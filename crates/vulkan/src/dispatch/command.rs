// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Command Pools & Buffers
//!
//! This module adds contracts and abstractions around raw command pools and buffers to encode the
//! answers to these kinds of questions:
//!
//! - What kinds of commands can this pool or its buffers execute?
//! - How long do the buffers live and how do we recycle them?
//! - What commands are valid to call in which circumstances?
//!
//! ## Recycling & Lifecycle
//!
//! Command pools can be reset as a whole (and their buffers must be recycled) or buffer-by-buffer.
//! It costs the driver a bit more to reset individual buffers, so we encourage reset of pools.  The
//! You semantics and ownership of pools and their buffers are very **arena-like**.
//!
//! You can obtain any number of buffers per pool.  This can be done to parallel record, although
//! that use case is waning.  Secondary buffers are likewise waning in utility as more work is being
//! delegated to the GPU.
//!
//! ## Valid Buffer Use
//!
//! - Command buffers must begin and end.  This involves a bit different ceremony for very simple
//!   cases like transfers versus complex cases like graphics.
//!
//! - Not all command buffers have the same capabilities.  A buffer tied to a compute queue cannot
//!   reliably be used to execute graphics commands (it may alias, but we still restrict based on
//!   types).
//!
//! - Not all commands are valid at all times.  We can enforce some of these contracts via
//!   typestate.
//!
//! ## The Command Pool
//!
//! Our [`CommandPool`] type is a lightweight abstraction over a pool and a set of buffers.  The
//! pool owns handles, so individual buffers do not need fine-grained management.  Reset the queue
//! and call `[buffer]` again to recieve recycled buffers.
//!
//! ## Recording Slot
//!
//! The very typical pattern for command buffer use is to cycle through a ring of usually just two
//! buffers, one front and one back buffer.  These are not held up on presentation semaphores etc
//! and so we only need two slots, a front and back buffer, one executing, one recording.  For
//! **longer serial workloads** that are intentionally throttled down over a small workgroup, there
//! may be more overlap, and so more parallelism of command buffers may pipeline over one another.
//! That's the use case where you will see 3-4 slots.

// This module is brand new.  if you find the module docs incomplete, please go to work on them.
// XXX create the pool from queues
// XXX create the slots owner to manage several pools
// XXX forward type restrictions to buffers
// XXX typed command buffer
// XXX Recording Ring seems like the abstraction about to happen.
// NEXT re-usable buffers for persistent workloads whose input and output change but whose commands
// do not.
// NEXT child buffers and their compositions.

use ash::vk;
use smallvec::SmallVec;

use crate::context::{DeviceContext, VkContext};

// The delightfully unsound thing about this abstraction for now is that CommandBuffers are freed
// whenever the pool is freed, so we don't track them, and everyone just needs to be adults for a
// little while. ☔  Maybe forever.  If you're an adult too long, they call you cowboy.  🤠
pub struct CommandPool {
    pool: vk::CommandPool,
    // XXX Store the more useful queue object when its ready
    queue: u32,
    outstanding: SmallVec<vk::CommandBuffer, 8>,
    recycled: SmallVec<vk::CommandBuffer, 8>,
}

impl CommandPool {
    // MAYBE temporary lifetimes for making things... for better chaining and less context
    // proliferation..  This is a good example.
    // XXX new queue
    pub fn new(device_context: &DeviceContext, queue_family_index: u32) -> Self {
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family_index);

        let pool = unsafe {
            device_context
                .device()
                .create_command_pool(&command_pool_ci, None)
                .unwrap()
        };

        Self {
            pool,
            queue: queue_family_index,
            outstanding: SmallVec::new(),
            recycled: SmallVec::new(),
        }
    }

    /// Return a single fresh command buffer.
    pub fn buffer(&mut self, device_context: &DeviceContext) -> vk::CommandBuffer {
        let buf = if let Some(buf) = self.recycled.pop() {
            buf
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.pool,
                command_buffer_count: 1,
                ..Default::default()
            };
            unsafe {
                device_context
                    .device()
                    .allocate_command_buffers(&alloc_info)
                    .unwrap()[0]
            }
        };

        self.outstanding.push(buf);
        buf
    }

    /// Return several command buffers.
    pub fn buffers(
        &mut self,
        device_context: &DeviceContext,
        count: u32,
    ) -> Vec<vk::CommandBuffer> {
        let count = count as usize;
        let from_recycled = self.recycled.len().min(count);
        let need_alloc = count - from_recycled;

        let mut result: Vec<vk::CommandBuffer> = self
            .recycled
            .drain(self.recycled.len() - from_recycled..)
            .collect();

        if need_alloc > 0 {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.pool,
                command_buffer_count: need_alloc as u32,
                ..Default::default()
            };
            let fresh = unsafe {
                device_context
                    .device()
                    .allocate_command_buffers(&alloc_info)
                    .unwrap()
            };
            result.extend_from_slice(&fresh);
        }

        self.outstanding.extend_from_slice(&result);
        result
    }

    /// Resetting the pool will reset all of its buffers.  You can either re-use handles or call
    /// [`buffer`] again.  The same set of handles will be returned in either case, but you are
    /// duplicating the ownership juggling if you retain handles, so only do this if it seems more
    /// ergonomic.
    pub fn reset(&mut self, device_context: &DeviceContext) {
        unsafe {
            device_context
                .device()
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        self.recycled.extend(self.outstanding.drain(..));
    }

    /// All outstanding handles are invalid after this call.
    pub fn destroy(&self, device_context: &DeviceContext) {
        unsafe {
            // When a pool is destroyed, all command buffers allocated from the pool are freed.
            device_context
                .device()
                .destroy_command_pool(self.pool, None);
        }
    }
}

// We might do some kind of RecordingRing as the abstraction.  Completely matches how we're really
// doing things in practice usually, recording buffers in pools with a pool reset for each fresh use.
pub struct RecordingSlot {
    pub command_buffer: vk::CommandBuffer,
}
