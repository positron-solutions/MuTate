// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Command Pool
//!
//! Command buffers come from pools.  This module provides pools that are consistent with the queues
//! they come from and accounting help for common reuse strategies.
//!
//! For typical use, either create a `CommandPool` directly or a ring of pools via `PoolRing`.  For
//! single epochs or maintaining longer lived pools of buffers, create a `CommandPool` directly.
//!
//! `PoolRing` is ideal when well-divided epochs of command buffer sets, such as rendering for one
//! frame, will be continously cycled.
//!
//! ## Pool Ring Size
//!
//! There are usually two generations in order to pipeline recording one epoch concurrent with
//! dispatch of another epoch.  More **concurrent** pipeline overlap requires longer rings to avoid
//! stalling re-acquisition of a pool still in flight.

// NOTE Pool Reset & Safety
//
// Valid reset of a `CommandPool` requires waiting on the final submission group containing a buffer
// from the pool to signal a unique semaphore.  Valid use of all command buffers from a reset pool
// would require transitioning them from initial to recording states again.

// NEXT Pool ring size options, but low priority
// MAYBE Tracking pool usage pattern where every submission extends the epoch

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;

use ash::vk::Handle;
use drop_bomb::DropBomb;

use crate::internal::*;

use super::cb;
use super::SubmissionModel;

/// Individual pools track the handles they have given out in order to support reset.
// MAYBE we might wind up with more than one Pool abstraction, but likely the PoolRing will always
// use just one of those abstractions.
pub struct CommandPool<C: Capability, M: SubmissionModel = OneTime> {
    raw: vk::CommandPool,
    queue: vk::Queue,
    primary_handles: smallvec::SmallVec<vk::CommandBuffer, 8>,
    primary_cursor: usize,
    secondary_handles: smallvec::SmallVec<vk::CommandBuffer, 8>,
    secondary_cursor: usize,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

impl<C: Capability, M: SubmissionModel> CommandPool<C, M> {
    pub fn new(device: &ash::Device, queue: &QueueRef<C>) -> Result<Self, VulkanError> {
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(M::POOL_FLAGS)
            .queue_family_index(queue.family());

        let pool = unsafe { device.create_command_pool(&command_pool_ci, None)? };

        Ok(Self {
            raw: pool,
            queue: unsafe { queue.as_raw() }, // We don't use for any calls requiring external synchronization.
            primary_handles: smallvec::SmallVec::new(),
            primary_cursor: 0,
            secondary_handles: smallvec::SmallVec::new(),
            secondary_cursor: 0,
            _cap: PhantomData,
            _model: PhantomData,
        })
    }

    // XXX build with Bon builder and provide some override gear for arguments too.
    pub fn primary(&mut self, device: &ash::Device) -> Result<RecordingBuffer<C, M>, VulkanError> {
        let cb = self.next_primary(device)?;
        let begin_info = vk::CommandBufferBeginInfo::default().flags(M::BUFFER_FLAGS);
        unsafe {
            device.begin_command_buffer(cb, &begin_info)?;
        }
        Ok(RecordingBuffer::from_raw(cb))
    }

    /// Reset the pool, returning all allocated command buffers to the initial state.
    ///
    /// # Safety
    ///
    /// Resetting a pool whose buffers are in-flight is undefined behavior per the Vulkan
    /// specification.  The caller must ensure that **every** command buffer allocated from this
    /// pool has finished execution on the GPU.  Any outstanding buffers must call `assume_reset` or
    /// rebuild as initial states.
    pub unsafe fn reset(
        &mut self,
        device: &ash::Device,
        release_resources: bool,
    ) -> Result<(), VulkanError> {
        let flags = if release_resources {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        } else {
            vk::CommandPoolResetFlags::empty()
        };
        device.reset_command_pool(self.raw, flags)?;
        // Restore both cursors to the full length of each list
        self.primary_cursor = self.primary_handles.len();
        self.secondary_cursor = self.secondary_handles.len();
        Ok(())
    }

    fn next_primary(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer, VulkanError> {
        if self.primary_cursor > 0 {
            self.primary_cursor -= 1;
            Ok(self.primary_handles[self.primary_cursor])
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.raw)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
            self.primary_handles.push(cb);
            Ok(cb)
        }
    }

    fn next_secondary(&mut self, device: &ash::Device) -> Result<vk::CommandBuffer, VulkanError> {
        if self.secondary_cursor > 0 {
            self.secondary_cursor -= 1;
            Ok(self.secondary_handles[self.secondary_cursor])
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.raw)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);
            let cb = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
            self.secondary_handles.push(cb);
            Ok(cb)
        }
    }

    // NEXT secondary support for GRAPHICS is dependent on some implementation of a rendering state
    // shadow.  It is not clear if we can provide that at compile or runtime ergonomically yet.  The
    // state shadow will likely be useful for runtime composition of unlike pipelines, so that is
    // the expected implementation driver.

    pub fn into_raw(self) -> vk::CommandPool {
        let CommandPool { raw, .. } = self;
        raw
    }

    pub fn as_raw(&self) -> &vk::CommandPool {
        &self.raw
    }

    pub fn destroy(self, device: &ash::Device) {
        let CommandPool { raw, .. } = self;
        unsafe { device.destroy_command_pool(raw, None) };
    }
}

impl<M> CommandPool<Compute, M>
where
    M: SubmissionModel,
{
    pub fn secondary(
        &mut self,
        device: &ash::Device,
    ) -> Result<RecordingBuffer<Compute, M>, VulkanError> {
        let cb = self.next_secondary(device)?;

        // Compute secondaries are never executed inside a rendering scope,
        // so inheritance is all defaults and RENDER_PASS_CONTINUE must not
        // be set in M::BUFFER_FLAGS.
        let inheritance = vk::CommandBufferInheritanceInfo::default();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(M::BUFFER_FLAGS)
            .inheritance_info(&inheritance);
        unsafe {
            device.begin_command_buffer(cb, &begin)?;
        }
        Ok(RecordingBuffer::from_raw(cb))
    }
}

// Implementation is specific to OneTime to enable easier type inferences.
impl<C: Capability> CommandPool<C, OneTime> {
    pub fn transient(device: &ash::Device, queue: &QueueRef<C>) -> Result<Self, VulkanError> {
        Self::new(device, queue)
    }
}

// NEXT with_pool macro for easy testing?

/// A ring of [`CommandPool`] leases.  Intended for one-pool-per-epoch style usages where a pool is
/// synchronized and reset before each acquisition.  Recording for the epoch and submission are
/// intended to complete before the next acquisition is attempted.
///
/// The default submission model is [`OneTime`].  Buffers in the pool are used once per epoch
pub struct PoolRing<C: Capability, const N: usize = 2, M: SubmissionModel = OneTime> {
    pools: [CommandPool<C, M>; N],
    timeline: TimelineSemaphore,
    /// Timeline value each slot's prior lease promised to signal.  `0` on the first lap matches
    /// the timeline's initial value, so the first `advance` per slot doesn't block.
    done_values: [WaitValue; N],
    /// Slot index vended by the next `acquire`.
    cursor: usize,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

impl<C: Capability, M: SubmissionModel, const N: usize> PoolRing<C, N, M> {
    pub fn new(device: &Device, queue: &QueueRef<C>) -> Result<Self, VulkanError> {
        const { assert!(N >= 1, "PoolRing requires at least one slot") };
        // FIXME Extension trait this over the regular ash device and get those traits into prelude.
        let timeline = device.make_timeline_semaphore()?;

        let mut pools: [MaybeUninit<CommandPool<C, M>>; N] = [const { MaybeUninit::uninit() }; N];
        for i in 0..N {
            match CommandPool::<C, M>::new(device, queue) {
                Ok(pool) => {
                    pools[i].write(pool);
                }
                Err(e) => {
                    // DEBT manual destruction of partially constructed ring resources
                    for slot in &mut pools[..i] {
                        unsafe { slot.assume_init_read() }.destroy(device);
                    }
                    unsafe { timeline.destroy(device) };
                    return Err(e);
                }
            }
        }
        // Bless the initialized pools
        let pools: [CommandPool<C, M>; N] = unsafe {
            // SAFETY: every slot in `pools` was written above; the only early-return
            // path destroys the partial prefix and returns before reaching here.
            let ptr = &pools as *const [MaybeUninit<CommandPool<C, M>>; N]
                as *const [CommandPool<C, M>; N];
            let init = ptr.read();
            std::mem::forget(pools);
            init
        };

        // The zero initial value is always signaled, so all waits immediately acquire on the first lap.
        let done_values: [WaitValue; N] = std::array::from_fn(|_| timeline.wait_initial());

        Ok(Self {
            pools,
            done_values,
            timeline,
            cursor: 0,
            _cap: PhantomData,
            _model: PhantomData,
        })
    }

    /// Wait for a slot and reset its pool to prepare for recording.  The pool can be used to give
    /// out any number of primary and secondary buffers.
    ///
    /// The return `SignalIntent` **must** be used in a queue submission. The user contract is that
    /// the `SignalIntent` must only be signaled by the `QueueSubmit` that will **finish** executing
    /// on the GPU last.  Signaling early will allow undefined behavior as the pool resets while
    /// command execution is in flight.
    ///
    /// In typical usage, the recording for epoch `n` is pipelined with dispatch for epoch `n - 1`,
    /// but recording and dispatch for a single epoch is exclusive.
    ///
    /// With external self-pacing on things like window events, the acquisition can be non-blocking,
    /// but without external self-pacing, the acquisition can block up to `timeout` nanoseconds.
    pub fn acquire(
        &mut self,
        device: &ash::Device,
        timeout: u64,
    ) -> Result<(&mut CommandPool<C, M>, SignalIntent), VulkanError> {
        let slot = self.cursor;
        self.done_values[slot].wait(device, timeout)?;
        unsafe {
            device.reset_command_pool(
                *self.pools[slot].as_raw(),
                vk::CommandPoolResetFlags::empty(),
            )?;
        }
        let intent = self.timeline.next_signal();
        self.done_values[slot] = intent.wait_value();
        self.cursor = (slot + 1) % N;
        Ok((&mut self.pools[slot], intent))
    }

    /// Wait for all command buffers in flight to be retired on the device.  Uses the highest
    /// timeline semaphore value for all pools.  If a pool has been acquired but will not yet
    /// signal, this method **will deadlock and you will trip and fall on your way home, so
    /// don't call this unless the pool you acquired has attached its signal semaphore to the final
    /// command buffer that will finish.**
    pub fn drain(&self, device: &ash::Device, timeout: u64) -> Result<u64, VulkanError> {
        let latest = self
            .done_values
            .iter()
            .max_by_key(|w| w.value())
            .expect("PoolRing always has N >= 1 slots"); // XXX this can be expressed as type contract
        Ok(latest.wait(device, timeout)?)
    }

    pub fn destroy(self, device: &Device) {
        let Self {
            pools, timeline, ..
        } = self;
        for pool in pools {
            pool.destroy(device);
        }
        timeline.destroy(device);
    }
}

#[cfg(test)]
mod test {
    use crate::with_context;

    use super::*;

    #[test]
    fn pool_instantiate() {
        with_context!(|device, instance| {
            let queue = device
                .queues
                .graphics_offscreen(QueuePriority::Low)
                .queue_ref();
            let pool = CommandPool::<Graphics, OneTime>::new(&device, &queue).unwrap();
            pool.destroy(&device);
        });
    }

    #[test]
    fn pool_buffer_create() {
        with_context!(|device, instance| {
            let queue = device
                .queues
                .graphics_offscreen(QueuePriority::Low)
                .queue_ref();
            let mut pool = CommandPool::<Compute, OneTime>::transient(&device, &queue).unwrap();
            let primary = pool.primary(&device).unwrap();
            let secondary = pool.secondary(&device).unwrap();
            pool.destroy(&device);

            std::mem::forget(primary);
            std::mem::forget(secondary);
        });
    }

    #[test]
    fn ring_instantiate() {
        with_context!(|device, instance| {
            let queue = device
                .queues
                .graphics_offscreen(QueuePriority::Low)
                .queue_ref();
            let ring = PoolRing::<Graphics>::new(&device, &queue).unwrap();
            ring.destroy(&device);
        });
    }

    #[test]
    fn acquire_pool() {
        with_context!(|device, instance| {
            let queue = device
                .queues
                .graphics_offscreen(QueuePriority::Low)
                .queue_ref();
            let mut ring = PoolRing::<Graphics, 2, OneTime>::new(&device, &queue).unwrap();

            let (pool, intent) = ring.acquire(&device, 16_000_000).unwrap();
            let cb = pool.primary(&device).unwrap();
            // In real usage, signal intent travels to the queue submit along with at least the cb
            // that will finish last.
            std::mem::forget(intent);
            std::mem::forget(cb);
            ring.destroy(&device);
        });
    }
}
