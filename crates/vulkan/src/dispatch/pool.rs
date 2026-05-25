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

pub struct CommandPool<C: Capability, M: SubmissionModel = OneTime> {
    raw: vk::CommandPool,
    queue: Queue<C>,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

impl<C: Capability, M: SubmissionModel> CommandPool<C, M> {
    pub fn new(device_ctx: &DeviceContext, queue: &Queue<C>) -> Result<Self, VulkanError> {
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(M::POOL_FLAGS)
            .queue_family_index(queue.family());

        let pool = unsafe {
            device_ctx
                .device()
                .create_command_pool(&command_pool_ci, None)?
        };

        Ok(Self {
            raw: pool,
            queue: queue.clone(),
            _cap: PhantomData,
            _model: PhantomData,
        })
    }

    // XXX build with Bon builder and provide some override gear for arguments too.
    pub fn primary(
        &self,
        device_ctx: &DeviceContext,
    ) -> Result<RecordingBuffer<C, M>, VulkanError> {
        let device = device_ctx.device();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.raw)
            .command_buffer_count(1);
        unsafe {
            let cb = device.allocate_command_buffers(&alloc_info)?[0];
            let cb_begin_info = vk::CommandBufferBeginInfo::default().flags(M::BUFFER_FLAGS);
            device.begin_command_buffer(cb, &cb_begin_info)?;
            Ok(RecordingBuffer::from_raw(cb))
        }
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
        &self,
        device_ctx: &DeviceContext,
        release: bool,
    ) -> Result<(), VulkanError> {
        let flags = if release {
            vk::CommandPoolResetFlags::empty()
        } else {
            vk::CommandPoolResetFlags::RELEASE_RESOURCES
        };
        device_ctx
            .device()
            .reset_command_pool(self.raw, flags)
            .map_err(Into::into)
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

    pub fn destroy(self, device_ctx: &DeviceContext) {
        let CommandPool { raw, .. } = self;
        unsafe { device_ctx.device().destroy_command_pool(raw, None) };
    }
}

impl<M> CommandPool<Compute, M>
where
    M: SubmissionModel,
{
    pub fn secondary(
        &self,
        device_ctx: &DeviceContext,
    ) -> Result<RecordingBuffer<Compute, M>, VulkanError> {
        let device = device_ctx.device();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_pool(self.raw)
            .command_buffer_count(1);

        unsafe {
            let cb = device.allocate_command_buffers(&alloc_info)?[0];

            // Compute secondaries are never executed inside a rendering scope,
            // so inheritance is all defaults and RENDER_PASS_CONTINUE must not
            // be set in M::BUFFER_FLAGS.
            let inheritance = vk::CommandBufferInheritanceInfo::default();
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(M::BUFFER_FLAGS)
                .inheritance_info(&inheritance);

            device.begin_command_buffer(cb, &begin)?;
            Ok(RecordingBuffer::from_raw(cb))
        }
    }
}

// Implementation is specific to OneTime to enable easier type inferences.
impl<C: Capability> CommandPool<C, OneTime> {
    pub fn transient(device_ctx: &DeviceContext, queue: &Queue<C>) -> Result<Self, VulkanError> {
        Self::new(device_ctx, queue)
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
    done_values: [u64; N],
    /// Slot index vended by the next `acquire`.
    cursor: usize,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

impl<C: Capability, M: SubmissionModel, const N: usize> PoolRing<C, N, M> {
    pub fn new(device_ctx: &DeviceContext, queue: &Queue<C>) -> Result<Self, VulkanError> {
        const { assert!(N >= 1, "PoolRing requires at least one slot") };

        let device = device_ctx.device();
        let timeline = device_ctx.make_timeline_semaphore()?;

        let mut pools: [MaybeUninit<CommandPool<C, M>>; N] = [const { MaybeUninit::uninit() }; N];
        for i in 0..N {
            match CommandPool::<C, M>::new(device_ctx, queue) {
                Ok(pool) => {
                    pools[i].write(pool);
                }
                Err(e) => {
                    // DEBT manual destruction of partially constructed ring resources
                    for slot in &mut pools[..i] {
                        unsafe { slot.assume_init_read() }.destroy(device_ctx);
                    }
                    unsafe { timeline.destroy(device_ctx) };
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
        let done_values = [0; N];

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
        device_ctx: &DeviceContext,
        timeout: u64,
    ) -> Result<(&CommandPool<C, M>, SignalIntent), VulkanError> {
        let device = device_ctx.device();
        let slot = self.cursor;

        // NEXT we can probably add some internal convenience via WaitValue or TimelineSemaphore
        // that would make this manual construction more expressive and re-use our
        // as_wait_submit_info method.
        let wait_values = [self.done_values[slot]];
        let wait_semaphores = [self.timeline.as_raw()];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&wait_semaphores)
            .values(&wait_values);
        unsafe { device.wait_semaphores(&wait_info, timeout)? };

        unsafe {
            device.reset_command_pool(
                *self.pools[slot].as_raw(),
                vk::CommandPoolResetFlags::empty(),
            )?;
        }
        let intent = self.timeline.next_signal();
        self.cursor = (slot + 1) % N;
        Ok((&self.pools[slot], intent))
    }

    pub fn destroy(self, device_ctx: &DeviceContext) {
        let Self {
            pools, timeline, ..
        } = self;
        for pool in pools {
            pool.destroy(device_ctx);
        }
        timeline.destroy(device_ctx);
    }
}

#[cfg(test)]
mod test {
    use crate::with_context;

    use super::*;

    #[test]
    fn pool_instantiate() {
        with_context!(|device_ctx, vk_ctx| {
            let queue = device_ctx.queues.graphics_offscreen(QueuePriority::Low);
            let pool = CommandPool::<Graphics, OneTime>::new(&device_ctx, &queue).unwrap();
            pool.destroy(&device_ctx);
        });
    }

    #[test]
    fn pool_buffer_create() {
        with_context!(|device_ctx, vk_ctx| {
            let queue = device_ctx.queues.graphics_offscreen(QueuePriority::Low);
            let pool = CommandPool::<Compute, OneTime>::transient(&device_ctx, &queue).unwrap();
            let primary = pool.primary(&device_ctx).unwrap();
            let secondary = pool.secondary(&device_ctx).unwrap();
            pool.destroy(&device_ctx);

            std::mem::forget(primary);
            std::mem::forget(secondary);
        });
    }

    #[test]
    fn ring_instantiate() {
        with_context!(|device_ctx, vk_ctx| {
            let queue = device_ctx.queues.graphics_offscreen(QueuePriority::Low);
            let ring = PoolRing::<Graphics>::new(&device_ctx, &queue).unwrap();
            ring.destroy(&device_ctx);
        });
    }

    #[test]
    fn acquire_lease() {
        with_context!(|device_ctx, vk_ctx| {
            let queue = device_ctx.queues.graphics_offscreen(QueuePriority::Low);
            let mut ring = PoolRing::<Graphics, 2, OneTime>::new(&device_ctx, &queue).unwrap();

            let (pool, intent) = ring.acquire(&device_ctx, 16_000_000).unwrap();
            let cb = pool.primary(&device_ctx).unwrap();
            // In real usage, signal intent travels to the queue submit along with at least the cb
            // that will finish last.
            std::mem::forget(intent);
            std::mem::forget(cb);
            ring.destroy(&device_ctx);
        });
    }
}
