// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Command Pool
//!
//! Command buffers come from pools.  This module provides pools that are consistent with the queues
//! they come from and accounting help for common reuse strategies.
//!
//! For typical use, either create a `CommandPool` directly or a ring of pools via `PoolRing`.  The
//! pool ring is ideal when well-divided epochs of command buffer sets, such as rendering for one
//! frame, will be continously cycled.  For single epochs or longer lived command buffers,
//! create a `CommandPool` directly.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::Deref;

use crate::internal::*;

use super::cb;
use super::SubmissionModel;

// MAYBE uncertain contract shape for buffer reset and synchronization data.

pub struct CommandPool<C: Capability, M: SubmissionModel = OneTime> {
    raw: vk::CommandPool,
    queue: Queue<C>,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

impl<C: Capability, M: SubmissionModel> CommandPool<C, M> {
    pub fn new(device_context: &DeviceContext, queue: &Queue<C>) -> Result<Self, VulkanError> {
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(M::POOL_FLAGS)
            .queue_family_index(queue.family());

        let pool = unsafe {
            device_context
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
    // XXX return typed buffers
    pub fn primary(
        &self,
        device_context: &DeviceContext,
    ) -> Result<RecordingBuffer<C, M>, VulkanError> {
        let device = device_context.device();
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

    pub fn secondary(
        &self,
        device_context: &DeviceContext,
    ) -> Result<RecordingBuffer<C, M>, VulkanError> {
        let device = device_context.device();
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_pool(self.raw)
            .command_buffer_count(1);
        unsafe {
            let cb = device.allocate_command_buffers(&alloc_info)?[0];
            let cb_begin_info = vk::CommandBufferBeginInfo::default().flags(M::BUFFER_FLAGS);
            device.begin_command_buffer(cb, &cb_begin_info)?;
            Ok(RecordingBuffer::from_raw(cb))
        }
    }

    pub fn into_raw(self) -> vk::CommandPool {
        let CommandPool { raw, .. } = self;
        raw
    }

    pub fn as_raw(&self) -> &vk::CommandPool {
        &self.raw
    }

    pub fn destroy(self, device_context: &DeviceContext) {
        let CommandPool { raw, .. } = self;
        unsafe { device_context.device().destroy_command_pool(raw, None) };
    }
}

// Implementation is specific to OneTime to enable easier type inferences.
impl<C: Capability> CommandPool<C, OneTime> {
    pub fn transient(
        device_context: &DeviceContext,
        queue: &Queue<C>,
    ) -> Result<Self, VulkanError> {
        Self::new(device_context, queue)
    }
}

// NEXT with_pool macro for easy testing?

/// A borrowed handle to a reset pool, valid for the duration of one epoch.  Derefs to the
/// underlying [`CommandPool`], so `primary` / `secondary` recording works transparently:
///
/// ```ignore
/// let lease = ring.acquire(&device_context)?;
/// let cb = lease.primary(&device_context)?;
/// // ...record...
/// // Submission signals `lease.signal_value()` on `lease.timeline()`.
/// ```
pub struct PoolLease<'ring, C: Capability, M: SubmissionModel> {
    pool: &'ring CommandPool<C, M>,
    timeline: TimelineSemaphore,
    signal_value: u64,
}

impl<'ring, C: Capability, M: SubmissionModel> PoolLease<'ring, C, M> {
    /// The timeline semaphore that the lease must signal on submission.
    pub fn timeline(&self) -> TimelineSemaphore {
        self.timeline
    }

    /// The timeline value to signal when this epoch's submission completes.
    pub fn signal_value(&self) -> u64 {
        self.signal_value
    }
}

impl<'ring, C: Capability, M: SubmissionModel> Deref for PoolLease<'ring, C, M> {
    type Target = CommandPool<C, M>;

    fn deref(&self) -> &Self::Target {
        self.pool
    }
}

/// A ring of [`CommandPool`] leases.  Intended for one-pool-per-epoch style usages where a pool is
/// synchronized and reset before each acquisition.  Recording for the epoch and submission are
/// intended to complete before the next acquisition is attempted.
///
/// The default submission model is [`OneTime`].  Buffers in the pool are used once per epoch
pub struct PoolRing<C: Capability, const N: usize = 2, M: SubmissionModel = OneTime> {
    pools: [CommandPool<C, M>; N],
    /// Timeline value each slot's prior lease promised to signal.  `0` on the first lap matches
    /// the timeline's initial value, so the first `advance` per slot doesn't block.
    done_values: [u64; N],
    timeline: TimelineSemaphore,
    /// Slot index vended by the next `acquire`.
    cursor: usize,
    /// Value tagged on the next lease.
    next_epoch: u64,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

struct Slot<C: Capability, M: SubmissionModel> {
    pool: CommandPool<C, M>,
    /// Timeline value that a prior lease promised to signal.  `0` means "never used yet."
    done_value: u64,
}

impl<C: Capability, M: SubmissionModel, const N: usize> PoolRing<C, N, M> {
    pub fn new(device_context: &DeviceContext, queue: &Queue<C>) -> Result<Self, VulkanError> {
        const { assert!(N >= 1, "PoolRing requires at least one slot") };

        let device = device_context.device();
        let timeline = device_context.make_timeline_semaphore(0)?;

        let mut pools: [MaybeUninit<CommandPool<C, M>>; N] = [const { MaybeUninit::uninit() }; N];
        for i in 0..N {
            match CommandPool::<C, M>::new(device_context, queue) {
                Ok(pool) => {
                    pools[i].write(pool);
                }
                Err(e) => {
                    // DEBT manual destruction of partially constructed ring resources
                    for slot in &mut pools[..i] {
                        let pool = unsafe { slot.assume_init_read() };
                        pool.destroy(device_context);
                    }
                    // XXX into raw and as_raw methods
                    unsafe { device.destroy_semaphore(timeline.0, None) };
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
            core::mem::forget(pools);
            init
        };

        Ok(Self {
            pools,
            done_values: [0; N],
            timeline,
            cursor: 0,
            next_epoch: 1,
            _cap: PhantomData,
            _model: PhantomData,
        })
    }

    /// Wait for a slot and reset its pool to prepare for recording.  In typical usage, the
    /// recording for epoch `n + 1` is pipelined with dispatch for epoch `n`, but recording and
    /// dispatch are never overlapping.  With external self-pacing on things like window events, the
    /// acquisition is non-blocking.
    pub fn acquire(
        &mut self,
        device_context: &DeviceContext,
    ) -> Result<PoolLease<'_, C, M>, VulkanError> {
        let device = device_context.device();
        let slot = self.cursor;

        let wait_values = [self.done_values[slot]];
        let wait_semaphores = [self.timeline.as_raw()];
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&wait_semaphores)
            .values(&wait_values);
        unsafe { device.wait_semaphores(&wait_info, u64::MAX)? };

        unsafe {
            device.reset_command_pool(
                *self.pools[slot].as_raw(),
                vk::CommandPoolResetFlags::empty(),
            )?;
        }

        let signal_value = self.next_epoch;
        self.done_values[slot] = signal_value;
        self.next_epoch += 1;
        self.cursor = (slot + 1) % N;

        Ok(PoolLease {
            pool: &self.pools[slot],
            timeline: self.timeline,
            signal_value,
        })
    }

    pub fn destroy(self, device_context: &DeviceContext) {
        let Self {
            pools, timeline, ..
        } = self;
        for pool in pools {
            pool.destroy(device_context);
        }
        unsafe {
            device_context
                .device()
                .destroy_semaphore(timeline.into_raw(), None)
        };
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
            let pool = CommandPool::<Graphics, OneTime>::transient(&device_ctx, &queue).unwrap();
            let _primary = pool.primary(&device_ctx).unwrap();
            let _secondary = pool.secondary(&device_ctx).unwrap();
            pool.destroy(&device_ctx);
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

            let lease = ring.acquire(&device_ctx).unwrap();
            let _cb = lease.primary(&device_ctx).unwrap();
            assert_eq!(lease.signal_value(), 1);
            // MAYBE If any ring were going to wait on next acquisition, it's unclear which
            // submission would signal that wait.  If no submission was done or that submission
            // won't signal, we don't have a sound contract in the API shape.  The only sound
            // possibility is to wait on the last buffer that will complete for an epoch.
            drop(lease);
            ring.destroy(&device_ctx);
        });
    }
}
