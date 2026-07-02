// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Sync
//!
//! MuTate focuses on the timeline semaphore for course-grained inter-submission synchronization.
//! Using timeline semaphores soundly and according to spec has several requirements:
//!
//! - **Monotonic signaling** - Each value must only be signaled exactly once.
//! - **No waiting without signaling** - Every wait is produced from an intent to signal.
//! - **No dropped signal intents** - Every intent to signal is used (runtime enforced via DropBomb)
//!
//! To help uphold no wait-without-signal, we require consumption of `SignalIntent` via drop bomb.
//! Adding the signal to a `QueueSubmit` is the only endorsed way to consume each value of the
//! timeline.  This is not fool-proof.  One can store several `QueueSubmits`, but there is little
//! semantic or natural inclination to do so.
//!
//! The natural tendency is to make each `QueueSubmit` wait on the previous value and signal the
//! current value, enforcing that at most one submission will be executing at a time.  The
//! [`PoolRing`](super::pool::PoolRing) uses this style by default.

// NEXT Any encountered needs for "I know what I'm doing" patterns.

use ash::vk::{self, Handle};
use drop_bomb::DropBomb;

use crate::prelude::*;

/// Vulkan timeline [Semaphore](vk::Semaphore) with some light abstraction to encourage valid usage
/// and shave off unneeded API surface.
pub struct TimelineSemaphore {
    semaphore: vk::Semaphore,
    next: u64,
}

impl TimelineSemaphore {
    /// Zero is the only value we treat as a valid initial value.
    pub(crate) fn new(semaphore: vk::Semaphore) -> Self {
        Self { semaphore, next: 0 }
    }

    /// Used to provide already-signaled wait values for initializing other structuress.
    pub(crate) fn wait_initial(&self) -> WaitValue {
        WaitValue {
            semaphore: self.semaphore,
            value: self.next,
        }
    }

    /// Return a [`SignalIntent`] and increment the counter for the next signal value.
    ///
    /// The `SignalIntent` **must** be used within a `QueueSubmit` that will signal before any wait
    /// is attempted or deadlocks may occur.  Callers are responsible for tracking the
    /// `SignalIntent` and propagating the `WaitValue` to waiters.
    pub fn next_signal(&mut self) -> SignalIntent {
        self.next += 1;
        SignalIntent {
            semaphore: self.semaphore,
            value: self.next,
            bomb: DropBomb::new(format!(
                "Signal intent {} for {:0x}",
                self.next,
                self.semaphore.as_raw()
            )),
        }
    }

    pub fn as_raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn into_raw(self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn destroy(self, device: &Device) {
        unsafe { device.as_raw().destroy_semaphore(self.into_raw(), None) };
    }
}

/// An intent to signal a [`Fence`] (backed by a Vulkan timeline semaphore).  This value **must** be
/// consumed by a `QueueSubmit` that will only signal after the previous `SignalIntent` has signaled
/// to uphold sound usage according to the Vulkan spec.  The value can also be consumed by signaling
/// it on the CPU, but this requires waiting on the previous value.
pub struct SignalIntent {
    semaphore: vk::Semaphore,
    value: u64,
    bomb: DropBomb,
}

impl SignalIntent {
    pub fn as_signal_submit_info(
        mut self,
        stage: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        self.bomb.defuse();
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.value)
            .stage_mask(stage)
    }

    /// Consume this intent early by performing a **CPU-side** wait-then-signal.  Waits on previous
    /// value and signals current.  Only necessary when something that waits on the value is
    /// necessary for consistent progression.  Skipping to the next value instead is valid Vulkan.
    ///
    /// * `device` - Logical device that owns the semaphore.
    /// * `timeout` - Nanosecond timeout forwarded to `vkWaitSemaphores`.  `u64::MAX` for
    ///    indefinite.  Zero for immediate return.
    ///
    /// On failure, it is valid Vulkan behavior to simply use the next Fence value.  Monotonic is a
    /// sufficient condition for valid usage.
    pub fn try_consume(mut self, device: &Device, timeout: u64) -> Result<u64, VulkanError> {
        self.bomb.defuse();

        // Wait for the previous value since GPU submissions might not be finished.  If value is
        // still zero, there is no previous submission, so we can safely skip this step.
        if self.value > 0 {
            let previous = self.value - 1;
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(std::slice::from_ref(&self.semaphore))
                .values(std::slice::from_ref(&previous));
            unsafe { device.as_raw().wait_semaphores(&wait_info, timeout) }?;
        }

        // Signal the value for this intent.
        let signal_info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(self.value);

        unsafe { device.as_raw().signal_semaphore(&signal_info) }?;
        Ok(self.value)
    }

    /// Produce a `WaitValue` for this intent.
    pub fn wait_value(&self) -> WaitValue {
        WaitValue {
            semaphore: self.semaphore,
            value: self.value,
        }
    }
}

unsafe impl Send for SignalIntent {}

/// A value that will be signaled (the `SignalIntent` was created and will not be dropped).  It is
/// valid to wait on a single value at multiple points, so `SignalValue` can be cloned.
#[derive(Clone)]
pub struct WaitValue {
    semaphore: vk::Semaphore,
    value: u64,
}

impl WaitValue {
    /// Build a `vk::SemaphoreSubmitInfo` wait entry for queue submission.
    pub fn as_wait_submit_info(
        &self,
        stage: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(self.value)
            .stage_mask(stage)
    }

    /// Block the calling thread until this value has been signaled.
    pub fn wait(&self, device: &Device, timeout: u64) -> Result<u64, vk::Result> {
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(std::slice::from_ref(&self.semaphore))
            .values(std::slice::from_ref(&self.value));
        unsafe {
            device.as_raw().wait_semaphores(&wait_info, timeout);
        }
        Ok(self.value)
    }

    // XXX Just being lazy.  Support sorting and comparison.
    pub(crate) fn value(&self) -> u64 {
        self.value
    }
}

/// Vulkan **binary** [Semaphore](vk::Semaphore) with some light abstraction to encourage valid
/// usage and shave off unneeded API surface.
// MAYBE as a matter of style, any zero-size wrappers are completely fine to implement copy for.  We
// might even just want to go ahead and add some derives for into_raw and as_raw.
#[derive(Clone, Copy)]
pub struct BinarySemaphore {
    semaphore: vk::Semaphore,
}

impl BinarySemaphore {
    pub(crate) fn new(semaphore: vk::Semaphore) -> Self {
        Self { semaphore }
    }

    /// Produce a paired signal and wait intent for one semaphore cycle.
    ///
    /// The `BinarySignal` **must** be consumed in a submission that signals this semaphore.  The
    /// `BinaryWait` must be consumed in a subsequent submission or deadlock may occur.
    pub fn next(&self) -> (BinarySignal, BinaryWait) {
        let signal = BinarySignal {
            semaphore: self.semaphore,
        };
        let wait = BinaryWait {
            semaphore: self.semaphore,
        };
        (signal, wait)
    }

    /// Produce only the signal intent.  Used when the wait side is external, such as presentation
    /// waiting on rendering to signal.
    pub fn signal(&self) -> BinarySignal {
        BinarySignal {
            semaphore: self.semaphore,
        }
    }

    /// Produce only the wait intent.  Used when the signal side is external, such as swapchain
    /// image acquisition completing.
    pub fn wait(&self) -> BinaryWait {
        BinaryWait {
            semaphore: self.semaphore,
        }
    }

    pub fn as_raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn into_raw(self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn destroy(self, device: &Device) {
        unsafe { device.as_raw().destroy_semaphore(self.semaphore, None) };
    }
}

/// Intent to signal a binary semaphore from a queue submission.  This value **must** be consumed or
/// else the wait may deadlock.
#[derive(Clone, Copy)]
pub struct BinarySignal {
    semaphore: vk::Semaphore,
}

impl BinarySignal {
    pub fn as_raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn into_raw(self) -> vk::Semaphore {
        self.semaphore
    }
}

/// Intent to wait on a binary semaphore from a queue submission.  This value **must** be consumed
/// or else a double-signal and undefined behavior may result.
#[derive(Clone, Copy)]
pub struct BinaryWait {
    semaphore: vk::Semaphore,
}

impl BinaryWait {
    pub fn as_raw(&self) -> vk::Semaphore {
        self.semaphore
    }

    pub fn into_raw(self) -> vk::Semaphore {
        self.semaphore
    }
}

#[cfg(test)]
mod test {
    use super::*;

    pub fn create() {}

    pub fn wait_cpu() {}

    // NEXT on-device test
}
