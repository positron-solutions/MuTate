// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Submission
//!
//! A reduced, convenience-wrapped API surface for
//! [`vkQueueSubmit2`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkQueueSubmit2.html)
//! that has a fluent builder interface and only supports timeline semaphores (publicly)
//!
//! Queue submissions can batch together buffers and synchronization data.  Many buffers, perhaps
//! from many pools on many threads, can be submitted in one [`SubmitInfo`] and many `SubmitInfos`
//! can be batched inside a single [`QueueSubmit`].
//!
//! The `SignalIntents` for pool epochs must be consumed in a `QueueSubmit`.  This shape pulls
//! together the buffers from pools on several threads and the synchronization data that allows safe
//! re-acquisition of each pool during the epoch that reuses them.
//!
//! ## Building a `QueueSubmit`
//!
//! - Use the destination queue to start a `QueueSubmit`.
//! - Add any wait semaphores that should control
//! - Add command buffers to the `SubmitInfo`.
//! - Adding a semapahore will implicitly begin a new `SubmitInfo` within the `QueueSubmit`.
//! - Submit the `QueueSubmit` to consume
//!
//! ## Ordering Concerns
//!
//! - Command buffers within a SubmitInfo act as one continuous command stream, so barrier behavior
//!   between buffers remains intact within a single `SubmitInfo`.
//! - The order of `SubmitInfo`s **must** match dependency order created by semaphores (from spec)
//! - Wait semaphores should be added **before** buffers that come after then while signal
//!   semaphores should come **after** the buffers that will signal it when completed.
//!
//! Acquiring a pool from a ring will return a `SignalIntent`.  You **must** signal that intent
//! (signaling the underlying timeline semaphore value) at the end of the command buffer batch that
//! will finish last during that pool's epoch.  Failure could result in reset of a pool with
//! commands in flight, resulting in invalid and undefined behavior.

// ROLL The chosen implementation is one way of doing a builder while waiting on generic_const_exprs
// to stabilize.  In the meantime, we use a fixed maximum size and try to just make it easy on the
// compiler to prove everything away.  We are aided by the user-facing API resulting in clean arrays
// that are laid out in order already, matching Vulkan's use of slices when assembling the
// submission for the C API.
// NEXT protected support can likely inherit from the queue or command buffer.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use super::SubmissionModel;
use crate::internal::*;

mod sealed {
    /// Marker trait for command buffers that may be used as part of a submission.  Submissions
    /// bound for less capable queues may not receive command buffers that require higher
    /// capabilities.  See [`QueueCapability`](crate::context::queues::QueueCapability).
    pub trait SubmittableTo<Q: super::Capability>: super::Capability {}
}

// Transfer buffers can run on any queue
impl sealed::SubmittableTo<Graphics> for Transfer {}
impl sealed::SubmittableTo<Compute> for Transfer {}
impl sealed::SubmittableTo<Transfer> for Transfer {}

// Compute buffers need at least a compute-capable queue
impl sealed::SubmittableTo<Graphics> for Compute {}
impl sealed::SubmittableTo<Compute> for Compute {}

// Graphics buffers need a graphics capable queue
impl sealed::SubmittableTo<Graphics> for Graphics {}

pub use sealed::SubmittableTo;

const MAX_OPS: usize = 32;
const MAX_SUBMIT_INFOS: usize = MAX_OPS; // One op per vk::SubmitInfo2

/// Raw byte spans because we're being cool.
#[derive(Clone, Copy, Default)]
struct Span {
    start: u8,
    end: u8,
}

impl Span {
    fn slice<'a, T>(self, buf: &'a [MaybeUninit<T>]) -> &'a [T] {
        let uninit = &buf[self.start as usize..self.end as usize];
        // SAFETY: caller ensures [start..end] is initialized
        unsafe { std::slice::from_raw_parts(uninit.as_ptr().cast::<T>(), uninit.len()) }
    }
}

/// Index ranges for one VkSubmitInfo2.
#[derive(Clone, Copy, Default)]
struct InfoSpans {
    waits: Span,
    cmds: Span,
    signals: Span,
}

pub struct QueueSubmit<'q, QC: Capability> {
    queue: &'q QueueRef<QC>,

    // Flat arrays of pre-built Vulkan structs — these are sliced directly at submit time.
    waits: [MaybeUninit<vk::SemaphoreSubmitInfo<'static>>; MAX_OPS],
    cmds: [MaybeUninit<vk::CommandBufferSubmitInfo<'static>>; MAX_OPS],
    signals: [MaybeUninit<vk::SemaphoreSubmitInfo<'static>>; MAX_OPS],
    nw: u8,
    nc: u8,
    ns: u8,

    // Closed SubmitInfo spans.
    info_spans: [InfoSpans; MAX_SUBMIT_INFOS],
    ni: usize,

    // Cursors for the currently-open SubmitInfo.
    cur_wait_start: u8,
    cur_cmd_start: u8,
    cur_sig_start: u8,
    has_signal: bool,
}

impl<'q, QC: Capability> QueueSubmit<'q, QC> {
    pub(crate) fn new(queue: &'q QueueRef<QC>) -> Self {
        Self {
            queue,
            waits: std::array::from_fn(|_| MaybeUninit::uninit()),
            cmds: std::array::from_fn(|_| MaybeUninit::uninit()),
            signals: std::array::from_fn(|_| MaybeUninit::uninit()),
            nw: 0,
            nc: 0,
            ns: 0,
            info_spans: [InfoSpans::default(); MAX_SUBMIT_INFOS],
            ni: 0,
            cur_wait_start: 0,
            cur_cmd_start: 0,
            cur_sig_start: 0,
            has_signal: false,
        }
    }

    /// If we've already accumulated signals, the next wait closes the current SubmitInfo.
    fn maybe_close_current(&mut self) {
        if self.has_signal {
            self.info_spans[self.ni] = InfoSpans {
                waits: Span {
                    start: self.cur_wait_start,
                    end: self.nw,
                },
                cmds: Span {
                    start: self.cur_cmd_start,
                    end: self.nc,
                },
                signals: Span {
                    start: self.cur_sig_start,
                    end: self.ns,
                },
            };
            self.ni += 1;
            self.cur_wait_start = self.nw;
            self.cur_cmd_start = self.nc;
            self.cur_sig_start = self.ns;
            self.has_signal = false;
        }
    }

    /// Wait on a timeline semaphore before executing subsequent commands.  If there are no commands
    /// after this wait, an empty submit info will be used instead.
    pub fn wait(mut self, wait: WaitValue, stage: vk::PipelineStageFlags2) -> Self {
        self.maybe_close_current();
        self.waits[self.nw as usize].write(wait.as_wait_submit_info(stage));
        self.nw += 1;
        self
    }

    /// Wait on a binary semaphore before executing subsequent commands.  This is only used for APIs
    /// that require binary semaphores, such as presentation and swapchain acquisition.  Prefer
    /// timeline semaphores elsewhere.
    // XXX make crate-public after pulling in swapchain things
    pub fn wait_binary(mut self, wait: BinaryWait, stage: vk::PipelineStageFlags2) -> Self {
        self.maybe_close_current();
        self.waits[self.nw as usize].write(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(wait.into_raw())
                .value(0)
                .stage_mask(stage),
        );
        self.nw += 1;
        self
    }

    /// Execute the commands of a command buffer.
    pub fn execute<C: Capability + SubmittableTo<QC>, M: SubmissionModel>(
        mut self,
        cmd: ExecutableBuffer<C, M>,
    ) -> Self {
        self.cmds[self.nc as usize]
            .write(vk::CommandBufferSubmitInfo::default().command_buffer(cmd.into_parts()));
        self.nc += 1;
        self
    }

    /// Signal a timeline semaphore after preceding commands finish executing.  If there are no
    /// commands before this semaphore, an empty submit info will be created.
    pub fn signal(mut self, intent: SignalIntent, stage: vk::PipelineStageFlags2) -> Self {
        // Defuses the DropBomb — intent is consumed here.
        self.signals[self.ns as usize].write(intent.as_signal_submit_info(stage));
        self.ns += 1;
        self.has_signal = true;
        self
    }

    /// Signal a binary semaphore after the commands in this `SubmitInfo` complete.  This is only
    /// used for APIs that require binary semaphores, such as presentation and swapchain
    /// acquisition.  Prefer timeline semaphores elsewhere.
    // XXX make crate-public again after pulling in swapchain things
    pub fn signal_binary(mut self, signal: BinarySignal, stage: vk::PipelineStageFlags2) -> Self {
        self.signals[self.ns as usize].write(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(signal.into_raw())
                .value(0)
                .stage_mask(stage),
        );
        self.ns += 1;
        self.has_signal = true;
        self
    }

    pub fn submit(mut self, device: &ash::Device, fence: vk::Fence) -> Result<(), VulkanError> {
        // Close the final (possibly only) SubmitInfo.
        self.info_spans[self.ni] = InfoSpans {
            waits: Span {
                start: self.cur_wait_start,
                end: self.nw,
            },
            cmds: Span {
                start: self.cur_cmd_start,
                end: self.nc,
            },
            signals: Span {
                start: self.cur_sig_start,
                end: self.ns,
            },
        };
        self.ni += 1;

        // Build SubmitInfo2 array by slicing the pre-built flat arrays.
        // No conversion — just pointer arithmetic into already-initialized memory.
        let mut submit_infos = [MaybeUninit::<vk::SubmitInfo2>::uninit(); MAX_SUBMIT_INFOS];
        for i in 0..self.ni {
            let s = &self.info_spans[i];
            submit_infos[i].write(
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(s.waits.slice(&self.waits))
                    .command_buffer_infos(s.cmds.slice(&self.cmds))
                    .signal_semaphore_infos(s.signals.slice(&self.signals)),
            );
        }

        Ok(unsafe {
            // `lock_raw` is perhaps not the most creative name, but this is internal API surface.
            // Can't be fucked for it.
            let (raw, _guard) = self.queue.lock_raw()?;
            device.as_raw().queue_submit2(
                raw,
                // SAFETY: [0..self.ni] written above
                std::slice::from_raw_parts(
                    submit_infos.as_ptr().cast::<vk::SubmitInfo2>(),
                    self.ni,
                ),
                fence,
            )
        }?)
    }
}

#[cfg(test)]
pub mod test {
    use super::*;

    #[test]
    fn start_submission() {
        with_context!(|device, instance| {
            let start = device
                .queues
                .compute(QueuePriority::High)
                .queue_ref()
                .submission();
        })
    }

    #[test]
    fn empty_signal() {
        with_context!(|device, instance| {
            let mut semaphore = device.make_timeline_semaphore().unwrap();
            let signal_intent = semaphore.next_signal();
            let wait_value = signal_intent.wait_value();

            // Signal and wait create an implicit submission boundary.
            device
                .queues
                .compute(QueuePriority::High)
                .queue_ref()
                .submission()
                .signal(signal_intent, vk::PipelineStageFlags2::ALL_COMMANDS)
                .submit(&device, vk::Fence::null());
            wait_value.wait(&device, 8_000_000).unwrap();
            semaphore.destroy(&device);
        })
    }

    #[test]
    fn binary_semaphores() {
        with_context!(|device, _instance| {
            let queue = device
                .queues
                .graphics_offscreen(QueuePriority::High)
                .queue_ref();

            // This test shape is designed to look like an acquire-render-present sequence.
            let image_ready = device.make_binary_semaphore().unwrap();
            let render_done = device.make_binary_semaphore().unwrap();

            let (image_ready_signal, image_ready_wait) = image_ready.next();
            let (render_done_signal, render_done_wait) = render_done.next();

            // Signal the first binary semaphore so that we can wait on it in a second submission.
            queue
                .submission()
                .signal_binary(image_ready_signal, vk::PipelineStageFlags2::ALL_COMMANDS)
                .submit(&device, vk::Fence::null())
                .unwrap();

            // The render will use a regular timeline semaphore.
            let mut completion = device.make_timeline_semaphore().unwrap();
            let intent = completion.next_signal();
            let wait_value = intent.wait_value();

            // Second submission looks like a render, signals a binary semaphore that a real present
            // would pick up.
            queue
                .submission()
                .wait_binary(
                    image_ready_wait,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                )
                .signal_binary(render_done_signal, vk::PipelineStageFlags2::ALL_COMMANDS)
                .signal(intent, vk::PipelineStageFlags2::ALL_COMMANDS)
                .submit(&device, vk::Fence::null())
                .unwrap();

            // We will wait on this timeline semaphore to give us something on the CPu side to wait for.
            let mut present_done = device.make_timeline_semaphore().unwrap();
            let present_intent = present_done.next_signal();
            let present_wait = present_intent.wait_value();

            queue
                .submission()
                .wait_binary(render_done_wait, vk::PipelineStageFlags2::ALL_COMMANDS)
                .signal(present_intent, vk::PipelineStageFlags2::ALL_COMMANDS)
                .submit(&device, vk::Fence::null())
                .unwrap();

            // Drain the chain
            wait_value.wait(&device, 8_000_000).unwrap();
            present_wait.wait(&device, 8_000_000).unwrap();

            image_ready.destroy(&device);
            render_done.destroy(&device);
            completion.destroy(&device);
            present_done.destroy(&device);
        })
    }
}
