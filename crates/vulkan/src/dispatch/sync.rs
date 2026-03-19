// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Sync
//!
//! This module focuses on the phase alignment and pacing of recording and submission.  More
//! granular timing or ordering of draw dependencies needs some other kind of timing &
//! synchronization support.
//!
//! - [`PresentWaitConsumer`] collect presentation times for phase alignment and present to present
//!   duration measurement using `VK_KHR_present_wait` and a waiter.
//!
//! - [`RenderTimings`] insert observable timestamp commands to estimate frame draw time.  Uses
//!   `VK_KHR_calibrated_timestamps` and injects / measures / calibrates timing events.
//!
//! - [`SwapchainContext`] wraps up swapchain herding 🐈‍⬛.  Centralizes decisions about when to
//!   attempt the next render & presentation.

use std::{
    thread,
    time::{Duration, Instant},
    u64,
};

use ash::{khr::present_wait::Device as PwDevice, vk};

use mutate_untorn::prelude::*;

use crate::{
    context::{DeviceContext, VkContext},
    VulkanError,
};

/// Observable state for the waiter.
#[derive(Clone, Copy)]
struct PresentConsumerState {
    /// Sentinel for the internal thread to know to die.
    alive: bool,
    /// Newest present ID that can be waited on
    next_id: u64,
    /// Swapchain we are presenting against.
    swapchain: vk::SwapchainKHR,
}

impl PresentConsumerState {
    fn with_id(self, next_id: u64) -> Self {
        Self { next_id, ..self }
    }

    fn with_swapchain(self, swapchain: vk::SwapchainKHR) -> Self {
        Self {
            swapchain,
            next_id: u64::MAX,
            ..self
        }
    }

    fn with_dead(self) -> Self {
        Self {
            alive: false,
            next_id: u64::MAX,
            swapchain: vk::SwapchainKHR::null(),
        }
    }
}

/// Polling handle for the incoming present data.  Notifies the waiter of the next waitable present
/// ID.  Presentation is just one piece of timing data, so the control loop is going to live up in a
/// higher context.
pub struct PresentConsumer {
    waiter_state: UntornReader<PresentWaiterState>,
    consumer_state: UntornWriter<PresentConsumerState>,
    waiter_thread: thread::JoinHandle<()>,
    /// Temporary internal ID used between creating the present info and notifying the waiter.
    id: u64,
    unparker: parking::Unparker,
}

impl PresentConsumer {
    pub fn new(
        vk_context: &VkContext,
        device_context: &DeviceContext,
        swapchain: vk::SwapchainKHR,
    ) -> Result<Self, VulkanError> {
        let pw_device = PwDevice::new(&vk_context.instance, &device_context.device());

        let (waiter_writer, waiter_reader) = Untorn::new(PresentWaiterState {
            last_window: Duration::MAX,
            last_present: Instant::now(),
            last_id: u64::MAX,
        })
        .split();

        let (consumer_writer, consumer_reader) = Untorn::new(PresentConsumerState {
            alive: true,
            swapchain: swapchain,
            next_id: 0,
        })
        .split();

        let (parker, unparker) = parking::pair();

        let waiter = PresentWaiter {
            waiter_state: waiter_writer,
            consumer_state: consumer_reader,
            pw_device,
            parker,
        };

        let waiter_thread = std::thread::Builder::new()
            .name("present-waiter".into())
            .spawn(move || waiter.run())?;

        Ok(Self {
            waiter_state: waiter_reader,
            consumer_state: consumer_writer,
            waiter_thread,
            id: 0,
            unparker,
        })
    }

    /// Returns None when the last frame timing was not for consecutive IDs, meaning some presents
    /// were missed or some other likely garbage state.
    pub fn read_last_present(&self) -> Option<Presentation> {
        let last = self.waiter_state.read();
        if last.last_window == Duration::MAX {
            None
        } else {
            Some(last)
        }
    }

    /// Get the next id that should be polled.  Without calling `notify_waiter`, the id will be
    /// skipped in the data stream, and the waiter will report a zero window until it waits on
    /// consecutive frames again.
    pub fn next_present_id(&mut self) -> u64 {
        self.id += 1;
        self.id
    }

    /// Updates state and wakes up the waiter if it was parked.
    pub fn notify_waiter(&mut self) {
        let current = self.consumer_state.read();
        self.consumer_state.write(current.with_id(self.id));
        self.unparker.unpark();
    }

    /// Provide a new swapchain.  (Indicates max ID to force reset).
    pub fn notify_swapchain_recreation(&mut self, swapchain: vk::SwapchainKHR) {
        let updated = self.consumer_state.read().with_swapchain(swapchain);
        self.consumer_state.write(updated);
    }

    pub fn destroy(mut self) {
        let current = self.consumer_state.read();
        self.consumer_state.write(current.with_dead());
        self.unparker.unpark();
        self.waiter_thread.join().ok();
    }
}

/// A snapshot of presentation data for the consumer
#[derive(Clone, Copy)]
pub struct PresentWaiterState {
    /// Most recent present-to-present window.
    pub last_window: Duration,
    /// Timing of most recent present.
    pub last_present: Instant,
    /// Most recently observed ID, not the most recently awaited.
    pub last_id: u64,
}
// If the present wait state gets more complex, these will diverge.
type Presentation = PresentWaiterState;

impl PresentWaiterState {
    fn missed(mut self) -> Self {
        Self {
            last_id: u64::MAX,
            last_present: Instant::now(),
            last_window: Duration::MAX,
        }
    }

    fn caught(mut self, id: u64) -> Self {
        let now = Instant::now();
        if id.saturating_sub(1) == self.last_id {
            self.last_window = now - self.last_present;
        } else {
            self.last_window = Duration::MAX;
        }
        self.last_present = now;
        self.last_id = id;
        self
    }
}

pub struct PresentWaiter {
    waiter_state: UntornWriter<PresentWaiterState>,
    consumer_state: UntornReader<PresentConsumerState>,

    // Device is a bit more durable than swapchains.  We probably have been detroyed if there's a
    // new device because it needs to reset a lot.
    pw_device: ash::khr::present_wait::Device,
    parker: parking::Parker,
}

impl PresentWaiter {
    pub fn run(&self) {
        loop {
            self.parker.park();
            let consumer = self.consumer_state.read();
            if !consumer.alive {
                break;
            }
            let mut state = self.waiter_state.read();

            // Nothing presented yet, or we've already waited on this ID.  Re-park.
            if consumer.next_id == u64::MAX || consumer.next_id == state.last_id {
                continue;
            }

            let id = consumer.next_id;
            let swapchain = consumer.swapchain;

            // XXX Pick an actual expected timeout duration.  We need the graphics timing in order
            // to do this (doesn't exist yet)
            match unsafe { self.pw_device.wait_for_present(swapchain, id, 6_000_000) } {
                Ok(_) => {
                    self.waiter_state.write(state.caught(id));
                }
                Err(vk::Result::TIMEOUT) => {
                    // Missed the latch window — loop and try again with the same id.
                    // Consumer may have advanced next_id in the meantime; we'll pick
                    // that up at the top of the loop.
                    self.waiter_state.write(state.missed());
                }
                Err(e) => {
                    self.waiter_state.write(state.missed());
                    eprintln!("present wait error: {:?}", e);
                }
            }
        }
    }
}
