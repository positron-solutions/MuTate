// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Recording & Dispatch
//!
//! Record commands into command buffers.  Synchronize and submit buffers to queues.  Recycle and
//! reset command buffers and pools.  This module encompasses how we feed the queues and enable
//! runtime components to get their commands recorded without knowing too much about the pools and
//! buffers we give to them.
//!
//! ## Pool Geometry
//!
//! - [`Pool`] - A single command pool, used alone for transient bulk operations or persistent
//!   buffer sets.  The vended buffers can use any recording model but the default is `OneTime`.
//!
//! - [`PoolRing`] - A literal ring of `Pool`s for epoch style use.  Per-cycle one pool is taken
//!   from the ring and all of the buffers that the pool vends will be used once and implicitly
//!   reset when the same pool is handed out again.
//!
//! ## Buffer Recording & Submission Models
//!
//! - [`OneTime`] - Recorded and submitted only once.  This is the default.
//!
//! - [`Sequential`] - May be re-submitted exclusively with synchronization on previous submission.
//!   Use this for repetitive work where re-recording can be avoided.
//!
//! - [`Simultaneous`] - May re-submitted even while in flight.  Reclaim requires synchronization.
//!   This style can be heavy on drivers.  It is only useful when overlapping dispatches of the same
//!   command buffer are required.
//!
//! ### Reset Support
//!
//! Reset of individual buffers is supported for `Sequential` and `Simultaneous` models.  When using
//! buffers multiple times from the same pool, it is natural to want to reset individual buffers
//! instead of the whole pool.  The extra recording cost is amortized over many queue submissions
//! and avoided for the other buffers in the pool that can avoid re-recording.
//!
//! Reset and re-use support works by splitting the buffer into two parts prior to submission.
//! Simultaneous use buffers may be submitted multiple times into the same `SubmitInfo` and so can
//! be re-used directly.  Sequential buffers may be submitted multiple times but must exclude
//! concurrent re-use via a synchronization token.
//!
//! Not all use cases can benefit from reset.  Individual buffer reset is unnatural for epoch style
//! `PoolRing` usage and is not supported. Streaming or large batch submissions can benefit from
//! resetting buffers from a small pool instead of waiting for moments to reset the entire pool.
//! Individual reset can also make sense for a pool of long-lived, rarely updated multi-submission
//! buffers that frequently are partially in flight and don't present a good timing to reset the
//! entire pool.
//!
//! Reset is distinct from reclaiming & re-vending a new buffer, which may require more
//! re-allocation overhead.  However, reclaim & re-vend is an available tactic even when reset is
//! not, so it works with all submission models.
//!
//! ## Safety
//!
//! Pools are not thread-safe.  To use a ring or pool on multiple threads, instead give each thread
//! its own ring or pool and use semaphores and host synchronization for coordinating submission /
//! order.  Recorded command buffers may be shared across threads for queue submission.  Queue
//! submission is thread-safe through the [`Queue`] abstraction.
//!
//! Reset of pool is difficult to make fully safe without reference counting and synchronization.
//! `PoolRing` usage is not fully safe if you leak command buffer handles outside the epoch.  Leaked
//! buffers will become invalid and the type-state transition cannot be enforced by compile-time
//! knowledge.  For standalone pools, it can be more natural to hold onto buffers somewhere and less
//! natural to reset the entire pool.  **Do not hold onto buffers if you want to reset a pool.**
//!
//! Completely safe runtime abstractions are straightforward to make but immediately begin taxing
//! other code with API shape or runtime overhead, so you're on your own for idiot-proof
//! abstractions.
//!
//! ## Queue Family Compatibility
//!
//! Buffers from different pools in the same queue family may be submitted together.  Secondary
//! buffers may be executed by primary buffers from a different pool as long as their pools belong
//! to the same queue family.
//!
//! ## Secondary Buffers
//!
//! A command pool can allocate secondary command buffers that may be used in a primary as long as
//! the pools are both in the same queue family.  Even if the primary or its pool is reset, the
//! secondary can be re-used if its model is `Sequential` or `Simultaneous`.  If `Simultaneous`, the
//! same secondary can be used in two different primaries concurrently.
//!
//! ## Render Phase Alignment
//!
//! Depending on the compositor and winit, the phase alignment between dispatch and presentation
//! could be quite off.  Since the swapchain might acquire at too fast of pace, only the command
//! pool epoch synchronization becomes both naturally aligned and reasonably self-pacing.
//!
//! The (present wait)[super::pw] module was intended to help with this problem, but the
//! capabilities demonstrated are not clearly going to fix problems just yet.

// NOTE This module seems really prone to covering all of the bases in terms of safe contracts vs
// API tax and capability.  For example, any sophisticated modular recording scheme will likely pass
// buffers between threads.  Resetting the source pools resets the buffers.  Expressing this remote
// type transition (spooky action at a distance) requires expressing relations that don't follow the
// scope or ownership topology.  Well, Rust can't express that yet.
//
// Perhaps someday there will be a mechanism of injecting type through scope that will affect all
// force-sensitive types.  Transmuting just one owning object is another option that propagates
// less spookily but still requires us to mint some kind of phantom tokens.  We can use any
// phase-coordinating point to mint such a token, but is that part of this crate's API or do we
// provide public methods for implementing such ghostery?
//
// Choices made so far are conservative in avoiding API tax and retaining full capability.  **This
// is at the expense of safety and validity.** Please guide the semantics, what the code says it
// does, towards natural soundness (ie the users get it right because the API suggests doing things
// right) and API convenience until real needs can pull development more directly.

// NEXT Secondary recording into a primary.
// NEXT The re-use token structure for sequential buffers is not yet decided, but simultaneous use
// buffers do not need any synchronization support.
// NEXT Sync is not well presented in these module docs.  The command pool docs probably need pushed
// into pool and a top level doc needs to present the high-level overview.

pub mod cb;
pub mod pool;
pub mod pw;
pub mod submit;
pub mod sync;

pub mod prelude {
    // Put traits into there as they show up

    // Marker types
    pub use super::OneTime;
    pub use super::Sequential;
    pub use super::Simultaneous;

    pub use super::cb::{
        ExecutableBuffer, ExecutableSecondary, RecordingBuffer, RecordingSecondary,
        RenderingBuffer, RenderingSecondary,
    };
    pub use super::pool::{CommandPool, PoolRing};
    pub use super::submit::QueueSubmit;
    // XXX make binary private after pulling in swapchain presentation gear
    pub use super::sync::{
        BinarySemaphore, BinarySignal, BinaryWait, SignalIntent, TimelineSemaphore, WaitValue,
    };
}

pub(crate) mod internal {
    pub use super::OneTime;
    pub use super::Sequential;
    pub use super::Simultaneous;

    pub use super::cb::{
        ExecutableBuffer, ExecutableSecondary, RecordingBuffer, RecordingSecondary,
        RenderingBuffer, RenderingSecondary,
    };
    pub use super::pool::{CommandPool, PoolRing};
    pub use super::submit::QueueSubmit;
    pub use super::sync::{
        BinarySemaphore, BinarySignal, BinaryWait, SignalIntent, TimelineSemaphore, WaitValue,
    };
}

use std::marker::PhantomData;

use ash::vk;

use crate::internal::*;

mod sealed {
    pub trait SubmissionModel {
        const BUFFER_FLAGS: ash::vk::CommandBufferUsageFlags;
        const POOL_FLAGS: ash::vk::CommandPoolCreateFlags;
    }
}

pub trait SubmissionModel: sealed::SubmissionModel {}

/// Buffer can only be recorded and submitted once.  It is considered reclaimed by the driver on
/// submission.
pub struct OneTime;
/// Buffer can be dispatched multiple times, but only one dispatch may be in-flight at any time.
pub struct Sequential;
/// Same buffer can be dispatched multiple times.
pub struct Simultaneous;

impl sealed::SubmissionModel for OneTime {
    const BUFFER_FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
    const POOL_FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::TRANSIENT;
}
impl sealed::SubmissionModel for Sequential {
    const BUFFER_FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::empty();
    const POOL_FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
}
impl sealed::SubmissionModel for Simultaneous {
    const BUFFER_FLAGS: vk::CommandBufferUsageFlags = vk::CommandBufferUsageFlags::SIMULTANEOUS_USE;
    const POOL_FLAGS: vk::CommandPoolCreateFlags = vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
}

impl SubmissionModel for OneTime {}
impl SubmissionModel for Sequential {}
impl SubmissionModel for Simultaneous {}

pub(crate) trait Resettable {}
impl Resettable for Sequential {}
impl Resettable for Simultaneous {}
