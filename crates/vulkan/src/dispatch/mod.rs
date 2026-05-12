// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Recording & Dispatch
//!
//! Record commands into command buffers.  Submit buffers to queues.  Recycle and reset command
//! buffers and pools.  This module encompasses how we feed the queues and enable runtime components
//! to get their commands recorded without knowing too much about the pools and buffers we give to
//! them.
//!
//! ## Pool Geometry
//!
//! - [`Pool`] - A single command pool, used alone for transient bulk operations or persistent
//!   buffer sets.  The vended buffers can use any recording model but the default is `OneTime`.
//!
//! - [`PoolRing`] - A literal `Pool` ring for epoch style use.  Per-cycle one pool is taken from
//!   the ring and all of the buffers that the pool vends will be used once and implicitly reset
//!   when the same pool is handed out again.
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
//! Reset of individual buffers is supported for use cases where it fits the pool better than
//! resetting the entire pool.  Individual buffer reset cannot make sense for epoch style `PoolRing`
//! usage and is not supported.
//!
//! Individual reset can make sense for bulk submissions of one-time-use buffers that don't present
//! a good timing to reset the entire pool.  Individual reset can also make sense for a pool of
//! long-lived, rarely updated multi-submission buffers.
//!
//! Reset is distinct from reclaiming & re-vending a new buffer, which may require more
//! re-allocation overhead.  However, reclaim & re-vend is an available tactic even when reset is
//! not.
//!
//! ## Thread Safety
//!
//! Pools are not thread-safe.  To use a ring or pool on multiple threads, instead give each thread
//! its own ring or pool and use semaphores and host synchronization for coordinating submission /
//! order.  Queue submission is thread-safe through the [`Queue`] abstraction.
//!
//! ## Queue Family Compatibility
//!
//! Buffers from different pools in the same queue family may be submitted together.  Secondary
//! buffers may be executed in primary buffers from a different pool as long as their pools belong
//! to the same family.
//!
//! ## Secondary Buffers
//!
//! A command pool can allocate secondary command buffers that may be used in a primary as long as
//! the pools are both in the same queue family.  Even if the primary or its pool is reset, the
//! secondary can be re-used if its model is `Sequential` or `Simultaneous`.  If `Simultaneous`, the
//! same secondary can be used in two different primaries concurrently.
//!
//! - [`SecondaryRecording`]

//! TODO move onto pool ring size argument
//! ## Pool Ring Size
//!
//! There are usually two generations in
//! order to pipeline recording beside parallel dispatch.  The epochs form a ring and the ring has
//! slots.
//!

// NEXT protected support

pub mod cb;
pub mod pool;
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
    pub use super::pool::{CommandPool, PoolLease};
}

pub(crate) mod internal {
    pub use super::OneTime;
    pub use super::Sequential;
    pub use super::Simultaneous;

    pub use super::cb::{
        ExecutableBuffer, ExecutableSecondary, RecordingBuffer, RecordingSecondary,
        RenderingBuffer, RenderingSecondary,
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

// XXX decide one
// pub(crate) use sealed::SubmissionModel;
pub trait SubmissionModel: sealed::SubmissionModel {}

/// Buffer can only be recorded and submitted once.  It is considered reclaimed by the driver on
/// submission.
pub struct OneTime;
/// Buffer can be dispatched multiple times, but only one dispatch may be in-flight at any time.
// NEXT it would be useful to enforce single-use-in-flight, but the ownership lifecycle for sync
// primitive and old-to-new-buffer type transition need to be figured out.  We may just tack sync
// data onto each buffer but hide the detail from consumers.
pub struct Sequential;
/// Same buffer can be dispatched multiple times.
// NEXT reclaiming this with copies in flight would be kind of bad.  Again, appending some sync data
// may allow this to clean up naturally.  Threading calls through some recording context that
// manages the pool's handles is another option.
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
