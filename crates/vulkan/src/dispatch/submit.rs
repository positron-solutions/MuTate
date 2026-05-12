// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Command Buffer Submission
//!
//! Command buffers are submitted in groups.  Submission groups are also the granularity of
//! synchronization between command buffers that are not in the same group.

// XXX synchronization for reclamation.  Use the pool / recording context?
// XXX sequential lock-out mechanism

use std::marker::PhantomData;

use crate::internal::*;

pub struct Submission<C: Capability> {
    // Order matters. Heterogeneous in M but homogeneous in C.
    entries: Vec<vk::CommandBufferSubmitInfo<'static>>,
    waits: Vec<vk::SemaphoreSubmitInfo<'static>>,
    signals: Vec<vk::SemaphoreSubmitInfo<'static>>,
    _cap: PhantomData<C>,
}
