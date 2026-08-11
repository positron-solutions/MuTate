// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Resource
//!
//! ⚠️ Very WIP.  The goal has become pretty clear:
//!
//! - Resource and workload specs, declarative render dependencies.
//! - Asynchronous loading via declarative state and reconciliation processes to approach that
//!   state.
//! - Dependency resolution & type / semantic compatibility checks to validate machine learning
//!   driven selection of pipeline compositions.
//!
//! Basically we want Kubernetes for Vulkan.  The machine learning gives us a desired pipeline
//! composition while the runtime creates and destroys whatever is needed, approaching the goal
//! state of the render.  When resources are ready in time for the dispatch, they start firing.
//!
//! Another need is to provide a thread safe runtime handle that sits above the device.  We want the
//! top level handle in each thread to be application, not device-centric.  There are sometimes
//! multiple devices.  Devices can be switched and used for different purposes.  The `Device` is not
//! the right top-level thing.

// This module began to grow into a fully fledged async resource creation system.  That work has
// been put off to allow more concrete code to drive the development.  What was learned is that we
// really, really want to get handles and pointers into consumers at the last possible moment and
// discourage persistent ownership of such views.  That will make every streaming, shared ownership,
// compaction problem so much easier.  Until then, we will focus on making the highly manual bits
// less manual.

pub mod buffer;
pub mod image;
pub mod retired;
pub mod shader;
pub mod ubo;

pub(crate) mod core {
    pub use super::buffer::core::*;
}
