// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Vulkan
//!
//! The little engine that does or does not, but never tries (because it's unsafe).
//!
//! Core types: WIP (LIES)
//!
//! - `VkContext`
//! - `Stage`
//! - `Pipeline`
//! - **Shader inputs**
//!   * `PushConstants`
//!     + `PushConstantRange`
//!   * `Descriptors`
//!   * `Image`
//!   * `Buffer`
//!   * `Uniform`
//! - **Targets**
//!   * `ComputeTarget`
//!   * `GraphicsTarget`

pub mod buffer;
pub mod context;
pub mod descriptors;
pub mod image;
pub mod queue;
pub mod util;

use ash::vk;

// The module structure is purely to allow the proc macros to be nicely developed and tested before
// everything gets re-exported behind feature flags in mutate-lib.

pub mod prelude {
    pub use super::VulkanError;
    pub use crate::context::VkContext;
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan: {0}")]
    // DEBT Use this only to begin returning results.  Use something else to actually start handling
    // them.
    ReplaceMe(&'static str),

    #[error("Ash: {0}")]
    Ash(#[from] vk::Result),
}
