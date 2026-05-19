// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused)]

//! # Vulkan
//!
//! *The little engine that does or does not, but never tries (it just unwraps).*
//!
//!  ⚠️ View all module documentation as design proposals.  If there are tests or something else
//!  using a thing, that thing works.  Otherwise it may not exist.  Contact the maintainers to get
//!  an overview of the most recent plan for the API shape and guidance on how to implement what you
//!  need.
//!
//! ## An Ergonomic Vulkan Subset
//!
//! Vulkan is a buffet.  Buffets have to support a variety of ever-changing dishes that not every
//! customer eats.  **Nobody stands at the buffet to eat.** They put some dishes that they want onto
//! their plate and then take it back to their table.  That's a Vulkan library.
//!
//! **Key Vulkan subset choices:**
//!
//!   + Bindless descriptor arrays
//!   + Dynamic rendering
//!   + Scalar block layouts preferred
//!
//! ### Runtime Support
//!
//! Selecting fixed infrastructure strategies further reduces the Vulkan subset.  This kind of
//! support goes further in the direction of a fixed engine but also allows more terse declarations
//! for things like pipelines, reactive parameter updates, and asset loading.
//!
//! **Core MuTate goals:**
//!
//! - Runtime-driven modulation of pipeline parameters and dispatch topology.
//! - Pipeline remixing that can adapt existing content to new pipelines.
//! - User-consultative tuning and permutation flows back upstream into procedural resources
//!   (runtime re-baking).
//!
//! Many of these goals can only be supported with deep runtime integration to enable vague user
//! declarations to be adapted at runtime.
//!
//! ### Ergonomic Correctness
//!
//! - Host-GPU type and layout agreement for [Slang](https://shader-slang.org/) shader language via
//!   build-time reflection and proc macro generated const witnesses.
//!
//! - Type-state and type-safe wrappers for raw Vulkan types, but only where the tradeoffs are a
//!   clear win.
//!
//! - On-stack builders for more fluent construction and usage of very common API paths.
//!
//! - Pipeline macros for compact declaration of several related types, using specs to combine
//!   independent declarations by name.
//!
//! - All types can drop back to raw [`ash`](ash) bindings to work around missing support or API
//!   friction.
//!
//! ## Module Outline (The Plan)
//!
//! - Context
//!   + Entry & Instance
//!   + Devices
//!     * Queue
//!   + memory (just raw allocation)
//!   + Descriptor table
//! - Resources
//!   + Image
//!     * Sampler
//!   + Buffer
//!   + UBO
//!   + Shader Modules
//! - Slang (types and alias macros)
//!   + Layout (agreement push constants, block-scalar UBO & SSBO etc)
//! - Pipeline
//!   + Stage
//!   + Pipeline
//! - Dispatch
//!   + CommandPool & PoolRing
//!   + Command Buffers
//!   + Queue Submissions
//!   + Synchronization
//! - Presentation
//!   + Swapchain
//!   + Alternative Frontends
//!
//! ## Context
//!
//! The `Context` covers only-once or very rarely touched things.  There's one Instance.  We
//! initialize devices and things only once-ish.  We create descriptor tables once per-device.
//! Queue families are probed once at device creation.
//!
//! ## Resource
//!
//! SSBOs, UBOs, Images, how we type their contents, how we get their addresses, how we hand out
//! their descriptors, how we swap their pointers in the GPU.
//!
//! ## Slang
//!
//! Size, alignment, and semantic agreement with Slang types.  Includes many inherent Slang types
//! and type-safe wrappers.  Semantic wrappers can be used to specialize Slang types in Rust to
//! avoid accidental mixing of byte-compatible types.
//!
//! ## Pipeline
//!
//! Agreement between Stages, their shaders, and Resources is handled here.  We use Slang
//! reflection data to ensure that the Rust code will emit types that match the Slang layout.
//!
//! ## Dispatch
//!
//! Command pool lifecycle, command buffer recording, synchoronization, and queue submission.  We
//! provide the common command pool ring, some type-state wrappers, and fluent builders for
//! submissions.
//!
//! ## Presentation
//!
//! Swapchain abstraction, the wrapping around recording for graphics commands that will be
//! presented, interfaces for alternative frontends.
//!
//! ## Current Back Burner
//!
//! Graphics engines, let alone game engines, are a **huge** topic.  Certain modules are planned,
//! but it is unwise to over-commit to designs before concrete needs are driving them.
//!
//! - Render graph for fine-grained aliasing, hazard detection, and automation of long-ranged sync
//!   dependencies within command buffers.
//!
//! - Async resource streaming, shared ownership, intent-based resource resolution, reactive
//!   resource updates, memory management, all mostly built on top of great **late binding
//!   support**.  This is probably needed first since users have a small limit for the number of
//!   driver-decided allocations.
//!
//! - Independent timelines to provide course-grained fencing, scheduled dispatch, and to handle
//!   self-pacing audio graph versus VRR synchronization problems.

pub mod context;
pub mod dispatch;
pub mod pipeline;
pub mod present;
pub mod resource;
pub mod slang;
pub mod util;

use ash::vk;

pub use mutate_macros::{compute_pipeline, graphics_pipeline, stage, GpuType, Push};

pub mod prelude {
    pub use mutate_macros::{compute_pipeline, graphics_pipeline, stage, GpuType, Push};

    pub use super::VulkanError;
    pub use crate::context::device::{BinarySemaphore, Fence, TimelineSemaphore};
    pub use crate::context::queue::prelude::*;
    pub use crate::context::{vulkan::SupportedDevice, DeviceContext, VkContext};
    pub use crate::descriptor_newtype;
    pub use crate::device_address_newtype;
    pub use crate::dispatch::prelude::*;
    pub use crate::pipeline::prelude::*;
    pub use crate::present::surface::VkSurface;
    pub use crate::present::swapchain::AcquiredImage;
    pub use crate::slang::prelude::*;
    pub use crate::slang_newtype;

    // test harness macros
    pub use crate::with_context;
}

// Use as a private prelude.  Convenience without public immodesty 😦.
pub(crate) mod internal {
    pub use std::ffi::CStr;

    pub use ash::vk;
    pub use smallvec::SmallVec;

    pub use super::VulkanError;
    pub use crate::context::device::{BinarySemaphore, Fence, TimelineSemaphore};
    pub use crate::context::queue::prelude::*;
    pub use crate::context::{vulkan::SupportedDevice, DeviceContext, VkContext};
    pub use crate::dispatch::internal::*;
    pub use crate::present::surface::VkSurface;
    pub use crate::present::swapchain::SwapchainContext;
    pub use crate::slang::prelude::*;

    // test harness macros
    pub use crate::with_context;
}

// Flattened paths and re-exports for use in proc macro expansions
#[doc(hidden)]
pub mod __ {
    pub use crate::pipeline::stage as stage_slots;
    pub use crate::pipeline::stage::{Stage, StageReflection, StageSlot, StageSpec};
    pub use crate::pipeline::{layout::LayoutSpec, push::PushConstants, ComputePipelineSpec};
    pub use crate::slang::{field_end, field_start, DataLayout, FieldNode, GpuType, Scalar};

    // Re-exports.
    pub use ash;
    pub use bytemuck;
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("Vulkan: {0}")]
    // DEBT Use this only to begin returning results.  Use something else to actually start handling
    // them.
    ReplaceMe(&'static str),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("vulkan: Swapchain out of date")]
    SwapchainOutOfDate,
    #[error("vulkan: Surface lost")]
    SurfaceLost,
    #[error("vulkan: Fullscreen exclusive mode lost")]
    FullscreenExclusiveLost,
    #[error("vuklan: Device lost")]
    DeviceLost,
    #[error("vulkan: Suboptimal swapchain")]
    Suboptimal,
    #[error("vulkan: Out of memory (host)")]
    OutOfHostMemory,
    #[error("vulkan: Out of memory (device)")]
    OutOfDeviceMemory,
    #[error("vulkan: Validation layer caught an error: this is a bug")]
    ValidationFailed,
    // NOTE Making this one super explicit since it's an upstream unknown.  Not even Vulkan knows
    // what's going on.  Don't coerce anything else into this variant ever.
    #[error("vulkan: VK_ERROR_UNKNOWN (-13)")]
    VulkanUnknown,

    #[error("ash: other {0}")]
    Ash(vk::Result),

    #[error("driver error: {0}")]
    DriverError(String),
}

impl From<vk::Result> for VulkanError {
    fn from(r: vk::Result) -> Self {
        match r {
            vk::Result::ERROR_OUT_OF_DATE_KHR => Self::SwapchainOutOfDate,
            vk::Result::ERROR_SURFACE_LOST_KHR => Self::SurfaceLost,
            vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfHostMemory,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Self::OutOfDeviceMemory,
            // NOTE most validation error codes are safe to recognize even when we will not see
            // them, so don't worry about gating too tightly.
            vk::Result::ERROR_VALIDATION_FAILED_EXT => Self::ValidationFailed,
            // Full screen exclusive not supported.  See context.
            // vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => Self::FullscreenExclusiveLost,

            // Other extensions may land us here with new error types to encode.
            other => Self::Ash(other),
            // But Unknown is an explicit indication that must never be created manually.
            vk::Result::ERROR_UNKNOWN => Self::VulkanUnknown,
        }
    }
}
