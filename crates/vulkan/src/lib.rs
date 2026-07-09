// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(warnings)]

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
//! support goes further in the direction of a fixed capability engine but also allows more terse
//! declarations for things like pipelines, reactive parameter updates, and asset loading.
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
//! - Instance
//!   + SupportedDevice
//! - Device
//!   + Queue
//!   + Memory
//!   + Descriptors
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
//!   + Surface
//!   + Alternative Frontends
//!
//! ## Instance
//!
//! Entry, instance, physical device scanning, and obtaining some other loaders that are most
//! tightly bound to the top level raw `ash::Instance`.
//!
//! ## Device
//!
//! The logical device.  We set up the queues, descriptor table, and will **soon** add a memory
//! sub-allocation manager.
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

pub mod device;
pub mod dispatch;
pub mod instance;
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
    // NEXT move fences out of prelude and de-emphasize in favor of consistent timeline semaphore
    // usage except where required (swapchain acquisition, presentation etc).
    pub use crate::descriptor_newtype;
    pub use crate::device::prelude::*;
    pub use crate::device_address_newtype;
    pub use crate::dispatch::prelude::*;
    pub use crate::instance::prelude::*;
    pub use crate::pipeline::prelude::*;
    pub use crate::present::prelude::*;
    pub use crate::present::surface::Surface;
    pub use crate::resource::buffer::{MappedAllocation, MappedWriteView};
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
    pub use crate::device::prelude::*;
    pub use crate::dispatch::internal::*;
    pub use crate::instance::prelude::*;
    pub use crate::present::prelude::*;
    pub use crate::present::swapchain::{AcquiredImage, Swapchain};
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
    #[error("vulkan: {0}")]
    // DEBT Use this only to begin returning results.  Use something else to actually start handling
    // them.
    ReplaceMe(&'static str),

    #[error("thread poisoned")]
    Poisoned,

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Attempted to acquire on a swapchain that is already marked for recreation needed.
    #[error("vulkan: Swapchain recreation needed.")]
    SwapchainRecreationRequired,

    // Errors that coerce from ash::vk results should not be constructed manually.
    /// Presentation succeeded, but the swapchain is no longer optimal for the surface and should be
    /// recreated.
    #[error("vulkan: Swapchain suboptimal")]
    SwapchainSuboptimal,
    /// Further presentation attempts will fail and the swapchain must be recreated to begin
    /// succeeding again.
    #[error("vulkan: Swapchain out of date.")]
    SwapchainOutOfDate,
    #[error("vulkan: Surface lost")]
    SurfaceLost,
    #[error("vulkan: Fullscreen exclusive mode lost")]
    FullscreenExclusiveLost,
    #[error("vuklan: Device lost")]
    DeviceLost,
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

    // Catch-all for ash results we don't know how to coerce yet.
    #[error("ash: other {0}")]
    Ash(vk::Result),

    /// Polling the Vulkan implementation or device returned results we could not make sense of or
    /// properly handle.  Use when something seems degenerate or inconsistent.
    // NEXT separate Instance and Device errors
    #[error("driver error: {0}")]
    DriverError(String),
    /// No queue family on the logical device had the correct capabilities or presentation
    /// capability on the surface.
    #[error("queue: no queue family with requested capabilities found")]
    QueueNotFound,

    /// Polling the window and compositor could not decide a useable swapchain size, and the correct
    /// behavior is to request redraw and wait for another event.
    #[error("surface: degenerate extent ({width}x{height}); window may be minimized")]
    DegenerateExtent { width: u32, height: u32 },
    /// Surface support inferred that the window is minimized and the application likely wants to
    /// idle while waiting for more event.
    // NOTE if you arrive here and the window does not "seem" minimized, check the surface module.
    // We inferred the minimized state from a zero inner pixel size.
    #[error("surface: window seems minimized")]
    WindowMinimized,
}

impl<T> From<std::sync::PoisonError<T>> for VulkanError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        VulkanError::Poisoned
    }
}

impl From<vk::Result> for VulkanError {
    fn from(r: vk::Result) -> Self {
        match r {
            vk::Result::ERROR_OUT_OF_DATE_KHR => Self::SwapchainOutOfDate,
            vk::Result::SUBOPTIMAL_KHR => Self::SwapchainSuboptimal,
            vk::Result::ERROR_SURFACE_LOST_KHR => Self::SurfaceLost,
            vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfHostMemory,
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Self::OutOfDeviceMemory,
            // NOTE most validation error codes are safe to recognize even when we will not see
            // them, so don't worry about gating too tightly.
            vk::Result::ERROR_VALIDATION_FAILED_EXT => Self::ValidationFailed,
            // Full screen exclusive not supported.  See context.
            // vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT => Self::FullscreenExclusiveLost,

            // Unknown is an explicit indication that must never be created manually.
            vk::Result::ERROR_UNKNOWN => Self::VulkanUnknown,
            // Other extensions may land us here with new error types to encode.
            other => Self::Ash(other),
        }
    }
}
