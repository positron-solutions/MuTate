// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused)]

//! # Vulkan
//!
//! *The little engine that does or does not, but never tries (it just unwraps).*
//!
//! - Ergonomic Vulkan subset:
//!   + Bindless descriptor arrays
//!   + Dynamic rendering
//!   + Scalar block layouts preferred
//! - Type-safe communication with Slang via build-time introspection and explicit Slang type
//!   analogs.
//! - Type-state and type-safe wrappers for raw Vulkan types, but only where the tradeoffs are a
//!   clear win.
//!
//! ## Back Burner
//!
//! Graphics engines, let alone game engines, are a **huge** topic.  Certain modules are planned,
//! but it is very dangerous to over-commit to designs before concrete needs are driving them.
//!
//! - Render graph for fine-grained aliasing, hazard detection, and automation of long-ranged sync
//!   dependencies within command buffers.
//! - Async resource streaming, shared ownership, intent-based resource resolution, reactive
//!   resource updates, memory management, all mostly built on top of great **late binding
//!   support**.
//! - Independent timelines to provide course-grained fencing, scheduled dispatch, and to handle
//!   self-pacing audio graph versus VRR synchronization problems.
//!
//! ## Ergonomics and Soundness
//!
//! Vulkan is extensible.  Once you choose what you will use, it's time to reduce that extensibility
//! into a terse sub-language that only does those chosen things really well.  The reduced API locks
//! the user into a specific model of using Vulkan in exchange for a vastly simplified set of
//! expressions to implement a small but good set of tools.
//!
//! Making that reduced surface ergonomic provides some opportunities to eliminate obviously wrong
//! choices by the user, and those are the only guarantees we chase at compile time.  Race towards
//! ergonomics and performance first.  Build contracts and guard rails only after there are
//! well-decided, high-value roads to guard.
//!
//! **Every GPU programmer has expressly opted into explicit synchronization of declared or easily
//! decided hazards.** This model means shared mutability is the *default*.  The window to
//! unsoundness is left open.  When you use this engine, you choose to be handed the keys.  We can
//! make sound expressions easier, but fully safe APIs on the GPU is a quagmire, and we will not lie
//! about **user responsibilities**.
//!
//! We don't want to over-specify contracts and get in the way of ergonomics or add onerous runtime
//! or compile time weight.  We just want to provide a limited API over a smaller toolbox, type-safe
//! code that can express obvious invariants, and macros to simplify emitting that type-safe code
//! without being distracted by it or unduly inconvenienced by satisfying it.
//!
//! ## Type Outline (The Plan)
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
//!   + Recording Ring
//!   + Recording Slot
//!     * Compute
//!     * Transfer
//!     * Graphics
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
//! avoid accidental mixing.
//!
//! ## Pipeline
//!
//! Agreement between Stages, their shaders, and Resources is handled here.  We use Slang
//! introspection data to ensure that the Rust code will emit types that match the Slang layout.
//!
//! ## Dispatch
//!
//! Recording, lifecycle, and submission of command buffers.
//!
//! ## Presentation
//!
//! Swapchain abstraction, the wrapping around recording for graphics commands that will be
//! presented, interfaces for alternative frontends.

pub mod context;
pub mod dispatch;
pub mod pipeline;
pub mod present;
pub mod resource;
pub mod slang;
pub mod util;

use ash::vk;

pub mod prelude {
    pub use super::VulkanError;
    pub use crate::context::{vulkan::SupportedDevice, DeviceContext, VkContext};
    pub use crate::present::surface::VkSurface;
}

// Re-export for slang's macros
#[doc(hidden)]
pub use bytemuck as __bytemuck;

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
