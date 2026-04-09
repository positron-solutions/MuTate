// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Push Constants
//!
//! Input data directly embedded into the command buffers.  The way we intend to use Vulkan
//! (bindless, descriptor table etc), one way or another, whether data is embedded directly or by
//! reference, the shader gets *it* or gets *to it* starting from push constants.
//!
//! ## Enabling Downstream Ergonomics
//!
//! The goal of this implementation is to enable ergonomic macros that can emit generated and
//! reusable push constant structures within broader declarations.  Type checking the host-GPU
//! agreement is part of ergonomics.  In addition to extremely straightforward default behaviors,
//! composing pipelines conveniently during development will benefit from flexible data sharing
//! between stages.  This is an example of the kind of declaration we want to enable:
//!
//! ```ignore
//! // XXX When more concrete code is available, update this example.
//! #[derive(Push)]
//! struct MyPush {
//!     #[stages(RAYGEN | INTERSECTION)]
//!     ray_ctrl_idx: UInt32,
//!     #[stages(RAYGEN | INTERSECTION)]
//!     intersection_ctrl_idx: UInt32,
//!     #[stages(CLOSEST)] // Creates a gap for INTERSECTION
//!     collision_ctrl_masks: [u8; 4],
//!     // undecorated fields are visible to all stages.  #[stages(ALL)] is also valid.
//!     reflection_ctrl_idx: UInt32,
//!     #[stages(CLOSEST | INTERSECTION)]
//!     collision_ctrl_idx: UInt32,
//! }
//! ```
//!
//! ## Implementation
//!
//! Ground facts that drive everything:
//!
//! - Push constants have a min-spec of 128 bytes.  With indirection, this is more than enough.
//! - Pipeline stages declare a single, contiguous constant block, the bytes they will *see*.
//! - Push constant ranges map a subset of the 128 bytes into the stage blocks using masking flags.
//! - Several ranges, with gaps, may together form a sparse, order-preserving subset of the 128 bytes into
//!   the contiguous constant block of a shader stage.
//! - Stage constant blocks my overlap in these mappings, so ranges and stages may not be
//!   one-to-one.
//! - Vulkan specifies that push constant range offsets must begin on a 4-byte boundary and sizes
//!   must be a multiple of four bytes.
//!
//! The key design insight is that no matter how the 128 bytes are viewed by stages, no matter which
//! subset of those bytes we write in which granularity, the type at every byte must be consistent
//! with a single layout for all 128 bytes, so a **struct** (or tuple, but that would be harder to
//! derive) is the natural choice to declare the pipeline's [`PushConstant`]s.
//!
//! Stages likewise only ever see a contiguous constant block.  Again, a **struct** is a natural way
//! for a stage to declare the host side of its constant block.  We can re-use the same witnesses on
//! both pipelines and stages because both are effectively a zero-offset view of their entire
//! domain.
//!
//! The mapping from pipeline to stages is where the tricky bits exist.  Each stage may *see* bytes
//! from two separate ranges e.g. `vk::StageFlags::Graphics` set on bytes 0-16 in one range and
//! 64-80 in another range.  We also don't always need to write to individual ranges.  However, we
//! must always declare all ranges precisely to the driver or else some stages will *see* the wrong
//! subset.
//!
//! Finally, **writing** ergonomics for some use cases (we re-used some stages from other
//! pipelines?) may favor sparse ranges and combined stage flags.  Vulkan allows this kind of mixed,
//! interspersed write, but **you cannot pass flags for stages that have no declared range covering
//! those bytes.** If you need to write bytes that are viewed by some stages but not all that you
//! want to set flags for, you must instead use multiple writes.
//!
//! From all this, we can begin to deduce the necessary types and relations:
//!
//! - One structure for a Pipeline's push constants
//! - One structure per-stage that is a *view* of constants
//! - Several typed ranges that connect the pipeline structure to stage views and are used to create
//!   the pipeline's concrete layout declared to Vulkan.
//! - Several typed ranges to support arbitrary writes and must agree with those declarations.
//!
//! To provide type-safety, we need to perform several checks.  The simplest check, the length of
//! byte ranges, must ensure that no ranges exceed byte 128 and that the ranges cover by stage flags
//! add up to the length of the view seen by that stage.
//!
//! This check and all checks leverage the `GpuType` witnesses, which enable traversing from root to
//! leaf type of each field, flattening types into the Slang primitives and scalars and providing
//! byte layout information.
//!
//! By declaring that a type is either push constants or a view, we also declare that it implements
//! the field-type witnesses of `GpuType`.  When we declare that a tuple or struct can write to a
//! range, using an offset and stage flags, we can verify the layout-view agreement from the push
//! constant declaration to the views that the range will write into.
//!
//! Taken together, this decides the `PushConstantRange` relation to the other types:
//!
//! - Push constants have no offset
//! - Push constant views have no offset, but do have a stage flag
//! - Push constant ranges have an offset and flags
//!
//! The dataflow for overall checks can make use of the fact that the pipeline must know all ranges
//! that will be declared to Vulkan and that this decides the overall stage flags and type layouts
//! of everything except writing.  For writing, we can utilize the pipeline as the source of truth.
//! Each implementation of writing to a pipeline's push constants must only agree with data that is
//! already available.
//!
//! ## Trading Size for Indirection
//!
//! Push constants have a limited size (only 128 bytes guaranteed).  If we lack sufficient space
//! within that 128 bytes, we must move some data into SSBOs or UBOs and point to that data.  This
//! is done by indexing into descriptor sets with 4 bytes.  If there are a lot of instances of this
//! control data, it might be more convenient to allocate an inner array and then use 4 bytes of
//! descriptor index alongside 4 bytes to index into that array.
//!
//! **In short, we use either 4 bytes or 8 bytes of data per control data structure**, meaning every
//! stage of a pipeline can use a separate control data structure while still affording us enough
//! room.  Even if every stage of a pipeline requires a separate control data
//! pointer, there are plenty of bytes.
//!
//! ### Indirect Type Agreement
//!
//! All of our SSBO, Image, and UBO types have compile-time checked typed handles. A handle is a
//! 32bit pointer with a descriptor type and an item layout.  The build-time check only ensures that
//! pipelines and shaders are in agreement, that handles agree with what the shader expects behind
//! the pointer.
//!
//! ## Scalar Layouts
//!
//! We enabled 8 bit and 16 bit scalars and scalar block storage extensions, giving us quite a bit
//! more flexibility.  `#[repr(C)]` and scalar data layout are equivalent.
//!
//! If you need std430, you will need to implement the [`Std430`](crate::slang::Std430) for leaf
//! types.  This will propagate through [`GpuType`](crate::slang::GpuType) into all other support.
//!
//! ### Stage Overlap & Range Alignment ⚠️
//!
//! A stage that overlaps other stages may not begin on a sub-32bit value unless that value happens
//! to be 32bit aligned.  You must rearrange fields or pack them into some explicitly aligned type
//! to satisfy this condition.  This restriction is only necessary for `Scalar` layout.
//!
//! ## Type Requirements
//!
//! - Each field used in push constants must implement [`GpuType`](crate::slang::GpuType).
//! - Types used at stage boundaries must be 32bit aligned under the layout rules in use (trivial in
//!   std430).
//!
//! ## Integrating With Slang
//!
//! In the most vanilla case, Slang simply declares a structure type and then to use that structure
//! as the push contant via `[[vk::push_constant]]`.
//!
//! ```slang
//!  struct PushData
//!  {
//!      uint  dispatch_id;
//!      float scale;
//!      uint  ubo_index;
//!      uint  ssbo_index;
//!  };
//!
//!  //!  Global declaration style, probably limits us to one entry per file.
//!  [[vk::push_constant]]
//!  PushData push;
//! ```
//!
//! The declarations can be used on function arguments **per entry point**, which is how you put
//! several entry points into one slang file.
//!
//! ```slang
//!  [shader("compute")]
//!  [numthreads(64, 1, 1)]
//!  void cs_push([[vk::push_constant]] PushData push, uint3 tid : SV_DispatchThreadID) {
//!      // The shader body
//!  }
//! ```
//!
//! Push constant ranges can be sparse per stage.  While adding complexity, this can make impromptu
//! stage combinations less rigid.
//!
//! ```slang
//!   struct PushData {
//!       [[vk::offset(0)]]  float4 uColor; // followed by 48-byte gap
//!       [[vk::offset(64)]] float scale;
//!   }
//!
//!   [[vk::push_constant]]
//!   PushData gPush;
//! ```
//!
//! Regardless of the user-facing API for declaring and writing constant ranges, support must always
//! first ensure the following:
//!
//! - The composite structure has a single coherent layout
//! - The declared views of that layout match the composite layout used in shaders
//!
//! We can write in whatever granularity we want, with whatever user API we want, but Vulkan must
//! always get the correct bytes to each shader program.

// NOTE So.  Where is all the code?  Well, after working out the relationships, there just wasn't a
// lot of need for types beyond vanilla vk::PushConstantRange and GpuType.  Every pipeline needs to
// *at least* push to its own push constants type, so even the need for basic writing moved
// elsewhere.  Unless someone feels the need for writing sub-ranges, this module is not expected to
// grow.  the above documentation will almost all flow into the proc macro docs.  Some of the check
// logic may have a runtime usage for dynamic module reloading, but until that time,
// NEXT struct-write ranges, providing a structure that can be used to write to a range.  The
// struct must agree with the block it will write to.

use crate::internal::*;

/// Vulkan min-spec guarantee is 128 bytes.  Perhaps in a newer Vulkan, the min-spec will expand,
/// but unless you have lots of stages, with indirection, there's little use for more push
/// constants.
pub const PUSH_CONSTANT_MAX_BYTES: usize = 128;

pub mod prelude {
    pub use super::PushConstants;
}

/// Implemented on types that describe a fixed sub-128 byte set of push constants.
pub trait PushConstants<D: DataLayout = Scalar>: GpuType<D> {}

#[cfg(test)]
mod tests {
    use super::*;
}
