// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pipeline
//!
//! ⚠️ This documentation is prospective and describes a design under active implementation.  These
//! docs will migrate into the public API as things shake out.
//!
//! Pipelines tie together stages and a layout, forming something we can dispatch.  There are
//! graphics and compute pipelines.  Compute is vastly simpler, so the types diverge a bit.
//!
//! ## Specs
//!
//! We can talk about pipelines without instantiating them, especially to represent partial
//! hydration states.  These are Specs.  Specs contain enough information to execute compile &
//! runtime host-GPU agreement checks and then to hydrate into a dispatch-ready pipeline.
//!
//! While Specs can be referenced by name, two identical specs at runtime will de-duplicate over the
//! hashes of the resources they load.  This can save some PSO compiling
//!
//! ## Declaration
//!
//! Pipelines can be declared conveniently either by assembling the pieces declared elsewhere or by
//! declaring them inline.
//!
//! ### Independent Declaration Style
//!
//! Declare layouts and stages separately, then combine them into a pipeline by name:
//!
//! ```ignore
//! #[stage(Vertex, "lighting/vertex")]
//! struct LightingVert;
//!
//! #[stage(Fragment, "lighting/fragment")]
//! struct LightingFrag;
//!
//! #[push]
//! struct ScenePush {
//!     #[visible(VERTEX | FRAGMENT)]
//!     matrix_idx: UInt32,
//!     #[visible(FRAGMENT)]
//!     light_idx:  UInt32,
//!     frame_time: F32,
//! }
//!
//! #[pipeline(Graphics)]
//! struct ScenePipeline {
//!     vert: LightingVert,
//!     frag: LightingFrag,
//!     push: ScenePush,
//! }
//! ```
//!
//! When sharing push constants or stages over several pipelines, using stages by name
//!
//! ### Inline Declaration Style
//!
//! Use the bang macros `stage!` and `push!` to declare those types inline.
//!
//! ```ignore
//! #[pipeline(Graphics)]
//! struct ScenePipeline {
//!     vert: stage!(Vertex, "lighting/vertex"),
//!     frag: stage!(Fragment, "lighting/fragment"),
//!     push: push! {
//!         #[visible(VERTEX | FRAGMENT)]
//!         matrix_idx: UInt32,
//!         #[visible(FRAGMENT)]
//!         light_idx:  UInt32,
//!         frame_time: F32,
//!     },
//! }
//! ```
//!
//! Inline declarations will be placed into a private module so that their unique types do not
//! pollute scope.
//!
//! ## Layouts
//!
//! We only support a single static descriptor table, defined in
//! [`descriptors`](crate::context::descriptors).  For the rest, we can focus on push constants.
//! To enforce type consistency, there is one definitive type for the overall push constants layout.
//! We can emit several views for this type, each a concrete push constant range that provides
//! visibility for pipeline stages.
//!
//! Writing to PushConstants is currently only supported for whole-range writes.  Sub-structures for
//! sub-ranges, sparse ranges, and sum ranges over tuple types are some possible directions to
//! expand support, but if nobody finds the time to do it, it probably isn't very useful.

use std::marker::PhantomData;

use crate::internal::*;
use crate::resource::shader;

pub mod layout;
pub mod push;
pub mod stage;

/// Hydrated graphics pipeline ready to dispatch.
pub struct GraphicsPipeline<S: GraphicsPipelineSpec> {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    _marker: PhantomData<S>,
    // XXX Extra fields for graphics things
}

/// Describes how to build and type-check a graphics pipeline.
pub trait GraphicsPipelineSpec {
    type Push: push::PushConstants;
    const STAGES: &'static [stage::StageSpec];
}

impl<S> GraphicsPipeline<S>
where
    S: GraphicsPipelineSpec,
{
    pub fn new(context: &DeviceContext) -> Result<Self, VulkanError> {
        todo!()
    }

    // XXX needs pipeline state (struct TBD) an a RecordingSlot<Graphics>
    pub fn draw(&self, device_context: &DeviceContext) {
        todo!()
    }
}

/// Describes how to build and type-check a compute pipeline.
pub trait ComputePipelineSpec {
    // No way in hell all of these fields stay put.  Just can't figure out the tension at this
    // point.  Keep coding and let the problems drive 😈.
    type Push: push::PushConstants;
    type LayoutSpec: layout::LayoutSpec<Push = Self::Push>;
    const STAGE: &'static stage::StageSpec;
}

/// Hydrated compute pipeline ready to dispatch.  Retains a type-level connection with the spec to
/// carry forward layout and stage information.
struct ComputePipeline<S: ComputePipelineSpec> {
    pipeline: vk::Pipeline,
    layout: layout::Layout<S::LayoutSpec>,
    _marker: PhantomData<S>,
}

impl<S: ComputePipelineSpec> ComputePipeline<S> {
    pub fn new(context: &DeviceContext) -> Result<Self, VulkanError> {
        let device = context.device();
        // Compute ranges only have one stage and only need one range.
        let layout = layout::Layout::<S::LayoutSpec>::new(context)?;

        // NOTE the shader module is still just half-baked fumbling in the dark at the shape of the
        // async loading code.  Not going to live long.
        let shader = shader::ShaderModule::load(context, S::STAGE.name)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(*shader)
            .name(S::STAGE.entry);

        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout.raw());
        // NEXT PSO compiling can take a while and definitely should be queued into background via resources.
        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
                .map_err(|huh| huh.1)?[0] // XXX 🤠
        };
        Ok(Self {
            pipeline,
            layout,
            _marker: PhantomData,
        })
    }

    // XXX typed recording slot
    // XXX possibly keep a device borrow on recording slots?
    pub fn push(&self, device: &ash::Device, cb: vk::CommandBuffer, data: &S::Push) {
        self.layout.push(device, cb, data);
    }

    /// Bind and dispatch this pipeline
    // NEXT bounds checked vs bounds guaranteed dispatch geometries for compute shaders.  There is a
    // correlation between `[numthreads(4, 8, 1)]` style geometry in the shader declaration and
    // dividing input geometry by 4 and 8 for the dispatch.  We should find a way to integrate
    // reflection and bounds checking (or omission).  Unchecked can use const expressions to ensure
    // perfect geometry by type contract.
    pub fn dispatch(&self, device: &ash::Device, cb: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_dispatch(cb, x, y, z);
        }
    }
}
