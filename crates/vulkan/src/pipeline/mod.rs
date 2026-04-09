// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pipeline
//!
//! This stub should be filled in with stages and other types that are strongly bound to realized
//! pipelines.
//!
//! ## Layouts
//!
//! We only support a single static descriptor table, defined in
//! [`descriptors`](crate::context::descriptors).
//!
//! Push constant ranges would be defined entirely by stage reflection if no stages were ever
//! intended to share data or write new push constants via partial updates of sub-structures instead
//! of always updating the entire 128byte set.

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
    const STAGE: &'static stage::StageSpec;
}

/// Hydrated compute pipeline ready to dispatch.  Retains a type-level connection with the spec to
/// carry forward layout and stage information.
struct ComputePipeline<S: ComputePipelineSpec> {
    pipeline: vk::Pipeline,
    layout: layout::Layout<S::Push>,
    _marker: PhantomData<S>,
}

impl<S: ComputePipelineSpec> ComputePipeline<S> {
    pub fn new(context: &DeviceContext) -> Result<Self, VulkanError> {
        let device = context.device();
        // Compute ranges only have one stage and only need one range.
        let layout = layout::Layout::<S::Push>::new(context)?;

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
    pub fn dispatch(&self, cmd: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        todo!()
    }
}
