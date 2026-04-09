// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pipeline Layout
//!
//! Layouts are basically the combination of descriptors and push constants that are input for a
//! pipeline.  Since we only support one kind of descriptor table, our layouts don't actually vary
//! over descriptors, only push constants.  Correspondingly, once we know any `PushConstants` type,
//! we also know how to hydrate a layout from just a logical device and we also know how to write
//! layout-compatible push constants.

use crate::internal::*;

use super::push;

pub struct Layout<P: push::PushConstants> {
    raw: vk::PipelineLayout,
    _push: std::marker::PhantomData<P>,
}

impl<P: push::PushConstants> Layout<P> {
    pub fn new(device_context: &DeviceContext) -> Result<Self, VulkanError> {
        let range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::ALL)
            .offset(0)
            .size(P::SIZE as u32);
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(std::slice::from_ref(&range))
            .set_layouts(device_context.descriptors.layout());
        let raw = unsafe {
            device_context
                .device()
                .create_pipeline_layout(&layout_ci, None)?
        };
        Ok(Self {
            raw,
            _push: std::marker::PhantomData,
        })
    }

    pub fn raw(&self) -> vk::PipelineLayout {
        self.raw
    }

    // DEBT Lifetime Agreement & Destructor.
    pub fn destroy(self, device_context: &DeviceContext) {
        unsafe {
            device_context
                .device()
                .destroy_pipeline_layout(self.raw, None)
        }
    }
}
