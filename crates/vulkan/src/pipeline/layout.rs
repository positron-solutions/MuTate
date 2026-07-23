// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pipeline Layout
//!
//! Layouts are basically the combination of descriptors and push constants that are input for a
//! pipeline.  Since we only support one kind of descriptor table, our layouts don't actually vary
//! over descriptors, only push constants.  Correspondingly, once we know any `PushConstants` type,
//! we also know how to hydrate a layout from just a logical device and we also know how to write
//! layout-compatible push constants.

use std::marker::PhantomData;

use crate::internal::*;

use super::push;

pub trait LayoutSpec {
    type D: DataLayout;
    type Push: push::PushConstants<Self::D>;
    const RANGES: &'static [vk::PushConstantRange];
}

pub struct Layout<S: LayoutSpec> {
    raw: vk::PipelineLayout,
    _spec: PhantomData<S>,
}

impl<S: LayoutSpec> Layout<S> {
    pub fn new(device: &Device) -> Result<Self, VulkanError> {
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(S::RANGES)
            .set_layouts(device.descriptors.layout());
        Ok(Self {
            raw: unsafe { device.as_raw().create_pipeline_layout(&layout_ci, None)? },
            _spec: PhantomData,
        })
    }

    pub fn as_raw(&self) -> vk::PipelineLayout {
        self.raw
    }

    // DEBT Lifetime Agreement & Destructor.
    pub fn destroy(self, device: &Device) {
        unsafe { device.as_raw().destroy_pipeline_layout(self.raw, None) }
    }

    /// Push all bytes of the PushConstants
    pub fn push(&self, device: &ash::Device, cb: vk::CommandBuffer, data: &S::Push) {
        // ROLL once again, I am asking for your consts https://github.com/rust-lang/rust/issues/132980
        // PUSH_CONSTANT_MAX_BYTES is the Vulkan-spec hard ceiling (128 nom sayan?)
        let mut buf = [0u8; push::PUSH_CONSTANT_MAX_BYTES];
        let packed = <S::Push as Pack<S::D>>::PACKED_SIZE;
        <S::Push as Pack<S::D>>::pack_into(data, &mut buf);
        unsafe {
            device.cmd_push_constants(cb, self.raw, vk::ShaderStageFlags::ALL, 0, &buf[..packed])
        };
    }
}
