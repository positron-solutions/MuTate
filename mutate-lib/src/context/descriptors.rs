// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Descriptors
//!
//! (◕‿◕)︵‿︵‿︵‿︵┻━┻
//!
//! (◕‿◕)ノ彡☆
//!
//! > You make a single descriptor set (or even better: descriptor heap!) with 2^20 texture
//! > descriptors and  2^12 sampler descriptors. You use buffer device address instead of buffer
//! > descriptors.  You pass around pointer and indices into the set via push constants. And then
//! > you never think about descriptor sets ever again. **- Afiery1**
//!
//! The implementation of this scheme requires thinking just one more time.  If it's not clear, the
//! resulting geometry is:
//!
//!   `[DescriptorSet[DescriptorArray[Index]]]`
//!
//! A descriptor array is a descriptor for an array, not an array of descriptors... well, maybe it
//! is an array of descriptors, but the point is we want to only care about indexes.  The size of
//! this array is:
//!
//! `[No. of Vulkan Applications (one)][No. of types (a few)][No. of items (a zillion)]`
//!
//! **So we have one array for each type and we index into it.**  Simple!
//!
//! The rest of the complexity pretty much boils down to re-claiming recycled descriptors.  We will
//! make one structure to track all of our descriptors.  It initializes with a big descriptor set.
//! It has a static fixed size because any kind of dynamic growth messes up the descriptor slots and
//! forces us to think about descriptors.  Okay, glad we are experts at Vulkan now!

use std::collections::VecDeque;

use ash::vk;

use crate::prelude::*;

// NEXT Type indexes
// NEXT Methods to hand out and recycle indexes, likely guarded through the context interface.
// Planning on coordinating Image and Buffer creation because not having descriptors would make them
// kind of useless.
pub struct Descriptors {
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,
    freelist_sampled_images: VecDeque<u32>,
    freelist_storage_images: VecDeque<u32>,
    freelist_ubos: VecDeque<u32>,
    freelist_ssbos: VecDeque<u32>,
}

impl Descriptors {
    pub fn new(device: &ash::Device, pool: &ash::vk::DescriptorPool) -> Result<Self, MutateError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(256) // max entries in freelist
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*pool)
            .set_layouts(&layouts);

        let set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        Ok(Self {
            set,
            layout,
            freelist_sampled_images: VecDeque::with_capacity(256),
            freelist_storage_images: VecDeque::with_capacity(256),
            freelist_ubos: VecDeque::with_capacity(256),
            freelist_ssbos: VecDeque::with_capacity(256),
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
