// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Vulkan Utils
//!
//! Junk drawer.  Move things out when there is a place for them to belong.
//!

use ash::vk;

use crate::VulkanError;

/// Implementation is just a first pass to get going.
// Probably goes in a dedicated memory management module.
// DEBT binding
pub fn find_memory_type_index(
    mem_req: &vk::MemoryRequirements,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        let type_supported = (mem_req.memory_type_bits & (1 << i)) != 0;
        let props = mem_props.memory_types[i as usize].property_flags;

        if type_supported && props.contains(required) {
            return Some(i);
        }
    }

    None
}

/// Layouts and descriptor management in general are annoying.
// Probably goes into a dedicated binding module.
// DEBT memory
pub fn descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, VulkanError> {
    let bindings = [
        vk::DescriptorSetLayoutBinding {
            binding: 0, // spectrum
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        },
        vk::DescriptorSetLayoutBinding {
            binding: 1, // output buffer
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        },
    ];

    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count: bindings.len() as u32,
        p_bindings: bindings.as_ptr(),
        ..Default::default()
    };

    Ok(unsafe { device.create_descriptor_set_layout(&layout_info, None)? })
}
