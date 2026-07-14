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
