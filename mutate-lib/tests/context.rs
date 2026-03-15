// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use ash::vk;
use mutate_lib::vulkan;

#[test]
fn test_context_lifecycle() {
    // XXX not actually good
    let vk_context = vulkan::context::VkContext::new();
    let context = vulkan::context::DeviceContext::new(&vk_context);
    context.destroy();
    vk_context.destroy();
}
