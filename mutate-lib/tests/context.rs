// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use ash::vk;
use mutate_lib::vulkan;

#[test]
fn test_context_lifecycle() {
    let context = vulkan::context::VkContext::new();
    context.destroy();
}
