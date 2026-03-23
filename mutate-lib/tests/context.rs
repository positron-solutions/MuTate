// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use ash::vk;
use mutate_lib::vulkan;

#[test]
fn test_context_lifecycle() {
    vulkan::with_context!(|context| {});
}

#[test]
fn test_device_context_lifecycle() {
    vulkan::with_context!(|vk_context, device_context| {});
}
