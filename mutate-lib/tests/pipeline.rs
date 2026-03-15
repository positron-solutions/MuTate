// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use mutate_vulkan as vulkan;

#[test]
fn stage_create() {
    vulkan::with_context!(|context| {
        let shader = vulkan::resource::shader::ShaderModule::load("test/compute", &context);
    })
}
