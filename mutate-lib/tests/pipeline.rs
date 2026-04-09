// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use mutate_macros::shader;
use mutate_vulkan as vulkan;

#[test]
fn stage_create() {
    vulkan::with_context!(|context| {
        let shader = vulkan::resource::shader::ShaderModule::load(&context, "test/compute");
    })
}

#[test]
fn declare_stage() {
    // Just a tripwire the stage macro.  Comprehensive testing upstream in the macros crate.
    #[shader("test/hello_compute", COMPUTE, c"main")]
    struct GoodStage {}
}

#[test]
fn declare_pipeline() {}
