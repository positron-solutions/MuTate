// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "vulkan")]

use mutate_macros::*;
use mutate_vulkan as vulkan;
use mutate_vulkan::prelude::*;

#[test]
fn stage_create() {
    vulkan::with_context!(|context| {
        let shader = vulkan::resource::shader::ShaderModule::load(&context, "test/compute");
    })
}

#[test]
fn declare_stage() {
    // Just a tripwire the stage macro.  Comprehensive testing upstream in the macros crate.
    #[stage("test/hello_compute", Compute, c"main")]
    struct GoodStage {}
}

#[test]
fn declare_pipeline() {
    #[stage("test/hello_compute", Compute, c"main")]
    pub struct ComputeStage {}

    #[derive(GpuType, Push)]
    #[repr(C)]
    pub struct ComputeConstants {
        #[visible(Compute)]
        foo: UInt,
    }

    #[compute_pipeline(
        compute = ComputeStage,
        push    = ComputeConstants,
    )]
    pub struct TestPipeline;

    vulkan::with_context!(|device_ctx| {
        // instantiate the pipeline!
        let pipeline: ComputePipeline<TestPipeline> =
            ComputePipeline::<TestPipeline>::new(&device_ctx)
                .expect("pipeline instantiation failed");

        pipeline.destroy(&device_ctx);
    })
}
