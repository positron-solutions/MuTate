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
fn stage_declare() {
    // Just a tripwire the stage macro.  Comprehensive testing upstream in the macros crate.
    #[stage("test/hello_compute", Compute, c"main")]
    struct GoodStage {}
}

#[test]
fn pipeline_declare() {
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
        let pipeline: ComputePipeline<TestPipeline> =
            ComputePipeline::<TestPipeline>::new(&device_ctx)
                .expect("pipeline instantiation failed");

        pipeline.destroy(&device_ctx);
    })
}

#[test]
fn pipeline_declare_inline_stage() {
    #[derive(GpuType, Push)]
    #[repr(C)]
    pub struct ComputeConstants {
        #[visible(Compute)]
        foo: UInt,
    }

    #[compute_pipeline(
        compute = stage!("test/hello_compute", Compute, c"main"),
        push    = ComputeConstants,
    )]
    pub struct TestPipeline;

    vulkan::with_context!(|device_ctx| {
        let pipeline: ComputePipeline<TestPipeline> =
            ComputePipeline::<TestPipeline>::new(&device_ctx)
                .expect("pipeline instantiation failed");

        pipeline.destroy(&device_ctx);
    })
}

#[test]
fn pipeline_declare_inline_push() {
    #[stage("test/hello_compute", Compute, c"main")]
    pub struct ComputeStage {}

    #[compute_pipeline(
        compute = ComputeStage,
        push = push!(ComputeConstants {
            #[visible(Compute)]
            foo: UInt,
        }),
    )]
    pub struct TestPipeline;

    vulkan::with_context!(|device_ctx| {
        let pipeline: ComputePipeline<TestPipeline> =
            ComputePipeline::<TestPipeline>::new(&device_ctx)
                .expect("pipeline instantiation failed");

        pipeline.destroy(&device_ctx);
    })
}

#[test]
fn pipeline_declare_inline_all() {
    #[compute_pipeline(
        compute = stage!("test/hello_compute", Compute, c"main"),
        push = push!(ComputeConstants {
            #[visible(Compute)]
            foo: UInt,
        }),
    )]
    pub struct TestPipeline;

    vulkan::with_context!(|device_ctx| {
        let pipeline: ComputePipeline<TestPipeline> =
            ComputePipeline::<TestPipeline>::new(&device_ctx)
                .expect("pipeline instantiation failed");

        pipeline.destroy(&device_ctx);
    })
}
