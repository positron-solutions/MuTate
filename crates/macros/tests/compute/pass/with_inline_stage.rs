// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Two fields, each visible to a distinct single stage.
// This is the case where the generated RANGES must contain exactly two entries
// and the ranges must not share a stage flag (VUID-00292).

use mutate_macros::*;
use mutate_vulkan::prelude::*;

#[derive(GpuType, Push)]
#[repr(C)]
pub struct ComputeConstants {
    #[visible(Compute)]
    foo: UInt,
}

#[compute_pipeline(
    compute = stage!("test/hello_compute", Compute, c"main"),
    push = ComputeConstants,
)]
pub struct TestPipeline;

fn main() {
    use ash::vk::ShaderStageFlags;
    use mutate_vulkan::pipeline::layout::LayoutSpec;
    use mutate_vulkan::pipeline::ComputePipelineSpec;

    fn assert_push<S: ComputePipelineSpec<Push = ComputeConstants>>() {}
    fn assert_stage<S: ComputePipelineSpec<Stage = TestPipeline>>() {}
    assert_push::<TestPipeline>();
    assert_stage::<TestPipeline>();

    let ranges = <ComputeConstants as LayoutSpec>::RANGES;
    assert_eq!(
        ranges.len(),
        1,
        "expected exactly one push constant range for ComputeConstants",
    );
    assert_eq!(
        ranges[0].stage_flags,
        ShaderStageFlags::COMPUTE,
        "the single range must be visible to COMPUTE only",
    );
    assert_eq!(
        ranges[0].offset, 0,
        "single-field push constant range must start at offset 0",
    );
    assert_eq!(
        ranges[0].size,
        std::mem::size_of::<u32>() as u32,
        "UInt is 4 bytes; range size must match",
    );
}
