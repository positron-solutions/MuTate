// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Two fields, each visible to a distinct single stage.
// This is the case where the generated RANGES must contain exactly two entries
// and the ranges must not share a stage flag (VUID-00292).

use mutate_macros::*;
use mutate_vulkan::prelude::*;

#[stage("test/hello_compute", Compute, c"main")]
pub struct ComputeStage {}

#[derive(GpuType, Push)]
#[repr(C)]
pub struct ComputeConstants {
    foo: UInt,
}

#[compute_pipeline(
    compute = ComputeStage,
    push = ComputeConstants,
)]
pub struct ComputePipeline;

fn main() {
    use ash::vk::ShaderStageFlags;
    use mutate_vulkan::pipeline::layout::LayoutSpec;
}
