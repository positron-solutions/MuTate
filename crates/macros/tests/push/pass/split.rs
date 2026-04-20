// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Two fields, each visible to a distinct single stage.
// This is the case where the generated RANGES must contain exactly two entries
// and the ranges must not share a stage flag (VUID-00292).

use mutate_macros::*;
use mutate_vulkan::slang::{Float, UInt};

#[derive(GpuType, Push)]
#[repr(C)]
struct SplitPush {
    #[visible(Vertex)]
    vertex_only: UInt,
    #[visible(Fragment)]
    fragment_only: Float,
}

fn main() {
    use ash::vk::ShaderStageFlags;
    use mutate_vulkan::pipeline::layout::LayoutSpec;

    let ranges = <SplitPush as LayoutSpec>::RANGES;

    // Two distinct ranges, one per stage.
    assert_eq!(ranges.len(), 2);

    // Neither range overlaps in stage flags (VUID-00292).
    assert_eq!(
        ranges[0].stage_flags & ranges[1].stage_flags,
        ShaderStageFlags::empty(),
    );

    // Vertex range: UInt at scalar offset 0, size 4.
    assert_eq!(ranges[0].stage_flags, ShaderStageFlags::VERTEX);
    assert_eq!(ranges[0].offset, 0);
    assert_eq!(ranges[0].size, 4);

    // Fragment range: Float at scalar offset 4, size 4.
    // This is the non-zero offset case the old hardcoded-zero emit could not produce.
    assert_eq!(ranges[1].stage_flags, ShaderStageFlags::FRAGMENT);
    assert_eq!(ranges[1].offset, 4);
    assert_eq!(ranges[1].size, 4);
}
