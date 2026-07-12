// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// This test was created from a buggy set of push constants.

use mutate_lib::prelude::*;
use mutate_lib::vulkan;

#[derive(GpuType, Push)]
#[repr(C)]
struct RawRingPushConstants {
    pub left_channel: DeviceAddress,
    pub right_channel: DeviceAddress,
    pub capacity: UInt,
    pub counter: UInt,
    pub window_width: Float,
    pub window_height: Float,
    pub output_idx: SsboIdx,
}

fn main() {
    use ash::vk::ShaderStageFlags;
    use vulkan::pipeline::layout::LayoutSpec;

    let ranges = <RawRingPushConstants as LayoutSpec>::RANGES;

    assert_eq!(ranges.len(), 1);

    // Vertex range: UInt at scalar offset 0, size 4.
    assert_eq!(ranges[0].offset, 0);
    assert_eq!(ranges[0].size, 40);
}
