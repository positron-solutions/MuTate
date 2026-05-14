// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Two stages with identical visible ranges must emit a single range with the combined flags.

use mutate_lib::prelude::*;
use mutate_lib::vulkan;

#[derive(GpuType, Push)]
#[repr(C)]
struct SharedPush {
    #[visible(Vertex | Fragment)]
    matrix_idx: UInt,
}

fn main() {
    use vulkan::pipeline::layout::LayoutSpec;
    assert_eq!(<SharedPush as LayoutSpec>::RANGES.len(), 1);
}
