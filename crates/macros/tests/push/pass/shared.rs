// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Two stages with identical visible ranges must emit a single range with the combined flags.

use mutate_macros::*;
use mutate_vulkan::slang::UInt;

#[derive(GpuType, Push)]
#[repr(C)]
struct SharedPush {
    #[visible(VERTEX | FRAGMENT)]
    matrix_idx: UInt,
}

fn main() {
    use mutate_vulkan::pipeline::layout::LayoutSpec;
    assert_eq!(<SharedPush as LayoutSpec>::RANGES.len(), 1);
}
