// tests/ui/push/pass_minimal.rs
// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// A single field, no #[visible], produces one all-stages range.

use mutate_macros::*;
use mutate_vulkan::prelude::*;

#[derive(GpuType, Push)]
#[repr(C)]
struct MinimalPush {
    dispatch_id: UInt,
}

fn main() {
    assert_eq!(<MinimalPush as LayoutSpec>::RANGES.len(), 1);
}
