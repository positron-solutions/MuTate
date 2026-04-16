// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_macros::*;

#[derive(GpuType, Push)]
#[repr(C)]
struct Empty {}

fn main() {}
