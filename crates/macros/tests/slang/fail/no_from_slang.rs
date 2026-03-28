// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must NOT compile: From<Float> to f32 is intentionally absent.
// Only the base to wrapper direction is provided.

use mutate_vulkan::slang::prelude::*;

fn main() {
    let wrapped = Float::from(1.0f32);
    let _raw: f32 = f32::from(wrapped);
}
