// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must compile: From<base> → wrapper conversions all exist.
// The reverse (From<wrapper> → base) is intentionally absent — tested in a fail case.

use mutate_vulkan::slang::prelude::*;

slang_newtype!(Hotness, Float32, "Hotness");

fn main() {
    // Scalar primitives accept their base Rust type
    let _f: Float32 = Float32::from(1.0f32);
    let _u: UInt16 = UInt16::from(42u16);
    let _b: Bool = Bool::from(true);
    let _h: Float16 = Float16::from(half::f16::from_f32(1.0));

    // Newtypes accept their inner slang scalar
    let _t: Hotness = Hotness::from(Float32::from(98.6f32));

    // Into<> works as the mirror (no separate impl needed)
    let _f2: Float32 = 1.0f32.into();
    let _t2: Hotness = Float32::from(98.6f32).into();
}
