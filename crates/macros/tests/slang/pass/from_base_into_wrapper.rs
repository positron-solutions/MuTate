// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must compile: From<base> → wrapper conversions all exist.
// The reverse (From<wrapper> → base) is intentionally absent — tested in a fail case.

use mutate_vulkan::slang::prelude::*;

slang_newtype!(Hotness, Float, "Hotness");

fn main() {
    // Scalar primitives accept their base Rust type
    let _f: Float = Float::from(1.0f32);
    let _u: UInt16 = UInt16::from(42u16);
    let _b: Bool = Bool::from(true);
    let _h: Half = Half::from(half::f16::from_f32(1.0));

    // Newtypes accept their inner slang scalar
    // XXX This is being put off because I want to check introspection data and actually understand
    // whether or not Slang new type introspection will point to the fundamental type or any
    // transitive newtype wrappers between (for single field struct style newtype wrappers).
    // let _t: Hotness = Hotness::from(Float::from(98.6f32));

    // Into<> works as the mirror (no separate impl needed)
    let _f2: Float = 1.0f32.into();
    // XXX same as above
    // let _t2: Hotness = Float::from(98.6f32).into();
}
