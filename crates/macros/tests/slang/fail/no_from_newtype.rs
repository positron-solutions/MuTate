// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must NOT compile: From<Hotness> to Float32 is intentionally absent.
// Newtypes are opaque in the outward direction — call .into_inner() explicitly.

use mutate_vulkan::slang::prelude::*;

slang_newtype!(Hotness, Float32, "Hotness");

fn main() {
    let h = Hotness::from(Float32::from(98.6f32));
    let _inner: Float32 = Float32::from(h); //~ ERROR
}
