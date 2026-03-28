// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must NOT compile: a StorageBuffer descriptor newtype cannot be constructed
// from a SampledImage base — the witness bound catches the kind mismatch
// at the From impl site.

use mutate_vulkan::slang::prelude::*;

descriptor_newtype!(MyStorageIdx, SsboIdx, "MyStorageIdx");

fn main() {
    let sampled = SampledImageIdx::new(0);
    let _wrong: MyStorageIdx = sampled.into(); //~ ERROR
}
