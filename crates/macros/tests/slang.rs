// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Slang Pro Macro Tests
//!
//! Using trybuild.

#[test]
fn descriptor_newtype() {
    let t = trybuild::TestCases::new();
    t.pass("tests/slang/pass/newtype_satisfies_gpu_type.rs");
    t.pass("tests/slang/pass/from_base_into_wrapper.rs");
}
