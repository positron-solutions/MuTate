// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Stage Proc Macro Tests
//!
//! Using trybuild.

#[test]
fn test_procmacro_stage_shader_exists() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/stage/fail-shader-missing.rs");
}
