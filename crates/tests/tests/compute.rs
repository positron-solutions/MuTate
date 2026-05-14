// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Compute Proc Macro Tests
//!
//! Using trybuild.

// DEBT centralize these macros
macro_rules! trybuild_pass {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.pass(concat!("tests/compute/pass/", stringify!($name), ".rs"));
        }
    };
}

macro_rules! trybuild_fail {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.compile_fail(concat!("tests/compute/fail/", stringify!($name), ".rs"));
        }
    };
}

// basic smoke tests
trybuild_pass!(independently_declared);
trybuild_pass!(with_inline_stage);
trybuild_pass!(with_inline_push);
