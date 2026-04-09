// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Stage Proc Macro Tests
//!
//! Using trybuild.

// DEBT centralize these macros
macro_rules! trybuild_pass {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.pass(concat!("tests/stage/pass/", stringify!($name), ".rs"));
        }
    };
}

macro_rules! trybuild_fail {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.compile_fail(concat!("tests/stage/fail/", stringify!($name), ".rs"));
        }
    };
}

// basic smoke tests
trybuild_pass!(compute);

trybuild_fail!(shader_missing);
