// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Push Proc Macro Tests
//!
//! Using trybuild.

// NEXT centralize these macros
macro_rules! trybuild_pass {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.pass(concat!("tests/push/pass/", stringify!($name), ".rs"));
        }
    };
}

macro_rules! trybuild_fail {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.compile_fail(concat!("tests/push/fail/", stringify!($name), ".rs"));
        }
    };
}

// Smoke Tests
trybuild_pass!(empty);
trybuild_pass!(minimal);
trybuild_pass!(split);
trybuild_pass!(shared);
