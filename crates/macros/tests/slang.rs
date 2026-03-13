// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Slang Pro Macro Tests
//!
//! Using trybuild.

// NEXT centralize these macros
macro_rules! trybuild_pass {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.pass(concat!("tests/slang/pass/", stringify!($name), ".rs"));
        }
    };
}

macro_rules! trybuild_fail {
    ($name:ident) => {
        #[test]
        fn $name() {
            let t = trybuild::TestCases::new();
            t.compile_fail(concat!("tests/slang/fail/", stringify!($name), ".rs"));
        }
    };
}

// Behavioral inventory, all the things the macros should give us.
trybuild_pass!(scalar);
trybuild_pass!(newtype);
trybuild_pass!(descriptor);
trybuild_pass!(descriptor_newtype);

// Granular
trybuild_pass!(newtype_satisfies_gpu_type);
trybuild_pass!(from_base_into_wrapper);
trybuild_pass!(device_address_newtype_and_null);

// Forbidden
trybuild_fail!(no_from_slang);
trybuild_fail!(no_from_newtype);
trybuild_fail!(descriptor_newtype_kind_mixing);
