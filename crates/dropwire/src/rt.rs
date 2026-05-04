// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Runtime
//!
//! This is an alternative implementation for debugging.  It uses the same API and semantics, but
//! fails with more information about the construction and drop site of each wire instance.  If this
//! does not find the erroneous drop, it will rule out the code paths you executed, narrowing the
//! search for a compile-time error.

use std::marker::PhantomData;
use std::panic::Location;

pub struct DropWire<O> {
    location: &'static Location<'static>,
    _owner: PhantomData<fn() -> O>,
}

impl<O> DropWire<O> {
    #[inline]
    #[track_caller]
    pub fn armed() -> Self {
        Self {
            location: Location::caller(),
            _owner: PhantomData,
        }
    }

    #[inline(always)]
    pub fn disarm(self) {
        std::mem::forget(self);
    }
}

impl<O> Drop for DropWire<O> {
    fn drop(&mut self) {
        panic!(
            "DropWire<{}> tripped: consumer did not call disarm\n\
             \twire armed at: {}",
            std::any::type_name::<O>(),
            self.location,
        );
    }
}

impl<O> std::fmt::Debug for DropWire<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DropWire<{}>(armed @ {})",
            std::any::type_name::<O>(),
            self.location,
        )
    }
}
