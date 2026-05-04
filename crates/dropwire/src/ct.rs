// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Compile Time
//!
//! The default implementations of dropwire use a ZST-based implementation.  If the program is
//! compiled with the ability to drop a guarded type, the dropwire will be tripped.  This rules out
//! all potential for erroneous drops at the expense of providing less information about the drop
//! site.

use std::marker::PhantomData;

// MAYBE variance needs further inspection.
pub struct DropWire<O>(PhantomData<fn() -> O>);

impl<O> DropWire<O> {
    /// Create an armed DropWire to place in a guarded value you are constructing.
    #[inline(always)]
    pub fn armed() -> Self {
        DropWire(PhantomData)
    }

    /// Disarm the wire, attesting that a valid consuming method has been called.
    ///
    /// This must be called after destructuring the guarded type and before the binding holding
    /// this wire goes out of scope.  Only blessed private methods of the guarded type should
    /// call this.
    #[inline(always)]
    pub fn disarm(self) {
        //
        std::mem::forget(self);
    }
}

impl<O> Drop for DropWire<O> {
    #[inline(always)]
    fn drop(&mut self) {
        const {
            panic!("DropWire tripped: consumer did not call disarm on a guarded value");
        }
    }
}

impl<O> std::fmt::Debug for DropWire<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DropWire<{}>(armed)", std::any::type_name::<O>())
    }
}
