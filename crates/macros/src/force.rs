// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Force (Correctness)
//!
//! *Luminous beings are we, not this crude matter!* - Yoda
//!
//! Implicitly adding things is not always the best style, but since many of our implementations
//! would have to instead error in all cases where the user does something else, we might as well
//! just do what is required, aka forcing the decision.

use proc_macro2::Span;
use syn::{Attribute, Error, Result};

/// If the correct repr has not been added, indicate an error.
pub(crate) fn assert_repr(attrs: &[syn::Attribute], span: Span) -> syn::Result<()> {
    match scan_repr(attrs, span) {
        ReprStatus::HasC | ReprStatus::HasTransparent => Ok(()),
        ReprStatus::HasConflict(s) => Err(syn::Error::new(
            s,
            "#[derive(GpuType)] is incompatible with `repr(packed)` and `repr(align(...))`; \
             use `#[repr(C)]` instead",
        )),
        ReprStatus::Absent => Err(syn::Error::new(
            span,
            "#[derive(GpuType)] requires `#[repr(C)]` or `#[repr(transparent)]` \
             to guarantee a stable, C-compatible layout",
        )),
    }
}

/// Add or detect `repr(C)` but error if there is some conflict with another choice of `repr`.
pub(crate) fn ensure_repr(attrs: &mut Vec<syn::Attribute>, span: Span) -> syn::Result<()> {
    match scan_repr(attrs, span) {
        ReprStatus::HasC | ReprStatus::HasTransparent => Ok(()),
        ReprStatus::HasConflict(s) => Err(syn::Error::new(
            s,
            "`repr(packed)` and `repr(align(...))` are incompatible with GPU layout requirements; \
             remove the conflicting repr so that `#[repr(C)]` can be applied",
        )),
        ReprStatus::Absent => {
            attrs.push(syn::parse_quote!(#[repr(C)]));
            Ok(())
        }
    }
}

enum ReprStatus {
    HasC,
    HasTransparent,
    HasConflict(Span),
    Absent,
}

fn scan_repr(attrs: &[syn::Attribute], span: Span) -> ReprStatus {
    let mut found_c = false;
    let mut found_transparent = false;
    let mut conflict: Option<Span> = None;

    for attr in attrs {
        if !attr.path().is_ident("repr") {
            continue;
        }
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("C") {
                found_c = true;
            } else if meta.path.is_ident("transparent") {
                found_transparent = true;
            } else if meta.path.is_ident("packed") || meta.path.is_ident("align") {
                let s = meta.path.get_ident().map(|i| i.span()).unwrap_or(span);
                conflict = Some(s);
            }
            Ok(())
        });
    }

    if let Some(s) = conflict {
        return ReprStatus::HasConflict(s);
    }
    if found_transparent {
        return ReprStatus::HasTransparent;
    }
    if found_c {
        return ReprStatus::HasC;
    }
    ReprStatus::Absent
}
