// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pipeline
//!
//! Emits trait implementations and the checks that fan-in const data from the comprising
//! components for final agreement checks.
//!
//! While different kinds of pipelines share a lot of similarity, pipelines for compute, graphics,
//! and ray tracing etc are pretty unique.  To keep the error messages and user-facing APIs from
//! leaking the entry points are kept separate.

use proc_macro2::TokenStream;
use syn::{
    parse::{Parse, ParseStream},
    Ident, Token,
};

/// Parsed fields from the attribute list:
///
/// ```text
/// compute = SomeStage, push = SomePush
/// ```
///
/// Both fields are required; order is flexible.
struct ComputePipelineAttr {
    compute: Ident,
    push: Ident,
}

struct KvIdent {
    key: Ident,
    value: Ident,
}

impl Parse for KvIdent {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let value: Ident = input.parse()?;
        Ok(KvIdent { key, value })
    }
}

impl Parse for ComputePipelineAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut compute: Option<Ident> = None;
        let mut push: Option<Ident> = None;

        // Accept the two kv pairs in any order, separated by commas.
        loop {
            if input.is_empty() {
                break;
            }
            let kv: KvIdent = input.parse()?;
            match kv.key.to_string().as_str() {
                "compute" => {
                    if compute.replace(kv.value).is_some() {
                        return Err(syn::Error::new(kv.key.span(), "duplicate `compute` field"));
                    }
                }
                "push" => {
                    if push.replace(kv.value).is_some() {
                        return Err(syn::Error::new(kv.key.span(), "duplicate `push` field"));
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        kv.key.span(),
                        format!("unknown field `{other}`, expected `compute` or `push`"),
                    ));
                }
            }
            // Consume optional trailing / separating comma.
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(ComputePipelineAttr {
            compute: compute
                .ok_or_else(|| syn::Error::new(input.span(), "missing required field `compute`"))?,
            push: push
                .ok_or_else(|| syn::Error::new(input.span(), "missing required field `push`"))?,
        })
    }
}

/// Express a shader stage as a concrete type.  Implements witness traits that enable downstream
/// type checking.  Accepts shader file name (without extension), stage flags, and entry point.
pub(crate) fn compute_pipeline(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let input = syn::parse2::<syn::DeriveInput>(item)?;
    let type_name = &input.ident;

    let ComputePipelineAttr {
        compute: stage_type,
        push: push_type,
    } = syn::parse2::<ComputePipelineAttr>(attr)?;

    Ok(quote::quote! {
        #input

        impl ::mutate_vulkan::pipeline::ComputePipelineSpec for #type_name {
            type Push       = #push_type;
            type LayoutSpec = #push_type;
            type Stage      = #stage_type;
        }
    })
}
