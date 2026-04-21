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

use crate::stage::{emit_stage_impls, StageAttr};

// NEXT likely this gets more general for other inline impls.
enum ComputeValue {
    External(Ident),
    Inline(StageAttr),
}

impl Parse for ComputeValue {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Ident) && input.peek2(Token![!]) {
            let macro_name: Ident = input.parse()?;
            if macro_name != "stage" {
                return Err(syn::Error::new(
                    macro_name.span(),
                    format!("expected `stage`, found `{macro_name}`"),
                ));
            }
            input.parse::<Token![!]>()?;
            let inner;
            let lookahead = input.lookahead1();
            if lookahead.peek(syn::token::Paren) {
                syn::parenthesized!(inner in input);
            } else if lookahead.peek(syn::token::Bracket) {
                syn::bracketed!(inner in input);
            } else if lookahead.peek(syn::token::Brace) {
                syn::braced!(inner in input);
            } else {
                return Err(lookahead.error());
            }
            Ok(ComputeValue::Inline(inner.parse()?))
        } else {
            Ok(ComputeValue::External(input.parse()?))
        }
    }
}

/// Parsed fields from the attribute list:
///
/// ```text
/// compute = SomeStage, push = SomePush
/// ```
///
/// Both fields are required; order is flexible.
struct ComputePipelineAttr {
    compute: ComputeValue,
    push: Ident,
}

impl Parse for ComputePipelineAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut compute: Option<ComputeValue> = None;
        let mut push: Option<Ident> = None;

        loop {
            if input.is_empty() {
                break;
            }
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "compute" => {
                    let value: ComputeValue = input.parse()?;
                    if compute.replace(value).is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `compute` field"));
                    }
                }
                "push" => {
                    let value: Ident = input.parse()?;
                    if push.replace(value).is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `push` field"));
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown field `{other}`, expected `compute` or `push`"),
                    ));
                }
            }

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
        compute,
        push: push_type,
    } = syn::parse2::<ComputePipelineAttr>(attr)?;

    let (stage_type, inline_stage_items) = match compute {
        ComputeValue::External(ident) => (ident, quote::quote! {}),
        ComputeValue::Inline(stage_attr) => {
            // Emit Stage<Slot> and StageReflection<Slot> directly onto the
            // pipeline type.  No synthetic type needed.
            let stage_impls = emit_stage_impls(type_name, stage_attr)?;
            (type_name.clone(), stage_impls)
        }
    };

    Ok(quote::quote! {
        #input

        #inline_stage_items

        impl ::mutate_vulkan::pipeline::ComputePipelineSpec for #type_name {
            type Push       = #push_type;
            type LayoutSpec = #push_type;
            type Stage      = #stage_type;
        }
    })
}
