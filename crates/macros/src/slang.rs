// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Slang
//!
//! GpuType derivation implementation.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Data, DeriveInput, Fields, LitStr, Token,
};

// XXX test with a structure that requires alignment

struct GpuTypeAttr {
    /// The user can provide a name different than the Rust type.  The Slang name will be used to
    /// check reflection agreement with semantic types.
    slang_name: Option<LitStr>,
}

struct KvPair {
    key: syn::Ident,
    value: LitStr,
}

impl Parse for KvPair {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: syn::Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let value: LitStr = input.parse()?;
        Ok(KvPair { key, value })
    }
}

impl Parse for GpuTypeAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let pairs = Punctuated::<KvPair, Token![,]>::parse_terminated(input)?;

        let mut slang_name = None;
        for kv in pairs {
            if kv.key == "slang_name" {
                if slang_name.is_some() {
                    return Err(syn::Error::new_spanned(
                        kv.key,
                        "duplicate `slang_name` key",
                    ));
                }
                slang_name = Some(kv.value);
            } else {
                return Err(syn::Error::new_spanned(
                    kv.key,
                    "unknown key — only `slang_name` is recognised",
                ));
            }
        }

        Ok(GpuTypeAttr { slang_name })
    }
}

pub(crate) fn derive_gpu_type(input: &DeriveInput) -> syn::Result<TokenStream> {
    // XXX well, we might need to support parameterized types for indirect types like Ssbo and BDA handles!
    if !input.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.generics,
            "#[derive(GpuType)] does not support generic structs",
        ));
    }

    crate::force::assert_repr_c(&input.attrs, input.ident.span())?;

    let fields = match &input.data {
        Data::Struct(s) => &s.fields,
        Data::Enum(e) => {
            return Err(syn::Error::new_spanned(
                e.enum_token,
                // NEXT enum is not totally different, just
                "#[derive(GpuType)] is only supported on structs, not enums",
            ));
        }
        Data::Union(u) => {
            return Err(syn::Error::new_spanned(
                u.union_token,
                "#[derive(GpuType)] is only supported on structs, not unions",
            ))
        }
    };

    let named_fields = match fields {
        Fields::Named(f) => f,
        Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "#[derive(GpuType)] requires named fields — tuple structs are not supported",
            ))
        }
        Fields::Unit => {
            return Err(syn::Error::new_spanned(
                &input.ident,
                "#[derive(GpuType)] requires at least one named field",
            ))
        }
    };

    // Provided Slang name or default
    let mut attr_slang_name: Option<LitStr> = None;
    for attr in &input.attrs {
        if !attr.path().is_ident("gpu_type") {
            continue;
        }
        let parsed: GpuTypeAttr = attr.parse_args()?;
        if attr_slang_name.is_some() {
            return Err(syn::Error::new_spanned(
                attr,
                "duplicate #[gpu_type(...)] attribute",
            ));
        }
        attr_slang_name = parsed.slang_name;
    }
    let struct_ident = &input.ident;
    let slang_name: LitStr = match attr_slang_name {
        Some(lit) => lit,
        None => LitStr::new(&struct_ident.to_string(), Span::call_site()),
    };

    // Build FieldNode list.
    //
    // Each entry looks like:
    //   <FieldType as ::mutate_vulkan::slang::GpuType<D>>::FIELD_NODE,
    let field_nodes = named_fields.named.iter().map(|f| {
        let ty = &f.ty;
        quote! {
            <#ty as ::mutate_vulkan::slang::GpuType<D>>::FIELD_NODE
        }
    });

    // DEBT bytemuck #[repr(C)] enforcement.
    let expanded = quote! {
        impl<D: ::mutate_vulkan::slang::DataLayout> ::mutate_vulkan::slang::GpuType<D>
            for #struct_ident
        {
            const FIELD_NODE: ::mutate_vulkan::slang::FieldNode =
                ::mutate_vulkan::slang::FieldNode::Tree {
                    slang_name: #slang_name,
                    fields: &[
                        #( #field_nodes ),*
                    ],
                };
        }

        // DEBT bytemuck
        impl Copy  for #struct_ident {}
        impl Clone for #struct_ident {
            fn clone(&self) -> Self { *self }
        }

        unsafe impl ::mutate_vulkan::__bytemuck::Zeroable for #struct_ident {}
        unsafe impl ::mutate_vulkan::__bytemuck::Pod    for #struct_ident {}
    };

    Ok(expanded.into())
}
