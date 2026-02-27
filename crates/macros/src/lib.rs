// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Proc Macros
//!
//! We use proc macros. Proc macros have to go in their own crates.  Feels like a scam.  Enough
//! complaining.  Time to let another program write our proc macros because generating your code
//! generation is like making ultramarine from lapis lazuli by hand and telling everyone your blue
//! bracelet is better than theirs and being mad when you must explain what you feel is important
//! yet exhibits no apparent concrete evidence of its truth.  🤖

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, LitStr, Type};

use mutate_assets as assets;

#[proc_macro]
pub fn sup_dawg(_input: TokenStream) -> TokenStream {
    let out = "sup dawg";
    let lit = syn::LitStr::new(&out, proc_macro2::Span::call_site());

    quote::quote! {
        #lit
    }
    .into()
}

#[proc_macro_attribute]
pub fn shader(attr: TokenStream, item: TokenStream) -> TokenStream {
    let file_literal = parse_macro_input!(attr as LitStr);
    let file_string = file_literal.value();
    let input = parse_macro_input!(item as DeriveInput);

    let dirs = assets::AssetDirs::new();
    let shader = dirs.find_shader(&file_string);
    let hash = dirs.find_hash(&file_string, assets::AssetKind::Hash);

    // DEBT maybe returning errors from assets functions is the better way 🤔.  We don't have a good
    // way to control if we tell the user about the error or the paths that were tried from here.
    if shader.is_err() {
        return syn::Error::new_spanned(
            input,
            format!("Missing required resource file: {}", &file_string),
        )
        .to_compile_error()
        .into();
    }
    if hash.is_none() {
        return syn::Error::new_spanned(
            input,
            format!("Missing required resource hash: {}", &file_string),
        )
        .to_compile_error()
        .into();
    }
    // XXX stop unwrapping... but later.
    let hash_path_string = hash.unwrap().into_os_string().into_string().unwrap();

    // Emit code that includes the bytes to ensure the compiler
    // watches this file for future changes.  We're just watching the hash to load less into memory
    // and then using a zero size range to save the linker the trouble.
    quote::quote! {
        const _WATCH_HASH: &[u8] = &include_bytes!(#hash_path_string)[0..0];
    }
    .into()
}
