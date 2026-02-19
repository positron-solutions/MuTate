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
use syn::{parse_macro_input, LitStr};

#[proc_macro]
pub fn sup_dawg(_input: TokenStream) -> TokenStream {
    let out = "sup dawg";
    let lit = syn::LitStr::new(&out, proc_macro2::Span::call_site());

    quote::quote! {
        #lit
    }
    .into()
}
