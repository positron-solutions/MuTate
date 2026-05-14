// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use proc_macro2::{Ident, Span, TokenStream};
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;

/// Resolves the root path to the `__` private module of mutate_vulkan.  The `::__` suffix is
/// included in the return token so you may write `#root::` directly to access `__`.
///
/// Uses `mutate_vulkan` or re-export as `vulkan` via `mutate_lib` facade.
pub(crate) fn mutate_vulkan_root() -> TokenStream {
    if let Ok(found) = crate_name("mutate-vulkan") {
        return match found {
            FoundCrate::Itself => quote!(crate::__),
            FoundCrate::Name(name) => {
                let ident = Ident::new(&name, Span::call_site());
                quote!(::#ident::__)
            }
        };
    }

    if let Ok(found) = crate_name("mutate-lib") {
        return match found {
            FoundCrate::Itself => quote!(crate::vulkan::__),
            FoundCrate::Name(name) => {
                let ident = Ident::new(&name, Span::call_site());
                quote!(::#ident::vulkan::__)
            }
        };
    }

    panic!(
        "mutate_vulkan proc macros did not find either `mutate-vulkan` or `mutate-lib` as a \
        dependency in your Cargo.toml"
    );
}
