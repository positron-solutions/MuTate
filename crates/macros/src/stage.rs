// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Stage
//!
//! Emits `Stage<Slot>` and `StageReflection<Slot>` impls onto any type, usually a pipeline or other
//! ZST used for naming.

use proc_macro2::{TokenStream, Span};
use syn::Token;
use syn::{
    parse::{Parse, ParseStream},
    LitCStr, LitStr,
};

use mutate_assets as assets;

/// Parsed stage.  Accepts shader file stem, stage flags, and entry point.
pub(crate) struct StageAttr {
    /// Shader file name, without extension.
    file: LitStr,
    /// Stage marker type, one that implements `[StageSlot](mutate_vulkan::pipeline::stage::StageSlot)`.
    stage: syn::Ident,
    entry: Option<LitCStr>,
}

impl Parse for StageAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let file: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let stage: syn::Ident = input.parse()?;
        let entry = if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            Some(input.parse::<LitCStr>()?)
        } else {
            None
        };
        Ok(StageAttr { file, stage, entry })
    }
}

/// Independent stage declaration entry point point.  Parses input and then delegates entirely to
/// [`emit_stage_impls`].
pub(crate) fn stage(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let input      = syn::parse2::<syn::DeriveInput>(item)?;
    let type_name  = &input.ident;
    let stage_attr = syn::parse2::<StageAttr>(attr)?;

    let impls = emit_stage_impls(type_name, stage_attr)?;

    Ok(quote::quote! {
        #input
        #impls
    })
}

// XXX Docs out of date
/// Attach knowledge of a shader stage to a concrete type.  Implements witness traits that enable
/// downstream type checking.
pub(crate) fn emit_stage_impls(
    target: &syn::Ident,
    attr: StageAttr,
) -> syn::Result<TokenStream> {
    let StageAttr {
        file: file_literal,
        stage,
        entry,
    } = attr;

    let file_string = file_literal.value();
    let dirs = assets::AssetDirs::new();
    let shader = dirs.find_shader(&file_string);
    let hash   = dirs.find_hash(&file_string, assets::AssetKind::Shader);

    if let Err(e) = &shader {
        let msg = match e {
            assets::AssetError::NotFound { name, tried } => {
                let paths = tried
                    .iter()
                    .map(|p| format!("\n  {}", p.display()))
                    .collect::<String>();
                format!("Missing required shader {name:?}, tried:{paths}")
            }
            _ => format!("Failed to load shader {file_string:?}: {e}"),
        };
        // Anchor the error to the file literal — that's what the user wrote.
        return Err(syn::Error::new_spanned(&file_literal, msg));
    }
    if let Err(e) = &hash {
        let msg = match e {
            assets::AssetError::NotFound { name, tried } => {
                let paths = tried
                    .iter()
                    .map(|p| format!("\n  {}", p.display()))
                    .collect::<String>();
                format!("Missing required shader hash for {name:?}, tried:{paths}")
            }
            _ => format!("Failed to load shader hash {file_string:?}: {e}"),
        };
        return Err(syn::Error::new_spanned(&file_literal, msg));
    }

    let entry: LitCStr = entry
        .unwrap_or_else(|| LitCStr::new(c"main", Span::call_site()));

    // DEBT error handling
    let hash_path_string = hash.unwrap().into_os_string().into_string().unwrap();

    // NEXT: drive CONSTANT_BLOCK_SIZE from actual reflection data read off `shader`.
    let constant_block_size: usize = 0;

    Ok(quote::quote! {
        impl ::mutate_vulkan::pipeline::stage::Stage
            <::mutate_vulkan::pipeline::stage::#stage> for #target {
            const SPEC: ::mutate_vulkan::pipeline::stage::StageSpec =
                ::mutate_vulkan::pipeline::stage::StageSpec {
                    name:  #file_literal,
                    stage: <::mutate_vulkan::pipeline::stage::#stage
                               as ::mutate_vulkan::pipeline::stage::StageSlot>::FLAGS,
                    entry: #entry,
                };
        }

        impl ::mutate_vulkan::pipeline::stage::StageReflection
            <::mutate_vulkan::pipeline::stage::#stage> for #target {
            const CONSTANT_BLOCK_SIZE: usize = #constant_block_size;
        }

        // NOTE Emit code that includes the bytes to ensure the compiler watches this file for future
        // changes.  We're just watching the hash to load less into memory and then using a zero
        // size range to save the linker the trouble.

        // Force the compiler to watch the hash file for changes.  Zero-length range keeps the bytes
        // out of the binary.
        const _: &[u8] = ::std::include_bytes!(#hash_path_string);
    })
}
