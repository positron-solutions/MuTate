// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Stage
//!
//! Declare a shader stage.  Compiled shader file and its reflection data will be read to emit
//! attributes necessary for downstream type-agreement checks.

use proc_macro::TokenStream;
use syn::Token;
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, Data, DeriveInput, Fields, LitCStr, LitStr, Type,
};

use mutate_assets as assets;

struct ShaderAttr {
    file: LitStr,
    stage: syn::Ident,
    entry: Option<LitCStr>,
}

impl Parse for ShaderAttr {
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
        Ok(ShaderAttr { file, stage, entry })
    }
}

/// Express a shader stage as a concrete type.  Implements witness traits that enable downstream
/// type checking.  Accepts shader file name (without extension), stage flags, and entry point.
pub(crate) fn shader(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::DeriveInput);
    let type_name = &input.ident;

    let ShaderAttr {
        file: file_literal,
        stage,
        entry,
    } = parse_macro_input!(attr as ShaderAttr);

    let file_string = file_literal.value();
    let dirs = assets::AssetDirs::new();
    let shader = dirs.find_shader(&file_string);
    let hash = dirs.find_hash(&file_string, assets::AssetKind::Shader);

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
        return syn::Error::new_spanned(input, msg)
            .to_compile_error()
            .into();
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
        return syn::Error::new_spanned(input, msg)
            .to_compile_error()
            .into();
    }

    let entry: LitCStr = entry
        .unwrap_or_else(|| syn::parse_str::<LitCStr>("c\"main\"").expect("default entry literal"));

    // DEBT error handling
    let hash_path_string = hash.unwrap().into_os_string().into_string().unwrap();

    // Emit code that includes the bytes to ensure the compiler
    // watches this file for future changes.  We're just watching the hash to load less into memory
    // and then using a zero size range to save the linker the trouble.

    // NEXT: drive CONSTANT_BLOCK_SIZE from actual reflection data read off `shader`.
    let constant_block_size: usize = 0;

    quote::quote! {
        #input

        // XXX export path stability!
        impl ::mutate_vulkan::pipeline::stage::StageReflection for #type_name {
            const CONSTANT_BLOCK_SIZE: usize = #constant_block_size;
        }

        // XXX implementing STAGE_SPEC as a constant for the leaf type gives us no way to fan in
        // properly.  Something is messed up.
        impl #type_name {
            pub const STAGE_SPEC: ::mutate_vulkan::pipeline::stage::StageSpec = ::mutate_vulkan::pipeline::stage::StageSpec {
                name:  #file_literal,
                stage: ::ash::vk::ShaderStageFlags::#stage,
                entry: #entry,
            };
        }

        // NEXT if shader modules can hydrate from specs, we can forward this into the spec for runtime
        // verification and then we don't need this statement to force re-eval.
        const _: &[u8] = ::std::include_bytes!(#hash_path_string);
    }
    .into()
}
