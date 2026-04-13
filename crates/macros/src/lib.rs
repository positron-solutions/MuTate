// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Mutate Macros
//!
//! Static checking host-GPU agreement would be quite a pain.  We use macros to emit the witnesses
//! on leaf types and the root witnesses that gather up leaf witness data to check agreement across
//! all leaves.
//!
//! - pipeline `Stages`
//! - `GpuTypes` and `PushConstants`
//! - `Pipelines`
//! - Ensembles of Pipelines and the inputs shapes they need to share between each other.

// We use proc macros. Proc macros have to go in their own crates.  Feels like a scam.  Enough
// complaining.  Time to let another program write our proc macros because generating your code
// generation is like making ultramarine from lapis lazuli by hand and telling everyone your blue
// bracelet is better than theirs and being mad when you must explain what you feel is important yet
// exhibits no apparent concrete evidence of its truth.  🤖

mod slang;
mod stage;

use proc_macro::TokenStream;

#[proc_macro_attribute]
pub fn shader(attr: TokenStream, item: TokenStream) -> TokenStream {
    stage::shader(attr, item)
}

#[proc_macro_derive(GpuType, attributes(gpu_type))]
pub fn gpu_type(input: TokenStream) -> TokenStream {
    slang::derive_gpu_type(input)
}
