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

mod force; // utilities for common assertions / ensures
mod slang;
mod stage;

#[proc_macro_attribute]
pub fn shader(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = proc_macro2::TokenStream::from(attr);
    let item = proc_macro2::TokenStream::from(item);
    stage::shader(attr, item)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// GpuType
///
/// Derive GpuType for new structs.  If all fields implement GpuType, this can be pretty trivial.
///
/// ## Usage
///
/// Structs must be `repr(C)`.  No other representation is valid.  Manual over-alignment is not
/// supported.  We need to be able to reason about the byte layout for copying the structure into
/// push constants and other writes to GPU memory.
///
/// ```
/// use mutate_vulkan::prelude::*;
///
/// #[derive(GpuType)]
/// #[repr(C)]
/// struct MySlangType {
///   foo: UInt,
///   bar: Float,
///   // XXX add some handles to the example
/// }
///
/// #[derive(GpuType, Clone, Copy, Debug)]
/// #[gpu_type(slang_name = "SpectralBand")]
/// #[repr(C)]
/// pub struct SpectralBand {
///     pub center_hz:  Float,
///     pub magnitude:  Float,
///     pub phase_rad:  Float,
///     pub sample_buf: SsboIdx,
/// }
///
/// #[derive(GpuType, Clone, Copy, Debug)]
/// #[repr(C)]
/// pub struct AudioPushConstants {
///     pub band:         SpectralBand, // nested 🕶️
///     pub waveform_buf: SsboIdx,
///     pub frame_index:  UInt,
///     pub delta_t:      Float,
///     pub flags:        UInt,
/// }
/// ```
///
#[proc_macro_derive(GpuType, attributes(gpu_type))]
pub fn gpu_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    let span = input.ident.span();

    slang::derive_gpu_type(&input, span)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
