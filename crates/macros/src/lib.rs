// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Mutate Macros
//!
//! *Let them call structs* - Marie Antoinette
//!
//! Maintaining host-GPU agreement can be a pain.  Comparing Rust code with Slang reflection data
//! requires a fair amount of unavoidable machinery to get layout and type information into one
//! place.  These macros emit that machinery from simple struct declarations with a minimum of extra
//! annotation.
//!
//! - [`Stage`](crate::stage::Stage) - declare pipeline stages, individual shader programs for
//!   assembly into pipelines.
//!
//! - [`GpuType`](crate::slang::GpuType) - enable uploading or reading a structure from GPU memory
//!   for a given `DataLayout`.
//!
//! - [`Push`](crate::push::Push) - declare pipeline push constant layouts by annotating a struct.
//!
//! - [`Pipeline`] - combine declarations for several stages, a layout, and pipeline states into a
//!   single tese expression.

mod force; // utilities for common assertions / ensures
mod push;
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

/// # `#[derive(Push)]`
///
/// Derives [`PushConstants`], a companion [`LayoutSpec`], and [`HasDefaultLayout`] for a struct
/// that already implements [`GpuType<Scalar>`].
/// ## Usage
///
/// ```ignore
/// #[derive(GpuType, Push)]
/// #[repr(C)]
/// struct MyPush {
///     ctrl_idx: UInt,
///     scale:    Float,
/// }
/// ```
///
/// ## What Is Emitted
///
/// ```ignore
/// impl PushConstants<Scalar> for MyPush {}
///
/// pub struct MyPushLayout;
///
/// impl LayoutSpec for MyPushLayout {
///     type D    = Scalar;
///     type Push = MyPush;
///     const RANGES: &'static [vk::PushConstantRange] = &[
///         vk::PushConstantRange {
///             stage_flags: vk::ShaderStageFlags::COMPUTE,
///             offset: 0,
///             size:   <MyPush as Pack<Scalar>>::PACKED_SIZE as u32,
///         },
///     ];
/// }
///
/// impl DefaultLayout for MyPush {
///     type D             = Scalar;
///     type DefaultLayout = MyPushLayout;
/// }
/// ```
///
/// # Including Fields in ranges
///
/// Fields can be exclusive to explicit ranges or implicitly included in all ranges.
///
/// ```ignore
/// #[derive(Push)]
/// #[repr(C)]
/// struct MyPush {
///     #[visible(ALL)]
///     shared: UInt32,
///     #[visible(RAYGEN)]
///     ray_only: UInt32,
///     #[visible(CLOSEST)]
///     hit_only: UInt32,
///     #[visible(CLOSEST | INTERSECTION)]
///     hit_and_intersect: UInt32,
/// }
///```
///
/// Gaps within a stage's range require explicit `[vk::offset(N)]` annotations on
/// subsequent fields so the Slang compiler assigns the correct push constant offset.
/// The macro generates one block per stage:
///
/// ```slang
/// [[vk::push_constant]] struct RayGen {
///     [[vk::offset(0)]] uint shared;
///     [[vk::offset(4)]] uint ray_only;
/// };
///
/// [[vk::push_constant]] struct Closest {
///     [[vk::offset(0)]] uint shared;
///     // gap [4,8)
///     [[vk::offset(8)]] uint hit_only;
///     uint hit_and_intersect;
/// };
///
/// [[vk::push_constant]] struct Intersection {
///     [[vk::offset(0)]]  uint shared;
///     // gap [4,12)
///     [[vk::offset(12)]] uint hit_and_intersect;
/// };
/// ```
///
/// Ranges delegate to `Pack<Scalar>::PACKED_SIZE`, rooted in the `GpuType` `FieldNode` tree.
/// Emitted calculations depend on the correctness of the backing `GpuType` implementaiton.

#[proc_macro_derive(Push, attributes(visible))]
pub fn derive_push(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    push::derive_push(&input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
