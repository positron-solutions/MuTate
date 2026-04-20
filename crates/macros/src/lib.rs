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
//! - [`stage`](crate::stage::stage) - declare pipeline stages, individual shader programs for
//!   assembly into pipelines.
//!
//! - [`GpuType`](crate::slang::GpuType) - derive the capability to upload or read a structure from
//!   GPU memory for a given `DataLayout`.
//!
//! - [`Push`](crate::push::Push) - derive pipeline push constant layouts by annotating a struct.
//!
//! ## Pipeline Macros
//!
//! Combine declarations for several stages, a layout, and pipeline states into a single
//! self-contained expression.  Include components by name if they were attached to other types.
//! Declare them inline, with optional names, if they should be used directly.
//!
//! - [`compute_pipeline`] - Compute pipelines are simple, consisting of a single stage and a layout
//!   that usually only needs one push constant range.
//!
//! - [`graphics_pipeline`] - Graphics pipelines consist of several stages, a layout compatible with
//!   them all, and required dynamic state.
//!
//! More pipeline macros may appear, especially to keep disjoint functionality (stage combination
//! checks and relevant dynamic states) separate.  Ray tracing and alternative mesh shaders are some
//! likely possibilities for further separation.

mod force; // utilities for common assertions / ensures
mod pipeline;
mod push;
mod slang;
mod stage;

/// # `#[stage]`
///
/// Declare a shader stage.  Compiled shader file and its reflection data will be read to emit
/// attributes necessary for downstream type-agreement checks.
///
/// ```ignore
/// #[stage("test/hello_compute", COMPUTE, c"main")]
/// struct GoodStage {}
/// ```
#[proc_macro_attribute]
pub fn stage(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = proc_macro2::TokenStream::from(attr);
    let item = proc_macro2::TokenStream::from(item);
    stage::stage(attr, item)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// # `#[derive(GpuType)]`
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
/// use mutate_macros::*;
///
/// #[derive(GpuType)]
/// #[repr(C)]
/// struct MySlangType {
///   foo: UInt,
///   bar: Float,
///   // XXX add some handles to the example
/// }
///
/// #[derive(GpuType, Debug)]
/// // XXX attribute ignored still
/// #[gpu_type(slang_name = "SpectralBand")]
/// #[repr(C)]
/// pub struct SpectralBand {
///     pub center_hz:  Float,
///     pub magnitude:  Float,
///     pub phase_rad:  Float,
///     pub sample_buf: SsboIdx,
/// }
///
/// #[derive(GpuType, Debug)]
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
pub fn derive_gpu_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    slang::derive_gpu_type(&input)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// # `#[derive(Push)]`
///
/// Derives [`PushConstants`] and a companion [`LayoutSpec`] for a struct that already implements
/// [`GpuType<Scalar>`].
///
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
/// impl PushConstants for MyPush {}
///
/// impl LayoutSpec for MyPush {
///     type D    = Scalar;
///     type Push = MyPush;
///     const RANGES: &'static [vk::PushConstantRange] = &[
///         vk::PushConstantRange {
///             stage_flags: <RayGen as StageSlot>::FLAGS | <Closest as StageSlot>::FLAGS
///                        | <Intersection as StageSlot>::FLAGS,
///             offset: field_start(&<MyPush as GpuType<Scalar>>::FIELD_NODE, 0, Scalar::DATA_LAYOUT) as u32,
///             size:   (field_end(…, N, …) - field_start(…, 0, …)) as u32,
///         },
///         // …one entry per distinct (first, last) span after merging…
///     ];
/// }
/// ```
///
/// ## Including Fields in ranges
///
/// Fields can be exclusive to explicit ranges or implicitly included in all ranges.
///
/// ```ignore
/// #[derive(Push)]
/// #[repr(C)]
/// struct MyPush {
///     // implicit ALL flags
///     shared: UInt32,
///     #[visible(RayGen)]
///     ray_only: UInt32,
///     #[visible(Closest)]
///     hit_only: UInt32,
///     #[visible(Closest | Intersection)]
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

/// # `#[compute_pipeline]`
///
/// Compute pipelines are simple:
///
/// - one stage
/// - a layout with push constants usually
///
/// The macro attaches the pipeline functionality to a type for reuse by name.
///
/// ```ignore
/// #[compute_pipeline
///     // Single stage
///     compute = stage!("indirect/compute"),
///     // Simple layout with one range covering all bytes.
///     push = push! {
///         matrix_idx: UInt,
///     }
/// )]
/// pub struct Compute;
/// ```
#[proc_macro_attribute]
pub fn compute_pipeline(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let attr = proc_macro2::TokenStream::from(attr);
    let item = proc_macro2::TokenStream::from(item);
    pipeline::compute_pipeline(attr, item)
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}

/// # `#[graphics_pipeline]`
///
/// Graphics pipelines are a bit more involved, usually combining several stages and requiring some
/// dynamic states be configured.
///
/// ```ignore
/// #[graphics_pipeline(
///     // The attribute names decide the stage flags and default entry points for the shader programs.
///     vert = stage!("lighting/vertex"),
///     frag = stage!("lighting/fragment"),
///
///     // `PushConstant` types all have a layout.  Since our descriptor table is held static, the
///     // push constants decide the layout.
///     push = push! {
///         #[visible(Vertex | Fragment)]
///         matrix_idx: UInt,
///         #[visible(Fragment)]
///         light_idx:  UInt,
///         frame_time: Float,
///     }
/// )]
/// pub struct ScenePipeline;
/// ```
#[proc_macro_attribute]
pub fn graphics_pipeline(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let _attr = proc_macro2::TokenStream::from(attr);
    let _item = proc_macro2::TokenStream::from(item);
    // pipeline::graphics_pipeline(attr, item)
    //     .unwrap_or_else(|e| e.to_compile_error())
    //     .into()
    todo!()
}
