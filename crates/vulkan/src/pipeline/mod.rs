// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pipeline
//!
//! This stub should be filled in with stages and other types that are strongly bound to realized
//! pipelines.

// NOTE see descriptors.rs for the static descriptor definition.  The lib/descriptors.slang contains
// yet another mirror of this single descriptor table scheme.

pub mod stage;

use ash::vk;

use crate::prelude::*;

struct ComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

// impl ComputePipeline {
//     // XXX no flags are supported yet.  Variadics are for macros '<@>_<@>'
//     fn new(
//         stage: vk::PipelineShaderStageCreateInfo,
//         ranges: &[vk::PushConstantRange],
//         context: &VkContext,
//     ) -> Result<Self, VulkanError> {
//         let device = context.device();
//         let layout_ci = vk::PipelineLayoutCreateInfo::default()
//             .push_constant_ranges(ranges)
//             .set_layouts(context.descriptors.layout());
//         let layout = unsafe { device.create_pipeline_layout(&layout_ci, None)? };
//         let pipeline_ci = vk::ComputePipelineCreateInfo::default()
//             .stage(stage)
//             .layout(layout);
//         let pipeline = unsafe {
//             device
//                 .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
//                 .map_err(|huh| huh.1)?[0] // XXX 🤠
//         };
//         Ok(Self {
//             pipeline: vk::Pipeline::null(),
//             layout: vk::PipelineLayout::null(),
//         })
//     }
// }

// trait Push {
//     const RANGE: vk::PushConstantRange;
// }

// vk::PushConstantRange = vk::PushConstantRange {
//     stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS,
//     offset: 0,
//     size: 128,
// };

// pipeline_layout: vk::PipelineLayout
// pipelines: Vec<vk::Pipeline>,

// #[derive(Hash, PartialEq, Eq)]
// struct PipelineLayoutKey {
//     set_layouts: Vec<vk::DescriptorSetLayout>,
//     push_constant_ranges: Vec<vk::PushConstantRange>,
// }

// pub(crate) struct Layouts {
//     layouts: std::collections::HashMap<LayoutKey, Layout>,
// }

// // ============================================================
// // Traits
// // ============================================================

// /// Witness trait for push constant structs.
// /// Supertrait bounds ensure safe cast to &[u8] for vkCmdPushConstants.
// pub trait PushConstants: bytemuck::NoUninit + Copy {
//     const STAGES: ash::vk::ShaderStageFlags;
//     const SIZE: usize = std::mem::size_of::<Self>();
//     const OFFSET: u32 = 0;
// }

// /// Witness trait for a single shader stage.
// /// Carries the stage flag and a key for cache lookup.
// pub trait ShaderStage {
//     const STAGE: ash::vk::ShaderStageFlags;
//     const KEY: ShaderKey;
// }

// /// Witness trait for a pipeline layout.
// /// Declares the push constant type, reads, and writes.
// /// Reads and writes are used by the render graph.
// pub trait PipelineLayout {
//     type PushConstants: PushConstants;

//     const READS: &'static [ResourceId];
//     const WRITES: &'static [ResourceId];
// }

// /// Witness trait for a fully resolved pipeline.
// /// Ties a layout to its concrete shader stages.
// pub trait Pipeline {
//     type Layout: PipelineLayout;

//     fn create(device: &ash::Device, cache: &PipelineCache) -> Self;
// }

// // ============================================================
// // Supporting types (handwritten)
// // ============================================================

// /// Logical identity of a shader — path plus spirv hash for freshness.
// #[derive(Clone, PartialEq, Eq, Hash)]
// pub struct ShaderKey {
//     pub path: &'static str,
//     pub spirv_hash: u64,
//     pub stage: ash::vk::ShaderStageFlags,
// }

// /// Opaque handle to a resource declared as a read or write in a layout.
// /// Used by the render graph to reason about dependencies.
// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
// pub struct ResourceId(pub u64);

// /// Runtime cache — totally separate from type identity.
// /// Keyed on ShaderKey or a pipeline layout hash.
// pub struct PipelineCache {
//     // TODO: inner cache storage
//     _priv: (),
// }

// impl PipelineCache {
//     pub fn new() -> Self {
//         Self { _priv: () }
//     }

//     pub fn get_shader(&self, _key: &ShaderKey) -> Option<ash::vk::ShaderModule> {
//         todo!()
//     }

//     pub fn get_pipeline(&self, _key: u64) -> Option<ash::vk::Pipeline> {
//         todo!()
//     }

//     pub fn insert_shader(&self, _key: ShaderKey, _module: ash::vk::ShaderModule) {
//         todo!()
//     }

//     pub fn insert_pipeline(&self, _key: u64, _pipeline: ash::vk::Pipeline) {
//         todo!()
//     }
// }

// // ============================================================
// // Macros
// // ============================================================

// /// Defines a PushConstants struct and its trait impl.
// ///
// /// # Mode 1 — define inline
// /// ```rust
// /// push_constants! {
// ///     name: GeoPassPC,
// ///     stages: VERTEX | FRAGMENT,
// ///     fields: {
// ///         model_matrix: Mat4,
// ///         flags: u32,
// ///     }
// /// }
// /// ```
// ///
// /// # Mode 2 — reference existing type
// /// ```rust
// /// push_constants! {
// ///     name: GeoPassPC,   // GeoPassPC already exists and impls PushConstants
// /// }
// /// ```
// #[macro_export]
// macro_rules! push_constants {
//     // Mode 1 — define inline
//     (
//         name: $name:ident,
//         stages: $($stage:ident)|+,
//         fields: { $($field:ident: $ty:ty),* $(,)? }
//     ) => {
//         #[repr(C)]
//         #[derive(Clone, Copy, bytemuck::NoUninit)]
//         pub struct $name {
//             $(pub $field: $ty,)*
//         }

//         impl $crate::PushConstants for $name {
//             const STAGES: ash::vk::ShaderStageFlags =
//                 $(ash::vk::ShaderStageFlags::$stage)|+;
//         }
//     };

//     // Mode 2 — reference existing type (just assert the bound, no expansion)
//     (name: $name:ident) => {
//         const _: () = {
//             fn _assert_push_constants<T: $crate::PushConstants>() {}
//             fn _check() { _assert_push_constants::<$name>(); }
//         };
//     };
// }

// /// Declares a shader stage by path and stage flag.
// /// Generates a zero-sized type + ShaderStage impl.
// ///
// /// ```rust
// /// shader! {
// ///     name: GeoVert,
// ///     path: "shaders/geo.vert.spv",
// ///     stage: VERTEX,
// ///     spirv_hash: 0xDEADBEEF,
// /// }
// /// ```
// #[macro_export]
// macro_rules! shader {
//     (
//         name: $name:ident,
//         path: $path:literal,
//         stage: $stage:ident,
//         spirv_hash: $hash:expr $(,)?
//     ) => {
//         pub struct $name;

//         impl $crate::ShaderStage for $name {
//             const STAGE: ash::vk::ShaderStageFlags =
//                 ash::vk::ShaderStageFlags::$stage;

//             const KEY: $crate::ShaderKey = $crate::ShaderKey {
//                 path: $path,
//                 spirv_hash: $hash,
//                 stage: ash::vk::ShaderStageFlags::$stage,
//             };
//         }
//     };
// }

// /// Declares a pipeline layout — push constants, reads, writes.
// /// Generates a zero-sized type + PipelineLayout impl.
// ///
// /// ```rust
// /// pipeline_layout! {
// ///     name: GeoPassLayout,
// ///     push_constants: GeoPassPC,
// ///     reads: [ShadowMap],
// ///     writes: [GBuffer0, GBuffer1, Depth],
// /// }
// /// ```
// #[macro_export]
// macro_rules! pipeline_layout {
//     (
//         name: $name:ident,
//         push_constants: $pc:ty,
//         reads: [$($read:ident),* $(,)?],
//         writes: [$($write:ident),* $(,)?] $(,)?
//     ) => {
//         pub struct $name;

//         impl $crate::PipelineLayout for $name {
//             type PushConstants = $pc;

//             const READS: &'static [$crate::ResourceId] = &[
//                 $($read::ID,)*
//             ];

//             const WRITES: &'static [$crate::ResourceId] = &[
//                 $($write::ID,)*
//             ];
//         }
//     };
// }

// /// Declares a full pipeline — composes layout and stages into a concrete type.
// /// Generates a struct + Pipeline impl.
// /// In inline mode, calls push_constants!, shader!, pipeline_layout! internally.
// ///
// /// ```rust
// /// pipeline! {
// ///     name: GeoPass,
// ///     layout: GeoPassLayout,
// ///     vertex: GeoVert,
// ///     fragment: GeoFrag,
// /// }
// /// ```
// #[macro_export]
// macro_rules! pipeline {
//     (
//         name: $name:ident,
//         layout: $layout:ty,
//         vertex: $vert:ty,
//         fragment: $frag:ty $(,)?
//     ) => {
//         pub struct $name {
//             pub raw: ash::vk::Pipeline,
//             pub raw_layout: ash::vk::PipelineLayout,
//         }

//         impl $crate::Pipeline for $name {
//             type Layout = $layout;

//             fn create(device: &ash::Device, cache: &$crate::PipelineCache) -> Self {
//                 todo!(
//                     "resolve {:?} and {:?} from cache, build PSO",
//                     <$vert as $crate::ShaderStage>::KEY,
//                     <$frag as $crate::ShaderStage>::KEY,
//                 )
//             }
//         }
//     };
// }
