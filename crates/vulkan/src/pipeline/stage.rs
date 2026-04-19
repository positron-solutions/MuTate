// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Stage
//!
//! *This stage has taken a while to reach.  Keep coding a lot.*
//!
//! At runtime, stages only need to be hydrated for building pipelines.  Actual loaded shader
//! modules can be dropped as soon as the pipeline handle is ready, so the spec data used to load
//! stages is a trait that can be attached to any ZST.  Pipelines still need to be able to specify
//! the paths to these ZSTs (or themselves) so that the list of modules can be produced for hydrating
//! a pipeline.
//!
//! Stage reflection data needed for later host-GPU agreement checks is attached via a second type
//! that can be compiled away after checks evaluate.  Expansion time obtains the data and then emits
//! them with predictable paths for combining with other const data from other types.
//!
//! Because we can attach stage specs to any ZST, they are a trait impl.  A pipeline needs only a
//! path to the impl in order to use the stage.  This also means we can either declare stages as
//! impls on some independent type or directly on the pipeline, and the pipeline just needs to able
//! to say where to look.
//!
//! Because [`StageSpec`] is type-erased, we can use the constants on several implementations in a
//! single list, enabling pipelines to list their stages without extra indirection.

use std::ffi::CStr;

use ash::vk::{self, ShaderStageFlags as S};

use crate::internal::*;
use crate::resource::shader::ShaderModule;

mod sealed {
    pub trait Sealed {}
}

/// A proxy for [`ash::vk::ShaderStageFlags`].  We would use those directly if they were a valid
/// type parameter, but they are not a primitive type.
// ROLL const machinery may later enable getting rid of this indirect mapping.
pub trait StageSlot: sealed::Sealed {
    const FLAGS: vk::ShaderStageFlags;
}

/// Map shader stage flags to marker types and seal.
macro_rules! slot {
    ($name:ident, $flag:expr) => {
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl StageSlot for $name {
            const FLAGS: vk::ShaderStageFlags = $flag;
        }
    };
}

slot!(Vertex, S::VERTEX);
slot!(TessellationControl, S::TESSELLATION_CONTROL);
slot!(TessellationEvaluation, S::TESSELLATION_EVALUATION);
slot!(Geometry, S::GEOMETRY);
slot!(Fragment, S::FRAGMENT);
slot!(Compute, S::COMPUTE);
slot!(RayGen, S::RAYGEN_KHR);
slot!(Miss, S::MISS_KHR);
slot!(ClosestHit, S::CLOSEST_HIT_KHR);
slot!(AnyHit, S::ANY_HIT_KHR);
slot!(Intersection, S::INTERSECTION_KHR);
slot!(Callable, S::CALLABLE_KHR);

/// Attach a stage spec to any type, usually a pipeline or ZST used for naming.  `Slot` implements
/// `StageSlot`.
pub trait Stage<slot: StageSlot> {
    const SPEC: StageSpec;
}

/// Declared data, what the user specified.  Sufficient to load a shader.
pub struct StageSpec {
    /// Shader program name for asset lookup, usually SPIR-V or Metal shader lib files.
    pub name: &'static str,
    /// Which stage of the pipeline this module will be used for.
    pub stage: vk::ShaderStageFlags,
    /// Shader program entry point must match the stage type found in the introspection data.
    // NEXT stage agreement check in macro
    pub entry: &'static CStr,
}

/// Introspected data.  Obtained by reflecting against the shader module.  Intended to be emitted by
/// macro.
pub trait StageReflection<Slot: StageSlot>: Stage<Slot> {
    /// Size in bytes of the push constants viewed by this shader.
    const CONSTANT_BLOCK_SIZE: usize;
    // NEXT keep pulling in the introspection data:
    // - push constants
    //   + alignment
    //   + types
    //     * indirect types (buffer types)
    // - descriptor bindings (should always be the same)
    // - hash (for load memoization)
}

// The functions below are likely to only find use at expansion time for now.  Possibly some later
// hot reloading may benefit from expanding runtime support.
enum PipelineFamily {
    Graphics,
    Compute,
    RayTracing,
}

fn has(stages: S, flag: S) -> bool {
    stages & flag != S::empty()
}

fn merge_all(flags: &[S]) -> S {
    flags.iter().fold(S::empty(), |acc, &f| acc | f)
}

const GRAPHICS_MASK: S = S::from_raw(
    S::VERTEX.as_raw()
        | S::TESSELLATION_CONTROL.as_raw()
        | S::TESSELLATION_EVALUATION.as_raw()
        | S::GEOMETRY.as_raw()
        | S::FRAGMENT.as_raw(),
);

const COMPUTE_MASK: S = S::COMPUTE;

const RAY_MASK: S = S::from_raw(
    S::RAYGEN_KHR.as_raw()
        | S::MISS_KHR.as_raw()
        | S::CLOSEST_HIT_KHR.as_raw()
        | S::ANY_HIT_KHR.as_raw()
        | S::INTERSECTION_KHR.as_raw()
        | S::CALLABLE_KHR.as_raw(),
);

pub fn family_of(stage: S) -> Option<PipelineFamily> {
    if has(GRAPHICS_MASK, stage) {
        Some(PipelineFamily::Graphics)
    } else if has(COMPUTE_MASK, stage) {
        Some(PipelineFamily::Compute)
    } else if has(RAY_MASK, stage) {
        Some(PipelineFamily::RayTracing)
    } else {
        None
    }
}

pub fn required_peers(stage: S, all_stages: S) -> S {
    let mut required = S::empty();
    match stage {
        // TCS and TES are always paired.
        S::TESSELLATION_CONTROL => {
            if !has(all_stages, S::TESSELLATION_EVALUATION) {
                required |= S::TESSELLATION_EVALUATION;
            }
        }
        S::TESSELLATION_EVALUATION => {
            if !has(all_stages, S::TESSELLATION_CONTROL) {
                required |= S::TESSELLATION_CONTROL;
            }
        }
        // GEOMETRY requires a vertex stage upstream.
        S::GEOMETRY => {
            if !has(all_stages, S::VERTEX) {
                required |= S::VERTEX;
            }
        }
        // FRAGMENT requires a vertex stage to have produced primitives.
        S::FRAGMENT => {
            if !has(all_stages, S::VERTEX) {
                required |= S::VERTEX;
            }
        }
        // INTERSECTION must have at least one hit shader to dispatch into.
        S::INTERSECTION_KHR => {
            if !has(all_stages, S::CLOSEST_HIT_KHR) && !has(all_stages, S::ANY_HIT_KHR) {
                required |= S::CLOSEST_HIT_KHR | S::ANY_HIT_KHR;
            }
        }
        _ => {}
    }
    required
}
