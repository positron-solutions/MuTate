// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Stage
//!
//! *This stage has taken a while to reach.  Keep coding a lot.*
//!
//! Stage declarations are both about getting stages from disk into pipelines and supporting
//! compile-time contracts.  For fanning stage information into pipelines at proc-macro expansion
//! time, the reflection data needs to be attached to stage declaration via proc macro.

use std::ffi::CStr;

use ash::vk::{self, ShaderStageFlags as S};

use crate::internal::*;
use crate::resource::shader::ShaderModule;

// Same as with PushConstants, a lot of thought was put into loading and what kinds of check logic
// needs to exist only to discover that a lot could boil away into the macro code and the runtime
// logic that hydrates the specs.

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
pub trait StageReflection {
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
