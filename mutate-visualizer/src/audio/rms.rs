// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Rms
//!
//! Expose RMS of the input as an output buffer.

// NEXT A-weight and interpolate outputs into a 240Hz output ring.  Single output is inappropriate
// for consumers.  This module was only built up just enough to be sure some things were working.

use ash::vk;
use mutate_lib::{self as utate, audio, prelude::*};

#[compute_pipeline(
    compute = stage!("audio/rms", Compute, c"main"),
    push = push!(RmsConstants {
        pub left_head: DeviceAddress,
        pub right_head: DeviceAddress,
        pub count_head: UInt,
        pub left_tail: DeviceAddress,
        pub right_tail: DeviceAddress,
        pub count_tail: UInt,
        pub output: DeviceAddress,
    }),
)]
pub struct RmsComputePipeline;

pub struct Rms {
    pub pipeline: ComputePipeline<RmsComputePipeline>,
}

impl Rms {
    pub fn new(device: &Device) -> Result<Self, MutateError> {
        Ok(Rms {
            pipeline: ComputePipeline::<RmsComputePipeline>::new(device)?,
        })
    }

    pub fn destroy(self, device: &Device) {
        self.pipeline.destroy(device);
    }
}
