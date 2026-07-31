// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Rms
//!
//! Expose RMS of the input as an output buffer.

// NEXT work on how we expose intended outputs for downstreams.  The output address is the key value
// of interest.  Need a way to declare that in the upcoming resources interfaces.
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
    pub constants: RmsConstants,
    pub output: DeviceBuffer,
    pub output_address: DeviceAddress,
}

impl Rms {
    pub fn new(device: &Device) -> Result<Self, MutateError> {
        // Four bytes to tell the world!  Note, this refreshes so fast that downstreams will not be
        // able to track it, leading to aliasing.
        let output = utate::vulkan::resource::buffer::DeviceBuffer::new(device, 4)?;
        let output_address = output.device_address(device)?;
        Ok(Rms {
            pipeline: ComputePipeline::<RmsComputePipeline>::new(device)?,
            constants: RmsConstants {
                left_head: DeviceAddress::NULL,
                right_head: DeviceAddress::NULL,
                count_head: 0.into(),
                left_tail: DeviceAddress::NULL,
                right_tail: DeviceAddress::NULL,
                count_tail: 0.into(),
                output: output_address.clone().into(),
            },
            output,
            output_address: output_address.into(),
        })
    }

    pub fn destroy(self, device: &Device) {
        self.pipeline.destroy(device);
        self.output.destroy(device);
    }
}
