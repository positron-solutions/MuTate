// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spectrograph
//!
//! > We sincerely apologize that some funds were temporarily located in places other than the
//! > places where those funds were intended to be located.
//! >
//! > - Samuel Benjamin Bankman-Fried
//!
//! The raw output of the CWT after an audio tick is essentially a scattered data set.  The purpose
//! of this module is to gather that scatter.  Readers want something closer to a spectrogram, a
//! ring buffer of more easily readable, time-aligned, de-messed-up output.  I'm getting dyslexia
//! just thinking about this data.
//!
//! ## Input
//!
//! - The bin ring buffer offsets.
//! - Bin table for tweaking reads across the bank.
//!
//! ## Output
//!
//! - Ring buffer per channel
//! - Rows correspond to time
//! - Columns correspond to bins
//! - Datum is a complex value...maybe magnitude and phase.  We'll see how complex it all is.

use ash::vk;

use mutate_lib::vulkan::resource::retired::RawHandle;
use mutate_lib::{self as utate, prelude::*};

use super::cwt;
use super::plan;

/// Output data location & geometry for consumers
#[derive(Clone, Copy, Debug)]
pub struct Output {
    pub base: DeviceAddress,
    pub channel_count: u32,
    pub channel_slots: u32,
    pub write_head: u64,

    // XXX compile
    pub ring_width: u32,
}

impl Output {
    pub fn bins(&self) -> u32 {
        0
    }

    pub fn slots(&self) -> u32 {
        0
    }
}

struct Config {}

#[compute_pipeline(
    compute = stage!("audio/spectrum", Compute, c"main"),
    push = push!(PushConstants {
        static_base: DeviceAddress,
    }),
)]
pub struct Pipeline;

pub struct Spectrum {
    allocation: MappedAllocation<u8>,
    static_base: DeviceAddress,
    output_base: DeviceAddress,
    pipeline: ComputePipeline<Pipeline>,
    write_head: u64,
}

impl Spectrum {
    pub fn new(device: &Device, input: cwt::Output) -> Result<Self, MutateError> {
        // XXX Get it from the input
        let channels = 2;

        let mut c = plan::Cursor::default();
        let config_offset = c.push::<Config>(1);
        let static_bytes = c.align_to(256);

        let mut c = plan::Cursor::default();
        let dynamic_bytes = c.len();

        // XXX padded to avoid dying
        let mut allocation =
            MappedAllocation::<u8>::new(device, (static_bytes + dynamic_bytes + 128) as usize)?;
        let base = allocation.device_address(device)?;
        let static_base: DeviceAddress = base.into();
        let output_base: DeviceAddress = (base + static_bytes as u64).into();

        let (stat, _dynam) = allocation
            .as_mut_slice()
            .split_at_mut(static_bytes as usize);
        plan::put(stat, config_offset, Config {});

        Ok(Self {
            allocation,
            pipeline: ComputePipeline::<Pipeline>::new(device)?,

            static_base,
            output_base,
            write_head: 0,
        })
    }

    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        input: &cwt::Output,
    ) -> Result<Output, MutateError> {
        // let count = self.ready_outputs(input.write_head);

        let constants = PushConstants {
            static_base: self.static_base, /* slots from head */
        };
        self.pipeline.push(device, **cb, &constants);

        let count: u32 = 1;
        unsafe {
            device.cmd_bind_pipeline(**cb, vk::PipelineBindPoint::COMPUTE, *self.pipeline);
            device.cmd_dispatch(**cb, count.div_ceil(256), 1, 2);
        }

        todo!()
    }

    pub fn destroy(self, device: &Device) {
        device.deletion_queue.push(self.allocation.buffer);
        device.deletion_queue.push(self.allocation.memory);
        device.deletion_queue.push(self.pipeline.into_raw());
    }

    // XXX state poll for setting up other things.

    /// Backpressure contract to upstream.
    pub fn retain_floor(&self) -> u64 {
        todo!()
    }
}
