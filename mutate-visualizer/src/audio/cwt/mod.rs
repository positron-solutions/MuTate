// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Continuous Wavelet Transform
//!
//! > **Colonel Harland Sandurz:** Sir, are we being too literal?
//! >
//! > **Dark Helmet:** No, you fool. We're following orders. We were told
//! >                  to comb the desert, so we're combing it.
//!
//! The world is too full of blocky FFTs, DFTs, STFTs, Constant-Q Transforms, and other ideas based
//! on making little phasors shaped like wavelets.  The wavelet is by comparison pre-multiplied.
//! This cuts out the awful serial dependency of spinning that little phasor for every single bin.
//! We can of course parallelize the phasor, but why keep generating the same numbers at all?
//!
//! Meet the wavelet transform.  It's cheap.  We basically just multiply the input by a comb with a
//! lot of teeth.  Some of the teeth are longer, so we run them on multiple lanes to balance waves
//! and then just shuffle-add reduce at end.  Because it's so cheap and embarrassingly parallel, we
//! can afford a whole lot of teeth.  Using downsampled inputs further helps level and reduce work.
//!
//! ## Outputs
//!
//! Raw complex numbers (and maybe phase timings?  Not sure yet. Just write the center frequency and
//! a zero phase for now?). Integration for reassignment is done downstream. We don't know which
//! wavelet will be processed when, so it's easier (and more flexible for multiple downstreams) to
//! just barrier after the complex values are out there.
//!
//! The data is provided as a device address of the rings, an array of their offsets, and a table of
//! bin indexes for lookups.  Since bins are the only read that can be guaranteed to benefit from
//! contiguous layout, each bin's ring is contiguous and neighboring bins **do not have a uniform
//! stride**.  Reading across time is difficult without a gather operation.  ꙰H̷͙̱̼̫̑̿̈́̈́͗̓͑͝ë̸̜̜́̓̽̔̎͛͌͝l̶͍̗̽̎̓͌̄͝͠p̴̯̺̠̈́͊̐͆̅̓꙰
//!
//! ## Implementation
//!
//! - Voices use downsampled input where available
//!
//! ### The Slang
//!
//! - Wavelets are mapped across several lanes to level their wavelet length-per-lane
//! - Warps are leveled as a whole within workgroups.
//! - Several workgroups are provided to enable the scheduler to level them.
//! - Lane assignment is calculated in a prologue front-load divergences into a single wider
//!   prologue.
//! - Lanes spiral inward on weights to accumulate from smallest to largest.
//! - Shuffle gather and reduce to accumulate with the final center weight.
//!
//! #### Dispatch Geometry
//!
//! In which we avoid tails at all costs.
//!
//! - Bins have approximately level *average* load.
//! - Some bins could have chaotic worst-case load on some audio ticks.
//! - We want to share the audio inputs in LDS because those don't depend on which bins we're
//!   processing.
//! - Large-ish workgroups will use LDS atomics to carve up coarse grained work chunks.
//! - Queue scoped atomics will enable several over-subscribed workgroups to reserve chunks of work.
//!
//! ### Indexes
//!
//! These bins are constant Q.  This means we are back to the world of different lengths of windows
//! and different sized hops, reading from different sample rates, we need a table of values to
//! assist downstream calculations.  Each bin has a delay, hop size, center frequency, sample rate,
//! and gain normalization factor.
//!
//! Where this really becomes a problem is for downstream readers.  Luckily, the plan is to
//! immediately gather the data back out of the rings in a series of re-samplers that provide
//! reassigned, cross-correlated outputs one row at a time with rows neatly dividing along time.
//! State shadows from the host would be rather complex to send over, so the audio read head is
//! provided and shaders perform lane assignment at the preamble of their run.
//!
//! ## Memory Layout
//!
//! ⚠️ The structures are defined as if intended for mutable and immutable sub-allocations, but in
//! the interest of time, one big mutable allocation is used instead.
//!
//! ## The Wavelet
//!
//! We're going with a Morse family wavelet to start with.  See [`wavelet`] module.

// NEXT We're not using any slang reflection.  This is the motivating problem for which we would
// even finish implementing the field checks 🤡.  Look at all of these buffer device addresses that
// have some coercion target!  This will require some newtype wrapping.  Probably the biggest time
// saver would be checking field orders.  99 times out of 10, if any type or alignment is wrong,
// it's because fields are missing, extra, or out of order.  That doesn't depend on fine-grained
// size and alignment accounting, so it's probably worth it to begin the JSON imports of reflection
// data and begin checks.

mod warps; // tests used to calibrate workgroup geometry
mod wavelet;

use std::mem::MaybeUninit;

use ash::vk;
use num_complex::Complex32 as Complex;
use num_traits::One;

use mutate_lib::tree::TreeSum;
use mutate_lib::{self as utate, audio::import::DeviceAudioView, prelude::*};
use utate::dsp::{self, bank, window::WindowFunction};

use crate::audio::downsample::DownsampleOutput;

use super::downsample;
use super::plan;

const BINS: u32 = 3840;
const WARP_SIZE: u32 = 32;

/// Published downstream for consumers.
#[derive(Debug, Clone)]
pub struct Output {
    // TODO fields
}

#[derive(Debug, Clone)]
pub struct Dispatch {
    pub output: Output,
    pub consumed: u32,
}

// TODO skeleton for the shader

struct Config {
    // TODO fields
}

struct ChannelConfig {
    // TODO input rings
    // TODO output rings
}

struct BinConfig {
    // TODO fields that we need per bin
}

#[compute_pipeline(
    compute = stage!("audio/cwt", Compute, c"main"),
    push = push!(DftPushConstants {
        /// Coerces to `DftConfig` in slang.  Static data offsets use this base.
        static_base: DeviceAddress,
        /// Coerces to `ChannelState*` in slang.  All dynamic offsets are relative to this base.
        dynamic_base: DeviceAddress,
    }),
)]
pub struct Pipeline;

/// Continuous Wave Transform
pub struct Cwt {
    allocation: MappedAllocation<u8>,
    pipeline: ComputePipeline<Pipeline>,
}

impl Cwt {
    pub fn new(device: &Device, input: DownsampleOutput) -> Result<Self, MutateError> {
        // Obtain a set of log spaced bin centers across the audible / prettily drawing range.
        let bins = bank::bins(dsp::MIN_FREQ_CHEAP_DRIVERS, 15_000.00, BINS as usize);

        // Laying out the memory is essentially just walking the sizes and offsets of everything we
        // want to put in it, remembering those values to later populate the memory, and allocating
        // the total sizes.

        // Walk the static section to find offsets and size.
        let mut c = plan::Cursor::default();
        let config_offset = c.push::<Config>(1);

        let static_bytes = c.align_to(256);

        // Walk the dynamic section to find offsets and size.
        let mut c = plan::Cursor::default();

        let dynamic_bytes = c.len();

        // allocate
        let mut allocation = utate::vulkan::resource::buffer::MappedAllocation::<u8>::new(
            device,
            (static_bytes + dynamic_bytes) as usize,
        )?;

        // write control data

        // set up state shadows

        // ROLL Flush likely redundant.  This will usually be fully coherent BAR/ReBAR window, but
        // that abstraction isn't ready yet.
        allocation.flush(device);

        let base = allocation.device_address(device)?;

        // construct self
        todo!()
    }

    /// For each processing quantum, dispatch once.  Barrier the hot region of the input and output
    /// buffers.
    // DEBT we could not provide a command buffer with a type-erased submission model.  This is an
    // oversight in the cb module.  Fix might involve some indirection on the SubmissionModel.  Not
    // really important, so just fully specified the submission model instead.
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        downsample: &downsample::DownsampleOutput,
    ) -> Result<Dispatch, MutateError> {
        // calculate indexes

        // push constants

        // dispatch

        // update state shadows

        // publish host data

        todo!()
    }

    pub fn destroy(self, device: &Device) {
        device.deletion_queue.push(self.allocation.buffer);
        device.deletion_queue.push(self.allocation.memory);
        self.pipeline.destroy(device);
    }

    /// Provides downstreams with an idea where to get output before it even starts flowing.  Hope
    /// your initializations aren't garbage. ⚝
    pub fn view(&self) -> Output {
        todo!()
    }
}
