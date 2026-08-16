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
//! ## Problem Symmetry & Similarity
//!
//! Unlike the downsampling, not everything fits in L1.  Reading one-weight-per-lane would likely
//! amplify reads, so we need to better take advantage of regularities.
//!
//! - All bins have a computation cost proportionate to their sample rate alone.  If bins are
//!   padded, this only slightly perturbs this relation.  This allows us to balance work groups over
//!   the sample rate alone!
//! - All three weights are Hermitian in time.  This is exploited for reducing storage and read.
//!   The multiply add also gets to add two input samples per weight multiply.
//! - Weights are re-used on each window.  The phase of the output requires correction, but this
//!   amortizes well over the window cost.
//! - To use weights contiguously, the number of unique bins in flight and the cache line for each
//!   weight read is shared by many lanes.  Faster waves complete more windows and their workgroups
//!   move on to the next bin.
//! - The same audio is read heavily by each workgroup, so it is stored in LDS.
//! - Most bin weights fit within a similar range of lengths thanks to the (coincidental but
//!   natural) octave structure of the downsampling.
//!
//! ## Outputs
//!
//! Most importantly, outputs are complex and laid out source-bin-contiguous.  During integration,
//! the complex values can be interpreted along with `BinConfig` to
//!
//! Each bin has a delay, hop size, center frequency, sample rate, and gain normalization factor.
//! There is a table of values to assist downstream calculations.
//!
//! ## Implementation Tradeoffs
//!
//! - Voices use downsampled input where available.  Small group delay.  Massive COLA cost
//!   improvement.  Some complexity.
//! - Constant Q.  Back to the world of different lengths of windows and different sized hops,
//!   reading from different sample rates.  Outputs per input quantum vary per bin.
//! - Batched output? Any output ring for windows with varying hop sizes would just lead to one
//!   independent ring per bin.  The gathers would be neither time nor bin contiguous and the
//!   indexing would the extremely tricky.  Reading across time is difficult without a gather
//!   operation.  ꙰H̷͙̱̼̫̑̿̈́̈́͗̓͑͝ë̸̜̜́̓̽̔̎͛͌͝l̶͍̗̽̎̓͌̄͝͠p̴̯̺̠̈́͊̐͆̅̓꙰
//!
//! ### The Slang
//!
//! - Workgroups target a single sampling rate to make use of LDS.
//! - Wavelets are mapped across several lanes to level their wavelet length-per-lane but cache
//!   behavior and avoiding read amplification is also taken into consideration.
//! - Workgroups overlap over bins to enable the scheduler to level hop noise.
//! - Lanes spiral inward on weights (using symmetry) to accumulate from smallest to largest.
//! - Shuffle gather and reduce to accumulate with the final center weight.
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
// NEXT Wavelet truncation should cooperate with bin length.  Padding helps avoid divergence. If
// we're going to pad, keep some extra accuracy!
// NEXT Some taps are possibly near identical, not just the same length, but actually identical,
// thanks to the octave structure of downsampling.
// XXX Fold is anti-Hermitian for reassignment weights, so we have to unpack the conjugates.

mod warps; // tests used to calibrate workgroup geometry
mod wavelet;

use std::mem::MaybeUninit;

use ash::vk;
use num_complex::Complex32 as Complex;
use num_traits::One;

use mutate_lib::{self as utate, prelude::*};
use utate::dsp::{self, bank};

use crate::audio::downsample::DownsampleOutput;

use super::downsample;
use super::plan;

const BINS: u32 = 3840;
const WARP_SIZE: u32 = 32;
const MAX_FREQ: f64 = 16_000.0;
const SAMPLE_RATE: f32 = 48_000.0;

/// Overlap factor for COLA.  Hop is `length / COLA`.
const COLA: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct Dispatch {
    // NOTE Using a GPU driven style.  Header structure is a `CwtOutput` from which the bin table,
    // and batch output can be interpreted downstream.  Address just enables host to dispatch.
    pub output: DeviceAddress,
}

// XXX does not exist
#[derive(Debug)]
pub struct Output {}

struct Config {
    // TODO fields
}

struct ChannelConfig {
    // TODO input rings
    // TODO output rings
}

// TODO skeleton for the shader
#[compute_pipeline(
    compute = stage!("audio/cwt", Compute, c"main"),
    push = push!(CwtPushConstants {
        /// Coerces to `DownsampleOutput`.
        input_base: DeviceAddress,
        /// Coerces to `DftConfig` in slang.  Static data offsets use this base.
        static_base: DeviceAddress,
        /// Coerces to `ChannelState*` in slang.  All dynamic offsets are relative to this base.
        output_base: DeviceAddress,

        // TODO figure out the actual necessary shape of control fields
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

        // XXX investigate half taps via warp tool!!
        // 24 bytes (six floats) per tap.  At 1,303,106 taps, that's  31,274,544 bytes worth of taps
        // table and the reason we try to keep each workgroup on a single set of taps.

        let static_bytes = c.align_to(256);

        // Walk the dynamic section to find offsets and size.
        let mut c = plan::Cursor::default();

        let dynamic_bytes = c.len();

        // allocate
        let mut allocation = utate::vulkan::resource::buffer::MappedAllocation::<u8>::new(
            device,
            // XXX wrong size
            (static_bytes + dynamic_bytes + 128) as usize,
        )?;

        // write control data

        // set up state shadows

        // ROLL Flush likely redundant.  This will usually be fully coherent BAR/ReBAR window, but
        // that abstraction isn't ready yet.
        allocation.flush(device);

        let base = allocation.device_address(device)?;

        Ok(Self {
            allocation,
            pipeline: ComputePipeline::<Pipeline>::new(device)?,
        })
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

        Ok(Dispatch {
            output: DeviceAddress::NULL,
        })
    }

    pub fn destroy(self, device: &Device) {
        device.deletion_queue.push(self.allocation.buffer);
        device.deletion_queue.push(self.allocation.memory);
        self.pipeline.destroy(device);
    }

    /// Provides downstreams with an idea where to get output before it even starts flowing.  Hope
    /// your initializations aren't garbage. ⚝
    pub fn view(&self) -> Output {
        Output {}
    }
}
