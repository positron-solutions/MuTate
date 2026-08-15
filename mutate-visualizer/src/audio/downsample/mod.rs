// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Downsample
//!
//! > I have supped full with horrors.
//! > Direness, familiar to my slaughterous thoughts,
//! > Cannot once start me.
//! >
//! > - Frederick Allan Moranis
//!
//! Long filter windows cost a lot.  Downsampling reduces the cost.  We FIR low-pass and decimate to
//! save on cost.  Savings taper off (fewer bins affected and less absolute input rate reduction)
//! while delay goes up for extreme downsampling, so we eat the rate cost in the lowest octaves.
//! High pitch samples are incidentally attenuated with a safety margin that makes the output signal
//! cleaner than it was if the same bins are filtering the full-rate input.
//!
//! - 2x downsampling for use below 1/4 Nyquist input
//! - 4x downsampling for use below 1/8 Nyquist input
//! - 8x downsampling for use below 1/16 Nyquist input
//! - 16x downsampling for use below 1/32 Nyquist input down to DC
//!
//! Biggest savings for the longest bins basically achieved already, so no 32x downsampling is
//! planned, although the GPU will happily crunch it in dozens of cycles at most.
//!
//! ## The Slang
//!
//! The shader is built around single-stage, direct FIR from full rate to target rate for every
//! downsampling level.  While the filters are long, the data sharing is minimal and we use several
//! lanes per frame, more lanes for longer filters to level the load across warps.  At usually 1k
//! lanes, the chip barely wakes up before the filtering is done.
//!
//! Feel free to look for other techniques, but the main on-device constraint for FIR downsampling
//! seems to be that serial dependencies across warps are absolutely terrible, so going directly
//! from signal to output is almost always going to win.  The device has enough lanes that a single
//! audio server tick will struggle to make use of the full width.
//!
//! ## The Weights
//!
//! Pre-decimation FIR low-pass that crushes everything above 3/4 input Nyquist, using a wide 1/4 to
//! 3/4 Nyquist as the transition band.  Parks McClellan Remez weight generation.  With the wide
//! transition, we get:
//!
//! - Low ripple in the pass so signal we want is unchanged.
//! - Post-fold transition noise is at least -6dB dampened, so it will net attenuate post-fold and be
//!   easier to filter from the pass band.
//! - Some extra stop band cushion so that bins at the top of the new pass band won't see folded
//!   transition.
//! - Some extra pass band cushion so that bins at the top of the pass can perform reassignment of
//!   their full main lobe.
//! - About -90dB stop band attenuation so our bins in the pass aren't seeing the folded signals.
//! - Odd numbers of taps, fitted to neatly level the load across warps.
//!
//! There is a **trick** of the down-sample folding used to achieve a really big transition band.
//! We guarantee *attenuation* of folded regions rather than setting the stop band to the new
//! Nyquist.  The ripple continues beyond the new Nyquist, all the way up to the folding mirror of
//! the pass band edge.
//!
//! We can obtain sufficient transition attenuation with a lower number of taps.  This would buy us
//! some delay reduction at the cost of more ripple, likely in the pass band, or less noise floor
//! margin.
//!
//! ### Filter Sharp Edges
//!
//! Input at the new sample rate Nyquist is not *amplified* from the source signal or noise, but
//! it would be evident within the usable *dynamic ranges* of the pass band (not the pitch ranges).
//! We don't use the area right at the new Nyquist, so this is fine?  In theory, but maybe not
//! practice.  We will defer to the opinions of long-time DSP professionals through an open source
//! development process.
//!
//! ⚠️ As a result of the transition admitting reflected signal, especially near the new Nyquist,
//! bins should be very careful to **not let their main lobe get into the transition band.** Beyond
//! the transition band guard, folded signals and noise begin ramping all the way to the new
//! Nyquist.  Use a higher sample rate and the transition will be nowhere near where you are
//! analyzing 😼.
//!
//! ⚠️ Filter bank design should be aware that the transition contains extreme grooves and undefined
//! attenuation that may scallop main lobes or obliterate side lobes.  This may lead to grooves in
//! the overall response if not calibrated.  Again, use a higher sample rate if there's a problem!
//!
//! ## Delay
//!
//! A uniform group delay per octave was chosen just to be tidy.  21/41/81/161 taps give 5 output
//! samples of relative delay at each successive level.

// NEXT There is a group delay introduced by INPUT_QUANTUM that needs to be either carried forward
// or removed by making each filter's quantum independent.  Implementation will likely involve
// passing LEVELS indexes via push constants.  Related, the shader
// NEXT Write the LEVELS output ring offset data to a header so that the host side of sharing
// downstream is dynamic state only.  We will need to start doing slang libraries, but we absolutely
// should!

mod weights;

use ash::vk;
use mutate_lib::{self as utate, audio::import::DeviceAudioView, prelude::*};

use super::plan;

/// LCM of the decimation factors.  The scheduling quantum, in input samples.
const INPUT_QUANTUM: u64 = 1 << LEVELS; // 16
/// Level-0 outputs per quantum.  `count[0]` is always a multiple of this.
const OUTPUT_QUANTUM: u64 = INPUT_QUANTUM / 2; // 8

const LEVELS: usize = 4;
pub const FILTERS: [weights::Lowpass; LEVELS] = [
    weights::DOWN_TWO,
    weights::DOWN_FOUR,
    weights::DOWN_EIGHT,
    weights::DOWN_SIXTEEN,
];

const LEVEL_SAMPLES: [u32; LEVELS] = [8192, 4096, 2048, 1024];

/// Must match `PAIRS_PER_LANE` in audio/downsample.slang.
const PAIRS_PER_LANE: u32 = 10;

/// Deepest lookahead demanded by any level.
const MAX_RADIUS: u64 = FILTERS[LEVELS - 1].radius() as u64;

const _: () = {
    let mut level = 0;
    while level < LEVELS {
        assert!(FILTERS[level].radius() == PAIRS_PER_LANE << level);
        assert!(FILTERS[level].decimation == 1 << (level + 1));
        level += 1;
    }
};

/// One level of downsampling.  A set of `Rings` and metadata about the decimation and cutoff
/// frequency.
#[derive(Clone, Copy, Debug)]
pub struct LevelOutput {
    /// Offset from the output base.
    pub offset: u32,
    /// Logical center?
    pub write_head: u64,
    /// Number of channels
    pub channel_count: u32,
    /// Number of samples in each channel
    pub sample_count: u32,
    /// Input samples per output sample.
    pub decimation: u32,
    /// Highest frequency (Hz) a bin center may sit at.
    pub cutoff: f32,
    /// Group delay in input samples.
    pub delay: u32,
}

impl LevelOutput {
    pub fn mask(&self) -> u32 {
        self.sample_count - 1
    }
}

/// Location,
#[derive(Clone, Copy, Debug)]
pub struct DownsampleOutput {
    /// All channels in the payload are based 🥷🏿.
    pub base_address: DeviceAddress,
    /// Each output level has metadata and channels.
    pub levels: [LevelOutput; LEVELS],
    /// Oldest input sample this module may still read.
    pub retain_floor: u64,
}

impl DownsampleOutput {
    // XXX pass sample rate.  Make level sample-rate normalized
    /// Deepest level whose read band covers `center_hz`.  `None` means the frequency belongs to
    /// the full-rate stream.  Assumes the last level is terminal, so nothing falls out the bottom.
    pub fn level_for(&self, center_hz: f32) -> Option<usize> {
        (0..LEVELS)
            .rev()
            .find(|&l| center_hz <= self.levels[l].cutoff)
    }
}

pub struct DownsampleDispatch {
    pub output: DownsampleOutput,
    pub consumed: u32,
}

/// Header of the static base.
#[repr(C)]
struct Config {
    /// Base of our dynamic section.  Written after allocation; see `new`.
    output_base: DeviceAddress,
    /// -> `RingView[LEVELS * CHANNELS]`, from `static_base`.
    output_views_offset: u32,
    /// Major stride of the output view table.  == CHANNELS.
    output_views_stride: u32,
    /// -> `LevelConfig[LEVELS]`, from `static_base`.
    level_configs_offset: u32,
}

// XXX maybe not actually pub
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RingView {
    /// Offset from the relevant base, in *bytes*.
    offset: u32,
    /// Index wrap mask, in *samples*.
    mask: u32,
}

#[repr(C)]
struct LevelConfig {
    /// -> `f32[radius + 1]`, from `static_base`, outward-in.
    weights_offset: u32,
    radius: u32,
    decimation_log2: u32,
    lanes_per_output_log2: u32,
}

// XXX first_window_center is identical.  We can use this to perhaps let each level decouple from
// the input quantum at the cost of a bit more state shadowing on the host.  Probably worth it to
// get rid of the group delay.
#[compute_pipeline(
    compute = stage!("audio/downsample", Compute, c"main"),
    push = push!(PushConstants {
        /// Base of the *input* rings.  Upstream's allocation, not ours.
        input_base: DeviceAddress,
        /// Coerces to `Config` in slang.  Static data offsets use this base.
        static_base: DeviceAddress,
        /// Bytes between input rings
        input_stride: UInt,
        /// Index wrapping mask for input ring, in samples
        input_mask: UInt,

        /// Physical index of the center sample where the first downsample filter will be applied.
        first_window_center: [UInt; 4],
        /// Physical index of the first output slot
        first_output_slot: [UInt; 4],
        /// How many outputs write,
        output_count: [UInt; 4],
    }),
)]
pub struct Pipeline;

pub struct Downsample {
    allocation: MappedAllocation<u8>,

    /// Bases are handed to the device every dispatch via push constants
    static_base: DeviceAddress,
    output_base: DeviceAddress,

    output_channel_count: u32,
    output_level_count: u32,

    /// Byte offset from `output_base` to each level's channel 0.
    output_level_offsets: [u32; LEVELS],

    pipeline: ComputePipeline<Pipeline>,
    timeline: TimelineSemaphore,

    /// For now, we might get away with a very specific rate of advance.  Given that downsample
    /// frames never tick downstream except on certain inputs, this might actually work.
    next_center: u64,

    /// Upstream sample rate.
    sample_rate: u32,
}

impl Downsample {
    pub fn new(device: &Device, channels: u32, sample_rate: u32) -> Result<Self, MutateError> {
        // NOTE The big idea is no different for any of these pipelines.  We plan the layout, grab
        // the allocation, initialize it, and set up our local state shadows we'll need for later
        // dispatch.  Return a dispatch that tells downstream how to use what we wrote.

        // plan static
        let mut c = plan::Cursor::default();
        let config_offset = c.push::<Config>(1);
        let output_views_offset = c.push::<RingView>((LEVELS as u32 * channels));
        let level_configs_offset = c.push::<LevelConfig>(LEVELS as u32);

        let mut weights_offsets = [0u32; LEVELS];
        for level in 0..LEVELS {
            weights_offsets[level] = c.push::<f32>(FILTERS[level].radius() + 1);
        }
        let static_bytes = c.align_to(256);

        // plan outputs
        let mut c = plan::Cursor::default();
        let mut output_level_offsets = [0u32; LEVELS];
        for level in 0..LEVELS {
            output_level_offsets[level] = c.push::<f32>(LEVEL_SAMPLES[level] * channels);
        }
        let dynamic_bytes = c.len();

        let mut allocation = utate::vulkan::resource::buffer::MappedAllocation::<u8>::new(
            device,
            (static_bytes + dynamic_bytes) as usize,
        )?;

        // Address first: an immutable borrow that ends here, before the split below takes the
        // allocation mutably for the rest of the function.
        let base = allocation.device_address(device)?;
        let static_base: DeviceAddress = base.into();
        let output_base: DeviceAddress = (base + static_bytes as u64).into();

        // XXX isn't the "dynamic" address the output address?  Where else would we enable writes if
        // we later do a static-dynamic allocation split?
        let (stat, _dynam) = allocation
            .as_mut_slice()
            .split_at_mut(static_bytes as usize);

        plan::put(
            stat,
            config_offset,
            Config {
                output_base,
                output_views_offset,
                output_views_stride: channels,
                level_configs_offset,
            },
        );

        // Weights are symmetric, so we store half.
        for level in 0..LEVELS {
            plan::put_slice(stat, weights_offsets[level], FILTERS[level].folded());
        }

        // Output views: [level][channel], stride == CHANNELS.
        for level in 0..LEVELS {
            for ch in 0..channels {
                let i = level as u32 * channels + ch;
                plan::put(
                    stat,
                    output_views_offset + i * size_of::<RingView>() as u32,
                    RingView {
                        offset: output_level_offsets[level] + ch * LEVEL_SAMPLES[level] * 4,
                        mask: LEVEL_SAMPLES[level] - 1,
                    },
                );
            }
        }

        for level in 0..LEVELS {
            plan::put(
                stat,
                level_configs_offset + level as u32 * size_of::<LevelConfig>() as u32,
                LevelConfig {
                    weights_offset: weights_offsets[level],
                    radius: FILTERS[level].radius(),
                    decimation_log2: level as u32 + 1,
                    lanes_per_output_log2: level as u32,
                },
            );
        }

        Ok(Self {
            allocation,
            static_base,
            output_base,
            output_level_offsets,
            output_channel_count: channels,
            output_level_count: LEVELS as u32,
            pipeline: ComputePipeline::<Pipeline>::new(device)?,
            timeline: device.make_timeline_semaphore()?,
            next_center: 0,
            sample_rate,
        })
    }

    // NOTE The dft.rs accepts a layout on new and stores it in the config.  This shader
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        input: &DeviceAudioView,
    ) -> Result<DownsampleDispatch, MutateError> {
        // Calculate the downsampler advances from the upstream write head.
        let count = self.ready_outputs(input.write_head);
        let input_mask = input.sample_count - 1;
        let constants = PushConstants {
            input_base: input.base_address,
            static_base: self.static_base,
            input_stride: (input.sample_count * 4).into(),
            input_mask: input_mask.into(),

            first_window_center: std::array::from_fn(|_| {
                ((self.next_center as u32) & input_mask).into()
            }),
            first_output_slot: std::array::from_fn(|l| {
                (((self.next_center >> (l + 1)) as u32) & (LEVEL_SAMPLES[l] - 1)).into()
            }),
            output_count: std::array::from_fn(|l| (count >> l).into()),
        };
        self.pipeline.push(device, **cb, &constants);

        let groups_x = count.div_ceil(256);
        unsafe {
            device.cmd_bind_pipeline(**cb, vk::PipelineBindPoint::COMPUTE, *self.pipeline);
            device.cmd_dispatch(**cb, groups_x, LEVELS as u32, self.output_channel_count);
        }

        self.next_center += count as u64 * 2;
        debug_assert_eq!(self.next_center % INPUT_QUANTUM, 0);

        Ok(DownsampleDispatch {
            output: self.view(),
            consumed: count,
        })
    }

    pub fn destroy(self, device: &Device) {
        device.deletion_queue.push(self.allocation.buffer);
        device.deletion_queue.push(self.allocation.memory);
        device.deletion_queue.push(self.timeline.into_raw());
        self.pipeline.destroy(device);
    }

    /// How many level-0 outputs are ready, given total samples ever written to the
    /// input ring.  Level `L` gets `count >> L`.
    ///
    /// An output centered at logical index `c` reads `[c - radius, c + radius]`.
    /// Lockstep scheduling means every level waits on `MAX_RADIUS` of lookahead,
    /// so the newest center we may close is `written - 1 - MAX_RADIUS`.
    fn ready_outputs(&self, written: u64) -> u32 {
        let Some(newest_center) = written.checked_sub(1 + MAX_RADIUS) else {
            return 0; // ring hasn't filled the first window yet
        };
        if newest_center < self.next_center {
            return 0;
        }

        // Centers are `next_center, next_center + 2, ..` up to and including newest.
        let count = (newest_center - self.next_center) / 2 + 1;

        // Floor to the quantum so every level gets a whole number of outputs, and
        // clamp so no level laps its output ring in a single dispatch.
        let count = count & !(OUTPUT_QUANTUM - 1);
        count.min(LEVEL_SAMPLES[0] as u64 / 2) as u32
    }

    /// Oldest input sample this module may still read.  Import must not reclaim below this.
    pub fn retain_floor(&self) -> u64 {
        self.next_center.saturating_sub(MAX_RADIUS)
    }

    /// Return the state and layout information independent of dispatching.  Useful for initializing
    /// downstreams.
    pub fn view(&self) -> DownsampleOutput {
        let sample_rate = self.sample_rate as f32;
        DownsampleOutput {
            base_address: self.output_base,
            levels: std::array::from_fn(|l| {
                let rate = self.sample_rate as f32 / (1 << (l + 1)) as f32;
                let f = &FILTERS[l];
                LevelOutput {
                    offset: self.output_level_offsets[l],
                    channel_count: self.output_channel_count,
                    sample_count: LEVEL_SAMPLES[l],
                    decimation: 1 << (l + 1),
                    cutoff: f.cutoff(sample_rate),
                    delay: f.radius(),
                    write_head: self.next_center >> (l + 1),
                }
            }),
            retain_floor: self.retain_floor(),
        }
    }
}
