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
//! save on cost.  Savings taper off (fewer bins affected and less absolute input length reduction)
//! while delay goes up for extreme downsampling, so we eat the cost in the lowest octaves.  High
//! pitch samples are incidentally attenuated with a safety margin that makes the output signal
//! cleaner than it was if the same bins are filtering the full-rate input.
//!
//! - 8x downsampling for use below 1/32 Nyquist input
//! - 4x downsampling for use below 1/16 Nyquist input
//! - 2x downsampling for use below 1/8 Nyquist input
//!
//! Biggest savings for the longest bins basically achieved.
//!
//! This is a fairly naive implementation, using several lanes per frame, just to get going. Feel
//! free to look for other techniques, but the main on-device constraint for FIR downsampling seems
//! to be that serial dependencies across warps are absolutely terrible, so going directly from
//! signal to output is almost always going to win.
//!
//! ## The Weights
//!
//! There is a trick of the down-sample folding used to achieve a really big transition band.  We
//! guarantee attenuation of noise from folding rather than no folded noise at all.  Input at the
//! new sample rate Nyquist, reach about 0.35 on a peak detector post filter, or about -9dB, leaving
//! a net of about -3dB for folded noise.  We don't use the area right at the new Nyquist, so this
//! is fine?  In theory, but maybe not practice.  We will defer to the opinions of long-time DSP
//! professionals through an open source development process.
//!
//! ⚠️ As a result of this design feature, bins should be very careful to **not let their main lobe
//! get into the transition band.** The transition band will be corrupted.  Use a higher sample rate
//! and the transition will be nowhere near where you are analyzing 😼.
//!
//! Pre-decimation FIR low-pass that crushes everything above 1/2 input Nyquist, using a wide 1/4 to
//! 3/4 Nyquist as the transition band.  Parks McClellan Remez weight generation.  With the wide
//! transition, we get:
//!
//! - Low ripple in the pass so signal we want is unchanged.
//! - Post-fold transition noise is at last -6dB dampened, so it will net attenuate post-fold and be
//!   easier to filter from the pass band.
//! - Some extra stop band cushion so that bins at the top of the new pass band won't see folded
//!   transition.
//! - Some extra pass band cushion so that bins at the top of the pass can perform reassignment of
//!   their full main lobe.
//! - About 80dB stop band attenuation so our bins in the pass aren't seeing the folded noise.
//! - Odd weights fitted to make decent use of warps.
//!
//! We can obtain sufficient transition attenuation with a lower number of taps.  This trades delay,
//! which would become significant at higher downsampling rates.
//!
//! ## Delay
//!
//! A uniform group delay per octave was chosen just to be tidy.  This kind of nicety on the output
//! may have usage for others.  The filters use 25/49/97 taps for a delay of 6 between each "stage"
//! (they are direct, not cascaded).

// DEBT I keep writing const generics for the channels.  At some point let's reverse this to runtime
// configuration.

mod weights;

use ash::vk;
use mutate_lib::{self as utate, audio::import::RingLayout, prelude::*};

use super::plan;

#[derive(Clone, Copy, Debug)]
pub struct LevelOutput<const CHANNELS: usize = 2> {
    /// Absolute base of each channel's ring.
    pub channels: [DeviceAddress; CHANNELS],
    /// Ring capacity in samples.  PoT; mask is `sample_count - 1`.
    pub sample_count: u32,
    /// Monotonic count of samples written at *this level's* rate.
    /// Physical slot is `write_head & (sample_count - 1)`.
    pub write_head: u64,
    /// `1 << (level + 1)`.  Input samples per output sample at this level.
    pub decimation: u32,
}

impl<const CHANNELS: usize> LevelOutput<CHANNELS> {
    pub fn mask(&self) -> u32 {
        self.sample_count - 1
    }
    pub fn write_slot(&self) -> u32 {
        self.write_head as u32 & self.mask()
    }
}

/// Geometry, location, and tracking for downstream consumers of all levels.
#[derive(Clone, Copy, Debug)]
pub struct DownsampleOutput<const CHANNELS: usize = 2> {
    base: DeviceAddress,
    ring_offsets: [[u32; CHANNELS]; LEVELS],
    /// Logical input index of the next window center.  The whole clock lives here:
    /// level `L`'s write head is `next_center >> (L + 1)`.
    pub next_center: u64,
    /// Oldest input sample still readable by this module.  Import honors this.
    pub retain_floor: u64,
    /// Signaled once this dispatch's writes are visible.
    pub data_ready: WaitValue,
}

impl<const CHANNELS: usize> DownsampleOutput<CHANNELS> {
    pub fn level(&self, level: usize) -> LevelOutput<CHANNELS> {
        LevelOutput {
            channels: std::array::from_fn(|ch| {
                (self.base.raw() + self.ring_offsets[level][ch] as u64).into()
            }),
            sample_count: LEVEL_SAMPLES[level],
            write_head: self.next_center >> (level + 1),
            decimation: 1 << (level + 1),
        }
    }
}

pub struct DownsampleDispatch<const CHANNELS: usize = 2> {
    pub output: DownsampleOutput<CHANNELS>,
    pub ready: SignalIntent,
    pub consumed: u32,
}

pub const LEVELS: usize = 3;

/// Folded weight counts: `radius + 1` for 25/49/97 taps.
const RADIUS: [u32; LEVELS] = [12, 24, 48];
const _: () = assert!(weights::DOWN_TWO.len() as u32 == RADIUS[0] * 2 + 1);
const _: () = assert!(weights::DOWN_FOUR.len() as u32 == RADIUS[1] * 2 + 1);
const _: () = assert!(weights::DOWN_EIGHT.len() as u32 == RADIUS[2] * 2 + 1);

/// Header of the static base.
#[repr(C)]
struct Config {
    /// Base of the *input* rings.  Upstream's allocation, not ours.
    input_base: DeviceAddress,
    /// Base of our dynamic section.  Written after allocation; see `new`.
    output_base: DeviceAddress,

    /// -> `RingView[CHANNELS]`, from `static_base`.
    input_views_offset: u32,
    /// -> `RingView[LEVELS * CHANNELS]`, from `static_base`.
    output_views_offset: u32,
    /// Major stride of the output view table.  == CHANNELS.
    output_views_stride: u32,
    /// -> `LevelConfig[LEVELS]`, from `static_base`.
    level_configs_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RingView {
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

#[compute_pipeline(
    compute = stage!("audio/downsample", Compute, c"main"),
    push = push!(PushConstants {
        /// Coerces to `Config` in slang.  Static data offsets use this base.
        static_base: DeviceAddress,
        /// Physical index of the center sample where the first downsample filter will be applied.
        first_window_center: [UInt; 3],
        /// Physical index of the first output slot
        first_output_slot: [UInt; 3],
        /// How many outputs write,
        output_count: [UInt; 3],
    }),
)]
pub struct Pipeline;

/// LCM of the decimation factors.  The scheduling quantum, in input samples.
const INPUT_QUANTUM: u64 = 1 << LEVELS; // 8
/// Level-0 outputs per quantum.  `count[0]` is always a multiple of this.
const OUTPUT_QUANTUM: u64 = INPUT_QUANTUM / 2; // 4
/// Deepest lookahead demanded by any level.  Level 2 binds.
const MAX_RADIUS: u64 = RADIUS[LEVELS - 1] as u64; // 48

// The levels saturate their output rings together: 4096 >> L == LEVEL_SAMPLES[L].

pub struct Downsample<const CHANNELS: usize = 2> {
    allocation: MappedAllocation<u8>,

    /// Bases are handed to the device every dispatch via push constants
    static_base: DeviceAddress,
    output_base: DeviceAddress,

    output_ring_offsets: [[u32; CHANNELS]; LEVELS],
    pipeline: ComputePipeline<Pipeline>,

    timeline: TimelineSemaphore,

    /// For now, we might get away with a very specific rate of advance.  Given that downsample
    /// frames never tick downstream except on certain inputs, this might actually work.
    next_center: u64,
}

const LEVEL_SAMPLES: [u32; LEVELS] = [4096, 2048, 1024];

impl<const CHANNELS: usize> Downsample<CHANNELS> {
    pub fn new(device: &Device, ring_layout: RingLayout<CHANNELS>) -> Result<Self, MutateError> {
        // NOTE The big idea is no different for any of these pipelines.  We plan the layout, grab
        // the allocation, initialize it, and set up our local state shadows we'll need for later
        // dispatch.

        // plan static
        let mut c = plan::Cursor::default();
        let config_offset = c.push::<Config>(1);
        let input_views_offset = c.push::<RingView>(CHANNELS as u32);
        let output_views_offset = c.push::<RingView>((LEVELS * CHANNELS) as u32);
        let level_configs_offset = c.push::<LevelConfig>(LEVELS as u32);

        let mut weights_offsets = [0u32; LEVELS];
        for level in 0..LEVELS {
            weights_offsets[level] = c.push::<f32>(RADIUS[level] + 1);
        }
        let static_bytes = c.align_to(256);

        // plan outputs
        let mut c = plan::Cursor::default();
        let mut output_ring_offsets = [[0u32; CHANNELS]; LEVELS];
        for level in 0..LEVELS {
            for ch in 0..CHANNELS {
                output_ring_offsets[level][ch] = c.push::<f32>(LEVEL_SAMPLES[level]);
            }
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
                input_base: ring_layout.base_address.into(),
                output_base,
                input_views_offset,
                output_views_offset,
                output_views_stride: CHANNELS as u32,
                level_configs_offset,
            },
        );

        // Folded, outward-in: index 0 is the outermost tap, index `radius` is the center.
        // Symmetry is asserted above, so the leading half is the whole story.
        plan::put_slice(
            stat,
            weights_offsets[0],
            &weights::DOWN_TWO[..=RADIUS[0] as usize],
        );
        plan::put_slice(
            stat,
            weights_offsets[1],
            &weights::DOWN_FOUR[..=RADIUS[1] as usize],
        );
        plan::put_slice(
            stat,
            weights_offsets[2],
            &weights::DOWN_EIGHT[..=RADIUS[2] as usize],
        );

        // TODO
        // Host state tracking data...

        // Input views: upstream ring, per channel.
        for ch in 0..CHANNELS {
            plan::put(
                stat,
                input_views_offset + ch as u32 * size_of::<RingView>() as u32,
                RingView {
                    offset: ring_layout.channel_offsets[ch],
                    mask: ring_layout.sample_count - 1,
                },
            );
        }

        // Output views: [level][channel], stride == CHANNELS.
        for level in 0..LEVELS {
            for ch in 0..CHANNELS {
                let i = (level * CHANNELS + ch) as u32;
                plan::put(
                    stat,
                    output_views_offset + i * size_of::<RingView>() as u32,
                    RingView {
                        offset: output_ring_offsets[level][ch],
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
                    radius: RADIUS[level],
                    decimation_log2: level as u32 + 1,
                    lanes_per_output_log2: level as u32,
                },
            );
        }

        Ok(Self {
            allocation,
            static_base,
            output_base,
            output_ring_offsets,
            timeline: device.make_timeline_semaphore()?,
            // NEXT Just...emit an alias already.
            pipeline: ComputePipeline::<Pipeline>::new(device)?,
            next_center: 0,
        })
    }

    // NOTE The dft.rs accepts a layout on new and stores it in the config.  This shader
    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        input: &DeviceRingView<CHANNELS>,
    ) -> Result<DownsampleDispatch<CHANNELS>, MutateError> {
        // Calculate the downsampler advances from the upstream write head.
        let count = self.ready_outputs(input.write_head);
        let input_mask = input.ring_layout.sample_count - 1;
        let constants = PushConstants {
            static_base: self.static_base,
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
            device.cmd_dispatch(**cb, groups_x, LEVELS as u32, CHANNELS as u32);
        }

        self.next_center += count as u64 * 2;
        debug_assert_eq!(self.next_center % INPUT_QUANTUM, 0);

        let ready = self.timeline.next_signal();
        let output = DownsampleOutput {
            base: self.output_base,
            ring_offsets: self.output_ring_offsets,
            next_center: self.next_center, // already advanced above
            retain_floor: self.retain_floor(),
            data_ready: ready.wait_value(),
        };

        Ok(DownsampleDispatch {
            output,
            ready,
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

    /// Returns on-device geometry for a single level of downsampling outputs.
    pub fn level_layout(&self, level: usize) -> RingLayout<CHANNELS> {
        RingLayout {
            base_address: self.output_base.raw(),
            channel_offsets: self.output_ring_offsets[level],
            sample_count: LEVEL_SAMPLES[level],
        }
    }
}
