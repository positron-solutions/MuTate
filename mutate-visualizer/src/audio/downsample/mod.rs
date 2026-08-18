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
//! High pitches are incidentally attenuated, providing a safety margin that makes the output signal
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
//! lanes per frame, more lanes for longer filters to level the load across warps.  For an audio
//! tick of 512 samples and two channels, there's about 80k MACs, barely enough to occupy all SMs
//! before the entire filter is already done.  The output sample rate goes down as fast as the
//! filter size increases, so this solution remains linear.
//!
//! Feel free to look for other techniques, but the main on-device constraint for FIR downsampling
//! seems to be that serial dependencies across warps are absolutely terrible, so going directly
//! from signal to output without any kind of cascading is almost always going to win.  The device
//! has enough lanes that a single audio server tick will struggle to make use of the full width
//! available anyway.
//!
//! The whole problem fits in L1, and there are barely enough outputs to issue on every warp as it
//! is, so hiding latency and leveling tails is probably best done with more workgroups rather than
//! LDS or pipelining outputs.
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
//! Output at the new sample rate Nyquist is only guaranteed to be *attenuated* from the source
//! signal or noise, but not attenuated as deeply as the true stop band, and would still be evident
//! within the usable *dynamic ranges* of the pass band (not the pitch ranges).  We don't use the
//! area right at the new Nyquist, so this is fine?  In theory, but maybe not practice.  We will
//! defer to the opinions of long-time DSP professionals through an open source development process.
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
//! A uniform group delay per octave was chosen just to be tidy.  21/41/81/161 taps give 5
//! **output** samples of relative delay at each successive level.

// MAYBE Publish a semaphore value to readback for the host to be able to decide how much input
// dispatched upon is now ready for producer reclaim.
// NEXT Write the LEVELS output ring offset data to a header so that the host side of sharing
// downstream is dynamic state only.  We will need to start doing slang libraries, but we absolutely
// should!
// NEXT Upstream bursts and downstream read floor contracts require one solution.  We should
// consider the oldest half of the output ring always ready for reclaim.  We should clamp the read
// onto the *newest* data on the input ring, adjusting our indexes and the consumed counts so that
// subsequent reads will properly jump ahead and skip our old data.  The audio graph dispatch cycle
// guarantees that this *effective double buffering* is good enough.  A follow-on plan for the audio
// import ring is to swallow such bursts using a triple-buffer with rotation so that, in the worst
// case, we see 1/3 of the ring size of fresh data when we come back around.  Downsampler will then
// fill half of its rings with fresh data and the audio graph will consume only that on the tick.
// Either the import or the downsampler should tell downtreams to flush the discontinuity by setting
// reclaim indexes.  Output rings will require a second triple-buffering fixup to account for
// consumers in flight.  Ring buffers man!

mod weights;

use ash::vk;
use mutate_lib::{self as utate, audio::import::DeviceAudioView, prelude::*};

use super::plan;

/// Must match `LANES` in audio/downsample.slang.
const LANES: u32 = 32;

/// Outputs a workgroup emits at `level`: 16 / 8 / 4 / 2.
const fn outputs_per_workgroup(level: usize) -> u32 {
    LANES / lanes_per_output(level)
}

const LEVELS: usize = 4;
pub const FILTERS: [weights::Lowpass; LEVELS] = [
    weights::DOWN_TWO,
    weights::DOWN_FOUR,
    weights::DOWN_EIGHT,
    weights::DOWN_SIXTEEN,
];

/// Input samples per output sample.  A stride in the sample domain, and a
/// property of the filter, not the level's position in the table.
const fn decimation_log2(level: usize) -> u32 {
    FILTERS[level].decimation.trailing_zeros()
}

/// Lanes cooperating on one output: 2 / 4 / 8 / 16.  A fanout in the lane
/// domain -- exactly the lanes needed to cover `radius` taps at
/// `PAIRS_PER_LANE` each.  Must stay <= LANES: above that the reduction
/// crosses the wave and `WaveShuffle` no longer reaches.
const fn lanes_per_output(level: usize) -> u32 {
    FILTERS[level].radius() / PAIRS_PER_LANE
}

const fn lanes_per_output_log2(level: usize) -> u32 {
    lanes_per_output(level).trailing_zeros()
}

/// Output rings are implicitly longer in time, enough to accommodate the longest spans any
/// downstream filter wants to read.  Low pitch filters will sometimes want to reach back in time,
/// and this storage is cheap, so we just keep it available.
// NOTE scaled back to input rate: 8192, 8192, 16384, 32768, almost 1s of 48kHz input.
const LEVEL_SAMPLES: [u32; LEVELS] = [4096, 2048, 2048, 2048];

/// How far back in *input* samples each level's ring reaches.  No longer uniform across levels
/// and not derivable from any other level -- downstream must read this rather than shift.
const fn level_span(level: usize) -> u64 {
    LEVEL_SAMPLES[level] as u64 * FILTERS[level].decimation as u64
}

/// Must match `PAIRS_PER_LANE` in audio/downsample.slang.
const PAIRS_PER_LANE: u32 = 5;

const _: () = {
    let mut level = 0;
    while level < LEVELS {
        // The fold divides exactly: every cooperating lane gets PAIRS_PER_LANE
        // pairs and the center lands on slice zero.
        assert!(FILTERS[level].radius() % PAIRS_PER_LANE == 0);
        // Fanout must be PoT for the mask/shift split and the butterfly.
        assert!(lanes_per_output(level).is_power_of_two());
        assert!(lanes_per_output(level) <= LANES);
        // Decimation must be PoT for the shift in write_head / first_output_slot.
        assert!(FILTERS[level].decimation.is_power_of_two());
        assert!(LEVEL_SAMPLES[level].is_power_of_two());
        level += 1;
    }
};

/// One level of downsampling.  A set of `Rings` and metadata about the decimation and cutoff
/// frequency.
#[derive(Clone, Copy, Debug)]
pub struct LevelOutput {
    /// Offset from the output base address.
    pub offset: u32,
    /// Output samples ever written at this level.
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
    /// Oldest input-domain sample still resident in this level's ring, relative to `write_head`
    /// scaled to input rate.  Reach-back budget for downstream filters.
    pub span_input_samples: u64,
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
}

impl DownsampleOutput {
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
    // NEXT make the upstream DeviceAudioImport accept an absolute head rather than relative
    // "consumed".
    /// Oldest input sample still needed.  Import may reclaim strictly below.
    pub retain_floor: u64,
    /// Floor delta.
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

    /// Next window center per level, in input samples.  Levels advance independently; each is a
    /// multiple of its own decimation, so `next_centers[l] >> decimation_log2(l)` is that level's
    /// output write head.
    next_centers: [u64; LEVELS],

    /// Upstream sample rate.
    sample_rate: u32,

    /// Tracking floor delta to report consumed.
    prev_retain_floor: u64,
}

impl Downsample {
    pub fn new(device: &Device, channels: u32, sample_rate: u32) -> Result<Self, MutateError> {
        // NOTE The big idea is no different for any of these pipelines.  We plan the layout, grab
        // the allocation, initialize it, and set up our local state shadows we'll need for later
        // dispatch.

        // plan static
        let mut c = plan::Cursor::default();
        let config_offset = c.push::<Config>(1);
        let output_views_offset = c.push::<RingView>(LEVELS as u32 * channels);
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

        let base = allocation.device_address(device)?;
        let static_base: DeviceAddress = base.into();
        // FIXME Output belongs in a separate allocation.  We can use device-only memory after the
        // transfer-initialization dance is set supported.
        let output_base: DeviceAddress = (base + static_bytes as u64).into();

        let stat = allocation.as_mut_slice();

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
                    decimation_log2: decimation_log2(level),
                    lanes_per_output_log2: lanes_per_output_log2(level),
                },
            );
            debug_assert!(lanes_per_output_log2(level) == decimation_log2(level));
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
            next_centers: std::array::from_fn(|l| FILTERS[l].radius() as u64),
            sample_rate,
            prev_retain_floor: 0,
        })
    }

    pub fn dispatch(
        &mut self,
        device: &ash::Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        input: &DeviceAudioView,
    ) -> Result<DownsampleDispatch, MutateError> {
        // Calculate the downsampler advances from the upstream write head.
        let counts: [u32; LEVELS] =
            std::array::from_fn(|l| self.ready_at(l, input.write_head, input.sample_count as u64));

        let input_mask = input.sample_count - 1;
        let constants = PushConstants {
            input_base: input.base_address,
            static_base: self.static_base,
            input_stride: (input.sample_count * 4).into(),
            input_mask: input_mask.into(),

            first_window_center: std::array::from_fn(|l| {
                ((self.next_centers[l] as u32) & input_mask).into()
            }),
            first_output_slot: std::array::from_fn(|l| {
                (((self.next_centers[l] >> decimation_log2(l)) as u32) & (LEVEL_SAMPLES[l] - 1))
                    .into()
            }),
            output_count: std::array::from_fn(|l| counts[l].into()),
        };
        self.pipeline.push(device, **cb, &constants);

        // NOTE Width covers the hungriest level.  Shallower levels over-provision no-op workgroups.
        // The upstream callback already early returns on zero-sized audio server ticks.
        let groups_x = (0..LEVELS)
            .map(|l| counts[l].div_ceil(outputs_per_workgroup(l)))
            .max()
            .unwrap()
            .max(1);

        unsafe {
            device.cmd_bind_pipeline(**cb, vk::PipelineBindPoint::COMPUTE, *self.pipeline);
            device.cmd_dispatch(**cb, groups_x, LEVELS as u32, self.output_channel_count);
        }

        for l in 0..LEVELS {
            self.next_centers[l] += counts[l] as u64 * FILTERS[l].decimation as u64;
        }

        let retain_floor = self.retain_floor();
        let consumed = (retain_floor - self.prev_retain_floor) as u32;
        self.prev_retain_floor = retain_floor;

        Ok(DownsampleDispatch {
            output: self.view(),
            retain_floor,
            consumed,
        })
    }

    pub fn destroy(self, device: &Device) {
        device.deletion_queue.push(self.allocation.buffer);
        device.deletion_queue.push(self.allocation.memory);
        device.deletion_queue.push(self.timeline.into_raw());
        self.pipeline.destroy(device);
    }

    /// Clamps maximum advance to last eligible center.
    ///
    /// An output centered at `c` reads `[c - radius, c + radius]`, so the newest eligible center
    /// is: `written - 1 - radius`, clamped before wrap or exhausting fresh input samples.
    fn ready_at(&self, level: usize, written: u64, input_samples: u64) -> u32 {
        let radius = FILTERS[level].radius() as u64;
        let decimation = FILTERS[level].decimation as u64;

        let Some(newest) = written.checked_sub(1 + radius) else {
            return 0;
        };
        let newest = newest & !(decimation - 1);
        let next = self.next_centers[level];
        if newest < next {
            return 0;
        }

        // NOTE During underruns, it would be preferred to consume the newest output and discard the
        // older output.  More details about double buffering at the mod doc trailer.
        let count = (newest - next) / decimation + 1;
        let by_output = LEVEL_SAMPLES[level] as u64 / 2;
        // Window for the last center reaches `next - radius` .. `next + (n-1)*dec + radius`.
        let by_input = (input_samples / 2).saturating_sub(2 * radius) / decimation;
        count.min(by_output).min(by_input) as u32
    }

    /// Oldest input sample this module may still read.  Import must not reclaim below this.
    pub fn retain_floor(&self) -> u64 {
        (0..LEVELS)
            .map(|l| self.next_centers[l].saturating_sub(FILTERS[l].radius() as u64))
            .min()
            .unwrap()
    }

    /// Return the state and layout information independent of dispatching.  Useful for initializing
    /// downstreams.
    pub fn view(&self) -> DownsampleOutput {
        let sample_rate = self.sample_rate as f32;
        DownsampleOutput {
            base_address: self.output_base,
            levels: std::array::from_fn(|l| {
                let f = &FILTERS[l];
                LevelOutput {
                    offset: self.output_level_offsets[l],
                    channel_count: self.output_channel_count,
                    sample_count: LEVEL_SAMPLES[l],
                    decimation: f.decimation,
                    cutoff: f.cutoff(sample_rate),
                    delay: f.radius(),
                    write_head: self.next_centers[l] >> decimation_log2(l),
                    span_input_samples: level_span(l),
                }
            }),
        }
    }
}
