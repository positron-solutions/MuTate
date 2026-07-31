//! # Discrete Fourier Transform
//!
//! A filter bank of many windowed Goertzel filters.
//!
//! - One independent DFT bank per channel
//! - Dolph-Chebyshev window (special case of ultra-spherical)
//! - Fixed quantum size, for direct use in audio callback without tracking
//! - 240Hz output ring buffer
//! - Downstream payloads
//!   + read-scoped barrier generation
//!   + timeline semaphore to ensure published dispatch has arrived
//!   + publish output control data at a single address
//! - ISO226 leveling for a 70phon reference curve, pre-applied to window weights
//!
//! - Relatively constant Q (musical octaves have constant visual spacing)
//!
//! ## Important Coherence Relations
//!
//! - Window overlap ratio and window length set the window step size.
//! - Window step size determines window sum tick rate.
//! - Output rate, fixed for all bins, is a re-sampling of each bin's window tick rate.
//! - Bin count is equal to output rows.
//!
//! Quality ratios are calculated to produce at least overlapping bandwidths to avoid pitches
//! falling through all bins.  Quality ratio determines the window length.  Window length,
//! bandwidth, noise suppression, and time resolution are all tradeoffs.
//!
//! ### Key Implementation Drivers
//!
//! - Window sum ticks can be a lot slower than the output rate (the 20Hz bin has a long window but
//!   output is 240Hz), requiring interpolation for updates to write smooth data, but introducing an
//!   interpolation phase delay 💀.
//! - Window sum ticks can be a lot faster than the output rate, requiring accumulation before
//!   writing outputs.
//! - In both the fast and slow case, each fraction of each output maps to a fraction of the output
//!   without overlap, in the same order.  A single accumulator can do the job in both cases.
//!
//! Window retirement may happen on any audio datum, but only one window is going to retire next, so
//! we track it and do the retirement check at one point in the loop, keeping divergence low.
//!
//! ## Data Structures
//!
//! - One `DftConfig` with values shared across channels
//! - One `ChannelConfig` and `ChannelState` per channel
//! - `BINS` bins per channel, accessed via `ChannelConfig` and `ChannelState`.
//! - One `BinConfig` and `BinState` per bin
//!
//! ## Memory Layout
//!
//! ⚠️ The structures are defined as if intended for mutable and immutable sub-allocations, but in
//! the interest of time, one big mutable allocation is used instead.
//!
//! Sized data such as the `BinConfig` is stored in a header.  Bin-dependent data such as window
//! weights follow.
//!
//! **Static data**:
//! - `DftConfig`s
//! - `BinConfig`s
//! - Window weights
//!
//! **Dynamic states**:
//! - DFT Output ring, column-major
//! - `BinState`s
//!
//! ### DFT Output Format
//!
//! Each channel uses an independent output ring.  The time axis advances by column and the memory
//! layout is column major, so barriers for a time slice of the ring cover a contiguous region of
//! memory or two for regions that straddle the end of the ring.
//!
//! Output is stored as an `f32` magnitude of the window sum to enable dynamic range interpretation
//! and more flexible downstream use, such as HDR.  Phase information is discarded, but may be
//! re-introduced if short-time phase rotation can be used as a pitch-resolution signal to gate
//! low-pitch bin activation (it can).
//!
//! ## Compute Pipeline Behavior
//!
//! - Each lane finds its channel and then bin using dispatch geometry, masking itself if out of
//!   the range.
//! - Each bin maintains [`OVERLAP_RATIO`] windows that are loaded to registers once.
//! - Each bin contains an output accumulator, again loaded to registers and updated for the timing
//!   re-anchor.
//! - Each input sample is first projected onto the Goertzel phasor once.
//! - Each window sum applies its next weight and accumulates.
//! - Only one window can possibly retire on every input datum.  If retirement happens, the output
//!   accumulator is updated, window sum reset, and retirement index rotated.
//! - When input time advances beyond the output tick, each output accumulator is partially drained
//!   for the consumed time segment into the output ring.
//! - After the dispatch completes, persistent states are written back to the bin's dynamic state,
//!   including the partial output residing in the accumulator.

// NEXT see discussion about perfecting the spectrogram, but the biggest, most obvious improvements
// will be dampening far off-bin pitches using pre-condition and using, narrow-dynamic range
// companion DFTs to create a very high time-resolution bank that can sharpen a slower,
// pitch-precise DFT.  This will require cross-reference training that is not available yet.
// DEBT DMA vs transfer abstraction.  The control data should be written device visible and then
// copied / handed off into device-only memory.  We would initialize this lazily, only beginning
// dispatches when the upload of control data is available and black-holing (later shared consumers
// will need a concept of the reclaim owner) data to keep upstream ring slack available.  There's a
// lot of runtime spec hydration and downstream dependency concepts in there.
// NEXT separate static and dynamic data.  Use UBO buffer flags etc.
// NEXT We're not using any slang introspection.  This is the motivating problem for which we would
// even finish implementing the field checks 🤡.  Look at all of these buffer device addresses that
// have some coercion target!  This will require some newtype wrapping.  Probably the biggest time
// saver would be checking field orders.  99 times out of 10, if any type or alignment is wrong,
// it's because fields are missing, extra, or out of order.  That doesn't depend on fine-grained
// size and alignment accounting, so it's probably worth it to begin the JSON imports of reflection
// data and begin checks.

mod plan;

use std::mem::MaybeUninit;

use ash::vk;
use mutate_lib::{self as utate, audio::import::RingLayout, prelude::*};
use utate::dsp::{self, bank, window::WindowFunction};

/// Number of bins.  Per channel.  Usually sampled in high resolution and re-sampled as needed.
const BINS: u32 = 2560 / 8;
/// To maintain COLA in time, each input datum is seen by at least this many windows.  Faster window
/// ticks lead to slightly better COLA for observing peaks and more finely spaced window sum ticks
/// on low pitch bins.  This product does directly scale the major axis of work for each lane!
// ♻️ Duplicated within the slang.
const OVERLAP_RATIO: u32 = 16;
const OUTPUT_COLUMNS: u32 = 128; // About 0.5s at 240Hz

// NOTE Example output buffer size: 2560 bins * 128 columns * 4 bytes per F32 output * 2 channels ~=
// 2.5MB.  At 60Hz, the 240Hz ring is using about 4 columns per frame.  Producer lapping is roughly
// impossible and the ring has extremely generous slack, about 10x over-provisioned, more-so on
// faster displays that hazard less data at a time and retire hazards faster.  The over-provision
// mainly provides buffer for tracking & slew.

/// Geometry, location, and tracking information for downstream consumers.
// XXX Create and publish this post dispatch!
#[derive(Clone, Debug)]
pub struct DftOutput<const CHANNELS: usize = 2> {
    /// Output buffer addresses.
    pub channels: [DeviceAddress; CHANNELS],
    /// Height in bins.  Number of rows.
    pub ring_height: u32,
    /// Width of each ring in pixels.  Number of columns.
    pub ring_width: u32,

    /// Timeline semaphore signaled on dispatch for latest quantum (witnessed by this structure
    /// during construction).
    pub data_ready: WaitValue,
    /// Consumed logical read head of the upstream audio input.  May be used for tracking & slewing.
    pub read_head: u64,
    /// Logical write head of the downstream **after** `data_ready` signals.  Note: this is output
    /// rate, 240Hz.
    pub write_head: u64,
    // XXX I don't like this name
    /// Sub-datum phase of the write head against the read head.  Use to time the input against the output.
    pub column_ticks_beg: u32,

    /// The rational data rate relation between input and output
    pub ticks_per_sample: u32,
    pub ticks_per_column: u32,
}

/// Result structure for dispatches.  Audio callback consumes and publishes `output` for downstream
/// consumers.
pub struct DftDispatch<const CHANNELS: usize = 2> {
    /// Input samples consumed.  Caller reclaims ring slack with this.
    pub consumed: u32,
    /// Signal when this dispatch's output is ready for consumers.
    pub ready: SignalIntent,
    /// Publish *after* a successful submit.
    pub output: DftOutput<CHANNELS>,
}

#[compute_pipeline(
    compute = stage!("audio/dft", Compute, c"main"),
    // XXX Was not allowed to write a doc comment in this position.
    // Each channel reads from a segment of the input ring, from start to end.  Thanks to PoT ring
    // width, straddled values are fine, but audio server quantums usually will not straddle the
    // device input rings.
    push = push!(DftPushConstants {
        /// Coerces to `DftConfig*` in slang.  Static data offsets use this base.
        static_base: DeviceAddress,
        /// Coerces to `ChannelState*` in slang.  All dynamic offsets are relative to this base.
        dynamic_base: DeviceAddress,

        /// Physical start of the input samples.
        input_beg: UInt,
        /// Physical end index of the input samples.
        input_end: UInt,

        /// Host calculated starting offset, the phase between the output and input clock.  Wish I can
        /// find a cleaner way to explain it.  In a hurry to get to the debugging.
        column_ticks_beg: UInt,
        /// Host pre-calculates end column and feeds it in.  Device counts down the difference.  Host
        /// and device *must* not disagree with
        output_column: UInt,
    }),
)]
pub struct DftComputePipeline;

/// Used to initialize static on-device config for a Dft channel.
#[repr(C)]
struct DftConfig {
    /// Number of bins.  Also interpreted as output image height.
    bin_count: u32,
    /// Input rings size informs wrapping logic.  Should be PoT.
    input_size: u32,
    /// Length of lines in the output image.  Each bin will write to a row of the
    /// output image, which is stored in column-major format for temporal
    /// barriers to cover contiguous regions.
    output_image_width: u32,
    /// Individual channel configurations.  Offset from `static_base`.  Coerce to
    /// `ChannelConfig*`. Index by tid.x.
    channel_configs_offset: u32,
    // NOTE count omitted.  Host controls maximum channel count with `tid.x`.
    /// Output grid as an exact rational on the input grid.  One tick is 1/`ticks_per_column` of an
    /// output column; one input sample advances the output position by `ticks_per_sample` ticks.
    /// Host reduces the pair.  48 kHz in, 240 Hz out: 200 and 1.  Boundaries land on sample
    /// instants.  44.1 kHz in, 240 Hz out: 735 and 4.  Boundaries land on quarter samples.
    ticks_per_sample: UInt,
    ticks_per_column: UInt,
}

#[repr(C)]
struct ChannelConfig {
    /// Input device address for this channel.  Pointer rather than offset
    /// because it comes from the upstream, not our base.
    input_ring_base: DeviceAddress,

    /// Individual bin configurations.  Offset from `static_base`.  Coerce to
    /// BinConfig* Index into by tid.y
    bin_configs_offset: u32,
}

/// Per-channel dynamic roots.  `dynamic_base` coerces to `ChannelState*`; index by tid.x.
#[repr(C)]
struct ChannelState {
    /// Offset from `dynamic_base`.  Coerce to `BinState*`.  Index by tid.y.
    bin_states_offset: u32,
    /// Offset from `dynamic_base`.  Coerce to `float*`.  Column-major, `output_image_width`
    /// columns of `bin_count` rows.
    output_image_offset: u32,
}

#[repr(C)]
struct Complex {
    real: f32,
    imag: f32,
}

/// Static configuration data for a single DFT bin.
#[repr(C)]
struct BinConfig {
    /// Bin phasor rotation per input.
    theta: Complex,

    /// Input sample offset of each window start.
    window_spacing: u32,

    /// Coerce to `float*` in slang.  Offset from `static_base`.
    window_weights_offset: u32,
    /// Number of window weights.
    window_weights_count: u32,
}

/// Bin dynamic states.  Window sums and accumulator state both live here.
#[repr(C)]
struct BinState {
    /// Goertzel filter's state variable.
    phasor: Complex,

    /// Coerces to `Complex*`.  Offset from `dynamic_base`
    window_sums_offset: u32,
    /// Window slot that retires next.
    next_retire: u32,
    // Samples elapsed since the retirement that produced `ramp_end`.  In
    // [0, window_spacing).  Retirement fires when it reaches window_spacing.
    ramp_samples: u32,

    // Accumulator states.
    /// Endpoints of the ramp currently on display.  `ramp_beg` is window n-1's
    /// level, `ramp_end` is window n's, and the pair is consumed over [T_n,
    /// T_n+1) *following* the retirement that produced `ramp_end`.
    ramp_beg: f32,
    ramp_end: f32,

    /// Trapezoid area accumulated into the current output interval
    open_area: f32,
}

/// 48kHz until we support rates like 44.1 etc
const SAMPLE_RATE: u32 = 48_000;
/// 240 is our default output rate on all video input sources.
const OUTPUT_RATE: u32 = 240;

/// The host-side handle for the control data and dispatches.  Owns, dispatches, and publishes
/// consumer control data.
pub struct Dft<const CHANNELS: usize = 2> {
    timeline: TimelineSemaphore,
    /// All control data and all buffers live in a single allocation.
    // DEBT Hack manual sub-allocation per channel for now.
    allocation: MappedAllocation<u8>,

    /// Bases handed to the device every dispatch.
    static_base: DeviceAddress,
    dynamic_base: DeviceAddress,

    /// Byte offsets of each channel's output ring from `dynamic_base`.  Barrier ranges.
    output_ring_offsets: [u32; CHANNELS],

    // Host-owned mirror of the device's output clock.  The device reads these from push
    // constants and never writes them back, so divergence here is silent corruption.
    /// Ticks into the currently open column.  In `[0, ticks_per_column)`.
    column_ticks: u32,
    /// Physical index of the open column.
    output_column: u32,
    /// Monotonic count of *closed* columns.  This is `DftOutput::write_head`.
    columns_written: u64,
    /// Logical sample index we have consumed through.
    read_head: u64,

    ticks_per_sample: u32,
    ticks_per_column: u32,
    /// `input_size - 1`, cached for wrapping `input_beg`.
    input_mask: u32,
    /// `BINS.div_ceil(32)`
    groups_y: u32,

    pipeline: ComputePipeline<DftComputePipeline>,
}

impl<const CHANNELS: usize> Dft<CHANNELS> {
    pub fn new(device: &Device, ring_layout: RingLayout<CHANNELS>) -> Result<Self, MutateError> {
        let input_size = ring_layout.sample_count;
        debug_assert!(input_size.is_power_of_two());

        let bins = bank::bins(
            dsp::MIN_FREQ_CHEAP_DRIVERS,
            dsp::MAX_FREQ_OLD_PEOPLE,
            BINS as usize,
        );

        // Window geometry.  Length is rounded to a multiple of OVERLAP_RATIO for shader index walking.
        let window_lengths: Vec<u32> = bins
            .iter()
            .map(|b| {
                // Conservative q just to get to debugging!
                let q: f32 = 8.0;
                ((q * SAMPLE_RATE as f32 / (b.center as f32)).ceil() as u32)
                    .next_multiple_of(OVERLAP_RATIO)
            })
            .collect();

        // Walk the static section to find offsets.
        let mut c = plan::Cursor::default();
        let dft_config = c.push::<DftConfig>(1);
        let channel_configs = c.push::<ChannelConfig>(CHANNELS as u32);
        let bin_configs = c.push::<BinConfig>(BINS);
        let weights: Vec<u32> = window_lengths.iter().map(|&n| c.push::<f32>(n)).collect();
        let static_bytes = c.align_to(256);

        // Walk the dynamic section to find offsets.
        let mut c = plan::Cursor::default();
        let channel_states = c.push::<ChannelState>(CHANNELS as u32);
        let mut bin_states = [0u32; CHANNELS];
        let mut window_sums = [0u32; CHANNELS];
        let mut output_rings = [0u32; CHANNELS];
        // Channel-major: each channel's regions are contiguous, matching the per-channel ring
        // barrier and the eventual per-channel split.
        for i in 0..CHANNELS {
            bin_states[i] = c.push::<BinState>(BINS);
            window_sums[i] = c.push::<Complex>(BINS * OVERLAP_RATIO);
            output_rings[i] = c.push::<f32>(BINS * OUTPUT_COLUMNS);
        }
        let dynamic_bytes = c.len();

        let mut allocation = utate::vulkan::resource::buffer::MappedAllocation::<u8>::new(
            device,
            (static_bytes + dynamic_bytes) as usize,
        )?;
        let (stat, dynam) = allocation
            .as_mut_slice()
            .split_at_mut(static_bytes as usize);

        let g = plan::gcd(SAMPLE_RATE, OUTPUT_RATE);
        let ticks_per_sample = OUTPUT_RATE / g;
        let ticks_per_column = SAMPLE_RATE / g;
        assert!(
            ticks_per_sample < ticks_per_column,
            "device closes at most one column per sample"
        );

        plan::put(
            stat,
            dft_config,
            DftConfig {
                bin_count: BINS,
                input_size: ring_layout.sample_count,
                output_image_width: OUTPUT_COLUMNS,
                channel_configs_offset: channel_configs,
                ticks_per_column: (SAMPLE_RATE / g).into(),
                ticks_per_sample: (OUTPUT_RATE / g).into(),
            },
        );

        for i in 0..CHANNELS {
            plan::put(
                stat,
                channel_configs + i as u32 * size_of::<ChannelConfig>() as u32,
                ChannelConfig {
                    input_ring_base: (ring_layout.base_address
                        + ring_layout.channel_offsets[i] as u64)
                        .into(),
                    bin_configs_offset: bin_configs.into(),
                },
            );
        }

        for (bin, (b, &n)) in bins.iter().zip(&window_lengths).enumerate() {
            plan::put(
                stat,
                bin_configs + bin as u32 * size_of::<BinConfig>() as u32,
                BinConfig {
                    theta: {
                        // sin = imag, cos = real.  This is intentional!
                        let (imag, real) = (std::f32::consts::TAU * (b.center as f32)
                            / SAMPLE_RATE as f32)
                            .sin_cos();
                        Complex { real, imag }
                    },
                    window_spacing: n / OVERLAP_RATIO,
                    window_weights_offset: weights[bin],
                    window_weights_count: n,
                },
            );

            // let window_function = WindowFunction::DolphChebyshev {…} {
            //     attenuation_db: 70.0,
            // };
            let window_function = WindowFunction::Bartlett;
            let mut w = window_function.make_window_32(n as usize);
            // Pre-multiply the weights by an iso226 gain levelling curve so that bins come out
            // approximately perceptually flat in relative RMS/SPL
            let scale = b.iso226_gain as f32;
            w.iter_mut().for_each(|x| *x *= scale);

            plan::put_slice(stat, weights[bin], &w);
        }

        for i in 0..CHANNELS {
            plan::put(
                dynam,
                channel_states + i as u32 * size_of::<ChannelState>() as u32,
                ChannelState {
                    bin_states_offset: bin_states[i],
                    output_image_offset: output_rings[i],
                },
            );
            for bin in 0..BINS {
                plan::put(
                    dynam,
                    bin_states[i] + bin * size_of::<BinState>() as u32,
                    BinState {
                        // Goertzel state is a unit phasor, not zero.
                        phasor: Complex {
                            real: 1.0,
                            imag: 0.0,
                        },
                        window_sums_offset: window_sums[i]
                            + bin * OVERLAP_RATIO * size_of::<Complex>() as u32,
                        next_retire: 0,
                        ramp_samples: 0,
                        ramp_beg: 0.0,
                        ramp_end: 0.0,
                        open_area: 0.0,
                    },
                );
            }
        }

        // ROLL Flush likely redundant.  This will usually be fully coherent BAR/ReBAR window, but
        // that abstraction isn't ready yet.
        allocation.flush(device);

        let timeline = device.make_timeline_semaphore()?;

        let base = allocation.device_address(device)?;

        Ok(Self {
            timeline,
            static_base: base.into(),
            dynamic_base: (base + static_bytes as u64).into(),
            output_ring_offsets: output_rings,
            pipeline: ComputePipeline::<DftComputePipeline>::new(device)?,
            allocation,
            column_ticks: 0,
            output_column: 0,
            columns_written: 0,
            read_head: 0,
            ticks_per_sample,
            ticks_per_column,
            input_mask: input_size - 1,
            groups_y: BINS.div_ceil(32),
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
        state: &DeviceRingView<CHANNELS>,
    ) -> Result<DftDispatch, MutateError> {
        // XXX upstream a length check before handing over an empty dispatch
        let count = state.occupied_len();

        // XXX Don't forget WAR/RAW against the previous dispatch's BinState and the trailing column
        // write.  Same queue, same buffer, so a single global barrier is cheapest and correct.
        let pre = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        let pre_barriers = [pre];
        let dep = vk::DependencyInfo::default().memory_barriers(&pre_barriers);
        unsafe { device.cmd_pipeline_barrier2(**cb, &dep) };

        // Push constants describe the clock as of the sample *before* input_beg.
        // XXX These push constants are wrong...
        let constants = DftPushConstants {
            static_base: self.static_base,
            dynamic_base: self.dynamic_base,
            input_beg: (self.read_head as u32 & self.input_mask).into(),
            input_end: (((self.read_head + count as u64) as u32) & self.input_mask).into(),
            column_ticks_beg: self.column_ticks.into(),
            output_column: self.output_column.into(),
        };
        self.pipeline.push(device, **cb, &constants);

        unsafe {
            device.cmd_bind_pipeline(**cb, vk::PipelineBindPoint::COMPUTE, *self.pipeline);
            device.cmd_dispatch(**cb, CHANNELS as u32, self.groups_y, 1);
        }

        // Advance the mirror.  Must match the device's per-sample loop exactly.
        let total = self.column_ticks as u64 + count as u64 * self.ticks_per_sample as u64;
        let closed = (total / self.ticks_per_column as u64) as u32;
        let first_closed = self.output_column;

        self.column_ticks = (total % self.ticks_per_column as u64) as u32;
        self.output_column = (self.output_column + closed) % OUTPUT_COLUMNS;
        self.columns_written += closed as u64;
        self.read_head += count as u64;

        // Hand the freshly closed columns to the consumer.
        if closed > 0 {
            // XXX Barrier after
            // self.barrier_after(device, cb, first_closed, closed);
        }
        // Must-consume SignalIntent for the dispatch
        let ready = self.timeline.next_signal();
        let output = DftOutput {
            channels: std::array::from_fn(|i| {
                (self.dynamic_base.raw() + self.output_ring_offsets[i] as u64).into()
            }),
            ring_height: BINS,
            ring_width: OUTPUT_COLUMNS,
            data_ready: ready.wait_value(), // 👍 nice!
            read_head: self.read_head,
            write_head: self.columns_written,
            column_ticks_beg: self.column_ticks,
            ticks_per_sample: self.ticks_per_sample,
            ticks_per_column: self.ticks_per_column,
        };

        Ok(DftDispatch {
            consumed: count,
            ready,
            output,
        })
    }

    pub fn destroy(self, device: &Device) {
        self.allocation.destroy(device);
        self.pipeline.destroy(device);
    }
}
