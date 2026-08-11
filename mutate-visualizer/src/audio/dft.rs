//! # Discrete Fourier Transform
//!
//! Short-time windowed discrete Fourier transform providing complex-valued output rings for
//! parallel downstream spectral analysis.  This transform just provides raw input for pitch
//! reassignment and synchrosqueezing etc.
//!
//! - One independent DFT bank per channel
//! - Logarithmic bin spacing, independently calculated Goertzel projections per bin.
//! - Short-time overlapping windowed block outputs.
//! - Dolph-Chebyshev windows (special case of ultra-spherical)
//! - Downstream payloads
//!   + Raw complex outputs stored in ring buffers for multiple downstreams
//!   + Time-slice contiguous barrier generation for consumer dispatches.
//!   + Timeline semaphore to ensure latest published dispatch can be made visible.
//!   + Dispatch published semaphore values for wait-free read.
//!   + Control data publish is wait-free, best-effort for consuming on another clock.
//!
//! ## Implementation Tradeoffs
//!
//! - Dispatch level window configuration for low divergence.  This makes many values such as Q
//!   bin-dependent, which has to be ironed out downstream.  Windows with padding weights would
//!   still require different spacing, and so the warp was chosen as the quantum of dispatch and
//!   configuration.
//!
//! ## Data Structures
//!
//! - One `DftConfig` with values shared across channels
//! - One `ChannelConfig` per channel.
//! - `BINS` `BinConfig`s per DFT.
//! - `BINS` `BinState`s per channel.
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
//! - `DftConfig`
//! - `ChannelConfig`s
//! - `Window weights`
//! - `BinConfig`s
//!
//! **Dynamic states**:
//! - `ChannelState`s
//! - Per Channel:
//!   + `BinState`s
//!   + `WindowSum`
//!   + Output rings. One per channel.  Column-major
//!
//! ### DFT Output Format
//!
//! Each channel uses an independent output ring of `Complex` values.  The time axis advances by
//! column and the memory layout is column major, so barriers for a time slice of the ring cover a
//! contiguous region of memory or two for regions that straddle the end of the ring.
//!
//! ## Compute Pipeline Behavior
//!
//! - Each lane finds its channel and then bin using dispatch geometry.
//! - Each bin maintains [`OVERLAP_RATIO`] window complex accumulators that are loaded to registers
//!   once.
//! - Each input sample is first projected onto the Goertzel phasor once.
//! - Each window sum applies its next weight to the phasor and accumulates.
//! - Only one window sum can possibly retire on every input datum.  If retirement happens, a
//!   complex output is written to the ring.
//! - After the dispatch completes, the bin's dynamic state and window sums are written back to
//!   VRAM.

// NOTE Other DFTs can be made from this structure.  When building composites with different tick
// rates, aligning the tracking is the responsibility of the user and should use the tick rate
// differential to deduce write-head offsets.
// DEBT DMA vs transfer abstraction.  The control data should be written device visible and then
// copied / handed off into device-only memory.  We would initialize this lazily, only beginning
// dispatches when the upload of control data is available and black-holing (later shared consumers
// will need a concept of the reclaim owner) data to keep upstream ring slack available.  There's a
// lot of runtime spec hydration and downstream dependency concepts in there.
// NEXT separate static and dynamic data.  Use UBO buffer flags etc.
// NEXT We're not using any slang reflection.  This is the motivating problem for which we would
// even finish implementing the field checks 🤡.  Look at all of these buffer device addresses that
// have some coercion target!  This will require some newtype wrapping.  Probably the biggest time
// saver would be checking field orders.  99 times out of 10, if any type or alignment is wrong,
// it's because fields are missing, extra, or out of order.  That doesn't depend on fine-grained
// size and alignment accounting, so it's probably worth it to begin the JSON imports of reflection
// data and begin checks.
// DEBT reading bins contiguously in a single time slice is the common access pattern.  The data has
// never cared about column / row major, only output[bin][time].  Transpose the output ring so that
// we can get on row-major, larger row index = time forward sensing, the super standard that is
// incidentally the sometimes weirdly regarded choice of X,Y = (0,0) being top-left pixel.  The
// "column major" thing started because scrolling an FFT in time to make a spectrogram has a strong
// right to left user interface tendency.

use std::mem::MaybeUninit;

use ash::vk;
use mutate_lib::tree::TreeSum;
use mutate_lib::{self as utate, audio::import::RingLayout, prelude::*};
use num_traits::One;
use utate::dsp::{self, bank, window::WindowFunction};

use super::downsample;
use super::plan;
use num_complex::Complex32 as Complex;

// NEXT suppose this could be configurable, but add some compile-time checks to configuration
// declarations.  Slang constant agreement via proc macro would be very welcome.
/// Number of bins.  `WARP_SIZE` multiple enforcement saves us from padding or a masking check
/// within the shader.
pub const BINS: u32 = 2560 / 2;
const WARP_SIZE: u32 = 32;
const _: () = assert!(BINS % WARP_SIZE == 0, "grouping assumes exact groups");
pub const WINDOW_LENGTH: u32 = 512;
const _: () = assert!(
    WINDOW_LENGTH.is_power_of_two(),
    "window length must be PoT for wrap masking",
);

/// To maintain COLA in time, each input datum is seen by at least this many windows.  Faster window
/// ticks lead to slightly better COLA for observing transients.
// ♻️ Duplicated within the slang.
const OVERLAP_RATIO: u32 = 4;
const WINDOW_SPACING: u32 = WINDOW_LENGTH / OVERLAP_RATIO;
const _: () = assert!(
    WINDOW_LENGTH % OVERLAP_RATIO == 0,
    "window spacing must divide evenly",
);
const _: () = assert!(
    WINDOW_LENGTH == WINDOW_SPACING * OVERLAP_RATIO,
    "shader can only walk weights that are a multiple of spacing.",
);

pub const OUTPUT_COLUMNS: u32 = 128;
const _: () = assert!(
    OUTPUT_COLUMNS.is_power_of_two(),
    "output ring must be PoT for wrap masking",
);

// NOTE Example output buffer size: 2560 bins * 128 columns * 8 bytes per Complex output * 2
// channels ~= 5MB.  With a window length of 256 and overlap ratio of 2, we get about 375.0Hz
// updates.  At that rate, 6.25 columns fit within a single 60Hz frame.  That means the output ring
// has over the pipeline depth + triple buffering of slack.  It's about 10x over-provisioned,
// more-so on faster displays that hazard less data at a time and retire hazards faster.  The
// over-provision mainly provides buffer for tracking & slew.

// DEBT sample rate
/// 48kHz until we support rates like 44.1 etc
const SAMPLE_RATE: u32 = 6_000;
const OUTPUT_RATE_NUM: u32 = SAMPLE_RATE;
const OUTPUT_RATE_DEN: u32 = WINDOW_SPACING;

/// Geometry, location, and tracking information for downstream consumers.
// XXX Publish this post dispatch!
#[derive(Clone, Debug)]
pub struct DftOutput<const CHANNELS: usize = 2> {
    /// Output buffer addresses.
    pub channels: [DeviceAddress; CHANNELS],
    /// Height in bins.  Number of rows.
    pub ring_height: u32,
    /// Width of each ring in pixels.  Number of columns.
    pub ring_width: u32,
    /// Consumed logical read head of the upstream audio input.  May be used for tracking & slewing.
    pub read_head: u64,

    /// Timeline semaphore signaled on dispatch for latest quantum (witnessed by this structure
    /// during construction).
    pub data_ready: WaitValue,
    /// Logical write head of the downstream **after** `data_ready` signals.  Note: this is output
    /// rate, 240Hz.
    pub write_head: u64,

    /// `[0, samples_per_column)`.  Consumers timing the input against the output use
    /// `read_head - column_phase` as the instant of the last column boundary.
    pub column_phase: u32,
    pub samples_per_column: u32,
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
    push = push!(DftPushConstants {
        /// Coerces to `DftConfig` in slang.  Static data offsets use this base.
        static_base: DeviceAddress,
        /// Coerces to `ChannelState*` in slang.  All dynamic offsets are relative to this base.
        dynamic_base: DeviceAddress,

        /// Physical start of the input samples.
        input_beg: UInt,
        /// Physical end index of the input samples.
        input_count: UInt,

        /// Host pre-calculates end column and feeds it in.  Device counts down the difference.  Host
        /// and device *must* not disagree with
        output_column: UInt,
        /// The number of input samples that have been consumed but did not emit an output yet.
        column_phase: UInt,
    }),
)]
pub struct DftComputePipeline;

/// Used to initialize static on-device config for a Dft channel.
#[repr(C)]
struct DftConfig {
    /// Number of bins.  Also interpreted as output image height.
    bin_count: u32,
    /// Input rings size informs wrapping logic.  Must be PoT.
    input_size: u32,
    /// Length of lines in the output image.  Each bin will write to a row of the
    /// output image, which is stored in column-major format for temporal
    /// barriers to cover contiguous regions.
    output_image_width: u32,
    /// Individual channel configurations.  Offset from `static_base`.  Coerce to
    /// `ChannelConfig*`. Index by tid.x.
    channel_configs_offset: u32,
    // NOTE channel count omitted.  Host controls maximum channel count with `tid.x`.
    /// Coerce to `float*` in slang.  Offset from `static_base`.
    window_weights_offset: u32,
    /// Input sample offset of each window start.  This *is* the output clock.  One column closes
    /// every `window_spacing` samples, on a sample instant.
    window_spacing: u32,
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
    /// Offset from `dynamic_base`.  Coerce to `Complex*`.  Column-major, `output_image_width`
    /// columns of `bin_count` rows.
    output_image_offset: u32,
}

/// Static configuration data for a single DFT bin.
#[repr(C)]
struct BinConfig {
    /// Bin phasor rotation per input sample.
    theta: Complex,
}

/// Bin dynamic states.  Window sums and accumulator state both live here.
#[repr(C)]
struct BinState {
    /// Goertzel filter's state variable.
    phasor: Complex,
    /// Coerces to `Complex*`.  Offset from `dynamic_base`
    window_sums_offset: u32,
}

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

    /// Samples accumulated into the open hop.  In `[0, WINDOW_SPACING)`.  Invariant is:
    /// `read_head == columns_written * WINDOW_SPACING + column_phase`
    column_phase: u32,
    // XXX I think we can derive this
    /// Physical index of the open column.
    output_column: u32,
    /// Monotonic count of *closed* columns.  This is `DftOutput::write_head`.
    columns_written: u64,
    /// Logical sample index we have consumed through.
    read_head: u64,

    /// `input_size - 1`, cached for wrapping `input_beg`.
    input_mask: u32,
    /// `BINS.div_ceil(32)`
    groups_y: u32,

    pipeline: ComputePipeline<DftComputePipeline>,
}

impl<const CHANNELS: usize> Dft<CHANNELS> {
    pub fn new(device: &Device, ring_layout: RingLayout<CHANNELS>) -> Result<Self, MutateError> {
        // XXX move this upstream and create a call-site compile time check
        let input_size = ring_layout.sample_count;
        debug_assert!(input_size.is_power_of_two());

        // Returns log-spaced bins.
        let bins = bank::bins(dsp::MIN_FREQ_CHEAP_DRIVERS, 3_000.0, BINS as usize);

        // Laying out the memory is essentially just walking the sizes and offsets of everything we
        // want to put in it, remembering those values to later populate the memory, and allocating
        // the total sizes.

        // Walk the static section to find offsets and size.
        let mut c = plan::Cursor::default();
        let dft_config_offset = c.push::<DftConfig>(1);
        let channel_configs_offset = c.push::<ChannelConfig>(CHANNELS as u32);
        let window_weights_offset = c.push::<f32>(WINDOW_LENGTH as u32);
        let bin_configs_offset = c.push::<BinConfig>(BINS);

        // NOTE alignment for within the buffer
        let static_bytes = c.align_to(256);

        // Walk the dynamic section to find offsets and size.
        let mut c = plan::Cursor::default();
        let channel_states_offset = c.push::<ChannelState>(CHANNELS as u32);

        // State for each channel is written contiguously.
        let mut bin_states = [0u32; CHANNELS];
        let mut window_sums = [0u32; CHANNELS];
        let mut output_rings = [0u32; CHANNELS];
        for i in 0..CHANNELS {
            bin_states[i] = c.push::<BinState>(BINS);
            window_sums[i] = c.push::<Complex>(BINS * OVERLAP_RATIO);
            output_rings[i] = c.push::<Complex>(BINS * OUTPUT_COLUMNS);
        }
        let dynamic_bytes = c.len();

        // XXX unknown alignment of this allocation
        let mut allocation = utate::vulkan::resource::buffer::MappedAllocation::<u8>::new(
            device,
            (static_bytes + dynamic_bytes) as usize,
        )?;
        let (stat, dynam) = allocation
            .as_mut_slice()
            .split_at_mut(static_bytes as usize);

        plan::put(
            stat,
            dft_config_offset,
            DftConfig {
                bin_count: BINS,
                input_size: ring_layout.sample_count,
                output_image_width: OUTPUT_COLUMNS,
                channel_configs_offset,

                window_weights_offset,
                window_spacing: WINDOW_LENGTH / OVERLAP_RATIO,
            },
        );

        for i in 0..CHANNELS {
            plan::put(
                stat,
                channel_configs_offset + i as u32 * size_of::<ChannelConfig>() as u32,
                ChannelConfig {
                    input_ring_base: (ring_layout.base_address
                        + ring_layout.channel_offsets[i] as u64)
                        .into(),
                    bin_configs_offset: bin_configs_offset.into(),
                },
            );
        }

        let window_function = WindowFunction::DolphChebyshev {
            attenuation_db: 120.0,
        };
        // let window_function = WindowFunction::Hamming;
        // let window_function = WindowFunction::BoxCar;
        let weights = window_function.make_window_32(WINDOW_LENGTH as usize);
        plan::put_slice(stat, window_weights_offset, &weights);

        for (i, bin) in bins.iter().enumerate() {
            plan::put(
                stat,
                bin_configs_offset + i as u32 * size_of::<BinConfig>() as u32,
                BinConfig {
                    theta: {
                        // sin = imag, cos = real.  This ordering is intentional!
                        let (imag, real) =
                            (std::f64::consts::TAU * (bin.center / SAMPLE_RATE as f64)).sin_cos();
                        Complex::new(real as f32, imag as f32)
                    },
                },
            );
        }

        for i in 0..CHANNELS {
            plan::put(
                dynam,
                channel_states_offset + i as u32 * size_of::<ChannelState>() as u32,
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
                        phasor: Complex::one(),
                        window_sums_offset: window_sums[i]
                            + bin * OVERLAP_RATIO * size_of::<Complex>() as u32,
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
            column_phase: 0,
            output_column: 0,
            columns_written: 0,
            read_head: 0,
            input_mask: input_size - 1,
            groups_y: BINS / WARP_SIZE, // NOTE enforced by const check
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
        downsample: &downsample::DownsampleOutput<CHANNELS>,
    ) -> Result<DftDispatch<CHANNELS>, MutateError> {
        let channels = downsample.level(2);
        let read_head = channels.write_head;
        let count = (read_head - self.read_head) as u32;

        // Push constants describe the clock as of the sample *before* input_beg.
        let constants = DftPushConstants {
            static_base: self.static_base,
            dynamic_base: self.dynamic_base,
            input_beg: (self.read_head as u32 & self.input_mask).into(),
            input_count: count.into(),
            output_column: self.output_column.into(),
            column_phase: self.column_phase.into(),
        };
        self.pipeline.push(device, **cb, &constants);

        unsafe {
            device.cmd_bind_pipeline(**cb, vk::PipelineBindPoint::COMPUTE, *self.pipeline);
            device.cmd_dispatch(**cb, CHANNELS as u32, self.groups_y, 1);
        }

        // Advance device state mirrors.
        let total = self.column_phase + count;
        let closed = total / WINDOW_SPACING;
        let first_closed = self.output_column;

        self.column_phase = total % WINDOW_SPACING;
        self.output_column = (self.output_column + closed) & (OUTPUT_COLUMNS - 1);
        self.columns_written += closed as u64;
        self.read_head += count as u64;

        debug_assert_eq!(
            self.read_head,
            self.columns_written * WINDOW_SPACING as u64 + self.column_phase as u64,
        );
        debug_assert_eq!(
            self.output_column as u64,
            self.columns_written & (OUTPUT_COLUMNS as u64 - 1),
        );

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
            column_phase: self.column_phase,
            samples_per_column: WINDOW_SPACING,
        };

        Ok(DftDispatch {
            consumed: count,
            ready,
            output,
        })
    }

    pub fn destroy(self, device: &Device) {
        device.deletion_queue.push(self.timeline.into_raw());
        self.allocation.destroy(device);
        self.pipeline.destroy(device);
    }
}
