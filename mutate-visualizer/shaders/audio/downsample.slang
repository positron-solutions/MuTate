// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// See the downsample.rs for more details.  Field descriptions in the Rust code
// are authoritative.

// LIES ⚠️ Same static/dynamic sub-allocator fiction as dft.slang.  Offsets are
// host-aligned for their target type and add directly to a device address.
// NOTE 🤖 Human designed and then re-generated.  The scheme takes advantage of
// a lot of symmetry.  The lane-filter geometry was intentionally designed to
// basically let most devices complete the downsample in a few cycles without
// any tails.  Lanes have approximately level work per output sample.  The log
// index coincidences were found by the LLM and left to be exploited.  Will
// clean up the comments later.

// Symmetric FIR, folded: a filter of L taps costs (L+1)/2 multiply-adds.  Tap
// counts are chosen so that cost, divided across the lanes cooperating on one
// output, is the same at every level:
//
// level  decimation  taps  pairs  lanes/output  pairs/lane  outputs/warp
// -----  ----------  ----  -----  ------------  ----------  ------------
//   0        2x       21     10        1            10           32
//   1        4x       41     20        2            10           16
//   2        8x       81     40        4            10            8
//   3       16x      161     80        8            10            4
//
// Every lane in the dispatch runs the same 10-iteration loop, then cooperating
// lanes reduce to a single lane, which folds in the center tap.
//
// A workgroup is WARPS * LANES = 256 threads and emits 256 / lanes_per_output
// outputs -- 256, 128, 64, or 32 -- which is exactly the yield of 512 input
// samples at that level.  So workgroup g covers input span [g*512, (g+1)*512)
// at every level, and one rectangular grid serves all four.
//
// NOTE The 16x downsample workgroup has so few outputs that it is instead
// beginning to use masking heavily.  Fine at this scale, but a bit sloppy

static const uint LANES = 32;
static const uint WARPS = 8;
static const uint LEVELS = 4;

/// Folded multiply-adds per lane, excluding the center tap.
static const uint PAIRS_PER_LANE = 10;

struct Config {
    uint64_t output_base;
    /// -> RingView[LEVELS * channels].  Indexed by
    /// `level * output_views_stride + channel`.
    uint output_views_offset;
    /// Major-axis stride of the output view table.  Host-supplied so the shader
    /// never needs to know the channel count.
    uint output_views_stride;

    /// -> LevelConfig[LEVELS].
    uint level_configs_offset;
}

/// Output ring and wrapping mask.
struct RingView {
    /// Offset from input device address.  Coerce to `float*`.
    uint offset;
    /// Index wrapping mask
    uint mask;
}

/// Pure filter description.  Nothing here depends on allocation or scheduling.
struct LevelConfig {
    /// -> float[radius + 1], outward-in: weights[0] is the outermost tap pair,
    /// weights[radius] is the unpaired center.
    uint weights_offset;

    /// Half-width of the filter support, in input samples.  Also the index of
    /// the center weight, and the fold count minus one.  10 / 20 / 40 / 80.
    uint radius;

    /// Input samples between consecutive outputs: 2 / 4 / 8 / 16.  A stride in
    /// the sample domain.
    uint decimation_log2;

    /// Lanes cooperating on one output: 1 / 2 / 4 / 8.  A fanout in the lane
    /// domain.  Equal to decimation/2 today, but they are different kinds of
    /// quantity and the shader keeps them apart.
    uint lanes_per_output_log2;
}

struct PushConstants {
    uint64_t input_base;
    uint64_t static_base;

    /// Bytes between consecutive channels in the input ring.
    uint input_stride;
    /// Index wrapping mask for the input ring, in samples.
    uint input_mask;

    /// Physical index, already reduced against the input view's mask, of the
    /// window center for this dispatch's first output at each level.
    uint first_window_center[LEVELS];

    /// Physical index, already reduced against that level's output view mask,
    /// of this dispatch's first output.
    uint first_output_slot[LEVELS];

    /// How many outputs this dispatch produces at each level.  Bounds the
    /// partial trailing workgroup.
    uint output_count[LEVELS];
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

// dispatch = (ceil(output_count[0] / 256), LEVELS, channels)
[WaveSize(LANES)]
[numthreads(LANES, WARPS, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID) {
    let workgroup = group_id.x;
    let level     = group_id.y;
    let channel   = group_id.z;

    let warp = thread_id.y;
    let lane = thread_id.x;

    Config config = *(Ptr<Config>)pc.static_base;
    LevelConfig level_config =
        ((Ptr<LevelConfig>)(pc.static_base + config.level_configs_offset))[level];

    Ptr<float> input_ring =
        (Ptr<float>)(pc.input_base + (uint64_t)(channel * pc.input_stride));

    RingView output_view =
        ((Ptr<RingView>)(pc.static_base + config.output_views_offset))
            [level * config.output_views_stride + channel];

    Ptr<float> output_ring = (Ptr<float>)(config.output_base + (uint64_t)output_view.offset);
    Ptr<float> weights = (Ptr<float>)(pc.static_base + level_config.weights_offset);

    let radius     = level_config.radius;
    let decimation = 1u << level_config.decimation_log2;

    // Split the lane id into (which output, which slice of that output's taps).
    let fanout_log2      = level_config.lanes_per_output_log2;
    let lanes_per_output = 1u << fanout_log2;
    let tap_slice        = lane & (lanes_per_output - 1);
    let output_slot      = lane >> fanout_log2;

    // Concatenate (workgroup, warp, output_slot) into this dispatch's output run.
    let outputs_per_warp      = LANES >> fanout_log2;
    let outputs_per_workgroup = WARPS * outputs_per_warp;
    let output_index = workgroup * outputs_per_workgroup
                     + warp * outputs_per_warp
                     + output_slot;

    // NOTE this check matches the output guard.
    if (output_index >= pc.output_count[level]) return;

    // Physical index of the sample at the center of this output's window.
    // Underflows below on `- radius + tap` are intentional: two's-complement
    // wrap composed with a power-of-two mask is the ring wrap.
    let window_center = pc.first_window_center[level] + output_index * decimation;

    // Ten unconditional folded pairs.  Each lane walks its own arithmetic
    // progression of tap indexes, striding by the fanout so the outermost (and
    // smallest) weights land in every accumulator first.
    float sum = 0.0f;
    [ForceUnroll]
    for (uint fold = 0; fold < PAIRS_PER_LANE; ++fold) {
        let tap = tap_slice + fold * lanes_per_output;
        let lower = (window_center - radius + tap) & pc.input_mask;
        let upper = (window_center + radius - tap) & pc.input_mask;
        sum = mad(weights[tap], input_ring[lower] + input_ring[upper], sum);
    }

    // The unpaired center tap.  `radius` is 10 / 20 / 40 / 80, all divisible by
    // the fanout, so it always falls to slice zero -- the same lane the
    // reduction below converges on, and the same lane that stores.
    if (tap_slice == 0) {
        sum = mad(weights[radius], input_ring[window_center & pc.input_mask], sum);
    }

    // Collapse the cooperating lanes onto slice zero.  Zero-trip at 2x.
    let lane_index = WaveGetLaneIndex();
    for (uint step = 1; step < lanes_per_output; step <<= 1) {
        sum += WaveShuffle(sum, lane_index ^ step);
    }

    if (tap_slice == 0) {
        let destination = (pc.first_output_slot[level] + output_index) & output_view.mask;
        output_ring[destination] = sum;
    }
}
