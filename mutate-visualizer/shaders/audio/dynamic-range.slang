// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// inputs are the raw outputs of the DFTs.  Output is a gain that will achieve
// the intended dynamic range of the inputs. One per channel
struct FloatBuffer {
    float samples[];
};

// A three-component peak filter:
//
// - Immediate decay hunts for peaks
// - Primary hold-decay filters immediate decay between local peaks
// - Output hold-decay only decays toward primary after a longer filter delay
//
// All three are reset whenever exceeded.  If output hold-decay is chasing decay
// of the primary, its progress is re-interpreted on every turn, and it updates
// to a new smooth location on the curve.
//
// The desired output is a gain_normalization.  It is the first field of the
// output structure so that it's easy to de-reference.  Downstreams read the
// gain_normalization and multiply the input, achieving a relatively constant
// dynamic range of 0 to -70dB.
//
// NEXT pop filtering to prevent micro-peaks from causing 4s holds.  As the
// filter complexity grows, dynamic machine learning will be a better
// investment of effort 😉.
struct DynamicRangeFilter {
    float output_hold_remaining;
    float output_db;
    float output_decay_start_db; // starting point of decay
    float primary_hold_remaining;
    float primary_db;
    float primary_decay_start_db;
    float immediate_db;
    float immediate_decay_start_db;
};

// TODO output array type
//

[[vk::push_constant]]
cbuffer DynamicRangeCd {
    // raw dft output slab, indexed into by dft configs
    FloatBuffer* raw_dft;
    // DftConfiguration* dft_configs;
    // DftState* dft_states;
    DynamicRangeFilter* filters;
};

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Obtain DFT states for this channel
    //
    // Run once per DFT output step (dynamic range)
    // Use filter state to catch up with DFT state.  Downstreams use the gain
    // output when interpreting the data... or... we overwrite the data after
    // calculating the threshold.  Either way, the thresholds are independently
    // useful.
    // Left and right should use one merged threshold.
    // Shuffle or merge the dispatch or something
}
