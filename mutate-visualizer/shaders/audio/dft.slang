// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// LIES ⚠️ "static allocation" and "dynamic allocation" are not yet part of the
// implementation, but the comment language will be consistent if they *were*.
// Device addresses will be used even though they come from the same allocation.
// Pretending we have a sub-allocator is part of the same lie already.
// NOTE Host always provides offsets that are aligned for the target type and
// add directly to a device address to form the destination array address.
// Device address + u32 yields a valid pointer for the T described by the field.
// NEXT As the common math begins to show up, start supporting newtypes and
// streamline the device address coercion so that our Slang is cleaner.
// NOTE Rate representation.  Output columns advance at a rational multiple of
// the input rate.  Windows retire on the input grid.  Columns close on the
// output grid.  The device carries the position within the open column.
// NEXT a lot of re-sampling phase values can be compressed to u16, but with
// extra conversions and no benefit unless registers are spilling back to
// memory, which would be dominated by window sum complexity, not the phase
// tracking data.
// NEXT We can zero-pad some weights to change the hot loop geometry.
// LMAO positional constructor arguments for structs?  Wow Slang.  We will
// definitely be discarding this awful practice via types and macros.
// NEXT Windows weights are symmetric.  We can halve the storage at the cost of
// a more complex inner loop, taking advantage of reflection and retirement
// occurring at the same point if we play everything right.

// See the dft.rs for more details.

struct DftPushConstants {
    // Root device address of static data, such as configs, window weights etc.
    // The static base has `DftConfig` at the header.
    uint64_t static_base;
    // Root device address of dynamic data, including bin states and output
    // data.  The dynamic address has `ChannelState*` at the header.  Index
    // directly by tid.x.
    uint64_t dynamic_base;

    // NOTE Every index here is physical and already wrapped by the host, so the
    // device never sees an absolute count and never has to reduce one.  Ring
    // wrap is handled by the read cursor and the column bump locally.

    // Physical index of the first sample to read.
    uint input_beg;
    // How many samples to read.  Wrapping is the cursor's problem.
    uint input_count;

    // Ticks elapsed in the open output column as of the sample immediately
    // before `input_beg`.  In [0, ticks_per_column).  Host-derived and shared
    // by every bin, so no bin can drift relative to another.
    uint column_ticks_beg;
    // Physical column index of the output column still open at that instant.
    uint output_column;
}

[[vk::push_constant]]
ConstantBuffer<DftPushConstants> pc;

// Configuration shared by all channels.  Found at `static_base`.
struct DftConfig {
    // Number of bins in this DFT.  Also interpreted as output image height.
    uint bin_count;
    // Upstream ring buffer sample count.  We require PoT length for cheap
    // modulus and other arithmetic benefits.
    uint input_size;
    // Length of lines in the output image.  Each bin will write to a row of the
    // output image, which is stored in column-major format for temporal
    // barriers to cover contiguous regions.
    uint output_image_width;

    // Individual channel configurations.  Offset from `static_base`.  Coerce to
    // ChannelConfig*. Index by tid.x.
    uint channel_configs_offset;
    // NOTE Count omitted.  Host controls maximum tid.x with no bounds check
    // necessary.

    // Output grid as an exact rational on the input grid.  One tick is
    // 1/`ticks_per_column` of an output column; one input sample advances the
    // output position by `ticks_per_sample` ticks.  Host reduces the pair.
    //   48 kHz in, 240 Hz out: 200 and 1.  Boundaries land on sample instants.
    //   44.1 kHz in, 240 Hz out: 735 and 4.  Boundaries land on quarter samples.
    uint ticks_per_sample;
    uint ticks_per_column;
}

// Configuration shared by all bins in the channel.
struct ChannelConfig {
    // Input device address for this channel.  Pointer rather than offset
    // because it comes from the upstream, not our base.
    Ptr<float>  input_ring_base;

    // Individual bin configurations.  Offset from `static_base`.  Coerce to
    // BinConfig* Index into by tid.y
    uint bin_configs_offset;
}

// State shared by all bins in the channel.
struct ChannelState {
    // Offset of this channel's bin state array from `dynamic_base`.  Coerce to
    // `BinState*`.  Index directly by tid.y.  Bounds check on
    // `DftConfig::bin_count`.
    uint bin_states_offset;

    // Offset of this channel's output image ring from `dynamic_base`.  Coerce
    // to `float*`.  Storage is column-major.  Column count is
    // `DftConfig::output_image_width`.  Row count is `DftConfig::bin_count`.
    uint output_image_offset;
}

// Static configuration data for a single DFT bin.
struct BinConfig {
    // Bin phasor rotation per input.
    Complex theta;
    // How many samples are between window starts.  Set by host to control the
    // overlap density in time.
    uint window_spacing;

    // Coerce to `float*`.  Offset from `static_base`.
    uint window_weights_offset;
    // Number of window weights.
    uint window_weights_count;
}

// Bin dynamic states.  Window sums and accumulator state both live here.
struct BinState {
    // The current phasor value, the state variable of the projected Goertzel filter.
    Complex phasor;
    // MAYBE Could store windows as an array if the length constant could be
    // propagated up to Rust.  Using a proc macro to pull out a shared constant
    // or to generate them for slang would not be too cancerous.

    // Coerces to `Complex*`.  Offset from `dynamic_base`.  Always
    // `WINDOW_SLOTS` slots.
    uint window_sums_offset;
    // Slot that is next to retire.
    uint next_retire;
    // Samples elapsed since the retirement that produced `ramp_end`.  In
    // [0, window_spacing).  Retirement fires when it reaches window_spacing.
    uint ramp_samples;

    // Accumulator states

    // Endpoints of the ramp currently on display.  `ramp_beg` is window n-1's
    // level, `ramp_end` is window n's, and the pair is consumed over [T_n,
    // T_n+1) *following* the retirement that produced `ramp_end`.
    float ramp_beg;
    float ramp_end;
    // Trapezoid area banked into the still-open output column, carried in
    // level-ticks.  Divided by `DftConfig::ticks_per_column` at the close,
    // which makes the stored value the column mean.
    float open_area;
}

// DEBT in the interest of time, not messing with specialization constants now.
static const uint WINDOW_SLOTS = 16;
static const uint WINDOW_SLOT_MASK = WINDOW_SLOTS - 1;

// Dispatch with x channels and divide bins by 32 and round up for y.
[numthreads(1, 32, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    let channel = tid.x;
    let bin = tid.y;

    // Unpack all of the states and retain pointers for write-back.
    let static_base = pc.static_base;
    let dynamic_base = pc.dynamic_base;

    DftConfig dft_cfg = *(Ptr<DftConfig>)static_base;
    if (bin >= dft_cfg.bin_count) { return; }

    Ptr<ChannelConfig> channel_cfgs =
        (Ptr<ChannelConfig>)(static_base + (uint64_t)dft_cfg.channel_configs_offset);
    ChannelConfig channel_cfg = channel_cfgs[channel];

    Ptr<BinConfig> bin_cfgs =
        (Ptr<BinConfig>)(static_base + (uint64_t)channel_cfg.bin_configs_offset);
    BinConfig bin_cfg = bin_cfgs[bin];

    Ptr<ChannelState> channel_states = (Ptr<ChannelState>)dynamic_base;
    ChannelState channel_state = channel_states[channel];

    Ptr<BinState> bin_states =
        (Ptr<BinState>)(dynamic_base + (uint64_t)channel_state.bin_states_offset);
    Ptr<BinState> state_ptr = bin_states + bin;
    BinState state = *state_ptr;

    Ptr<float> input_ring = channel_cfg.input_ring_base;
    let input_mask  = dft_cfg.input_size - 1;

    Ptr<float> window_weights =
        (Ptr<float>)(static_base + (uint64_t)bin_cfg.window_weights_offset);

    Ptr<Complex> window_sums =
        (Ptr<Complex>)(dynamic_base + (uint64_t)state.window_sums_offset);

    Ptr<float> out_image =
        (Ptr<float>)(dynamic_base + (uint64_t)channel_state.output_image_offset);

    // Carried filter state.
    Complex phasor = state.phasor;
    uint next_retire  = state.next_retire;
    uint ramp_samples = state.ramp_samples;

    // Carried integrator state.  The ramp on display over the segment we are
    // entering, plus whatever has been banked into the open output interval.
    float ramp_beg = state.ramp_beg;
    float ramp_end = state.ramp_end;
    float open_area = state.open_area;

    // Reciprocals hoisted out of the sample loop.  Each is a ratio against an
    // integer grid: samples within a ramp, ticks within a sample, ticks within
    // a column.
    let inv_window_spacing   = 1.0f / (float)bin_cfg.window_spacing;
    let inv_ticks_per_column = 1.0f / (float)dft_cfg.ticks_per_column;
    let inv_ticks_per_sample = 1.0f / (float)dft_cfg.ticks_per_sample;

    // Position on the output grid, carried in from the host.
    uint column_ticks  = pc.column_ticks_beg;
    uint output_column = pc.output_column;
    uint banked_ticks  = column_ticks;

    // Level at the last banked event.  Every dispatch closes with a bank at its
    // final sample instant, so that event is always the previous dispatch's
    // last sample.
    float banked_level = lerp(ramp_beg, ramp_end,
                              (float)ramp_samples * inv_window_spacing);

    uint read_index = pc.input_beg;
    for (uint s = 0; s < pc.input_count; ++s) {
        // Advance both clocks to this sample's instant.  Everything below reads
        // them as "now", so they must move together.
        ramp_samples += 1;                  // in [1, window_spacing] within the body
        column_ticks += dft_cfg.ticks_per_sample;

        // Close the column if the boundary fell at or before this instant.
        // NOTE column tick rate always < input tick rate.
        if (column_ticks >= dft_cfg.ticks_per_column) {
            column_ticks -= dft_cfg.ticks_per_column;

            // The leftover *is* how far back the boundary sits from this sample,
            // in ticks.  With ticks_per_sample == 1 it is always 0 and boundaries
            // coincide with sample instants.
            let samples_back = (float)column_ticks * inv_ticks_per_sample;
            let level = lerp(ramp_beg, ramp_end,
                             ((float)ramp_samples - samples_back) * inv_window_spacing);

            bank(open_area, banked_level, level,
                 dft_cfg.ticks_per_column - banked_ticks);
            out_image[output_column * dft_cfg.bin_count + bin] =
                open_area * inv_ticks_per_column;

            open_area    = 0.0f;
            banked_level = level;
            banked_ticks = 0;

            output_column += 1;
            if (output_column == dft_cfg.output_image_width) { output_column = 0; }
        }

        let sample = input_ring[read_index];
        read_index = (read_index + 1) & input_mask;

        // The Goertzel.  Why we're here.  ٩(⊙‿⊙)۶
        let projected = cscale(phasor, sample);
        phasor = cmul(phasor, bin_cfg.theta);

        // Fan into the live windows.
        [ForceUnroll]
        for (uint i = 0; i < WINDOW_SLOTS; ++i) {
            // Age in hops; 0 is the window that retires next.
            let age = (i + WINDOW_SLOTS - next_retire) & WINDOW_SLOT_MASK;
            // Next-to-retire window's read head.  It reaches the last weight
            // exactly when ramp_samples reaches window_spacing.  Host guarantees
            // window_weights_count >= window_spacing, so this never underflows.
            let head = bin_cfg.window_weights_count - 1
                     - bin_cfg.window_spacing + ramp_samples;
            let hop_back = age * bin_cfg.window_spacing;   // needs 32 bits: 15 * 4800
            // Only way out of range is stepping back past the start.  Explicit
            // test, no reliance on underflow.
            let live     = hop_back <= head;
            // Something something underflow OpSelect...  Be my guest if this
            // code breaks on your driver.
            let w_index  = live ? head - hop_back : 0u;
            let w        = live ? window_weights[w_index] : 0.0f;
            window_sums[i] = cmad(window_sums[i], projected, w);
        }

        // The ramp is exactly at `ramp_end` here; retire the window.
        if (ramp_samples == bin_cfg.window_spacing) {
            let sum   = window_sums[next_retire];
            let level = sqrt(sum.real * sum.real + sum.imag * sum.imag);

            bank(open_area, banked_level, ramp_end, column_ticks - banked_ticks);
            banked_level = ramp_end;
            banked_ticks = column_ticks;

            ramp_beg = ramp_end;
            ramp_end = level;

            window_sums[next_retire] = Complex(0.0f, 0.0f);
            next_retire  = (next_retire + 1) & WINDOW_SLOT_MASK;
            ramp_samples = 0;
        }
    }

    // Close the dispatch on a banked event at the final sample instant, so the
    // next dispatch resumes with banked_ticks == pc.column_ticks_beg and can
    // rebuild banked_level from the ramp alone.
    bank(open_area, banked_level,
         lerp(ramp_beg, ramp_end, (float)ramp_samples * inv_window_spacing),
         column_ticks - banked_ticks);

    // Renormalize once per quantum.  Kills the exponential magnitude drift;
    // phase drift is a bounded random walk and needs no correction.
    phasor = cscale(phasor, rsqrt(phasor.real * phasor.real +
                                  phasor.imag * phasor.imag));

    // Write back.  Repacking into the loaded struct and storing once keeps the
    // unpack/repack symmetric and lets this lower to a single contiguous store;
    // the immutable fields are rewritten with the values they already held.
    state.phasor = phasor;
    state.next_retire = next_retire;
    state.ramp_samples = ramp_samples;
    state.ramp_beg = ramp_beg;
    state.ramp_end = ramp_end;
    state.open_area = open_area;
    *state_ptr = state;
}

// NEXT library dawg
struct Complex {
    float real;
    float imag;
}

Complex cmul(Complex a, Complex b) {
    return Complex( a.real * b.real - a.imag * b.imag,
                    a.real * b.imag + a.imag * b.real );
}

Complex cmad(Complex acc, Complex a, float s) {
    return Complex(acc.real + a.real * s, acc.imag + a.imag * s);
}

Complex cscale(Complex a, float s) {
    return Complex(a.real * s, a.imag * s);
}

// Bank the trapezoid under the ramp from the last banked event to this one.
// `ticks` is the exact width in ticks, so `open_area` carries in level-ticks and
// dividing by `ticks_per_column` at the close *is* the column mean.
void bank(inout float open_area, float a, float b, uint ticks) {
    open_area += 0.5f * (a + b) * (float)ticks;
}
