// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// See the dft.rs for more details.  Field descriptions in the Rust code are
// authoritative.

// LIES ⚠️ "static allocation" and "dynamic allocation" are not yet part of the
// implementation, but the comment language will be consistent if they *were*.
// Device addresses will be used even though they come from the same allocation.
// Pretending we have a sub-allocator is part of the same lie already.
// NOTE Host always provides offsets that are aligned for the target type and
// add directly to a device address to form the destination array address.
// Device address + u32 yields a valid pointer for the T described by the field.
// NEXT As the common math begins to show up, start supporting newtypes and
// streamline the device address coercion so that our Slang is cleaner.
// NEXT a lot of physical ring indexes can be compressed to u16, but with
// extra conversions and no benefit unless registers are spilling back to
// memory, which would be dominated by window sum complexity, not the phase
// tracking data.
// NEXT phasor updates can be split into 2x theta at the cost of some prologue /
// epilogue.  The operations serialize and may be creating latency we can hide
// by providing a second phasor to rotate.
// NEXT if we have enough register space and it won't change the loads, some
// ripple-carry-adds will increase the precision of long sums and convert some
// adds to simple moves.  Requires an extra Complex slot per window and
// retirement epilogue unless window length guarantees an empty carry slot.
// DEBT Warp geometry!

// LMAO positional constructor arguments for structs?  Wow Slang.  We will
// definitely be discarding this awful practice via types and macros.

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
    uint input_beg;
    uint input_count;
    uint output_column;
    uint column_phase;
}

[[vk::push_constant]]
ConstantBuffer<DftPushConstants> pc;

// Configuration shared by all channels.  Found at `static_base`.
struct DftConfig {
    uint bin_count;
    uint input_size;
    uint output_image_width;

    // Offset from `static_base`.  Coerce to ChannelConfig*. Index by tid.x.
    uint channel_configs_offset;
    // NOTE Count omitted.  Host controls maximum tid.x with no bounds check
    // necessary.

    // Coerce to `float*`.  Offset from `static_base`.
    uint window_weights_offset;
    uint window_spacing;
}

struct ChannelConfig {
    // Input device address for this channel.  Pointer rather than offset
    // because it comes from the upstream, not our base.
    Ptr<float>  input_ring_base;

    // Individual bin configurations.  Offset from `static_base`.  Coerce to
    // BinConfig* Index into by tid.y
    uint bin_configs_offset;
}

struct ChannelState {
    //  Coerce to `BinState*`.  Offset from `dynamic_base`.   Index directly by
    //  tid.y.
    uint bin_states_offset;

    // Coerce to `Complex*`.  Offset from `dynamic_base`.
    uint output_image_offset;
}

struct BinConfig {
    // Bin phasor rotation per sample.
    Complex theta;
}

struct BinState {
    // The current phasor value, the state variable of the projected Goertzel filter.
    Complex phasor;
    // Coerces to `Complex*`.  Offset from `dynamic_base`.  Always
    // `WINDOW_SLOTS` slots.
    uint window_sums_offset;
}

// DEBT in the interest of time, not messing with specialization constants now.
static const uint WINDOW_SLOTS = 4;

// Dispatch with x channels and divide bins by 32 and round up for y.
[numthreads(1, 32, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    let channel = WaveReadLaneFirst(tid.x);
    let bin = tid.y;

    // Unpack all of the states and retain pointers for write-back.
    let static_base = pc.static_base;
    let dynamic_base = pc.dynamic_base;

    DftConfig dft_cfg = *(Ptr<DftConfig>)static_base;

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

    Ptr<float> window_weights =
        (Ptr<float>)(static_base + (uint64_t)dft_cfg.window_weights_offset);

    Ptr<Complex> window_sums_spill =
        (Ptr<Complex>)(dynamic_base + (uint64_t)state.window_sums_offset);

    Ptr<Complex> out_image =
        (Ptr<Complex>)(dynamic_base + (uint64_t)channel_state.output_image_offset);

    let window_spacing = WaveReadLaneFirst(dft_cfg.window_spacing);
    let input_size = WaveReadLaneFirst(dft_cfg.input_size);
    let input_mask  = input_size - 1;
    let bin_count = WaveReadLaneFirst(dft_cfg.bin_count);
    let output_image_mask = WaveReadLaneFirst(dft_cfg.output_image_width) - 1;

    // Carried filter state and window sums
    Complex phasor = state.phasor;

    Complex sums[WINDOW_SLOTS];
    [ForceUnroll]
    for (uint i = 0; i < WINDOW_SLOTS; ++i) { sums[i] = window_sums_spill[i]; }

    // Samples elapsed in the current hop.  Retires at `window_spacing`.
    // INVARIANT Host guarantees ramp_beg < window_spacing.  Equality yields
    // run == 0 and hangs the wave
    uint column_phase = pc.column_phase;

    // Pre-build slot stride (pre-build on the host?)
    uint slot_stride[WINDOW_SLOTS];
    [ForceUnroll]
        for (uint i = 0; i < WINDOW_SLOTS; ++i) {
            slot_stride[i] = (WINDOW_SLOTS - 1 - i) * window_spacing;
        }

    // Set a pointer into window weights for use with our strides
    Ptr<float> wp = window_weights + column_phase;

    uint output_column = pc.output_column;

    let input_count = pc.input_count;
    uint read_index = pc.input_beg;
    uint s = 0;

    while (s < input_count) {
        // Clamp to the hop boundary and to the ring wrap
        uint run = min(window_spacing - column_phase, input_count - s);
        run = min(run, input_size - read_index);

        Ptr<float> rp = input_ring + read_index;
        for (uint k = 0; k < run; ++k) {
            let sample = rp[k];
            // The Goertzel.  Why we're here.  ٩(⊙‿⊙)۶
            phasor = cmul(phasor, bin_cfg.theta);

            // Pre-multiply the input sample by the weights since it will carry
            // through to the phasor projection and addition.
            //
            // Uses window spacing to walk through window weights.
            float sw[WINDOW_SLOTS];
            [ForceUnroll]
                for (uint i = 0; i < WINDOW_SLOTS; ++i) { sw[i] = sample * wp[slot_stride[i]]; }

            // Fan into the live windows
            [ForceUnroll]
                for (uint i = 0; i < WINDOW_SLOTS; ++i) { sums[i] = cmad(sums[i], phasor, sw[i]); }

            wp = wp + 1;
        }

        s += run;
        column_phase += run;
        read_index = (read_index + run) & input_mask;

        // Window retirement.  Write the output, rotate the sums, reset loop
        // loop control.
        if (column_phase == window_spacing) {
            out_image[output_column * bin_count + bin] = sums[0];
            output_column = (output_column + 1) & output_image_mask;

            // Rotate the window sums and zero the last slot
            [ForceUnroll]
            for (uint i = 0; i < WINDOW_SLOTS - 1; ++i) { sums[i] = sums[i + 1]; }
            sums[WINDOW_SLOTS - 1] = Complex(0.0f, 0.0f);

            column_phase = 0;
            wp = window_weights;
        }
    }

    // Renormalize once per quantum.  Kills the exponential magnitude drift;
    // phase drift is a bounded random walk and needs no correction.
    phasor = cscale(phasor, rsqrt(phasor.real * phasor.real +
                                  phasor.imag * phasor.imag));

    // Write window states and phasor state back to VRAM
    [ForceUnroll]
    for (uint i = 0; i < WINDOW_SLOTS; ++i) { window_sums_spill[i] = sums[i]; }

    state.phasor = phasor;
    *state_ptr = state;
}

// DEBT Slang libraries
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
