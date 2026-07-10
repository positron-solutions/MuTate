// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct FloatBuffer {
    float samples[];
};

[[vk::push_constant]]
cbuffer HelloConstants {
    FloatBuffer* left_channel;
    FloatBuffer* right_channel;
    uint capacity;
    uint counter;
    float2 window_size;
    uint output_idx;
};

[[vk::binding(5, 0)]]
RWByteAddressBuffer storage_buffers[];

[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Bounds check masks off lanes that would draw outside of the available area.
    uint2 pixel = tid.xy;
    if (pixel.x >= (uint)window_size.x || pixel.y >= (uint)window_size.y)
        return;

    // Pull in the raw audio from each channel
    let index = (pixel.x + pixel.y * window_size.x) % capacity;
    float left_sample = left_channel.samples[index];
    float right_sample = right_channel.samples[index];

    // Blue is linear mapped with constant gain.  Green and Red use a log
    // mapping.  Red maxes out at half brightness.
    uint b = (uint)round(saturate(right_sample * 24.0) * 255.0);
    float g_log = saturate(log2(1.0 + saturate(left_sample) * 1023.0) / log2(1024.0));
    uint g = (uint)round(g_log * 255.0);
    float r_log = saturate(log2(1.0 + saturate(right_sample) * 255.0) / log2(256.0));
    uint r = (uint)round(r_log * 200.0);

    // BGRA
    uint packed = b
        | (g <<  8)
        | (r << 16)
        | (0xFFu << 24);

    uint byte_offset = (tid.y * (uint)window_size.x + tid.x) * 4; // placeholder stride
    // Descriptor lookup (BDA would work here too)
    RWByteAddressBuffer output = storage_buffers[output_idx];
    output.Store(byte_offset, packed);
}
