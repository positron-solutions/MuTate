// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct PackedBuffer {
    uint texels[];
};

[[vk::push_constant]]
cbuffer PulseConstants {
    uint window_width;
    uint window_height;
    PackedBuffer* input;
    uint output_idx;
};

[[vk::binding(5, 0)]]
RWByteAddressBuffer storage_buffers[];

[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Bounds check masks off lanes that would draw outside of the available area.
    uint2 pixel = tid.xy;
    if (pixel.x >= window_width || pixel.y >= window_height)
        return;

    let packed = (*input).texels[0];
    // let packed = 0xFFFFFFFFu;

    // NEXT go ahead and switch to BDA.
    RWByteAddressBuffer output = storage_buffers[output_idx];
    uint byte_offset = (tid.y * window_width + tid.x) * 4;
    output.Store(byte_offset, packed);
}
