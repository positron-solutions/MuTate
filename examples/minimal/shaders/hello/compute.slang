// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

[[vk::push_constant]]
cbuffer HelloConstants {
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

    RWByteAddressBuffer output = storage_buffers[output_idx];


    uint x = (pixel.x + counter) & 0xFF;
    uint y = (pixel.y + counter) & 0xFF;
    uint s = (pixel.x + pixel.y + counter) & 0xFF;

    // BGRA: b=x, g=y, r=x+y, a=255
    uint packed = (0xFFu << 24)
                | (x    << 16)
                | (y    <<  8)
                |  s;

    uint byte_offset = (tid.y * (uint)window_size.x + tid.x) * 4; // placeholder stride
    output.Store(byte_offset, packed);
}
