// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

[[vk::push_constant]]
cbuffer HelloConstants {
    float4 color; // rgba
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

    // BGRA?
    uint packed = (uint(round(color.a * 255.0)) << 24)
                | (uint(round(color.b * 255.0)) << 16)
                | (uint(round(color.g * 255.0)) << 8)
                |  uint(round(color.r * 255.0));

    uint byte_offset = (tid.y * (uint)window_size.x + tid.x) * 4; // placeholder stride
    output.Store(byte_offset, packed);
}
