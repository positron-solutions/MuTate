// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct FloatBuffer {
    float samples[];
};

[[vk::push_constant]]
cbuffer RmsConstants {
    FloatBuffer* input;
    FloatBuffer* output;
    uint head;
};

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    float input_sample = input.samples[0];

    // do the RMS
    //...
    // something something
}
