// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct FloatBuffer {
    float samples[];
};

struct PackedBuffer {
    uint texels[];
};

[[vk::push_constant]]
cbuffer RmsConstants {
    FloatBuffer* left_head;
    FloatBuffer* right_head;
    uint count_head;
    FloatBuffer* left_tail;
    FloatBuffer* right_tail;
    uint count_tail;
    PackedBuffer* output;
};

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    float left_sum_sq  = 0.0;
    float right_sum_sq = 0.0;
    float diff_sum_sq = 0.0;

    for (uint k = 0; k < count_head; ++k) {
        float l = (*left_head).samples[k];
        float r = (*right_head).samples[k];
        left_sum_sq  += l * l;
        right_sum_sq += r * r;
        diff_sum_sq  += (r - l) * (r - l);
    }

    for (uint k = 0; k < count_tail; ++k) {
        float l = (*left_tail).samples[k];
        float r = (*right_tail).samples[k];
        left_sum_sq  += l * l;
        right_sum_sq += r * r;
        diff_sum_sq  += (r - l) * (r - l);
    }

    uint count = count_head + count_tail;
    float inv_n = (count != 0) ? (1.0 / float(count)) : 0.0;
    float left_rms  = sqrt(left_sum_sq  * inv_n) * 5.0;
    float right_rms = sqrt(right_sum_sq * inv_n) * 1.0;
    float diff_rms = sqrt(diff_sum_sq * inv_n) * 10.0;

    // NEXT peak detection!
    // float max_rms = max(left_rms, right_rms);

    // This is a unified dispatch, but next we will separate peak detection and
    // use it as a feedback value rather than a primary output.
    (*output).texels[0] = pack_bgra_unorm(float4(left_rms, right_rms, diff_rms, 1.0));
}

uint pack_bgra_unorm(float4 bgra) {
    uint4 q = uint4(round(saturate(bgra) * 255.0));
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}
