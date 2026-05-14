// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// NOTE 16 bytes under scalar data layout
struct GpuData {
    float3 position;
    uint   flags;
};

struct PushConstants {
    uint image_idx;
    uint sampler_idx;
    uint sampler_compare_idx;
    uint storage_img_idx;
    uint ubo_idx;
    uint ssbo_idx;
};

[[vk::push_constant]]
PushConstants push_constants;

[[vk::binding(0,0)]]
Texture2D sampled_images[];

// NOTE re-use of indexes.  The handle type accounting is doing some lifting for
// us here.  Not the only place.
#pragma warning(disable: 39001)
[[vk::binding(0, 1)]]
SamplerState samplers[];

[[vk::binding(0, 1)]]
SamplerComparisonState comparison_samplers[];
#pragma warning(default: 39001)

[[vk::binding(0,2)]]
RWTexture2D storage_images[];

[[vk::binding(0,3)]]
ConstantBuffer<GpuData> ubos[];

#pragma warning(disable: 39001)
[[vk::binding(0, 4)]]
StructuredBuffer<GpuData> ssbos_input[];

[[vk::binding(0, 4)]]
RWByteAddressBuffer ssbos_output[];
#pragma warning(default: 39001)

[shader("compute")]
[numthreads(64, 1, 1)]
// NOTE annotated the entry point function arg instead of using a global
void main(uint3 tid : SV_DispatchThreadID)
{

}
