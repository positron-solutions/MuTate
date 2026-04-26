// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
#language slang 2026

/// * Test Compute
///
/// Declare push constants.  Look up an input buffer.  Write to an output
/// buffer.o

// An interface to identify resource categories
interface IResourceType {}
struct MainColorTex : IResourceType {}
struct NormalMapTex : IResourceType {}

// The "Newtype" wrapper
struct DescriptorHandle<T : IResourceType>
{
    uint index;

    // Helper to make usage cleaner in the shader
    uint get() { return index; }
}

struct GlobalPushConstants {
    uint normal_scalar;
    uint16_t half_scalar;
    uint8_t byte_scalar;

    DescriptorHandle<MainColorTex> diffuseIdx;
    DescriptorHandle<NormalMapTex> normalIdx;

    // Range [64, 80): Fragment Shader data
    [[vk::offset(64)]]
    float4 lightColor;
};

[vk::push_constant]
uniform GlobalPushConstants pushData;

[[vk::binding(0, 0)]]
RWByteAddressBuffer outputBuffer;

[numthreads(1,1,1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    outputBuffer.Store(0, pushData.byte_scalar);
}
