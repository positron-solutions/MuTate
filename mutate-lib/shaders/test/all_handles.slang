// test_all_handles.slang
// Compile-test: exercises every handle type from descriptors.slang
// to prevent DCE from eliminating the bindings.

import descriptors;

struct AllHandles {
    SamplerIdx          sampler;
    CompSamplerIdx      comp_sampler;
    SampledImageIdx     sampled_image;
    StorageImageIdx     storage_image;
    UboIdx              ubo;
    SsboIdx             ssbo;
    UniformTexelBufferIdx  uniform_texel;
    StorageTexelBufferIdx  storage_texel;
};

[[vk::binding(4, 1)]] ConstantBuffer<AllHandles> g_handles;

[[vk::binding(5, 1)]] RWStructuredBuffer<uint> g_sink;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x != 0) return;

    AllHandles h = g_handles;

    // Slot 0 — SAMPLER
    // SamplerState and SamplerComparisonState have no standalone load;
    // we read the handle's inner index as a proxy touch.
    uint sink = h.sampler.inner + h.comp_sampler.inner;

    // Slots 2–3 — IMAGE (plain uint aliases)
    sink += h.sampled_image + h.storage_image;

    // Slot 4 — UNIFORM_BUFFER
    // Load a single uint from offset 0 of the bound byte-address buffer.
    uint ubo_val = h.ubo.get<uint>();
    sink += ubo_val;

    // Slot 5 — STORAGE_BUFFER (read + write round-trip)
    uint ssbo_val = h.ssbo.get<uint>();
    sink += ssbo_val;
    h.ssbo.set<uint>(sink);          // also exercises Store path

    // Slots 6–7 — TEXEL BUFFER (plain uint aliases, no global array yet)
    sink += h.uniform_texel + h.storage_texel;

    // Commit accumulated value so nothing above is DCE'd.
    g_sink[0] = sink;
}
