// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct PushConstants {
    uint outputBufferIdx;
};

[vk::push_constant]
uniform PushConstants pushData;

[[vk::binding(5, 0)]] // XXX use centralized slot index constants
RWByteAddressBuffer storageBuffers[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint current = storageBuffers[pushData.outputBufferIdx].Load(0);
    storageBuffers[pushData.outputBufferIdx].Store(0, current + 1);
}
