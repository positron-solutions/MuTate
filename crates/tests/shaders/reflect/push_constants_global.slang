// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct PushData
{
    uint  dispatch_id;
    float scale;
    // XXX duplicate the newtypes from mutate-lib for a more representative test
    // vector.
    uint  ubo_index;
    uint  ssbo_index;
};

// Global declaration style, probably limits us to one entry per file.
[[vk::push_constant]]
PushData push;

[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    // just touch each field so the compiler can't DCE them
    uint _ = push.dispatch_id + push.ubo_index + push.ssbo_index;
    float __ = push.scale;
}
