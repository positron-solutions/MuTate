// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Using entry point push constants will throw some warnings.
// XXX unknown truthiness!  We need to test the entry point style in a concrete setting.
#pragma warning(disable: 38010)
#pragma warning(disable: 38040)

struct PushData
{
    uint  dispatch_id;
    float scale;
    // XXX duplicate the newtypes from mutate-lib for a more representative test
    // vector.
    uint  ubo_index;
    uint  ssbo_index;
};

// Slang: push constant = function parameter decorated with [[vk::push_constant]]
// This is the *only* correct place for push constants in Slang for SPIR-V.
// Global-scope [[vk::push_constant]] is also accepted but entry-point params
// are easier to reflect — the entry point's parameter list is where the
// reflection API reports them under getParameterCount().

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_push(
    [[vk::push_constant]] PushData push,
    uint3 tid : SV_DispatchThreadID)
{
    // just touch each field so the compiler can't DCE them
    uint _ = push.dispatch_id + push.ubo_index + push.ssbo_index;
    float __ = push.scale;
}
