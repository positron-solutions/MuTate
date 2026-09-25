// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// See the cwt.rs for more details.  Field descriptions in the Rust code are
// authoritative.

struct PushConstants {
    // Device address of input data, including the output header state.
    uint64_t input_base;
    // Device address of static data, such as configs, wavelet table, bin table
    // etc.  Header is a Config and may point to an Output struct describing
    uint64_t static_base;
    // Device address of output data and any dynamic state.  Read information is
    // contained in the Output in the static allocation.
    uint64_t output_base;
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

static const uint LANES = 32;
[WaveSize(LANES)]
[numthreads(8, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {

// Compile-time: L lanes/window, T = pairs/lane, P hops in flight, U = 4
//
// - warp = G = 32/L groups
// - group g owns hops [hbase + g*P, +P)
//
// lane l within group owns pairs [l*T, l*T + T)
//

// uint kbase = lane_in_group * T;              // this lane's tap slice
// uint c0    = center_of(hbase + g*P) + kbase; // swizzled word coords

// float4 acc[P];                               // (re_psi, im_psi, re_tpsi, im_tpsi)

// for (uint kp = 0; kp < T; kp += U) {
//     // single-buffered: 64 MADs of cover below
//     float4 w[U] = load4x(wbase + (((kbase + kp) & kmask) << 4));

//     [unroll] for (uint p = 0; p < P; ++p) {
//         float4 fwd = lds_load4(c0 + p*H + kp);
//         float4 rev = lds_load4_rev(c0 + p*H - kp - (U-1) - 2*kbase);

//         [unroll] for (uint u = 0; u < U; ++u) {
//             float s = fwd[u] + rev[U-1-u];
//             float d = fwd[u] - rev[U-1-u];
//             acc[p].x = mad(s, w[u].x, acc[p].x);
//             acc[p].y = mad(d, w[u].y, acc[p].y);
//             acc[p].z = mad(d, w[u].z, acc[p].z);
//             acc[p].w = mad(s, w[u].w, acc[p].w);
//         }
//     }
// }

// epilogue: log2(L) butterfly reduce acc[p] within group, lane 0 emits P outputs,
// then rotate c0 forward by G*P*H and repeat.

}
