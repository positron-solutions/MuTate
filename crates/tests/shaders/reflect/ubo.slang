// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct CameraData
{
    float4x4 view;
    float4x4 proj;
    float4   eye_pos;
    float    near_z;
    float    far_z;
    uint     frame_index;
    uint     _pad;
};

ConstantBuffer<CameraData> g_camera;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    float4 _ = mul(g_camera.view, float4(g_camera.eye_pos.xyz, 1.0));
}
