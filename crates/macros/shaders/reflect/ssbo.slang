// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

struct Particle
{
    float3 position;
    float  mass;
    float3 velocity;
    float  radius;
};

StructuredBuffer<Particle>   g_particles_in;
RWStructuredBuffer<Particle> g_particles_out;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    Particle p = g_particles_in[tid.x];
    p.velocity += float3(0.0, -9.8 * 0.016, 0.0);
    g_particles_out[tid.x] = p;
}
