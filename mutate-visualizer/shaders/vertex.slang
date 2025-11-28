static const float2 kPositions[3] = {
    float2( 0.0,  0.5),
    float2(-0.5, -0.5),
    float2( 0.5, -0.5)
};

struct VSOut {
    float4 position : SV_Position;
};

[shader("vertex")]
VSOut mainVS(uint vertexID : SV_VertexID)
{
    VSOut o;
    float2 pos = kPositions[vertexID];
    o.position = float4(pos.x, -pos.y, 0.0, 1.0);
    return o;
}
