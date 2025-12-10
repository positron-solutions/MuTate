static const float2 kPositions[3] = {
    float2( 0.0,  0.5),
    float2(-0.5, -0.5),
    float2( 0.5, -0.5)
};

struct PushData {
    float4 _uColor;
    float scale;
}

[[vk::push_constant]]
PushData gPush;

struct VSOut {
    float4 position : SV_Position;
};

[shader("vertex")]
VSOut mainVS(uint vertexID : SV_VertexID)
{

    VSOut o;
    float2 basePos = kPositions[vertexID];
    float2 scaledPos = basePos * gPush.scale;
    o.position = float4(scaledPos.x, -scaledPos.y, 0.0, 1.0);
    return o;
}
