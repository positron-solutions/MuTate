struct PushData {
    float4 uColor;
}

[[vk::push_constant]]
PushData gPush;

struct FSOut {
    float4 color : SV_Target0;
}

[shader("fragment")]
FSOut mainFS()
{
    FSOut o;
    o.color = gPush.uColor;
    return o;
}
