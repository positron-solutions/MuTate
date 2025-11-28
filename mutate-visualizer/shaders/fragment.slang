struct FSOut {
    float4 color : SV_Target0;
};

[shader("fragment")]
FSOut mainFS()
{
    FSOut o;
    o.color = float4(1.0, 0.2, 0.2, 1.0);
    return o;
}
