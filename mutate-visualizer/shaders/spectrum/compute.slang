struct SpectrumSample
{
    float left_decibels;
    float left_mag;
    float right_decibels;
    float right_mag;
};

[[vk::push_constant]]
cbuffer PushConstants
{
    float2 window_size; // (width, height)
};

[[vk::binding(0, 0)]]
StructuredBuffer<SpectrumSample> spectrum;

[[vk::binding(1, 0)]]
RWByteAddressBuffer outputBuffer;

[numthreads(8,16,1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= (uint)window_size.x || pixel.y >= (uint)window_size.y)
        return;

    SpectrumSample s = spectrum[window_size.y - (pixel.y + 1)];

    // red
    float left_low = hill_function(s.left_mag, 80.0, 1.0, 4.1);
    float right_low = hill_function(s.right_mag, 80.0, 1.0, 4.1);
    float both_ramped = (left_low + right_low) * 0.5;
    // green
    float left_ramped = hill_function((s.left_decibels + 50.5), 22.5, 1.0, 4.1);
    // blue
    float right_ramped = hill_function((s.right_decibels + 50.5), 22.5, 1.0, 4.1);

    // NOTE Hill function also kind of clamps.
    uint packed = (255 << 24)
                  | ((uint)round(both_ramped * 255.0) << 16)
                  | ((uint)round(left_ramped * 255.0) << 8)
                  | (uint)round(right_ramped * 255.0);

    uint out_index = (pixel.y * uint(window_size.x) + pixel.x) * 4;
    outputBuffer.Store(out_index, packed);
}


/// Hill function starts at zero, has a controllable halfway point, asymptote, and shape.
///
/// - x: the variable input, expected to be on-scale
/// - half_x: x value that reaches half of the asymptote
/// - max: the asymptote
/// - c_hill: Hill coefficient (> 1.0 gives sharper / double-inflection shapes)
float hill_function(
    float x,
    float half_x,
    float max,
    float c_hill)
{
    float t_n = pow(x, c_hill);
    return max * (t_n / (t_n + pow(half_x, c_hill)));
}