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
    float left_low = power_ramp(s.left_mag, 0.0, 0.01, 1.0);
    float right_low = power_ramp(s.right_mag, 0.0, 0.01, 1.0);
    float both_ramped = (left_low + right_low) * 0.5;

    // green
    float left_ramped = power_ramp(s.left_decibels, -84.5, -16.5, 4.1);

    // blue
    float right_ramped = power_ramp(s.right_decibels, -52.5, -28.5, 3.8);

    uint packed = (255 << 24)
                  | ((uint)round(both_ramped * 255.0) << 16)
                  | ((uint)round(left_ramped * 255.0) << 8)
                  | (uint)round(right_ramped * 255.0);

    uint out_index = (pixel.y * uint(window_size.x) + pixel.x) * 4;
    outputBuffer.Store(out_index, packed);
}


// Power ramp is a simple function that maps a fixed range from 0 to 1 and can
// dampen or accelerate the curve.  Powers greater than 1.0 cause dampening.
// Powers less than 1.0 will magnify very small values quickly.
// NEXT add an asymptote to preserve a tiny amount of dynamic range on excessive values.
float power_ramp (float x, float min, float max, float power) {
      return pow((clamp(x, min, max) - min) / (max - min), power);
}
