// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Verticlysm
//
// DFT bins mapped across the x-axis of the output buffer and rendered
// vertically.
//
// The upstream DFT output is a ring for each channel stored as a column-major
// buffer of f32s.  The rows of the input are DFT bins.  The columns of the ring
// are time.  The data is stored column major.
//
// We map the bin rows onto the output buffer's x-axis.  A slice of audio output
// bins lands within each x pixel.  This slice of rows is then integrated over a
// time slice, a range of the columns.  The integration produces the magnitude
// of energy within the time slice and the audio pitch range.
//
// The y axis of the output buffer re-uses the magnitude and only the way
// magnitude is drawn changes along the y axis.  Dispatch geometry takes
// advantage of this re-use of the by giving each lane a output buffer column
// and using one lane of dispatch to write `LANE_ROWS` of that column.
//
// The host has converted logical index to physical index already.  Physical
// wrap is performed at the input ring row width, which is guaranteed to be PoT.
//
// Both left and right channels are drawn as a vertical gradient along the
// column.  Left is blue and makes a gradient from top to bottom.  Right is
// green and makes a gradient from bottom to top.  The gradient math starts at
// the magnitude and drops by 1.0 per output buffer.  A magnitude of 0.6 will
// become zero when the gradient is 0.6 output buffers from the edge.  A
// magnitude of 1.0 will be 0.5 in the middle of the output buffer.  A magnitude
// of 0.1 is 0.0 at 0.1 output buffers from the edge.

// NEXT look at snapping the bin edges during interpolation to avoid
// pathological slivers of bins when bins and pixels align almost exactly.

[[vk::push_constant]]
cbuffer VerticlasmConstants {
    // Input buffers
    float* left_channel;
    float* right_channel;

    // Column height == bin count of the DFT
    uint input_height;
    // Ring buffer width.  Rows are time.  Always PoT for easy wrap.
    uint input_width;

    // Physical column index.
    uint  beg_col;
    // Phase within beg_col.  [0.0, 1.0)
    float beg_phase;
    // Width of the time slice in input columns.
    float span;

    // NEXT a gain correction on the left and right channel will be applied to
    // each input using the identical indexing.
    // float* gains;

    // Output buffer, row major and usually copied to a swapchain or other image.
    uint* output;
    uint output_width;
    uint output_height;
};

// Each mangitude calculation is re-used to write outputs on this many rows.
static const uint LANE_ROWS = 128;

// Warps operate on 32 pixel wide, LANE_ROWS tall areas of the output image.
// The magnitude calculation is re-used by each lane for LANE_ROWS rows.
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Bounds check masks off lanes that would draw outside of the available area.
    if ((tid.x >= window_width) || (tid.y * LANE_ROWS >= window_height))
        return;

    // Calculate magnitude.  We will re-sample the bins that land within this
    // output buffer x and integrate the same bin slice over the time slice to obtain
    // the magnitude.  Remember, bins are rows and columns are time.
    float acc_l = 0.0;
    float acc_r = 0.0;

    // Time axis: columns of the ring.  `col_beg`/`col_count` are relative to
    // the slice start, so they never wrap; only the address does.
    let col_mask  = input_width - 1; // Works via PoT invariant from the host.
    let col_phase = beg_phase;
    let col_first = beg_col & col_mask;
    let col_count = uint(ceil(col_phase + span)); // host ensures <= input_width

    // Bin axis: rows of the ring, re-sampled onto output buffer x.
    let bins_per_px = float(input_height) / float(half_width);
    let bin_beg   = float(tid.x) * bins_per_px;
    let bin_end   = (float(tid.x) + 1.0) * bins_per_px;
    let bin_first = uint(floor(bin_beg));
    let bin_count = uint(ceil(bin_end)) - bin_first;

    // Time is the outer loop because bins are contiguous along columns in the
    // input ring.
    for (uint c = 0; c < col_count; ++c) {
        // Overlap of column `c` with [col_phase, col_phase + span), in
        // relative space.  Relative, so this arithmetic never sees the seam.
        let col_lo = max(float(c),       col_phase);
        let col_hi = min(float(c) + 1.0, col_phase + span);
        let col_weight = max(col_hi - col_lo, 0.0);

        // Only the address wraps.
        let phys_col = (col_first + c) & col_mask;
        let col_base = phys_col * input_height;

        float col_acc_l = 0.0;
        float col_acc_r = 0.0;
        for (uint b = 0; b < bin_count; ++b) {
            let bin = bin_first + b;
            let bin_lo = max(float(bin),       bin_beg);
            let bin_hi = min(float(bin) + 1.0, bin_end);
            let bin_weight = max(bin_hi - bin_lo, 0.0);

            let sample_l = float(left_channel [col_base + bin]);
            let sample_r = float(right_channel[col_base + bin]);
            col_acc_l += bin_weight * sample_l * sample_l;
            col_acc_r += bin_weight * sample_r * sample_r;
        }

        acc_l += col_weight * col_acc_l;
        acc_r += col_weight * col_acc_r;
    }

    // We integrated over a columns * bins area.  Normalize to area.
    let inv_area = 1.0 / max(span * bins_per_px, 1e-6);

    // XXX Raw window sums are not in DB and there is no dynamic range yet!.
    // let mag_l = acc_l * inv_area;
    // let mag_r = acc_r * inv_area;

    static const float DEBUG_GAIN = 0.001;
    let mag_l = sqrt(acc_l * inv_area) * DEBUG_GAIN;
    let mag_r = sqrt(acc_r * inv_area) * DEBUG_GAIN;

    // Iterate over Y for LANE_ROWS output rows, writing BGRA.
    let row_height  = 1.0 / float(output_height);
    let y_beg = float(tid.y * LANE_ROWS) * row_height;

    float grad_l = mag_l - y_beg; // blue, top -> bottom
    float grad_r = mag_r - (1.0 - y_beg); // green, bottom -> top

    uint y = tid.y * LANE_ROWS;
    let y_end = min(y + LANE_ROWS, output_height);
    while (y < y_end) {
        output[y * window_width + tid.x] = pack_bgra(grad_l, grad_r, 0.0, 1.0);
        grad_l -= row_height;
        grad_r += row_height;
        ++y;
    }
}

uint pack_bgra(float b, float g, float r, float a) {
    let c = saturate(float4(b, g, r, a));
    let q = uint4(round(c * 255.0));
    return (q.w << 24) | (q.z << 16) | (q.y << 8) | q.x;
}
