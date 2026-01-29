// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # ISO 226
//!
//! [ISO 226](https://cdn.standards.iteh.ai/samples/83117/6afa5bd94e0e4f32812c28c3b0a7b8ac/ISO-226-2023.pdf)
//! defines curves that we can use to correct the sensitivity of frequency bins to match human
//! perception.
//!
//! This isn't really used for processing in the amplitude domain but instead to create
//! weights for bins, such as those used for a CQT.  If you need amplitude domain mapping,
//! look for something with computationally simple rules like the K weights filter, which just uses
//! a high-pass and a shelf.

use mutate_lib as utate;

/// 70.0 phons was chosen as a hardcode because this was used in the ITU-R BS.1770-5 standard and
/// stated to be the usual TV listening volume.  Americans measure TV volumes in gallons per slug or
/// something, so we used the EU standard.
// LIES technically, we should be using a different curve for each input SPL because the 70 phon
// curve is only correct for SPLs on the 70 phon curve.  SPLs off of the 70 phon curve need a
// different correction from a different curve, but how would we choose that curve?
const CURVE_PHONS: f32 = 70.0;

/// Return signal gain dB necessary to map perceptually iso-loud tones (which follow a curve in
/// measured SPL) to flat, perceptually scaled outputs.
///
/// The reference value is 1kHz, and this function solves for a gain summand (in dB) that will
/// "correct" an iso-loud value at another frequency to a perceptually accurate relative value at
/// the target frequency.  If you *add the summand* to the observed dB value, it will become a
/// perceptually mapped value.
///
/// It is advised to subtract some noise floor from the measured values before applying gain because
/// their logs are actually near negative infinity, but tiny amounts of noise in calculations can be
/// combined with very large gain corrections to produce spurious signals after gain that are well
/// above the chosen noise floor.
pub fn iso226_gain(freq: f32) -> Result<f32, utate::MutateError> {
    let ref_spl = iso226_phon2spl(1000.0);
    let target_spl = iso226_phon2spl(freq);

    // Convert this to a correction, how much gain should be applied relative to 1kHz to obtain a
    // value relative to the iso-loud value on the 70 phon curve?
    Ok(ref_spl - target_spl)
}

// This is the C implementation without indirection.
fn iso226_phon2spl(freq: f32) -> f32 {
    let (af, tf, lu) = interpolate_table(freq);

    let af_part = (0.4 * 10f32.powf(((tf + lu) / 10.0) - 9.0)).powf(af);
    let af_total = 4.47 / 1000.0 * (10f32.powf(0.025 * CURVE_PHONS) - 1.15) + af_part;

    ((10.0 / af) * af_total.log10()) - lu + 94.0
}

/// Interpolate ISO226 table values to return constants.  Return values are (AF, TF, and LU), in the
/// same order as the source table columns.
fn interpolate_table(freq: f32) -> (f32, f32, f32) {
    // Clamp frequencies outside the table to the first/last table values
    if freq <= FREQ[0] {
        return (AF[0], TF[0], LU[0]);
    }
    if freq >= FREQ[28] {
        return (AF[AF.len() - 1], TF[TF.len() - 1], LU[LU.len() - 1]);
    }

    // Find the interval [FREQ[i], FREQ[i+1]] that contains freq
    for i in 0..28 {
        if freq >= FREQ[i] && freq < FREQ[i + 1] {
            let k = (freq - FREQ[i]) / (FREQ[i + 1] - FREQ[i]);
            let af_interp = (AF[i + 1] - AF[i]) * k + AF[i];
            let tf_interp = (TF[i + 1] - TF[i]) * k + TF[i];
            let lu_interp = (LU[i + 1] - LU[i]) * k + LU[i];
            return (af_interp, tf_interp, lu_interp);
        }
    }
    unreachable!()
}

// 🤖 The numbers have been eyeballed in a test.  Figures were transformed via LLM and the
// implementation was checked against libiso226 C implementation.

const FREQ: [f32; 29] = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
    500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
    8000.0, 10000.0, 12500.0,
];

const AF: [f32; 29] = [
    0.635, 0.602, 0.569, 0.537, 0.509, 0.482, 0.456, 0.433, 0.412, 0.391, 0.373, 0.357, 0.343,
    0.330, 0.320, 0.311, 0.303, 0.300, 0.295, 0.292, 0.290, 0.290, 0.289, 0.289, 0.289, 0.293,
    0.303, 0.323, 0.354,
];

const LU: [f32; 29] = [
    -31.5, -27.2, -23.1, -19.3, -16.1, -13.1, -10.4, -8.2, -6.3, -4.6, -3.2, -2.1, -1.2, -0.5, 0.0,
    0.4, 0.5, 0.0, -2.7, -4.2, -1.2, 1.4, 2.3, 1.0, -2.3, -7.2, -11.2, -10.9, -3.5,
];

const TF: [f32; 29] = [
    78.1, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0,
    2.2, 2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3,
];

#[cfg(test)]
pub mod test {

    use super::*;

    const TOLERANCE: f32 = 0.001;

    #[test]
    fn test_iso226_curve() {
        let at_20_hz = iso226_gain(20.0).unwrap();
        let at_1000hz = iso226_gain(1000.0).unwrap();

        // The iso-loud SPL for 20Hz is about 45dB above the SPL at 1kHz.
        assert!(((at_1000hz - at_20_hz) - 45.0).abs() < 5.0);

        // 1/3 octaves, just good enough resolution to eyeball the numbers
        let freqs = std::iter::successors(Some(20.0), |&f| Some(f * 2f32.powf(1.0 / 3.0)))
            .take_while(|&f| f <= 24_000.0)
            .collect::<Vec<f32>>();

        // print these if you need to re-calibrate
        for freq in &freqs {
            println!(
                "freq: {:10.0} gain dB: {:10.6}",
                freq,
                iso226_gain(*freq).unwrap()
            );
        }

        // Eyeball confirmed that these values appear to match the difference required to convert
        // the target frequency SPL to a value that is instead relative to the iso-loud SPL on the
        // 70 phon curve.  If we get 80dB at 20Hz and 60dB at 40Hz, the 60dB will produce a larger
        // perceived volume, and this correction will approximate that.
        let expected: [f32; 32] = [
            -42.125954, -37.151352, -32.474823, -28.164536, -24.230034, -20.573692, -17.254478,
            -14.266739, -11.596863, -9.248718, -7.047577, -5.185562, -3.613426, -2.271393,
            -1.181274, -0.264160, 0.294724, -0.150620, -2.466782, -3.531448, -0.326263, 2.199837,
            2.970627, 1.579132, -1.993591, -7.124924, -11.449837, -11.762466, -6.538109, -6.538109,
            -6.538109, -6.538109,
        ];

        // Tests
        for (freq, expected) in freqs.iter().zip(expected) {
            let res = iso226_gain(*freq).unwrap(); // IDENTITY for development
            let err = (res - expected).abs() / expected;
            assert!(
                err < TOLERANCE,
                "mismatch: result={} expected={}",
                res,
                expected
            );
        }
    }
}
