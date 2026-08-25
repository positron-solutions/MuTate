// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Generate Wavelets
//!
//! > I am incensed that the features I require have been so thoroughly neglected only so that
//! > this utter slop could be rushed into the hands of mere others who do not share my specific
//! > requirements, and I will hold accountable those responsible for what yesterday I did not know
//! > that I must have.
//! >
//! > - Ttang Kong
//!
//! High resolution mother wavelets, ideally both quickly and accurate, but perhaps split between
//! debug and release where tradeoffs must be made.  Downstream uses `f64` for several steps, but
//! the stencil itself washes away *unbiased* noise during the reduction to f32.  The moments that
//! the stencil attempts to restore

use core::f64::consts::TAU;
use rustfft::{num_complex::Complex64, FftPlanner};

use crate::dsp::wavelet::Shape;

// NEXT Did not compare any other FFT libraries or the quadrature of the Fourier integral method in
// either speed or accuracy.  Above 2048 resolution (steps per wave), both amplitude and quadrature
// reach convergence.  This convergence sits below f32 accuracy.

///  `periods` and `samples_per_period`.
pub fn morse_half_taps(
    shape: Shape,
    periods: usize,
    samples_per_period: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let half_len = periods * samples_per_period + 1;
    let n_fft = 2 * periods * samples_per_period + 1;

    // Carrier at one cycle per `samples_per_period` samples, mapped onto the shape's
    // dimensionless spectral peak via a frequency-axis dilation.
    let target_omega = TAU / samples_per_period as f64;
    let scale = target_omega / shape.peak();

    let mut psi_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut d_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];

    // Fill the input spectrums with a constant magnitude from the shape.
    for k in 1..=n_fft / 2 {
        let omega = k as f64 * TAU / n_fft as f64;
        let omega_norm = omega / scale;
        let mag = omega_norm.powf(shape.beta) * (-omega_norm.powf(shape.gamma)).exp();
        psi_spectrum[k] = Complex64::new(mag, 0.0);
        d_spectrum[k] = Complex64::new(mag * omega, 0.0);
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut psi_spectrum);
    ifft.process(&mut d_spectrum);

    let norm = 1.0 / n_fft as f64;
    let psi = psi_spectrum[..half_len].iter().map(|x| x * norm).collect();
    let d = d_spectrum[..half_len].iter().map(|x| x * norm).collect();
    (psi, d)
}

// Add functionality to shape.
pub fn head_db_periods(tail_db: f64, shape: Shape) -> f64 {
    let q = shape.q();
    let gamma = shape.gamma;
    let periods = q * (tail_db.abs() / 20.0).powf((gamma - 1.0) / gamma);
    periods
}

#[cfg(test)]
mod test {

    use super::super::whatsleft::Accumulator;
    use super::*;

    // Simpson's rule integration, using a high-precision accumulator.
    fn simpson(vals: &[Complex64], i0: usize, i1: usize) -> Complex64 {
        let n = i1 - i0;
        let mut real: Accumulator<f64> = Accumulator::default();
        let mut imag: Accumulator<f64> = Accumulator::default();
        let init = (vals[i0] + vals[i1]);
        real.add(init.re);
        imag.add(init.im);
        for i in i0 + 1..i1 {
            let weight = if (i - i0) % 2 == 1 { 4.0 } else { 2.0 };
            let val = (vals[i] * weight);
            real.add(val.re);
            imag.add(val.im);
        }
        Complex64::new(real.sum(), imag.sum())
    }

    fn fmt_e(x: f64) -> String {
        let s = format!("{x:+.2e}");
        // split "±m.mme±dd" into mantissa and exponent, then zero-pad the exponent
        let (mantissa, exp) = s.split_once('e').unwrap();
        let exp: i32 = exp.parse().unwrap();
        format!("{mantissa}e{exp:+03}")
    }

    #[ignore] // 12.5ms on a Zen2+ core
    #[test]
    fn full_resolution() {
        let shape = Shape::from_q(3.5, 3.0);
        let now = std::time::Instant::now();
        let (d, psi) = morse_half_taps(shape, 32, 1024);

        let elapsed = now.elapsed().as_micros();
        println!("elapsed: {:?}", elapsed);

        const SLOW_MICROS: u128 = 512000;
        assert!(elapsed < SLOW_MICROS, "FFT slow: {}µs ", elapsed);
    }

    #[test]
    fn amplitude_delta() {
        // Render at high precision and then look for the changes in values. When the implementation
        // is out of precision juice, the numbers should begin to oscillate at low amplitude.  This
        // test does not investigate the relationship between quadrature between points, only the
        // value at a specific point (half of the first period).

        let shape = Shape::from_q(3.5, 3.0);
        const FULL_RES_EXP: u32 = 15;
        const FULL_RES: usize = 1 << FULL_RES_EXP; // 2^15 = 32,768
        const HALF_RES: usize = FULL_RES / 2;

        let (psi, d) = morse_half_taps(shape, 8, FULL_RES);
        let psi0 = psi[HALF_RES] * (FULL_RES as f64);
        let d0 = d[HALF_RES] * (FULL_RES as f64) * (FULL_RES as f64);

        for i in 1..=FULL_RES_EXP {
            let resolution = 2_i32.pow(i) as usize;
            let half_res = resolution / 2;
            let (psi, d) = morse_half_taps(shape, 8, resolution);
            let mag_norm = resolution as f64;

            let norm_psi = psi[half_res] * mag_norm;
            let delta_psi_pct = (norm_psi - psi0) / psi0 * 100.0;
            let norm_d = d[half_res] * mag_norm * mag_norm;
            let delta_d_pct = (norm_d - d0) / d0 * 100.0;

            println!(
                "resolution: {resolution:>5}, psi0: {:+8.7} (𝛅% {:>10} {:>10}) d0: {:+8.7} (𝛅% {:>10} {:>10})",
                norm_psi,
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                norm_d,
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }
    }

    #[test]
    fn quadrature_delta() {
        // Simpson's-rule quadrature of psi and d over a half period centered between the 4th and
        // 5th periods. Bounds are quarter-period fractions, so any power-of-two samples_per_period
        // >= 8 lands them exactly on a sample index with an even interval count, as Simpson
        // requires.
        //
        // psi's raw taps carry a 1/samples_per_period scale that a dt = 1/samples_per_period
        // step cancels exactly, so the raw sum is directly comparable across resolutions.
        // d carries 1/samples_per_period^2, so its integral needs one leftover factor of
        // samples_per_period.

        // NOTE At around 1024 resolution, the quadratures we're cooking begin to converge within
        // f32 precision, and no further effort will survive a target-aware f32 rounding.  Later
        // periods are both smaller and more noisy.  This does NOT mean we found the best values that
        // can be found within f32, just that we won't get a different f32 by squeezing f64 harder.

        let shape = Shape::from_q(3.5, 3.0);
        const FULL_RES_EXP: u32 = 16;
        const FULL_RES: usize = 1 << FULL_RES_EXP; // 2^15 = 32,768

        let (psi, d) = morse_half_taps(shape, 8, FULL_RES);
        let (i0, i1) = (17 * FULL_RES / 4, 21 * FULL_RES / 4);
        let psi0 = simpson(&psi, i0, i1);
        let d0 = simpson(&d, i0, i1) * FULL_RES as f64;

        for i in 3..FULL_RES_EXP {
            let resolution = 2_i32.pow(i) as usize;
            let (i0, i1) = (17 * resolution / 4, 21 * resolution / 4);
            let (psi, d) = morse_half_taps(shape, 8, resolution);

            let norm_psi = simpson(&psi, i0, i1);
            let delta_psi_pct = (norm_psi - psi0) / psi0 * 100.0;
            let norm_d = simpson(&d, i0, i1) * resolution as f64;
            let delta_d_pct = (norm_d - d0) / d0 * 100.0;

            println!(
            "resolution: {resolution:>5}, psi0: {:+8.7} (𝛅% {:>10} {:>10}) d0: {:+8.7} (𝛅% {:>10} {:>10})",
            norm_psi,
            fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
            norm_d,
            fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
        );
        }
    }
}
