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
//! High resolution mother wavelets from which our highly tuned but fundamentally rough N-tap
//! filters are made.  The moments and shape that the stencil and tapering solution attempts to
//! restore are first measured from these high-fidelity inputs, so we try to make them good.
//!
//! Ideally both quick and accurate, but perhaps split between debug and release or offline baking
//! where tradeoffs must be made.  Downstream uses `f64` for several steps, but the stencil itself
//! washes away *unbiased* noise during the reduction to `N` taps.  The final shape-aware rounding
//! to `f32` forgets any inaccuracy accumulated below its own precision.  The final word is **avoid
//! bias.**
//!
//! ## The Implementations
//!
//! Two implementations are contained.  We use them to corroborate the wavelet engineering and
//! verify the precision convergence of our approaches.  The requirement is to establish convergence
//! over the `f32` filter performance within the application.  If numerical precision no longer
//! limits our filter, we can focus on improving the filter we are approximating in N taps.
//!
//! - IFFT
//! - Inverse Fourier integration by direct quadrature
//!
//! Both generators evaluate the same one-sided inverse transform in the same
//! units. `samples_per_period` (`S`) resolves the carrier into samples and *is* the 𝛚 in carrier
//! radians per sample.  The output taps are `ψ/S` and `d/S²`.  The dilation step, where the
//! daughter wavelet sample omega is known, applies both.
//!
//! Because the mother wavelet will only be sampled over a fixed number of periods, there is one
//! normalization.  The expression `1.0 / (2 * periods)` appears in several places and normalizes
//! sample magnitude for the higher pitch of the wavelet necessary to fit the specified `periods`
//! within the IFFT output.  The inverse Fourier integral matches this normalization to create
//! regularity across the module.

use core::f64::consts::{PI, TAU};

use rustfft::{num_complex::Complex64, FftPlanner};

use crate::dsp::wavelet::whatsleft::Accumulator;
use crate::dsp::wavelet::Shape;

// NOTE The doc comments are getting really dense.  While it's fun to play educator, I'd like to
// focus on getting the nomenclature to presume the reader is familiar with a **good** model for
// these problems and focus on getting across the choices in place for key conventions where most
// people choose one or the other formalism.
// NEXT Did not compare any other FFT libraries or the quadrature of the Fourier integral method in
// either speed or accuracy.  Above 2048 resolution (steps per wave), both amplitude and quadrature
// reach convergence.  This convergence sits below f32 accuracy.

/// Use an IFFT to generate time-domain solutions for psi and d.  `periods` sets the
/// pitch. `samples_per_period` determines how much tail survives.
///
// XXX IIRC the final period of the output is inaccurate or doesn't converge well
pub fn morse_half_taps(
    shape: Shape,
    periods: usize,
    samples_per_period: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let half_len = periods * samples_per_period + 1;
    // Also decides the Nyquist
    let n_fft = 2 * periods * samples_per_period;

    // - Samples per period normalizes omega to the target resolution
    // - The carrier is bin `2 * periods`
    let s_per_bin = shape.peak() / (2 * periods) as f64;
    let zeta_per_bin = TAU / (2 * periods) as f64;

    let mut psi_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut d_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];

    for k in 1..n_fft / 2 {
        let s = k as f64 * s_per_bin;
        let mag = s.powf(shape.beta) * (-s.powf(shape.gamma)).exp();
        psi_spectrum[k] = Complex64::new(mag, 0.0);
        d_spectrum[k] = Complex64::new(mag * k as f64 * zeta_per_bin, 0.0);
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut psi_spectrum);
    ifft.process(&mut d_spectrum);

    let norm = 1.0 / (2 * periods) as f64;

    let psi = psi_spectrum[..half_len].iter().map(|x| x * norm).collect();
    let d = d_spectrum[..half_len].iter().map(|x| x * norm).collect();
    (psi, d)
}

/// Tune the inverse Fourier integration knob.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IfiSettings {
    /// Relative log-amplitude cut for the integration limits: e^-64 is ~1.6e-28, below the f64
    /// epsilon of the peak contribution.
    log_tail: f64,
    /// Grid points per e-fold of `s`, resolving the shape's log-domain lobe.
    log_steps: f64,
}

impl Default for IfiSettings {
    fn default() -> Self {
        Self {
            log_tail: 22.0,
            log_steps: 2.0,
        }
    }
}

const BEND_OFFSET: f64 = 1.5;
const BEND_SCALE: f64 = 0.33;

/// Time-domain amplitude at `u` in carrier periods.  Returns the same dimensionless `psi` and `d`
/// that `morse_half_taps` should converge to.
///
///
/// Steepest-descent quadrature of the inverse Fourier integral, carried out in `v = ln rho` where
/// the integrand
///
///     exp(beta v - P e^{gamma v} + i a e^v)
///
/// is entire.  The path is a single line through the saddle, bent by a `tanh` so that its left end
/// is the ray at `arg rho_star` and its right end returns to the real `rho` axis, which is where
/// the cap term needs it to be for the closing arc to vanish.
pub fn morse_tap_at(shape: Shape, u: f64, settings: IfiSettings) -> (Complex64, Complex64) {
    // 🤖 Hugely generated.

    let (beta, gamma) = (shape.beta, shape.gamma);
    let s_peak = shape.peak();
    let peak_pow = s_peak.powf(gamma);
    let a = TAU * u;

    // Continuation in `a` from the real-axis peak.  A cold Newton at large `a` jumps branches;
    // the stage spacing keeps each restart well inside the quadratic basin.
    let mut rho = Complex64::ONE;

    let stages = (a / 4.0).ceil().max(8.0) as usize;
    for k in 1..=stages {
        let a_k = a * k as f64 / stages as f64;
        for _ in 0..3 {
            let pow = rho.powf(gamma);
            let g = beta * (Complex64::ONE - pow) + Complex64::I * a_k * rho;
            let dg = -beta * gamma * pow / rho + Complex64::I * a_k;
            rho -= g / dg;
        }
    }

    let cap = peak_pow * rho.powf(gamma);
    let osc = Complex64::I * a * rho;

    // Curvature at the saddle sets the only length scale on the contour: the bend is placed and
    // shaped in widths, so `log_steps` is samples per width for the core and the turn alike.
    let phi2 = gamma * gamma * cap - osc;
    let width = 1.0 / phi2.norm().sqrt();
    // let h = width / settings.log_steps;
    let h = width.min(BEND_SCALE) / settings.log_steps;

    // The cap term needs `|gamma arg rho| < pi/2` at the right end.  Stopping short of the axis
    // keeps a margin there and leaves the oscillation damped rather than free-running.
    const SECTOR: f64 = 0.35;
    let phi_end = SECTOR * PI / (2.0 * gamma);

    // Below the sector limit the saddle ray already lands where the cap term needs it, so the
    // contour is straight and the bend contributes nothing.
    let swing = (rho.arg() - phi_end).max(0.0);

    let bend = BEND_OFFSET * width;
    let bend_scale = BEND_SCALE;

    let mut psi_re = Accumulator::<f64>::default();
    let mut psi_im = Accumulator::<f64>::default();
    let mut d_re = Accumulator::<f64>::default();
    let mut d_im = Accumulator::<f64>::default();

    // Returns the log-magnitude of the larger of the two terms, carrying the extra `rho_j` that
    // `d` holds, so the caller can walk out to the tail cut.
    let mut tap = |x: f64| {
        let tanh = ((x - bend) / bend_scale).tanh();
        let z = Complex64::new(x, -swing * (1.0 + tanh) * 0.5);
        let dz = Complex64::new(1.0, -swing * (1.0 - tanh * tanh) * 0.5 / bend_scale);

        let delta = beta * z - cap * ((gamma * z).exp() - Complex64::ONE)
            + osc * (z.exp() - Complex64::ONE);

        let mag = delta.re + x.max(2.0 * x);

        if !(mag > -settings.log_tail) {
            return mag;
        }

        let rho_j = rho * z.exp();
        let term = delta.exp() * (h * dz) * rho_j;
        let dv = term * rho_j * TAU;

        psi_re.add(term.re);
        psi_im.add(term.im);
        d_re.add(dv.re);
        d_im.add(dv.im);

        mag
    };

    tap(0.0);

    let mut j = 1;
    while tap(j as f64 * h) > -settings.log_tail {
        j += 1;
    }
    let mut j = 1;
    while tap(-(j as f64) * h) > -settings.log_tail {
        j += 1;
    }

    let ln_scale = beta * rho.ln() - cap + osc;
    let scale = ln_scale.exp() * s_peak.powf(beta);

    let psi = Complex64::new(psi_re.sum(), psi_im.sum()) * scale;
    let d = Complex64::new(d_re.sum(), d_im.sum()) * scale;

    (psi, d)
}

#[cfg(test)]
mod test {

    use super::super::whatsleft::Accumulator;
    use super::*;

    // These tests are mostly print tests used to calibrate, fill in empirical values, and design
    // the accuracy of the wavelet.
    // NEXT feature rule?

    // Simpson's rule integration, using a high-precision accumulator.
    fn simpson_weighted_sum(vals: &[Complex64], i0: usize, i1: usize) -> Complex64 {
        let n = i1 - i0;
        debug_assert!(n % 2 == 0);
        let mut real: Accumulator<f64> = Accumulator::default();
        let mut imag: Accumulator<f64> = Accumulator::default();

        // Initialize with the endpoints
        real.add(vals[i0].re);
        real.add(vals[i1].re);
        imag.add(vals[i0].im);
        imag.add(vals[i1].im);

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
        let (mantissa, exp) = s.split_once('e').unwrap_or(("999", "999"));
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
    fn ifft_amplitude_convergence() {
        // Render amplitudes at increasing precision and then look for convergence in values. When
        // the implementation is out of precision juice, the numbers should begin to oscillate at
        // low amplitude.  Test condition is that error is small, agreement high.

        // NEXT compare convergence at Nth periods and convergence at different phases.  Quadrature
        // is broad spectrum.  Probing amplitude shows us the floor of what we're integrating.

        let shape = Shape::from_q(3.5, 3.0);
        const MIN_RES_EXP: u32 = 8;
        const FULL_RES_EXP: u32 = 17;
        const PERIODS: usize = 8;
        const FULL_RES: usize = 1 << FULL_RES_EXP;

        // Snap the requested point onto the coarsest transform's sample grid.  Resulting pitch is
        // `1 / MIN_RES` and every finer resolution in the sweep is a power-of-two refinement of it:
        // The oracle is then given the snapped `u` multiplied by periods per sample, `min_s`.
        let min_s = 1.0 / (1usize << MIN_RES_EXP) as f64;
        let u = 1.25;
        let probe_bin = (u / min_s).round() as usize;
        let u = probe_bin as f64 * min_s;

        let (psi, d) = morse_half_taps(shape, PERIODS, FULL_RES);
        let psi0 = psi[probe_bin << (FULL_RES_EXP - MIN_RES_EXP)];
        let d0 = d[probe_bin << (FULL_RES_EXP - MIN_RES_EXP)];

        // The integral method is used here as an "oracle", but it's more of a second opinion,
        // corroborating evidence.
        let settings = IfiSettings::default();
        let (oracle_psi, oracle_d) = morse_tap_at(shape, u, settings);
        let oracle_psi_pct = (oracle_psi - psi0) / psi0 * 100.0;
        let oracle_d_pct = (oracle_d - d0) / d0 * 100.0;
        println!(
            "oracle:            psi: {oracle_psi:+8.7} (𝛅% {:>10} {:>10}) d: {oracle_d:+8.7} (𝛅% {:>10} {:>10})\n",
            fmt_e(oracle_psi_pct.re), fmt_e(oracle_psi_pct.im),
            fmt_e(oracle_d_pct.re), fmt_e(oracle_d_pct.im),
        );

        for i in MIN_RES_EXP..FULL_RES_EXP {
            let resolution = 1usize << i;
            let (psi, d) = morse_half_taps(shape, PERIODS, resolution);
            let index = probe_bin << (i - MIN_RES_EXP);

            let tap_psi = psi[index];
            let tap_d = d[index];
            let delta_psi_pct = (tap_psi - psi0) / psi0 * 100.0;
            let delta_d_pct = (tap_d - d0) / d0 * 100.0;

            println!(
                "resolution: {resolution:>5}, psi: {tap_psi:+8.7} (𝛅% {:>10} {:>10}) d: {tap_d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }
    }

    #[test]

        let shape = Shape::from_q(4.5, 3.0);
        const ROWS: u32 = 12;
        const PERIODS: usize = 16;

        // High res reference settings, high enough to get post-convergence in all dimensions, not
        // too high to start suffering numerical conditioning issues.
        let high_res = IfiSettings {
            log_tail: 20.0,
            log_steps: 8.0,
        };

        let u: f64 = 2.0; // One and a third carrier periods
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);

        println!("=== Cranking log tail === u: {u}");
        let mut settings = IfiSettings::default();
        settings.log_steps = 6.0;
        for i in 0..ROWS {
            let mut settings = settings.clone();
            let log_tail = 2 * i + 2;
            settings.log_tail = log_tail as f64;
            let (psi, d) = morse_tap_at(shape, u, settings);

            let delta_psi_pct = (psi - psi_ref) / psi_ref * 100.0;
            let delta_d_pct = (d - d_ref) / d_ref * 100.0;

            println!(
                "  log tail: {log_tail:>5}, psi: {psi:+8.7} (𝛅% {:>10} {:>10}) d: {d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }

        let u: f64 = 2.33; // One and a third carrier periods
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);

        println!("=== Cranking log tail === u: {u}");
        for i in 0..ROWS {
            let mut settings = settings.clone();
            let log_tail = 2 * i + 2;
            settings.log_tail = log_tail as f64;
            let (psi, d) = morse_tap_at(shape, u, settings);

            let delta_psi_pct = (psi - psi_ref) / psi_ref * 100.0;
            let delta_d_pct = (d - d_ref) / d_ref * 100.0;

            println!(
                "  log tail: {log_tail:>5}, psi: {psi:+8.7} (𝛅% {:>10} {:>10}) d: {d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }

        let u: f64 = 4.7; // One and a third carrier periods
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);

        println!("=== Cranking log tail === u: {u}");
        for i in 0..ROWS {
            let mut settings = settings.clone();
            let log_tail = 2 * i + 2;
            settings.log_tail = log_tail as f64;
            let (psi, d) = morse_tap_at(shape, u, settings);

            let delta_psi_pct = (psi - psi_ref) / psi_ref * 100.0;
            let delta_d_pct = (d - d_ref) / d_ref * 100.0;

            println!(
                "  log tail: {log_tail:>5}, psi: {psi:+8.7} (𝛅% {:>10} {:>10}) d: {d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }

        let u: f64 = 8.2; // One and a third carrier periods
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);

        println!("=== Cranking log tail === u: {u}");
        for i in 0..ROWS {
            let mut settings = settings.clone();
            let log_tail = 2 * i + 2;
            settings.log_tail = log_tail as f64;
            let (psi, d) = morse_tap_at(shape, u, settings);

            let delta_psi_pct = (psi - psi_ref) / psi_ref * 100.0;
            let delta_d_pct = (d - d_ref) / d_ref * 100.0;

            println!(
                "  log tail: {log_tail:>5}, psi: {psi:+8.7} (𝛅% {:>10} {:>10}) d: {d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }

        let mut settings = IfiSettings::default();
        settings.log_tail = 18.0;

        let u: f64 = 0.8; // One and a third carrier periods
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);

        println!("\n=== Cranking log steps === u: {u}");
        for i in 0..ROWS {
            let mut settings = settings.clone();
            let log_steps = (1 + i) as f64 * 0.5;
            settings.log_steps = log_steps as f64;
            let (psi, d) = morse_tap_at(shape, u, settings);

            let delta_psi_pct = (psi - psi_ref) / psi_ref * 100.0;
            let delta_d_pct = (d - d_ref) / d_ref * 100.0;

            println!(
                "  log steps: {log_steps:>5}, psi: {psi:+8.7} (𝛅% {:>10} {:>10}) d: {d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }

        let u: f64 = 3.7; // One and a third carrier periods
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);

        println!("\n=== Cranking log steps === u: {u}");
        for i in 0..ROWS {
            let mut settings = settings.clone();
            let log_steps = (1 + i) as f64 * 0.5;
            settings.log_steps = log_steps as f64;
            let (psi, d) = morse_tap_at(shape, u, settings);

            let delta_psi_pct = (psi - psi_ref) / psi_ref * 100.0;
            let delta_d_pct = (d - d_ref) / d_ref * 100.0;

            println!(
                "  log steps: {log_steps:>5}, psi: {psi:+8.7} (𝛅% {:>10} {:>10}) d: {d:+8.7} (𝛅% {:>10} {:>10})",
                fmt_e(delta_psi_pct.re), fmt_e(delta_psi_pct.im),
                fmt_e(delta_d_pct.re), fmt_e(delta_d_pct.im),
            );
        }

        let u: f64 = 8.33;
        let (psi_ref, d_ref) = morse_tap_at(shape, u, high_res);
    #[test]
    fn ifi_amplitude_convergence() {
        let shape = Shape::from_q(4.5, 3.0);

        let reference = IfiSettings {
            log_tail: 24.0,
            log_steps: 6.5,
        };

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Peak, shoulder, the old sick zone around 3.7-4.7, and deep tail.
        let probes = [0.0, 0.8, 2.0, 2.33, 3.7, 4.7, 5.5, 8.2, 8.33, 11.0, 14.0];

        let sweep = |name: &str, knob: &str, vary: &dyn Fn(u32) -> IfiSettings, rows: u32| {
            println!("\n=== {name} ===");
            print!("  {knob:>10} |");
            for u in probes {
                print!(" {u:>9.2}");
            }
            println!();

            for i in 0..rows {
                let settings = vary(i);
                let label = if knob == "log tail" {
                    settings.log_tail
                } else {
                    settings.log_steps
                };

                print!("  {label:>10.1} |");
                for u in probes {
                    let (psi_ref, d_ref) = morse_tap_at(shape, u, reference);
                    let (psi, d) = morse_tap_at(shape, u, settings);
                    print!(" {:>9}", fmt_e(rel(psi, psi_ref).max(rel(d, d_ref))));
                }
                println!();
            }
        };

        sweep(
            "Cranking log tail",
            "log tail",
            &|i| IfiSettings {
                log_tail: (2 * i + 6) as f64,
                log_steps: 5.0,
            },
            12,
        );

        sweep(
            "Cranking log steps",
            "log steps",
            &|i| IfiSettings {
                log_tail: 24.0,
                log_steps: (1 + i) as f64 * 0.5,
            },
            12,
        );

        // Acceptance: at the settings we intend to ship, the floor is flat in u.  A dense grid so a
        // narrow seam can't hide between probes.
        println!("\n=== Current Defaults ===");
        let settings = IfiSettings::default();
        let mut worst: f64 = 0.0;

        for k in 0..=280 {
            let u = k as f64 * 0.05;
            let (psi_ref, d_ref) = morse_tap_at(shape, u, reference);
            let (psi, d) = morse_tap_at(shape, u, settings);
            let err = rel(psi, psi_ref).max(rel(d, d_ref));
            worst = worst.max(err);

            if k % 10 == 0 {
                println!("  u: {u:>6.2}, err: {:>9}", fmt_e(err));
            }
        }

        println!("  worst over grid: {}", fmt_e(worst));
        assert!(worst < 1e-6);
    }

    #[test]
    fn quadrature_delta() {
        // Simpson's-rule quadrature of psi and d over a half period centered between the 4th and
        // 5th periods. Bounds are quarter-period fractions, so any power-of-two
        // `samples_per_period` >= 8 lands them exactly on a sample index with an even interval
        // count, as Simpson's rule requires.

        let shape = Shape::from_q(3.5, 3.0);
        const FULL_RES_EXP: u32 = 16;
        const FULL_RES: usize = 1 << FULL_RES_EXP; // 2^15 = 32,768

        let (psi, d) = morse_half_taps(shape, 8, FULL_RES);
        let (i0, i1) = (17 * FULL_RES / 4, 21 * FULL_RES / 4);
        let psi0 = simpson_weighted_sum(&psi, i0, i1);
        let d0 = simpson_weighted_sum(&d, i0, i1) * FULL_RES as f64;

        for i in 3..(FULL_RES_EXP - 1) {
            let resolution = 2_i32.pow(i) as usize;
            let (i0, i1) = (17 * resolution / 4, 21 * resolution / 4);
            let (psi, d) = morse_half_taps(shape, 8, resolution);

            let norm_psi = simpson_weighted_sum(&psi, i0, i1);
            let delta_psi_pct = (norm_psi - psi0) / psi0 * 100.0;
            let norm_d = simpson_weighted_sum(&d, i0, i1) * resolution as f64;
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
