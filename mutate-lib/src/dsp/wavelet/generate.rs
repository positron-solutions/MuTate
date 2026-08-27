// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Generate Wavelets
//!
//! > I am incensed that the features I require have been so thoroughly neglected only so that
//! > nthis utter slop could be rushed into the hands of mere others who inconsiderately do not share
//! > my specific requirements, and I will hold accountable the selfish cretins responsible for what
//! > yesterday I did not know that I must have.
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
// NEXT DFT tests are the ultimate discriminator and confirmation that we have generated a wavelet,
// but only an untruncated wavelet, which does not exist, can truly give us an answer to whether the
// IFFT or IFI method is actually the oracle.  One may yet bet an imposter. 🥷🏿
// NOTE We're interested in bias, not extra precision.  We can barely measure bias any more
// accurately with extra precision, but a large bias, we have plenty of precision to measure.  Bias
// will show up even after we squeeze the result through a stencil.  Noise will just get washed away
// in the stencil and f32 truncation.
// NEXT Did not compare any other FFT libraries or the quadrature of the Fourier integral method in
// either speed or accuracy.  Above 2048 resolution (steps per wave), both amplitude and quadrature
// reach convergence.  This convergence sits below f32 accuracy.

/// Use an IFFT to generate time-domain solutions for psi, d, and dd.
///
/// The result covers `periods` carrier cycles at `resolution` samples per cycle, so
/// `psi.len() == periods * resolution + 1`.
///
/// `(periods + pad) * resolution` controls IFFT size.
///
/// Each successive array is the `u`-derivative of the one before it up to a quarter turn, so
/// `dpsi/du == i * d` and `dd/du == i * dd`.  Consumers interpolating `psi` or `d` have exact
/// slopes available.  This enables both `psi` and `d` to use Hermitian interpolation.
pub fn morse_half_taps(
    shape: Shape,
    periods: usize,
    pad: usize,
    resolution: usize,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let record = periods + pad;
    let n_fft = record * resolution;
    let half_len = periods * resolution + 1;

    // The carrier is bin `record`, which puts `resolution` samples on a cycle and spans the record
    // over `record` cycles.
    let s_per_bin = shape.peak() / record as f64;
    let zeta_per_bin = TAU / record as f64;

    let mut psi_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut d_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];
    let mut dd_spectrum = vec![Complex64::new(0.0, 0.0); n_fft];

    for k in 1..n_fft / 2 {
        let s = k as f64 * s_per_bin;
        let mag = s.powf(shape.beta) * (-s.powf(shape.gamma)).exp();
        let zeta = k as f64 * zeta_per_bin;
        psi_spectrum[k] = Complex64::new(mag, 0.0);
        d_spectrum[k] = Complex64::new(mag * zeta, 0.0);
        dd_spectrum[k] = Complex64::new(mag * zeta * zeta, 0.0);
    }

    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);
    ifft.process(&mut psi_spectrum);
    ifft.process(&mut d_spectrum);
    ifft.process(&mut dd_spectrum);

    // The quadrature weight `Δs` in units of the peak.  It tracks the record, so a longer record
    // means more bins of proportionally smaller weight and the amplitude is invariant under `pad`.
    let norm = 1.0 / record as f64;

    let sample = |spectrum: &[Complex64]| {
        (0..half_len)
            .map(|i| spectrum[i % n_fft] * norm)
            .collect::<Vec<_>>()
    };

    (
        sample(&psi_spectrum),
        sample(&d_spectrum),
        sample(&dd_spectrum),
    )
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
        let _ = morse_half_taps(shape, 16, 8, 512);

        let elapsed = now.elapsed().as_micros();
        println!("elapsed: {:?}", elapsed);

        const SLOW_MICROS: u128 = 512000;
        assert!(elapsed < SLOW_MICROS, "FFT slow: {}µs ", elapsed);
    }

    /// Cubic Hermite reconstruction from a tap and its derivative.
    ///
    /// `d` is the `u`-derivative up to a quarter turn (`dpsi/du == i * d`), so the slopes are exact
    /// rather than estimated and the stencil stays at two taps.  Error is O(dt^4 |psi''''|).
    ///
    /// `t` is in tap units; `resolution` converts the exact `u`-derivatives to that spacing.
    fn resample_hermite(
        taps: &[Complex64],
        d: &[Complex64],
        t: f64,
        resolution: usize,
    ) -> Complex64 {
        let floor = t.floor();
        let f = t - floor;
        let i = floor as usize;

        let dt = 1.0 / resolution as f64;
        let m0 = Complex64::I * d[i] * dt;
        let m1 = Complex64::I * d[i + 1] * dt;

        let f2 = f * f;
        let f3 = f2 * f;

        let h00 = 2.0 * f3 - 3.0 * f2 + 1.0;
        let h10 = f3 - 2.0 * f2 + f;
        let h01 = -2.0 * f3 + 3.0 * f2;
        let h11 = f3 - f2;

        taps[i] * h00 + m0 * h10 + taps[i + 1] * h01 + m1 * h11
    }

    #[test]
    fn ifft_amplitude_convergence() {
        // Reconstruct taps at arbitrary `u` by Hermite resampling against the IFI oracle.  Errors
        // are normalized to the oracle's amplitude at the nearest half-phase, so a column tracks
        // the local extrema.
        //
        // Relative error against a decaying signal is only meaningful while that signal stands
        // above the IFFT's absolute roundoff, which is `n_fft * EPSILON * peak` because the
        // transform sums `n_fft / 2` bins coherently.  Cells below that are masked rather than
        // printed, since they measure the floor and not the grid.

        let shape = Shape::from_q(3.5, 3.0);
        let oracle = IfiSettings {
            log_tail: 24.0,
            log_steps: 5.0,
        };

        // Amplitude has to clear the roundoff floor by this much before a cell reports.
        const LIVE: f64 = 16.0;
        const TOL: f64 = 1e-6;

        let rel = |v: Complex64, r: Complex64, scale: f64| (v - r).norm() / scale;

        // Snap to a half period so a probe is scored against the extremum it sits nearest.
        let local_scale = |u: f64| {
            let anchor = (u * 2.0).round() / 2.0;
            let (psi_a, d_a) = morse_tap_at(shape, anchor, oracle);
            (psi_a.norm(), d_a.norm())
        };

        // Deliberately off-grid at every resolution in the sweep.
        let probes = [
            0.37, 1.31, 2.66, 3.42, 4.19, 4.77, 5.31, 5.88, 6.42, 6.94, 7.54,
        ];

        let refs: Vec<_> = probes
            .iter()
            .map(|&u| (morse_tap_at(shape, u, oracle), local_scale(u)))
            .collect();

        let peak = refs[0].1 .0;

        print!("\n  {:>10} |", "decay");
        for &(_, (ps, _)) in &refs {
            print!(" {:>9}", fmt_e(ps / peak));
        }
        println!();

        #[derive(Clone, Copy)]
        enum Tap {
            Psi,
            D,
        }

        let sweep = |tap: Tap,
                     name: &str,
                     knob: &str,
                     vary: &dyn Fn(u32) -> (usize, usize, usize),
                     rows: u32| {
            let label_tap = match tap {
                Tap::Psi => "psi",
                Tap::D => "d",
            };
            println!("\n=== {name} ({label_tap}) ===");
            print!("  {knob:>10} |");
            for u in probes {
                print!(" {u:>9.2}");
            }
            println!();

            for i in 0..rows {
                let (periods, pad, resolution) = vary(i);
                let label = match knob {
                    "periods" => periods,
                    "pad" => pad,
                    "record" => periods + pad,
                    _ => resolution,
                };
                let (psi, d, dd) = morse_half_taps(shape, periods, pad, resolution);
                let (value, slope) = match tap {
                    Tap::Psi => (&psi, &d),
                    Tap::D => (&d, &dd),
                };
                let reach = (psi.len() - 1) as f64 / resolution as f64;
                let n_fft = ((periods + pad) * resolution) as f64;
                let noise = n_fft.log2().sqrt() * f64::EPSILON * peak;

                print!("  {label:>10} |");
                for (&u, &((psi_ref, d_ref), (ps, ds))) in probes.iter().zip(&refs) {
                    let (reference, scale) = match tap {
                        Tap::Psi => (psi_ref, ps),
                        Tap::D => (d_ref, ds),
                    };

                    // Skip cells that won't have an index due to being too short.
                    if u >= reach {
                        print!(" {:>9}", "-");
                        continue;
                    }

                    let t = u * resolution as f64;
                    let cell = format!(
                        "{:>9}",
                        fmt_e(rel(
                            resample_hermite(value, slope, t, resolution),
                            reference,
                            scale
                        ))
                    );

                    // LIES When the predicted IFFT noise floor is higher than the scale of the
                    // features we are attempting to draw.  The prediction may be loose or we may
                    // just be truly above the noise floor in average cases.  Leaving this here in
                    // case this failure mode is encountered.
                    if scale < noise * LIVE {
                        print!(" \x1b[2;90m{cell}\x1b[0m");
                    } else {
                        print!(" {cell}");
                    }
                }

                println!();
            }
        };

        // Each sweep is over-provisioned in the knobs it isn't varying, so the only thing that can
        // bind is the one in the label.
        for tap in [Tap::Psi, Tap::D] {
            // Drive up the FFT grid points per period, resulting in finer sampling of the wavelet
            // in the time domain.
            sweep(
                tap,
                "Cranking resolution",
                "resolution",
                &|i| (8, 56, 1usize << (i + 5)),
                8,
            );

            // Add padding to constant period count.
            sweep(
                tap,
                "Cranking pad",
                "pad",
                &|i| (8, (1usize << i) - 1, 1 << 8),
                10,
            );

            // Fixed reach with a growing transform.  Once pad clears truncation the rows go flat
            // and stay flat across seven doublings of `n_fft`, so the floor doesn't scale with
            // transform size the way a coherent-sum bound would predict.  Extra record past that
            // point is free and useless.
            sweep(
                tap,
                "Cranking record",
                "record",
                &|i| (6, (1usize << (i + 3)) - 6, 1 << 8),
                10,
            );
        }

        println!("\n=== Shipping Grid ===");
        let (periods, pad, resolution) = (6, 26, 1usize << 10);
        let (psi, d, dd) = morse_half_taps(shape, periods, pad, resolution);
        let n_fft = ((periods + pad) * resolution) as f64;
        let noise = n_fft.log2().sqrt() * f64::EPSILON * peak;
        let reach = (psi.len() - 1) as f64 / resolution as f64;

        let mut worst = [(0.0f64, 0.0f64); 2];
        let mut live_to = [0.0f64; 2];

        println!("  {:>6} | {:>9} {:>9} | {:>9}", "u", "psi", "d", "decay");

        let steps = ((reach - 0.011) / 0.05) as u32;

        for k in 0..steps {
            let u = k as f64 * 0.05 + 0.011;
            let t = u * resolution as f64;
            let (psi_ref, d_ref) = morse_tap_at(shape, u, oracle);
            let (ps, ds) = local_scale(u);

            let e = [
                rel(resample_hermite(&psi, &d, t, resolution), psi_ref, ps),
                rel(resample_hermite(&d, &dd, t, resolution), d_ref, ds),
            ];
            let live = [ps > noise * LIVE, ds > noise * LIVE];

            for j in 0..2 {
                if !live[j] {
                    continue;
                }
                live_to[j] = u;
                if e[j] > worst[j].0 {
                    worst[j] = (e[j], u);
                }
            }

            if k % 10 == 0 {
                println!(
                    "  {u:>6.2} | {:>9} {:>9} | {:>9}",
                    fmt_e(e[0]),
                    fmt_e(e[1]),
                    fmt_e(ps / peak)
                );
            }
        }

        println!("\n  noise floor {} at peak scale", fmt_e(noise / peak));
        for (j, name) in ["psi", "d"].iter().enumerate() {
            println!(
                "  {name:>3}: worst {} at u {:.3} ({:.0}x under tol), live to u {:.2}",
                fmt_e(worst[j].0),
                worst[j].1,
                TOL / worst[j].0,
                live_to[j]
            );
        }

        for j in 0..2 {
            assert!(worst[j].0 < TOL);
            assert!(live_to[j] > 5.5);
        }
    }

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

        // NEXT Using dd, we can interpolate and get to convergence a lot faster
        let (psi, d, _) = morse_half_taps(shape, 8, 8, FULL_RES);

        let (i0, i1) = (17 * FULL_RES / 4, 21 * FULL_RES / 4);
        let psi0 = simpson_weighted_sum(&psi, i0, i1);
        let d0 = simpson_weighted_sum(&d, i0, i1) * FULL_RES as f64;

        for i in 3..(FULL_RES_EXP - 1) {
            let resolution = 2_i32.pow(i) as usize;
            let (i0, i1) = (17 * resolution / 4, 21 * resolution / 4);
            let (psi, d, _) = morse_half_taps(shape, 8, 8, resolution);

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
