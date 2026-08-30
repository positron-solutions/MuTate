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

use core::f64::consts::{FRAC_PI_2, LN_2, PI, TAU};
use std::ops::Div;

use rustfft::{num_complex::Complex64, FftPlanner};

use crate::dsp::wavelet::whatsleft::Accumulator;
use crate::dsp::wavelet::Shape;

// NOTE As in other files, `𝐮` is the normalized time coordinate, `𝛚𝐭`, periods at the wavelet's
// angular velocity.

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
// NEXT Did not compare any other FFT libraries, just went with stock standard.

/// Grid for the IFFT generator.
///
/// `periods` is the reach of the returned half-taps in carrier cycles; `pad` extends the record
/// past that reach so the periodic wrap lands in the decayed tail; `resolution` is samples per
/// carrier cycle.  Only `record` and `n_fft` are derived, so no call site recomputes them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IfftSettings {
    pub periods: usize,
    pub pad: usize,
    pub resolution: usize,
}

impl Default for IfftSettings {
    /// Past both elbows in `ifft_precision_convergence` and `ifft_quadrature_precision`: pad is
    /// into the flat rows around 1e-14 at the near lobes, and resolution 1024 puts Hermite
    /// interpolation at ~2e-12 where the truncation floor no longer binds.
    fn default() -> Self {
        Self {
            periods: 8,
            pad: 26,
            resolution: 1 << 10,
        }
    }
}

impl IfftSettings {
    /// Periods + padding periods
    pub fn record(self) -> usize {
        self.periods + self.pad
    }

    /// Dimensions of the IFFT that will result
    pub fn n_fft(self) -> usize {
        self.record() * self.resolution
    }

    /// Folded legth in taps
    pub fn half_len(self) -> usize {
        self.periods * self.resolution + 1
    }

    /// Reach of the returned taps in `u`, i.e. the largest `u` a resample can index.
    pub fn reach(self) -> f64 {
        (self.half_len() - 1) as f64 / self.resolution as f64
    }

    /// Post-elbow reference settings.  Additional precision will buy only noise or regression.
    pub fn reference() -> Self {
        Self {
            periods: 8,
            pad: 34,
            resolution: 1 << 11,
        }
    }
}

/// Use an IFFT to generate time-domain solutions for psi, d, and dd.
///
/// The result covers `settings.periods` carrier cycles at `settings.resolution` samples per cycle,
/// so `psi.len() == settings.half_len()`.
///
/// `settings.n_fft()` is the transform size.
///
/// Each successive array is the `u`-derivative of the one before it up to a quarter turn, so
/// `dpsi/du == i * d` and `dd/du == i * dd`.  Consumers interpolating `psi` or `d` have exact
/// slopes available.  This enables both `psi` and `d` to use Hermitian interpolation.
pub(crate) fn morse_half_taps(
    shape: Shape,
    settings: IfftSettings,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let record = settings.record() as f64;
    let half_len = settings.half_len();
    let resolution = settings.resolution as f64;

    let s_per_bin = shape.peak() / record;
    let zeta_per_bin = TAU / record;

    // Integer binade shift placing the spectral peak in [1, 2).  Exponent-only, so folding it
    // back out through `norm` restores the mantissa exactly.
    let peak = shape.peak();
    let shift = -((shape.beta * peak.ln() - peak.powf(shape.gamma)) / LN_2).floor();
    let norm = (-shift).exp2() / record;

    // The envelope underflows well before the Nyquist bin, so the significant support is far
    // shorter than the transform would be.  Collect it once and reuse across all samples.
    let bins: Vec<[f64; 4]> = (1..settings.n_fft() / 2)
        .map(|k| {
            let s = k as f64 * s_per_bin;
            let mag = ((shape.beta * s.ln() - s.powf(shape.gamma)) / LN_2 + shift).exp2();
            let zeta = k as f64 * zeta_per_bin;
            [k as f64, mag, mag * zeta, mag * zeta * zeta]
        })
        .filter(|b| b[1] > 0.0)
        .collect();

    let mut psi = Vec::with_capacity(half_len);
    let mut d = Vec::with_capacity(half_len);
    let mut dd = Vec::with_capacity(half_len);

    for i in 0..half_len {
        let u = i as f64 / resolution;
        let mut acc = [0.0f64; 6];
        let mut comp = [0.0f64; 6];

        for &[k, w_psi, w_d, w_dd] in &bins {
            let (sin, cos) = (TAU * frac_turns(k, u, record)).sin_cos();
            for (j, (w, trig)) in [
                (w_psi, cos),
                (w_psi, sin),
                (w_d, cos),
                (w_d, sin),
                (w_dd, cos),
                (w_dd, sin),
            ]
            .into_iter()
            .enumerate()
            {
                let x = w * trig;
                let x_err = w.mul_add(trig, -x);
                let t = acc[j] + x;
                comp[j] += (acc[j] - (t - x)) + (x - (t - acc[j])) + x_err;
                acc[j] = t;
            }
        }

        let mut out =
            |j: usize| Complex64::new((acc[j] + comp[j]) * norm, (acc[j + 1] + comp[j + 1]) * norm);
        psi.push(out(0));
        d.push(out(2));
        dd.push(out(4));
    }

    (psi, d, dd)
}

/// Fractional turns of `k*u/record`, carrying the low half of the product.
///
/// The phase reaches tens of thousands of radians at high `u`; reducing in turns before
/// scaling by TAU keeps the argument reduction out of `sin_cos`.
#[inline]
fn frac_turns(k: f64, u: f64, record: f64) -> f64 {
    let p_hi = k * u;
    let p_lo = k.mul_add(u, -p_hi);
    let q_hi = p_hi / record;
    let q_lo = ((-q_hi).mul_add(record, p_hi) + p_lo) / record;
    (q_hi - q_hi.floor()) + q_lo
}

#[inline]
fn two_sum_into(acc: &mut f64, comp: &mut f64, x: f64) {
    let t = *acc + x;
    *comp += (*acc - (t - x)) + (x - (t - *acc));
    *acc = t;
}

/// Tune the inverse Fourier integration knob.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IfiSettings {
    /// Relative log-amplitude cut for the integration limits: e^-64 is ~1.6e-28, below the f64
    /// epsilon of the peak contribution.
    log_tail: f64,
    /// Grid points per e-fold of `s`, resolving the shape's log-domain lobe.  Middle `u`, going
    /// into the tail starts to sag in accuracy at lower settings.
    log_steps: f64,
}

impl Default for IfiSettings {
    fn default() -> Self {
        Self {
            log_tail: 32.0,
            log_steps: 4.0,
        }
    }
}

/// Dominant saddle: the root continuously connected to `rho = 1` at `lambda = 0`, which is the
/// one of largest real part throughout.  The endpoint root sits near `i / lambda` and the rest
/// are rotated into the left half-plane, so the selection never becomes ambiguous.
fn saddle(shape: Shape, a: f64) -> Complex64 {
    let gamma = shape.gamma as usize;
    let mut rho = saddle_set(gamma, a / shape.beta)[..gamma]
        .iter()
        .copied()
        .fold(Complex64::new(f64::NEG_INFINITY, 0.0), |best, r| {
            if r.re > best.re {
                r
            } else {
                best
            }
        });

    // Durand-Kerner leaves a few ulp; polish in `v = ln rho` so the exponent the contour is
    // built from is stationary to full precision.
    let (beta, gamma) = (shape.beta, shape.gamma);
    let mut v = rho.ln();
    for _ in 0..3 {
        let (pow, lin) = ((gamma * v).exp(), v.exp());
        let g = beta * (Complex64::ONE - pow) + Complex64::I * a * lin;
        let dg = -beta * gamma * pow + Complex64::I * a * lin;
        v -= g / dg;
    }
    rho = v.exp();

    rho
}

/// Saddles of the normalized exponent: roots of `rho^gamma - i*lambda*rho - 1`,
/// `lambda = a / beta`. Durand-Kerner, which needs no branch choice for integer `gamma`.
fn saddle_set(gamma: usize, lambda: f64) -> [Complex64; 4] {
    let p = |r: Complex64| r.powi(gamma as i32) - Complex64::I * lambda * r - Complex64::ONE;

    let radius = 1.0 + lambda.powf(1.0 / (gamma - 1) as f64);
    let mut z = [Complex64::ONE; 4];
    for k in 0..gamma {
        let t = TAU * k as f64 / gamma as f64 + 0.4;
        z[k] = Complex64::from_polar(radius, t);
    }

    for _ in 0..60 {
        for k in 0..gamma {
            let mut denom = Complex64::ONE;
            for j in 0..gamma {
                if j != k {
                    denom *= z[k] - z[j];
                }
            }
            z[k] -= p(z[k]) / denom;
        }
    }
    z
}

/// Log-distance from the dominant saddle to its nearest neighbor, measured in the
/// contour's own `v = ln rho`.
fn neighbor_gap(gamma: usize, lambda: f64, rho: Complex64) -> f64 {
    saddle_set(gamma, lambda)[..gamma]
        .iter()
        .map(|r| (r / rho).ln().norm())
        .filter(|d| *d > 1e-6)
        .fold(f64::INFINITY, f64::min)
}

/// Time-domain amplitude at `u` in carrier periods.  Returns the same dimensionless `psi`, `d`, and
/// `dd` that `morse_half_taps` should converge to.
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
pub(crate) fn morse_tap_at(
    shape: Shape,
    u: f64,
    settings: IfiSettings,
) -> (Complex64, Complex64, Complex64) {
    // 🤖 Hugely generated.

    const SECTOR: f64 = 0.35;
    const BEND_OFFSET: f64 = 2.8;
    const BEND_SCALE: f64 = 0.60;

    let (beta, gamma) = (shape.beta, shape.gamma);

    let s_peak = shape.peak();
    let peak_pow = s_peak.powf(gamma);
    let a = TAU * u;

    let mut rho = saddle(shape, a);

    let cap = peak_pow * rho.powf(gamma);
    let osc = Complex64::I * a * rho;

    let phi2 = gamma * gamma * cap - osc;
    let phi3 = osc - gamma * gamma * gamma * cap;
    let gap = neighbor_gap(gamma as usize, a / beta, rho);

    // Away from a coalescence the curvature is the only scale and this is the old
    // `1/sqrt(|phi2|)`.  Where two saddles graze, `phi2` softens and the cubic scale takes
    // over, which is the Airy width without the Airy machinery.
    let width = (1.0 / phi2.norm().sqrt()).min((6.0 / phi3.norm()).cbrt());

    let phi_end = SECTOR * PI / (2.0 * gamma);
    let swing = (rho.arg() - phi_end).max(0.0);

    // The turn has to happen before `Re(cap * e^{gamma z})` changes sign, so it is placed and
    // shaped in absolute `v`, never in a saddle-derived scale that can grow.
    let bend_scale = BEND_SCALE;
    let bend = BEND_OFFSET * width.min(bend_scale);

    let h = width.min(bend_scale) / settings.log_steps;

    let mut psi_re = Accumulator::<f64>::default();
    let mut psi_im = Accumulator::<f64>::default();
    let mut d_re = Accumulator::<f64>::default();
    let mut d_im = Accumulator::<f64>::default();
    let mut dd_re = Accumulator::<f64>::default();
    let mut dd_im = Accumulator::<f64>::default();

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
        let step = rho_j * TAU;
        let term = delta.exp() * (h * dz) * rho_j;

        let dv = term * rho_j * TAU;
        let ddv = dv * step;

        psi_re.add(term.re);
        psi_im.add(term.im);
        d_re.add(dv.re);
        d_im.add(dv.im);
        dd_re.add(ddv.re);
        dd_im.add(ddv.im);

        mag
    };

    tap(0.0);

    let margin = (gap.min(2.0 * width) / h).ceil() as usize + 2;

    for dir in [1.0, -1.0] {
        let (mut j, mut below) = (1usize, 0usize);
        while below < margin {
            if tap(dir * j as f64 * h) > -settings.log_tail {
                below = 0;
            } else {
                below += 1;
            }
            j += 1;
        }
    }

    let ln_scale = beta * rho.ln() - cap + osc;
    let scale = ln_scale.exp() * s_peak.powf(beta);

    let psi = Complex64::new(psi_re.sum(), psi_im.sum()) * scale;
    let d = Complex64::new(d_re.sum(), d_im.sum()) * scale;
    let dd = Complex64::new(dd_re.sum(), dd_im.sum()) * scale;

    (psi, d, dd)
}

#[cfg(test)]
mod test {

    use super::super::whatsleft::Accumulator;
    use super::*;

    // These tests are mostly print tests used to calibrate, fill in empirical values, and design
    // the accuracy of the wavelet.
    // NEXT feature rule?

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
        let _ = morse_half_taps(
            shape,
            IfftSettings {
                periods: 16,
                pad: 8,
                resolution: 512,
            },
        );

        let elapsed = now.elapsed().as_micros();
        println!("elapsed: {:?}", elapsed);

        const SLOW_MICROS: u128 = 512000;
        assert!(elapsed < SLOW_MICROS, "FFT slow: {}µs ", elapsed);
    }

    #[inline]
    fn two_diff(a: f64, b: f64) -> (f64, f64) {
        let s = a - b;
        let bv = s - a;
        (s, (a - (s - bv)) + (-b - bv))
    }

    /// Hermite basis in delta form, anchored on the nearer endpoint.
    ///
    /// `a` and `b` are the one-sided second differences; they are the only place cancellation
    /// occurs, and the two-sum residuals are folded back before they reach the Horner chain.
    #[inline]
    fn hermite_1d(p0: f64, p1: f64, m0: f64, m1: f64, f: f64) -> f64 {
        let (delta, delta_err) = two_diff(p1, p0);
        let (a, a_err) = two_diff(delta, m0);
        let (b, b_err) = two_diff(m1, delta);
        let a = a + (a_err + delta_err);
        let b = b + (b_err - delta_err);

        if f <= 0.5 {
            p0 + f * (b - a).mul_add(f, 2.0 * a - b).mul_add(f, m0)
        } else {
            let g = 1.0 - f;
            p1 + g * (a - b).mul_add(g, 2.0 * b - a).mul_add(g, -m1)
        }
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

        let res = resolution as f64;
        let m0 = Complex64::new(-d[i].im / res, d[i].re / res);
        let m1 = Complex64::new(-d[i + 1].im / res, d[i + 1].re / res);

        let p0 = taps[i];
        let p1 = taps[i + 1];

        Complex64::new(
            hermite_1d(p0.re, p1.re, m0.re, m1.re, f),
            hermite_1d(p0.im, p1.im, m0.im, m1.im, f),
        )
    }

    /// Exact integral of the cubic Hermite reconstruction over `[i0, i1]`, in `u`.
    ///
    /// Same stencil as `resample_hermite`, so this measures the area under the curve consumers
    /// will actually see rather than the area under an independent quadrature rule.
    fn hermite_integral(
        vals: &[Complex64],
        d: &[Complex64],
        i0: usize,
        i1: usize,
        resolution: usize,
    ) -> Complex64 {
        let dt = 1.0 / resolution as f64;
        let mut real: Accumulator<f64> = Accumulator::default();
        let mut imag: Accumulator<f64> = Accumulator::default();

        for i in i0..i1 {
            let m0 = Complex64::I * d[i] * dt;
            let m1 = Complex64::I * d[i + 1] * dt;
            let seg = (vals[i] + vals[i + 1]) * 0.5 + (m0 - m1) / 12.0;
            real.add(seg.re);
            imag.add(seg.im);
        }

        Complex64::new(real.sum(), imag.sum()) * dt
    }

    #[test]
    fn ifft_precision_convergence() {
        // Find the grid and padding elbows.  Remaining disagreement in other tests is disagreement
        // about the wavelet rather than disagreement with self. Tests show that padding has the
        // largest effect on accuracy.  Grid size has relatively little influence relative to padding.

        let shape = Shape::from_q(3.5, 3.0);

        let reference = IfftSettings::reference();

        let (ref_psi, ref_d, _) = morse_half_taps(shape, reference);

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Deliberately off-grid at every resolution in the sweep.
        let probes = [
            0.0, 0.37, 0.89, 1.31, 2.07, 2.66, 3.42, 4.19, 5.88, 6.42, 7.911,
        ];

        let refs: Vec<_> = probes
            .iter()
            .map(|&u| {
                resample_hermite(
                    &ref_psi,
                    &ref_d,
                    u * reference.resolution as f64,
                    reference.resolution,
                )
            })
            .collect();

        let sweep =
            |name: &str, knob: &str, vary: &dyn Fn(u32) -> (usize, IfftSettings), rows: u32| {
                println!("\n=== {name} ===");
                print!("  {knob:>10} |");
                for u in probes {
                    print!(" {u:>9.2}");
                }
                println!();

                for i in 0..rows {
                    let (label, settings) = vary(i);
                    let (psi, d, _) = morse_half_taps(shape, settings);

                    print!("  {label:>10} |");
                    for (&u, &psi_ref) in probes.iter().zip(&refs) {
                        let t = u * settings.resolution as f64;
                        let e = rel(resample_hermite(&psi, &d, t, settings.resolution), psi_ref);
                        print!(" {:>9}", fmt_e(e));
                    }
                    println!();
                }
            };

        sweep(
            "Cranking pad",
            "pad",
            &|i| {
                let pad = 2 * i as usize + 2;
                (pad, IfftSettings { pad, ..reference })
            },
            16,
        );
        sweep(
            "Cranking resolution",
            "resolution",
            &|i| {
                let resolution = (i as usize + 1) * 64;
                (
                    resolution,
                    IfftSettings {
                        resolution,
                        ..reference
                    },
                )
            },
            14,
        );

        println!("\n=== Shipping Grid ===");
        let shipping = IfftSettings::default();
        let (psi, d, _) = morse_half_taps(shape, shipping);

        let mut worst: f64 = 0.0;
        let mut worst_u: f64 = 0.0;

        for k in 0..=150 {
            let u = k as f64 * 0.05 + 0.011;
            let e = rel(
                resample_hermite(
                    &psi,
                    &d,
                    u * shipping.resolution as f64,
                    shipping.resolution,
                ),
                resample_hermite(
                    &ref_psi,
                    &ref_d,
                    u * reference.resolution as f64,
                    reference.resolution,
                ),
            );
            if e > worst {
                worst = e;
                worst_u = u;
            }

            if k % 10 == 0 {
                println!("  u: {u:>6.2}, err: {:>9}", fmt_e(e));
            }
        }

        println!("  worst over grid: {} at {:0.2}", fmt_e(worst), worst_u);
        assert!(worst < 1e-5);
    }

    #[test]
    fn ifft_accuracy_convergence() {
        // Reconstruct taps at arbitrary `u` by Hermite resampling against the IFI oracle.  Errors
        // are normalized to the oracle's amplitude at the nearest half-phase, so a column tracks
        // the local extrema.
        //
        // Relative error against a decaying signal is only meaningful while that signal stands
        // above the IFFT's roundoff floor, which the `record` sweep shows is incoherent in `n_fft`
        // rather than linear.  Cells below that are masked rather than printed, since they measure
        // the floor and not the grid.

        let shape = Shape::from_q(3.5, 3.0);
        let oracle = IfiSettings {
            log_tail: 24.0,
            log_steps: 6.5,
        };

        let base = IfftSettings {
            periods: 8,
            pad: 56,
            resolution: 1 << 8,
        };
        // Amplitude has to clear the roundoff floor by this much before a cell reports.
        const LIVE: f64 = 16.0;
        const TOL: f64 = 1e-6;

        let rel = |v: Complex64, r: Complex64, scale: f64| (v - r).norm() / scale;

        // Snap to a half period so a probe is scored against the extremum it sits nearest.
        let local_scale = |u: f64| {
            let anchor = (u * 2.0).round() / 2.0;
            let (psi_a, d_a, _) = morse_tap_at(shape, anchor, oracle);
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

        let sweep =
            |tap: Tap, name: &str, knob: &str, vary: &dyn Fn(u32) -> IfftSettings, rows: u32| {
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
                    let settings = vary(i);
                    let label = match knob {
                        "periods" => settings.periods,
                        "pad" => settings.pad,
                        "record" => settings.record(),
                        _ => settings.resolution,
                    };
                    let (psi, d, dd) = morse_half_taps(shape, settings);
                    let (value, slope) = match tap {
                        Tap::Psi => (&psi, &d),
                        Tap::D => (&d, &dd),
                    };
                    let reach = settings.reach();
                    let floor = (settings.n_fft() as f64).log2().sqrt() * f64::EPSILON * peak;

                    print!("  {label:>10} |");
                    for (&u, &((psi_ref, d_ref, _), (ps, ds))) in probes.iter().zip(&refs) {
                        let (reference, scale) = match tap {
                            Tap::Psi => (psi_ref, ps),
                            Tap::D => (d_ref, ds),
                        };

                        // Skip cells that won't have an index due to being too short.
                        if u >= reach - 1.0 / settings.resolution as f64 {
                            print!(" {:>9}", "-");
                            continue;
                        }

                        let t = u * settings.resolution as f64;
                        let cell = format!(
                            "{:>9}",
                            fmt_e(rel(
                                resample_hermite(value, slope, t, settings.resolution),
                                reference,
                                scale
                            ))
                        );

                        // LIES When the predicted IFFT noise floor is higher than the scale of the
                        // features we are attempting to draw.  The prediction may be loose or we may
                        // just be truly above the noise floor in average cases.  Leaving this here in
                        // case this failure mode is encountered.
                        if scale < floor * LIVE {
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
                &|i| IfftSettings {
                    resolution: 1 << (i + 5),
                    ..base
                },
                8,
            );

            // Add padding to constant period count.
            sweep(
                tap,
                "Cranking pad",
                "pad",
                &|i| IfftSettings {
                    pad: (1usize << i) - 1,
                    ..base
                },
                8,
            );

            // Fixed reach with a growing transform.  Once pad clears truncation the rows go flat
            // and stay flat across seven doublings of `n_fft`, so the floor doesn't scale with
            // transform size the way a coherent-sum bound would predict.  Extra record past that
            // point is free and useless.
            sweep(
                tap,
                "Cranking record",
                "record",
                &|i| IfftSettings {
                    periods: 6,
                    pad: (1usize << (i + 3)) - 6,
                    ..base
                },
                8,
            );
        }

        println!("\n=== Shipping Grid ===");

        let shipping = IfftSettings::default();
        let (psi, d, dd) = morse_half_taps(shape, shipping);
        let noise = (shipping.n_fft() as f64).log2().sqrt() * f64::EPSILON * peak;
        let floor = noise.max((-oracle.log_tail).exp() * peak);
        let reach = shipping.reach();

        let mut worst = [(0.0f64, 0.0f64); 2];
        let mut live_to = [0.0f64; 2];

        println!("  {:>6} | {:>9} {:>9} | {:>9}", "u", "psi", "d", "decay");

        let steps = ((reach - 0.011) / 0.05) as u32;

        for k in 0..steps {
            let u = k as f64 * 0.05 + 0.011;
            if u >= reach {
                print!(" {:>9}", "-");
                continue;
            }

            let t = u * shipping.resolution as f64;
            let (psi_ref, d_ref, _) = morse_tap_at(shape, u, oracle);
            let (ps, ds) = local_scale(u);

            let e = [
                rel(
                    resample_hermite(&psi, &d, t, shipping.resolution),
                    psi_ref,
                    ps,
                ),
                rel(resample_hermite(&d, &dd, t, shipping.resolution), d_ref, ds),
            ];

            let live = [ps > floor * LIVE, ds > floor * LIVE];

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

        println!(
            "\n  floor {} at peak scale (ifft {}, oracle {})",
            fmt_e(floor / peak),
            fmt_e(noise / peak),
            fmt_e((-oracle.log_tail).exp())
        );
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
    fn ifi_precision_convergence() {
        let shape = Shape::from_q(3.5, 3.0);
        let reference = IfiSettings {
            log_tail: 40.0,
            log_steps: 7.0,
        };

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Peak, shoulder, the old sick zone around 3.7-4.7, and deep tail.
        let probes = [
            0.0, 0.8, 2.0, 2.33, 3.7, 4.7, 5.5, 7.5, 8.0, 8.2, 8.33, 11.0, 14.0,
        ];

        #[derive(Clone, Copy)]
        enum Tap {
            Psi,
            D,
            Dd,
        }

        let pick = |t: Tap, (psi, d, dd): (Complex64, Complex64, Complex64)| match t {
            Tap::Psi => psi,
            Tap::D => d,
            Tap::Dd => dd,
        };

        let sweep =
            |tap: Tap, name: &str, knob: &str, vary: &dyn Fn(u32) -> IfiSettings, rows: u32| {
                let label_tap = match tap {
                    Tap::Psi => "psi",
                    Tap::D => "d",
                    Tap::Dd => "dd",
                };
                println!("\n=== {name} ({label_tap}) ===");
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
                        let r = pick(tap, morse_tap_at(shape, u, reference));
                        let v = pick(tap, morse_tap_at(shape, u, settings));
                        print!(" {:>9}", fmt_e(rel(v, r)));
                    }
                    println!();
                }
            };

        for tap in [Tap::Psi, Tap::D, Tap::Dd] {
            sweep(
                tap,
                "Cranking log tail",
                "log tail",
                &|i| IfiSettings {
                    log_tail: (4 * i + 2) as f64,
                    ..reference
                },
                12,
            );
            sweep(
                tap,
                "Cranking log steps",
                "log steps",
                &|i| IfiSettings {
                    log_steps: (1 + i) as f64 * 1.0,
                    ..reference
                },
                12,
            );
        }

        // Acceptance: at the settings we intend to ship, the floor is flat in u.  A dense grid so a
        // narrow seam can't hide between probes.
        println!("\n=== Current Defaults ===");
        let settings = IfiSettings::default();

        let mut worst: f64 = 0.0;
        let mut worst_u: f64 = 0.0;

        for k in 0..=2048 {
            let u = k as f64 * 0.0125;

            let (psi_ref, d_ref, dd_ref) = morse_tap_at(shape, u, reference);
            let (psi, d, dd) = morse_tap_at(shape, u, settings);
            let err = rel(psi, psi_ref).max(rel(d, d_ref)).max(rel(dd, dd_ref));
            if err > worst {
                worst = err;
                worst_u = u;
            }

            if k % 100 == 0 {
                println!("  u: {u:>6.2}, err: {:>9}", fmt_e(err));
            }
        }

        println!("  worst over grid: {} at {:0.2}", fmt_e(worst), worst_u);
        assert!(worst < 1e-9);
    }

    #[test]
    fn ifft_quadrature_precision() {
        // Hermite quadrature of psi over half periods, against an over-provisioned instance of
        // itself.  This is the same stencil `resample_hermite` uses, so a column reports when the
        // stored psi stops carrying the area downstream will reconstruct.
        //
        // Bounds are quarter-period fractions straddling the carrier's zero crossings, so each
        // column is one signed lobe and successive columns alternate sign.  A whole-period window
        // would cancel and report on the cancellation instead of the grid.

        let shape = Shape::from_q(3.5, 3.0);

        let reference = IfftSettings {
            periods: 8,
            pad: 72,
            resolution: 2048,
        };

        // Half periods 0..10, covering the first five carrier periods.
        const LOBES: usize = 10;

        let (ref_psi, ref_d, _) = morse_half_taps(shape, reference);

        let bounds = |k: usize, resolution: usize| {
            ((1 + 2 * k) * resolution / 4, (3 + 2 * k) * resolution / 4)
        };

        let refs: Vec<Complex64> = (0..LOBES)
            .map(|k| {
                let (i0, i1) = bounds(k, reference.resolution);
                hermite_integral(&ref_psi, &ref_d, i0, i1, reference.resolution)
            })
            .collect();

        let sweep = |name: &str, knob: &str, vary: &dyn Fn(u32) -> IfftSettings, rows: u32| {
            println!("\n=== {name} ===");
            print!("  {knob:>10} |");
            for k in 0..LOBES {
                print!(" {:>9.2}", 0.25 + k as f64 * 0.5);
            }
            println!();

            for i in 0..rows {
                let settings = vary(i);
                let label = if knob == "pad" {
                    settings.pad
                } else {
                    settings.resolution
                };
                let (psi, d, _) = morse_half_taps(shape, settings);

                print!("  {label:>10} |");
                for (k, &area_ref) in refs.iter().enumerate() {
                    let (i0, i1) = bounds(k, settings.resolution);
                    let area = hermite_integral(&psi, &d, i0, i1, settings.resolution);
                    print!(" {:>9}", fmt_e((area - area_ref).norm() / area_ref.norm()));
                }
                println!();
            }
        };

        sweep(
            "Cranking pad",
            "pad",
            &|i| IfftSettings {
                pad: 2 * i as usize + 2,
                resolution: 1 << 12,
                ..reference
            },
            12,
        );
        // Truncation floor has to sit below the interpolation error at every row, so pad tracks the
        // resolution rather than sitting at a fixed over-provision.
        sweep(
            "Cranking resolution",
            "resolution",
            &|i| IfftSettings {
                pad: 32,
                resolution: 1 << (i + 5),
                ..reference
            },
            7,
        );

        // Acceptance: the shipping grid sits past both elbows, so the area it carries differs from
        // the reference only by floor.
        println!("\n=== Shipping Grid ===");
        let shipping = IfftSettings::default();
        let (psi, d, _) = morse_half_taps(shape, shipping);
        let mut worst: f64 = 0.0;

        for (k, &area_ref) in refs.iter().enumerate() {
            let (i0, i1) = bounds(k, shipping.resolution);
            let area = hermite_integral(&psi, &d, i0, i1, shipping.resolution);
            let e = (area - area_ref).norm() / area_ref.norm();
            worst = worst.max(e);

            println!(
                "  u: {:>6.2}, area: {:+9.7}, err: {:>9}",
                0.25 + k as f64 * 0.5,
                area,
                fmt_e(e)
            );
        }

        println!("  worst over lobes: {}", fmt_e(worst));
        assert!(worst < 1e-8);
    }

    #[test]
    fn multi_cross_reference() {
        // NEXT Probably this belongs in an integration test module.
        use super::super::reference;

        let shape = Shape {
            gamma: 3.0,
            beta: 3.5,
        };
        let beta = shape.beta;

        let settings = IfftSettings {
            periods: 12,
            ..IfftSettings::default()
        };
        let (psi, _, _) = morse_half_taps(shape, settings);

        let ifi_settings = IfiSettings::reference();

        let res = settings.resolution as f64;
        let stride = settings.resolution / 4;
        let last = (10.0 * res) as usize;

        let digits = |x: f64| if x <= 1e-17 { 17.0 } else { -x.log10() };

        let mut worst_ifft: f64 = 17.0;
        let mut worst_ifft_u: f64 = 0.0;
        let mut worst_pair: f64 = 17.0;
        let mut worst_pair_u: f64 = 0.0;
        let mut optimistic: f64 = 0.0;
        let mut optimistic_u: f64 = 0.0;

        // All three pairs as digits of agreement, so the columns compare directly.  `ifft` is the
        // IFFT against the mean of the two integrators, which agree far better with each other
        // than either does with the grid.  `pred` is what contour's own residual claims for
        // `ifi-c`; it runs conservative, so `short` is only interesting when positive.
        println!("\n=== IFFT vs IFI vs deformed contour (beta {beta}) ===");
        println!(
            "  {:>6} | {:>9} | {:>5} {:>5} {:>5} {:>5} {:>6}",
            "u", "scale", "ifft-ifi", "i-c", "pred", "short", "resid"
        );

        for k in (0..=last).step_by(stride) {
            let u = k as f64 / res;
            let v = psi[k];

            let (ifi, _, _) = morse_tap_at(shape, u, ifi_settings);
            let contour = reference::deformed_contour_morse(shape, u);

            let scale = ifi.norm().max(contour.value.norm());
            let truth = (ifi + contour.value) * 0.5;

            let ifft_d = digits((v - truth).norm() / scale);
            let tc = digits((ifi - contour.value).norm() / scale);
            let pred = digits(contour.residual / scale);
            let short = tc - pred;

            if ifft_d < worst_ifft {
                worst_ifft = ifft_d;
                worst_ifft_u = u;
            }
            if tc < worst_pair {
                worst_pair = tc;
                worst_pair_u = u;
            }
            if short < 0.0 && -short > optimistic {
                optimistic = -short;
                optimistic_u = u;
            }

            println!(
                "  {u:>6.3} | {:>9} | {ifft_d:>5.1} {tc:>5.1} {pred:>5.1} {short:>5.1} {:>6}",
                fmt_e(scale),
                fmt_e(contour.residual),
            );
        }

        println!("  worst ifft vs integrators: {worst_ifft:.1} digits at u {worst_ifft_u:.3}");
        println!("  worst ifi/contour: {worst_pair:.1} digits at u {worst_pair_u:.3}");
        println!("  residual optimistic by up to {optimistic:.1} digits at u {optimistic_u:.3}");

        assert!(worst_pair > 5.5);
    }
}
