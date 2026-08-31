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

/// Relative accuracy target for the inverse Fourier integration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IfiSettings {
    tol: f64,
    /// Ceiling on the endpoint series.  The working count is solved per call from the same Stirling
    /// bound that gates the splice; this only has to be past what any `(cap, osc, tol)` in range
    /// can ask for.  The `[Complex64; N + 1]` in `tail_moments` is sized by `MAX_TAIL_TERMS`, so a
    /// larger value here is a request, not a guarantee.
    terms: usize,
    /// `ln((terms + 1)!)`, the Stirling denominator.  Fixed by `terms`, so it is paid once here
    /// rather than per evaluation of the bracket.
    ln_fact: f64,
}

/// Backing size for the coefficient array.  `terms` clamps to this.
const MAX_TAIL_TERMS: usize = 128;

impl IfiSettings {
    fn with(tol: f64, terms: usize) -> Self {
        let terms = terms.min(MAX_TAIL_TERMS);
        Self {
            tol,
            terms,
            ln_fact: (1..=terms as u64 + 1).map(|k| (k as f64).ln()).sum(),
        }
    }

    fn reference() -> Self {
        Self::with(1e-15, MAX_TAIL_TERMS)
    }

    /// Nats of dynamic range
    fn range(self) -> f64 {
        -self.tol.ln()
    }
}

impl Default for IfiSettings {
    fn default() -> Self {
        Self::with(1e-9, MAX_TAIL_TERMS)
    }
}

/// Exact remainder of the three contour integrals on `(-inf, x]`, where the path lies on the real
/// axis and `dz = 1`.  Returns `∫ exp(delta) e^{k z} dz` for `k = 1, 2, 3` -- the same three
/// moments the quadrature accumulates into `psi`, `d`, and `dd`, less their `rho` and `TAU`
/// prefactors.
///
/// With `delta = beta z + (cap - osc) - cap e^{gamma z} + osc e^z`, the residual exponential is a
/// power series in `w = e^z` whose coefficients are the Cauchy product of `exp(-cap w^gamma)` and
/// `exp(osc w)`.  Each monomial integrates to `e^{(beta + k + m) x} / (beta + k + m)`.
/// Exact remainder of the three contour integrals on `(-inf, x]`, where the path lies on the real
/// axis and `dz = 1`.  Returns `∫ exp(delta) e^{k z} dz` for `k = 1, 2, 3` -- the same three
/// moments the quadrature accumulates into `psi`, `d`, and `dd`, less their `rho` and `TAU`
/// prefactors.
///
/// With `delta = beta z + (cap - osc) - cap e^{gamma z} + osc e^z`, the residual exponential is a
/// power series in `w = e^z` whose coefficients are the Cauchy product of `exp(-cap w^gamma)` and
/// `exp(osc w)`.  Each monomial integrates to `e^{(beta + k + m) x} / (beta + k + m)`.
///
/// `terms` carries `1/p!`, so the truncation error is below `1/terms!` once the caller has placed
/// the cut where `|cap| e^{gamma x}` and `|osc| e^x` are inside the unit disk.
fn tail_moments(
    beta: f64,
    gamma: f64,
    cap: Complex64,
    osc: Complex64,
    x: f64,
    h: f64,
    terms: usize,
) -> [Complex64; 3] {
    let g = gamma as usize;
    let mut c = [Complex64::ZERO; MAX_TAIL_TERMS + 1];

    let mut a = Complex64::ONE;
    let mut p = 0usize;
    while p * g <= terms {
        let mut b = a;
        for q in 0..=terms - p * g {
            c[p * g + q] += b;
            b *= osc / (q + 1) as f64;
        }
        p += 1;
        a *= -cap / p as f64;
    }

    let w = x.exp();
    let mut out = [Complex64::ZERO; 3];
    for k in 0..3 {
        let order = beta + (k + 1) as f64;

        // The quadrature is a trapezoid sum on the `h` grid, so the remainder must be the matching
        // geometric sum -- not the exact integral -- or the splice carries an `O((order*h)^2)` step.
        let mut pw = (cap - osc + order * x).exp();
        let mut acc = Complex64::ZERO;

        for m in 0..=terms {
            let r = order + m as f64;
            acc += c[m] * pw * h / (1.0 - (-r * h).exp());
            pw *= w;
        }

        out[k] = acc;
    }
    out
}

/// Saddles of the normalized exponent: roots of `rho^gamma - i*lambda*rho - 1`,
/// `lambda = a / beta`.  Slot 0 is the dominant root -- the one continuously connected to
/// `rho = 1` at `lambda = 0`, of largest real part throughout -- polished in `v = ln rho`.
///
/// Durand-Kerner needs no branch choice for integer `gamma`, and seeding it from the two
/// asymptotic regimes rather than a rotated circle cuts the sweep count.  Slots `1..gamma`
/// are unlabeled; those seeds are paired by index and only need to be distinct.
/// All `gamma` roots of `rho^gamma - i*lambda*rho - 1`, slot 0 dominant and polished in
/// `v = ln rho`.  Slots `1..gamma` carry Durand-Kerner accuracy only.
fn saddles(shape: Shape, lambda: f64) -> ([Complex64; 4], usize) {
    let gamma = shape.gamma as usize;
    let p = |r: Complex64| r.powi(gamma as i32) - Complex64::I * lambda * r - Complex64::ONE;

    let mu = lambda.powf(1.0 / (gamma - 1) as f64);
    let w = mu / (1.0 + mu);

    let mut z = [Complex64::ONE; 4];
    for k in 0..gamma - 1 {
        let small = TAU * k as f64 / gamma as f64;
        let big = (FRAC_PI_2 + TAU * k as f64) / (gamma - 1) as f64;
        z[k] = Complex64::from_polar(1.0 + mu, (1.0 - w) * small + w * big);
    }
    let endpoint = TAU * (gamma - 1) as f64 / gamma as f64;
    z[gamma - 1] =
        Complex64::from_polar(1.0 / (1.0 + lambda), (1.0 - w) * endpoint + w * FRAC_PI_2);

    for _ in 0..12 {
        let mut moved = 0.0f64;
        for k in 0..gamma {
            let mut denom = Complex64::ONE;
            for j in 0..gamma {
                if j != k {
                    denom *= z[k] - z[j];
                }
            }
            let step = p(z[k]) / denom;
            z[k] -= step;
            moved = moved.max(step.norm());
        }
        if moved < 1e-3 {
            break;
        }
    }

    let mut best = 0;
    for k in 1..gamma {
        if z[k].re > z[best].re {
            best = k;
        }
    }
    z.swap(0, best);

    let (beta, gammaf) = (shape.beta, shape.gamma);
    let a = lambda * beta;
    let mut v = z[0].ln();
    for _ in 0..6 {
        let (pow, lin) = ((gammaf * v).exp(), v.exp());
        let g = beta * (Complex64::ONE - pow) + Complex64::I * a * lin;
        let dg = -beta * gammaf * pow + Complex64::I * a * lin;
        let step = g / dg;
        v -= step;
        if step.norm() < f64::EPSILON * v.norm().max(1.0) {
            break;
        }
    }
    z[0] = v.exp();

    (z, gamma)
}

fn saddle(shape: Shape, lambda: f64) -> Complex64 {
    saddles(shape, lambda).0[0]
}

/// Time-domain amplitude at `u` in carrier periods.  Returns the same dimensionless `psi`, `d`, and
/// `dd` that `morse_half_taps` should converge to.
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
    let range = settings.range();

    const SECTOR: f64 = 0.31;
    const BEND_OFFSET: f64 = 1.3;
    const BEND_SCALE: f64 = 0.33;
    const POLE_MARGIN: f64 = 0.90;

    let (beta, gamma) = (shape.beta, shape.gamma);
    let gi = gamma as i32;

    let s_peak = shape.peak();
    let a = TAU * u;

    let rho = saddle(shape, a / beta);

    // `s_peak^gamma = beta / gamma` for the Morse peak, so `gamma * cap = beta * rho^gamma` and the
    // saddle equation `beta(1 - rho^gamma) + osc = 0` is exactly `delta'(0) = 0`.  The stationary
    // point of the quadrature exponent is the origin of the `z` frame, not merely near it.
    let cap = s_peak.powf(gamma) * rho.powi(gi);
    let osc = Complex64::I * a * rho;

    let phi2 = gamma * gamma * cap - osc;
    let phi3 = osc - gamma * gamma * gamma * cap;
    let width = (1.0 / phi2.norm().sqrt()).min((6.0 / phi3.norm()).cbrt());

    let phi_end = SECTOR * PI / (2.0 * gamma);
    let swing = (rho.arg() - phi_end).max(0.0);
    let bend = BEND_OFFSET * width.min(BEND_SCALE);

    // The turn is centered near the saddle, so the raw tanh displaces `x = 0` off the stationary
    // point.  Referencing the shift to its own value at the origin makes `z(0) = 0` identically,
    // which is the property the quadratic model and `peak` both assume.  `dz` is unchanged --
    // the offset is a constant.
    let tanh0 = (-bend / BEND_SCALE).tanh();
    let shift = |t: Complex64| (t - tanh0) * 0.5;
    let path = |x: Complex64| {
        let tanh = ((x - bend) / BEND_SCALE).tanh();
        let z = x - Complex64::I * swing * shift(tanh);
        let dz = Complex64::ONE
            - Complex64::I * swing * (Complex64::ONE - tanh * tanh) * 0.5 / BEND_SCALE;
        (z, dz)
    };

    // `gamma` is integral and the `rho_j^3` weight is a pure magnitude, so the exponent needs one
    // complex `exp` rather than two and the weight closes to `3(ln|rho| + Re z)`.
    let ln_rho = rho.norm().ln();
    let exponent = |z: Complex64| {
        let w = z.exp();
        let delta = beta * z - cap * (w.powi(gi) - Complex64::ONE) + osc * (w - Complex64::ONE);
        (delta, w)
    };
    let height = |x: f64, y: f64| {
        let (z, dz) = path(Complex64::new(x, y));
        let (delta, _) = exponent(z);
        delta.re + 0.5 * dz.norm_sqr().ln() + 3.0 * (ln_rho + z.re)
    };

    // The tail assumes `dz = 1`, which needs `1 - |tanh|` below `tol` relative to the swing the
    // moments would otherwise miss.  Only meaningful when the path actually bends.
    let x_bend = if swing > 0.0 {
        bend - 0.5 * BEND_SCALE * (range + (swing * (beta + 3.0)).ln())
    } else {
        f64::INFINITY
    };

    let ln_fact = settings.ln_fact;
    let max_terms = settings.terms;

    let a_of = |x: f64| {
        let (c, o) = (cap.norm() * (gamma * x).exp(), osc.norm() * x.exp());
        (c + o, gamma * c + o)
    };

    // Radius of the Cauchy product, not its truncation: below this the series argument is inside
    // the unit disk, the terms decay from the first, and the sum has no cancellation to lose.
    // Truncation is then a consequence rather than a separate claim.
    let x_disk = {
        let mut x = 0.0f64;
        for _ in 0..8 {
            let (a_val, a_prime) = a_of(x);
            let step = a_val.ln() / (a_prime / a_val);
            x -= step;
            if step.abs() < 1e-12 * x.abs().max(1.0) {
                break;
            }
        }
        x
    };

    // provisional peak
    let peak = height(0.0, 0.0);

    // Where the integrand is `range` below the crest.  The tail is exact in form but truncated in
    // practice, so its share of the answer has to be small on its own terms -- the disk radius
    // bounds the series' validity, not its weight.  On the settled left the path is straight and
    // the cap term is negligible, so the height is affine and the cut is explicit.
    let x_mag = (peak - range - (cap - osc).re) / (beta + 3.0);
    let x_cut = x_disk.min(x_bend).min(x_mag);

    let x_settle = bend + 3.0 * BEND_SCALE;

    let phi_tail = if swing > 0.0 { phi_end } else { rho.arg() };
    let damp = (gamma * phi_tail).cos().max(f64::MIN_POSITIVE);

    let x_max = {
        let osc_phase = (osc.arg() - swing).cos();
        let target = peak - range;
        let mut x = (((range + peak.abs() + 1.0) / (cap.norm() * damp)).ln() / gamma)
            .max(x_settle + BEND_SCALE);
        for _ in 0..6 {
            let (c, o) = (cap.norm() * (gamma * x).exp(), osc.norm() * x.exp());
            let f =
                beta * x + (cap - osc).re - c * damp + o * osc_phase + 3.0 * (ln_rho + x) - target;
            let df = beta + 3.0 - gamma * c * damp + o * osc_phase;
            let step = f / df;
            x = (x - step).max(x_settle);
            if step.abs() < 1e-12 * x.abs().max(1.0) {
                break;
            }
        }
        x
    };

    // Right end of the scan, solved on the line rather than inherited from the real axis: the cap
    // term damps as `cos(gamma * (phi_tail + y))` off-axis, so the real-axis `x_max` can still be
    // on the rising side.  Same closed form as `x_max`, one parameter different.
    let x_end = |y: f64| {
        let damp_y = (gamma * (phi_tail + y)).cos().max(f64::MIN_POSITIVE);
        let osc_phase = (osc.arg() - swing + y).cos();
        let target = peak - range;
        let mut x = x_max;
        for _ in 0..8 {
            let (c, o) = (cap.norm() * (gamma * x).exp(), osc.norm() * x.exp());
            let f = beta * x + (cap - osc).re - c * damp_y + o * osc_phase + 3.0 * (ln_rho + x)
                - target;
            let df = beta + 3.0 - gamma * c * damp_y + o * osc_phase;
            let step = f / df;
            x = (x - step).max(x_settle);
            if step.abs() < 1e-12 * x.abs().max(1.0) {
                break;
            }
        }
        x
    };

    // `d/dx height` on the line at `y`, carrying the deformation: the crest sits inside the bend
    // whenever the path swings, so the tanh Jacobian is part of the stationarity condition, not a
    // correction to it.  Secant rather than Newton -- the second derivative of the Jacobian term
    // is not worth forming, and the bracket below is what actually guarantees the basin.
    let slope = |x: f64, y: f64| {
        let c = Complex64::new(x, y);
        let tanh = ((c - bend) / BEND_SCALE).tanh();
        let sech2 = Complex64::ONE - tanh * tanh;
        let dz = Complex64::ONE - Complex64::I * swing * sech2 * 0.5 / BEND_SCALE;
        let ddz = Complex64::I * swing * sech2 * tanh / (BEND_SCALE * BEND_SCALE);
        let (z, _) = path(c);
        let w = z.exp();
        let ddelta = beta - gamma * cap * w.powi(gi) + osc * w;
        ((ddelta + 3.0) * dz + ddz / dz).re
    };

    // Need peak
    /// Roots of `beta + 3 - gamma*cap*w^gamma + osc*w = 0`, the stationarity condition for
    /// `Re(delta + 3z)`.  These are the saddles of the weighted exponent -- the local maxima of the
    /// height along the path -- so the crest is one of them rather than something to be searched
    /// for.  Durand-Kerner from a circle: the roots are well separated here and only need to reach
    /// the basin of the polish below.
    let crest_roots = || {
        let g = gamma as usize;
        let lead = -gamma * cap;
        let mut w = [Complex64::ONE; 4];
        let seed = ((beta + 3.0) / lead.norm()).powf(1.0 / gamma);
        for k in 0..g {
            w[k] = Complex64::from_polar(seed, TAU * k as f64 / g as f64 + 0.4);
        }
        for _ in 0..40 {
            let mut moved = 0.0f64;
            for k in 0..g {
                let p = lead * w[k].powi(gi) + osc * w[k] + (beta + 3.0);
                let mut den = lead;
                for j in 0..g {
                    if j != k {
                        den *= w[k] - w[j];
                    }
                }
                let step = p / den;
                w[k] -= step;
                moved = moved.max(step.norm());
            }
            if moved < 1e-13 {
                break;
            }
        }
        (w, g)
    };

    let crest = |y: f64| -> Option<f64> {
        let hi_end = x_end(y);
        let (w, g) = crest_roots();

        let mut best: Option<(f64, f64)> = None;
        for k in 0..g {
            // `z.re = x` on the path up to the swing's own imaginary part, so the root's log is
            // already the right seed; the secant carries the Jacobian correction.
            // let mut x0 = w[k].ln().re;

            // The polynomial lives in `z`, the bracket in `x`, and on the bent path they differ by
            // the swing's real part -- which depends on `y`, so a seed taken straight from
            // `Re(ln w)` is displaced on exactly the lines where the bend is doing the most work.
            // Two fixed-point passes invert `Re path(x + iy) = Re ln w`; the map is a contraction
            // here because the tanh's slope is bounded by `1 / BEND_SCALE` and the swing is small
            // against it.
            let target = w[k].ln().re;
            let mut x0 = target;
            for _ in 0..2 {
                let t = ((Complex64::new(x0, y) - bend) / BEND_SCALE).tanh();
                x0 = target - swing * shift(t).im;
            }
            let mut x1 = x0 + 1e-3;

            let mut s0 = slope(x0, y);
            for _ in 0..24 {
                let s1 = slope(x1, y);
                if (s1 - s0).abs() < f64::MIN_POSITIVE {
                    break;
                }
                let x2 = x1 - s1 * (x1 - x0) / (s1 - s0);
                x0 = x1;
                s0 = s1;
                x1 = x2;
                if (x1 - x0).abs() < 1e-12 * x1.abs().max(1.0) {
                    break;
                }
            }

            if x1 > x_cut && x1 < hi_end && slope(x1, y).abs() < 1.0 {
                let h = height(x1, y);
                if best.map_or(true, |(bh, _)| h > bh) {
                    best = Some((h, x1));
                }
            }
        }
        best.map(|(_, x)| x)
    };

    // The good peak
    // let peak = match crest(0.0) {
    //     Some(x) => height(x, 0.0).max(height(0.0, 0.0)),
    //     None => height(0.0, 0.0),
    // };

    let strip = |y: f64| {
        let ends = height(x_cut, y).max(height(x_end(y), y));
        match crest(y) {
            Some(x) => height(x, y).max(ends),
            None => ends,
        }
    };

    let y_tanh = FRAC_PI_2 * BEND_SCALE;
    let y_cap = FRAC_PI_2 / gamma;
    let blend = (swing / BEND_SCALE).min(1.0);
    let y_pole = POLE_MARGIN * (blend * y_tanh.min(y_cap) + (1.0 - blend) * y_cap);

    let h = (1..=5)
        .map(|k| {
            let y = y_pole * k as f64 / 5.0;
            TAU * y / (range + (strip(y).max(strip(-y)) - peak).max(0.0))
        })
        .fold(0.0f64, f64::max);

    let lo = (x_cut / h).ceil() as i64;
    let hi = (x_max / h).ceil() as i64;

    let mut psi_re = Accumulator::<f64>::default();
    let mut psi_im = Accumulator::<f64>::default();
    let mut d_re = Accumulator::<f64>::default();
    let mut d_im = Accumulator::<f64>::default();
    let mut dd_re = Accumulator::<f64>::default();
    let mut dd_im = Accumulator::<f64>::default();

    // Largest height reached on the walked path, and where.  On a correct single-thimble
    // deformation this is `peak` at `j = 0`.
    let mut walk_top = f64::NEG_INFINITY;
    let mut walk_top_x = 0.0f64;

    for j in lo..=hi {
        let x = j as f64 * h;
        let (z, dz) = path(Complex64::new(x, 0.0));
        let (delta, w) = exponent(z);

        let hgt = delta.re + 0.5 * dz.norm_sqr().ln() + 3.0 * (ln_rho + z.re);
        if hgt > walk_top {
            walk_top = hgt;
            walk_top_x = x;
        }

        let step = rho * w * TAU;
        let term = delta.exp() * (h * dz) * rho * w;
        let dv = term * step;
        let ddv = dv * step;

        psi_re.add(term.re);
        psi_im.add(term.im);
        d_re.add(dv.re);
        d_im.add(dv.im);
        dd_re.add(ddv.re);
        dd_im.add(ddv.im);
    }

    let edge = lo as f64 * h - h;

    let mut tail_terms = 0usize;
    if height(edge, 0.0) - peak > -range {
        let terms = {
            let a_ser = cap.norm() * (gamma * edge).exp() + osc.norm() * edge.exp();
            let head = beta * edge + (cap - osc).re + a_ser + range;
            let ln_a = a_ser.ln();
            let mut ln_f = 0.0;
            let mut n = max_terms;
            for k in 0..max_terms {
                ln_f += ((k + 1) as f64).ln();
                if head + (k + 1) as f64 * ln_a - ln_f < 0.0 {
                    n = k;
                    break;
                }
            }
            n
        };
        tail_terms = terms + 1;

        let m = tail_moments(beta, gamma, cap, osc, edge, h, terms);

        let tail_psi = rho * m[0];
        let tail_d = rho * rho * TAU * m[1];
        let tail_dd = rho * rho * rho * TAU * TAU * m[2];

        psi_re.add(tail_psi.re);
        psi_im.add(tail_psi.im);
        d_re.add(tail_d.re);
        d_im.add(tail_d.im);
        dd_re.add(tail_dd.re);
        dd_im.add(tail_dd.im);
    }

    let scale = (beta * rho.ln() - cap + osc).exp() * s_peak.powf(beta);

    let psi = Complex64::new(psi_re.sum(), psi_im.sum()) * scale;
    let d = Complex64::new(d_re.sum(), d_im.sum()) * scale;
    let dd = Complex64::new(dd_re.sum(), dd_im.sum()) * scale;

    (psi, d, dd)
}

fn fmt_e(x: f64) -> String {
    let s = format!("{x:+.2e}");
    // split "±m.mme±dd" into mantissa and exponent, then zero-pad the exponent
    let (mantissa, exp) = s.split_once('e').unwrap_or(("999", "999"));
    let exp: i32 = exp.parse().unwrap();
    format!("{mantissa}e{exp:+03}")
}

#[cfg(test)]
mod test {

    use super::super::whatsleft::Accumulator;
    use super::*;

    // These tests are mostly print tests used to calibrate, fill in empirical values, and design
    // the accuracy of the wavelet.
    // NEXT feature rule?

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
        let oracle = IfiSettings::reference();

        let base = IfftSettings {
            periods: 8,
            pad: 56,
            resolution: 1 << 8,
        };
        // Amplitude has to clear the roundoff floor by this much before a cell reports.
        const LIVE: f64 = 16.0;
        const TOL: f64 = 1e-4; // Around u = 7, IFFT and IFI start to disagree enough to trip any more.

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
        let floor = noise.max(oracle.tol * peak);

        let reach = shipping.reach();

        let mut worst = [(0.0f64, 0.0f64); 2];
        let mut live_to = [0.0f64; 2];

        println!(
            "  {:>6} | {:>9} {:>9} | {:>9} {:>9} | {:>9}",
            "u", "psi", "d", "ifft-psi", "ifft-d", "decay"
        );

        let mut drift = [(0.0f64, 0.0f64); 2];

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

            // Nearest grid index; the oracle is exact at any `u`, so evaluating it *there*
            // isolates solver-vs-solver drift from Hermite error.
            let idx = t.round() as usize;
            let u_grid = idx as f64 / shipping.resolution as f64;
            let (psi_g, d_g, _) = morse_tap_at(shape, u_grid, oracle);
            let g = [rel(psi[idx], psi_g, ps), rel(d[idx], d_g, ds)];

            if k % 10 == 0 {
                println!(
                    "  {u:>6.2} | {:>9} {:>9} | {:>9} {:>9} | {:>9}",
                    fmt_e(e[0]),
                    fmt_e(e[1]),
                    fmt_e(g[0]),
                    fmt_e(g[1]),
                    fmt_e(ps / peak)
                );
            }

            let live = [ps > floor * LIVE, ds > floor * LIVE];
            for j in 0..2 {
                if live[j] && g[j] > drift[j].0 {
                    drift[j] = (g[j], u);
                }
            }

            for j in 0..2 {
                if !live[j] {
                    continue;
                }
                live_to[j] = u;
                if e[j] > worst[j].0 {
                    worst[j] = (e[j], u);
                }
            }
        }

        println!(
            "\n  floor {} at peak scale (ifft {}, oracle {})",
            fmt_e(floor / peak),
            fmt_e(noise / peak),
            fmt_e((-oracle.tol))
        );
        for (j, name) in ["psi", "d"].iter().enumerate() {
            println!(
                "  {name:>3}: worst {} at u {:.3}, oracle drift {} at u {:.3}, live to u {:.2}",
                fmt_e(worst[j].0),
                worst[j].1,
                fmt_e(drift[j].0),
                drift[j].1,
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
        let reference = IfiSettings::reference();

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Peak, shoulder, the old sick zone around 3.7-4.7, and deep tail.
        let probes = [
            0.0, 0.8, 2.0, 2.4, 3.7, 4.7, 5.5, 6.7, 7.911, 8.2, 8.33, 11.0, 14.0,
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
                    let label = settings.tol;

                    print!("  {label:>10.1e} |");
                    for u in probes {
                        let r = pick(tap, morse_tap_at(shape, u, reference));
                        let v = pick(tap, morse_tap_at(shape, u, settings));
                        print!(" {:>9}", fmt_e(rel(v, r)));
                    }
                    println!();
                }
            };

        let matrix_start = std::time::Instant::now();
        for tap in [Tap::Psi, Tap::D, Tap::Dd] {
            sweep(
                tap,
                "Cranking tol",
                "tol",
                &|i| IfiSettings {
                    tol: 1.0 / 10.0f64.powf((i + 1) as f64),
                    ..reference
                },
                13,
            );
        }
        let matrix_time = matrix_start.elapsed();

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
        println!("  matrix completed in: {}", matrix_time.as_millis());
        assert!(worst < 1e-8);
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
        use super::super::jetmorse;
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

        // Every pair, so no method sits in the denominator of its own score.  `f` is IFFT,
        // `i` is IFI, `c` is deformed contour, `j` is the jet.  If the three integrators
        // agree with each other better than any of them agrees with `j`, the jet is the
        // outlier; if `j-c` tracks `i-c`, the jet is inside the reference noise.
        println!("\n=== pairwise agreement, beta {beta} ===");
        println!(
            "  {:>6} | {:>9} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5}",
            "u", "scale", "i-c", "f-i", "f-c", "j-i", "j-c", "j-f", "pred"
        );

        let mut worst_ic: f64 = 17.0;
        let mut worst_ic_u: f64 = 0.0;
        let mut worst_jc: f64 = 17.0;
        let mut worst_jc_u: f64 = 0.0;
        let mut worst_gap: f64 = 17.0;
        let mut worst_gap_u: f64 = 0.0;

        for k in (0..=last).step_by(stride) {
            let u = k as f64 / res;

            let ifft = psi[k];
            let (ifi, _, _) = morse_tap_at(shape, u, ifi_settings);
            let contour = reference::deformed_contour_morse(shape, u);
            let jet = jetmorse::jet_morse(shape, u).psi;

            let scale = ifi.norm().max(contour.value.norm());
            let d = |a: Complex64, b: Complex64| digits((a - b).norm() / scale);

            let ic = d(ifi, contour.value);
            let fi = d(ifft, ifi);
            let fc = d(ifft, contour.value);
            let ji = d(jet, ifi);
            let jc = d(jet, contour.value);
            let jf = d(jet, ifft);
            let pred = digits(contour.residual / scale);

            // How much worse the jet's best pairing is than the integrators' best pairing.
            let gap = ic.max(fi).max(fc) - ji.max(jc).max(jf);

            if ic < worst_ic {
                worst_ic = ic;
                worst_ic_u = u;
            }
            if jc < worst_jc {
                worst_jc = jc;
                worst_jc_u = u;
            }
            if gap > 0.0 && 17.0 - gap < worst_gap {
                worst_gap = 17.0 - gap;
                worst_gap_u = u;
            }

            println!(
                "  {u:>6.3} | {:>9} | {ic:>5.1} {fi:>5.1} {fc:>5.1} | {ji:>5.1} {jc:>5.1} {jf:>5.1} | {pred:>5.1}",
                fmt_e(scale),
            );
        }

        println!("  worst ifi/contour: {worst_ic:.1} digits at u {worst_ic_u:.3}");
        println!("  worst jet/contour: {worst_jc:.1} digits at u {worst_jc_u:.3}");
        println!(
            "  largest jet deficit: {:.1} digits at u {worst_gap_u:.3}",
            17.0 - worst_gap
        );

        assert!(worst_ic > 5.5);
    }
}
