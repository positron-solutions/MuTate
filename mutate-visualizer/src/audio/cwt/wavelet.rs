// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # The Wavelet
//!
//! > The traveler who fears the unknown road will eventually learn that known roads return
//! > to where they began.
//! >
//! > - Anthony L. Ray
//!
//!               #
//!               #
//!               #
//!
//!                ###
//!                ######
//!                ######
//!
//!       #########
//! ###############
//!    ############
//!
//!                ##############
//!                ####################
//!                ##############
//!
//!    ############
//! ###############
//!       #########
//!
//!                ######
//!                ######
//!                ###
//!
//!               #
//!               #
//!               #
//!
//! This module generates our wavelet tables. Morse wavelet is the first chosen implementation.
//!
//! - Easy to generate without dependencies
//! - Regarded as nice for time and frequency reassignment
//! - Parameterized
//!
//! None of these things matter unless people can see pretty pixels, so get to waving!
//!
//! ## Usage
//!
//! One [`Plan`] per Q. It holds the spectrum in a normalized frequency variable, so it is
//! independent of center frequency and sample rate, and every voice sharing that Q may reuse it.
//!
//! ```
//! # use mutate::wavelet::{Spec, Taper};
//! let mut plan = Spec::default()
//!     .taper(Taper{eps_time: 1e-3, rho: 0.1})
//!     .plan();
//!
//! let bin = plan.bin(1000.0, 8000.0);
//! let mut taps = vec![(0.0, 0.0); bin.taps()];
//! plan.taps_into(bin, &mut taps);
//! ```
//!
//! Reassignment consumes the plan and adds the two derivative spectra. The `d` and `t` taps
//! carry psi's normalization, so all three convolve against the same signal scale.
//!
//! ```
//! # use mutate::wavelet::{Spec, Taper};
//! let mut plan = Spec::default()
//!     .taper(Taper{eps_time: 1e-3, rho: 0.1})
//!     .plan_with_reassignment();
//!
//! let bin = plan.bin(1000.0, 8000.0);
//! let n = bin.taps();
//! let (mut psi, mut d, mut t) = (vec![(0.0, 0.0); n], vec![(0.0, 0.0); n], vec![(0.0, 0.0); n]);
//!
//! // d and t are multiplied by i at use.
//! plan.taps_into(bin, &mut psi, &mut d, &mut t);
//! ```

// NEXT ReassignPlan and Plan are too similar.  Go ahead and re-combine them over the 1-vs-3 weights
// axis.
// NOTE We have logarithmic bin spacings, but the cutoff frequencies that determine which downsample
// will be used are not particularly aware, so it's not expected that we can re-use exact bins in
// any kind of octave structure.  Mel scaling etc also defeats this, so there's no point.
// NEXT Run time of the bin generation test (not reflective of actual sample rates and Q) is about
// 33ms on a Zen2+ part.  This affects CWT startup time.
// NEXT We must do a very fine frequency sweep on a resulting filter bank sampling edge to really
// have an idea of the correctness of the gain peak location.  If there is a bias, it probably is
// consistent, but if the design bias doesn't remain consistent across the resampling edges, we will
// start to see bands in the results.
// NOTE Peak gain 2 per bin (demodulated L1 = 2), so a unit real tone reads |W| = 1.
// Steady tones are flat across the bank.  Noise floor rises as sqrt(f).
// NOTE Zero phase. Conjugate-symmetric taps give a real (one-sided, not even) frequency response
// and no group delay.
// 🤖 Heavy generation.  Should be pretty standard academic stuff, so not expecting a lot of
// surprises.  We will, for the most part, swiftly and knowingly eat shit if the wavelet is busted.
// Well-formalized stuff doesn't have a lot of wiggle room to violate the consistency of the
// formalism.
//
// === RESPONSE (Q = 2.5, gamma = 3, eps = 1e-8, taper 1e-3 rho 0.1, rel width exact 0.39865 gaussian 0.40000) ===
//
// fc    1000 sr   6000  taps    31  w0 1.047198
//   peak gain 2.000000060  dev +2.978e-8 rel
//   rel width 0.39875  exact 0.39865  ratio 1.00026
//   peak -0.0804 cents  -2.33e-4 of half-width
//   negative-freq max   -83.56 dB
//   stopband floor      -85.86 dB
//   peak gain 2.00000  analytic 2.00000  ratio 1.00000
//
// fc     250 sr   3000  taps    61  w0 0.523599
//   peak gain 2.000000073  dev +3.642e-8 rel
//   rel width 0.39883  exact 0.39865  ratio 1.00045
//   peak -0.1200 cents  -3.48e-4 of half-width
//   negative-freq max   -78.37 dB
//   stopband floor      -80.65 dB
//   peak gain 2.00000  analytic 2.00000  ratio 1.00000
//
// fc   12000 sr  48000  taps    21  w0 1.570796
//   peak gain 1.999999972  dev -1.424e-8 rel
//   rel width 0.39870  exact 0.39865  ratio 1.00012
//   peak -0.0448 cents  -1.30e-4 of half-width
//   negative-freq max   -80.34 dB
//   stopband floor      -80.34 dB
//   peak gain 2.00000  analytic 2.00000  ratio 1.00000

/// Filter peak gain. Analytic taps see half a real tone's amplitude,
/// so |H| = 2 makes a unit tone read |W| = 1.
const PEAK_GAIN: f64 = 2.0;

/// A center frequency resolved against a plan and a sample rate.
#[derive(Clone, Copy)]
pub struct Bin {
    // rad/sample at the decimated rate
    w0: f64,
    taps: usize,
}

impl Bin {
    /// Number of taps.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Rotational velocity ദ്ദി(•̀ω-)✧ in radians.
    pub fn velocity(&self) -> f64 {
        self.w0
    }
}

/// Controls Q and other critical tradeoffs of the wavelets.
#[derive(Clone, Copy)]
pub struct Shape {
    /// `gamma` sets the exponent of the spectral envelope, trading Gaussian core against flank
    /// weight. Taps are zero-phase, so the time envelope stays even for any gamma.
    pub gamma: f64,
    /// Adjusting beta at fixed gamma is adjusting Q.
    pub beta: f64,
}

impl Shape {
    /// `q` is the quality factor on the -3 dB energy width. Higher `q` narrows the band and costs
    /// proportionally more taps at a given center frequency.
    pub fn from_q(q: f64, gamma: f64) -> Self {
        let p = 1.6651 * q;
        Shape {
            gamma,
            beta: p * p / gamma,
        }
    }

    /// P = sqrt(beta*gamma), the -3 dB width parameter.  Q = P/1.6651.
    pub fn p(&self) -> f64 {
        (self.beta * self.gamma).sqrt()
    }

    /// Argmax of the spectral envelope, in rad/sample.
    pub fn peak(&self) -> f64 {
        if self.gamma == 3.0 {
            (self.beta / 3.0).cbrt()
        } else {
            (self.beta / self.gamma).powf(1.0 / self.gamma)
        }
    }
}

/// Trades stopband depth for tap count.
///
/// - `rho` is the fraction of the half-span over which the truncation is smoothed.
/// - `eps_time` truncates the tap envelope.
///
/// Other parameters have their normal behavior.
#[derive(Clone, Copy)]
pub struct Taper {
    /// Error tolerance with respect to time truncation.
    pub eps_time: f64,
    /// Controls width of the taper.
    pub rho: f64,
}

/// A builder struct for configuring a plan to build wavelets with a given shape.
///
/// ```
/// # use mutate::wavelet::{Spec, Taper};
///
/// let spec = Spec::default()
///    .taper(Taper { eps_time: 1e-3, rho: 0.1 });
/// ```
#[derive(Clone, Copy)]
pub struct Spec {
    shape: Shape,
    eps: f64,
    tail_a: f64,
    taper: Option<Taper>,
    max_taps: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            shape: Shape::from_q(2.5, 3.0),
            eps: 1e-8,
            tail_a: 1.0,
            taper: None,
            max_taps: 0,
        }
    }
}

impl Spec {
    /// Set mother wavelet [`Shape`].
    pub fn shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// `eps` is the spectral truncation floor relative to the peak. It sets how far the baked grid
    /// extends, and through that the tap count, but not the shape. 1e-8 lands near the f32 noise
    /// floor of the output taps.  Never use `eps_time` (on [`Taper`]) lower than `eps` because it
    /// means avoiding truncation of what is already inexact.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// `tail_a` scales the algebraic-tail branch of the half-width estimate, which dominates the
    /// Gaussian core at low `q`. Raise it to buy more tail at the cost of taps; 1.0 is the
    /// calibrated value and the one every response measurement in this module was taken with.
    pub fn tail_a(mut self, tail_a: f64) -> Self {
        self.tail_a = tail_a;
        self
    }
    /// Set taper and error tolerance
    pub fn taper(mut self, taper: Taper) -> Self {
        self.taper = Some(taper);
        self
    }

    /// `max_taps` is a **hint** to allocate a larger scratch `Vec`, which will of course resize if
    /// necessary.  Uses [`Vec::with_capacity`](std::vec::Vec::with_capacity).
    pub fn max_taps(mut self, max_taps: usize) -> Self {
        self.max_taps = max_taps;
        self
    }

    /// Half-span that sets tap count, and the grid extent that feeds it.
    fn spans(&self) -> (f64, f64) {
        let grid = half_width_scaled(self.shape, self.eps, self.tail_a);
        let taps = match self.taper {
            Some(t) => half_width_scaled(self.shape, t.eps_time, self.tail_a),
            None => grid,
        };
        (taps, grid.max(taps))
    }

    pub fn plan(self) -> Plan {
        let (c, extent) = self.spans();
        let peak = self.shape.peak();

        // Aliasing in t occurs at tau/(du*omega0); du < pi/extent keeps the replica clear.
        let du = 0.8 * core::f64::consts::PI / extent;
        let (lo, m) = support(self.shape, du, self.eps);

        let mut psi = vec![0.0; m];
        fill_grid(&self.shape, du, lo, &mut psi);

        Plan {
            shape: self.shape,
            peak,
            c,
            du,
            psi,
            lo,
            rho: self.taper.map_or(0.0, |t| t.rho),
            buf: Vec::with_capacity(3 * self.max_taps),
        }
    }

    pub fn plan_with_reassignment(self) -> ReassignPlan {
        self.plan().with_reassignment()
    }
}

/// Shape-only. Everything here is independent of center frequency and rate.
pub struct Plan {
    shape: Shape,
    peak: f64,            // argmax of w^beta e^{-w^gamma}
    c: f64,               // half_width_scaled; taps = 2*ceil(c/omega0)+1
    du: f64,              // uniform step in u = w/w_peak
    psi: Vec<f64>,        // psi at u_j = j*du, peak 1
    lo: usize,            // first grid point above eps; psi[..lo] is zero
    rho: f64,             // taper fraction of the half-span; 0.0 disables
    buf: Vec<(f64, f64)>, // bake scratch, 3 * longest bin seen
}

impl Plan {
    /// Adds the reassignment spectra. Costs two more grids of plan memory.
    fn with_reassignment(self) -> ReassignPlan {
        let m = self.psi.len();

        // g(u) = beta*ln(u) - (beta/gamma)*u^gamma, g(1) = -beta/gamma.
        // Truncate where g(u) - g(1) < ln(eps), on both flanks.
        let (beta, gamma) = (self.shape.beta, self.shape.gamma);
        let mut spec = vec![[0.0; 3]; m];

        for j in 1..m {
            let w = self.peak * j as f64 * self.du;
            let wg = w.powf(gamma);
            let p = self.psi[j];
            spec[j] = [p, p * w, p * (beta / w - gamma * wg / w)];
        }
        ReassignPlan { plan: self, spec }
    }

    /// Calculate angular velocity for a bin at `center`, sampled at `rate`.  Ties together center
    /// radial velocity and tap count.
    ///
    /// `center` below `rate` and its Nyquist limit.
    pub fn bin(&self, center: f64, rate: f64) -> Bin {
        let nyquist = rate / 2.0;
        debug_assert!(
            center < rate / 2.0,
            "center {center:.1}Hz above Nyquist {:.0}Hz",
            rate / 2.0
        );
        let w0 = core::f64::consts::TAU * center / rate;
        Bin {
            w0,
            taps: 2 * (self.c / w0).ceil() as usize + 1,
        }
    }

    /// Writes bin.taps() complex taps, centered, peak gain 2, DC removed.
    pub fn taps_into(&mut self, bin: Bin, out: &mut [(f32, f32)]) {
        let mut buf = self.take_buf(1, bin.taps);
        {
            let buf = &mut buf[..];

            self.transform(bin, buf);
            self.taper(buf);
            Self::center(buf);
            Self::scale_by(buf, PEAK_GAIN / Self::gain_at(buf, bin.w0));
            Self::quantize(buf, out);
        }
        self.buf = buf;
    }

    /// Single-spectrum rotor walk over psi.  Takes advantage of odd symmetry.
    fn transform(&self, bin: Bin, out: &mut [(f64, f64)]) {
        let n = out.len();
        let half = n / 2;
        let step = self.du * bin.w0;
        let (ss, sc) = Plan::seed_step(step, self.lo);
        let (mut sr, mut si) = (1.0f64, 0.0f64);

        for i in half..n {
            let d = step * (i - half) as f64;
            let (ds, dc) = d.sin_cos();
            let (mut cr, mut ci) = (sr, si);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for &p in &self.psi[self.lo..] {
                re += p * cr;
                im += p * ci;
                let (nr, ni) = (cr * dc - ci * ds, cr * ds + ci * dc);
                cr = nr;
                ci = ni;
            }
            out[i] = (re, im);
            out[n - 1 - i] = (re, -im);

            let (nr, ni) = (sr * sc - si * ss, sr * ss + si * sc);
            sr = nr;
            si = ni;
        }
    }

    /// DC removal only. Hermitian: pairs contribute 2*re, center once.
    fn center(out: &mut [(f64, f64)]) {
        let n = out.len();
        let half = n / 2;
        let mr = (2.0 * out[half + 1..].iter().map(|s| s.0).sum::<f64>() + out[half].0) / n as f64;
        for s in out.iter_mut() {
            s.0 -= mr;
        }
    }

    fn scale_by(out: &mut [(f64, f64)], norm: f64) {
        for s in out.iter_mut() {
            s.0 *= norm;
            s.1 *= norm;
        }
    }

    /// Planck taper over the outer `rho` of the half-span.  Smooth to all orders at
    /// the join and at the edge, so the truncation step stops setting the stopband.
    fn taper(&self, out: &mut [(f64, f64)]) {
        if self.rho == 0.0 {
            return;
        }
        let n = out.len();
        let half = n / 2;

        // zero lands one tap past the end, so the outermost tap still carries weight.
        let end = half as f64 + 1.0;
        let inv = (end * self.rho).recip();
        let start = end * (1.0 - self.rho);

        for i in half..n {
            let s = ((i - half) as f64 - start) * inv;
            if s <= 0.0 {
                continue;
            }
            let w = 1.0 / (1.0 + (1.0 / (1.0 - s) - 1.0 / s).exp());
            out[i].0 *= w;
            out[i].1 *= w;
            out[n - 1 - i].0 *= w;
            out[n - 1 - i].1 *= w;
        }
    }

    /// Per-tap advance for a rotor seeded at grid index `base`.
    fn seed_step(step: f64, base: usize) -> (f64, f64) {
        (step * base as f64).sin_cos()
    }

    /// Takes the bake buffer out, sized for `k` spans of `n`. Caller puts it back.
    fn take_buf(&mut self, k: usize, n: usize) -> Vec<(f64, f64)> {
        let mut buf = core::mem::take(&mut self.buf);
        buf.resize(k * n, (0.0, 0.0));
        buf
    }

    /// Real part quantized with error feedback along i, weighting the center tap
    /// once and the mirrored pairs twice, so residual DC after rounding is
    /// minimized rather than accumulating as a random walk. The imaginary part is
    /// left alone because DC removal only touched the real part.
    fn quantize(src: &[(f64, f64)], out: &mut [(f32, f32)]) {
        let n = src.len();
        let half = n / 2;
        let mut err = 0.0f64;
        for i in half..n {
            let (r, im) = src[i];
            let w = if i == half { 1.0 } else { 2.0 };
            let rr = (r + err / w) as f32;
            err += w * (r - rr as f64);
            let ii = im as f32;
            out[i] = (rr, ii);
            out[n - 1 - i] = (rr, -ii);
        }
    }

    /// H(w0) of centered Hermitian taps. Real by symmetry.
    fn gain_at(out: &[(f64, f64)], w0: f64) -> f64 {
        let n = out.len();
        let half = n / 2;
        let (s, c) = w0.sin_cos();
        let (mut cr, mut ci) = (1.0f64, 0.0f64);
        let mut acc = out[half].0;
        for &(r, im) in &out[half + 1..] {
            let (nr, ni) = (cr * c - ci * s, cr * s + ci * c);
            cr = nr;
            ci = ni;
            acc += 2.0 * (r * cr + im * ci);
        }
        acc
    }
}

/// A [`Plan`] carrying the two reassignment spectra.
pub struct ReassignPlan {
    plan: Plan,
    spec: Vec<[f64; 3]>, // [psi, d, t] per grid point
}

impl ReassignPlan {
    /// Writes psi, pitch weights, and time weights for `bin`. All three output
    /// slices must be `bin.taps()`.
    ///
    /// Centered, peak gain 2, DC removed.
    ///
    /// `d` and `t` are to be multiplied by `i` at use.
    pub fn taps_into(
        &mut self,
        bin: Bin,
        psi: &mut [(f32, f32)],
        d: &mut [(f32, f32)],
        t: &mut [(f32, f32)],
    ) {
        let n = bin.taps;
        let mut buf = self.plan.take_buf(3, n);
        {
            let (bp, rest) = buf.split_at_mut(n);
            let (bd, bt) = rest.split_at_mut(n);
            self.transform3(bin, bp, bd, bt);

            // d and t are re-weightings of psi.  Taper each on its own:
            // windowing in time convolves each spectrum with the taper kernel, so d = w*psi holds to
            // the kernel's second moment.  Differentiating tapered taps instead would add a
            // product-rule term carrying the taper's derivative, a first-order bump sitting exactly
            // where the reassignment weights should be going quiet.
            self.plan.taper(bp);
            self.plan.taper(bd);
            self.plan.taper(bt);

            Plan::center(bp);
            Plan::center(bd);
            Plan::center(bt);

            let norm = PEAK_GAIN / Plan::gain_at(bp, bin.w0);
            let axis = bin.w0 / self.plan.peak;

            Plan::scale_by(bp, norm);
            Plan::scale_by(bd, norm * axis);
            Plan::scale_by(bt, norm / axis);

            for (out, src) in [(&mut *psi, &*bp), (&mut *d, &*bd), (&mut *t, &*bt)] {
                Plan::quantize(src, out);
            }
        }
        self.plan.buf = buf;
    }

    /// Rotor walk over the interleaved spectra, one pass for all three outputs.
    /// Takes advantage of odd symmetry.
    fn transform3(
        &self,
        bin: Bin,
        psi: &mut [(f64, f64)],
        d: &mut [(f64, f64)],
        t: &mut [(f64, f64)],
    ) {
        let n = psi.len();
        let half = n / 2;
        let step = self.plan.du * bin.w0;

        let (ss, sc) = Plan::seed_step(step, 1);
        let (mut sr, mut si) = (1.0f64, 0.0f64);

        for i in half..n {
            let dt = step * (i - half) as f64;
            let (ds, dc) = dt.sin_cos();
            let (mut cr, mut ci) = (sr, si);
            let (mut a0, mut a1, mut a2) = ((0.0f64, 0.0f64), (0.0f64, 0.0f64), (0.0f64, 0.0f64));

            for &[sp, sd, st] in &self.spec[1..] {
                a0 = (a0.0 + sp * cr, a0.1 + sp * ci);
                a1 = (a1.0 + sd * cr, a1.1 + sd * ci);
                a2 = (a2.0 + st * cr, a2.1 + st * ci);
                let (nr, ni) = (cr * dc - ci * ds, cr * ds + ci * dc);
                cr = nr;
                ci = ni;
            }

            for (out, acc) in [(&mut *psi, a0), (&mut *d, a1), (&mut *t, a2)] {
                out[i] = acc;
                out[n - 1 - i] = (acc.0, -acc.1);
            }

            let (nr, ni) = (sr * sc - si * ss, sr * ss + si * sc);
            sr = nr;
            si = ni;
        }
    }

    /// See [`Plan::bin`].
    pub fn bin(&self, center: f64, rate: f64) -> Bin {
        self.plan.bin(center, rate)
    }

    fn du(&self) -> f64 {
        self.plan.du
    }

    fn peak(&self) -> f64 {
        self.plan.peak
    }
}

/// Grid indices bracketing the spectrum above `eps`: `[lo, m)`.
/// g(u) = beta*ln(u) - (beta/gamma)*u^gamma, normalized so g(1) = 0.
fn support(shape: Shape, du: f64, eps: f64) -> (usize, usize) {
    let (beta, bg) = (shape.beta, shape.beta / shape.gamma);
    let le = eps.ln();
    let g = |u: f64| beta * u.ln() - bg * u.powf(shape.gamma) + bg;

    let mut j = 1usize;
    while g(j as f64 * du) < le {
        j += 1;
    }
    let lo = j;
    while g(j as f64 * du) >= le {
        j += 1;
    }
    (lo, j + 1)
}

fn fill_grid(shape: &Shape, du: f64, lo: usize, psi: &mut [f64]) {
    let (beta, gamma) = (shape.beta, shape.gamma);
    let bg = beta / gamma;
    for (j, p) in psi.iter_mut().enumerate().skip(lo) {
        let u = j as f64 * du;
        *p = (beta * u.ln() - bg * u.powf(gamma) + bg).exp();
    }
}

/// Half width in samples, times omega0. Pure function of shape and leakage.
fn half_width_scaled(s: Shape, eps: f64, tail_a: f64) -> f64 {
    let core = (2.0 * eps.recip().ln()).sqrt() * s.p();
    let tail = (tail_a / eps).powf(1.0 / (2.0 * s.beta + 1.0));
    core.max(tail)
}

#[cfg(test)]
mod test {
    use core::f64::consts::{LN_2, PI};

    use super::*;

    const BINS: usize = 1024;
    const RATE: f64 = 48_000.0;

    fn spec(q: f64, eps: f64) -> Spec {
        Spec::default().shape(Shape::from_q(q, 3.0)).eps(eps)
    }

    fn taps(plan: &mut Plan, bin: Bin) -> Vec<(f32, f32)> {
        let mut t = vec![(0.0, 0.0); bin.taps()];
        plan.taps_into(bin, &mut t);
        t
    }

    fn mag((re, im): (f32, f32)) -> f64 {
        let (re, im) = (re as f64, im as f64);
        (re * re + im * im).sqrt()
    }

    /// |H(w)| of centered taps, w in rad/sample.
    fn dtft(taps: &[(f32, f32)], w: f64) -> f64 {
        let half = (taps.len() / 2) as isize;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (j, &(r, i)) in taps.iter().enumerate() {
            let (s, c) = (w * (j as isize - half) as f64).sin_cos();
            let (r, i) = (r as f64, i as f64);
            re += r * c + i * s;
            im += -r * s + i * c;
        }
        (re * re + im * im).sqrt()
    }

    /// Bisect for |H| = target on [a, b], target bracketed.
    fn crossing(taps: &[(f32, f32)], mut a: f64, mut b: f64, target: f64) -> f64 {
        let above = dtft(taps, a) > target;
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            if (dtft(taps, m) > target) == above {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    }

    /// Signed bar, zero at column `cols`.
    fn bar(v: f64, max: f64, cols: usize) -> String {
        let col = ((v / max) * cols as f64).round() as isize;
        let (pad, fill) = if col >= 0 {
            (cols, col as usize)
        } else {
            ((cols as isize + col) as usize, (-col) as usize)
        };
        format!("{}{}", " ".repeat(pad), "#".repeat(fill))
    }

    /// Real part of `taps`, centered, scaled to the largest magnitude present.
    fn print_wave(label: &str, taps: &[(f32, f32)], cols: usize) {
        let n = taps.len();
        let max = taps
            .iter()
            .map(|&(r, _)| (r as f64).abs())
            .fold(0.0, f64::max);
        println!("\n=== {label} ({n} taps) ===");
        for (j, &(re, _)) in taps.iter().enumerate() {
            println!(
                "{:>6} {:>12.7} {}",
                j as isize - (n / 2) as isize,
                re,
                bar(re as f64, max, cols)
            );
        }
    }

    #[test]
    fn print_gamma_sweep() {
        println!("\n=== ENVELOPE vs GAMMA (Q = 2.4) ===");
        // P = 4.0 is Q = 2.4; holding it fixed keeps the -3 dB width constant across gamma.
        let p = 4.0;
        for gamma in [1.0f64, 2.0, 3.0, 6.0] {
            let mut plan = Spec::default()
                .shape(Shape {
                    gamma,
                    beta: p * p / gamma,
                })
                .plan();
            let bin = plan.bin(1000.0, 8000.0);
            let t = taps(&mut plan, bin);
            let n = bin.taps();

            let mags: Vec<f64> = t.iter().copied().map(mag).collect();
            let max = mags.iter().fold(0.0f64, |a, &b| a.max(b));
            let ctr = (n / 2) as f64;
            let m: f64 = mags
                .iter()
                .enumerate()
                .map(|(j, &v)| (j as f64 - ctr) * v * v)
                .sum();
            let e: f64 = mags.iter().map(|v| v * v).sum();

            println!(
                "\ngamma = {:.1}  taps {}  centroid offset = {:+.3}",
                gamma,
                n,
                m / e
            );
            for (j, &v) in mags.iter().enumerate() {
                println!(
                    "{:>4} {}",
                    j as isize - (n / 2) as isize,
                    "#".repeat((v / max * 40.0).round() as usize)
                );
            }
            assert!((m / e).abs() < 1e-3, "gamma {gamma} centroid {:+.4}", m / e);
        }
    }

    #[test]
    fn print_spectrum() {
        let mut plan = spec(3.0, 1e-6).plan_with_reassignment();

        println!(
            "\n=== SPECTRUM ({} grid points, u = w/w_peak) ===",
            plan.spec.len()
        );
        println!(
            "{:>4} {:>8} {:>8} {:>12} {:>12} {:>12}",
            "j", "u", "omega", "psi", "dpsi", "tpsi"
        );
        for (j, &[p, pd, pt]) in plan.spec.iter().enumerate() {
            if p < 1e-7 {
                continue;
            }
            let u = j as f64 * plan.du();
            println!(
                "{:>4} {:>8.4} {:>8.4} {:>12.6} {:>12.6} {:>12.6}",
                j,
                u,
                plan.peak() * u,
                p,
                pd,
                pt
            );
        }

        let pk = (1..plan.spec.len())
            .max_by(|&a, &b| plan.spec[a][0].total_cmp(&plan.spec[b][0]))
            .unwrap();
        let upk = pk as f64 * plan.du();
        println!("peak at u = {:.4}, wanted 1.0", upk);

        // The grid is sampled, so the argmax can only land within half a step of 1.0.
        assert!(
            (upk - 1.0).abs() <= plan.du(),
            "peak u {upk:.6} du {:.6}",
            plan.du()
        );

        // d = w*psi exactly
        for j in 1..plan.spec.len() {
            let [p, d, _] = plan.spec[j];
            let w = plan.peak() * j as f64 * plan.du();
            assert!((d - p * w).abs() <= 1e-12 * (p * w).abs(), "d at {j}");
        }

        // t = dpsi/dw, checked by central difference.  This catches sign and
        // factor errors, not small ones.
        let dw = plan.peak() * plan.du();
        let pmax = plan.spec.iter().map(|s| s[0]).fold(0.0f64, f64::max);
        let tmax = plan.spec.iter().map(|s| s[2].abs()).fold(0.0f64, f64::max);

        let mut checked = 0;
        for j in 1..plan.spec.len() - 1 {
            let t = plan.spec[j][2];
            if plan.spec[j][0] < 0.5 * pmax {
                continue;
            }
            let fd = (plan.spec[j + 1][0] - plan.spec[j - 1][0]) / (2.0 * dw);
            assert!(
                (t - fd).abs() < 0.10 * tmax,
                "t at {j}: analytic {t:.6e} finite-diff {fd:.6e}"
            );
            checked += 1;
        }
        assert!(checked >= 3, "only {checked} points above half-peak");
    }

    /// Bakes the full bank at production-ish scale.
    ///
    /// ```text
    /// cargo test --release wavelet::test::bake_bank -- --ignored --nocapture
    /// ```
    // NEXT this should be a benchmark, but we don't have any set up.  Most of our GPU driven world
    // will not care about the host code.  But faster, less UI delay, and lower power is always
    // better.
    #[test]
    #[ignore]
    fn bake_bank() {
        use mutate_lib::dsp::bank;

        let start = std::time::Instant::now();
        let mut plan = Spec::default()
            .max_taps(4096)
            .shape(Shape::from_q(20.0, 3.0))
            .plan();
        let bins = bank::bins(2_000.0, 20_000.0, BINS);
        println!("planning time: {:?}µs", start.elapsed().as_micros());

        let voices: Vec<Bin> = bins
            .iter()
            .map(|b| b.center)
            .map(|c| plan.bin(c, RATE))
            .collect();

        let total: usize = voices.iter().map(|b| b.taps()).sum();
        let mut taps = vec![(0.0f32, 0.0f32); total];
        let mut offsets = Vec::with_capacity(voices.len());

        let mut cursor = 0;

        let start = std::time::Instant::now();
        for &bin in &voices {
            let n = bin.taps();
            offsets.push(cursor);
            plan.taps_into(bin, &mut taps[cursor..cursor + n]);
            cursor += n;
        }

        let elapsed = start.elapsed();
        let worst = voices
            .iter()
            .zip(offsets.iter())
            .map(|(b, &o)| (dtft(&taps[o..o + b.taps()], b.velocity()) - PEAK_GAIN).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-3, "worst peak gain error {worst:.3e}");

        let n = voices[0].taps();
        print_wave(
            &format!(
                "LOWEST BIN ({:.0}Hz, omega0 {:.5})",
                bins[0].center,
                voices[0].velocity()
            ),
            &taps[..n],
            30,
        );

        println!(
            "voices {} of {}  taps {}  longest {}  shortest {}",
            voices.len(),
            BINS,
            total,
            voices[0].taps(),
            voices[voices.len() - 1].taps(),
        );

        println!("bin filling time: {:?}µs", elapsed.as_micros());
    }

    /// psi from transform3 must match transform, which starts at `lo`.
    /// Also the cheapest place to see the reassignment envelopes.
    #[test]
    fn transform3_matches_transform() {
        let base = spec(2.5, 1e-6);
        let mut plain_plan = base.plan();
        let bin = plain_plan.bin(2_000.0, RATE);
        let n = bin.taps();

        let mut plain = vec![(0.0f32, 0.0f32); n];
        plain_plan.taps_into(bin, &mut plain);

        let mut plan = base.plan_with_reassignment();

        let (mut psi, mut d, mut t) = (
            vec![(0.0f32, 0.0f32); n],
            vec![(0.0f32, 0.0f32); n],
            vec![(0.0f32, 0.0f32); n],
        );
        plan.taps_into(bin, &mut psi, &mut d, &mut t);

        print_wave("PSI (transform3)", &psi, 30);
        print_wave("D = w*psi", &d, 30);
        print_wave(
            "T = dpsi/dw (bipolar spectrum, expect a node at center)",
            &t,
            30,
        );
        print_wave("PLAIN (transform)", &plain, 30);

        let skew = plain
            .iter()
            .zip(psi.iter())
            .map(|(a, b)| (a.0 - b.0).abs().max((a.1 - b.1).abs()) as f64)
            .fold(0.0, f64::max);
        println!(
            "omega0 {:.5}  taps {}  psi max diff {:.3e}",
            bin.velocity(),
            n,
            skew
        );

        let worst = plain
            .iter()
            .zip(psi.iter())
            .enumerate()
            .max_by(|a, b| {
                let f = |(_, (p, q)): &(usize, (&(f32, f32), &(f32, f32)))| {
                    (p.0 - q.0).abs().max((p.1 - q.1).abs())
                };
                f(a).total_cmp(&f(b))
            })
            .unwrap();
        println!("worst at {:+}", worst.0 as isize - (n / 2) as isize);

        let pk = |v: &[(f32, f32)]| v.iter().map(|&(r, _)| r.abs()).fold(0.0f32, f32::max);
        let peak = pk(&plain);
        println!("peak plain {:.7}  peak psi {:.7}", peak, pk(&psi));

        let differing = plain.iter().zip(psi.iter()).filter(|(a, b)| a != b).count();
        let dc = |v: &[(f32, f32)]| v.iter().map(|&(r, _)| r as f64).sum::<f64>();
        println!(
            "differing taps {differing} of {n}  dc plain {:.3e}  dc psi {:.3e}",
            dc(&plain),
            dc(&psi)
        );
        assert_eq!(plain, psi.as_slice(), "psi paths diverge");

        // t is the bipolar spectrum; its real part passes through zero at the center.
        let half = n / 2;
        assert!(
            (t[half].0 as f64).abs() < 1e-2 * pk(&t) as f64,
            "t center {:.3e} peak {:.3e}",
            t[half].0,
            pk(&t)
        );

        for (label, v) in [("d", &d), ("t", &t)] {
            let pk = v.iter().map(|&(r, _)| r.abs()).fold(0.0f32, f32::max) as f64;
            let dc = v.iter().map(|&(r, _)| r as f64).sum::<f64>();
            assert!(dc.abs() < 1e-5 * pk, "{label} dc {dc:.3e} peak {pk:.3e}");
        }
    }

    /// Both rotors against direct evaluation. Agreement with each other follows.
    #[test]
    fn rotor_vs_reference() {
        let mut plan = spec(20.0, 1e-8).plan_with_reassignment();

        let bin = plan.bin(12_000.0, RATE);
        let n = bin.taps();
        let (half, step) = (n / 2, plan.du() * bin.w0);

        let mut want = [
            vec![(0.0f64, 0.0f64); n],
            vec![(0.0f64, 0.0f64); n],
            vec![(0.0f64, 0.0f64); n],
        ];
        for i in half..n {
            let d = step * (i - half) as f64;
            let mut acc = [(0.0f64, 0.0f64); 3];
            for (j, &spec) in plan.spec.iter().enumerate() {
                let (s, c) = (j as f64 * d).sin_cos();
                for (a, v) in acc.iter_mut().zip(spec) {
                    *a = (a.0 + v * c, a.1 + v * s);
                }
            }
            for (w, a) in want.iter_mut().zip(acc) {
                w[i] = a;
                w[n - 1 - i] = (a.0, -a.1);
            }
        }

        let e = |a: &[(f64, f64)], b: &[(f64, f64)]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x.0 - y.0).abs().max((x.1 - y.1).abs()))
                .fold(0.0f64, f64::max)
        };

        let mut got = vec![(0.0f64, 0.0f64); n];
        // 🤮 Anyway...
        plan.plan.transform(bin, &mut got);
        assert!(
            e(&want[0], &got) < 1e-9,
            "transform {:.3e}",
            e(&want[0], &got)
        );

        let (mut gp, mut gd, mut gt) = (
            vec![(0.0f64, 0.0f64); n],
            vec![(0.0f64, 0.0f64); n],
            vec![(0.0f64, 0.0f64); n],
        );
        plan.transform3(bin, &mut gp, &mut gd, &mut gt);
        for (label, w, g) in [
            ("psi", &want[0], &gp),
            ("d", &want[1], &gd),
            ("t", &want[2], &gt),
        ] {
            let err = e(w, g);
            // d carries a factor of w ~ peak, so scale the bound by the magnitude present.
            let scale = w
                .iter()
                .map(|s| s.0.abs().max(s.1.abs()))
                .fold(1.0f64, f64::max);
            assert!(
                err < 1e-9 * scale,
                "transform3 {label} {err:.3e} scale {scale:.3e}"
            );
        }
    }

    /// Smoke test.  Hermitian, DC-free.
    #[test]
    fn taps_are_conditioned() {
        let mut p = spec(3.0, 1e-8).plan();
        for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 8000.0), (12_000.0, RATE)] {
            let bin = p.bin(fc, sr);
            let t = taps(&mut p, bin);
            let n = bin.taps();

            for j in 0..n / 2 {
                let (a, b) = (t[j], t[n - 1 - j]);
                assert_eq!(a.0, b.0);
                assert_eq!(a.1, -b.1);
            }

            let g = dtft(&t, bin.velocity());
            let dc = t.iter().map(|&(r, _)| r as f64).sum::<f64>();
            assert!((g - PEAK_GAIN).abs() < 1e-3, "fc {fc} peak gain {g:.6}");
            assert!(dc.abs() < 1e-5 * g, "fc {fc} dc {dc:.3e}");
        }
    }

    /// The four numbers a spectrogram actually depends on: where the bin sits,
    /// how wide it is, how much leaks to negative frequency, and the stopband floor.
    #[test]
    fn response_is_characterized() {
        const SWEEP: usize = 8192;

        let (q, eps, eps_time, rho) = (2.5, 1e-8, 1e-3, 0.1);
        let mut p = spec(q, eps).taper(Taper { eps_time, rho }).plan();

        fn ref_rel_width(s: Shape) -> f64 {
            let (beta, gamma) = (s.beta, s.gamma);
            let g = |x: f64| beta * x - (beta / gamma) * ((gamma * x).exp() - 1.0);
            let target = -LN_2 / 2.0;
            // Lol.. we'll just solve that real quick eh?
            let root = |mut a: f64, mut b: f64| {
                for _ in 0..80 {
                    let m = 0.5 * (a + b);
                    if g(m) > target {
                        a = m
                    } else {
                        b = m
                    }
                }
                0.5 * (a + b)
            };
            root(0.0, 2.0).exp() - root(0.0, -2.0).exp()
        }

        let omega = |k: usize| -PI + 2.0 * PI * k as f64 / SWEEP as f64;

        let want_rel = 2.0 * LN_2.sqrt() / p.shape.p();
        let exact_rel = ref_rel_width(p.shape);
        let mut widths = Vec::new();

        println!(
            "\n=== RESPONSE (Q = {q}, gamma = {}, eps = {eps:.0e}, taper {eps_time:.0e} rho {rho}, \
             rel width exact {exact_rel:.5} gaussian {want_rel:.5}) ===",
            p.shape.gamma
        );

        for (fc, sr) in [(1000.0f64, 6000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = p.bin(fc, sr);
            let t = taps(&mut p, bin);
            let w0 = bin.velocity();

            println!(
                "\nfc {:>7.0} sr {:>6.0}  taps {:>5}  w0 {:.6}",
                fc,
                sr,
                bin.taps(),
                w0
            );

            // Coarse sweep: peak location, negative-frequency leakage.
            let mut peak = (0.0f64, 0.0f64);
            let mut neg = 0.0f64;
            for k in 0..=SWEEP {
                let w = omega(k);
                let h = dtft(&t, w);
                if w < 0.0 {
                    neg = neg.max(h);
                }
                if h > peak.1 {
                    peak = (w, h);
                }
            }

            // Refine the peak on the coarse cell, then the half-power edges.
            let cell = 2.0 * PI / SWEEP as f64;
            let (mut a, mut b) = (peak.0 - cell, peak.0 + cell);
            for _ in 0..80 {
                let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
                if dtft(&t, m1) < dtft(&t, m2) {
                    a = m1;
                } else {
                    b = m2;
                }
            }
            let wpk = 0.5 * (a + b);
            let hpk = dtft(&t, wpk);
            println!(
                "  peak gain {:.9}  dev {:+.3e} rel",
                hpk,
                hpk / PEAK_GAIN - 1.0
            );

            let half = hpk / 2.0f64.sqrt();
            let lo = crossing(&t, wpk - w0, wpk, half);
            let hi = crossing(&t, wpk, (wpk + w0).min(PI), half);
            let rel = (hi - lo) / wpk;

            // Stopband: everything past three half-power widths from the peak.
            let guard = 3.0 * (hi - lo);
            let mut floor = 0.0f64;
            for k in 0..=SWEEP {
                let w = omega(k);
                if (w - wpk).abs() > guard {
                    floor = floor.max(dtft(&t, w));
                }
            }

            // Detuning in cents against w0, and as a fraction of the half-power half-width.
            // Both are yardsticks: 1 cent is inaudible, and 1.0 here would put the argmax
            // on the -3 dB edge.
            let cents = 1200.0 * (wpk / w0).log2();
            let detune = (wpk - w0) / (0.5 * rel * wpk);
            println!(
                "  rel width {:.5}  exact {:.5}  ratio {:.5}",
                rel,
                exact_rel,
                rel / exact_rel
            );
            println!("  peak {:+.4} cents  {:+.2e} of half-width", cents, detune);
            widths.push(rel);

            let db = |v: f64| 20.0 * (v / hpk).log10();

            println!("  negative-freq max {:>8.2} dB", db(neg));
            println!("  stopband floor    {:>8.2} dB", db(floor));
            println!(
                "  peak gain {:.5}  analytic {:.5}  ratio {:.5}",
                hpk,
                PEAK_GAIN,
                hpk / PEAK_GAIN
            );

            assert!(
                (hpk / PEAK_GAIN - 1.0).abs() < 1e-5,
                "fc {fc} peak gain {hpk:.9}"
            );

            assert!(
                cents.abs() < 1.0,
                "fc {fc} peak {cents:+.4} cents off center"
            );
        }

        let lo = widths.iter().copied().fold(f64::MAX, f64::min);
        let hi = widths.iter().copied().fold(0.0f64, f64::max);
        assert!(
            hi / lo - 1.0 < 2e-3,
            "rel width varies across rates: {lo:.5} to {hi:.5}"
        );
    }

    /// A real unit tone reads |W| = 1 even though |H| = 2: the analytic taps
    /// see only the +w half of the cosine.
    #[test]
    fn unit_tone_reads_unity() {
        let mut p = spec(3.0, 1e-8).plan();
        let bin = p.bin(1000.0, 8000.0);
        let t = taps(&mut p, bin);
        let (n, w0) = (bin.taps(), bin.velocity());

        // taps are centered, so m is the sample under tap index n/2.
        for m in 0..8 {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (j, &(r, i)) in t.iter().enumerate() {
                let x = (w0 * (m as isize + (n / 2) as isize - j as isize) as f64).cos();
                re += x * r as f64;
                im += x * i as f64;
            }
            let env = (re * re + im * im).sqrt();
            assert!((env - 1.0).abs() < 1e-3, "phase {m} envelope {env:.6}");
        }
    }

    /// Peak-normalized constant-Q puts noise gain proportional to center
    /// frequency: length goes as 1/w0, amplitude as 1/N, so energy tracks w0.
    /// White noise therefore floors at a fixed level per bin once w0 is divided out.
    #[test]
    fn noise_gain_tracks_center() {
        // Tap count is an integer, so envelope truncation loses O(1/N) of the
        // energy, worst at the top of the range. Anchored mid-range so the
        // deviation splits either side.
        const NOISE_GAIN: f64 = 0.224814;
        const TOL: f64 = 1e-3;

        let mut p = spec(3.0, 1e-8)
            .taper(Taper {
                eps_time: 1e-3,
                rho: 0.1,
            })
            .plan();

        println!("\n=== NOISE GAIN (Q = 3, sr = {RATE}) ===");

        for fc in [500.0f64, 1000.0, 2000.0, 4000.0, 8000.0] {
            let bin = p.bin(fc, RATE);
            let t = taps(&mut p, bin);
            let e: f64 = t
                .iter()
                .map(|&(r, i)| (r as f64).powi(2) + (i as f64).powi(2))
                .sum();
            let ratio = e / bin.velocity();

            println!(
                "  fc {:>6.0}  taps {:>5}  energy {:.6}  e/w0 {:.6}  dev {:+.2e}",
                fc,
                bin.taps(),
                e,
                ratio,
                ratio / NOISE_GAIN - 1.0
            );

            assert!(
                (ratio / NOISE_GAIN - 1.0).abs() < TOL,
                "fc {fc} noise gain {ratio:.6}"
            );
        }
    }
}
