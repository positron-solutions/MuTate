// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # The Wavelet
//!
//! > The traveler who fears the unknown road will eventually learn that known roads return
//! > to where they began.
//! >
//! > - Anthony L. Ray
//!
//!                 ●
//!                 ●
//!                 •
//!                 ·
//!                ●··
//!               ●●·
//!              ●●●
//!              ●••
//!             ··••
//!            ·····●
//!            ·····●●●●●
//!                ·●●●●●●●
//!                 •••●●●●●
//!                 ••••••··
//!                 ···········
//!          ●●●●●●●··········
//!     ●●●●●●●●●●●●·····
//!   ●●●●●●●●●●●•••
//!      •••••••••••
//! ·············•••
//!  ···············●●●●●●
//!        ·········●●●●●●●●●●●●●●
//!                 ●●●●●●●●●●●●●●●●●
//!                 •••••••••●●●●●
//!                 ••••••·········
//!              ●●●················
//!      ●●●●●●●●●●●···········
//!   ●●●●●●●●●●●●●●···
//!     ●●●●●●●•••••
//!       ···•••••••
//!      ···········
//!         ········●●●●●●
//!              ···●●●●●●●●
//!                 •●●●●●●
//!                 •••••
//!                 •····
//!               ●●····
//!              ●●●··
//!              ●●●
//!               ●•
//!               ·•
//!                ·
//!                ·●
//!                 ●
//!                 ●
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
//! One [`Plan`] per Q. It holds the spectrum in a frequency independent form, so every voice
//! sharing that Q may reuse one `Plan`.
//!
//! ```
//! # use mutate::wavelet::{Spec, Taper};
//! const QUANTUM: usize = 4;
//!
//! let mut plan = Spec::default()
//!     .taper(Taper { eps_time: 1e-3, rho: 0.0 })
//!     .max_load_quantum(QUANTUM)
//!     .plan();
//!
//! let bin = plan.bin(1000.0, 8000.0);
//! let mut weights = vec![[0.0f32; 4]; bin.folded_taps(QUANTUM)];
//! plan.taps_into(bin, QUANTUM, &mut weights);
//! ```
//!
//! The `max_load_quantum` adjusts truncation and conditioning to land the mirrored tap length
//! (excluding center tap!) on a quantum matching the size you intend to load.  The center tap is
//! usually loaded individually for a final reduction.
//!
//! For example, if the implementation loads eight weights at a time on mirrored taps, the
//! half-weight size is eight.  If truncation for the spectrum would lead to 17 folded weights, 24
//! will be used instead. The extra weights are used to truncate & shape less aggressively,
//! increasing precision and hopefully de-correlating some bias.
//!
//! ```
//! # use mutate::wavelet::{Spec, Taper};
//! const QUANTUM: usize = 8;
//!
//! let mut plan = Spec::default()
//!     .taper(Taper { eps_time: 1e-3, rho: 0.0 })
//!     .max_load_quantum(QUANTUM)
//!     .plan();
//!
//! let bin = plan.bin(1000.0, 8000.0);
//! let mut weights = vec![[0.0f32; 4]; bin.folded_taps(QUANTUM)];
//!
//! // float4(Re ψ, Im ψ, Re d, Im d); index 0 is the center, with Re halved.
//! plan.taps_into(bin, QUANTUM, &mut weights);
//! ```
//!
//! ## Weight Table Format
//!
//! Hermitian folded with derived t.  Exploit symmetry and derive weight components that are cheaper
//! (and more accurate) to derive than to load.  Taps are odd and `ψ`, d, and `t` are all Hermitian.
//!
//! ### Storage Format
//!
//! - Weight `k` is `float4( Re ψ_k, Im ψ_k, Re d_k, Im d_k )`
//! - The center tap has a slightly simpler representation.  `float4( Re ψ₀/2, 0, Re d₀/2, 0 )`
//!
//! `t` is derived.  About a center sample `m`, lanes will own the pair `k = |ν|` from `m`.  For
//! weight `ν`, `t_ν = −i·ν·ψ_ν`.
//!
//! ### Compute & Register Cost
//!
//! - 5 ops per tap
//! - 4 f32s per 2 mirrored taps, 2 per tap
//! - 3 (W_ψ, W_d, W_t) complex accumulators, 6 registers per pipelined hop.
//!
//! For `P` pipelines, we need `6P + 6 + 2` non-uniform registers per lane and some scratch for
//! temporaries.  Audio is 8 bytes per sample at two channels and each 8 bytes read can be re-used
//! with `P` weights for each read.
//!
//! `d` carries `w0/peak` so its ratio against ψ reads in rad/sample. `t` derived this way
//! reads in samples with no scaling.

// NEXT Offline SOCP oracle to see what we're leaving on the table with our approximations.
// MAYBE More deliberate taper shaping, using taper shape to ceiling and distribute any halo the
// taper introduces.  Shape-aware taper.  Length padding-aware truncation & taper.  Try to land the
// taper where it won't slice a half period?
// MAYBE Integrate late quantization to bias rounding towards filter precision.
// NOTE We have logarithmic bin spacings, but the cutoff frequencies that determine which downsample
// will be used are not particularly aware, so it's not expected that we can re-use exact bins in
// any kind of octave structure.  Mel scaling etc also defeats this, so there's no point.
// NEXT Run time of the bin generation test (not reflective of actual sample rates and Q) is about
// 17ms on a Zen2+ part.  This affects CWT startup time.
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
// === TABLE RESPONSE (Q = 3.5, quantum 4) ===
//
// fc    1000 sr   6000  weights    25 (unfolded    49)  w0 1.047198
//   peak gain 1.999999982  dev -8.799e-9 rel
//   rel width 0.28523
//   peak -0.0047 cents
//   negative-freq max  -104.02 dB
//   stopband floor     -101.53 dB
//
// fc     250 sr   3000  weights    49 (unfolded    97)  w0 0.523599
//   peak gain 1.999999993  dev -3.306e-9 rel
//   rel width 0.28524
//   peak -0.0071 cents
//   negative-freq max  -103.05 dB
//   stopband floor      -98.91 dB
//
// fc   12000 sr  48000  weights    17 (unfolded    33)  w0 1.570796
//   peak gain 2.000000014  dev +7.186e-9 rel
//   rel width 0.28523
//   peak -0.0030 cents
//   negative-freq max  -104.40 dB
//   stopband floor     -104.40 dB

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
    /// Rotational velocity ദ്ദി(•̀ω-)✧ in radians per sample.
    pub fn velocity(&self) -> f64 {
        self.w0
    }

    /// Compute storage for folded taps with the center tap, rounding up for the load quantum.  If
    /// your usage will load taps four at a time, the load quantum is four.
    pub fn folded_taps(&self, load_quantum: usize) -> usize {
        (self.taps / 2).div_ceil(load_quantum) * load_quantum + 1
    }

    /// Compute the effective taps after unfolding, including the center tap.  The effect of the
    /// load quantum is doubled due to mirroring
    pub fn unfolded_taps(&self, load_quantum: usize) -> usize {
        2 * self.folded_taps(load_quantum) - 1
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
            // cbrt is just the fast path
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
    // Error tolerance for the spectrum.
    eps: f64,
    tail_a: f64,
    taper: Option<Taper>,
    max_taps: usize,
    max_load_quantum: usize,
    sobolev: f64,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            shape: Shape::from_q(3.0, 3.0),
            eps: 1e-8,
            tail_a: 1.0,
            taper: None,
            max_taps: 0,
            max_load_quantum: 1,
            sobolev: 2.0,
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

    /// Largest load quantum any bake will pass in. Sizes the rotor grid so the emitted span stays
    /// clear of the time-domain replica.
    pub fn max_load_quantum(mut self, q: usize) -> Self {
        self.max_load_quantum = q;
        self
    }

    /// Sobolev weight on the conditioning correction, dimensionless against the emitted
    /// half-span. The correction minimizes `∫|E|² + λ|E'|²` of its own spectral footprint,
    /// so raising this pulls the correction toward the center tap and flattens the error
    /// slope that reassignment divides through. 0.0 is the plain minimum-norm correction.
    /// Above roughly 8.0 the Gram solution gets soft and the stopband starts paying for it.
    pub fn sobolev(mut self, sigma: f64) -> Self {
        self.sobolev = sigma;
        self
    }

    /// Half-span that sets tap count, and the grid extent that feeds it.
    fn spans(&self) -> (f64, f64) {
        let grid = half_width_scaled(self.shape, self.eps, self.tail_a);
        let taps = match self.taper {
            Some(t) => half_width_scaled(self.shape, t.eps_time, self.tail_a),
            None => grid,
        };

        // Quantum rounding plus the center tap extend the emitted span by up to q+1 samples,
        // worst at w0 = PI in scaled units.
        let pad = (self.max_load_quantum + 1) as f64 * core::f64::consts::PI;

        // Rectangle rule on a uniform grid aliases rather than truncates: the error is the
        // time-domain replica at period TAU/du. Placing it a full eps half-width past the
        // emitted span puts the fold-back at eps.
        (taps, core::f64::consts::TAU / (taps + pad + grid.max(taps)))
    }

    pub fn plan(self) -> Plan {
        let (c, du) = self.spans();
        let du = snap(du);
        let peak = self.shape.peak();
        let (lo, m) = support(self.shape, du, self.eps);

        // d = w*psi shares psi's support, so lo bounds both.
        let env = log_env(self.shape);
        let mut spec = vec![[0.0; 2]; m];
        for (j, s) in spec.iter_mut().enumerate().skip(lo) {
            let u = j as f64 * du;
            let p = env(u).exp();
            *s = [p, p * peak * u];
        }

        Plan {
            peak,
            c,
            du,
            spec,
            lo,
            rho: self.taper.map_or(0.0, |t| t.rho),
            sobolev: self.sobolev,
            buf: Vec::with_capacity(2 * (self.max_taps / 2 + self.max_load_quantum + 1)),
        }
    }
}

/// Taper produces a ramp from the design rho out.  Centering re-uses the ramp to avoid
/// re-introducing what the ramp tapered away.
#[derive(Clone, Copy)]
struct Ramp {
    start: f64,
    inv: f64,
}

impl Ramp {
    fn new(end: f64, width: f64) -> Self {
        Ramp {
            start: end - width,
            inv: width.recip(),
        }
    }

    /// Identity window; `x` pins to 0 for every index.
    fn none() -> Self {
        Ramp {
            start: 0.0,
            inv: 0.0,
        }
    }

    fn at(&self, i: usize) -> f64 {
        let x = (i as f64 - self.start) * self.inv;
        if x <= 0.0 {
            1.0
        } else if x >= 1.0 {
            0.0
        } else {
            1.0 / (1.0 + (1.0 / (1.0 - x) - 1.0 / x).exp())
        }
    }

    fn apply(&self, out: &mut [(f64, f64)]) {
        for (i, s) in out.iter_mut().enumerate() {
            let w = self.at(i);
            s.0 *= w;
            s.1 *= w;
        }
    }
}

// CWT weight table generator.
pub struct Plan {
    peak: f64,            // argmax of w^beta e^{-w^gamma}
    c: f64,               // half_width_scaled
    du: f64,              // uniform step in u = w/w_peak
    spec: Vec<[f64; 2]>,  // [psi, d] at u_j = j*du
    lo: usize,            // first grid point above eps
    rho: f64,             // taper fraction of the emitted half-span; 0.0 disables
    buf: Vec<(f64, f64)>, // bake scratch, 2 spans
    sobolev: f64,
}

impl Plan {
    /// Angular velocity and tap count for a bin at `center`, sampled at `rate`.
    /// `center` above Nyquist will panic in debug.
    pub fn bin(&self, center: f64, rate: f64) -> Bin {
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

    /// Writes `bin.folded_taps(quantum)` folded weights. Peak gain 2, DC removed, tapered over
    /// the quantized half-span. `out.len()` must be `bin.folded_taps(quantum)`.
    pub fn taps_into(&mut self, bin: Bin, quantum: usize, out: &mut [[f32; 4]]) {
        let k = out.len();
        let mut buf = core::mem::take(&mut self.buf);
        buf.resize(2 * k, (0.0, 0.0));
        {
            let (psi, d) = buf.split_at_mut(k);

            self.transform2(bin, psi, d);
            // Ramp width comes from the design span, so it is one shape for every bin and
            // quantum.
            let ramp = match self.rho {
                0.0 => Ramp::none(),
                // Ramp size is calculated from the effective size,
                rho => Ramp::new(out.len() as f64, rho * self.c / bin.w0),
            };

            // Taper ψ and d separately: windowing convolves each spectrum with the taper
            // kernel, so d = w*ψ survives to the kernel's second moment. Tapering ψ and then
            // deriving d would add a product-rule term exactly where d should be going quiet.
            ramp.apply(psi);
            ramp.apply(d);

            self.condition(psi, ramp);
            self.condition(d, ramp);

            let norm = PEAK_GAIN / Self::gain_at(psi, bin.w0);

            Self::scale_by(psi, norm);
            Self::scale_by(d, norm * bin.w0 / self.peak);

            Self::quantize(psi, d, out);
        }
        self.buf = buf;
    }

    /// Rotor walk from the center outward, both spectra in one pass.
    fn transform2(&self, bin: Bin, psi: &mut [(f64, f64)], d: &mut [(f64, f64)]) {
        let step = self.du * bin.w0;

        // NOTE implementation is a phasor walk with Newton's correction.  Got -270dB versus sin_cos
        // and on our very short filters, the phase error accumulates very slowly.  Tests run quite
        // a bit quicker.

        // Outer angles are linear in i: dt advances by step, the j = lo seed by step*lo.
        let (ps, pc) = step.sin_cos();
        let (qs, qc) = (step * self.lo as f64).sin_cos();
        let (mut dc, mut ds) = (1.0f64, 0.0f64);
        let (mut sr, mut si) = (1.0f64, 0.0f64);

        for i in 0..psi.len() {
            let (mut cr, mut ci) = (sr, si);
            let (mut a0, mut a1) = ((0.0f64, 0.0f64), (0.0f64, 0.0f64));

            for &[sp, sd] in &self.spec[self.lo..] {
                a0 = (a0.0 + sp * cr, a0.1 + sp * ci);
                a1 = (a1.0 + sd * cr, a1.1 + sd * ci);
                let (nr, ni) = (cr * dc - ci * ds, cr * ds + ci * dc);
                let k = 0.5 * (3.0 - (nr * nr + ni * ni));
                cr = nr * k;
                ci = ni * k;
            }

            psi[i] = a0;
            d[i] = a1;

            let (nr, ni) = (dc * pc - ds * ps, dc * ps + ds * pc);
            let k = 0.5 * (3.0 - (nr * nr + ni * ni));
            dc = nr * k;
            ds = ni * k;

            let (nr, ni) = (sr * qc - si * qs, sr * qs + si * qc);
            let k = 0.5 * (3.0 - (nr * nr + ni * ni));
            sr = nr * k;
            si = ni * k;
        }
    }

    /// Zeroes H(0), H'(0), H''(0), H'''(0) with the minimum-norm correction under the metric
    /// `(1 + (sigma·x)²)/window`, x = j/n. The window factor keeps the correction inside the
    /// taper so the stopband survives; the Sobolev factor concentrates it near the center so
    /// the correction contributes little slope to the error, which is what the reassignment
    /// ratios divide through.
    ///
    /// Parity splits the problem: the even taps carry the even derivatives at DC and the odd
    /// taps the odd ones, so this is two independent 2x2 solves sharing one moment table.
    /// Assumes the taper leaves enough support for the Gram to stay invertible; a bin baked
    /// down to a handful of weights with a large sigma will not.
    fn condition(&self, out: &mut [(f64, f64)], ramp: Ramp) {
        let n = out.len();
        let inv = (n as f64).recip();

        let s2 = {
            // Correction span in taps, not in fraction of span. Short bins can't afford
            // to concentrate the correction: the Gram goes soft and the flanks pay.
            let s = self.sobolev * (n as f64 / 64.0).sqrt().min(1.0);
            s * s
        };

        // Correction shape, common to both parities.
        let g = |j: usize| {
            let x = j as f64 * inv;
            (x, ramp.at(j) / (1.0 + s2 * x * x))
        };

        // Moments of the shape and the residuals in one pass. Pair multiplicity rides the
        // functionals, not the correction: it cancels out of the stationarity condition.
        let (mut m0, mut m1, mut m2, mut m3) = (0.0f64, 0.0, 0.0, 0.0);
        let (mut r0, mut r2, mut i1, mut i3) = (0.0f64, 0.0, 0.0, 0.0);
        for (j, s) in out.iter().enumerate() {
            let (x, w) = g(j);
            let c = if j == 0 { 1.0 } else { 2.0 };
            let (x2, cw) = (x * x, c * w);

            m0 += cw;
            m1 += cw * x2;
            m2 += cw * x2 * x2;
            m3 += cw * x2 * x2 * x2;

            r0 += c * s.0;
            r2 += c * x2 * s.0;
            i1 += c * x * s.1;
            i3 += c * x2 * x * s.1;
        }

        // [a b; b d] · coef = -[p; q]
        let solve = |a: f64, b: f64, d: f64, p: f64, q: f64| {
            let det = a * d - b * b;
            ((b * q - d * p) / det, (b * p - a * q) / det)
        };
        let (ea, eb) = solve(m0, m1, m2, r0, r2); // even shapes {1, x²}
        let (oa, ob) = solve(m1, m2, m3, i1, i3); // odd shapes {x, x³}

        for (j, s) in out.iter_mut().enumerate() {
            let (x, w) = g(j);
            let x2 = x * x;
            s.0 += (ea + eb * x2) * w;
            s.1 += (oa + ob * x2) * x * w;
        }
    }

    fn scale_by(out: &mut [(f64, f64)], norm: f64) {
        for s in out.iter_mut() {
            s.0 *= norm;
            s.1 *= norm;
        }
    }

    /// H(w0) of the folded Hermitian taps. Real by symmetry.
    fn gain_at(out: &[(f64, f64)], w0: f64) -> f64 {
        let (s, c) = w0.sin_cos();
        let (mut cr, mut ci) = (1.0f64, 0.0f64);
        let mut acc = out[0].0;
        for &(r, im) in &out[1..] {
            let (nr, ni) = (cr * c - ci * s, cr * s + ci * c);
            cr = nr;
            ci = ni;
            acc += 2.0 * (r * cr + im * ci);
        }
        acc
    }

    /// Interleave into `float4`, with error feedback on both real parts. Pairs carry twice the
    /// DC weight of the center, so the residual is minimized rather than random-walking.
    fn quantize(psi: &[(f64, f64)], d: &[(f64, f64)], out: &mut [[f32; 4]]) {
        let (mut ep, mut ed) = (0.0f64, 0.0f64);
        for (i, (&(pr, pi), &(dr, di))) in psi.iter().zip(d).enumerate() {
            let w = if i == 0 { 1.0 } else { 2.0 };

            let qp = (pr + ep / w) as f32;
            ep += w * (pr - qp as f64);
            let qd = (dr + ed / w) as f32;
            ed += w * (dr - qd as f64);

            out[i] = if i == 0 {
                // Halving a rounded f32 is exact, so the feedback above stays consistent.
                [0.5 * qp, 0.0, 0.5 * qd, 0.0]
            } else {
                [qp, pi as f32, qd, di as f32]
            };
        }
    }
}

/// g(u) = beta*ln(u) - (beta/gamma)*u^gamma, normalized so g(1) = 0.
fn log_env(s: Shape) -> impl Fn(f64) -> f64 {
    let bg = s.beta / s.gamma;
    move |u| s.beta * u.ln() - bg * u.powf(s.gamma) + bg
}

// /// Grid indices bracketing the spectrum above `eps`: `[lo, m)`.
// fn support(shape: Shape, du: f64, eps: f64) -> (usize, usize) {
//     let g = log_env(shape);
//     let le = eps.ln();

//     let mut j = 1usize;
//     while g(j as f64 * du) < le {
//         j += 1;
//     }
//     let lo = j;
//     while g(j as f64 * du) >= le {
//         j += 1;
//     }
//     (lo, j + 1)
// }

/// Half width in samples, times omega0. Pure function of shape and leakage.
fn half_width_scaled(s: Shape, eps: f64, tail_a: f64) -> f64 {
    let core = (2.0 * eps.recip().ln()).sqrt() * s.p();
    let tail = (tail_a / eps).powf(1.0 / (2.0 * s.beta + 1.0));
    core.max(tail)
}

/// Largest power-of-two step at or below `du`. Grids at different steps are then nested, so a
/// change to quantum or eps that doesn't cross a dyadic boundary leaves every u_j where it was.
fn snap(du: f64) -> f64 {
    du.log2().floor().exp2()
}

/// Roots of g(u) = ln(eps). g rises to 0 at u = 1 and falls after, so Newton from each
/// asymptotic branch converges monotonically inward.
fn roots(s: Shape, eps: f64) -> (f64, f64) {
    let g = log_env(s);
    let le = eps.ln();
    let dg = |u: f64| s.beta / u - s.beta * u.powf(s.gamma - 1.0);

    let solve = |mut u: f64| {
        for _ in 0..40 {
            u -= (g(u) - le) / dg(u);
        }
        u
    };
    (
        solve(eps.powf(1.0 / s.beta)),
        solve(1.0 + (2.0 * -le / s.beta).sqrt()),
    )
}

/// Grid indices bracketing the spectrum above `eps`: `[lo, m)`.
fn support(shape: Shape, du: f64, eps: f64) -> (usize, usize) {
    let (u_lo, u_hi) = roots(shape, eps);
    let lo = ((u_lo / du).ceil() as usize).max(1);
    (lo, (u_hi / du).floor() as usize + 1)
}

#[cfg(test)]
mod test {
    use core::f64::consts::PI;

    use super::*;

    const BINS: usize = 1024;
    const RATE: f64 = 48_000.0;

    fn spec(q: f64, eps: f64) -> Spec {
        Spec::default().shape(Shape::from_q(q, 3.0)).eps(eps)
    }

    // NOTE we have num_complex btw.  Just bing lazy.
    fn mag((re, im): (f32, f32)) -> f64 {
        let (re, im) = (re as f64, im as f64);
        (re * re + im * im).sqrt()
    }

    /// |H(w)| of centered taps, w in rad/sample.
    fn dtft(taps: &[(f32, f32)], w: f64) -> f64 {
        // Phase drift accumulates proportionate to sqrt(n), and reseed caps the walk.
        // Measured vs full re-seed out to about -270dB of difference, so well below what our
        // eventual storage is losing to f32 truncation already.
        //
        // Set RESEED to 1 for full seeding if this test device is under scrutiny.
        const RESEED: usize = 512;

        let half = (taps.len() / 2) as f64;
        let (s, c) = w.sin_cos();
        let (sh, ch) = (w * half).sin_cos();
        let (mut cr, mut ci) = (ch, sh);
        let (mut re, mut im) = (0.0f64, 0.0f64);

        for (j, &(r, i)) in taps.iter().enumerate() {
            if j & (RESEED - 1) == 0 {
                let (sj, cj) = (w * (half - j as f64)).sin_cos();
                cr = cj;
                ci = sj;
            }
            let (r, i) = (r as f64, i as f64);
            re += r * cr - i * ci;
            im += r * ci + i * cr;
            let (nr, ni) = (cr * c + ci * s, ci * c - cr * s);
            let k = 0.5 * (3.0 - (nr * nr + ni * ni));
            cr = nr * k;
            ci = ni * k;
        }

        (re * re + im * im).sqrt()
    }

    /// Worst |H| gap between two responses, split at the reference filter's half-power
    /// edges. Passband gap is what a tone's magnitude sees; stopband gap is the leakage
    /// truncation bought.
    ///
    /// Truncation cost split by region. Passband is measured as relative error against the
    /// reference and restricted to the band core, because at the half-power edges a sub-percent
    /// width change swamps any in-band ripple — the flanks are steep and the absolute gap there
    /// reports skirt motion, not fidelity. Stopband stays absolute, relative to peak, since
    /// that is what leakage means.
    fn gap_split(a: &dyn Fn(f64) -> f64, b: &dyn Fn(f64) -> f64, lo: f64, hi: f64) -> (f64, f64) {
        const SWEEP: usize = 8192;
        // Core is the middle half of the -3 dB band, clear of the flanks.
        let (mid, quarter) = (0.5 * (lo + hi), 0.25 * (hi - lo));
        let (core_lo, core_hi) = (mid - quarter, mid + quarter);

        let (mut pass, mut stop) = (0.0f64, 0.0f64);
        for k in 0..=SWEEP {
            let w = -PI + 2.0 * PI * k as f64 / SWEEP as f64;
            let (ra, rb) = (a(w), b(w));
            if (core_lo..=core_hi).contains(&w) {
                pass = pass.max((ra - rb).abs() / ra);
            } else if !(lo..=hi).contains(&w) {
                stop = stop.max((ra - rb).abs());
            }
        }
        (pass, stop)
    }

    /// Folded weights back to centered taps. Lane `c` selects psi (0) or d (2).
    /// The doubled center undoes the halving in `quantize`.
    fn unfold(w: &[[f32; 4]], c: usize) -> Vec<(f32, f32)> {
        let k = w.len();
        let mut out = vec![(0.0f32, 0.0f32); 2 * k - 1];
        out[k - 1] = (2.0 * w[0][c], 0.0);
        for (j, q) in w.iter().enumerate().skip(1) {
            out[k - 1 + j] = (q[c], q[c + 1]);
            out[k - 1 - j] = (q[c], -q[c + 1]);
        }
        out
    }

    /// t_nu = -i*nu*psi_nu, matching what a consumer reconstructs per lane.
    fn derive_t(psi: &[(f32, f32)]) -> Vec<(f32, f32)> {
        let half = (psi.len() / 2) as isize;
        psi.iter()
            .enumerate()
            .map(|(j, &(r, i))| {
                let nu = (j as isize - half) as f32;
                (nu * i, -nu * r)
            })
            .collect()
    }

    /// Bisect for |H| = target on [a, b], target bracketed.
    fn crossing(h: &dyn Fn(f64) -> f64, mut a: f64, mut b: f64, target: f64) -> f64 {
        let above = h(a) > target;
        for _ in 0..60 {
            let m = 0.5 * (a + b);
            if (h(m) > target) == above {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    }
    /// Signed bars for `re` and `im` overlaid on one axis, zero between cells `cols / 2 - 1` and
    /// `cols / 2`. Caller guarantees `max >= |re|` and `max >= |im|`, which keeps both spans inside
    /// the `cols` field.
    fn bar(re: f64, im: f64, max: f64, cols: usize) -> String {
        let (cr, ci, cb) = ('●', '·', '•');
        let half = cols / 2;
        let span = |v: f64| {
            let col = ((v / max) * half as f64).round() as isize;
            let end = (half as isize + col) as usize;
            if col >= 0 {
                half..end
            } else {
                end..half
            }
        };
        let (r, i) = (span(re), span(im));
        let cells: String = (0..cols)
            .map(|c| match (r.contains(&c), i.contains(&c)) {
                (true, true) => cb,
                (true, false) => cr,
                (false, true) => ci,
                (false, false) => ' ',
            })
            .collect();
        cells.trim_end().to_string()
    }

    /// Real and imaginary parts of `taps`, centered, on a shared scale.
    fn print_wave(label: &str, taps: &[(f32, f32)], cols: usize) {
        let n = taps.len();
        println!("\n=== {label} ===");
        let max = taps
            .iter()
            .map(|&(r, i)| (r as f64).abs().max((i as f64).abs()))
            .fold(0.0, f64::max);

        for (j, &(re, im)) in taps.iter().enumerate() {
            println!(
                "{:>6} {:>12.7} {:>12.7} {}",
                j as isize - (n / 2) as isize,
                re,
                im,
                bar(re as f64, im as f64, max, cols)
            );
        }
    }

    struct Response {
        peak_w: f64,
        peak_h: f64,
        edges: (f64, f64),
        rel_width: f64,
        neg: f64,
        floor: f64,
    }

    /// Peak location and gain, -3 dB relative width, negative-frequency max, and the
    /// floor outside three half-power widths. `w0` only sets the bracket for the edges.
    fn characterize(h: &dyn Fn(f64) -> f64, w0: f64) -> Response {
        const SWEEP: usize = 8192;
        let omega = |k: usize| -PI + 2.0 * PI * k as f64 / SWEEP as f64;

        let (mut peak, mut neg) = ((0.0f64, 0.0f64), 0.0f64);
        for k in 0..=SWEEP {
            let w = omega(k);
            let v = h(w);
            if w < 0.0 {
                neg = neg.max(v);
            }
            if v > peak.1 {
                peak = (w, v);
            }
        }

        let cell = 2.0 * PI / SWEEP as f64;
        let (mut a, mut b) = (peak.0 - cell, peak.0 + cell);
        for _ in 0..80 {
            let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
            if h(m1) < h(m2) {
                a = m1;
            } else {
                b = m2;
            }
        }
        let peak_w = 0.5 * (a + b);
        let peak_h = h(peak_w);

        let half = peak_h / 2.0f64.sqrt();
        let lo = crossing(h, peak_w - w0, peak_w, half);
        let hi = crossing(h, peak_w, (peak_w + w0).min(PI), half);

        let guard = 3.0 * (hi - lo);
        let mut floor = 0.0f64;
        for k in 0..=SWEEP {
            let w = omega(k);
            if (w - peak_w).abs() > guard {
                floor = floor.max(h(w));
            }
        }

        Response {
            peak_w,
            peak_h,
            edges: (lo, hi),
            rel_width: (hi - lo) / peak_w,
            neg,
            floor,
        }
    }

    #[test]
    fn print_gamma_sweep() {
        const QUANTUM: usize = 4;

        println!("\n=== ENVELOPE vs GAMMA (Q = 2.4) ===");
        // P = 4.0 is Q = 2.4; holding it fixed keeps the -3 dB width constant across gamma.
        let p = 4.0;
        for gamma in [1.0f64, 2.0, 3.0, 6.0] {
            let mut plan = Spec::default()
                .shape(Shape {
                    gamma,
                    beta: p * p / gamma,
                })
                .max_load_quantum(QUANTUM)
                .plan();
            let bin = plan.bin(1000.0, 8000.0);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps(QUANTUM)];
            plan.taps_into(bin, QUANTUM, &mut w);

            let t = unfold(&w, 0);
            let n = t.len();

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
                "\ngamma = {:.1}  weights {}  taps {}  centroid offset = {:+.3}",
                gamma,
                w.len(),
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

        let load_quantum = 4;
        let start = std::time::Instant::now();
        let mut plan = Spec::default()
            .max_taps(1024)
            .shape(Shape::from_q(5.0, 3.0))
            .max_load_quantum(load_quantum)
            .taper(Taper {
                eps_time: 5e-3,
                rho: 0.00,
            })
            .plan();
        let bins = bank::bins(2_000.0, 20_000.0, BINS);
        println!("planning time: {:?}µs", start.elapsed().as_micros());

        let voices: Vec<Bin> = bins
            .iter()
            .map(|b| b.center)
            .map(|c| plan.bin(c, RATE))
            .collect();

        let mut offsets = Vec::with_capacity(voices.len());
        let strides: Vec<usize> = voices.iter().map(|b| b.folded_taps(load_quantum)).collect();
        let total: usize = strides.iter().sum();
        let mut weights = vec![[0.0f32; 4]; total];

        let mut cursor = 0;
        for (&bin, &n) in voices.iter().zip(&strides) {
            offsets.push(cursor);
            plan.taps_into(bin, load_quantum, &mut weights[cursor..cursor + n]);
            cursor += n;
        }
        let elapsed = start.elapsed();

        let worst = voices
            .iter()
            .zip(offsets.iter().zip(&strides))
            .map(|(b, (&o, &n))| {
                (dtft(&unfold(&weights[o..o + n], 0), b.velocity()) - PEAK_GAIN).abs()
            })
            .fold(0.0f64, f64::max);
        println!("worst peak gain error: {worst:.3e}");
        assert!(worst < 1e-3, "worst peak gain error {worst:.3e}");

        let lowest = unfold(&weights[..strides[0]], 0);
        print_wave(
            &format!(
                "LOWEST BIN ({:.0}Hz, omega0 {:.5})",
                bins[0].center,
                voices[0].velocity()
            ),
            &lowest,
            30,
        );

        println!(
            "voices {} of {}  weights {}  longest {}  shortest {}",
            voices.len(),
            BINS,
            total,
            strides[0],
            strides[strides.len() - 1],
        );

        println!("bin filling time: {:?}µs", elapsed.as_micros());
    }

    /// A real unit tone reads |W| = 1 even though |H| = 2: the analytic taps
    /// see only the +w half of the cosine. Swept over the quantum because
    /// padding moves the taper start and with it the emitted energy.
    #[test]
    fn unit_tone_reads_unity() {
        for quantum in [1usize, 4, 8] {
            let mut p = spec(3.0, 1e-8)
                .taper(Taper {
                    eps_time: 1e-3,
                    rho: 0.00,
                })
                .max_load_quantum(quantum)
                .plan();

            for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = p.bin(fc, sr);
                let mut w = vec![[0.0f32; 4]; bin.folded_taps(quantum)];
                p.taps_into(bin, quantum, &mut w);

                let psi = unfold(&w, 0);
                let (n, w0) = (psi.len(), bin.velocity());
                let half = (n / 2) as isize;

                // taps are centered, so m is the sample under tap index n/2.
                for m in 0..8 {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for (j, &(r, i)) in psi.iter().enumerate() {
                        let x = (w0 * (m as isize + half - j as isize) as f64).cos();
                        re += x * r as f64;
                        im += x * i as f64;
                    }
                    let env = re.hypot(im);
                    assert!(
                        (env - 1.0).abs() < 1e-3,
                        "quantum {quantum} fc {fc} phase {m} envelope {env:.6}"
                    );
                }
            }
        }
    }

    ///  Peak-normalized constant-Q puts noise gain proportional to center
    /// frequency: length goes as 1/w0, amplitude as 1/N, so energy tracks w0.
    /// White noise therefore floors at a fixed level per bin once w0 is divided out.
    #[test]
    fn noise_gain_tracks_center() {
        // Quantum rounding pads the emitted half-span, so the energy sum picks up
        // whatever the taper leaves on the pad. Anchored at the shipping quantum.
        const QUANTUM: usize = 4;

        // Tap count is an integer, so envelope truncation loses O(1/N) of the
        // energy, worst at the top of the range. Anchored to split the sweep
        // rather than to any one bin.
        const NOISE_GAIN: f64 = 0.224777;
        const TOL: f64 = 5e-4;

        let mut p = spec(3.0, 1e-8)
            .taper(Taper {
                eps_time: 1e-3,
                rho: 0.00,
            })
            .max_load_quantum(QUANTUM)
            .plan();

        println!("\n=== NOISE GAIN (Q = 3, sr = {RATE}, quantum {QUANTUM}) ===");

        for fc in [500.0f64, 1000.0, 2000.0, 4000.0, 8000.0] {
            let bin = p.bin(fc, RATE);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps(QUANTUM)];
            p.taps_into(bin, QUANTUM, &mut w);

            let psi = unfold(&w, 0);
            let e: f64 = psi
                .iter()
                .map(|&(r, i)| (r as f64).powi(2) + (i as f64).powi(2))
                .sum();
            let ratio = e / bin.velocity();

            println!(
                "  fc {:>6.0}  taps {:>5}  energy {:.6}  e/w0 {:.6}  dev {:+.2e}",
                fc,
                bin.unfolded_taps(QUANTUM),
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

    /// Tests for bias correlation and magnitude in reassignment.
    #[test]
    fn reassignment_is_unbiased() {
        const QUANTUM: usize = 4;

        let mut plan = spec(3.5, 1e-8)
            .taper(Taper {
                eps_time: 1e-4,
                rho: 0.00,
            })
            .max_load_quantum(QUANTUM)
            .plan();

        for (fc, sr) in [(2_000.0f64, RATE), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = plan.bin(fc, sr);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps(QUANTUM)];
            plan.taps_into(bin, QUANTUM, &mut w);

            let psi = unfold(&w, 0);
            let (d, t) = (unfold(&w, 2), derive_t(&psi));

            let (n, half, w0) = (psi.len(), psi.len() / 2, bin.velocity());

            // W(m) = sum_j x[m + half - j] * h[j], matching unit_tone_reads_unity.
            let conv = |h: &[(f32, f32)], x: &dyn Fn(isize) -> f64, m: isize| {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (j, &(r, i)) in h.iter().enumerate() {
                    let s = x(m + half as isize - j as isize);
                    re += s * r as f64;
                    im += s * i as f64;
                }
                (re, im)
            };
            let div = |a: (f64, f64), b: (f64, f64)| {
                let q = b.0 * b.0 + b.1 * b.1;
                ((a.0 * b.0 + a.1 * b.1) / q, (a.1 * b.0 - a.0 * b.1) / q)
            };

            println!("\n=== REASSIGN fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6} ===");

            for cents in [-600.0f64, -300.0, 0.0, 300.0, 600.0] {
                let w = w0 * (cents / 1200.0).exp2();
                let tone = |k: isize| (w * k as f64).cos();
                let (mut worst, mut quad) = (0.0f64, 0.0f64);
                for m in 0..8 {
                    let (re, im) = div(conv(&d, &tone, m), conv(&psi, &tone, m));
                    worst = worst.max((1200.0 * (re / w).log2()).abs());
                    quad = quad.max((im / w).abs());
                }
                println!("  detune {cents:+6.0}c  bias {worst:.4}c  quad {quad:.2e}");
                assert!(worst < 2.0, "fc {fc} detune {cents} bias {worst:.4}c");
                assert!(quad < 1e-3, "fc {fc} detune {cents} quad {quad:.3e}");
            }

            let imp = |k: isize| if k == 0 { 1.0 } else { 0.0 };
            let env0 = (psi[half].0 as f64).hypot(psi[half].1 as f64);
            let (mut worst, mut real) = (0.0f64, 0.0f64);
            for m in -(half as isize)..=(half as isize) {
                let wp = conv(&psi, &imp, m);
                if wp.0.hypot(wp.1) < 1e-2 * env0 {
                    continue;
                }
                let (re, im) = div(conv(&t, &imp, m), wp);
                worst = worst.max((m as f64 + im).abs());
                real = real.max(re.abs());
            }
            println!("  impulse: worst t_hat {worst:.4} samples  real leak {real:.2e}");
            assert!(worst < 0.05, "fc {fc} t_hat off by {worst:.4} samples");
            assert!(real < 1e-2, "fc {fc} t real leak {real:.3e}");
        }
    }

    /// Truncation cost against a full-length bake, swept over `eps_time`.
    ///
    /// Also the only coverage of the untapered path: with `rho == 0` the taper is a no-op
    /// and `center` folds against a flat window, so the full-length bake exercises branches
    /// the tapered tests never reach.
    ///
    /// Cross-references: peak gain against `taps_are_conditioned` (which only sees tapered
    /// bakes), and `rel_width` against `Shape::from_q`, which no other test measures.
    #[test]
    fn truncation_is_predictable() {
        const QUANTUM: usize = 4;
        const Q: f64 = 4.0;

        // NOTE these are empirically discovered values stored to catch regressions.

        // Measured 0.9984 at Q = 3.5.
        const WIDTH_Q: f64 = 0.998;
        const WIDTH_TOL: f64 = 0.03;

        // Stopband gap relative to PEAK_GAIN, as a multiple of eps_time.
        const LEAK_PER_EPS: f64 = 1.0;

        // In-band relative error as a multiple of eps_time.
        const PASS_PER_EPS: f64 = 0.3;

        let base = Spec::default()
            .shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM);
        let mut full = base.plan();

        println!("\n=== TRUNCATION (Q = {Q}, rho 0.00, quantum {QUANTUM}) ===");

        for fc in [2_000.0f64, 4_000.0, 8_000.0, 14_000.0] {
            let bf = full.bin(fc, RATE);
            let nf = bf.folded_taps(QUANTUM);
            let mut wf = vec![[0.0f32; 4]; nf];
            full.taps_into(bf, QUANTUM, &mut wf);
            let pf = unfold(&wf, 0);
            let hf = |w: f64| dtft(&pf, w);

            let w0 = bf.velocity();
            let rf = characterize(&hf, w0);

            println!(
                "\n  fc {fc:>6.0}  full weights {nf:>5}  peak {:.9}  rel width {:.5} \
                 (x Q = {:.4})",
                rf.peak_h,
                rf.rel_width,
                rf.rel_width * Q
            );

            // The untapered bake is held to the same conditioning as the tapered ones.
            assert!(
                (rf.peak_h / PEAK_GAIN - 1.0).abs() < 1e-5,
                "fc {fc} full peak gain {:.9}",
                rf.peak_h
            );
            let dc = pf.iter().map(|&(r, _)| r as f64).sum::<f64>();
            assert!(dc.abs() < 1e-5 * PEAK_GAIN, "fc {fc} full dc {dc:.3e}");

            // -3 dB width is set by P = sqrt(beta*gamma) and Q = P/1.6651. Nothing else
            // measures whether that conversion actually lands.
            assert!(
                (rf.rel_width * Q / WIDTH_Q - 1.0).abs() < WIDTH_TOL,
                "fc {fc} rel width {:.5} x Q = {:.4}",
                rf.rel_width,
                rf.rel_width * Q
            );

            let (mut prev_taps, mut prev_stop) = (0usize, f64::INFINITY);

            for eps_time in [1e-2f64, 1e-3, 1e-4] {
                let mut cut = base
                    .taper(Taper {
                        eps_time,
                        rho: 0.00,
                    })
                    .plan();
                let bc = cut.bin(fc, RATE);
                let nc = bc.folded_taps(QUANTUM);
                let mut wc = vec![[0.0f32; 4]; nc];
                cut.taps_into(bc, QUANTUM, &mut wc);
                let pc = unfold(&wc, 0);
                let hc = |w: f64| dtft(&pc, w);

                let (pass, stop) = gap_split(&hf, &hc, rf.edges.0, rf.edges.1);
                let rc = characterize(&hc, w0);
                let cents = 1200.0 * (rc.peak_w / rf.peak_w).log2();

                println!(
                    "    eps_time {eps_time:>7.0e}  weights {nc:>5} ({:.3})  \
                     pass {:>7.2} dB rel  stop {:>7.2} dB  peak {:+.4}c  width {:+.3}%",
                    nc as f64 / nf as f64,
                    20.0 * pass.log10(),
                    20.0 * (stop / PEAK_GAIN).log10(),
                    cents,
                    100.0 * (rc.rel_width / rf.rel_width - 1.0)
                );

                // In-band magnitude is flat to well under the leakage budget. The peak is
                // pinned by gain_at; this checks the core around it didn't tilt.
                assert!(
                    pass < PASS_PER_EPS * eps_time,
                    "fc {fc} eps_time {eps_time:e} passband {:.2} dB rel",
                    20.0 * pass.log10()
                );

                // Peak gain survives truncation, and the band neither moves nor widens.
                assert!(
                    cents.abs() < 1.0,
                    "fc {fc} eps_time {eps_time:e} peak moved {cents:+.4}c"
                );
                assert!(
                    (rc.rel_width / rf.rel_width - 1.0).abs() < 0.02,
                    "fc {fc} eps_time {eps_time:e} width {:+.3}%",
                    100.0 * (rc.rel_width / rf.rel_width - 1.0)
                );

                // Tighter eps_time truncates less, so it costs taps and buys stopband.  Taps may
                // stay the same due to quantum rounding.
                assert!(
                    nc >= prev_taps,
                    "fc {fc} eps_time {eps_time:e} taps {nc} < {prev_taps}"
                );
                // If taps go up, stop band must go down.
                assert!(
                    (nc == prev_taps && stop >= prev_stop) || stop < prev_stop,
                    "fc {fc} eps_time {eps_time:e} taps {prev_taps} -> {nc} without stopband gain"
                );

                assert!(
                    stop < LEAK_PER_EPS * eps_time * PEAK_GAIN,
                    "fc {fc} eps_time {eps_time:e} stop {:.2} dB",
                    20.0 * (stop / PEAK_GAIN).log10()
                );

                (prev_taps, prev_stop) = (nc, stop);
            }
        }
    }

    /// Same four numbers as `response_is_characterized`, measured on the folded weight
    /// table. Sweeps the load quantum, because the quantum pads the emitted half-span and moves
    /// where the taper starts.
    #[test]
    fn table_response_is_characterized() {
        let (q, eps, eps_time, rho) = (3.5, 1e-8, 2e-4, 0.0);
        for quantum in [1usize, 4, 8, 16] {
            let mut p = spec(q, eps)
                .taper(Taper { eps_time, rho })
                .max_load_quantum(quantum)
                .plan();

            println!("\n=== TABLE RESPONSE (Q = {q}, quantum {quantum}) ===");

            for (fc, sr) in [(1000.0f64, 6000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = p.bin(fc, sr);
                let n = bin.folded_taps(quantum);
                let mut w = vec![[0.0f32; 4]; n];
                p.taps_into(bin, quantum, &mut w);

                let w0 = bin.velocity();
                let psi = unfold(&w, 0);
                let r = characterize(&|omega| dtft(&psi, omega), w0);

                let db = |v: f64| 20.0 * (v / r.peak_h).log10();
                let taps = bin.taps;

                println!(
                    "\nfc {fc:>7.0} sr {sr:>6.0} taps {taps:>5} weights {n:>5} \
                     (unfolded {:>5})  w0 {w0:.6}",
                    bin.unfolded_taps(quantum)
                );
                println!(
                    "  peak gain {:.9}  dev {:+.3e} rel",
                    r.peak_h,
                    r.peak_h / PEAK_GAIN - 1.0
                );
                println!("  rel width {:.5}", r.rel_width);
                println!("  peak {:+.4} cents", 1200.0 * (r.peak_w / w0).log2());
                println!("  negative-freq max {:>8.2} dB", db(r.neg));
                println!("  stopband floor    {:>8.2} dB", db(r.floor));

                assert!(
                    (r.peak_h / PEAK_GAIN - 1.0).abs() < 1e-5,
                    "fc {fc} q {quantum} peak gain {:.9}",
                    r.peak_h
                );
            }
        }
    }

    /// Smoke test. DC-free and correct peak gain, measured with the linear DTFT so a broken fold
    /// convention can't agree with itself. First thing to look at if the bake goes sideways.
    #[test]
    fn taps_are_conditioned() {
        const QUANTUM: usize = 4;

        let mut p = spec(3.0, 1e-8)
            .taper(Taper {
                eps_time: 1e-3,
                rho: 0.0,
            })
            .max_load_quantum(QUANTUM)
            .plan();

        for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = p.bin(fc, sr);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps(QUANTUM)];
            p.taps_into(bin, QUANTUM, &mut w);

            let (psi, d) = (unfold(&w, 0), unfold(&w, 2));
            let w0 = bin.velocity();

            let g = dtft(&psi, w0);
            assert!((g - PEAK_GAIN).abs() < 1e-3, "fc {fc} peak gain {g:.6}");

            // Analytic taps: the mirror image is stopband, not signal.
            let neg = dtft(&psi, -w0);
            assert!(
                neg < 1e-3 * g,
                "fc {fc} negative-freq leak {:.2} dB",
                20.0 * (neg / g).log10()
            );

            let dc = psi.iter().map(|&(r, _)| r as f64).sum::<f64>();
            assert!(dc.abs() < 1e-5 * g, "fc {fc} dc {dc:.3e}");

            // d carries w0/peak, so its ratio against psi reads in rad/sample.
            let gd = dtft(&d, w0);
            assert!(
                (gd / g - w0).abs() < 1e-3 * w0,
                "fc {fc} d/psi {:.6} want {w0:.6}",
                gd / g
            );

            let m1 = psi
                .iter()
                .enumerate()
                .map(|(j, &(_, i))| (j as isize - (psi.len() / 2) as isize) as f64 * i as f64)
                .sum::<f64>();
            // measured: fc 1000 first moment -1.123e-7
            assert!(m1.abs() < 1e-5 * g, "fc {fc} first moment {m1:.3e}");
        }
    }
}
