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
//! # use mutate::wavelet::Spec;
//! const QUANTUM: usize = 8;
//!
//! let mut plan = Spec::default()
//!     .max_load_quantum(QUANTUM)
//!     .plan();
//!
//! let bin = plan.bin(1000.0, 8000.0, QUANTUM);
//! let mut weights = vec![[0.0f32; 4]; bin.folded_taps()];
//! plan.taps_into(bin, &mut weights);
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
//! # use mutate::wavelet::Spec;
//! const QUANTUM: usize = 8;
//!
//! let mut plan = Spec::default()
//!     .max_load_quantum(QUANTUM)
//!     .plan();
//!
//! let bin = plan.bin(1000.0, 8000.0, QUANTUM);
//! let mut weights = vec![[0.0f32; 4]; bin.folded_taps()];
//!
//! // float4(Re ψ, Im ψ, Re d, Im d); index 0 is the center, with Re halved.
//! plan.taps_into(bin, &mut weights);
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
//! `d/ψ` reads in rad/sample and `t/ψ` in samples. No scaling required at the use site.

// 🤖 Heavy generation.  Should be pretty standard academic stuff, so not expecting a lot of
// surprises.  We will, for the most part, swiftly and knowingly eat shit if the wavelet is busted.
// Well-formalized stuff doesn't have a lot of wiggle room to violate the consistency of the
// formalism.

// NEXT Characterizing results needs a more stable benchmark test that is not configured with user
// API independent knobs.  More automation of checking performance is pretty much required to
// progress much farther.
// NEXT Noise floor performance, which provides our dynamic range resolution, is very sensitive to
// the number of taps for a given Q.  We would like to improve Q without raising N taps and we would
// like to make more use of our padding quantum, but only an integrated design process, one that
// solves for measured goals, such as noise floor and reassignment accuracy, is going to squeeze
// more f32 pixie dust out.  Several tapering schemes ranging from Plank envelope to a higher order
// solution were tried, but none consistently improved the result of simple truncation.
// NEXT Offline SOCP oracle to see what we're leaving on the table with our approximations.
// NOTE Peak gain 2 per bin (demodulated L1 = 2), so a unit real tone reads |W| = 1.
// Steady tones are flat across the bank.  Noise floor rises as sqrt(f).
// NOTE We have logarithmic bin spacings, but the cutoff frequencies that determine which downsample
// will be used are not particularly aware, so it's not expected that we can re-use exact bins in
// any kind of octave structure.  Mel scaling etc also defeats this, so there's no point.
// NOTE Run time of the bin generation test (not reflective of actual sample rates and Q) is about
// 30ms on a Zen2+ part in release.  This affects CWT startup time.

// === TABLE RESPONSE (Q = 3.5, quantum 4) ===
//
// fc  1000 sr  6000  w0 1.047198  quantized  25 (unfolded  49)
//   peak gain 2.000000001  dev +3.297e-10 rel
//   rel width 0.28523
//   peak -0.0033 cents
//   negative-freq max  -103.74 dB
//   stopband floor     -103.69 dB
//
// fc   250 sr  3000  w0 0.523599  quantized  49 (unfolded  97)
//   peak gain 2.000000000  dev +1.824e-10 rel
//   rel width 0.28523
//   peak -0.0041 cents
//   negative-freq max  -102.57 dB
//   stopband floor     -102.02 dB
//
// fc 12000 sr 48000  w0 1.570796  quantized  17 (unfolded  33)
//   peak gain 1.999999996  dev -2.022e-9 rel
//   rel width 0.28523
//   peak -0.0028 cents
//   negative-freq max  -105.28 dB
//   stopband floor     -105.28 dB

use core::f64::consts::{LN_10, LN_2, PI, TAU};

/// Filter peak gain. Analytic taps see half a real tone's amplitude,
/// so |H| = 2 makes a unit tone read |W| = 1.
const PEAK_GAIN: f64 = 2.0;

/// A center frequency resolved against a plan, a sample rate, and a load quantum.
#[derive(Clone, Copy)]
pub struct Bin {
    w0: f64,
    quantum: usize,
    sigmas: f64,
    k: usize,
}

impl Bin {
    /// Rotational velocity ദ്ദി(•̀ω-)✧ in radians per sample.
    pub fn velocity(&self) -> f64 {
        self.w0
    }

    pub fn quantum(&self) -> usize {
        self.quantum
    }

    /// Folded weights including the center tap.  **Exactly** the length [`taps_into`] will write to
    /// the destination.
    pub fn folded_taps(&self) -> usize {
        self.k
    }

    /// Effective taps after unfolding, including the center tap.
    pub fn unfolded_taps(&self) -> usize {
        2 * self.k - 1
    }

    /// Truncation radius actually emitted, after the extremum snap and quantum rounding.
    /// At or above what the plan asked for; the excess is free stopband.
    pub fn sigmas(&self) -> f64 {
        self.sigmas
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
        let p = 2.0 * LN_2.sqrt() * q;
        Shape {
            gamma,
            beta: p * p / gamma,
        }
    }

    /// P = sqrt(beta*gamma), the -3 dB width parameter.  Q = P/1.6651.
    pub fn p(&self) -> f64 {
        (self.beta * self.gamma).sqrt()
    }

    pub fn q(&self) -> f64 {
        self.p() / (2.0 * LN_2.sqrt())
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

#[derive(Clone, Copy)]
enum Truncation {
    Sigmas(f64),
    FloorDb(f64),
}

/// Truncation leakage is bounded by the envelope tail, `exp(-n²/2)` at `n` sigmas.
/// `10/ln(10)` is the dB conversion, so a floor request and a sigma request are the
/// same number in two units.
impl Truncation {
    fn sigmas(&self) -> f64 {
        match *self {
            Truncation::Sigmas(n) => n,
            Truncation::FloorDb(db) => (db.abs() * LN_10 / 10.0).sqrt(),
        }
    }
}

/// A builder struct for configuring a plan to build wavelets with a given shape.
///
/// ```
/// # use mutate::wavelet::Spec;
///
/// let spec = Spec::default();
/// ```
#[derive(Clone, Copy)]
pub struct Spec {
    shape: Shape,
    /// Error tolerance for the spectrum.
    grid_eps: f64,
    /// Error tolerance for filter length truncation
    truncation: Truncation,
    max_taps: usize,
    max_load_quantum: usize,
    sobolev: f64,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            shape: Shape::from_q(3.0, 3.0),
            grid_eps: 1e-14,
            truncation: Truncation::Sigmas(4.0),
            max_taps: 0,
            max_load_quantum: 1,
            sobolev: 0.0,
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
    /// floor of the output taps.  1e-10 has measurable effects at 5.5sigmas.
    pub fn grid_eps(mut self, eps: f64) -> Self {
        self.grid_eps = eps;
        self
    }

    /// `max_taps` is a **hint** to allocate a larger scratch `Vec`, which will of course resize if
    /// necessary.  Uses [`Vec::with_capacity`](std::vec::Vec::with_capacity).  No effect on quality.
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

    /// Set error tolerance by envelope geometry.  This will control the selected N taps for each
    /// [`Bin`], so it's one of the most powerful knobs.  Values below 3.0 truncate too hard to
    /// ship until a better numerical solver is available.  Values over 5.5 begin grinding up the
    /// dust of departed f32s.
    pub fn sigmas(mut self, sigmas: f64) -> Self {
        self.truncation = Truncation::Sigmas(sigmas);
        self
    }

    /// Set error tolerance by desired noise floor, which is **estimated** to an envelope geometry
    /// and ultimately used to select a value for [`sigmas`].  A different way to attempt to say the
    /// same thing.  State dB if you do not measure sigmas.
    pub fn floor_db(mut self, db: f64) -> Self {
        self.truncation = Truncation::FloorDb(db);
        self
    }

    /// Half-span that sets tap count, and the grid extent that feeds it.
    fn spans(&self) -> (f64, f64) {
        let taps = self.truncation.sigmas() * self.shape.p();
        let grid = half_width_scaled(self.shape, self.grid_eps);

        // Quantum rounding plus the center tap extend the emitted span by up to 2q+1 samples,
        // worst at w0 = PI in scaled units.
        let pad = (2 * self.max_load_quantum + 1) as f64 * PI;
        // Rectangle rule on a uniform grid aliases rather than truncates: the error is the
        // time-domain replica at period TAU/du. Placing it a full eps half-width past the
        // emitted span puts the fold-back at eps.
        (taps, TAU / (taps + pad + grid.max(taps)))
    }

    pub fn plan(self) -> Plan {
        let (c, du) = self.spans();
        let du = snap(du);
        let (lo, m) = support(self.shape, du, self.grid_eps);

        // d = w*psi shares psi's support, so lo bounds both.
        let env = log_env(self.shape);
        let mut spec = vec![[0.0; 2]; m];
        for (j, s) in spec.iter_mut().enumerate().skip(lo) {
            let u = j as f64 * du;
            let p = env(u).exp();
            *s = [p, p * u];
        }

        Plan {
            shape: self.shape,
            c,
            du,
            spec,
            lo,
            sobolev: self.sobolev,
            buf: Vec::with_capacity(3 * (self.max_taps / 2 + self.max_load_quantum + 1)),
            max_load_quantum: self.max_load_quantum,
        }
    }
}

// CWT weight table generator.
pub struct Plan {
    shape: Shape,
    c: f64,               // half_width_scaled
    du: f64,              // uniform step in u = w/w_peak
    spec: Vec<[f64; 2]>,  // [psi, d] at u_j = j*du
    lo: usize,            // first grid point above eps
    buf: Vec<(f64, f64)>, // bake scratch, 2 spans
    sobolev: f64,
    max_load_quantum: usize,
}

impl Plan {
    /// Generate a bin definition at `center` frequency, sampled at `rate`, loaded `quantum` weights
    /// at a time.
    ///
    /// Picks between the two shortest quantum-legal half-spans such that the last nonzero tap lands
    /// on a carrier extremum: `m` whole cycles of aperture, `m` fixed by the plan, so every bin
    /// sees the same scaled half-span and the same response. Quantum rounding pads past it with
    /// zeros rather than aperture.
    pub fn bin(&self, center: f64, rate: f64, quantum: usize) -> Bin {
        debug_assert!(
            center < rate / 2.0,
            "center {center:.1}Hz above Nyquist {:.0}Hz",
            rate / 2.0
        );
        debug_assert!(
            quantum <= self.max_load_quantum,
            "quantum {quantum} above plan ceiling {}",
            self.max_load_quantum
        );

        let w0 = TAU * center / rate;
        let span = (self.c / PI).round() * PI / w0;
        let n = (span / quantum as f64).ceil() as usize * quantum;

        Bin {
            w0,
            quantum,
            k: n + 1,
            sigmas: n as f64 * w0 / self.shape.p(),
        }
    }

    /// Writes `bin.folded_taps()` weights and returns that count. Peak gain 2, DC removed.
    ///
    /// Upstream owes an `out` at least that long.
    pub fn taps_into(&mut self, bin: Bin, out: &mut [[f32; 4]]) -> usize {
        let k = bin.k;
        let out = &mut out[..k];

        let mut buf = core::mem::take(&mut self.buf);
        buf.resize(3 * k, (0.0, 0.0));
        {
            let (psi, rest) = buf.split_at_mut(k);
            let (d, e) = rest.split_at_mut(k);

            self.transform2(bin.w0, psi, d);

            Self::scale_by(psi, PEAK_GAIN / Self::gain_at(psi, bin.w0));

            // Trades a bit of stop band depth for much lower DC.  Check the gains between 3.5 and
            // 4.5 sigma.  Sometimes longer truncation beats conditioning, but it may depend on
            // other knobs like the sobolev order.
            self.condition(psi, bin.w0, Some(0.5 * PEAK_GAIN));

            let half_bw_cents = 1200.0 * (1.0 + 0.5 / self.shape.q()).log2();
            let fit_span_cents = 3.0 * half_bw_cents;
            Self::align_d(psi, d, e, bin.w0, fit_span_cents);

            self.condition(d, bin.w0, None);

            Self::quantize(psi, d, bin.w0, out);
        }
        self.buf = buf;
        k
    }

    /// Least-squares fit of H_d(w) = w*H_psi(w) over the intended detune range, using
    /// d and nu^2*psi as the basis. nu^2 is even, so the fold and the derived t are
    /// untouched; adding a multiple of nu^2*psi subtracts a multiple of H_psi'', which
    /// is the curvature the ratio error is made of.
    ///
    /// `cents` brackets the detune range the consumer will reassign over, clamped so the
    /// top probe stays clear of Nyquist: past it the fold-over dominates gain_at and the
    /// fit chases the image instead of the band.
    fn align_d(
        psi: &[(f64, f64)],
        d: &mut [(f64, f64)],
        e: &mut [(f64, f64)],
        w0: f64,
        cents: f64,
    ) {
        const PROBES: usize = 9;

        for (j, (s, &(pr, pi))) in e.iter_mut().zip(psi).enumerate() {
            let j2 = (j * j) as f64;
            *s = (pr * j2, pi * j2);
        }

        let cents = cents.min(1200.0 * (0.9 * PI / w0).log2());

        let (mut aa, mut ab, mut bb, mut ap, mut bp) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        for i in 0..PROBES {
            let c = cents * (2.0 * i as f64 / (PROBES - 1) as f64 - 1.0);
            let w = w0 * (c / 1200.0).exp2();
            let (hd, he) = (Self::gain_at(d, w), Self::gain_at(e, w));
            let hp = w * Self::gain_at(psi, w);
            aa += hd * hd;
            ab += hd * he;
            bb += he * he;
            ap += hd * hp;
            bp += he * hp;
        }

        let det = aa * bb - ab * ab;
        let (a, b) = ((bb * ap - ab * bp) / det, (aa * bp - ab * ap) / det);

        for (s, c) in d.iter_mut().zip(&*e) {
            s.0 = a * s.0 + b * c.0;
            s.1 = a * s.1 + b * c.1;
        }
    }

    /// Rotor walk from the center outward, both spectra in one pass.
    fn transform2(&self, w0: f64, psi: &mut [(f64, f64)], d: &mut [(f64, f64)]) {
        let step = self.du * w0;

        // NOTE implementation is a phasor walk with Newton's correction.  Got -270dB versus sin_cos
        // and on our very short filters, the phase error accumulates very slowly.  Tests run quite
        // a bit quicker.

        // Outer angles are linear in i: dt advances by step, the j = lo seed by step*lo.
        let (ps, pc) = step.sin_cos();
        let (qs, qc) = (step * self.lo as f64).sin_cos();
        let (mut dc, mut ds) = (1.0f64, 0.0f64);
        let (mut sr, mut si) = (1.0f64, 0.0f64);

        // for i in 0..=bin.last {
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
    /// `1 + (sigma·x)²`, x = j/n. The correction lands as `shape(x)/(1 + (sigma·x)²)`, so
    /// raising sigma pulls it toward the center tap and flattens the slope it contributes to
    /// the error, which is what the reassignment ratios divide through.
    ///
    /// `pin` sets the carrier functionals E and O to a common target, which lands peak gain
    /// and nulls the image in one solve. `None` holds them at their current values, so the
    /// DC correction cannot tilt the band response; that is what `d` wants, since `align_d`
    /// owns its scale.
    ///
    /// Parity splits the problem: the even taps carry the even derivatives at DC and the
    /// carrier's cosine quadrature, the odd taps the odd derivatives and the sine quadrature.
    /// Two independent 3x3 solves sharing one moment table. Sigma is de-rated by sqrt(n/64)
    /// below 64 weights: a short bake has too little support to concentrate the correction
    /// and still keep the Gram invertible.
    fn condition(&self, out: &mut [(f64, f64)], w0: f64, pin: Option<f64>) {
        let n = out.len();
        let inv = (n as f64).recip();

        let s2 = {
            let s = self.sobolev * (n as f64 / 64.0).sqrt().min(1.0);
            s * s
        };

        let g = |j: usize| {
            let x = j as f64 * inv;
            (x, 1.0 / (1.0 + s2 * x * x))
        };

        // Newton-corrected phasor, same walk as transform2.
        let (ws, wc) = w0.sin_cos();
        let step = |cr: &mut f64, ci: &mut f64| {
            let (nr, ni) = (*cr * wc - *ci * ws, *cr * ws + *ci * wc);
            let k = 0.5 * (3.0 - (nr * nr + ni * ni));
            *cr = nr * k;
            *ci = ni * k;
        };

        // Moments of the shapes and the residuals in one pass. Pair multiplicity rides the
        // functionals, not the correction: it cancels out of the stationarity condition.
        let (mut m0, mut m1, mut m2, mut m3) = (0.0f64, 0.0, 0.0, 0.0);
        let (mut q0, mut q1, mut q2) = (0.0f64, 0.0, 0.0);
        let (mut p0, mut p1, mut p2) = (0.0f64, 0.0, 0.0);
        let (mut r0, mut r2, mut i1, mut i3) = (0.0f64, 0.0, 0.0, 0.0);
        let (mut e_car, mut o_car) = (0.0f64, 0.0);

        let (mut cr, mut ci) = (1.0f64, 0.0f64);
        for (j, s) in out.iter().enumerate() {
            let (x, w) = g(j);
            let c = if j == 0 { 1.0 } else { 2.0 };
            let (x2, cw) = (x * x, c * w);

            m0 += cw;
            m1 += cw * x2;
            m2 += cw * x2 * x2;
            m3 += cw * x2 * x2 * x2;

            q0 += cw * cr;
            q1 += cw * x2 * cr;
            q2 += cw * cr * cr;

            p0 += cw * x * ci;
            p1 += cw * x2 * x * ci;
            p2 += cw * ci * ci;

            r0 += c * s.0;
            r2 += c * x2 * s.0;
            i1 += c * x * s.1;
            i3 += c * x2 * x * s.1;

            e_car += c * s.0 * cr;
            o_car += c * s.1 * ci;

            step(&mut cr, &mut ci);
        }

        let (re, ro) = match pin {
            Some(t) => (e_car - t, o_car - t),
            None => (0.0, 0.0),
        };

        // Symmetric 3x3, upper triangle by entry. Solves M·coef = -rhs.
        let solve = |a: [f64; 6], p: f64, q: f64, r: f64| {
            let [a00, a01, a02, a11, a12, a22] = a;
            let (c0, c1, c2) = (
                a11 * a22 - a12 * a12,
                a02 * a12 - a01 * a22,
                a01 * a12 - a02 * a11,
            );
            let det = a00 * c0 + a01 * c1 + a02 * c2;
            let (d0, d1, d2) = (
                a00 * a22 - a02 * a02,
                a02 * a01 - a00 * a12,
                a00 * a11 - a01 * a01,
            );

            (
                -(c0 * p + c1 * q + c2 * r) / det,
                -(c1 * p + d0 * q + d1 * r) / det,
                -(c2 * p + d1 * q + d2 * r) / det,
            )
        };

        // even shapes {1, x², cos}
        let (ea, eb, ec) = solve([m0, m1, q0, m2, q1, q2], r0, r2, re);
        // odd shapes {x, x³, sin}
        let (oa, ob, oc) = solve([m1, m2, p0, m3, p1, p2], i1, i3, ro);

        let (mut cr, mut ci) = (1.0f64, 0.0f64);
        for (j, s) in out.iter_mut().enumerate() {
            let (x, w) = g(j);
            let x2 = x * x;
            s.0 += (ea + eb * x2 + ec * cr) * w;
            s.1 += ((oa + ob * x2) * x + oc * ci) * w;
            step(&mut cr, &mut ci);
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

    /// Rounding functionals. Even lane (real) carries H(0) and H(w0); odd lane (imag)
    /// carries H'(0) and the w0 quadrature. Pair multiplicity rides the row, matching
    /// `condition`. DC curvature was tried here and dropped: those rows go as x² and x³,
    /// so they vanish near the center where the ulps are coarse enough to steer, and only
    /// gain magnitude in the tail where the steps are decades too fine to move anything.
    /// `condition` nulls those moments in f64 and the cast's perturbation clears
    /// `taps_are_conditioned` with room.
    fn rows(j: usize, inv: f64, w0: f64) -> ([f64; 2], [f64; 2]) {
        let c = if j == 0 { 1.0 } else { 2.0 };
        let (s, k) = (w0 * j as f64).sin_cos();
        ([c, c * k], [c * j as f64 * inv, c * s])
    }

    /// Neighbor of `v` in f32 that leaves `e` smallest, residual folded back in.
    /// Walked center-outward, so the coarse center ulps take the gross correction and
    /// the tail grinds the remainder down with progressively finer steps.
    fn round_shaped<const N: usize>(v: f64, row: [f64; N], w: [f64; N], e: &mut [f64; N]) -> f32 {
        let lo = v as f32;
        let hi = if (lo as f64) > v {
            lo.next_down()
        } else {
            lo.next_up()
        };
        let cost = |c: f32| {
            let r = v - c as f64;
            (0..N)
                .map(|k| {
                    let t = e[k] + r * row[k];
                    w[k] * t * t
                })
                .sum::<f64>()
        };
        let c = if cost(hi) < cost(lo) { hi } else { lo };
        let r = v - c as f64;
        for k in 0..N {
            e[k] += r * row[k];
        }
        c
    }

    /// Interleave into `float4`, choosing each tap's rounding direction to keep a small
    /// residual vector small rather than letting it random-walk. Four independent lanes:
    /// ψ and d, real and imaginary.
    fn quantize(psi: &[(f64, f64)], d: &[(f64, f64)], w0: f64, out: &mut [[f32; 4]]) {
        // Commensurate weights. Over-weighting the peak-gain row (W[1]) made the greedy solution
        // myopic: the coarse center taps chase H(w0) and push the DC residuals out where the fine
        // tail taps can't retire them. Balanced residuals converge together and leave the tail
        // enough freedom to land peak gain within an ulp.
        const W: [f64; 2] = [1.0, 1.0];

        let inv = (out.len() as f64).recip();
        let (mut pr_e, mut pi_e) = ([0.0f64; 2], [0.0f64; 2]);
        let (mut dr_e, mut di_e) = ([0.0f64; 2], [0.0f64; 2]);

        for (i, (&(pr, pi), &(dr, di))) in psi.iter().zip(d).enumerate() {
            let (even, odd) = Self::rows(i, inv, w0);

            let qpr = Self::round_shaped(pr, even, W, &mut pr_e);
            let qpi = Self::round_shaped(pi, odd, W, &mut pi_e);
            let qdr = Self::round_shaped(dr, even, W, &mut dr_e);
            let qdi = Self::round_shaped(di, odd, W, &mut di_e);

            out[i] = if i == 0 {
                // Halving a rounded f32 is exact, so the feedback stays consistent.
                [0.5 * qpr, 0.0, 0.5 * qdr, 0.0]
            } else {
                [qpr, qpi, qdr, qdi]
            };
        }
    }
}

/// g(u) = beta*ln(u) - (beta/gamma)*u^gamma, normalized so g(1) = 0.
fn log_env(s: Shape) -> impl Fn(f64) -> f64 {
    let bg = s.beta / s.gamma;
    move |u| s.beta * u.ln() - bg * u.powf(s.gamma) + bg
}

/// Half width in samples, times omega0. Pure function of shape and leakage.
fn half_width_scaled(s: Shape, eps: f64) -> f64 {
    let core = (2.0 * eps.recip().ln()).sqrt() * s.p();
    let tail = eps.recip().powf(1.0 / (2.0 * s.beta + 1.0));
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
    use super::*;

    const BINS: usize = 1024;
    const RATE: f64 = 48_000.0;

    fn spec(q: f64, eps: f64) -> Spec {
        Spec::default().shape(Shape::from_q(q, 3.0)).grid_eps(eps)
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
        // Set RESEED to 1 for full seeding if this test device is under scrutiny.  **Must be power
        // of two for iteration mask.**
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

    /// Max of `f` over `n` samples of [lo, hi].
    fn sweep(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
        (0..=n)
            .map(|k| f(lo + (hi - lo) * k as f64 / n as f64))
            .fold(0.0f64, f64::max)
    }

    /// Worst |H| gap over the middle half of the -3 dB band.
    fn passband_gap(a: &[(f32, f32)], b: &[(f32, f32)], lo: f64, hi: f64) -> f64 {
        let (mid, quarter) = (0.5 * (lo + hi), 0.25 * (hi - lo));
        sweep(mid - quarter, mid + quarter, 2048, |w| {
            (dtft(a, w) - dtft(b, w)).abs()
        })
    }

    /// Truncation floor, swept two octaves starting three octaves off center. Far
    /// enough out that the skirt is gone. Both sides when the upper band fits under
    /// Nyquist, low side alone otherwise.
    fn stopband(a: &[(f32, f32)], b: &[(f32, f32)], w0: f64) -> f64 {
        let gap = |w: f64| (dtft(a, w) - dtft(b, w)).abs();
        let low = sweep(w0 / 32.0, w0 / 8.0, 2048, gap);
        if 32.0 * w0 < PI {
            low.max(sweep(8.0 * w0, 32.0 * w0, 2048, gap))
        } else {
            low
        }
    }

    /// Peak |H| from DC to 5% of center.
    fn dc_leak(taps: &[(f32, f32)], peak_w: f64) -> f64 {
        const STEPS: usize = 64;

        let top = 0.05 * peak_w;
        let mut peak = 0.0f64;
        for k in 0..=STEPS {
            peak = peak.max(dtft(taps, top * k as f64 / STEPS as f64));
        }
        peak
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

    /// W(m) = sum_j x[m + half - j] * h[j], matching `unit_tone_reads_unity`.
    fn conv(h: &[(f32, f32)], x: impl Fn(isize) -> f64, m: isize) -> (f64, f64) {
        let half = (h.len() / 2) as isize;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (j, &(r, i)) in h.iter().enumerate() {
            let s = x(m + half - j as isize);
            re += s * r as f64;
            im += s * i as f64;
        }
        (re, im)
    }

    fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        let q = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / q, (a.1 * b.0 - a.0 * b.1) / q)
    }

    /// Worst frequency bias in cents and worst quadrature leak of `d/psi` against a real tone
    /// detuned `cents` from `w0`, over eight carrier phases.
    fn tone_bias(psi: &[(f32, f32)], d: &[(f32, f32)], w0: f64, cents: f64) -> (f64, f64) {
        let w = w0 * (cents / 1200.0).exp2();
        let tone = |k: isize| (w * k as f64).cos();
        let (mut bias, mut quad) = (0.0f64, 0.0f64);
        for m in 0..8 {
            let (re, im) = cdiv(conv(d, tone, m), conv(psi, tone, m));
            bias = bias.max((1200.0 * (re / w).log2()).abs());
            quad = quad.max((im / w).abs());
        }
        (bias, quad)
    }

    /// Worst group-delay error in samples and worst real leak of `t/psi` across the impulse
    /// response, skipping positions where the envelope is too small to divide through.
    fn impulse_delay(psi: &[(f32, f32)], t: &[(f32, f32)]) -> (f64, f64) {
        let half = (psi.len() / 2) as isize;
        let imp = |k: isize| if k == 0 { 1.0 } else { 0.0 };
        let env0 = (psi[half as usize].0 as f64).hypot(psi[half as usize].1 as f64);

        let (mut worst, mut leak) = (0.0f64, 0.0f64);
        for m in -half..=half {
            let wp = conv(psi, imp, m);
            if wp.0.hypot(wp.1) < 1e-2 * env0 {
                continue;
            }
            let (re, im) = cdiv(conv(t, imp, m), wp);
            worst = worst.max((m as f64 + im).abs());
            leak = leak.max(re.abs());
        }
        (worst, leak)
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
    fn characterize(taps: &[(f32, f32)], w0: f64) -> Response {
        let sweep = (16 * taps.len()).next_power_of_two();
        let omega = |k: usize| -PI + 2.0 * PI * k as f64 / sweep as f64;

        let (mut peak, mut neg) = ((0.0f64, 0.0f64), 0.0f64);
        for k in 0..=SWEEP {
            let w = omega(k);
            let v = dtft(taps, w);
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
            if dtft(taps, m1) < dtft(taps, m2) {
                a = m1;
            } else {
                b = m2;
            }
        }
        let peak_w = 0.5 * (a + b);
        let peak_h = dtft(taps, peak_w);

        let half = peak_h / 2.0f64.sqrt();
        let lo = crossing(taps, peak_w - w0, peak_w, half);
        let hi = crossing(taps, peak_w, (peak_w + w0).min(PI), half);

        let guard = 3.0 * (hi - lo);
        let mut floor = 0.0f64;
        for k in 0..=SWEEP {
            let w = omega(k);
            if (w - peak_w).abs() > guard {
                floor = floor.max(dtft(taps, w));
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
            let bin = plan.bin(1000.0, 8000.0, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
            plan.taps_into(bin, &mut w);

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
            .plan();
        let bins = bank::bins(2_000.0, 20_000.0, BINS);
        println!("planning time: {:?}µs", start.elapsed().as_micros());

        let voices: Vec<Bin> = bins
            .iter()
            .map(|b| plan.bin(b.center, RATE, load_quantum))
            .collect();

        let total: usize = voices.iter().map(Bin::folded_taps).sum();
        let mut weights = vec![[0.0f32; 4]; total];

        let mut offsets = Vec::with_capacity(voices.len());
        let mut cursor = 0;
        for &bin in &voices {
            offsets.push(cursor);
            cursor += plan.taps_into(bin, &mut weights[cursor..]);
        }

        let elapsed = start.elapsed();

        let worst = voices
            .iter()
            .zip(&offsets)
            .map(|(b, &o)| {
                let n = b.folded_taps();
                (dtft(&unfold(&weights[o..o + n], 0), b.velocity()) - PEAK_GAIN).abs()
            })
            .fold(0.0f64, f64::max);
        println!("worst peak gain error: {worst:.3e}");
        assert!(worst < 1e-3, "worst peak gain error {worst:.3e}");

        let lowest = unfold(&weights[..voices[0].folded_taps()], 0);
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
            voices[0].folded_taps(),
            voices[voices.len() - 1].folded_taps(),
        );

        println!("bin filling time: {:?}µs", elapsed.as_micros());
    }

    /// A real unit tone reads |W| = 1 even though |H| = 2: the analytic taps
    /// see only the +w half of the cosine. Swept over the quantum.
    #[test]
    fn unit_tone_reads_unity() {
        for quantum in [1usize, 4, 8] {
            let mut p = spec(3.0, 1e-8).max_load_quantum(quantum).plan();

            for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = p.bin(fc, sr, quantum);
                let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
                p.taps_into(bin, &mut w);

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
        // Quantum rounding pads the emitted half-span.
        const QUANTUM: usize = 4;

        // Tap count is an integer, so envelope truncation loses O(1/N) of the
        // energy, worst at the top of the range. Anchored to split the sweep
        // rather than to any one bin.
        const NOISE_GAIN: f64 = 0.224777;
        const TOL: f64 = 2e-3;

        let mut p = spec(3.0, 1e-8).max_load_quantum(QUANTUM).plan();

        println!("\n=== NOISE GAIN (Q = 3, sr = {RATE}, quantum {QUANTUM}) ===");

        for fc in [500.0f64, 1000.0, 2000.0, 4000.0, 8000.0] {
            let bin = p.bin(fc, RATE, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
            p.taps_into(bin, &mut w);

            let psi = unfold(&w, 0);
            let e: f64 = psi
                .iter()
                .map(|&(r, i)| (r as f64).powi(2) + (i as f64).powi(2))
                .sum();
            let ratio = e / bin.velocity();

            println!(
                "  fc {:>6.0}  taps {:>5}  energy {:.6}  e/w0 {:.6}  dev {:+.2e}",
                fc,
                bin.unfolded_taps(),
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

        let mut plan = spec(3.5, 1e-10).max_load_quantum(QUANTUM).plan();

        for (fc, sr) in [(2_000.0f64, RATE), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = plan.bin(fc, sr, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
            plan.taps_into(bin, &mut w);
            let psi = unfold(&w, 0);
            let (d, t) = (unfold(&w, 2), derive_t(&psi));

            let (n, w0) = (psi.len(), bin.velocity());
            println!("\n=== REASSIGN fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6} ===");

            // Pitch reassignment
            for cents in [-600.0f64, -300.0, 0.0, 300.0, 600.0] {
                let (bias, quad) = tone_bias(&psi, &d, w0, cents);
                println!("  detune {cents:+6.0}c  bias {bias:.5}c  quad {quad:.3e}");
                assert!(bias < 2.0, "fc {fc} detune {cents} bias {bias:.3}c");
                assert!(quad < 1e-3, "fc {fc} detune {cents} quad {quad:.3e}");
            }

            // Time reassignment
            let (worst, real) = impulse_delay(&psi, &t);
            println!("  impulse: worst t_hat {worst:.4} samples  real leak {real:.2e}");
            assert!(worst < 0.05, "fc {fc} t_hat off by {worst:.4} samples");
            assert!(real < 1e-2, "fc {fc} t real leak {real:.3e}");
        }
    }

    /// Truncation cost against a full-length bake, swept over `sigmas`.  Stop band, DC leak,
    /// ripple in the pass, width, and gain are all compared.
    #[test]
    fn truncation_is_predictable() {
        const QUANTUM: usize = 4;
        const Q: f64 = 3.5;

        // NOTE these are empirically discovered values stored to catch regressions.

        // Measured 0.9984 at Q = 3.5.
        const WIDTH_Q: f64 = 0.998;
        const WIDTH_TOL: f64 = 0.001;

        // Stopband gap relative to PEAK_GAIN, as a multiple of sigmas.
        const LEAK_PER_SIGMA: f64 = 0.0001;

        // In-band relative error as a multiple of sigmas
        const PASS_PER_SIGMA: f64 = 0.001;

        let base = Spec::default()
            .shape(Shape::from_q(Q, 3.0))
            .sigmas(8.0)
            .max_load_quantum(QUANTUM);
        let mut full = base.plan();

        let db = |v: f64| 20.0 * (v / PEAK_GAIN).log10();

        println!(
            "\n=== TRUNCATION (Q = {Q}, quantum {QUANTUM}) ===\n\
            dB reference peak gain {PEAK_GAIN:.1}; values except DC are dB relative to full-length bake"
        );

        for fc in [2_000.0f64, 4_000.0, 8_000.0, 14_000.0] {
            let bf = full.bin(fc, RATE, QUANTUM);
            let nf = bf.folded_taps();
            let mut wf = vec![[0.0f32; 4]; nf];
            full.taps_into(bf, &mut wf);
            let pf = unfold(&wf, 0);

            let w0 = bf.velocity();
            let rf = characterize(&pf, w0);

            println!(
                "\n  fc {fc:>6.0}  full weights {nf:>5}  peak {:.9}  rel width {:.5} \
                 (x Q = {:.4})",
                rf.peak_h,
                rf.rel_width,
                rf.rel_width * Q
            );

            // The full length bake is held to the same conditioning as the time-truncated ones.
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

            for sigmas in [3.5, 4.5, 5.5, 6.5] {
                let mut cut = base.sigmas(sigmas).plan();
                let bc = cut.bin(fc, RATE, QUANTUM);
                let nc = bc.folded_taps();
                let mut wc = vec![[0.0f32; 4]; nc];
                cut.taps_into(bc, &mut wc);
                let pc = unfold(&wc, 0);

                let pass = passband_gap(&pf, &pc, rf.edges.0, rf.edges.1);
                let stop = stopband(&pf, &pc, w0);
                let dc = dc_leak(&pc, w0);

                let rc = characterize(&pc, w0);
                let cents = 1200.0 * (rc.peak_w / rf.peak_w).log2();

                println!(
                    "    sigmas {sigmas:>3.2}  weights (folded) {nc:>4} ({:.3})  \
                    pass {:>7.2} dB  stop {:>7.2} dB  dc {:>7.2} dB  peak {:+.4}c  width {:+.3}%",
                    nc as f64 / nf as f64,
                    db(pass),
                    db(stop),
                    db(dc),
                    cents,
                    100.0 * (rc.rel_width / rf.rel_width - 1.0)
                );

                // In-band magnitude is flat to well under the leakage budget. The peak is
                // pinned by gain_at; this checks the core around it didn't tilt.
                assert!(
                    pass < PASS_PER_SIGMA * sigmas,
                    "fc {fc} sigmas {sigmas:e} passband {:.2} dB rel",
                    20.0 * pass.log10()
                );

                // Peak gain survives truncation, and the band neither moves nor widens.
                assert!(
                    cents.abs() < 8.0,
                    "fc {fc} sigmas {sigmas:e} peak moved {cents:+.4}c"
                );
                assert!(
                    (rc.rel_width / rf.rel_width - 1.0).abs() < 0.02,
                    "fc {fc} sigmas {sigmas:e} width {:+.3}%",
                    100.0 * (rc.rel_width / rf.rel_width - 1.0)
                );

                // Turning off conditioning should break this, but the more heavily truncated
                // filters also tend to trip it.
                assert!(
                    dc < 1e-2 * PEAK_GAIN,
                    "fc {fc} sigmas {sigmas:e} dc {:.2} dB",
                    20.0 * (dc / PEAK_GAIN).log10()
                );

                // More sigmas truncates less, so it uses more taps to buy stopband.  Taps may stay
                // the same due to quantum rounding.
                assert!(
                    nc >= prev_taps,
                    "fc {fc} sigmas {sigmas:e} taps {nc} < {prev_taps}"
                );
                // If taps go up, stop band must go down.
                assert!(
                    (nc >= prev_taps && stop >= prev_stop) || stop < prev_stop,
                    "fc {fc} sigmas {sigmas:e} taps {prev_taps} -> {nc} without stopband gain"
                );

                assert!(
                    stop < LEAK_PER_SIGMA * sigmas * PEAK_GAIN,
                    "fc {fc} sigmas {sigmas:e} stop {:.2} dB",
                    20.0 * (stop / PEAK_GAIN).log10()
                );

                (prev_taps, prev_stop) = (nc, stop);
            }
        }
    }

    /// Same four numbers as `response_is_characterized`, measured on the folded weight
    /// table. Sweeps the load quantum, because the quantum pads the emitted half-span.
    #[test]
    fn table_response_is_characterized() {
        let (q, grid_eps, sigmas) = (3.5, 1e-10, 4.0);
        for quantum in [2usize, 4, 8, 16] {
            let mut p = spec(q, grid_eps)
                .sigmas(sigmas)
                .max_load_quantum(quantum)
                .plan();

            println!("\n=== TABLE RESPONSE (Q = {q}, quantum {quantum}) ===");

            for (fc, sr) in [(1000.0f64, 6000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = p.bin(fc, sr, quantum);
                let n = bin.folded_taps();
                let mut w = vec![[0.0f32; 4]; n];
                p.taps_into(bin, &mut w);

                let w0 = bin.velocity();
                let psi = unfold(&w, 0);
                let r = characterize(&psi, w0);

                let db = |v: f64| 20.0 * (v / r.peak_h).log10();

                println!(
                    "\nfc {fc:>5.0} sr {sr:>5.0}  w0 {w0:.6}  quantized {n:>3} (unfolded {:>3})",
                    bin.unfolded_taps()
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

        let mut p = spec(3.0, 1e-10).max_load_quantum(QUANTUM).plan();

        for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = p.bin(fc, sr, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
            p.taps_into(bin, &mut w);

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

            let mom = |p: u32| {
                psi.iter()
                    .enumerate()
                    .map(|(j, &(r, i))| {
                        let nu = (j as isize - (psi.len() / 2) as isize) as f64;
                        nu.powi(p as i32) * if p % 2 == 0 { r as f64 } else { i as f64 }
                    })
                    .sum::<f64>()
            };

            // H''(0) and H'''(0), the two the solve nulls that nothing else measures.
            let (m2, m3) = (mom(2), mom(3));
            assert!(m2.abs() < 1e-3 * g, "fc {fc} second moment {m2:.3e}");
            assert!(m3.abs() < 1e-3 * g, "fc {fc} third moment {m3:.3e}");
        }
    }
}
