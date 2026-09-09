// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// ⚠️ The module documents are aspirational and not implemented yet.  We are in the middle of
// implementing the results of the design process conclusions.

//! # The Wavelet
//!
//! > The traveler who fears the unknown road will eventually learn that known roads return
//! > to where they began.
//! >
//! > - Anthony L. Ray
//!
//!                ·
//!                ·
//!                ·●
//!                 ●
//!                 •●
//!                 •·
//!                ●···
//!              ●●●··
//!             ●●●●
//!             ●•••
//!            ···••
//!           ······●●
//!             ····●●●●●●
//!                 ●●●●●●●●●
//!                 ••••••●●
//!                 •••·······
//!             ●●●●···········
//!       ●●●●●●●●●●·······
//!    ●●●●●●●●●●●●●
//!      ●●●••••••••
//!   ··········••••
//!   ··············●●●●●
//!        ·········●●●●●●●●●●●●
//!                 ●●●●●●●●●●●●●●●
//!                 •••••••••●●●
//!                 •••••·········
//!             ●●●●··············
//!      ●●●●●●●●●●●········
//!    ●●●●●●●●●●●●●
//!       ●●●•••••••
//!      ·······••••
//!       ··········●●●
//!           ······●●●●●●●●
//!                 ●●●●●●●●●
//!                 ••••●●
//!                 ••····
//!               ●●·····
//!             ●●●●···
//!             ●●●●
//!              ●••
//!              ··•
//!               ··●
//!                ·●●
//!                 ●
//!                 •
//!                 ·
//!                 ·
//!
//! This module generates our wavelet tables.  The Morse wavelet is the chosen implementation.
//!
//! - Easy to generate
//! - Regarded as nice for time and frequency reassignment
//! - Parameterized (but only a little!)
//!
//! ## Usage
//!
//! Configure a [`Spec`] and use it to build a [`Plan`]. Each [`Plan`] holds the intermediate data
//! in a frequency independent form, so every voice sharing those settings may reuse one `Plan`.
//!
//! ```
//! # use mutate_lib::dsp::wavelet::Spec;
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
//! usually loaded individually for a final reduction, also giving us an **odd tap invariant**.
//!
//! **Example:**
//!
//! - Implementation loads eight weights at a time for use in evaluating sixteen unfolded taps.
//! - The if truncation for the configured options would normally lead to 17 *folded* weights, the
//!   implementation will round up to 24 instead, meaning 48 *unfolded* taps and 49 including the
//!   center.
//! - The extra weights are used to shape truncation aggressively, increasing precision and
//!   hopefully de-correlating some locally biased response to transients, smearing artifacts.
//!
//! ```
//! # use mutate_lib::dsp::wavelet::Spec;
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
//! To support both pitch and time reassignment, we need three channels:
//!
//! | symbol | Rust | object | units | frame |
//! |---|---|---|---|---|
//! | `ψ` | `psi` | the wavelet | — | `u` |
//! | `d` || `−(i/2π)·dψ/du`, the rotated derivative for pitch reassignment | dimensionless | `u` |
//! | `t` || `ν·ψ(ρν)`, the wavelet's first moment kernel for time reassignment | samples | `ν` |
//!
//! Every wavelet uses an odd number of taps and each tap carries three weights, but we exploit the
//! following relations:
//!
//! - `ψ` is Hermitian.
//! - `−(i/2π)·dψ/du` is the rotated derivative and therefore also Hermitian.
//! - `ν·ψ(νρ)` is anti-Hermitian.  It carries `ν`, so `t` reads the same two half-pair
//!   products with the roles exchanged, `dif` against `Re ψ` and `sum` against `Im ψ`.
//!
//! So we fold `ψ` and `d` as sums and use `k` to derive `t` as the difference of the same products.
//! The resulting table is `N/2 + 1` entries of four floats, laid out as a Slang `float4` in the
//! order `(Re ψ, Im ψ, Re d, Im d)`.  The center tap is real in both channels, `(Re ψ₀, 0, Re d₀,
//! 0)`, and contributes nothing to `t`.
//!
//! ## Generation Pipeline
//!
//! Low quality wavelets can cause many bad things that good things cannot fix.  A principled
//! approach upstream is required.  We can divide the strategy into five phases:
//!
//! - Define coherent goals, such as `load_quantum` and `Q`, obtaining a [`Spec`].
//! - Generate high resolution mother wavelet sufficient to fill any tap count that the `Spec` may
//!   produce.
//! - Dilate and truncate the mother wavelet to the length required to fill `N` taps.
//! - Restrict (fancy downsampling) the high resolution daughter wavelet into the coarse `N` taps.
//! - Repair the truncation to more closely recreate the ideal infinite wavelet's **transient**
//!   response.
//!
//! The final repair steps use the measured (by quadrature) moments of the high-fidelity daughterlet
//! and analytically determined contributions from the truncated ends.  Together, these are the mass
//! and moments of the complete ideal wavelet.  Corrections approach this ideal wavelet to minimize
//! the spectral ringing of the truncation in a minimal number of taps without damaging the
//! desirable response of the wavelet's main body.
//!
//! ## Symbols and Nomenclature
//!
//! Unrealized wavelets are frequently handled using period normalized coordinates, `u` periods from
//! the center.   Only when realizing a [`Bin`] into memory do frequency `fs` and center
//! frequency `fc` appear.  The concrete using the appropriate `ρ` period density.
//!
//! ### Coordinates
//!
//! Wavelets are created and dilated using the real coordinate and mapped to and from integral
//! indexed grids using `ρ`, periods per tap or grid sample (depends on context, but the choice is
//! always obvious).
//!
//! | symbol | Rust | object | units |
//! |---|---|---|---|
//! | `u` || periods from the wavelet's center, a real | carrier periods |
//! | `ρ` | `rho` | the conversion ratio, a linear density | periods per tap |
//!
//! `resolution` is the number of grid points per period `u` and decides how finely each period of
//! the mother wavelet will be resolved before restriction to `N` taps.
//!
//! There are additionally three integral coordinates distinguished by the use case:
//!
//! | symbol | Rust | object | units |
//! |---|---|---|---|
//! | `m` || center sample index of the analysis window, the origin | samples |
//! | `ν` | `nu` | signed integer tap index relative to `m`, in `(-K, K)` | taps |
//! | `k` || taps from the center tap, in `[0, K)` and only used for symmetric work | taps |
//! | `n` || unfolded tap index, `n = ν + K − 1` on `[0, N)` | taps |
//!
//! - `n` is used for indexing unfolded taps, such as for computering DFTs of filters for tests.
//! - `m` shows back up in shaders as the index of the center tap sample.  That anchors the analysis
//!   window for each hop.
//! - `k` makes the most sense when modifying taps since there are not actually two distinct taps,
//!    while `nu` makes the most sense when using taps, where lane assignment will use `m` for the
//!    center index.
//!
//! ### Projections
//!
//! Correlations of the signal `x` against each channel about center `m`.
//!
//! | symbol | Rust | object |
//! |---|---|---|
//! | `Ψ` | `psi_sum` | `Σ_ν conj(ψ_ν)·x_{m+ν}` |
//! | `D` | `d_sum` | the same sum against `d_ν` |
//! | `T` | `t_sum` | the same sum against `t_ν` |
//!
//! ### Estimators
//!
//! | symbol | Rust | object | units | frame |
//! |---|---|---|---|---|
//! | `r̂` | `r_hat` | `Re(D/Ψ)`, the detuning | multiples of the carrier | `u` |
//! | `t̂` | `t_hat` | `m + Re(T/Ψ)`, reassigned time | samples | `ν` |
//!
//! Each inherits its channel's frame.  `t̂` is already on the output grid.  `r̂` reaches physical
//! frequency through `f = r̂·f_c`, and bank placement wants `log2 r̂`.
//!
//! ## Sample Computation
//!
//! Unpack weights and obtain `r_hat` and `t_hat`.
//!
//! ```slang
//! // per hop, K = N/2 + 1 entries, w[k] = (Re ψ, Im ψ, Re d, Im d)
//! float2 psi = float2(w[0].x, 0.0) * x[m];
//! float2 dee = float2(w[0].z, 0.0) * x[m];
//! float2 tee = float2(0.0, 0.0);
//!
//! for (uint k = 1; k < K; ++k) {
//!     float sum = x[m + k] + x[m - k];
//!     float dif = x[m + k] - x[m - k];
//!     float4 c = w[k];
//!
//!     psi += float2( c.x * sum, -c.y * dif);
//!     dee += float2( c.z * sum, -c.w * dif);
//!     tee += float(k) * float2( c.x * dif, -c.y * sum);
//! }
//!
//! // Re(D·conj Ψ)/|Ψ|² and Re(T·conj Ψ)/|Ψ|²
//! float inv   = 1.0 / dot(psi, psi);
//! float r_hat = dot(dee, psi) * inv;
//! float t_hat = float(m) + dot(tee, psi) * inv;
//! ```
//!
//! ## Compute & Register Cost
//!
//! - 5 ops per tap
//! - 4 f32s per 2 mirrored taps, 2 per tap
//! - 3 (W_ψ, W_d, W_t) complex accumulators, 6 registers per pipelined hop.
//!
//! For `P` pipelines, we need `6P + 6 + 2` non-uniform registers per lane and some scratch for
//! temporaries.  Audio is 8 bytes per sample at two channels and each 8 bytes of audio read can be
//! re-used for `P` taps for each read.  Each mirrored tap applies to two audio samples per
//! pipelined hop.

// 🤖 Heavy generation.  Should be pretty standard academic stuff, so not expecting a lot of
// surprises.  We will, for the most part, swiftly and knowingly eat shit if the wavelet is busted.
// Well-formalized stuff doesn't have a lot of wiggle room to violate the consistency of the
// formalism.

// NEXT A ton of the characterization gear for testing belongs in the dsp module.
// NEXT Transient behavior evaluation to look for negative frequency response under impure tones.
// NEXT PSSL and skirt detection to qualify the near-field spectral leakage.  Fine scan to search
// for combs.
// NEXT High omega filters, starting at around 60% of Nyquist, begin to degrade at low Q.  The
// carrier doesn't have enough detail to represent a fast-changing envelope.  A numerical solution
// for these heavily aliased wavelets may succeed or we may use a Plan with a higher Q beyond some
// empirical cutoff.  Truncating less aggressively can't help because the main lobe itself is
// dominating the breakdown.
// NEXT Noise floor performance, which provides our dynamic range resolution, is very sensitive to
// the number of taps for a given Q.  We would like to improve Q without raising N taps and we would
// like to make more use of our padding quantum, but only an integrated design process, one that
// solves for measured goals, such as noise floor and reassignment accuracy, is going to squeeze
// more f32 pixie dust out.  Several tapering schemes ranging from Plank envelope to a higher order
// solution were tried, but none consistently improved the result of simple truncation.
// NOTE We have logarithmic bin spacings, but the cutoff frequencies that determine which downsample
// will be used are not particularly aware, so it's not expected that we can re-use exact bins in
// any kind of octave structure.  Mel scaling etc also defeats this, so there's no point.
// NOTE Run time of the filter bank generation test (not reflective of actual sample rates and Q) is
// about 30ms on a Zen2+ part in release.  This affects CWT startup time.

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

pub mod generate;
pub mod restrict;
pub mod spec;
pub mod whatsleft;

use core::f64::consts::{LN_10, LN_2, PI, TAU};

use num_complex;

use spec::{Shape, Spec};

/// Filter peak gain. Analytic taps see half a real tone's amplitude, so |H| = 2 makes a unit tone
/// read |W| = 1.
const PEAK_GAIN: f64 = 2.0;

/// A center frequency resolved against a plan, a sample rate, and a load quantum.  Allocating
/// memory and estimating compute load for a filter bank requires at least approximate knowledge of
/// the taps.  Calibration wants to know how those filters are expected to perform.  The `Bin`
/// stores this queryable information before the filter taps are realized.
#[derive(Clone, Copy)]
pub struct Bin {
    w0: f64,
    quantum: usize,
    k: usize,
}

impl Bin {
    /// Rotational velocity ദ്ദി(•̀ω-)✧ in radians per sample.  Learn to speak 𝛑. 🥧
    pub fn velocity(&self) -> f64 {
        self.w0
    }

    /// Periods per tap, the cycle density of the output taps.
    pub fn rho(&self) -> f64 {
        todo!()
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
}

// CWT weight table generator.
pub struct Plan {
    shape: Shape,
    c: f64,               // truncation half-span, scaled: sigmas * P
    du: f64,              // uniform step in u = w/w_peak
    spec: Vec<[f64; 2]>,  // [psi, d] at u_j = j*du
    lo: usize,            // first grid point above eps
    buf: Vec<(f64, f64)>, // bake scratch, 2 spans
    max_load_quantum: usize,
    /// Worst unretired row from the last `correct`, row-normalized. Nonzero means the
    /// corrector basis went collinear.
    pub floor: f64,
}

impl Plan {
    /// Generate a bin definition at `center` frequency, sampled at `rate`, loaded `quantum` weights
    /// at a time.
    ///
    /// Snaps the requested half-span to a half-integer number of carrier cycles so the last
    /// tap lands on a carrier extremum, then rounds up to the quantum. Every bin sees the
    /// same scaled half-span and so the same response; the quantum padding is extra aperture
    /// past the snap, not zeros.
    pub fn bin(&self, center: f64, rate: f64, quantum: usize) -> Bin {
        debug_assert!(
            center < rate / 2.0,
            "center {center:.1}Hz above Nyquist {:.0}Hz",
            rate / 2.0,
        );
        debug_assert!(
            quantum <= self.max_load_quantum,
            "quantum {quantum} above plan ceiling {}",
            self.max_load_quantum
        );

        let w0 = TAU * center / rate;

        // XXX Fix here
        let span = (self.c / PI).round() * PI / w0;
        let n = (span / quantum as f64).ceil() as usize * quantum;

        Bin {
            w0,
            quantum,
            k: n + 1,
        }
    }

    /// Writes `bin.folded_taps()` weights and returns that count.
    ///
    /// Upstream owes an `out` at least that long.
    pub fn taps_into(&mut self, bin: Bin, out: &mut [[f32; 4]]) -> usize {
        let k = bin.k;
        let out = &mut out[..k];

        let mut buf = core::mem::take(&mut self.buf);
        buf.resize(2 * k, (0.0, 0.0));
        {
            let (psi, d) = buf.split_at_mut(k);

            let rho = bin.w0;

            // Self::quantize(psi, d, bin.w0, out);
        }
        self.buf = buf;
        k
    }

    // XXX This will go away soon.  The idea that will stick around is we want to know the last f64
    // goal that dipped below f32 representation and then round in a goal aware way.  This function
    // never expressed the goal part.
    /// Rounding functionals.
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

        // NEXT very fine differences, such as comparing dtft of psi with the filter, will clearly
        // demonstrate jumps from -177dB to -215dB after enabling rounding conditioning.  Since our
        // targets are around -100dB of usable dynamic range, an integrated approach to filter
        // tuning and final f32 truncation appears beneficial.
        let round = |v: f64, row: [f64; 2], e: &mut [f64; 2]| Self::round_shaped(v, row, W, e);

        // Disable weights and truncate to f32 naively.
        // let round = |v: f64, row: [f64; 2], e: &mut [f64; 2]| v as f32;

        let inv = (out.len() as f64).recip();
        let (mut pr_e, mut pi_e) = ([0.0f64; 2], [0.0f64; 2]);
        let (mut dr_e, mut di_e) = ([0.0f64; 2], [0.0f64; 2]);

        for (i, (&(pr, pi), &(dr, di))) in psi.iter().zip(d).enumerate() {
            let (even, odd) = Self::rows(i, inv, w0);

            let qpr = round(pr, even, &mut pr_e);
            let qpi = round(pi, odd, &mut pi_e);
            let qdr = round(dr, even, &mut dr_e);
            let qdi = round(di, odd, &mut di_e);

            out[i] = if i == 0 {
                // Halving a rounded f32 is exact, so the feedback stays consistent.
                [0.5 * qpr, 0.0, 0.5 * qdr, 0.0]
            } else {
                [qpr, qpi, qdr, qdi]
            };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const BINS: usize = 1024;
    const RATE: f64 = 48_000.0;

    fn spec(q: f64, eps: f64) -> Spec {
        Spec::default().with_shape(Shape::from_q(q, 3.0)).eps(eps)
    }

    // NOTE we have num_complex btw.  Just bing lazy.
    fn mag((re, im): (f32, f32)) -> f64 {
        let (re, im) = (re as f64, im as f64);
        (re * re + im * im).sqrt()
    }

    // XXX combine these two and use num_complex
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

    // XXX use num_complex
    fn dtft_c(taps: &[(f32, f32)], w: f64) -> (f64, f64) {
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
        (re, im)
    }

    /// |H_d(w) − w·H_psi(w)|, the pairing the taper and the DC corrector both promise to
    /// preserve at every w. Absolute, because dividing by H_psi is exactly what turns a flat
    /// floor into a skirt blowup in the cents column.
    fn pairing_residual(psi: &[(f32, f32)], d: &[(f32, f32)], w: f64) -> f64 {
        let (pr, pi) = dtft_c(psi, w);
        let (dr, di) = dtft_c(d, w);
        (dr - w * pr).hypot(di - w * pi)
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
    fn derive_t(psi: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let half = (psi.len() / 2) as isize;
        psi.iter()
            .enumerate()
            .map(|(j, &(r, i))| {
                let nu = (j as isize - half) as f64;
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

    fn widen(taps: &[(f32, f32)]) -> Vec<(f64, f64)> {
        taps.iter().map(|&(r, i)| (r as f64, i as f64)).collect()
    }

    /// W(m) = sum_j x[m + half - j] * h[j], matching `unit_tone_reads_unity`.
    fn conv(h: &[(f64, f64)], x: impl Fn(isize) -> f64, m: isize) -> (f64, f64) {
        let half = (h.len() / 2) as isize;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for (j, &(r, i)) in h.iter().enumerate() {
            let s = x(m + half - j as isize);
            re += s * r;
            im += s * i;
        }
        (re, im)
    }

    fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
        let q = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / q, (a.1 * b.0 - a.0 * b.1) / q)
    }

    /// Worst frequency bias in cents and worst quadrature leak of `d/psi` against a real tone
    /// detuned `cents` from `w0`, over eight carrier phases.
    fn tone_bias(psi: &[(f64, f64)], d: &[(f64, f64)], w0: f64, cents: f64) -> (f64, f64) {
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

    /// Untruncated f64 taps from `plan`'s grid, unconditioned and unquantized: the estimator's
    /// own answer, so a gap against the shipped table is ours and not the wavelet's.
    ///
    /// `k` folded weights must clear the replica `spans` placed for this plan.
    fn reference(plan: &Plan, w0: f64, k: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let (mut psi, mut d) = (vec![(0.0, 0.0); k], vec![(0.0, 0.0); k]);

        // XXX naked transform ah, so this really is just to get a raw wavelet
        // plan.transform2(w0, &mut psi, &mut d);

        // XXX This is a conditioning that sets the wavelet's gain.
        // Belongs as a feature of a daughter wavelet.
        // let g = Plan::gain_at(&psi, w0);
        // Plan::scale_by(&mut psi, PEAK_GAIN / g);

        let mut out = vec![(0.0f64, 0.0); 2 * k - 1];
        out[k - 1] = (psi[0].0, 0.0);
        for (j, &(r, i)) in psi.iter().enumerate().skip(1) {
            out[k - 1 + j] = (r, i);
            out[k - 1 - j] = (r, -i);
        }
        let t = derive_t(&out);
        (out, t)
    }

    /// Gaussian tone burst, carrier `w`, envelope sd in samples, centered at `p`.
    fn burst(w: f64, sd: f64, p: f64) -> impl Fn(isize) -> f64 {
        move |k| {
            let z = (k as f64 - p) / sd;
            (-0.5 * z * z).exp() * (w * (k as f64 - p)).cos()
        }
    }

    /// Hop levels the buckets accumulate over, dB below the loudest hop.
    const LEVELS: [f64; 3] = [-20.0, -40.0, -60.0];

    /// Worst |t̂ − t̂_ref| in samples per level bucket, plus the worst real leak in the top
    /// bucket. Cumulative, so the -60 dB entry contains the -20 dB one. Reassignment only has
    /// to hold where the pixel is bright enough to see.
    fn t_hat_profile(
        table: (&[(f64, f64)], &[(f64, f64)]),
        reference: (&[(f64, f64)], &[(f64, f64)]),
        x: impl Fn(isize) -> f64,
        span: isize,
    ) -> ([f64; 3], f64) {
        let ((psi, t), (rpsi, rt)) = (table, reference);

        let hops: Vec<(f64, f64, f64)> = (-span..=span)
            .map(|m| {
                let wp = conv(psi, &x, m);
                let (re, im) = cdiv(conv(t, &x, m), wp);
                let r = cdiv(conv(rt, &x, m), conv(rpsi, &x, m));
                (wp.0.hypot(wp.1), (im - r.1).abs(), (re - r.0).abs())
            })
            .collect();

        let peak = hops.iter().fold(0.0f64, |a, h| a.max(h.0));
        let (mut worst, mut leak) = ([0.0f64; 3], 0.0f64);
        for (lvl, err, re) in hops {
            let db = 20.0 * (lvl / peak).log10();
            for (w, &l) in worst.iter_mut().zip(&LEVELS) {
                if db >= l {
                    *w = w.max(err);
                }
            }
            if db >= LEVELS[0] {
                leak = leak.max(re);
            }
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

        let mut mag = Vec::with_capacity(sweep + 1);
        let (mut peak, mut neg) = ((0.0f64, 0.0f64), 0.0f64);
        for k in 0..=sweep {
            let w = omega(k);
            let v = dtft(taps, w);
            mag.push(v);
            if w < 0.0 {
                neg = neg.max(v);
            }
            if v > peak.1 {
                peak = (w, v);
            }
        }

        let cell = 2.0 * PI / sweep as f64;
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
        for (k, &v) in mag.iter().enumerate() {
            if (omega(k) - peak_w).abs() > guard {
                floor = floor.max(v);
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
                .with_shape(Shape {
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
        use crate::dsp::bank;

        let load_quantum = 4;
        let start = std::time::Instant::now();
        let mut plan = Spec::default()
            .max_taps(1024)
            .with_shape(Shape::from_q(5.0, 3.0))
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

    /// Pitch reassignment bias and quadrature leak against steady tones.  Scans across the range
    /// where the wavelet's ideal response is in `[-GATE_DB, GATE_DB)`.  The predicted bias is
    /// compared to the observed.  Too large of bias in the main lobe will trip the asserts.
    #[test]
    fn reassignment_is_unbiased() {
        const QUANTUM: usize = 4;

        /// Cents readings stop meaning anything once the skirt is down in truncation ripple:
        /// the denominator is no longer the envelope, so the ratio is measuring the stopband.
        const GATE_DB: f64 = -20.0;

        /// `pred` is the bias the pairing residual alone implies. The rest is the negative
        /// frequency image, flat in level and so stated absolutely.
        const MIRROR_C: f64 = 0.25;
        const SLOP: f64 = 10.0; // 🫠

        const STEP: f64 = 100.0;
        const SPAN: isize = 12;

        let mut plan = spec(3.5, 1e-10)
            .sigmas(4.5)
            .max_load_quantum(QUANTUM)
            .plan();

        for (fc, sr) in [(2_000.0f64, RATE), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = plan.bin(fc, sr, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
            plan.taps_into(bin, &mut w);

            let psi32 = unfold(&w, 0);
            let d32 = unfold(&w, 2);
            let psi = widen(&psi32);
            let d = widen(&d32);

            let (n, w0) = (psi.len(), bin.velocity());
            println!("\n=== REASSIGN fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6} ===");

            for k in -SPAN..=SPAN {
                let cents = k as f64 * STEP;
                let wd = w0 * (cents / 1200.0).exp2();
                let h = dtft(&psi32, wd);
                let h_db = 20.0 * (h / PEAK_GAIN).log10();
                if h_db < GATE_DB {
                    continue;
                }

                let (bias, quad) = tone_bias(&psi, &d, w0, cents);
                let r = pairing_residual(&psi32, &d32, wd);
                let pred = 1200.0 / LN_2 * r / (wd * h);
                let budget = SLOP * pred + MIRROR_C;

                println!(
                    "  {cents:+6.0}c  |H| {h_db:>6.1} dB  R {:>6.1} dB  \
                     pred {pred:>8.3}c  bias {bias:>8.3}c  ({:.2}x)  quad {quad:.1e}",
                    20.0 * (r / PEAK_GAIN).log10(),
                    bias / budget,
                );

                assert!(
                    bias < budget,
                    "fc {fc} detune {cents} bias {bias:.3}c over {budget:.3}c"
                );
                assert!(quad < 5e-3, "fc {fc} detune {cents} quad {quad:.3e}");
            }
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
            .with_shape(Shape::from_q(Q, 3.0))
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

    // Basically just a wavelet without the sauce
    // /// Fixed-aperture quality assurance™.  Deliberately circumvents load quantum & truncation phase
    // /// heuristics to provide a stable evaluation of wavelet shaping.
    // ///
    // /// ```text
    // /// cargo test --release wavelet::test::quality_sweep -- --nocapture
    // /// ```
    // // DEBT No assertions yet because there's almost always an edge case either very near DC or very
    // // near Nyquist.  The minimum Q kicking in at near-Nyquist values bites hard.
    // #[test]
    // fn quality_assurance() {
    //     const Q: f64 = 3.5;
    //     const GAMMA: f64 = 3.0;
    //     const EPS: f64 = 1e-14; // Using a really low grid floor.

    //     // Omegas sweeping the edge cases, 20Hz at 3kHz sample rate to 15kHz at 48kHz sample rate.
    //     // This is a fraction of the sample rate, so Nyquist is 0.5.  Multiplied by TAU to obtain
    //     // radians.
    //     //
    //     // This is a Representation of the downsample ladder.  Decimated rates are only used below
    //     // their own 1/4 band, so the 3kHz sample rate spans 20Hz up to 750Hz and the octave above
    //     // each cutoff lands against the next rate up, topping out near 15kHz of the 48kHz input, a
    //     // little over half Nyquist.
    //     const OMEGAS: [f64; 8] = [0.00667, 0.0116, 0.02, 0.035, 0.060, 0.104, 0.180, 0.312];

    //     // Emitted grid half-spans of replica clearance, a pad for TAU/(du*w0).  Putting the image
    //     // this far out leaves truncation as the only error the stop band column reports.
    //     const CLEARANCE: f64 = 4.0;

    //     // Untruncated enough that the reference t̂ is the estimator's own answer.
    //     const REF_SIGMAS: f64 = 10.0;

    //     // Measured -3 dB width times Q
    //     const WIDTH_Q: f64 = 0.998;
    //     // Negative dB usually detaches from noise floor at around 5.5.  After 5.5, f32 truncation
    //     // should be taking over.  7.5 to confirm if you're curious.
    //     const SIGMAS: [f64; 6] = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5];

    //     let shape = Shape::from_q(Q, GAMMA);
    //     let env = log_env(shape);
    //     let db = |v: f64| 20.0 * v.log10();

    //     println!("\n=== QUALITY (Q {Q}, gamma {GAMMA}, eps {EPS:e}) ===");

    //     for sigmas in SIGMAS {
    //         println!("\n  truncation {sigmas} sigmas (bound {:.1} dB)\n  {:>7} {:>6} {:>10} {:>8} {:>8} \
    //             {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}",
    //             db((-0.5 * sigmas * sigmas).exp()),
    //             "w/fs",
    //             "taps",
    //             "gain",
    //             "center",
    //             "width",
    //             "neg dB",
    //             "stop dB",
    //             "dc dB",
    //             "bias ct",
    //             "t_hat",
    //             "quad",
    //             "t leak",
    //         );

    //         for nyq in OMEGAS {
    //             let w0 = TAU * nyq;
    //             // Half-span in samples is sigmas * P / w0. Rounding to an integer tap moves
    //             // the achieved radius by up to w0/2P, which is under 0.15 sigma at the top of
    //             // the sweep, so the request and the emission agree to well under a dB.
    //             let n = (sigmas * shape.p() / w0).round() as usize;
    //             let bin = Bin {
    //                 w0,
    //                 quantum: 1,
    //                 k: n + 1,
    //                 sigmas: n as f64 * w0 / shape.p(),
    //             };

    //             // Reference half-span, past every `sigmas` in the sweep so one grid serves all.
    //             let kref = (REF_SIGMAS * shape.p() / w0).round() as usize + 1;

    //             // Grid built by hand to bypass all the heuristics.
    //             let du = snap(TAU / (CLEARANCE * w0 * kref as f64));

    //             let (lo, m) = support(shape, du, EPS);
    //             let mut spec = vec![[0.0; 2]; m];
    //             for (j, s) in spec.iter_mut().enumerate().skip(lo) {
    //                 let u = j as f64 * du;
    //                 let p = env(u).exp();
    //                 *s = [p, p * u];
    //             }
    //             let mut plan = Plan {
    //                 shape,
    //                 c: 0.0,
    //                 du,
    //                 spec,
    //                 lo,
    //                 buf: Vec::new(),
    //                 max_load_quantum: 1,
    //                 floor: 0.0,
    //             };

    //             let mut w = vec![[0.0f32; 4]; bin.k];
    //             plan.taps_into(bin, &mut w);

    //             let psi = unfold(&w, 0);
    //             let psi64 = widen(&psi);
    //             let d = widen(&unfold(&w, 2));
    //             let t = derive_t(&psi64);
    //             let (rpsi, rt) = reference(&plan, w0, kref);

    //             let r = characterize(&psi, w0);
    //             let gain = r.peak_h / PEAK_GAIN - 1.0;
    //             let cents = 1200.0 * (r.peak_w / w0).log2();
    //             let width = r.rel_width * Q;
    //             let (neg, floor) = (r.neg / r.peak_h, r.floor / r.peak_h);
    //             let dc = dc_leak(&psi, w0) / r.peak_h;

    //             let (mut bias, mut quad) = (0.0f64, 0.0f64);

    //             let half_bw = 1200.0 * (1.0 + 0.5 / Q).log2();
    //             let top = (1.5 * half_bw).min(1200.0 * (0.9 * PI / w0).log2());
    //             for detune in [-top, -0.5 * top, 0.0, 0.5 * top, top] {
    //                 let (a, q) = tone_bias(&psi64, &d, w0, detune);
    //                 bias = bias.max(a);
    //                 quad = quad.max(q);
    //             }

    //             // One burst at a tenth of the aperture, fixed across the sweep so the column
    //             // is comparable. The full profile is `t_hat_survives_transients`.
    //             let half = (psi64.len() / 2) as isize;
    //             let (t_err, t_leak) = t_hat_profile(
    //                 (&psi64, &t),
    //                 (&rpsi, &rt),
    //                 burst(w0, 0.1 * half as f64, 0.0),
    //                 2 * half,
    //             );
    //             let t_hat = t_err[2];

    //             println!(
    //                 "  {nyq:>7.4} {:>6} {gain:>+10.2e} {cents:>+8.4} {width:>8.5} \
    //                  {:>8.2} {:>8.2} {:>8.2} {bias:>8.4} {t_hat:>8.4} {quad:>9.2e} {t_leak:>9.2e}",
    //                 bin.k,
    //                 db(neg),
    //                 db(floor),
    //                 db(dc),
    //             );
    //         }
    //     }
    // }

    /// t̂ against an untruncated bake, on transients short enough that the estimator has to
    /// actually integrate the envelope. Swept from near-impulsive to comparable to the
    /// wavelet's own support, which is where the pull toward the hop takes over.
    #[test]
    fn t_hat_survives_transients() {
        const QUANTUM: usize = 4;
        const REF_TAIL_DB: f64 = -80.0;
        const TAIL_DB: f64 = -40.0;

        // Samples, per level bucket. Calibrate from the first run; these are a starting bracket.
        const TOL: [f64; 3] = [0.05; 3];

        let mut plan = spec(3.5, 1e-10).max_load_quantum(QUANTUM).plan();
        let long = spec(3.5, 1e-14)
            .truncate(REF_TAIL_DB)
            .max_load_quantum(QUANTUM)
            .plan();

        // NEXT adapt for same omegas as the quality assurance.
        for (fc, sr) in [(40.0f64, 3000.0), (200.0, 3000.0), (800.0, 3000.0)] {
            let bin = plan.bin(fc, sr, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];
            plan.taps_into(bin, &mut w);

            let psi = widen(&unfold(&w, 0));
            let t = derive_t(&psi);
            let w0 = bin.velocity();

            // The reference plan sized its own grid for this span, so the replica is clear.
            let k = long.bin(fc, sr, QUANTUM).folded_taps();
            let (rpsi, rt) = reference(&long, w0, k);

            let half = (psi.len() / 2) as isize;
            println!(
                "\n=== T_HAT fc {fc:.0} sr {sr:.0} taps {} ref {} ({:.1}x) ===",
                psi.len(),
                rpsi.len(),
                rpsi.len() as f64 / psi.len() as f64,
            );

            for frac in [0.02f64, 0.1, 0.35] {
                let sd = (frac * half as f64).max(1.0);
                for detune in [0.0f64, 400.0] {
                    let w = w0 * (detune / 1200.0).exp2();
                    let (err, leak) =
                        t_hat_profile((&psi, &t), (&rpsi, &rt), burst(w, sd, 0.0), 2 * half);

                    let pct = |v: f64| 100.0 * v / half as f64;
                    println!(
                        "  sd {sd:>7.2}  detune {detune:>4.0}c  err {:.4} / {:.4} / {:.4} \
                         ({:.3} / {:.3} / {:.3} %sup)  skew {leak:.2e}",
                        err[0],
                        err[1],
                        err[2],
                        pct(err[0]),
                        pct(err[1]),
                        pct(err[2])
                    );

                    for (e, tol) in err.iter().zip(&TOL) {
                        let frac = e / half as f64;
                        assert!(
                            frac < *tol,
                            "fc {fc} sd {sd:.2} detune {detune} t_hat gap {e:.4} samples"
                        );
                    }
                }
            }
        }
    }

    /// Rough magnitude response, about four main lobes wide, centered on the measured peak.
    ///
    #[test]
    fn print_response() {
        const QUANTUM: usize = 1;
        const ROWS: usize = 64;
        const COLS: usize = 80;
        const FLOOR_DB: f64 = -100.0;
        const LOBES: f64 = 32.0;

        // Expressed as a fraction of the sample rate (2pi).
        // const OMEGAS: [f64; 8] = [0.00667, 0.0116, 0.02, 0.035, 0.060, 0.104, 0.180, 0.312];
        // Show just one peak
        const OMEGAS: [f64; 1] = [0.180];

        let mut p = spec(12.5, 1e-10)
            .sigmas(3.5)
            .max_load_quantum(QUANTUM)
            .plan();

        for fcfs in OMEGAS {
            let w0 = TAU * fcfs;
            let bin = p.bin(fcfs, 1.0, QUANTUM);
            let mut w = vec![[0.0f32; 4]; bin.folded_taps()];

            // p.taps_into(bin, &mut w);

            let psi = unfold(&w, 0);
            let w0 = bin.velocity();
            let r = characterize(&psi, w0);

            let lobe = r.edges.1 - r.edges.0;
            let span = LOBES * lobe;
            let lo = (r.peak_w - 0.5 * span).max(-PI);
            let hi = (r.peak_w + 0.5 * span).min(PI);
            let step = (hi - lo) / ROWS as f64;

            println!(
                "\n=== RESPONSE w/fs {fcfs:.4} taps {} w0 {w0:.6} \
                 lobe {lobe:.6} ({:.5} w/fs) ===",
                psi.len(),
                lobe / TAU
            );
            println!(
                "  columns span {FLOOR_DB:.0} dB (left) to peak (right); \
                 {:.1} of {LOBES:.0} lobes in window",
                (hi - lo) / lobe
            );

            for k in 0..=ROWS {
                let w = lo + step * k as f64;
                let db = 20.0 * (dtft(&psi, w) / r.peak_h).log10();
                let cells = ((1.0 - db / FLOOR_DB) * COLS as f64)
                    .round()
                    .clamp(0.0, COLS as f64) as usize;

                let near = |t: f64| (w - t).abs() <= 0.5 * step;
                let tag = if near(r.peak_w) {
                    '<'
                } else if near(-r.peak_w) {
                    '>'
                } else if near(r.edges.0) || near(r.edges.1) {
                    '-'
                } else {
                    ' '
                };

                println!(
                    "{:>10.5} {:>9.5} {:>8.2} {tag}{}",
                    w,
                    w / TAU,
                    db,
                    "#".repeat(cells)
                );
            }
        }
    }

    #[test]
    fn print_bin() {
        const QUANTUM: usize = 4;

        let spec = Spec::default()
            .q(3.5)
            .truncate(-60.0)
            .max_load_quantum(QUANTUM);

        for (fc, sr) in [(1000.0f64, 8000.0), (300.0, 3000.0), (12_000.0, RATE)] {
            let p = spec.bin_planner(fc, sr);
            let mut w = vec![[0.0f32; 4]; p.folded_taps()];
            p.taps_into(&mut w);

            let (psi, d) = (unfold(&w, 0), unfold(&w, 2));
            let w0 = p.velocity();

            print_wave(
                &format!(
                    "BIN fc {fc:.0} sr {sr:.0} w0 {w0:.6} rho {:.6} taps {}",
                    p.rho(),
                    p.unfolded_taps()
                ),
                &psi,
                30,
            );

            let g = dtft(&psi, w0);
            let gd = dtft(&d, w0);
            let dc = psi.iter().map(|&(r, _)| r as f64).sum::<f64>();
            let neg = dtft(&psi, -w0);

            println!(
                "  peak {g:.9}  d/psi {:.9}  dc {dc:.3e}  neg {:.2} dB",
                gd / g,
                20.0 * (neg / g).log10()
            );

            assert!((g - PEAK_GAIN).abs() < 1e-6);
            assert!((gd / g - 1.0).abs() < 1e-6);
            assert!(dc.abs() < 1e-5 * g);

            let psi64 = widen(&psi);
            for m in 0..8 {
                let (re, im) = conv(&psi64, |k| (w0 * k as f64).cos(), m);
                assert!((re.hypot(im) - 1.0).abs() < 1e-3, "fc {fc} phase {m}");
            }
        }
    }
}
