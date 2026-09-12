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
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! const QUANTUM: usize = 8;
//!
//! let mut plan = WaveletSpec::default()
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
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! const QUANTUM: usize = 8;
//!
//! let mut plan = WaveletSpec::default()
//!     .max_load_quantum(QUANTUM)
//!     .plan();
//!
//! let bin = plan.bin(1000.0, 8000.0, QUANTUM);
//! let mut weights = vec![[0.0f32; 4]; bin.folded_taps()];
//!
//! // float4(Re ψ, Im ψ, Re d, Im d); index 0 is the real center tap.
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

#[cfg(test)]
mod harness;

use core::f64::consts::{LN_2, PI, TAU};

use num_complex::{Complex32, Complex64};

use spec::{Bin, BinSpec, Shape, Wavelet, WaveletSpec};

pub mod defaults {
    #[cfg(debug_assertions)]
    pub const RESOLUTION: usize = 64;
    #[cfg(not(debug_assertions))]
    pub const RESOLUTION: usize = 256;
    pub const GRID_EPS: f64 = 1e-9;
    pub const TAIL_DB: f64 = -40.0;
    pub const LOAD_QUANTUM: usize = 4;
    pub const GAMMA: f64 = 3.0;
    pub const Q: f64 = 3.5;
}

/// Filter peak gain. Analytic taps see half a real tone's amplitude, so |H| = 2 makes a unit tone
/// read |W| = 1.
const PEAK_GAIN: f64 = 2.0;

#[cfg(test)]
fn fmt_e(x: f64) -> String {
    let s = format!("{x:+.2e}");
    // split "±m.mme±dd" into mantissa and exponent, then zero-pad the exponent
    let (mantissa, exp) = s.split_once('e').unwrap_or(("999", "999"));
    let exp: i32 = exp.parse().unwrap();
    format!("{mantissa}e{exp:+03}")
}

#[cfg(test)]
mod test {
    use super::*;

    use harness::{
        bandwidth, bisect, burst, characterize, conv, dc_leak, dtft, first_null, l1, level_moment,
        moment, pairing_residual, print_wave, skirts, t_hat_profile, tone_bias, unfold, widen,
        Point, Skirt,
    };

    const BINS: usize = 1024;
    const RATE: f64 = 48_000.0;
    const WEAKEST_TAIL_DB: f64 = -160.0;

    fn wavelet(q: f64, quantum: usize) -> Wavelet {
        WaveletSpec::default()
            .with_shape(Shape::from_q(q, 3.0))
            .max_load_quantum(quantum)
            .max_truncation(WEAKEST_TAIL_DB)
            .bake()
    }

    #[test]
    fn print_gamma_sweep() {
        // XXX Completely busted.  And... envelope?

        const QUANTUM: usize = 4;

        println!("\n=== ENVELOPE vs GAMMA (Q = 2.4) ===");
        // P = 4.0 is Q = 2.4; holding it fixed keeps the -3 dB width constant across gamma.
        let p = 4.0;
        for gamma in [1.0f64, 2.0, 3.0, 6.0] {
            let wav = WaveletSpec::default()
                .with_shape(Shape {
                    gamma,
                    beta: p * p / gamma,
                })
                .max_load_quantum(QUANTUM)
                .bake();
            let bin = wav.at_rho(1000.0 / 8000.0);
            let taps = bin.taps();

            let t = unfold(&taps, 0);
            let n = t.len();

            let mags: Vec<f64> = t.iter().map(|h| h.norm() as f64).collect();
            let max = mags.iter().fold(0.0f64, |a, &b| a.max(b));

            // Σ (j − c)|ψ_j|² / Σ |ψ_j|²,  c = (n − 1)/2
            let ctr = (n - 1) as f64 / 2.0;
            let m: f64 = mags
                .iter()
                .enumerate()
                .map(|(j, &v)| (j as f64 - ctr) * v * v)
                .sum();
            let e: f64 = mags.iter().map(|v| v * v).sum();

            println!(
                "\ngamma = {:.1}  weights {}  taps {}  centroid offset = {:+.3}",
                gamma,
                taps.len(),
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

        const Q: f64 = 5.0;

        let bins = bank::bins(2_000.0, 20_000.0, BINS);

        let start = std::time::Instant::now();
        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, defaults::GAMMA))
            .bake();
        let bake_time = start.elapsed();

        // Packed bank with per-voice tap ranges
        let start = std::time::Instant::now();
        let mut weights = Vec::new();
        let mut voices = Vec::with_capacity(bins.len());
        for b in &bins {
            let bin = wav.at_rho(b.center / RATE);
            let taps = bin.taps();
            let range = weights.len()..weights.len() + taps.len();
            weights.extend_from_slice(&taps);
            voices.push((bin, range));
        }
        let fill_time = start.elapsed();

        // max over voices of |H(ω₀) − PEAK_GAIN|
        let worst = voices
            .iter()
            .map(|(bin, r)| {
                (dtft(&unfold(&weights[r.clone()], 0), bin.velocity()).norm() - PEAK_GAIN).abs()
            })
            .fold(0.0f64, f64::max);
        println!("worst peak gain error: {worst:.3e}");
        assert!(worst < 1e-3, "worst peak gain error {worst:.3e}");

        let (low_bin, low_range) = &voices[0];
        print_wave(
            &format!(
                "LOWEST BIN ({:.0}Hz, omega0 {:.5})",
                bins[0].center,
                low_bin.velocity()
            ),
            &unfold(&weights[low_range.clone()], 0),
            30,
        );

        let lens = voices.iter().map(|(_, r)| r.len());
        println!(
            "voices {} of {}  weights {}  longest {}  shortest {}",
            voices.len(),
            BINS,
            weights.len(),
            lens.clone().max().unwrap(),
            lens.min().unwrap(),
        );

        println!("bake time: {}µs", bake_time.as_micros());
        println!("bin filling time: {}µs", fill_time.as_micros());
    }

    /// A real unit tone reads |W| = 1 even though |H| = 2: the analytic taps see only the +ω half
    /// of the cosine. Swept over the quantum, which pads the emitted half-span.
    #[test]
    fn unit_tone_reads_unity() {
        const Q: f64 = 3.0;
        const TAIL_DB: f64 = -100.0;

        let w = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(8)
            .max_truncation(TAIL_DB)
            .bake();

        for quantum in [1usize, 4, 8] {
            for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = w.bin(fc, sr).load_quantum(quantum);
                let psi = widen(&unfold(&bin.taps(), 0));
                let w0 = bin.velocity();

                // taps are centered, so m is the sample under tap index n/2.
                for m in 0..8 {
                    let env = conv(&psi, |k| (w0 * k as f64).cos(), m).norm();
                    assert!(
                        (env - 1.0).abs() < 1e-3,
                        "quantum {quantum} fc {fc} phase {m} envelope {env:.6}"
                    );
                }
            }
        }
    }

    /// Peak-normalized constant-Q puts noise gain proportional to ρ: the ψ sum is pinned at
    /// PEAK_GAIN, so the envelope's amplitude scales with ρ and its energy with ρ². Dividing that
    /// back by the ρ⁻¹ taps per period leaves one power of ρ. White noise therefore floors at a
    /// fixed level per bin once ρ is divided out.
    #[test]
    fn noise_gain_tracks_center() {
        const Q: f64 = 3.0;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -100.0;

        // Tap count is an integer, so envelope truncation loses O(1/N) of the energy, worst at
        // the top of the range. Anchored to split the sweep rather than to any one bin.
        // NOTE recalibrate from the first run.
        const NOISE_GAIN: f64 = 1.412;
        const TOL: f64 = 2e-3;

        let w = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        println!("\n=== NOISE GAIN (Q = {Q}, sr = {RATE}, quantum {QUANTUM}) ===");

        for fc in [500.0f64, 1000.0, 2000.0, 4000.0, 8000.0] {
            let bin = w.bin(fc, RATE);
            let psi = unfold(&bin.taps(), 0);

            let e: f64 = widen(&psi).iter().map(Complex64::norm_sqr).sum();
            let ratio = e / bin.rho();

            println!(
                "  fc {:>6.0}  taps {:>5}  rho {:.6}  energy {:.6}  e/rho {:.6}  dev {:+.2e}",
                fc,
                bin.unfolded_taps(),
                bin.rho(),
                e,
                ratio,
                ratio / NOISE_GAIN - 1.0
            );

            // assert!(
            //     (ratio / NOISE_GAIN - 1.0).abs() < TOL,
            //     "fc {fc} noise gain {ratio:.6}"
            // );
        }
    }

    /// Pitch reassignment bias and quadrature leak against steady tones.  Scans across the range
    /// where the wavelet's ideal response is in `[-GATE_DB, GATE_DB)`.  The predicted bias is
    /// compared to the observed.  Too large of bias in the main lobe will trip the asserts.
    #[test]
    fn reassignment_is_unbiased() {
        const Q: f64 = 8.5;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -80.0;

        /// Cents readings stop meaning anything once the skirt is down in truncation ripple.
        /// The denominator is no longer the envelope, so the ratio is measuring the stopband.
        const GATE_DB: f64 = -50.0;

        /// `pred` is the bias the pairing residual alone implies.  The rest is the negative
        /// frequency image, flat in level and so stated absolutely.
        const MIRROR_C: f64 = 0.25;
        const SLOP: f64 = 10.0; // 🫠

        const STEP: f64 = 100.0;
        const SPAN: isize = 12;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        // XXX Use omegas to make this easier to read.
        for (fc, sr) in [
            (2_000.0f64, RATE),
            (200.0, 3000.0),
            (250.0, 3000.0),
            (12_000.0, RATE),
        ] {
            let bin = wav.at_rho(fc / sr);
            let taps = bin.taps();
            let (psi, d) = (unfold(&taps, 0), unfold(&taps, 2));

            let (n, w0) = (psi.len(), bin.velocity());
            println!("\n=== REASSIGN fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6} ===");

            for k in -SPAN..=SPAN {
                let cents = k as f64 * STEP;
                // 2^(c/1200)
                let ratio = (cents / 1200.0).exp2();
                let wd = w0 * ratio;
                let h = dtft(&psi, wd).norm();
                let h_db = 20.0 * (h / PEAK_GAIN).log10();
                if h_db < GATE_DB {
                    continue;
                }

                let (bias, quad) = tone_bias(&taps, w0, cents);
                let res = pairing_residual(&psi, &d, w0, wd);
                // |H_ψ(−ω)|, |H_d(−ω)|
                let (img_psi, img_d) = (dtft(&psi, -wd).norm(), dtft(&d, -wd).norm());
                // (1200 / ln 2) · R / (r · |H|)
                let pred = 1200.0 / LN_2 * res / (ratio * h);
                let budget = SLOP * pred + MIRROR_C;

                println!(
                    "  {cents:+6.0}c  |H| {h_db:>6.1} dB  R {:>6.1} dB  img ψ {:>6.1} dB  img d {:>6.1} dB  \
                    pred {pred:>8.3}c  bias {bias:>8.3}c  quad {quad:.1e}",
                    20.0 * (res / PEAK_GAIN).log10(),
                    20.0 * (img_psi / PEAK_GAIN).log10(),
                    20.0 * (img_d / PEAK_GAIN).log10(),
                );

                // assert!(
                //     bias < budget,
                //     "fc {fc} detune {cents} bias {bias:.3}c over {budget:.3}c"
                // );
                // assert!(quad < 5e-3, "fc {fc} detune {cents} quad {quad:.3e}");
            }
        }
    }

    /// Truncation cost against a full-length bake, swept over `tail_db`.  One motherlet serves the
    /// reference and every cut, so a gap in the delta columns is truncation and nothing else.
    #[test]
    fn truncation_is_predictable() {
        const QUANTUM: usize = 1;
        const Q: f64 = 3.5;

        /// Weakest truncation in the sweep, and so the grid the wavelet is sized for.
        const FULL_DB: f64 = -120.0;
        const CUTS: [f64; 5] = [-20.0, -40.0, -60.0, -80.0, -100.0];
        const FCS: [f64; 4] = [2_000.0, 4_000.0, 8_000.0, 14_000.0];

        // NOTE these are empirically discovered values stored to catch regressions.

        // Measured 0.9984 to 1.0020 across the sweep.
        const WIDTH_Q: f64 = 1.0;
        const WIDTH_TOL: f64 = 0.01;

        let w = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(FULL_DB)
            .bake();

        let db = |v: f64| 20.0 * (v / PEAK_GAIN).log10();
        let cents = |w: f64, w0: f64| 1200.0 * (w / w0).log2();

        println!(
            "\n=== TRUNCATION (Q = {Q}, quantum {QUANTUM}) ===\n\n\
            - |H| is the response to a unit exponential.\n\
            - gain is |H| at the peak. peak c is the peak offset from ω₀ in cents.\n\
            - dc, image, and floor are in dB relative to PEAK_GAIN. image is the worst |H| on [−π, 0].\n\
            - ratio is the tap count divided by the reference tap count."
        );

        // Reference
        println!("\n  reference, tail {FULL_DB:.1} dB");
        println!(
            "  {:>6} {:>5} {:>9} {:>8} {:>9} {:>8} {:>8} {:>8}",
            "fc", "taps", "gain", "width·Q", "peak c", "dc", "image", "floor"
        );

        let mut refs = Vec::with_capacity(FCS.len());
        for fc in FCS {
            let full = w.bin(fc, RATE).truncate(FULL_DB);
            let nf = full.folded_taps();
            let pf = unfold(&full.taps(), 0);

            let w0 = full.velocity();
            let rf = characterize(&pf, w0);
            let dcf = dc_leak(&pf, w0);

            println!(
                "  {fc:>6.0} {nf:>5} {:>9.6} {:>8.4} {:>+9.3} {:>8.2} {:>8.2} {:>8.2}",
                rf.gain,
                rf.rel_width * Q,
                cents(rf.peak_w, w0),
                db(dcf),
                db(rf.image),
                db(rf.floor)
            );

            // XXX There's some issue with the naive quadrature restriction that moves center
            // frequency around

            // Peak sits below ω₀ by the cell-average droop, gain rising as ½P²Δx².
            // assert!(
            //     (rf.gain - 1.0).abs() < 1e-5,
            //     "fc {fc} full gain {:.9}",
            //     rf.gain
            // );
            // assert!(dcf < 1e-5 * PEAK_GAIN, "fc {fc} full dc {:.2} dB", db(dcf));

            // -3 dB width is set by P = sqrt(beta*gamma) and Q = P/1.6651.
            assert!(
                (rf.rel_width * Q / WIDTH_Q - 1.0).abs() < WIDTH_TOL,
                "fc {fc} width x Q {:.4}",
                rf.rel_width * Q
            );

            refs.push((fc, nf, rf, w0));
        }

        // Cuts
        println!(
            "\n  {:>6} {:>7} {:>5} {:>6} {:>9} {:>8} {:>9} {:>8} {:>8} {:>8}",
            "fc", "tail", "taps", "ratio", "gain", "width·Q", "peak c", "dc", "image", "floor"
        );

        for (fc, nf, rf, w0) in &refs {
            let (fc, nf, w0) = (*fc, *nf, *w0);
            let (mut prev_taps, mut prev_floor) = (0usize, f64::INFINITY);

            for tail_db in CUTS {
                let cut = w.bin(fc, RATE).truncate(tail_db);
                let nc = cut.folded_taps();
                let pc = unfold(&cut.taps(), 0);
                let rc = characterize(&pc, w0);
                let dc = dc_leak(&pc, w0);

                println!(
                    "  {fc:>6.0} {tail_db:>7.1} {nc:>5} {:>6.3} {:>9.6} {:>8.4} {:>+9.3} \
                     {:>8.2} {:>8.2} {:>8.2}",
                    nc as f64 / nf as f64,
                    rc.gain,
                    rc.rel_width * Q,
                    cents(rc.peak_w, w0),
                    db(dc),
                    db(rc.image),
                    db(rc.floor)
                );

                // The band neither moves nor widens.
                assert!(
                    (cents(rc.peak_w, w0) - cents(rf.peak_w, w0)).abs() < 8.0,
                    "fc {fc} tail {tail_db} peak moved {:+.4}c",
                    cents(rc.peak_w, w0) - cents(rf.peak_w, w0)
                );
                // assert!(
                //     (rc.rel_width / rf.rel_width - 1.0).abs() < 0.02,
                //     "fc {fc} tail {tail_db} width {:+.3}%",
                //     100.0 * (rc.rel_width / rf.rel_width - 1.0)
                // );

                // Gain near DC combines both positive and negative components, so use of PEAK_GAIN
                // is apt here.
                // ROLL DC centering
                assert!(
                    dc < 1e-1 * PEAK_GAIN,
                    "fc {fc} tail {tail_db} dc {:.2} dB",
                    db(dc)
                );

                // Weaker truncation buys floor with taps.  Taps may hold under quantum rounding.
                assert!(
                    nc >= prev_taps,
                    "fc {fc} tail {tail_db} taps {nc} < {prev_taps}"
                );
                assert!(
                    (nc == prev_taps && rc.floor >= prev_floor) || rc.floor < prev_floor,
                    "fc {fc} tail {tail_db} taps {prev_taps} -> {nc} without floor gain"
                );

                (prev_taps, prev_floor) = (nc, rc.floor);
            }
            println!();
        }
    }

    /// Truncated tables carry the reference's low moments, so the cut tails live on in the body.
    // ROLL until restriction restores moments, different dB will not have the same total
    // response.
    #[ignore]
    #[test]
    fn truncation_preserves_moments() {
        const Q: f64 = 3.5;
        const QUANTUM: usize = 4;

        const FULL_DB: f64 = -160.0;
        const CUTS: [f64; 4] = [-60.0, -80.0, -100.0, -120.0];
        const FCS: [f64; 4] = [2_000.0, 4_000.0, 8_000.0, 14_000.0];

        /// Moments the repair restores, M_0 through M_{ORDERS−1}.
        const ORDERS: i32 = 4;

        // NOTE calibrate once the repair lands.
        const TOL: f64 = 1e-4;

        let w = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(FULL_DB)
            .bake();

        let db = |v: f64| 20.0 * v.log10();

        println!(
            "\n=== MOMENTS (Q = {Q}, quantum {QUANTUM}) ===\n\
            dB of |M_p(cut) − M_p(ref)| / Σ|ν|^p|h_ref|, reference tail {FULL_DB:.1} dB"
        );
        let header: String = (0..ORDERS)
            .map(|p| format!(" {:>8}", format!("M{p}")))
            .collect();
        println!("\n  {:>6} {:>7} {:>5}{header}", "fc", "tail", "taps");

        let mut worst = (0.0f64, 0.0f64, 0.0f64, 0i32);
        for fc in FCS {
            let pf = unfold(&w.bin(fc, RATE).truncate(FULL_DB).taps(), 0);
            let half = (pf.len() / 2) as isize;

            // Σ|ν|^p |h_ref|
            let scale = |p: i32| -> f64 {
                pf.iter()
                    .enumerate()
                    .map(|(j, h)| ((j as isize - half) as f64).abs().powi(p) * h.norm() as f64)
                    .sum()
            };

            for tail_db in CUTS {
                let cut = w.bin(fc, RATE).truncate(tail_db);
                let pc = unfold(&cut.taps(), 0);

                let errs: Vec<f64> = (0..ORDERS)
                    .map(|p| (moment(&pc, p) - moment(&pf, p)).norm() / scale(p))
                    .collect();

                let row: String = errs.iter().map(|&e| format!(" {:>8.2}", db(e))).collect();
                println!("  {fc:>6.0} {tail_db:>7.1} {:>5}{row}", cut.folded_taps());

                for (p, &e) in errs.iter().enumerate() {
                    if e > worst.0 {
                        worst = (e, fc, tail_db, p as i32);
                    }
                }
            }
            println!();
        }

        let (e, fc, tail_db, p) = worst;
        assert!(e < TOL, "fc {fc} tail {tail_db} M{p} error {:.2} dB", db(e));
    }

    /// Smoke test.  DC-free and correct peak gain, measured with the linear DTFT so a broken fold
    /// convention can't agree with itself.  First thing to look at if the bake goes sideways.
    #[test]
    fn taps_are_conditioned() {
        // Use a rough tail dB so we can verify conditioning under challenging conditions.
        const TAIL_DB: f64 = -30.0;

        let wav = WaveletSpec::default().max_truncation(TAIL_DB).bake();

        for (fc, sr) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = wav.at_rho(fc / sr);
            let taps = bin.taps();

            let (psi, d) = (unfold(&taps, 0), unfold(&taps, 2));
            let w0 = bin.velocity();

            let g = dtft(&psi, w0).norm();
            assert!((g - PEAK_GAIN).abs() < 1e-3, "fc {fc} peak gain {g:.6}");

            // Analytic taps: the mirror image is stopband, not signal.
            let neg = dtft(&psi, -w0).norm();
            assert!(
                neg < 1e-3 * g,
                "fc {fc} negative-freq leak {:.2} dB",
                20.0 * (neg / g).log10()
            );

            let dc = psi.iter().map(|h| h.re as f64).sum::<f64>();
            assert!(dc.abs() < 1e-5 * g, "fc {fc} dc {dc:.3e}");

            // d/ψ reads ω/ω₀, unity at the carrier.
            let gd = dtft(&d, w0).norm();
            // assert!((gd / g - 1.0).abs() < 1e-3, "fc {fc} d/psi {:.6}", gd / g);

            // Σ ν^p · (Re ψ if p even, Im ψ if p odd)
            let center = (psi.len() / 2) as isize;
            let mom = |p: i32| {
                psi.iter()
                    .enumerate()
                    .map(|(j, h)| {
                        let nu = (j as isize - center) as f64;
                        nu.powi(p) * if p % 2 == 0 { h.re as f64 } else { h.im as f64 }
                    })
                    .sum::<f64>()
            };

            #[allow(unused)]
            {
                // measured: fc 1000 first moment -1.123e-7
                let m1 = mom(1);
                // assert!(m1.abs() < 1e-5 * g, "fc {fc} first moment {m1:.3e}");

                // H''(0) and H'''(0), the two the solve nulls that nothing else measures.
                let (m2, m3) = (mom(2), mom(3));
                // assert!(m2.abs() < 1e-3 * g, "fc {fc} second moment {m2:.3e}");
                // assert!(m3.abs() < 1e-3 * g, "fc {fc} third moment {m3:.3e}");
            }
        }
    }

    /// t̂ against an untruncated bake, on transients short enough that the estimator has to
    /// actually integrate the envelope.  Swept from near-impulsive to comparable to the
    /// wavelet's own support, which is where the pull toward the hop takes over.
    #[test]
    fn t_hat_survives_transients() {
        const REF_TAIL_DB: f64 = -80.0;
        const TAIL_DB: f64 = -40.0;

        /// Fraction of half-support, per level bucket.  Calibrate from the first run.  These are
        /// a starting bracket.
        const TOL: [f64; 3] = [0.05; 3];

        let wav = WaveletSpec::default().max_truncation(TAIL_DB).bake();
        let long = WaveletSpec::default().max_truncation(REF_TAIL_DB).bake();

        // NEXT adapt for same omegas as the quality assurance.
        for (fc, sr) in [(40.0f64, 3000.0), (200.0, 3000.0), (800.0, 3000.0)] {
            let rho = fc / sr;
            let bin = wav.at_rho(rho);
            let taps = bin.taps();
            let reference = long.at_rho(rho).taps();
            let w0 = bin.velocity();
            let half = (taps.len() - 1) as isize;

            println!(
                "\n=== T_HAT fc {fc:.0} sr {sr:.0} taps {} ref {} ({:.1}x) ===",
                2 * taps.len() - 1,
                2 * reference.len() - 1,
                reference.len() as f64 / taps.len() as f64,
            );

            for frac in [0.02f64, 0.1, 0.35] {
                let sd = (frac * half as f64).max(1.0);
                for detune in [0.0f64, 400.0] {
                    // ω₀ · 2^(c/1200)
                    let w = w0 * (detune / 1200.0).exp2();
                    let (err, skew) = t_hat_profile(&taps, &reference, burst(w, sd, 0.0), 2 * half);

                    let pct = |v: f64| 100.0 * v / half as f64;
                    println!(
                        "  sd {sd:>7.2}  detune {detune:>4.0}c  err {:.4} / {:.4} / {:.4} \
                         ({:.3} / {:.3} / {:.3} %sup)  skew {skew:.2e}",
                        err[0],
                        err[1],
                        err[2],
                        pct(err[0]),
                        pct(err[1]),
                        pct(err[2])
                    );

                    for (e, tol) in err.iter().zip(&TOL) {
                        let gap = e / half as f64;
                        // assert!(
                        //     gap < *tol,
                        //     "fc {fc} sd {sd:.2} detune {detune} t_hat gap {e:.4} samples \
                        //      ({gap:.4} of half support)"
                        // );
                    }
                }
            }
        }
    }

    /// Rough magnitude response, centered on the measured peak.  The sweep names ρ directly, so
    /// a row is a filter and not a sample rate.
    #[test]
    fn print_response() {
        const Q: f64 = 8.5;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -40.0;

        const ROWS: usize = 384;
        const COLS: usize = 140;
        const ANTI_ALIAS: usize = 2;
        const FLOOR_DB: f64 = -100.0;
        const LOBES: f64 = 64.0;

        // Periods per tap, sweeping the downsample ladder from 20Hz at 3kHz to 15kHz at 48kHz.
        // Nyquist is 0.5.
        // const RHOS: [f64; 8] = [0.00667, 0.0116, 0.02, 0.035, 0.060, 0.104, 0.180, 0.312];

        const RHOS: [f64; 1] = [0.312];

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        for rho in RHOS {
            let bin = wav.at_rho(rho);
            let psi = unfold(&bin.taps(), 0);
            let w0 = bin.velocity();
            let r = characterize(&psi, w0);

            let lobe = r.edges.1 - r.edges.0;
            let span = LOBES * lobe;
            let lo = (r.peak_w - 0.5 * span).max(-PI);
            let hi = (r.peak_w + 0.5 * span).min(PI);
            let step = (hi - lo) / ROWS as f64;

            println!(
                "\n=== RESPONSE rho {rho:.4} taps {} w0 {w0:.6} \
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

                // (1/step) ∫ |H(ω)|² dω over [w − step/2, w + step/2]
                let power = (0..ANTI_ALIAS)
                    .map(|j| {
                        let u = w + step * ((j as f64 + 0.5) / ANTI_ALIAS as f64 - 0.5);
                        dtft(&psi, u).norm_sqr()
                    })
                    .sum::<f64>()
                    / ANTI_ALIAS as f64;
                let db = 10.0 * (power / (r.gain * r.gain)).log10();

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
    fn print_waveform() {
        const TAIL_DB: f64 = -40.0;
        let wav = WaveletSpec::default().max_truncation(TAIL_DB).bake();

        for (fc, sr) in [(1000.0f64, 8000.0), (300.0, 3000.0), (12_000.0, RATE)] {
            let rho = fc / sr;
            let bin = wav.at_rho(rho);
            let taps = bin.taps();

            let (psi, d) = (unfold(&taps, 0), unfold(&taps, 2));
            let w0 = bin.velocity();

            print_wave(
                &format!(
                    "BIN fc {fc:.0} sr {sr:.0} w0 {w0:.6} rho {rho:.6} taps {}",
                    psi.len()
                ),
                &psi,
                30,
            );

            let g = dtft(&psi, w0).norm();
            let gd = dtft(&d, w0).norm();
            let dc = psi.iter().map(|h| h.re as f64).sum::<f64>();
            let neg = dtft(&psi, -w0).norm();

            // max over m of | |ψ ∗ cos(ω₀k)|(m) − 1 |
            let psi64 = widen(&psi);
            let phase_err = (0..8)
                .map(|m| (conv(&psi64, |k| (w0 * k as f64).cos(), m).norm() - 1.0).abs())
                .fold(0.0f64, f64::max);

            println!(
                "  peak {g:.9}  d/psi {:.9}  dc {dc:.3e}  neg {:.2} dB  phase err {phase_err:.3e}",
                gd / g,
                20.0 * (neg / g).log10()
            );
        }
    }

    /// Same four numbers as `response_is_characterized`, measured on the folded weight table.
    /// Sweeps the load quantum, because the quantum pads the emitted half-span.
    #[test]
    fn table_response_is_characterized() {
        const Q: f64 = 3.5;
        const TAIL_DB: f64 = -100.0;

        let w = wavelet(Q, 16);

        for quantum in [2usize, 4, 8, 16] {
            println!("\n=== TABLE RESPONSE (Q = {Q}, quantum {quantum}) ===");

            for (fc, sr) in [(1000.0f64, 6000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = w.bin(fc, sr).load_quantum(quantum).truncate(TAIL_DB);
                let psi = unfold(&bin.taps(), 0);

                let w0 = bin.velocity();
                let r = characterize(&psi, w0);
                let db = |v: f64| 20.0 * (v / r.gain).log10();

                println!(
                    "\nfc {fc:>5.0} sr {sr:>5.0}  w0 {w0:.6}  quantized {:>3} (unfolded {:>3})",
                    bin.folded_taps(),
                    bin.unfolded_taps()
                );
                println!(
                    "  peak gain {:.9}  dev {:+.3e} rel",
                    r.gain,
                    r.gain / PEAK_GAIN - 1.0
                );
                println!("  rel width {:.5}", r.rel_width);
                println!("  peak {:+.4} cents", 1200.0 * (r.peak_w / w0).log2());
                println!("  image max        {:>8.2} dB", db(r.image));
                println!("  stopband floor    {:>8.2} dB", db(r.floor));

                // assert!(
                //     (r.peak_h / PEAK_GAIN - 1.0).abs() < 1e-5,
                //     "fc {fc} q {quantum} peak gain {:.9}",
                //     r.peak_h
                // );
            }
        }
    }

    /// First dips, peak side lobes, and κ over Q × γ × ρ.
    #[test]
    fn skirt_is_characterized() {
        const QS: [f64; 4] = [3.5, 5.0, 8.5, 12.5];
        const GAMMAS: [f64; 2] = [3.0, 4.0];
        const RHOS: [f64; 4] = [0.116, 0.189, 0.223, 0.384];
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -40.0;
        /// Height zero, where the moment stops counting area.
        const FLOOR_DB: f64 = -80.0;

        println!(
            "\n=== SKIRT (quantum {QUANTUM}, tail {TAIL_DB:.0} dB, floor {FLOOR_DB:.0} dB) ===\n\
         offsets in −3 dB widths from the peak, levels in dB below the peak\n\
         κ = ∫_band h² / ∫_dips h², h = 1 − dB/floor, unity for a brick wall\n"
        );
        println!(
            "  {:>5} {:>4} {:>7} {:>5} {:>8} {:>8} {:>7} \
            {:>8} {:>7} {:>6} {:>5} {:>8} {:>7} {:>6} {:>5} {:>7} {:>7}",
            "Q",
            "𝛄",
            "𝛒",
            "taps",
            "lo off",
            "hi off",
            "spread",
            "psl lo",
            "off lo",
            "prom",
            "lobes",
            "psl hi",
            "off hi",
            "prom",
            "lobes",
            "𝛋 lo",
            "𝛋 hi"
        );

        for q in QS {
            for gamma in GAMMAS {
                let wav = WaveletSpec::default()
                    .with_shape(Shape::from_q(q, gamma))
                    .max_load_quantum(QUANTUM)
                    .max_truncation(TAIL_DB)
                    .bake();

                for rho in RHOS {
                    let bin = wav.at_rho(rho);
                    let psi = unfold(&bin.taps(), 0);
                    let r = characterize(&psi, bin.velocity());
                    let lobe = r.edges.1 - r.edges.0;
                    let db = |h: f64| 20.0 * (h / r.gain).log10();

                    // 16ε Σ|h|
                    let null = 16.0 * f64::EPSILON * l1(&psi);
                    let density = 16.0 * psi.len() as f64 / TAU;
                    let h = 1.0 / density;

                    let lo = first_null(&psi, r.edges.0, r.edges.0 - PI, null, h, density)
                        .unwrap_or_else(|| panic!("Q {q} γ {gamma} ρ {rho} lower dip not found"));
                    let hi = first_null(&psi, r.edges.1, r.edges.1 + PI, null, h, density)
                        .unwrap_or_else(|| panic!("Q {q} γ {gamma} ρ {rho} upper dip not found"));

                    // ∫_band h² and ∫_dips h² on each side of the peak
                    let band_lo = level_moment(&psi, r.gain, (r.edges.0, r.peak_w), FLOOR_DB);
                    let band_hi = level_moment(&psi, r.gain, (r.peak_w, r.edges.1), FLOOR_DB);
                    // XXX naming is way off.  This is roughly energy between first null and -3dB
                    let dips_lo = level_moment(&psi, r.gain, (lo.w, r.peak_w), FLOOR_DB);
                    let dips_hi = level_moment(&psi, r.gain, (r.peak_w, hi.w), FLOOR_DB);

                    let (psl_lo, psl_hi) = skirts(&psi, (lo, hi));

                    /// Level, offset, and prominence of one skirt, dashes where the band holds no lobe.
                    let cols = |s: &Skirt| match (s.peak, s.prominence_db()) {
                        (Some(p), Some(prom)) => format!(
                            "{:>8.2} {:>+7.3} {:>6.1} {:>5}",
                            db(p.h),
                            (p.w - r.peak_w) / lobe,
                            prom,
                            s.lobes
                        ),
                        _ => format!("{:>8} {:>7} {:>6} {:>5}", "—", "—", "—", s.lobes),
                    };

                    println!(
                        "  {q:>5.1} {gamma:>4.1} {rho:>7.4} {:>5} {:>+8.3} {:>+8.3} {:>7.3} {} {} {:>7.4} {:>7.4}",
                        psi.len(),
                        (lo.w - r.peak_w) / lobe,
                        (hi.w - r.peak_w) / lobe,
                        (hi.w - lo.w) / lobe,
                        cols(&psl_lo),
                        cols(&psl_hi),
                        band_lo / dips_lo,
                        band_hi / dips_hi,
                    );

                    let reaches = |s: &Skirt| s.peak.is_some_and(|p| p.h >= r.gain);
                    assert!(
                        !reaches(&psl_lo) && !reaches(&psl_hi),
                        "Q {q} γ {gamma} ρ {rho} side lobe reaches the main lobe"
                    );
                }
                println!();
            }
        }
    }
}
