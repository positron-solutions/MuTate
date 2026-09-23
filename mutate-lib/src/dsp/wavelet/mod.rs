// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # The Wavelet
//!
//! > The traveler who fears the unknown road will eventually learn that known roads return
//! > to where they began.
//! >
//! > - Anthony L. Ray
//!
//! ```text
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
//! ```
//!
//! This module generates families of wavelets for use in wavelet tables.  Our wavelets are Morse
//! family:
//!
//! - Easy to generate (reference IFFT method and high-precision
//!   [`QuadJet`](generate::quadjet::QuadJet) implementations available).
//! - Fully analytic, one-sided response, making it nice for time and frequency reassignment
//! - Very well studied, providing many closed-form analytic expressions that ease writing solvers
//!   or estimating tap lengths necessary to achieve performance goals.
//! - Simple parameterization of time vs pitch precision tradeoffs.
//!
//! ## Usage
//!
//! Define a wavelet family with [`WaveletSpec`] and bake it into a [`Wavelet`].  Spec settings are
//! maxima, and the bake is sized to serve every bin under them.
//!
//! ```
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! let wavelet = WaveletSpec::default()
//!     .q(5.5)
//!     .max_truncation(-140.0)
//!     .bake();
//! ```
//!
//! Resolve a bin with [`Wavelet::bin`] and realize its folded taps.  Bins may use a tighter load
//! quantum but may not exceed the [`Wavelet`].
//!
//! ```
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! # let wavelet = WaveletSpec::default().q(5.5).max_truncation(-140.0).bake();
//! let bin = wavelet.bin(440.0, 48_000.0).with_truncation(-100.0);
//!
//! // The `Bin` can be used to calculate allocation sizes.
//! let mut taps = vec![[0.0f32; 4]; bin.len_folded()];
//! bin.taps_into(&mut taps);
//!
//! // float4(Re ψ, Im ψ, Re d, Im d); index 0 is the real center tap.
//! assert!(taps[0][1] == 0.0 && taps[0][3] == 0.0);
//! ```
//!
//! ## Customizing the Wavelet Family
//!
//! The primary knobs are [`WaveletSpec::max_truncation`] and [`WaveletSpec::with_shape`], which modulate
//! filter length, bandwidth, stop band, and skirt depth.  All are inherently coupled.  At a fixed
//! filter length, raising Q while truncating harder (smaller magnitude of tail dB) trades time
//! precision for pitch precision for likely a bit of noise floor.
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
//! - The extra weights are used to shape truncation less aggressively, increasing precision and
//!   hopefully de-correlating some locally biased response to transients, diffusing artifacts.
//!
//! Setting [`max_load_quantum`] tells the motherlet (mother wavelet) to bake a longer grid so that
//! if a wavelet's length is rounded up, there is enough grid to fill the `N` taps.
//!
//! ```
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! # let wavelet = WaveletSpec::default()
//! #     .q(5.5)
//! #     .max_truncation(-140.0)
//! #     .max_load_quantum(32)
//! #     .bake();
//! let bin = wavelet.bin(240.0, 3_000.0)
//!     .with_load_quantum(32);
//!
//! let mut taps = vec![[0.0f32; 4]; bin.len_folded()];
//! let wrote = bin.taps_into(&mut taps);
//!
//! // mirrored half lands on the quantum, center tap excluded
//! assert!((wrote - 1) % 32 == 0);
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
//! The resulting table is `(N + 1)/2` entries of four floats, laid out as a Slang `float4` in the
//! order `(Re ψ, Im ψ, Re d, Im d)`.  The center tap is real in both channels, `(Re ψ₀, 0, Re d₀,
//! 0)`, and contributes nothing to `t`.
//!
//! ## Generation Pipeline
//!
//! Low quality wavelets can cause many bad things that good things cannot fix.  A principled
//! approach upstream is required.  We can divide the strategy into five phases:
//!
//! - Define coherent goals, such as `load_quantum` and `Q`, obtaining a [`WaveletSpec`].
//! - Generate high resolution mother wavelet sufficient to fill any tap count that a [`BinSpec`]
//!   may need.
//! - Dilate and truncate the mother wavelet to the length required to fill `N` taps with a
//!   [Restriction](restrict::Restriction) operator (fancy downsampling) squeezing the high
//!   resolution mother wavelet into the coarse `N` taps.
//! - [Taper](restrict::Taper) the truncation to lessen the spectral damage of the cliff and obtain
//!   a better starting point for refinement.
//! - [Refine](refile::Refinement) the final result to restore certain analytic properties such as
//!   moments and peak curvature while taking liberties to clean up spectral artifacts and enhance
//!   transient response where possible.
//!
//! Everything before refinement is almost purely driven by time-domain geometry.  Refinement itself
//! is the first spectrally aware step.  The derivative generation is downstream of all work on `Ψ`.
//! The time derivative is re-generated from the other taps on the fly and therefore only concretely
//! exists at the point of use.
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
//! | symbol | variable | object | units |
//! |---|---|---|---|
//! | `u` || periods from the wavelet's center, a real | carrier periods |
//! | `ρ` | `rho` | the conversion ratio, a linear density | periods per tap |
//!
//! `resolution` is the number of grid points per period `u` and decides how finely each period of
//! the mother wavelet will be resolved before restriction to `N` taps.
//!
//! `ρ` was intended to only be used when converting `u` periods to grid or sample points, but the
//! same `ρ` in context turns out to be identical to `fc/fs` and a convenient way to write `ω` from
//! `(0.0,0.5)` where `ρ = 0.5` is a function of the Nyquist, no `2π` in sight.
//!
//! There are additionally three integral coordinates distinguished by the use case:
//!
//! | symbol | variable | object | units |
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
//! | symbol | variable | object |
//! |---|---|---|
//! | `Ψ` | `psi_sum` | `Σ_ν conj(ψ_ν)·x_{m+ν}` |
//! | `D` | `d_sum` | the same sum against `d_ν` |
//! | `T` | `t_sum` | the same sum against `t_ν` |
//!
//! ### Estimators
//!
//! | symbol | variable | object | units | frame |
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
//! // per hop, K = (N + 1)/2 entries, w[k] = (Re ψ, Im ψ, Re d, Im d)
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

// #![warn(warnings, dead_code, unused_variables)]

// 🤖 Heavy generation all over this module.  Should be pretty standard academic stuff, so not
// expecting a lot of surprises.  We will, for the most part, swiftly and knowingly eat shit if the
// wavelet is busted. Well-formalized stuff doesn't have a lot of wiggle room to violate the
// consistency of the formalism.

// DEBT The newer `max_rho` and `max_noise_floor` API replacing `truncation` and `tail_db` needs to
// be migrated.  Unfortunately this affects basically all tests.  Most will adapt smoothly.  Fix up
// the mapping constant in spec while you're at it. 🤖
// MAYBE The `d` provided for Hermite interpolation is basically not used at all for the actual `N`
// tap outputs.  It may be appropriate to remove the turn from everywhere upstream of `taps_into`,
// but that's also a ton of little edits that I want to think about before committing to.  Obvious
// LLM task.  Tell the young, impressionable puritans to cry more into my cup.
// MAYBE A ton of the characterization gear for testing belongs higher in the dsp module?
// NEXT High omega filters, starting at around 60% of Nyquist, begin to degrade at low Q.  The
// carrier doesn't have enough detail to represent a fast-changing envelope.  A numerical solution
// for these heavily aliased wavelets may succeed or we may use a Plan with a higher Q beyond some
// empirical cutoff.  Truncating less aggressively can't help because the main lobe itself is
// dominating the breakdown.  The Weighted quadrature might be helpful if its drawbacks (center is
// not precise) can be mitigated.
// NOTE We have logarithmic bin spacings, but the cutoff frequencies that determine which downsample
// will be used are not particularly aware, so it's not expected that we can re-use exact bins in
// any kind of octave structure.  Mel scaling etc also defeats this, so there's no point.
// NOTE Run time of the filter bank generation test (not reflective of actual sample rates and Q) is
// about 220ms on a Zen2+ part in release.  This affects CWT startup time.

// === TABLE RESPONSE (Q = 3.5, quantum 8) ===
//
// fc  1000 fs  6000  w0 1.047198  quantized  25 (unfolded  49)
//   peak gain 2.000000032  dev +1.578e-8 rel
//   width 0.99725
//   peak -0.0125 cents
//   image max         -111.29 dB
//   stopband floor     -109.30 dB
//
// fc   250 fs  3000  w0 0.523599  quantized  49 (unfolded  97)
//   peak gain 2.000000003  dev +1.666e-9 rel
//   width 0.99838
//   peak -0.0164 cents
//   image max         -110.83 dB
//   stopband floor     -106.17 dB
//
// fc 12000 fs 48000  w0 1.570796  quantized  17 (unfolded  33)
//   peak gain 1.999999994  dev -3.209e-9 rel
//   width 0.99550
//   peak -0.0088 cents
//   image max         -111.95 dB
//   stopband floor     -108.55 dB

pub(self) mod generate;
pub(self) mod inspect;
mod refine;
pub(self) mod restrict;
pub(self) mod spec;
pub mod whatsleft;

#[cfg(test)]
mod harness;
#[cfg(test)]
mod tests {
    mod print;
}

use core::f64::consts::{PI, TAU};

use num_complex::Complex64;

use generate::hermite;
use inspect::{Inspect, Sample, OVERSAMPLE};
use spec::TAIL_OVER_FLOOR_DB;
pub use spec::{BinSpec, Shape, WaveletSpec};

pub mod defaults {
    #[cfg(debug_assertions)]
    pub const RESOLUTION: usize = 64;
    #[cfg(not(debug_assertions))]
    pub const RESOLUTION: usize = 256;

    pub const DELAY: usize = 0;
    pub const GAMMA: f64 = 3.0;
    pub const GRID_EPS: f64 = 1e-9;
    pub const LOAD_QUANTUM: usize = 4;
    pub const Q: f64 = 3.5;
    pub const TAIL_DB: f64 = -40.0;
    pub const NOISE_FLOOR: f64 = -70.0;
}

/// Filter peak gain. Analytic taps see half a real tone's amplitude, so |H| = 2 makes a unit tone
/// read |W| = 1.
const PEAK_GAIN: f64 = 2.0;

/// The mother wavelet (aka Motherlet®) that has been densely sampled and can serve every [`Bin`]
/// sharing a [`Shape`], up to the maxima specified in the [`WaveletSpec`].
pub struct Wavelet {
    shape: Shape,
    /// Periods per grid point.
    du: f64,
    /// Ψ
    psi: Vec<Complex64>,
    /// Rotated (for symmetry) derivative, `−(i/2π)·dψ/du`.
    d: Vec<Complex64>,
    /// Bin defaults, and the ceiling the grid extent was sized against.
    limits: BinSpec,
    /// Strategy for reducing grid to `N` taps.
    restriction: restrict::Restriction,
    /// Strategy for refining the post-restiction spectral and analytic characteristics.
    refinement: Option<refine::Refinement>,
}

impl Wavelet {
    /// Return a `Bin` definition that may be used to bake `N` taps into some destination memory.
    /// New bin will use the limits from the `WaveletSpec`
    pub fn bin(&self, center: f64, rate: f64) -> Bin<'_> {
        Bin::new(
            self,
            BinSpec {
                center,
                rate,
                ..self.limits
            },
        )
    }

    pub fn from_spec(&self, spec: BinSpec) -> Bin<'_> {
        Bin::new(self, spec)
    }

    /// A bin named by ρ directly, periods per tap.  `bin(fc, fs)` is `at_rho(fc / fs)`.
    pub fn at_rho(&self, rho: f64) -> Bin<'_> {
        self.bin(rho, 1.0)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn restriction(&self) -> restrict::Restriction {
        self.restriction
    }

    fn grid(&self) -> Grid<'_> {
        Grid {
            psi: &self.psi,
            d: &self.d,
            du: self.du,
        }
    }
}

/// The baked motherlet, resolved on `u`.
#[derive(Clone, Copy)]
struct Grid<'w> {
    pub psi: &'w [Complex64],
    pub d: &'w [Complex64],
    pub du: f64,
}

impl<'w> Grid<'w> {
    pub fn at(&self, u: f64) -> Complex64 {
        hermite::eval(self.psi, self.d, u, self.du)
    }

    pub fn mass(&self, u_beg: f64, u_end: f64) -> Complex64 {
        hermite::integrate(self.psi, self.d, u_beg, u_end, self.du)
    }
}

/// A spec resolved against one bake.  Carrying the borrow keeps a bin off the wrong motherlet,
/// where the tap count would be wrong outright.
#[derive(Clone, Copy)]
pub struct Bin<'w> {
    wavelet: &'w Wavelet,
    spec: BinSpec,
    /// Periods per tap, `fc / fs`.  The only bridge from sample rate into `u`.
    rho: f64,
    /// Folded weights including the center tap.
    k: usize,
}

impl<'w> Bin<'w> {
    fn new(wavelet: &'w Wavelet, spec: BinSpec) -> Self {
        let l = wavelet.limits;
        let rho = spec.center / spec.rate;

        debug_assert!(rho <= l.center, "rho {rho} over the wavelet's {}", l.center);
        debug_assert!(
            spec.noise_floor >= l.noise_floor,
            "noise floor {} under the wavelet's {}",
            spec.noise_floor,
            l.noise_floor
        );
        debug_assert!(spec.load_quantum <= l.load_quantum);
        debug_assert!(spec.delay <= l.delay);

        let tail_db = spec.noise_floor - TAIL_OVER_FLOOR_DB;
        let reach = (wavelet.shape.truncation_u(tail_db) / rho).ceil() as usize;
        let half = (reach + spec.delay).div_ceil(spec.load_quantum) * spec.load_quantum;

        Bin {
            wavelet,
            spec,
            rho,
            k: half + 1,
        }
    }

    /// Set the bin's delay.  Must respect `Wavelet` limits.
    pub fn with_delay(self, delay: usize) -> Self {
        Self::new(self.wavelet, BinSpec { delay, ..self.spec })
    }

    /// Set the bin's load quantum.  Must respect `Wavelet` limits.
    pub fn with_load_quantum(self, load_quantum: usize) -> Self {
        Self::new(
            self.wavelet,
            BinSpec {
                load_quantum,
                ..self.spec
            },
        )
    }

    pub fn with_noise_floor(self, db: f64) -> Self {
        Self::new(
            self.wavelet,
            BinSpec {
                noise_floor: -db.abs(),
                ..self.spec
            },
        )
    }

    /// Set the bin's center frequency, using its sample rate multiplied by the periods-per-sample
    /// density, `rho`.
    pub fn with_rho(self, rho: f64) -> Self {
        let center = self.spec.rate * rho;
        Self::new(
            self.wavelet,
            BinSpec {
                center,
                ..self.spec
            },
        )
    }

    /// Set the bin's tail dB.  Must respect `Wavelet` limits.
    pub fn with_truncation(self, tail_db: f64) -> Self {
        self.with_noise_floor(tail_db.abs() - TAIL_OVER_FLOOR_DB)
    }

    /// Options used to create this bin.
    pub fn spec(&self) -> BinSpec {
        self.spec
    }

    /// Radians per tap.  `ω₀ = 2π·ρ`
    pub fn velocity(&self) -> f64 {
        TAU * self.rho
    }

    /// Periods per tap.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Number of folded weights, including a center tap weighs.  `K = (N + 1) / 2`.
    pub fn len_folded(&self) -> usize {
        self.k
    }

    /// Number of real taps after weights are unfolded.  `2K - 1`.  Center tap is still just one
    /// tap.
    pub fn len_unfolded(&self) -> usize {
        2 * self.k - 1
    }

    /// ψ and d for this bin, normalized at the measured crest of |H|.
    ///
    /// ```text
    /// H(ω_peak) = PEAK_GAIN
    /// H_d(ω) ≈ (ω/ω₀)·H(ω)
    /// ```
    pub fn write(&self, bake: &mut Bake) {
        let (wav, rho, w0) = (self.wavelet, self.rho, self.velocity());
        let Bake { weights: w, probe } = bake;

        w.resize(self.k);
        wav.restriction.psi_into(wav.grid(), rho, &mut w.psi);
        debug_assert!(
            w.psi.iter().all(|p| p.is_finite()),
            "restriction produced a non-finite weight at rho {rho}"
        );

        if let Some(refinement) = wav.refinement {
            refinement.apply(&mut w.psi, wav.shape, rho);
        }

        // ω_peak and H(ω_peak)
        let (peak, gain) = Inspect::new(w.psi(), probe, OVERSAMPLE)
            .peak(w0, 0.0, PI)
            .unwrap_or((w0, w.psi().dtft(w0)));

        let scale = PEAK_GAIN / gain;
        for p in w.psi.iter_mut() {
            *p *= scale;
        }

        wav.restriction
            .derivative
            .write(&w.psi, peak / TAU, w0, &mut w.d);
    }

    /// Upstream owes an `out` at least `folded_taps()` long.
    pub fn taps_into(&self, out: &mut [[f32; 4]]) -> usize {
        let mut bake = Bake::default();
        self.write(&mut bake);
        bake.weights.pack_into(out)
    }

    /// Only used by storage format aware consumers.
    pub fn taps(&self) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; self.k];
        self.taps_into(&mut out);
        out
    }

    /// Write the bin to an owned vector and and return it as a [`Weights]` for inspection.
    pub fn weights(&self) -> Weights {
        Weights::unpack(&self.taps())
    }
}

/// Folded ψ and d for one bin.
#[derive(Default)]
pub struct Weights {
    psi: Vec<Complex64>,
    d: Vec<Complex64>,
}

impl Weights {
    pub(self) fn psi(&self) -> Fold<'_> {
        Fold::new(&self.psi)
    }

    #[cfg(test)]
    pub(self) fn d(&self) -> Fold<'_> {
        Fold::new(&self.d)
    }

    fn resize(&mut self, k: usize) {
        self.psi.clear();
        self.d.clear();
        self.psi.resize(k, Complex64::default());
        self.d.resize(k, Complex64::default());
    }

    /// float4(Re ψ, Im ψ, Re d, Im d), index 0 real in both lanes.
    pub fn pack_into(&self, out: &mut [[f32; 4]]) -> usize {
        let k = self.psi.len();
        out[0] = [self.psi[0].re, 0.0, self.d[0].re, 0.0].map(|v| v as f32);
        for (o, (p, q)) in out[1..k]
            .iter_mut()
            .zip(self.psi[1..].iter().zip(&self.d[1..]))
        {
            *o = [p.re, p.im, q.re, q.im].map(|v| v as f32);
        }
        k
    }

    /// Lanes recovered from a packed table, carrying its f32 rounding.
    pub fn unpack(table: &[[f32; 4]]) -> Self {
        let lane = |c: usize| {
            table
                .iter()
                .map(|w| Complex64::new(w[c] as f64, w[c + 1] as f64))
                .collect()
        };
        Weights {
            psi: lane(0),
            d: lane(2),
        }
    }

    /// Ψ, D, T about center `m`, accumulated as the shader does.
    #[cfg(test)]
    pub(super) fn project(&self, x: impl Fn(isize) -> f64, m: isize) -> [Complex64; 3] {
        let x0 = x(m);
        let mut psi = Complex64::new(self.psi[0].re * x0, 0.0);
        let mut dee = Complex64::new(self.d[0].re * x0, 0.0);
        let mut tee = Complex64::default();

        for (j, (p, q)) in self.psi[1..].iter().zip(&self.d[1..]).enumerate() {
            let k = j as isize + 1;
            let (hi, lo) = (x(m + k), x(m - k));
            let (sum, dif) = (hi + lo, hi - lo);

            psi += Complex64::new(p.re * sum, -p.im * dif);
            dee += Complex64::new(q.re * sum, -q.im * dif);
            tee += k as f64 * Complex64::new(p.re * dif, -p.im * sum);
        }
        [psi, dee, tee]
    }
}

/// Reusable scratch for filling bins.  Separate fields so the probe stays borrowable while the
/// weights are being read.
#[derive(Default)]
pub struct Bake {
    weights: Weights,
    probe: Vec<Sample>,
}

impl Bake {
    pub fn weights(&self) -> &Weights {
        &self.weights
    }
}

/// A borrowed view of a single channel of [`Weights`].
///
/// `ψ` indexed over `[0, K)`, the mirror `ψ₋ₖ = conj ψₖ` implied.  Entry 0 is real.
// Making the channel first class could prevent some kinds of mishandling
#[derive(Clone, Copy)]
pub(super) struct Fold<'a>(&'a [Complex64]);

impl<'a> Fold<'a> {
    pub(self) fn new(psi: &'a [Complex64]) -> Self {
        Fold(psi)
    }

    /// Number of real taps after weights are unfolded, in `[0, 2K - 1]`.  Center tap is still just
    /// one tap.
    pub fn len_unfolded(&self) -> usize {
        2 * self.0.len() - 1
    }

    /// Number of physical weights that unfold into taps.  Includes a center weight.  Always equal to `K`.
    #[allow(unused)]
    pub fn len_folded(&self) -> usize {
        self.0.len()
    }

    /// `ψ₀ + 2 Σ_{k≥1} Re(ψ_k e^{−iωk})`
    pub(self) fn dtft(&self, w: f64) -> f64 {
        // k = LANES·b + i + 1
        const LANES: usize = 16;

        // e^{−iω(i+1)}
        let mut lane = [Complex64::default(); LANES];
        for i in 0..LANES {
            lane[i] = Complex64::from_polar(1.0, -w * (i + 1) as f64);
        }
        // e^{−iω·LANES}
        let block = lane[LANES - 1];

        let (mut a_re, mut a_im) = ([0.0; LANES], [0.0; LANES]);
        let (mut b_re, mut b_im) = ([0.0; LANES], [0.0; LANES]);
        let mut phi = Complex64::new(1.0, 0.0);

        let (blocks, tail) = self.0[1..].as_chunks::<LANES>();

        for c in blocks {
            for i in 0..LANES {
                a_re[i] += phi.re * c[i].re;
                a_im[i] += phi.re * c[i].im;
                b_re[i] += phi.im * c[i].re;
                b_im[i] += phi.im * c[i].im;
            }
            phi *= block;
            // Newton step toward |phi| = 1
            phi *= 0.5 * (3.0 - phi.norm_sqr());
        }

        for (i, h) in tail.iter().enumerate() {
            a_re[i] += phi.re * h.re;
            a_im[i] += phi.re * h.im;
            b_re[i] += phi.im * h.re;
            b_im[i] += phi.im * h.im;
        }

        // cos(ωk) = pc − qs, sin(ωk) = −(ps + qc)
        let mut sum = 0.0;
        for i in 0..LANES {
            let (c, s) = (lane[i].re, lane[i].im);
            sum += a_re[i] * c - b_re[i] * s - a_im[i] * s - b_im[i] * c;
        }

        self.0[0].re + 2.0 * sum
    }

    /// M_p = Σ_ν ν^p ψ_ν
    #[cfg(test)]
    pub(self) fn moment(&self, p: i32) -> Complex64 {
        let head = match p {
            0 => self.0[0],
            _ => Complex64::default(),
        };
        self.0[1..].iter().enumerate().fold(head, |m, (j, h)| {
            let w = 2.0 * ((j + 1) as f64).powi(p);
            m + w * match p % 2 == 0 {
                true => Complex64::new(h.re, 0.0),
                false => Complex64::new(0.0, h.im),
            }
        })
    }

    /// A_p = Σ_ν |ν|^p |ψ_ν|
    #[cfg(test)]
    pub(self) fn abs_moment(&self, p: i32) -> f64 {
        let head = match p {
            0 => self.0[0].norm(),
            _ => 0.0,
        };
        head + 2.0
            * self.0[1..]
                .iter()
                .enumerate()
                .map(|(j, h)| ((j + 1) as f64).powi(p) * h.norm())
                .sum::<f64>()
    }

    /// Σ_ν |ψ_ν|
    #[cfg(test)]
    pub(self) fn l1(&self) -> f64 {
        self.0[0].norm() + 2.0 * self.0[1..].iter().map(|h| h.norm()).sum::<f64>()
    }

    /// Σ_ν |ψ_ν|²
    #[cfg(test)]
    pub(self) fn energy(&self) -> f64 {
        self.0[0].norm_sqr() + 2.0 * self.0[1..].iter().map(|h| h.norm_sqr()).sum::<f64>()
    }

    /// Variance of the magnitude envelope.
    ///
    /// Σ ν²|ψ_ν| / Σ |ψ_ν|
    #[cfg(test)]
    pub(self) fn envelope_var(&self) -> f64 {
        let num: f64 = self.0[1..]
            .iter()
            .enumerate()
            .map(|(j, h)| {
                let k = (j + 1) as f64;
                2.0 * k * k * h.norm()
            })
            .sum();
        num / self.l1()
    }

    /// ψ over [−K, K), for display.
    #[cfg(test)]
    pub(super) fn mirrored(&self) -> impl Iterator<Item = Complex64> + '_ {
        let tail = self.0[1..].iter().copied();
        tail.clone()
            .rev()
            .map(|h| h.conj())
            .chain(self.0.iter().copied())
    }
}

// Proper location for this has become amorphous.  Burn something when convenient.
#[cfg(test)]
fn fmt_e(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x:>9}");
    }
    let s = format!("{x:+.2e}");
    // split "±m.mme±dd" into mantissa and exponent, then zero-pad the exponent
    let (mantissa, exp) = s.split_once('e').unwrap_or(("999", "999"));
    let exp: i32 = exp.parse().unwrap();
    format!("{mantissa}e{exp:+03}")
}

#[cfg(test)]
mod test {
    // DEBT a lot of the matrix code is becoming pretty redundant.  It's simple, but every test
    // builds a matrix slightly differently.  See the `unit_tone_reads_unity` test for some raw
    // harness ideas taking shape.  It does a much more complete matrix than other tests and reports
    // threshold headroom and total error mass.o
    // DEBT the print tests are secretly being used in a manner that the workbench binary is
    // intended for.  As things are becoming mature enough, moving some features over to the
    // workbench tools would be welcome, although it may need some redesign since basically all IIR
    // solutions and therefore most of the tests with time dependency are DoA on the GPU.

    use core::f64::consts::LN_2;

    use super::*;

    use harness::*;
    use inspect::*;

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

    /// Whether the filter answers the same at every input phase.  Each row drives a steady tone
    /// through the full 2π of carrier phase and demodulates each lane, so the reported number is
    /// the worst relative departure from that lane's phase mean.  Zero is phase blind.
    #[test]
    fn response_is_phase_independent() {
        const Q: f64 = 8.5;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -60.0;

        const GATE_DB: f64 = -50.0;
        const RESOLUTION: f64 = 0.05;

        /// Well above anything the image alone produces in band.
        const SWING_TOL: f64 = 5e-2;

        const STEP: f64 = 25.0;
        const SPAN: isize = 6;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        println!("\n=== RESPONSE PHASE DEPENDENCE ===");
        println!("  detune      |H|    img ψ         psi           d           t");

        for (fc, fs) in [
            (2_000.0f64, RATE),
            (200.0, 3000.0),
            (250.0, 3000.0),
            (12_000.0, RATE),
        ] {
            let bin = wav.at_rho(fc / fs);
            let wts = bin.weights();
            let psi = wts.psi();

            let (n, w0) = (psi.len_unfolded(), bin.velocity());
            println!("  fc {fc:.0} fs {fs:.0} taps {n} w0 {w0:.6}");

            for k in -SPAN..=SPAN {
                let cents = k as f64 * STEP;
                let wd = w0 * (cents / 1200.0).exp2();
                let h_db = db(psi.dtft(wd).abs()) - db(PEAK_GAIN);
                if h_db < GATE_DB {
                    continue;
                }

                let [sp, sd, st] = tone_response(&wts, w0, cents, RESOLUTION);
                let img = psi.dtft(-wd).abs();

                println!(
                    "  {cents:+6.0}c {h_db:>7.1} {:>8.1} {sp:>11.2e} {sd:>11.2e} {st:>11.2e}",
                    db(img) - db(PEAK_GAIN),
                );

                for (lane, s) in ["psi", "d", "t"].iter().zip([sp, sd, st]) {
                    assert!(
                        s < SWING_TOL,
                        "fc {fc} detune {cents} lane {lane} swing {s:.3e}"
                    );
                }
            }
        }
    }

    /// Reassignment error over the detuning each bin is responsible for.  A bank at `SPACING`
    /// cents hands off at half that, so beyond it the reading belongs to a neighbor.
    #[test]
    fn reassignment_is_unbiased() {
        const Q: f64 = 3.5;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -40.0;

        const SPACING: f64 = 400.0;
        const STEPS: isize = 8;
        const RESOLUTION: f64 = 0.05;

        /// Worst per hop error in cents, bias plus the swing across carrier phase.
        const ERROR_C: f64 = 4.0;

        const SKIRT_DB: f64 = -30.0;
        const SKIRT_C: f64 = 20.0;
        /// Reach of the crossing search, an octave either side.
        const SPAN: f64 = 1.0;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        println!("\n=== REASSIGN ===");

        for (fc, fs) in [
            (2_000.0f64, RATE),
            (200.0, 3000.0),
            (250.0, 3000.0),
            (12_000.0, RATE),
        ] {
            let bin = wav.at_rho(fc / fs);
            let wts = bin.weights();
            let (psi, d) = (wts.psi(), wts.d());

            let (n, w0) = (psi.len_unfolded(), bin.velocity());
            println!("  fc {fc:.0} fs {fs:.0} taps {n} w0 {w0:.6}");
            println!("  detune      |H|     pred      bias     swing      leak");

            for k in -STEPS..=STEPS {
                let cents = k as f64 * 0.5 * SPACING / STEPS as f64;
                // 2^(c/1200)
                let ratio = (cents / 1200.0).exp2();
                let h = psi.dtft(w0 * ratio).abs();

                let ((bias, swing), (leak, _)) = tone_bias(&wts, w0, cents, RESOLUTION);
                let (_, dr) = pairing_residual(psi, d, w0, w0 * ratio);
                // (1200 / ln 2) · Re(R/Ψ̂) / r
                let pred = 1200.0 / LN_2 * dr / ratio;

                println!(
                    "  {cents:+6.1}c {:>7.1} {pred:>8.3}c {bias:>8.3}c {swing:>9.4}c {leak:>9.2e}",
                    20.0 * (h / PEAK_GAIN).log10(),
                );

                assert!(
                    bias.abs() + swing < ERROR_C,
                    "fc {fc} detune {cents} error {:.4}c",
                    bias.abs() + swing
                );
            }

            let peak = psi.dtft(w0).abs();
            let (lo, hi) = shoulders(psi, peak, w0, SKIRT_DB, w0 * SPAN);

            for w in [lo, hi].into_iter().flatten() {
                // 1200 log2(ω/ω₀)
                let cents = 1200.0 * (w / w0).log2();
                let ((bias, swing), _) = tone_bias(&wts, w0, cents, RESOLUTION);
                let worst = bias.abs() + swing;
                println!("  skirt {cents:+7.1}c  bias {bias:+8.3}c  swing {swing:8.3}c");
                assert!(
                    worst < SKIRT_C,
                    "fc {fc} skirt {cents:.1} worst {worst:.3}c"
                );
            }
        }
    }

    /// Whether the reported pitch and quadrature move with the carrier phase.  Each row sweeps
    /// the full 2π of input phase at `RESOLUTION` and reports the departure from the phase mean,
    /// so a filter that answers the same for every phase reads zero across the board.  The swing
    /// is the negative frequency image beating against the signal, so the budget scales with
    /// image over signal rather than being flat.
    #[test]
    fn reassignment_is_phase_independent() {
        const Q: f64 = 8.5;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -60.0;

        const GATE_DB: f64 = -50.0;
        const RESOLUTION: f64 = 0.05;

        /// Swing the image alone implies, within this factor.  Second order in |α|, so tight.
        const SWING_C: f64 = 1.3;

        const STEP: f64 = 25.0;
        const SPAN: isize = 6;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        println!("\n=== PHASE DEPENDENCE ===");
        println!("  detune      |H|    img ψ    img d     pred     swing     ratio");

        for (fc, fs) in [
            (2_000.0f64, RATE),
            (200.0, 3000.0),
            (250.0, 3000.0),
            (12_000.0, RATE),
        ] {
            let bin = wav.at_rho(fc / fs);
            let wts = bin.weights();
            let (psi, d) = (wts.psi(), wts.d());

            let (n, w0) = (psi.len_unfolded(), bin.velocity());
            println!("  fc {fc:.0} fs {fs:.0} taps {n} w0 {w0:.6}");

            for k in -SPAN..=SPAN {
                let cents = k as f64 * STEP;
                let wd = w0 * (cents / 1200.0).exp2();
                let h = psi.dtft(wd).abs();
                let h_db = 20.0 * (h / PEAK_GAIN).log10();
                if h_db < GATE_DB {
                    continue;
                }

                let ((_, swing), (quad, _)) = tone_bias(&wts, w0, cents, RESOLUTION);

                // H_ψ(±ω), H_d(±ω)
                let (p, p_img) = (psi.dtft(wd), psi.dtft(-wd));
                let (q, q_img) = (d.dtft(wd), d.dtft(-wd));
                // α = H_d(−ω)/H_d(ω) − H_ψ(−ω)/H_ψ(ω)
                let alpha = q_img / q - p_img / p;
                let pred = 1200.0 / LN_2 * alpha.abs();

                println!(
                    "  {cents:+6.0}c {h_db:>7.1} {:>8.1} {:>8.1} {pred:>8.3}c {swing:>8.3}c {:>9.3}",
                    db(p_img.abs()) - db(PEAK_GAIN),
                    db(q_img.abs()) - db(PEAK_GAIN),
                    swing / pred,
                );

                assert!(
                    swing < SWING_C * pred + 1e-3,
                    "fc {fc} detune {cents} swing {swing:.4}c over {:.4}c",
                    SWING_C * pred + 1e-3
                );
                assert!(
                    quad.abs() < 1e-12,
                    "fc {fc} detune {cents} quad mean {quad:.3e}"
                );
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
        const WIDTH_TOL: f64 = 0.02;

        let w = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(FULL_DB)
            .bake();

        let to_peak = |v: f64| db(v) - db(PEAK_GAIN);
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
            let full = w.bin(fc, RATE).with_truncation(FULL_DB);
            let nf = full.len_folded();
            let wts = full.weights();
            let psi = wts.psi();

            let w0 = full.velocity();
            let rf = characterize(psi, w0);
            let dcf = dc_leak(psi, w0);

            println!(
                "  {fc:>6.0} {nf:>5} {:>9.6} {:>8.4} {:>+9.3} {:>8.2} {:>8.2} {:>8.2}",
                rf.gain,
                rf.rel_width * Q,
                cents(rf.peak_w, w0),
                to_peak(dcf),
                to_peak(rf.image),
                to_peak(rf.floor)
            );

            // Peak sits below ω₀ by the cell-average droop, gain rising as ½P²Δx².
            assert!(
                (rf.gain - 2.0).abs() < 1e-5,
                "fc {fc} full gain {:.9}",
                rf.gain
            );
            assert!(
                dcf < 1e-5 * PEAK_GAIN,
                "fc {fc} full dc {:.2} dB",
                to_peak(dcf)
            );

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
                let cut = w.bin(fc, RATE).with_truncation(tail_db);
                let nc = cut.len_folded();
                let wts = cut.weights();
                let psi = wts.psi();
                let rc = characterize(psi, w0);
                let dc = dc_leak(psi, w0);

                println!(
                    "  {fc:>6.0} {tail_db:>7.1} {nc:>5} {:>6.3} {:>9.6} {:>8.4} {:>+9.3} \
                     {:>8.2} {:>8.2} {:>8.2}",
                    nc as f64 / nf as f64,
                    rc.gain,
                    rc.rel_width * Q,
                    cents(rc.peak_w, w0),
                    to_peak(dc),
                    to_peak(rc.image),
                    to_peak(rc.floor)
                );

                // The band neither moves nor widens.
                assert!(
                    (cents(rc.peak_w, w0) - cents(rf.peak_w, w0)).abs() < 8.0,
                    "fc {fc} tail {tail_db} peak moved {:+.4}c",
                    cents(rc.peak_w, w0) - cents(rf.peak_w, w0)
                );
                if tail_db.abs() > 20.0 {
                    assert!(
                        (rc.rel_width / rf.rel_width - 1.0).abs() < 0.1,
                        "fc {fc} tail {tail_db} width {:+.3}%",
                        100.0 * (rc.rel_width / rf.rel_width - 1.0)
                    );
                }

                // Gain near DC combines both positive and negative components, so use of PEAK_GAIN
                // is apt here.
                // ROLL DC centering
                assert!(
                    dc < 1e-1 * PEAK_GAIN,
                    "fc {fc} tail {tail_db} dc {:.2} dB",
                    to_peak(dc)
                );

                // Weaker truncation buys floor with taps.  Taps may hold under quantum rounding.
                assert!(
                    nc >= prev_taps,
                    "fc {fc} tail {tail_db} taps {nc} < {prev_taps}"
                );

                // XXX More reliable floor needed before we can assert this.
                // assert!(
                //     (nc == prev_taps && rc.floor >= prev_floor) || rc.floor < prev_floor,
                //     "fc {fc} tail {tail_db} taps {prev_taps} -> {nc} without floor gain: {} to {}",
                //     prev_floor,
                //     rc.floor,
                // );

                (prev_taps, prev_floor) = (nc, rc.floor);
            }
            println!();
        }
    }

    /// Same four numbers as `response_is_characterized`, measured on the folded weight table.
    /// Sweeps the load quantum, because the quantum pads the emitted half-span.
    #[test]
    fn table_response_is_characterized() {
        const Q: f64 = 3.5;
        const TAIL_DB: f64 = -65.0;

        let w = wavelet(Q, 16);

        for quantum in [2usize, 4, 8, 16] {
            println!("\n=== TABLE RESPONSE (Q = {Q}, quantum {quantum}) ===");

            for (fc, fs) in [(1000.0f64, 6000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = w
                    .bin(fc, fs)
                    .with_load_quantum(quantum)
                    .with_truncation(TAIL_DB);
                let wts = bin.weights();

                let w0 = bin.velocity();
                let r = characterize(wts.psi(), w0);
                let rel = |v: f64| db(v) - db(r.gain);

                println!(
                    "\nfc {fc:>5.0} fs {fs:>5.0}  w0 {w0:.6}  quantized {:>3} (unfolded {:>3})",
                    bin.len_folded(),
                    bin.len_unfolded()
                );
                println!(
                    "  peak gain {:.9}  dev {:+.3e} rel",
                    r.gain,
                    r.gain / PEAK_GAIN - 1.0
                );

                println!("  width {:.5}", r.rel_width * Q);
                println!("  peak {:+.4} cents", 1200.0 * (r.peak_w / w0).log2());
                println!("  image max        {:>8.2} dB", rel(r.image));
                println!("  stopband floor    {:>8.2} dB", rel(r.floor));
            }
        }
    }

    #[test]
    fn print_gamma_sweep() {
        const QUANTUM: usize = 4;
        const STEP: f64 = 100.0;
        const SPAN: isize = 4;
        /// Worst reassignment bias over the scan, in cents.
        const BIAS_C: f64 = 1.2;
        const TAIL_DB: f64 = 40.0;

        println!("\n=== TAP PROFILE vs GAMMA (Q = 2.4) ===");
        // P² = beta gamma
        let p = 4.0;
        for gamma in [1.0f64, 2.0, 3.0, 6.0] {
            let wav = WaveletSpec::default()
                .with_shape(Shape {
                    gamma,
                    beta: p * p / gamma,
                })
                .max_load_quantum(QUANTUM)
                .max_truncation(TAIL_DB)
                .bake();
            let bin = wav.at_rho(1000.0 / 8000.0);
            let w0 = bin.velocity();
            let wts = bin.weights();
            let (psi, d) = (wts.psi(), wts.d());
            let n = psi.len_unfolded();

            // M₁ / M₀
            let delay = (psi.moment(1) / psi.moment(0)).re;

            // worst over detuning of |R| / ‖ψ‖₁ and of (1200/ln 2)·(R/H)/r
            let (floor, worst) = (-SPAN..=SPAN).fold((0.0f64, 0.0f64), |acc, k| {
                let ratio = (k as f64 * STEP / 1200.0).exp2();
                let (res, dr) = pairing_residual(psi, d, w0, w0 * ratio);
                (
                    acc.0.max(res / psi.l1()),
                    // (1200 / ln 2) · (R/H) / r
                    acc.1.max((1200.0 / LN_2 * dr / ratio).abs()),
                )
            });

            println!(
                "\ngamma = {gamma:.1}  weights {}  taps {n}  delay = {delay:+.3e}  \
             floor = {}  bias = {worst:.3}c",
                psi.len_folded(),
                fmt_e(floor),
            );

            let mags: Vec<f64> = psi.mirrored().map(|h| h.norm()).collect();
            let max = mags.iter().fold(0.0f64, |a, &b| a.max(b));
            for (j, &v) in mags.iter().enumerate() {
                println!(
                    "{:>4} {}",
                    j as isize - (n / 2) as isize,
                    "#".repeat((v / max * 40.0).round() as usize)
                );
            }

            assert!(worst < BIAS_C, "gamma {gamma} bias {worst:.3}c");
        }
    }

    /// A real unit tone at the crest reads |W| = 1 even though |H| = 2: the analytic taps see only
    /// the +ω half of the cosine, and the −ω half beats against it by ½|H(−ω)|.
    #[test]
    fn unit_tone_reads_unity() {
        const EPS: f64 = 1e-6;
        const BINS: usize = 16;
        const WORST: usize = 2;

        const QS: Axis = Axis::Levels(&[3.0, 4.25, 6.0]);
        const GAMMAS: Axis = Axis::Levels(&[3.0, 4.0, 6.0]);
        const RHO: Span = Span::Log(0.004, 0.45);
        const TAIL_DB: Span = Span::Lin(-20.0, -100.0);

        const GOOD: f64 = 0.99;
        const EXCESS: f64 = 5e-3;

        let mut probe = Vec::new();
        let mut all = Ledger::default();

        println!("\n=== UNIT TONE ===");
        println!(
            "  good is the weighted share within tolerance, excess the capped overshoot mass\n"
        );
        println!("  {:>6} {:>6} {:>8} {:>10}", "Q", "γ", "good", "excess");

        for (_, gamma) in GAMMAS.levels() {
            for (_, q) in QS.levels() {
                let wav = WaveletSpec::default()
                    .with_shape(Shape::from_q(q, gamma))
                    .max_truncation(TAIL_DB.end())
                    .bake();

                let mut shape = Ledger::default();
                for (w, [rho, tail_db]) in survey([RHO, TAIL_DB], BINS) {
                    let bin = wav.at_rho(rho).with_truncation(tail_db);
                    let wts = bin.weights();
                    let psi = wts.psi();

                    // ω_peak
                    let Some((wp, _)) =
                        Inspect::new(psi, &mut probe, OVERSAMPLE).peak(bin.velocity(), 0.0, PI)
                    else {
                        shape.record(w, f64::NAN, at!(q, gamma, rho, tail_db));
                        continue;
                    };

                    // ½|H(−ω_peak)| + ε
                    let tol = 0.5 * psi.dtft(-wp).abs() + EPS;

                    // max over m of | |Ψ(m)| − 1 |
                    let err = (0..8)
                        .map(|m| (wts.project(|k| (wp * k as f64).cos(), m)[0].norm() - 1.0).abs())
                        .fold(0.0f64, f64::max);

                    shape.record(w, err / tol, at!(q, gamma, rho, tail_db));
                }

                println!(
                    "  {q:>6.2} {gamma:>6.2} {:>8.4} {:>10.2e}",
                    shape.good(),
                    shape.excess()
                );
                all.absorb(shape);
            }
        }

        all.print_worst(WORST);
        let [good, excess] = all.verdict([GOOD, EXCESS]);

        assert!(good > GOOD, "good mass {good:.4}");
        assert!(excess < EXCESS, "excess mass {excess:.2e}");
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
        const NOISE_GAIN: f64 = 1.4105;
        const TOL: f64 = 2e-3;

        let w = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        println!("\n=== NOISE GAIN (Q = {Q}, fs = {RATE}, quantum {QUANTUM}) ===");

        for fc in [500.0f64, 1000.0, 2000.0, 4000.0, 8000.0] {
            let bin = w.bin(fc, RATE);
            let wts = bin.weights();
            let psi = wts.psi();

            let e = psi.energy();
            let ratio = e / bin.rho();

            println!(
                "  fc {:>6.0}  taps {:>5}  rho {:.6}  energy {:.6}  e/rho {:.6}  dev {:+.2e}",
                fc,
                bin.len_unfolded(),
                bin.rho(),
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

    /// Smoke test.  DC-free and correct peak gain, measured with the linear DTFT so a broken fold
    /// convention can't agree with itself.  First thing to look at if the bake goes sideways.
    #[test]
    fn taps_are_conditioned() {
        // Use a rough tail dB so we can verify conditioning under challenging conditions.
        const TAIL_DB: f64 = -80.0;
        const MOMENT_TOL: f64 = 1e-6;

        // NOTE the refinement is explicitly set so that this test doesn't regress whenever the
        // default moment conditioning is updated.
        let wav = WaveletSpec::default()
            .max_truncation(TAIL_DB)
            .with_refinement(Some(refine::Refinement::Reach {
                moments: &[0, 1, 2, 3],
                jet: &[0, 1, 2, 3],
                tangent: true,
                spare: 16,
                turns: 3.0,
            }))
            .bake();

        for (fc, fs) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
            let bin = wav.at_rho(fc / fs);
            let wts = bin.weights();
            let (psi, d) = (wts.psi(), wts.d());

            let w0 = bin.velocity();

            let g = psi.dtft(w0).abs();
            assert!((g - PEAK_GAIN).abs() < 1e-2, "fc {fc} peak gain {g:.6}");

            // Analytic taps: the mirror image is stopband, not signal.
            let neg = psi.dtft(-w0).abs();
            assert!(
                neg < 1e-3 * g,
                "fc {fc} negative-freq leak {:.2} dB",
                20.0 * (neg / g).log10()
            );

            let dc = psi.dtft(0.0);
            assert!(dc.abs() < 1e-2 * g, "fc {fc} dc {dc:.3e}");

            // d/ψ reads ω/ω₀, unity at the carrier.
            let gd = d.dtft(w0).abs();
            assert!((gd / g - 1.0).abs() < 1e-2, "fc {fc} d/psi {:.6}", gd / g);

            // Σ ν^p · (Re ψ if p even, Im ψ if p odd)
            let mom = |p: i32| match p % 2 == 0 {
                true => psi.moment(p).re,
                false => psi.moment(p).im,
            };

            // |M_p| / A_p, the share of each moment's scale left uncancelled.  H^(p)(0) for p = 2, 3
            // are the ones the solve nulls that nothing else measures.
            for p in 0..=3 {
                let residue = mom(p).abs() / psi.abs_moment(p);
                println!("  fc {fc:>6.0}  M{p} residue {residue:.3e}");
                assert!(
                    residue < MOMENT_TOL,
                    "fc {fc} moment {p} residue {residue:.3e}"
                );
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
        const TAIL_DB: f64 = -30.0;
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
                    let wts = bin.weights();
                    let psi = wts.psi();
                    let r = characterize(psi, bin.velocity());
                    let lobe = r.edges.1 - r.edges.0;
                    let to_gain = |h: f64| db(h) - db(r.gain);

                    let mut buf = Vec::new();
                    let mut insp = Inspect::new(psi, &mut buf, OVERSAMPLE);
                    let lo = insp.null(r.edges.0, r.edges.0 - PI).map(|(w, _)| w);
                    let hi = insp.null(r.edges.1, r.edges.1 + PI).map(|(w, _)| w);

                    // ∫_band h² on each side of the peak
                    let band_lo = level_moment(psi, r.gain, (r.edges.0, r.peak_w), FLOOR_DB);
                    let band_hi = level_moment(psi, r.gain, (r.peak_w, r.edges.1), FLOOR_DB);

                    let (psl_lo, psl_hi) = skirts(psi, (lo, hi));

                    /// Offset of a dip in −3 dB widths, dash where the skirt never reaches zero.
                    let off = |w: Option<f64>| match w {
                        Some(w) => format!("{:>+8.3}", (w - r.peak_w) / lobe),
                        None => format!("{:>8}", "—"),
                    };
                    let spread = match (lo, hi) {
                        (Some(lo), Some(hi)) => format!("{:>7.3}", (hi - lo) / lobe),
                        _ => format!("{:>7}", "—"),
                    };
                    /// ∫_band h² over ∫_dips h², the dip side measured from the dip.
                    let kappa = |band: f64, dip: Option<f64>, to: (f64, f64)| match dip {
                        Some(_) => {
                            format!("{:>7.4}", band / level_moment(psi, r.gain, to, FLOOR_DB))
                        }
                        None => format!("{:>7}", "—"),
                    };

                    /// Level, offset, and prominence of one skirt, dashes where the band holds
                    /// no lobe.
                    let cols = |s: &Skirt| match (s.peak, s.prominence_db()) {
                        (Some((w, h)), Some(prom)) => format!(
                            "{:>8.2} {:>+7.3} {:>6.1} {:>5}",
                            to_gain(h),
                            (w - r.peak_w) / lobe,
                            prom,
                            s.lobes
                        ),
                        _ => format!("{:>8} {:>7} {:>6} {:>5}", "—", "—", "—", s.lobes),
                    };

                    println!(
                        "  {q:>5.1} {gamma:>4.1} {rho:>7.4} {:>5} {} {} {spread} {} {} {} {}",
                        psi.len_unfolded(),
                        off(lo),
                        off(hi),
                        cols(&psl_lo),
                        cols(&psl_hi),
                        kappa(band_lo, lo, (lo.unwrap_or(r.peak_w), r.peak_w)),
                        kappa(band_hi, hi, (r.peak_w, hi.unwrap_or(r.peak_w))),
                    );

                    let reaches = |s: &Skirt| s.peak.is_some_and(|(_, h)| h >= r.gain);
                    assert!(
                        !reaches(&psl_lo) && !reaches(&psl_hi),
                        "Q {q} γ {gamma} ρ {rho} side lobe reaches the main lobe"
                    );
                }
                println!();
            }
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
                (Weights::unpack(&weights[r.clone()])
                    .psi()
                    .dtft(bin.velocity())
                    .abs()
                    - PEAK_GAIN)
                    .abs()
            })
            .fold(0.0f64, f64::max);

        let (low_bin, low_range) = &voices[0];
        let low = Weights::unpack(&weights[low_range.clone()]);
        print_wave(
            &format!(
                "LOWEST BIN ({:.0}Hz, omega0 {:.5})",
                bins[0].center,
                low_bin.velocity()
            ),
            low.psi(),
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

    // Transport (time-reassignment mis-location) Tests

    /// (|Ψ|², Re(T Ψ̄)) about `m`
    fn moments(wts: &Weights, x: &dyn Fn(isize) -> f64, m: isize) -> (f64, f64) {
        let [psi, _, tee] = wts.project(x, m);
        (psi.norm_sqr(), (tee * psi.conj()).re)
    }

    /// About `n` hops on a uniform stride across [−r, r]
    fn hops(r: f64, n: usize) -> impl Iterator<Item = isize> {
        let stride = (2.0 * r / (n - 1) as f64).ceil().max(1.0) as usize;
        let r = r.ceil() as isize;
        (-r..=r).step_by(stride)
    }

    /// {(q + ½) φ}
    fn offset(q: usize) -> f64 {
        const GOLDEN: f64 = 0.618_033_988_749_894_8;
        ((q as f64 + 0.5) * GOLDEN).fract()
    }

    /// Prints a reassignment transport report and returns the median and worst row E·d, dB re one
    /// period.
    ///
    /// Each row holds (ρ, taps, readings per family), a reading being (E, d) with d in carrier
    /// periods.  Families enter each row at unit energy, so a row's pooled E·d is their mean.
    fn transport_report(
        title: &str,
        names: &[&str],
        rows: &[(f64, usize, Vec<Vec<(f64, f64)>>)],
        limits: [f64; 2],
    ) -> [f64; 2] {
        /// Displacement thresholds, periods.
        const BEYOND: [f64; 4] = [1e-3, 1e-2, 1e-1, 1.0];
        /// Energy band edges, least displaced first.
        const EDGES: [f64; 5] = [0.25, 0.5, 0.75, 0.99, 1.0];
        const BAND_LABELS: [&str; 5] = ["0-25%", "25-50%", "50-75%", "75-99%", "99-100%"];

        // Σ E d / Σ E
        let e_d = |r: &[(f64, f64)]| {
            let (ed, e) = r
                .iter()
                .fold((0.0, 0.0), |(ed, e), v| (ed + v.0 * v.1, e + v.0));
            ed / e
        };
        // 10 log10 u
        let db_u = |u: f64| 10.0 * u.max(f64::MIN_POSITIVE).log10();
        // d below which fraction f of the energy lands
        let quantile = |r: &[(f64, f64)], f: f64| weighted_quantile(&mut r.to_vec(), f);
        let pooled_col = names.len() > 1;

        println!("\n=== {title} ===");
        println!("  E is reported energy |Ψ|², d is displacement ρ |t̂ − t| in carrier periods");
        println!("  E·d = Σ E·d / Σ E, in dB re one period");
        println!("  q3 and q99 are d in periods, below which 75% and 99% of E lands\n");

        print!("  {:>7} {:>5}", "rho", "taps");
        for n in names {
            print!(" {n:>8}");
        }
        if pooled_col {
            print!(" {:>8}", "pooled");
        }
        println!(" {:>9} {:>9}", "q3 d", "q99 d");
        let cols = names.len() + pooled_col as usize;
        println!("  {}", "-".repeat(13 + 9 * cols + 20));

        // Rows
        let mut all = Vec::new();
        let mut row_ed = Vec::with_capacity(rows.len());
        for (rho, taps, families) in rows {
            let mut pooled = Vec::new();
            print!("  {rho:>7.4} {taps:>5}");
            for f in families {
                let total: f64 = f.iter().map(|v| v.0).sum();
                print!(" {:>8.2}", db_u(e_d(f)));
                pooled.extend(f.iter().map(|&(e, d)| (e / total, d)));
            }
            let ed = e_d(&pooled);
            if pooled_col {
                print!(" {:>8.2}", db_u(ed));
            }
            println!(
                " {:>9.2e} {:>9.2e}",
                quantile(&pooled, 0.75),
                quantile(&pooled, 0.99)
            );
            row_ed.push((*rho, ed));
            all.extend(pooled);
        }

        // Row quartiles
        let mut spread: Vec<f64> = row_ed.iter().map(|r| r.1).collect();
        spread.sort_by(f64::total_cmp);
        let at = |f: f64| db_u(spread[((spread.len() - 1) as f64 * f).round() as usize]);

        println!("\n  E·d quartiles over rows, dB re one period");
        println!("  {:>8} {:>8} {:>8}", "q1", "median", "q3");
        println!("  {:>8.2} {:>8.2} {:>8.2}", at(0.25), at(0.5), at(0.75));

        // Row extremes
        let by_ed = |a: &&(f64, f64), b: &&(f64, f64)| a.1.total_cmp(&b.1);
        let (best_rho, best) = *row_ed.iter().min_by(by_ed).unwrap();
        let (worst_rho, worst) = *row_ed.iter().max_by(by_ed).unwrap();

        println!("\n  E·d row extremes");
        println!("  {:>8} {:>8} {:>8}", "", "rho", "dB");
        println!("  {:>8} {best_rho:>8.4} {:>8.2}", "best", db_u(best));
        println!("  {:>8} {worst_rho:>8.4} {:>8.2}", "worst", db_u(worst));

        // Energy beyond each displacement
        let total: f64 = all.iter().map(|v| v.0).sum();

        println!("\n  share of E displaced beyond d periods");
        print!(" ");
        for b in BEYOND {
            print!(" {:>9}", format!(">{b:e}"));
        }
        print!("\n ");
        for b in BEYOND {
            // Σ_{d > b} E / Σ E
            let share = all.iter().filter(|v| v.1 > b).map(|v| v.0).sum::<f64>() / total;
            print!(" {:>8.3}%", 100.0 * share);
        }
        println!();

        // Energy bands, least displaced first
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut banded = [(0.0f64, 0.0f64, 0.0f64); EDGES.len()];
        let mut acc = 0.0;
        for &(e, d) in &all {
            let b = EDGES
                .partition_point(|&x| x <= (acc + 0.5 * e) / total)
                .min(EDGES.len() - 1);
            let o = &mut banded[b];
            *o = (o.0 + e / total, d, o.2 + e * d / total);
            acc += e;
        }
        let total_ed: f64 = banded.iter().map(|b| b.2).sum();

        println!("\n  E·d by energy band, bands sum to the mean row E·d");
        println!("  E is the band's share of energy, d is displacement in periods");
        println!(
            "  {:>8} {:>7} {:>10} {:>10} {:>10} {:>8} {:>7}",
            "band", "E", "mean d", "max d", "E·d", "E·d dB", "% E·d"
        );
        for (l, (e, dmax, ed)) in BAND_LABELS.iter().zip(banded) {
            println!(
                "  {l:>8} {:>6.1}% {:>10.2e} {dmax:>10.2e} {ed:>10.2e} {:>8.2} {:>6.1}%",
                100.0 * e,
                ed / e,
                db_u(ed),
                100.0 * ed / total_ed
            );
        }

        // Headline
        let measured = [at(0.5), db_u(worst)];
        let verdict = |h: f64| if h > 0.0 { "pass" } else { "FAIL" };
        let rule = "=".repeat(31);

        println!("\n  {rule}");
        println!("  {:>9} {:>10} {:>10}", "E·d dB", "median row", "worst row");
        println!("  {rule}");
        println!(
            "  {:>9} {:>10.2} {:>10.2}",
            "measured", measured[0], measured[1]
        );
        println!("  {:>9} {:>10.2} {:>10.2}", "limit", limits[0], limits[1]);
        print!("  {:>9}", "headroom");
        for (m, l) in measured.iter().zip(limits) {
            print!(" {:>10.2}", l - m);
        }
        print!("\n  {:>9}", "");
        for (m, l) in measured.iter().zip(limits) {
            print!(" {:>10}", verdict(l - m));
        }
        println!("\n  {rule}");

        measured
    }

    /// Energy that time reassignment moves off an impulse, and how far.
    ///
    ///     E·d = Σ E ρ |t̂ − p| / Σ E
    ///
    /// A band-limited impulse at sub-sample `p` has exact truth at every hop.  E = |Ψ|² is reported
    /// energy and d is displacement in carrier periods, the dilation invariant frame.
    #[test]
    fn t_hat_impulse_transport() {
        const Q: f64 = 3.5;
        const GAMMA: f64 = 3.0;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -40.0;

        const PITCHES: usize = 48;
        const RHO_LO: f64 = 0.01;
        const RHO_HI: f64 = 0.375;
        const HOPS: usize = 64;
        const PHASES: usize = 6;

        const MEDIAN_DB: f64 = -35.0;
        const WORST_DB: f64 = -10.0;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, GAMMA))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        let rows: Vec<_> = (0..PITCHES)
            .map(|i| {
                // ρ_lo (ρ_hi / ρ_lo)^(i / (n − 1))
                let rho = RHO_LO * (RHO_HI / RHO_LO).powf(i as f64 / (PITCHES - 1) as f64);
                let bin = wav.at_rho(rho);
                let wts = bin.weights();
                let half = (bin.len_folded() - 1) as f64;

                let mut readings = Vec::with_capacity(PHASES * HOPS);
                for q in 0..PHASES {
                    let p = offset(q);
                    // sinc(k − p)
                    let x = move |k: isize| {
                        let d = PI * (k as f64 - p);
                        d.sin() / d
                    };
                    for m in hops(half, HOPS) {
                        let (e, c) = moments(&wts, &x, m);
                        if e > 0.0 {
                            // ρ |m + c/e − p|
                            readings.push((e, rho * (m as f64 + c / e - p).abs()));
                        }
                    }
                }
                (rho, bin.len_unfolded(), vec![readings])
            })
            .collect();

        let [median, worst] = transport_report(
            &format!("IMPULSE T_HAT (Q {Q}, γ {GAMMA}, quantum {QUANTUM}, tail {TAIL_DB:.0} dB)"),
            &["impulse"],
            &rows,
            [MEDIAN_DB, WORST_DB],
        );

        assert!(
            median < MEDIAN_DB,
            "median row E·d {median:.2} dB re one period"
        );
        assert!(
            worst < WORST_DB,
            "worst row E·d {worst:.2} dB re one period"
        );
    }

    /// Energy that time reassignment moves off a stationary tone, and how far, per detune.
    ///
    ///     E·d = Σ E ρ |t̂ − m| / Σ E
    ///
    /// A stationary tone belongs to every hop, so truth is the hop itself.  Error comes from the
    /// image beating against the signal, which differs above and below the carrier.
    #[test]
    fn t_hat_tone_transport() {
        const Q: f64 = 3.5;
        const GAMMA: f64 = 3.0;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -40.0;

        const PITCHES: usize = 48;
        const RHO_LO: f64 = 0.01;
        const RHO_HI: f64 = 0.375;
        const DETUNES_C: [f64; 5] = [-300.0, -150.0, 0.0, 150.0, 300.0];
        const PHASES: usize = 12;

        const MEDIAN_DB: f64 = -30.0;
        const WORST_DB: f64 = -20.0;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, GAMMA))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        let rows: Vec<_> = (0..PITCHES)
            .map(|i| {
                // ρ_lo (ρ_hi / ρ_lo)^(i / (n − 1))
                let rho = RHO_LO * (RHO_HI / RHO_LO).powf(i as f64 / (PITCHES - 1) as f64);
                let bin = wav.at_rho(rho);
                let wts = bin.weights();
                let w0 = bin.velocity();

                let families = DETUNES_C
                    .iter()
                    .map(|cents| {
                        // ω₀ 2^(c/1200)
                        let w = w0 * (cents / 1200.0).exp2();
                        (0..PHASES)
                            .map(|q| {
                                let theta = TAU * q as f64 / PHASES as f64;
                                let x = move |k: isize| (w * k as f64 + theta).cos();
                                let (e, c) = moments(&wts, &x, 0);
                                // ρ |c/e|
                                (e, rho * (c / e).abs())
                            })
                            .collect()
                    })
                    .collect();
                (rho, bin.len_unfolded(), families)
            })
            .collect();

        let names = DETUNES_C.map(|c| format!("{c:+.0}c"));
        let names: Vec<&str> = names.iter().map(String::as_str).collect();

        let [median, worst] = transport_report(
            &format!("TONE T_HAT (Q {Q}, γ {GAMMA}, quantum {QUANTUM}, tail {TAIL_DB:.0} dB)"),
            &names,
            &rows,
            [MEDIAN_DB, WORST_DB],
        );

        assert!(
            median < MEDIAN_DB,
            "median row E·d {median:.2} dB re one period"
        );
        assert!(
            worst < WORST_DB,
            "worst row E·d {worst:.2} dB re one period"
        );
    }

    /// Energy that time reassignment moves off a Gaussian burst's center, and how far, per width.
    ///
    ///     ĉ = Σ E t̂ / Σ E
    ///     E·d = Σ E_b ρ |ĉ − p| / Σ E_b
    ///
    /// Reassignment squeezes a burst toward its center by design, so each burst is one reading,
    /// its total reported energy E_b at the displacement of its centroid.
    #[test]
    fn t_hat_burst_transport() {
        const Q: f64 = 3.5;
        const GAMMA: f64 = 3.0;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -40.0;

        const PITCHES: usize = 48;
        const RHO_LO: f64 = 0.01;
        const RHO_HI: f64 = 0.375;
        const HOPS: usize = 64;
        const PHASES: usize = 6;
        /// Burst widths, envelope σ.
        const WIDTHS: [f64; 3] = [0.25, 0.75, 2.0];
        const DETUNES_C: [f64; 3] = [-300.0, 0.0, 300.0];

        const MEDIAN_DB: f64 = -20.0;
        const WORST_DB: f64 = -10.0;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, GAMMA))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        let rows: Vec<_> = (0..PITCHES)
            .map(|i| {
                // ρ_lo (ρ_hi / ρ_lo)^(i / (n − 1))
                let rho = RHO_LO * (RHO_HI / RHO_LO).powf(i as f64 / (PITCHES - 1) as f64);
                let bin = wav.at_rho(rho);
                let wts = bin.weights();
                let w0 = bin.velocity();
                let half = (bin.len_folded() - 1) as f64;
                // P / 2πρ
                let sigma = wav.shape().p() / (TAU * rho);

                let families = WIDTHS
                    .iter()
                    .map(|scale| {
                        let sd = (scale * sigma).max(1.0);
                        let mut readings = Vec::with_capacity(DETUNES_C.len() * PHASES);
                        for cents in DETUNES_C {
                            let w = w0 * (cents / 1200.0).exp2();
                            for q in 0..PHASES {
                                let (theta, p) = (TAU * q as f64 / PHASES as f64, offset(q));
                                // e^{−z²/2} cos(ω(k − p) + θ), z = (k − p) / sd
                                let x = move |k: isize| {
                                    let t = k as f64 - p;
                                    let z = t / sd;
                                    (-0.5 * z * z).exp() * (w * t + theta).cos()
                                };
                                // (Σ E, Σ E t̂), E t̂ = E m + c
                                let (e, et) =
                                    hops(half + 3.0 * sd, HOPS).fold((0.0, 0.0), |(e, et), m| {
                                        let (em, c) = moments(&wts, &x, m);
                                        (e + em, et + em * m as f64 + c)
                                    });
                                // ρ |ĉ − p|
                                readings.push((e, rho * (et / e - p).abs()));
                            }
                        }
                        readings
                    })
                    .collect();
                (rho, bin.len_unfolded(), families)
            })
            .collect();

        let names = WIDTHS.map(|w| format!("b {w}σ"));
        let names: Vec<&str> = names.iter().map(String::as_str).collect();

        let [median, worst] = transport_report(
            &format!("BURST T_HAT (Q {Q}, γ {GAMMA}, quantum {QUANTUM}, tail {TAIL_DB:.0} dB)"),
            &names,
            &rows,
            [MEDIAN_DB, WORST_DB],
        );

        assert!(
            median < MEDIAN_DB,
            "median row E·d {median:.2} dB re one period"
        );
        assert!(
            worst < WORST_DB,
            "worst row E·d {worst:.2} dB re one period"
        );
    }

    /// Energy time reassignment moves off a step's edge, and how far, for the default filter over
    /// γ × Q × tail.
    ///
    ///     E·d = Σ E ρ |t̂ − p| / Σ E
    ///
    /// Far from the edge the step is DC, so Ψ settles to ½M₀ and each leaked reading lands displaced
    /// by its distance to the edge.  The step's 1/ω spectrum weights the low skirt.
    #[test]
    fn t_hat_step_transport() {
        const HOPS: usize = 96;
        const PHASES: usize = 6;
        /// Hop reach in half-spans, past where the edge leaves the window.
        const REACH: f64 = 3.0;
        /// Edge width, samples.
        const EDGE: f64 = 0.5;

        // (γ, Q, tail, [median row, worst row])
        let mut summary = Vec::new();

        for gamma in PITCH_GAMMAS {
            for q in PITCH_QS {
                for tail in PITCH_TAILS {
                    let wav = WaveletSpec::default()
                        .with_shape(Shape::from_q(q, gamma))
                        .max_load_quantum(PITCH_QUANTUM)
                        .max_truncation(tail)
                        .bake();

                    let rows: Vec<_> = PITCH_RHOS
                        .into_iter()
                        .map(|rho| {
                            let bin = wav.at_rho(rho);
                            let wts = bin.weights();
                            let half = (bin.len_folded() - 1) as f64;

                            let mut readings = Vec::with_capacity(PHASES * HOPS);
                            for j in 0..PHASES {
                                let p = offset(j);
                                // ½ tanh((k − p) / w)
                                let x = move |k: isize| 0.5 * ((k as f64 - p) / EDGE).tanh();
                                for m in hops(REACH * half, HOPS) {
                                    let (e, c) = moments(&wts, &x, m);
                                    if e > 0.0 {
                                        // ρ |m + c/e − p|
                                        readings.push((e, rho * (m as f64 + c / e - p).abs()));
                                    }
                                }
                            }
                            (rho, bin.len_unfolded(), vec![readings])
                        })
                        .collect();

                    let measured = transport_report(
                        &format!(
                            "STEP T_HAT (Q {q}, γ {gamma}, quantum {PITCH_QUANTUM}, \
                             tail {tail:.0} dB)"
                        ),
                        &["step"],
                        &rows,
                        [0.0, 0.0],
                    );
                    summary.push((gamma, q, tail, measured));
                }
            }
        }

        let tails = PITCH_TAILS.len();
        let rule = "=".repeat(14 + 18 * tails);
        let header = |lead: &str| {
            print!("  {lead}");
            for tail in PITCH_TAILS {
                print!(" {:>17}", format!("{tail:.0} dB"));
            }
            println!();
        };

        // Worst over γ × Q per tail
        let worst: Vec<[f64; 2]> = (0..tails)
            .map(|t| {
                summary
                    .iter()
                    .skip(t)
                    .step_by(tails)
                    .fold([f64::NEG_INFINITY; 2], |w, r| {
                        [w[0].max(r.3[0]), w[1].max(r.3[1])]
                    })
            })
            .collect();

        println!("\n  {rule}");
        println!("  STEP T_HAT worst over γ × Q, E·d dB re one period");
        println!("  {rule}");
        header(&format!("{:>12}", ""));
        for (label, i) in [("median row", 0), ("worst row", 1)] {
            print!("  {label:>12}");
            for w in &worst {
                print!(" {:>17.2}", w[i]);
            }
            println!();
        }
        println!("  {rule}");

        // Median and worst row E·d by tail, one row per γ × Q
        println!("\n  {rule}");
        println!("  STEP T_HAT E·d dB re one period by tail, median / worst row");
        println!("  {rule}");
        header(&format!("{:>4} {:>5}  ", "γ", "Q"));
        for row in summary.chunks(tails) {
            print!("  {:>4.1} {:>5.1}  ", row[0].0, row[0].1);
            for s in row {
                print!(" {:>17}", format!("{:.2} / {:.2}", s.3[0], s.3[1]));
            }
            println!();
        }
        println!("  {rule}");
    }

    /// The band edge and noise floor a user names, and the bin the wavelet hands back.
    #[test]
    fn noise_floor_is_delivered() {
        const GAMMA: f64 = 3.0;

        /// Where the fold binds.  Below this a workable Q already buries the image.
        const RHOS: [f64; 5] = [0.31, 0.35, 0.38, 0.40, 0.42];
        const FLOORS: [f64; 4] = [-60.0, -70.0, -90.0, -110.0];

        /// Measured under the floor, taps nobody needed.  The restriction kernel at π against ω₀,
        /// which shrinks toward Nyquist.
        // Measured 0.50 at rho 0.31, -110 dB.
        const UNDER_DB: f64 = 1.0;
        // Measured 0.9939 to 0.9998, tightening with Q.
        const WIDTH_TOL: f64 = 0.02;
        const PEAK_C: f64 = 0.5;
        const OVER_DB: f64 = 1.0;
        const GAIN_TOL: f64 = 1e-4;
        const BREACH: f64 = 1.08;

        println!("\n=== NOISE FLOOR DELIVERED (γ {GAMMA}) ===");
        println!(
            "  {:>7} {:>8} {:>6} {:>5} {:>9} {:>9} {:>9} {:>8} {:>9}",
            "rho", "floor", "Q", "taps", "fold", "stop", "width·Q", "peak c", "breach"
        );

        for floor in FLOORS {
            for rho in RHOS {
                let shape = Shape::from_noise_floor(rho, floor, GAMMA);
                let q = shape.q();
                let spec = WaveletSpec::default()
                    .with_shape(shape)
                    .max_noise_floor(floor);

                let wav = spec.max_rho(rho).bake();
                let bin = wav.at_rho(rho);
                let wts = bin.weights();
                let w0 = bin.velocity();
                let r = characterize(wts.psi(), w0);

                let fold = db(r.image) - db(r.gain);
                let stop = db(r.floor) - db(r.gain);
                let peak_c = 1200.0 * (r.peak_w / w0).log2();

                let breach = {
                    let wav = WaveletSpec::default()
                        .with_shape(shape)
                        .max_noise_floor(floor)
                        .bake();
                    let bin = wav.at_rho(rho * BREACH);
                    let r = characterize(bin.weights().psi(), bin.velocity());
                    db(r.image) - db(r.gain)
                };

                println!(
                    "  {rho:>7.4} {floor:>8.1} {q:>6.2} {:>5} {fold:>9.2} {stop:>9.2} {:>9.4} \
                 {peak_c:>+8.3} {breach:>9.2}",
                    bin.len_unfolded(),
                    r.rel_width * q,
                );

                assert!(
                    fold < floor + OVER_DB && fold > floor - UNDER_DB,
                    "rho {rho} floor {floor} fold {fold:.2}"
                );
                assert!(
                    stop < floor + OVER_DB,
                    "rho {rho} floor {floor} stopband {stop:.2}"
                );
                assert!((r.gain - PEAK_GAIN).abs() < GAIN_TOL * PEAK_GAIN);
                assert!(
                    (r.rel_width * q - 1.0).abs() < WIDTH_TOL,
                    "rho {rho} floor {floor} width·Q {:.4}",
                    r.rel_width * q
                );
                assert!(
                    peak_c.abs() < PEAK_C,
                    "rho {rho} floor {floor} peak {peak_c:+.3}c"
                );
                assert!(
                    breach > floor,
                    "rho {:.4} past the ceiling still holds {floor:.1} at {breach:.2}",
                    rho * BREACH
                );
            }
            println!();
        }
    }

    // Pitch transport

    /// 1200 log2(1 + 1/600)
    const PITCH_RES_C: f64 = 2.883;
    /// Share of a filter's energy misplaced before the filter is unusable.
    const PITCH_FAIL: f64 = 0.5;
    /// Share of log ρ held by unusable filters before the wavelet is.
    const WAVELET_FAIL: f64 = 0.1;

    const PITCH_GAMMAS: [f64; 2] = [3.0, 4.0];
    const PITCH_QS: [f64; 2] = [3.5, 12.5];
    const PITCH_TAILS: [f64; 3] = [-40.0, -60.0, -80.0];
    const PITCH_QUANTUM: usize = 4;

    /// Sampled centers, denser where the top of the band degrades.
    const PITCH_RHOS: [f64; 7] = [0.01, 0.03, 0.1, 0.2, 0.28, 0.33, 0.375];

    /// Detunes in half-power half-widths.
    const PITCH_DETUNES: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

    /// 1200 log2(1 + 1/2Q)
    fn half_width_c(q: f64) -> f64 {
        1200.0 * (1.0 + 0.5 / q).log2()
    }

    /// Share of the log ρ span each center stands for, cells split at geometric midpoints.
    fn rho_weights() -> [f64; PITCH_RHOS.len()] {
        let l = PITCH_RHOS.map(f64::ln);
        let last = l.len() - 1;
        core::array::from_fn(|i| {
            let lo = if i == 0 {
                l[0]
            } else {
                0.5 * (l[i - 1] + l[i])
            };
            let hi = if i == last {
                l[last]
            } else {
                0.5 * (l[i] + l[i + 1])
            };
            (hi - lo) / (l[last] - l[0])
        })
    }

    /// (|Ψ|², ω₀ Re(D/Ψ), m + Re(T/Ψ))
    fn reassign(wts: &Weights, x: &dyn Fn(isize) -> f64, m: isize, w0: f64) -> (f64, f64, f64) {
        let [psi, dee, tee] = wts.project(x, m);
        (
            psi.norm_sqr(),
            w0 * (dee / psi).re,
            m as f64 + (tee / psi).re,
        )
    }

    /// min(|1200 log2(ω̂/ω)|, 1200)
    fn miss_c(w_hat: f64, w: f64) -> f64 {
        (1200.0 * (w_hat / w).log2()).abs().min(1200.0)
    }

    /// 10 log10 u, dash where nothing was measured
    fn db_cell(u: f64, width: usize) -> String {
        if u > 0.0 {
            format!("{:>width$.2}", 10.0 * u.log10())
        } else {
            format!("{:>width$}", "—")
        }
    }

    /// Prints one IQM and failed share against their limits, returning whether both hold.
    fn pitch_headline(label: &str, iqm_db: f64, share: f64, iqm_limit: f64) -> bool {
        let verdict = |h: f64| if h >= 0.0 { "pass" } else { "FAIL" };
        let (h_iqm, h_share) = (iqm_limit - iqm_db, WAVELET_FAIL - share);
        let rule = "=".repeat(33);

        println!("\n  {rule}");
        println!("  {label}");
        println!("  {rule}");
        println!("  {:>9} {:>10} {:>10}", "", "IQM dB", "failed");
        println!("  {:>9} {iqm_db:>10.2} {:>9.1}%", "measured", 100.0 * share);
        println!(
            "  {:>9} {iqm_limit:>10.2} {:>9.1}%",
            "limit",
            100.0 * WAVELET_FAIL
        );
        println!(
            "  {:>9} {h_iqm:>10.2} {:>9.1}%",
            "headroom",
            100.0 * h_share
        );
        println!(
            "  {:>9} {:>10} {:>10}",
            "",
            verdict(h_iqm),
            verdict(h_share)
        );
        println!("  {rule}");

        h_iqm >= 0.0 && h_share >= 0.0
    }

    /// Pitch transport over Q × γ × tail × ρ.  `readings` returns one family of
    /// (E, e, e_phys) per column in `names`, e measured from the physical model's truth and
    /// e_phys the model's own departure from the carrier, both in cents.
    ///
    ///     E·d  = Σ E e/res / Σ E
    ///     miss = Σ E [e > res] / Σ E
    ///     IQM  = E·d over the middle two quartiles of surviving filters
    fn pitch_matrix(
        title: &str,
        names: &[String],
        iqm_limits: [f64; PITCH_TAILS.len()],
        readings: impl Fn(Bin<'_>, f64) -> Vec<Vec<(f64, f64, f64)>>,
    ) {
        // Σ E v / Σ E
        let mean = |r: &[(f64, f64, f64)], v: &dyn Fn(&(f64, f64, f64)) -> f64| {
            let (num, e) = r
                .iter()
                .fold((0.0, 0.0), |(n, e), x| (n + x.0 * v(x), e + x.0));
            num / e
        };
        let ed = |r: &[(f64, f64, f64)]| mean(r, &|x| x.1 / PITCH_RES_C);
        let phys = |r: &[(f64, f64, f64)]| mean(r, &|x| x.2 / PITCH_RES_C);
        let miss = |r: &[(f64, f64, f64)]| mean(r, &|x| (x.1 > PITCH_RES_C) as u8 as f64);
        let db = |u: f64| {
            if u > 0.0 {
                10.0 * u.log10()
            } else {
                f64::INFINITY
            }
        };

        let weights = rho_weights();
        // (γ, Q, tail, IQM dB, failed share, [q1, q2, q3], phys median)
        let mut summary = Vec::new();
        let mut failures = Vec::new();

        for gamma in PITCH_GAMMAS {
            for q in PITCH_QS {
                for (t, tail) in PITCH_TAILS.into_iter().enumerate() {
                    let wav = WaveletSpec::default()
                        .with_shape(Shape::from_q(q, gamma))
                        .max_load_quantum(PITCH_QUANTUM)
                        .max_truncation(tail)
                        .bake();

                    // (ρ, w, taps, per family E·d, pooled readings)
                    let rows: Vec<_> = PITCH_RHOS
                        .into_iter()
                        .zip(weights)
                        .map(|(rho, w)| {
                            let bin = wav.at_rho(rho);
                            let families = readings(bin, q);
                            let per: Vec<f64> = families.iter().map(|f| ed(f)).collect();
                            let pooled: Vec<(f64, f64, f64)> = families
                                .iter()
                                .flat_map(|f| {
                                    let total: f64 = f.iter().map(|v| v.0).sum();
                                    f.iter().map(move |&(e, c, p)| (e / total, c, p))
                                })
                                .collect();
                            (rho, w, bin.len_unfolded(), per, pooled)
                        })
                        .collect();
                    let has_phys = rows.iter().any(|r| r.4.iter().any(|v| v.2 > 0.0));

                    println!(
                        "\n=== {title} (Q {q}, γ {gamma}, quantum {PITCH_QUANTUM}, \
                         tail {tail:.0} dB, res {PITCH_RES_C}c) ==="
                    );
                    println!("  E·d in dB re one resolution per family and pooled");
                    if has_phys {
                        println!("  phys is the model's own pull, excluded from E·d");
                    }
                    println!("  miss is E beyond res, q50 and q99 are e in cents\n");
                    print!("  {:>7} {:>5}", "rho", "taps");
                    for n in names {
                        print!(" {n:>8}");
                    }
                    print!(" {:>8}", "pooled");
                    if has_phys {
                        print!(" {:>8}", "phys");
                    }
                    println!(" {:>7} {:>8} {:>8}", "miss", "q50 c", "q99 c");

                    // (w, E·d, phys, failed)
                    let mut stats = Vec::with_capacity(rows.len());
                    for (rho, w, taps, per, pooled) in &rows {
                        let (row_ed, row_phys, row_miss) = (ed(pooled), phys(pooled), miss(pooled));
                        let failed = row_miss > PITCH_FAIL;
                        let mut spread: Vec<(f64, f64)> =
                            pooled.iter().map(|&(e, c, _)| (e, c)).collect();

                        print!("  {rho:>7.4} {taps:>5}");
                        for f in per {
                            print!(" {}", db_cell(*f, 8));
                        }
                        print!(" {}", db_cell(row_ed, 8));
                        if has_phys {
                            print!(" {}", db_cell(row_phys, 8));
                        }
                        println!(
                            " {:>6.2}% {:>8.3} {:>8.3} {}",
                            100.0 * row_miss,
                            weighted_quantile(&mut spread, 0.5),
                            weighted_quantile(&mut spread, 0.99),
                            if failed { "FAIL" } else { "" }
                        );
                        stats.push((*w, row_ed, row_phys, failed));
                    }

                    // Survivors, weighted by log ρ share
                    let share = stats.iter().filter(|r| r.3).fold(0.0, |s, r| s + r.0);
                    let mut eds: Vec<(f64, f64)> =
                        stats.iter().filter(|r| !r.3).map(|r| (r.0, r.1)).collect();
                    let quartiles = if eds.is_empty() {
                        [f64::NAN; 3]
                    } else {
                        [0.25, 0.5, 0.75].map(|f| weighted_quantile(&mut eds, f))
                    };
                    // Σ w E·d / Σ w over q1 ≤ E·d ≤ q3
                    let (num, den) = eds
                        .iter()
                        .filter(|r| r.1 >= quartiles[0] && r.1 <= quartiles[2])
                        .fold((0.0, 0.0), |(n, d), r| (n + r.0 * r.1, d + r.0));
                    let iqm = db(num / den);
                    let mut physs: Vec<(f64, f64)> = stats.iter().map(|r| (r.0, r.2)).collect();
                    let phys_med = weighted_quantile(&mut physs, 0.5);

                    println!(
                        "\n  E·d quartiles over survivors  q1 {}  median {}  q3 {}",
                        db_cell(quartiles[0], 0),
                        db_cell(quartiles[1], 0),
                        db_cell(quartiles[2], 0),
                    );
                    let label = format!("Q {q} γ {gamma} tail {tail:.0}");
                    if !pitch_headline(&label, iqm, share, iqm_limits[t]) {
                        failures.push(label);
                    }
                    summary.push((gamma, q, tail, iqm, share, quartiles, phys_med));
                }
            }
        }

        // IQM by tail, one row per γ × Q
        // Worst over γ × Q per tail
        let tails = PITCH_TAILS.len();
        let rule = "=".repeat(14 + 16 * tails);
        let worst: Vec<(f64, f64)> = (0..tails)
            .map(|t| {
                summary
                    .iter()
                    .skip(t)
                    .step_by(tails)
                    .fold((f64::NEG_INFINITY, 0.0f64), |(i, s), r| {
                        (i.max(r.3), s.max(r.4))
                    })
            })
            .collect();
        let verdict = |h: f64| if h >= 0.0 { "pass" } else { "FAIL" };
        let header = |lead: &str| {
            print!("  {lead:>12}");
            for tail in PITCH_TAILS {
                print!(" {:>15}", format!("{tail:.0} dB"));
            }
            println!();
        };
        let line = |label: &str, cell: &dyn Fn(usize) -> String| {
            print!("  {label:>12}");
            for t in 0..tails {
                print!(" {:>15}", cell(t));
            }
            println!();
        };

        println!("\n  {rule}");
        println!("  {title} worst over γ × Q");
        println!("  {rule}");
        header("");
        line("limit", &|t| format!("{:.2}", iqm_limits[t]));
        line("headroom", &|t| {
            format!("{:.2}", iqm_limits[t] - worst[t].0)
        });
        line("verdict", &|t| {
            let h = (iqm_limits[t] - worst[t].0).min(WAVELET_FAIL - worst[t].1);
            verdict(h).to_string()
        });
        println!("  {rule}");

        // IQM by tail, one row per γ × Q
        println!("\n  {rule}");
        println!("  {title} IQM dB re {PITCH_RES_C}c by tail, failed share where nonzero");
        println!("  {rule}");
        print!("  {:>4} {:>5}  ", "γ", "Q");
        for tail in PITCH_TAILS {
            print!(" {:>15}", format!("{tail:.0} dB"));
        }
        println!();
        for row in summary.chunks(tails) {
            print!("  {:>4.1} {:>5.1}  ", row[0].0, row[0].1);
            for s in row {
                let cell = match s.4 > 0.0 {
                    true => format!("{:.2} ({:.0}%)", s.3, 100.0 * s.4),
                    false => format!("{:.2}", s.3),
                };
                print!(" {cell:>15}");
            }
            println!();
        }
        println!("  {rule}");

        assert!(
            failures.is_empty(),
            "{title} failed wavelets: {}",
            failures.join(", ")
        );
    }

    /// Pitch misplaced off a stationary tone across the band a filter answers for.
    ///
    ///     e = 1200 log2(ω̂ / ω)
    #[test]
    fn r_hat_tone_transport() {
        const PHASES: usize = 12;
        // Empirical.  Keep updated.  Maps to the truncation dB.
        const IQM_DB: [f64; 3] = [-12.0, -18.0, -29.0];

        let names = PITCH_DETUNES.map(|s| format!("{s:+.1}h"));
        pitch_matrix("TONE R_HAT", &names, IQM_DB, |bin, q| {
            let (wts, w0) = (bin.weights(), bin.velocity());
            PITCH_DETUNES
                .iter()
                .map(|s| {
                    // ω₀ 2^(s h / 1200)
                    let w = w0 * (s * half_width_c(q) / 1200.0).exp2();
                    (0..PHASES)
                        .map(|p| {
                            let theta = TAU * p as f64 / PHASES as f64;
                            let x = move |k: isize| (w * k as f64 + theta).cos();
                            let (e, w_hat, _) = reassign(&wts, &x, 0, w0);
                            (e, miss_c(w_hat, w), 0.0)
                        })
                        .collect()
                })
                .collect()
        });
    }

    /// Pitch misplaced off a linear chirp, read at its reassigned time.
    ///
    ///     e = 1200 log2(ω̂ / (ω₀ + a t̂))
    ///
    /// Rates are c = a σ², σ² the envelope variance in samples.
    #[test]
    fn r_hat_chirp_transport() {
        const RATES: [f64; 3] = [0.05, 0.2, 0.5];
        const READINGS: usize = 32;
        const PHASES: usize = 6;
        const IQM_DB: [f64; 3] = [-5.0, -15.0, -25.0];

        let names = RATES.map(|c| format!("aσ² {c}"));
        pitch_matrix("CHIRP R_HAT", &names, IQM_DB, |bin, q| {
            let (wts, w0) = (bin.weights(), bin.velocity());
            let var = wts.psi().envelope_var();
            let span = half_width_c(q);
            RATES
                .iter()
                .map(|c| {
                    let a = c / var;
                    let mut r = Vec::with_capacity(READINGS * PHASES);
                    for j in 0..READINGS {
                        // ω₀ 2^(δ/1200) = ω₀ + a m
                        let detune = span * (2.0 * j as f64 / (READINGS - 1) as f64 - 1.0);
                        let m = (w0 * ((detune / 1200.0).exp2() - 1.0) / a).round() as isize;
                        for p in 0..PHASES {
                            let theta = TAU * p as f64 / PHASES as f64;
                            // φ(k) = ω₀k + ½ak² + θ
                            let x = move |k: isize| {
                                let k = k as f64;
                                (w0 * k + 0.5 * a * k * k + theta).cos()
                            };
                            let (e, w_hat, t_hat) = reassign(&wts, &x, m, w0);
                            r.push((e, miss_c(w_hat, w0 + a * t_hat), 0.0));
                        }
                    }
                    r
                })
                .collect()
        });
    }

    /// Pitch misplaced off Gaussian bursts at every hop they reach, measured from the product
    /// centroid a Gaussian envelope pair would reassign to.
    ///
    ///     ω* = (ω σ_x² + ω₀ σ_ψ²) / (σ_x² + σ_ψ²)
    ///     e  = 1200 log2(ω̂ / ω*)
    ///     e_phys = 1200 log2(ω* / ω)
    #[test]
    fn r_hat_burst_transport() {
        /// Burst widths, filter envelope σ.
        const WIDTHS: [f64; 3] = [1.0, 2.0, 4.0];
        const HOPS: usize = 32;
        const PHASES: usize = 4;
        // XXX Would set limits here, but the "worst" is always a Q = 3.5 row that is probably just
        // barely big enough to locate the burst.  With some more effort I'm sure something will
        // become apparent.
        const IQM_DB: [f64; 3] = [0.0; 3];

        let names = WIDTHS.map(|w| format!("b {w}σ"));
        pitch_matrix("BURST R_HAT", &names, IQM_DB, |bin, q| {
            let (wts, w0) = (bin.weights(), bin.velocity());
            let half = (bin.len_folded() - 1) as f64;
            let var = wts.psi().envelope_var();
            WIDTHS
                .iter()
                .map(|scale| {
                    let sd = (scale * var.sqrt()).max(1.0);
                    let sd2 = sd * sd;
                    let mut r = Vec::new();
                    for s in PITCH_DETUNES {
                        let w = w0 * (s * half_width_c(q) / 1200.0).exp2();
                        // ω*
                        let w_model = (w * sd2 + w0 * var) / (sd2 + var);
                        let phys = miss_c(w_model, w);
                        for j in 0..PHASES {
                            let (theta, p) = (TAU * j as f64 / PHASES as f64, offset(j));
                            // e^{−z²/2} cos(ω(k − p) + θ), z = (k − p) / sd
                            let x = move |k: isize| {
                                let t = k as f64 - p;
                                let z = t / sd;
                                (-0.5 * z * z).exp() * (w * t + theta).cos()
                            };
                            for m in hops(half + 3.0 * sd, HOPS) {
                                let (e, w_hat, _) = reassign(&wts, &x, m, w0);
                                r.push((e, miss_c(w_hat, w_model), phys));
                            }
                        }
                    }
                    r
                })
                .collect()
        });
    }
}
