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
//! This module generates families of wavelets for use in wavelet tables.  Our wavelets are Morse
//! family:
//!
//! - Easy to generate
//! - Regarded as nice for time and frequency reassignment
//! - Parameterized (but only a little!)
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
//!     .truncate(-140.0)
//!     .bake();
//! ```
//!
//! Resolve a bin with [`Wavelet::bin`] and realize its folded taps.  Bins may use a tighter load
//! quantum but may not exceed the [`Wavelet`].
//!
//! ```
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! # let wavelet = WaveletSpec::default().q(5.5).truncate(-140.0).bake();
//!
//! let bin = wavelet.bin(440.0, 48_000.0).truncate(-100.0);
//!
//! // The `Bin` can be used to calculate allocation sizes.
//! let mut taps = vec![[0.0f32; 4]; bin.folded_taps()];
//! bin.taps_into(&mut taps);
//!
//! // float4(Re ψ, Im ψ, Re d, Im d); index 0 is the real center tap.
//! assert!(taps[0][0].im == taps[0][2].im == 0.0);
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
//! # let wavelet = WaveletSpec::default().q(5.5).truncate(-140.0).bake();
//!
//! let bin = wavelet.bin(240.0, 3_000.0)
//!     .max_load_quantum(32);
//!
//! let mut taps = vec![[0.0f32; 4]; bin.folded_taps()];
//! let wrote = bin.taps_into(&mut taps);
//! assert!(wrote % 32 == 0);
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

#![warn(warnings, dead_code, unused_variables)]

// 🤖 Heavy generation.  Should be pretty standard academic stuff, so not expecting a lot of
// surprises.  We will, for the most part, swiftly and knowingly eat shit if the wavelet is busted.
// Well-formalized stuff doesn't have a lot of wiggle room to violate the consistency of the
// formalism.

// NEXT A ton of the characterization gear for testing belongs in the dsp module.
// NEXT Transient behavior evaluation to look for negative frequency response under impure tones.
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
// about 220ms on a Zen2+ part in release.  This affects CWT startup time.

// === TABLE RESPONSE (Q = 3.5, quantum 8) ===
//
// fc  1000 sr  6000  w0 1.047198  quantized  25 (unfolded  49)
//   peak gain 2.000000008  dev +3.910e-9 rel
//   width 0.99697
//   peak -0.0000 cents
//   image max         -109.59 dB
//   stopband floor     -109.20 dB
//
// fc   250 sr  3000  w0 0.523599  quantized  49 (unfolded  97)
//   peak gain 1.999999991  dev -4.253e-9 rel
//   width 0.99797
//   peak -0.0000 cents
//   image max         -107.93 dB
//   stopband floor     -106.56 dB
//
// fc 12000 sr 48000  w0 1.570796  quantized  17 (unfolded  33)
//   peak gain 2.000000009  dev +4.465e-9 rel
//   width 0.99531
//   peak -0.0000 cents
//   image max         -111.14 dB
//   stopband floor     -110.96 dB

pub(self) mod generate;
pub(self) mod inspect;
pub(self) mod restrict;
pub(self) mod spec;
pub mod whatsleft;

#[cfg(test)]
mod harness;

use core::f64::consts::{LN_2, PI, TAU};

use num_complex::{Complex32, Complex64};

use generate::hermite;
use inspect::{Inspect, Sample, OVERSAMPLE};
pub use spec::{BinSpec, Shape, WaveletSpec};

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

    // NEXT `bin_with_spec` method.
    // XXX Without debug checks against requesting a `BinSpec` that exceeds maxima, we are letting
    // bad behavior through even runtime.

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

    fn linear(&self, u: f64) -> Complex64 {
        let x = u / self.du;
        let i = x as usize;
        let f = x - i as f64;
        self.psi[i] * (1.0 - f) + self.psi[i + 1] * f
    }

    /// ψ_T at the upper edge of cell j, zero past the cut
    fn edge(&self, j: usize, k: usize, rho: f64) -> Complex64 {
        if j + 1 < k {
            self.at((j as f64 + 0.5) * rho)
        } else {
            Complex64::default()
        }
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
        let rho = spec.center / spec.rate;
        let reach = (wavelet.shape.truncation_u(spec.tail_db) / rho).ceil() as usize;
        let half = (reach + spec.delay).div_ceil(spec.load_quantum) * spec.load_quantum;

        Bin {
            wavelet,
            spec,
            rho,
            k: half + 1,
        }
    }

    pub fn load_quantum(self, load_quantum: usize) -> Self {
        Self::new(
            self.wavelet,
            BinSpec {
                load_quantum,
                ..self.spec
            },
        )
    }

    pub fn delay(self, delay: usize) -> Self {
        Self::new(self.wavelet, BinSpec { delay, ..self.spec })
    }

    pub fn truncate(self, tail_db: f64) -> Self {
        let tail_db = -tail_db.abs();
        Self::new(
            self.wavelet,
            BinSpec {
                tail_db,
                ..self.spec
            },
        )
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

    /// Number of folded taps, including a center tap.  In [0, K].
    pub fn folded_taps(&self) -> usize {
        self.k
    }

    /// Number of real taps after unfolded, in [0, 2K - 1].  Center tap is still just one tap.
    pub fn unfolded_taps(&self) -> usize {
        2 * self.k - 1
    }

    /// ψ and d for this bin, normalized at the measured crest of |H|.
    ///
    ///     H(ω_peak) = PEAK_GAIN
    ///     H_d(ω) ≈ (ω/ω₀)·H(ω)
    pub fn write(&self, bake: &mut Bake) {
        let (wav, rho, w0) = (self.wavelet, self.rho, self.velocity());
        let Bake { weights: w, probe } = bake;

        w.resize(self.k);
        wav.restriction.psi_into(wav.grid(), rho, &mut w.psi);
        debug_assert!(
            w.psi.iter().all(|p| p.is_finite()),
            "restriction produced a non-finite weight at rho {rho}"
        );

        // ω_peak and H(ω_peak)
        let (peak, gain) = Inspect::new(w.psi(), probe, OVERSAMPLE)
            .peak(w0, 0.0, PI)
            .unwrap_or((w0, w.psi().dtft(w0)));

        let scale = PEAK_GAIN / gain;
        for p in w.psi.iter_mut() {
            *p *= scale;
        }

        // XXX This needs to not presume any favored omega
        restrict::derivative_into(&w.psi, peak / TAU, &mut w.d);
    }

    /// Upstream owes an `out` at least `folded_taps()` long.
    pub fn taps_into(&self, out: &mut [[f32; 4]]) -> usize {
        let mut bake = Bake::default();
        self.write(&mut bake);
        bake.weights.pack_into(out)
    }

    pub fn taps(&self) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; self.k];
        self.taps_into(&mut out);
        out
    }
}

/// Folded ψ and d for one bin.
#[derive(Default)]
pub struct Weights {
    psi: Vec<Complex64>,
    d: Vec<Complex64>,
}

impl Weights {
    pub(super) fn psi(&self) -> Fold<'_> {
        Fold::new(&self.psi)
    }

    pub(super) fn d(&self) -> Fold<'_> {
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
/// ψ over [0, K), the mirror ψ₋ₖ = conj ψₖ implied.  Entry 0 is real.
// Making the channel first class could prevent some kinds of mishandling
#[derive(Clone, Copy)]
pub(super) struct Fold<'a>(&'a [Complex64]);

impl<'a> Fold<'a> {
    pub(super) fn new(psi: &'a [Complex64]) -> Self {
        Fold(psi)
    }

    /// 2K − 1
    pub(super) fn taps(&self) -> usize {
        2 * self.0.len() - 1
    }

    /// ψ₀ + 2 Σ_{k≥1} Re(ψ_k e^{−iωk})
    pub(super) fn dtft(&self, w: f64) -> f64 {
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
    pub(super) fn moment(&self, p: i32) -> Complex64 {
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

    /// Σ_ν |ψ_ν|
    pub(super) fn l1(&self) -> f64 {
        self.0[0].norm() + 2.0 * self.0[1..].iter().map(|h| h.norm()).sum::<f64>()
    }

    /// Variance of the magnitude envelope.
    ///
    /// Σ ν²|ψ_ν| / Σ |ψ_ν|
    pub(super) fn envelope_var(&self) -> f64 {
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

    /// Envelope under a fixed carrier.
    ///
    /// a_k = Re(ψ_k e^{−2πikρ})
    pub(super) fn demodulate(&self, rho: f64) -> impl Iterator<Item = f64> + '_ {
        self.0.iter().enumerate().map(move |(j, p)| {
            let (s, c) = (TAU * j as f64 * rho).sin_cos();
            p.re * c + p.im * s
        })
    }

    /// ψ over [−K, K), for display.
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
    use super::*;

    use harness::*;

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
        const QUANTUM: usize = 4;
        const STEP: f64 = 100.0;
        const SPAN: isize = 4;
        /// Worst reassignment bias over the scan, in cents.
        const BIAS_C: f64 = 2.0;

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
                .bake();
            let bin = wav.at_rho(1000.0 / 8000.0);
            let w0 = bin.velocity();
            let weights = bin.taps();

            let psi = unfold(&weights, 0);
            let d = unfold(&weights, 2);
            let n = psi.len();

            // M₁ / M₀
            let delay = (moment(&psi, 1) / moment(&psi, 0)).re;

            // worst over detuning of |R| / ‖ψ‖₁ and of (1200/ln 2)·Re(R/Ψ̂)/r
            let (floor, worst) = (-SPAN..=SPAN).fold((0.0f64, 0.0f64), |acc, k| {
                let ratio = (k as f64 * STEP / 1200.0).exp2();
                let (res, dr) = pairing_residual(&psi, &d, w0, w0 * ratio);
                (
                    acc.0.max(res / l1(&psi)),
                    acc.1.max((1200.0 / LN_2 * dr / ratio).abs()),
                )
            });

            println!(
                "\ngamma = {gamma:.1}  weights {}  taps {n}  delay = {delay:+.3e}  \
             floor = {}  bias = {worst:.3}c",
                weights.len(),
                fmt_e(floor),
            );

            let mags: Vec<f64> = psi.iter().map(|h| h.norm() as f64).collect();
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
        const TAIL_DB: f64 = -50.0;

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
        const NOISE_GAIN: f64 = 1.4105;
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

            assert!(
                (ratio / NOISE_GAIN - 1.0).abs() < TOL,
                "fc {fc} noise gain {ratio:.6}"
            );
        }
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

        for (fc, sr) in [
            (2_000.0f64, RATE),
            (200.0, 3000.0),
            (250.0, 3000.0),
            (12_000.0, RATE),
        ] {
            let bin = wav.at_rho(fc / sr);
            let taps = bin.taps();
            let psi = unfold(&taps, 0);

            let (n, w0) = (psi.len(), bin.velocity());
            println!("  fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6}");

            for k in -SPAN..=SPAN {
                let cents = k as f64 * STEP;
                let wd = w0 * (cents / 1200.0).exp2();
                let h = dtft(&psi, wd).norm();
                let h_db = 20.0 * (h / PEAK_GAIN).log10();
                if h_db < GATE_DB {
                    continue;
                }

                let [sp, sd, st] = tone_response(&taps, w0, cents, RESOLUTION);
                let img = dtft(&psi, -wd).norm();

                println!(
                    "  {cents:+6.0}c {h_db:>7.1} {:>8.1} {sp:>11.2e} {sd:>11.2e} {st:>11.2e}",
                    20.0 * (img / PEAK_GAIN).log10(),
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
        const Q: f64 = 8.5;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -60.0;

        const SPACING: f64 = 100.0;
        const STEPS: isize = 8;
        const RESOLUTION: f64 = 0.05;

        /// Worst per hop error in cents, bias plus the swing across carrier phase.
        const ERROR_C: f64 = 0.5;

        const SKIRT_DB: f64 = -30.0;
        const SKIRT_C: f64 = 10.0;
        /// Reach of the crossing search, an octave either side.
        const SPAN: f64 = 1.0;

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();

        println!("\n=== REASSIGN ===");

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
            println!("  fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6}");
            println!("  detune      |H|     pred      bias     swing      leak");

            for k in -STEPS..=STEPS {
                let cents = k as f64 * 0.5 * SPACING / STEPS as f64;
                // 2^(c/1200)
                let ratio = (cents / 1200.0).exp2();
                let h = dtft(&psi, w0 * ratio).norm();

                let ((bias, swing), (leak, _)) = tone_bias(&taps, w0, cents, RESOLUTION);
                let (_, dr) = pairing_residual(&psi, &d, w0, w0 * ratio);
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

            let peak = dtft(&psi, w0).norm();
            let (lo, hi) = shoulders(&psi, peak, w0, SKIRT_DB, w0 * SPAN);

            for w in [lo, hi].into_iter().flatten() {
                // 1200 log2(ω/ω₀)
                let cents = 1200.0 * (w / w0).log2();
                let ((bias, swing), _) = tone_bias(&taps, w0, cents, RESOLUTION);
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
            println!("  fc {fc:.0} sr {sr:.0} taps {n} w0 {w0:.6}");

            for k in -SPAN..=SPAN {
                let cents = k as f64 * STEP;
                let wd = w0 * (cents / 1200.0).exp2();
                let h = dtft(&psi, wd).norm();
                let h_db = 20.0 * (h / PEAK_GAIN).log10();
                if h_db < GATE_DB {
                    continue;
                }

                let ((_, swing), (quad, quad_swing)) = tone_bias(&taps, w0, cents, RESOLUTION);

                // H_ψ(±ω), H_d(±ω)
                let (p, n) = (dtft(&psi, wd), dtft(&psi, -wd));
                let (q, m) = (dtft(&d, wd), dtft(&d, -wd));
                // α = H_d(−ω)/H_d(ω) − H_ψ(−ω)/H_ψ(ω)
                let alpha = m / q - n / p;
                // (1200 / ln 2) · |α|
                let pred = 1200.0 / LN_2 * alpha.norm();

                println!(
                    "  {cents:+6.0}c {h_db:>7.1} {:>8.1} {:>8.1} {pred:>8.3}c {swing:>8.3}c {:>9.3}",
                    20.0 * (n.norm() / PEAK_GAIN).log10(),
                    20.0 * (m.norm() / PEAK_GAIN).log10(),
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
            let taps = full.taps();
            let psi64 = lane(&taps, 0);
            let pf = unfold(&taps, 0);

            let w0 = full.velocity();
            let rf = characterize(Fold(&psi64), w0);
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

            // Peak sits below ω₀ by the cell-average droop, gain rising as ½P²Δx².
            assert!(
                (rf.gain - 2.0).abs() < 1e-5,
                "fc {fc} full gain {:.9}",
                rf.gain
            );
            assert!(dcf < 1e-5 * PEAK_GAIN, "fc {fc} full dc {:.2} dB", db(dcf));

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
                let taps = cut.taps();
                let psi64 = lane(&taps, 0);
                let pc = unfold(&taps, 0);
                let rc = characterize(Fold(&psi64), w0);
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
                if tail_db.abs() > 20.0 {
                    assert!(
                        (rc.rel_width / rf.rel_width - 1.0).abs() < 0.02,
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

        const FULL_DB: f64 = -140.0;
        const CUTS: [f64; 4] = [-40.0, -60.0, -80.0, -100.0];
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
        const TAIL_DB: f64 = -40.0;

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
            assert!(dc.abs() < 1e-2 * g, "fc {fc} dc {dc:.3e}");

            // d/ψ reads ω/ω₀, unity at the carrier.
            let gd = dtft(&d, w0).norm();
            assert!((gd / g - 1.0).abs() < 1e-3, "fc {fc} d/psi {:.6}", gd / g);

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

            // XXX pretty loose!

            // measured: fc 1000 first moment -1.123e-7
            let m1 = mom(1);
            assert!(m1.abs() < 1e-1 * g, "fc {fc} first moment {m1:.3e}");

            // H''(0) and H'''(0), the two the solve nulls that nothing else measures.
            let (m2, m3) = (mom(2), mom(3));
            assert!(m2.abs() < 100.0 * g, "fc {fc} second moment {m2:.3e}");
            assert!(m3.abs() < 100.0 * g, "fc {fc} third moment {m3:.3e}");
        }
    }

    /// t̂ against an untruncated bake, on transients short enough that the estimator has to
    /// actually integrate the envelope.  Swept from near-impulsive to comparable to the
    /// wavelet's own support, which is where the pull toward the hop takes over.
    #[test]
    fn t_hat_survives_transients() {
        /// Reference tail, past which f32 storage zeroes the taps anyway.
        const REF_TAIL_DB: f64 = -100.0;
        /// Second reference, confirming the first has converged.
        const DEEP_TAIL_DB: f64 = -140.0;
        const TAIL_DB: f64 = -40.0;

        /// Burst widths in units of the filter's own envelope σ.
        const WIDTHS: [f64; 3] = [0.25, 0.75, 2.0];

        /// Reference movement between the two tails, as a fraction of the measured error.
        const CONVERGED: f64 = 0.05;

        /// Worst |t̂ − t̂_ref| in burst sd, per level bucket.
        const TOL: [f64; 3] = [0.1, 1.2, 2.0];
        /// Skew against Δω·σ_c², bounding Morse departure from the Gaussian model.
        const SKEW_TOL: f64 = 1.1;
        /// Leakage floor in burst sd where the model predicts no skew.
        const LEAK_TOL: f64 = 0.1;

        let wav = WaveletSpec::default().max_truncation(TAIL_DB).bake();
        let long = WaveletSpec::default().max_truncation(REF_TAIL_DB).bake();
        let deep = WaveletSpec::default().max_truncation(DEEP_TAIL_DB).bake();

        for (fc, sr) in [(40.0f64, 3000.0), (200.0, 3000.0), (800.0, 3000.0)] {
            let rho = fc / sr;
            let bin = wav.at_rho(rho);
            let taps = bin.taps();
            let psi_taps = unfold(&taps, 0);
            let reference = long.at_rho(rho).taps();
            let deeper = deep.at_rho(rho).taps();
            let w0 = bin.velocity();
            let half = (taps.len() - 1) as isize;
            // σ² of |ψ|
            let var = envelope_var(&psi_taps);
            let sigma = var.sqrt();

            println!(
                "\n=== T_HAT fc {fc:.0} sr {sr:.0} taps {} ref {} sigma {sigma:.1} ===",
                2 * taps.len() - 1,
                2 * reference.len() - 1,
            );
            println!("  err in burst sd, skew against Δω·σ_c², drift against the -60 dB bucket\n");
            println!(
                "  {:>5} {:>7} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
                "width", "sd", "detune", "-20 dB", "-40 dB", "-60 dB", "drift", "skew"
            );

            for scale in WIDTHS {
                let sd = (scale * sigma).max(1.0);
                for detune in [0.0f64, 400.0] {
                    // ω₀ · 2^(c/1200)
                    let w = w0 * (detune / 1200.0).exp2();
                    // reach plus four burst σ, so the tails clear the level buckets
                    let span = 2 * half + (4.0 * sd).ceil() as isize;
                    let (err, skew) = t_hat_profile(&taps, &reference, burst(w, sd, 0.0), span);
                    let (drift, _) = t_hat_profile(&reference, &deeper, burst(w, sd, 0.0), span);

                    // Δω·σ_c², σ_c² = (σ_ψ⁻² + σ_x⁻²)⁻¹
                    let dw = w0 * ((detune / 1200.0).exp2() - 1.0);
                    let pred = dw * (var.recip() + (sd * sd).recip()).recip();

                    // error in units of the burst being located
                    let rel = |v: f64| v / sd;
                    // skew against the model, or the leakage floor where the model is zero
                    let s = if pred > 0.0 {
                        format!("{:>8.3}", skew / pred)
                    } else {
                        format!("{:>8.1e}", skew / sd)
                    };

                    println!(
                        "  {scale:>4.2}σ {sd:>7.2} {detune:>5.0}c {:>8.3} {:>8.3} {:>8.3} {:>8.1e} {s}",
                        rel(err[0]),
                        rel(err[1]),
                        rel(err[2]),
                        drift[2] / err[2].max(1e-9),
                    );

                    for (e, tol) in err.iter().zip(&TOL) {
                        let gap = rel(*e);
                        assert!(
                            gap < *tol,
                            "fc {fc} sd {sd:.2} detune {detune} t_hat gap {gap:.3} sd"
                        );
                    }

                    assert!(
                        drift[2] < CONVERGED * err[2].max(1e-3),
                        "fc {fc} sd {sd:.2} detune {detune} reference moves {:.4} against err {:.4}",
                        drift[2], err[2]
                    );

                    // a burst at the floor resolves too few samples for σ_x to mean anything
                    if pred > 0.0 {
                        // a burst at the floor resolves too few samples for σ_x to mean anything
                        assert!(
                            sd < 2.0 || (skew / pred - 1.0).abs() < SKEW_TOL - 1.0,
                            "fc {fc} sd {sd:.2} skew off model by {:.3}",
                            skew / pred - 1.0
                        );
                    } else {
                        assert!(
                            skew / sd < LEAK_TOL,
                            "fc {fc} sd {sd:.2} leakage {:.3e} sd",
                            skew / sd
                        );
                    }
                }
            }
        }
    }

    /// Rough magnitude response, centered on the measured peak.  The sweep names ρ directly, so
    /// a row is a filter and not a sample rate.
    #[test]
    fn print_response() {
        const Q: f64 = 40.0;
        const QUANTUM: usize = 4;
        const TAIL_DB: f64 = -60.0;

        const ROWS: usize = 200;
        const COLS: usize = 200;
        const ANTI_ALIAS: usize = 8;
        const FLOOR_DB: f64 = -100.0;
        const LOBES: f64 = 96.0;

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
            let taps = &bin.taps();
            let psi = unfold(taps, 0);
            let psi64 = lane(taps, 0);
            let w0 = bin.velocity();
            let r = characterize(Fold(&psi64), w0);

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
        const TAIL_DB: f64 = -65.0;

        let w = wavelet(Q, 16);

        for quantum in [2usize, 4, 8, 16] {
            println!("\n=== TABLE RESPONSE (Q = {Q}, quantum {quantum}) ===");

            for (fc, sr) in [(1000.0f64, 6000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
                let bin = w.bin(fc, sr).load_quantum(quantum).truncate(TAIL_DB);
                let taps = &bin.taps();
                let psi = unfold(taps, 0);
                let psi64 = lane(taps, 0);

                let w0 = bin.velocity();
                let r = characterize(Fold(&psi64), w0);
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

                println!("  width {:.5}", r.rel_width * Q);
                println!("  peak {:+.4} cents", 1200.0 * (r.peak_w / w0).log2());
                println!("  image max        {:>8.2} dB", db(r.image));
                println!("  stopband floor    {:>8.2} dB", db(r.floor));
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
                    let taps = &bin.taps();
                    let psi = unfold(taps, 0);
                    let psi64 = lane(taps, 0);
                    let r = characterize(Fold(&psi64), bin.velocity());
                    let lobe = r.edges.1 - r.edges.0;
                    let db = |h: f64| 20.0 * (h / r.gain).log10();

                    // 16ε Σ|h|
                    let null = 16.0 * f64::EPSILON * l1(&psi);

                    let lo = first_null(&psi, r.edges.0, r.edges.0 - PI, null)
                        .unwrap_or_else(|| panic!("Q {q} γ {gamma} ρ {rho} lower dip not found"));
                    let hi = first_null(&psi, r.edges.1, r.edges.1 + PI, null)
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

    #[test]
    fn print_chirp_transform() {
        // Make a little plot similar to the one show in this paper:
        // https://jmlilly.net/papers/lilly09-itsp-cp.pdf
        //
        // The paper uses a beta and gamma we can't quite draw (betas this low are not generally
        // supported yet), but our outputs are somewhat similar at relevant beta.

        const TAIL_DB: f64 = -40.0;
        const SPAN: isize = 400;
        const SD: f64 = 90.0;
        const ROWS: usize = 40;

        // ρ of the top and bottom rows, the ridge reaching ρ_top at the window edge
        const RHO_TOP: f64 = 0.10;
        const RHO_BOT: f64 = 0.002;

        let wav = WaveletSpec::default()
            .with_shape(Shape {
                beta: 3.0,
                gamma: 3.0,
            })
            .max_truncation(TAIL_DB)
            .bake();

        // ρ_top (ρ_bot/ρ_top)^(r/(ROWS−1))
        let bins: Vec<_> = (0..ROWS)
            .map(|r| {
                let f = r as f64 / (ROWS - 1) as f64;
                wav.at_rho(RHO_TOP * (RHO_BOT / RHO_TOP).powf(f))
            })
            .collect();
        let tables: Vec<_> = bins.iter().map(|b| b.taps()).collect();
        let bank: Vec<(f64, &[[f32; 4]])> = bins
            .iter()
            .zip(&tables)
            .map(|(b, t)| (b.velocity(), t.as_slice()))
            .collect();

        // ω(t) = a·t
        let a = bank[0].0 / SPAN as f64;
        let x = chirp(a, SD, 0.0);

        print_transform("MORSE", &bank, &x, SPAN, 160, -40.0);
    }
}
