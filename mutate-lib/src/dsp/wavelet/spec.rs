// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spec
//!
//! > We reject: kings, presidents, and voting.
//! > We believe in: rough consensus and running code
//! >
//! > - David "D" Clark
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
//! Resolve a bin with [`Wavelet::bin`] and realize its folded taps.  Bins may tighten the spec's
//! maxima but not exceed them.
//!
//! ```
//! # use mutate_lib::dsp::wavelet::WaveletSpec;
//! # let wavelet = WaveletSpec::default().q(5.5).truncate(-140.0).bake();
//! let bin = wavelet.bin(440.0, 48_000.0).truncate(-100.0);
//! let mut taps = vec![[0.0f32; 4]; bin.folded_taps()];
//! bin.taps_into(&mut taps);
//! ```
//!
//! ## Customizing the Wavelet Family
//!
//! The primary knobs are [`WaveletSpec::truncate`] and [`WaveletSpec::with_shape`], which modulate
//! filter length, bandwidth, stop band, and skirt depth.  All are inherently coupled.  At a fixed
//! filter length, raising Q while truncating harder (smaller magnitude of tail dB) trades time
//! precision for pitch precision.

// MAYBE Gamma = 4 is not that wild, but has a flatter top and a steeper main lobe, things we are
// interested in.  It's possibly worth a bit of Q unless reassignment becomes broken.

use core::f64::consts::{LN_10, LN_2, PI, TAU};

use libm::lgamma;
use num_complex::Complex64;

use super::defaults;
use super::generate::{hermite, quadjet::QuadJet};
use super::PEAK_GAIN;

/// Controls Q and other critical tradeoffs of the Morse family wavelet parameters.  For exact
/// details, consult [real graphs](https://arxiv.org/pdf/1203.3380).
#[derive(Clone, Copy)]
pub struct Shape {
    /// The 𝛄 value of 3 results in a useful frequency domain uniformity and is the standard choice.
    pub gamma: f64,
    /// Adjusting beta at fixed gamma is adjusting Q.  The same envelope shape will be dilated over
    /// more carrier periods, resulting in more taps required to approximate the wavelet, the
    /// familiar time vs pitch resolution tradeoff.
    pub beta: f64,
}

impl Shape {
    /// `q` is the quality factor `fc / BW` on the half-power bandwidth.  Higher `q` narrows the
    /// band and costs proportionally more taps at a given center frequency.
    ///
    /// The width at the start of the skirt, which must be controlled to avoid transition bands of
    /// downsampled inputs, is usually not more than `2 fc / q`.
    pub fn from_q(q: f64, gamma: f64) -> Self {
        // p = 2.0 * LN_2.sqrt() * q
        // beta = p * p / gamma
        Shape {
            gamma,
            beta: (4.0 * LN_2) * (q * q / gamma),
        }
    }

    /// Morse wavelet time-bandwidth product (P = √(βγ)).  It can be interpreted as a surrogate for
    /// Q.
    pub fn p(&self) -> f64 {
        (self.beta * self.gamma).sqrt()
    }

    /// Return the **estimated** `q` for this shape.  Realized `q` will be close.
    pub fn q(&self) -> f64 {
        self.p() / (2.0 * LN_2.sqrt())
    }

    /// `ω_p = (β/γ)^{1/γ}`, argmax of `ω^β e^{-ω^γ}` and the frequency `u` counts cycles of.
    pub fn peak(&self) -> f64 {
        if self.gamma == 3.0 {
            // cbrt is just the fast path
            (self.beta / 3.0).cbrt()
        } else {
            (self.beta / self.gamma).powf(1.0 / self.gamma)
        }
    }

    /// Truncation point, in periods of the carrier, where the omitted tails carry `tail_db` of the
    /// total energy of the entire wavelet.
    ///
    ///     M(t) = μ,  μ = 10^(-|tail_db| / 10)
    ///     u = t ω_p / 2π
    pub fn truncation_u(&self, tail_db: f64) -> f64 {
        let Shape { beta, gamma } = *self;

        let p = 2.0 * beta + 1.0;

        // log 2C, both tails against the whole mass
        let log_c = LN_2 + gamma.ln() + (p / gamma) * LN_2 + 2.0 * lgamma(beta + 1.0)
            - TAU.ln()
            - p.ln()
            - lgamma(p / gamma);

        // log t = (log 2C - log μ) / p
        let log_t = (log_c + tail_db.abs() / 10.0 * LN_10) / p;

        log_t.exp() * self.peak() / TAU
    }
}

/// Maxima for the family. Every bin served by the bake sits under these.
#[derive(Clone, Copy)]
pub struct WaveletSpec {
    shape: Shape,
    resolution: usize,
    max_tail_db: f64,
    max_load_quantum: usize,
    max_delay: usize,
}

impl Default for WaveletSpec {
    fn default() -> Self {
        WaveletSpec {
            shape: Shape::from_q(defaults::Q, defaults::GAMMA),
            resolution: defaults::RESOLUTION,
            max_tail_db: defaults::TAIL_DB,
            max_load_quantum: defaults::LOAD_QUANTUM,
            max_delay: 0,
        }
    }
}

impl WaveletSpec {
    /// Set shape by quality factor, holding gamma.
    pub fn q(mut self, q: f64) -> Self {
        self.shape = Shape::from_q(q, self.shape.gamma);
        self
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// Grid points per period of `u`.
    pub fn resolution(mut self, resolution: usize) -> Self {
        self.resolution = resolution;
        self
    }

    /// Weakest truncation any bin will ask for.  -10dB truncates hard, -80dB very weakly.
    /// Supporting more truncation
    pub fn max_truncation(mut self, tail_db: f64) -> Self {
        self.max_tail_db = -tail_db.abs();
        self
    }

    /// Largest load quantum any bin will ask for.
    pub fn max_load_quantum(mut self, quantum: usize) -> Self {
        self.max_load_quantum = quantum;
        self
    }

    /// Largest group delay any bin will ask for, in taps.
    pub fn max_delay(mut self, delay: usize) -> Self {
        self.max_delay = delay;
        self
    }

    pub fn bake(self) -> Wavelet {
        let du = (self.resolution as f64).recip();

        // u_max = u_trunc + (quantum + delay + 1/2) rho,  rho < 1/2 at Nyquist
        let u_max = self.shape.truncation_u(self.max_tail_db)
            + 0.5 * (self.max_load_quantum + self.max_delay) as f64
            + 0.25;

        let jet = QuadJet::standard(self.shape);
        let (psi, d) = (0..=(u_max / du).ceil() as usize + 1)
            .map(|j| {
                let t = jet.tap_at(j as f64 * du);
                (t.psi, t.d)
            })
            .unzip();

        Wavelet {
            shape: self.shape,
            du,
            psi,
            d,
            limits: BinSpec {
                center: 0.0,
                rate: 1.0,
                load_quantum: self.max_load_quantum,
                delay: self.max_delay,
                tail_db: self.max_tail_db,
            },
        }
    }
}

/// One motherlet, resolved on `u` alone.  Serves every `(fc, fs)` under the spec's maxima.
pub struct Wavelet {
    shape: Shape,
    /// Periods per grid point.
    du: f64,
    /// Ψ
    psi: Vec<Complex64>,
    /// `−(i/2π)·dψ/du`
    d: Vec<Complex64>,
    /// Bin defaults, and the ceiling the grid extent was sized against.
    limits: BinSpec,
}

impl Wavelet {
    /// Inherits quantum, delay, and tail_db from the spec that baked it.
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

    /// A bin named by ρ directly, periods per tap.  `bin(fc, fs)` is `at_rho(fc / fs)`.
    pub fn at_rho(&self, rho: f64) -> Bin<'_> {
        self.bin(rho, 1.0)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    fn at(&self, u: f64) -> Complex64 {
        hermite::eval(&self.psi, &self.d, u, self.du)
    }

    fn mass(&self, u_beg: f64, u_end: f64) -> Complex64 {
        hermite::integrate(&self.psi, &self.d, u_beg, u_end, self.du)
    }
}

/// The record a runtime `Bin` hydrates from.  No borrow, no realized geometry.
#[derive(Clone, Copy)]
pub struct BinSpec {
    center: f64,
    rate: f64,
    load_quantum: usize,
    /// Extra group delay, extra folded taps, padding that will be opportunistically used during
    /// restriction.
    // XXX has not be reconciled with load quantum!
    delay: usize,
    tail_db: f64,
}

impl BinSpec {
    pub fn load_quantum(self, quantum: usize) -> Self {
        Self {
            load_quantum: quantum,
            ..self
        }
    }

    pub fn delay(self, delay: usize) -> Self {
        Self { delay, ..self }
    }

    pub fn truncate(self, tail_db: f64) -> Self {
        let tail_db = -tail_db.abs();
        Self { tail_db, ..self }
    }

    pub fn bin<'w>(self, wavelet: &'w Wavelet) -> Bin<'w> {
        Bin::new(wavelet, self)
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

    pub fn spec(&self) -> BinSpec {
        self.spec
    }

    /// Radians per sample.  `ω₀ = 2π·ρ`
    pub fn velocity(&self) -> f64 {
        TAU * self.rho
    }

    /// Periods per tap.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    pub fn folded_taps(&self) -> usize {
        self.k
    }

    pub fn unfolded_taps(&self) -> usize {
        2 * self.k - 1
    }

    pub fn taps(&self) -> Vec<[f32; 4]> {
        let mut out = vec![[0.0f32; 4]; self.k];
        self.taps_into(&mut out);
        out
    }

    /// Writes `folded_taps()` weights and returns that count.  Each weight is
    /// `[Re ψ, Im ψ, Re d, Im d]` and the center is `[Re ψ₀, 0, Re d₀, 0]`.
    ///
    ///     H(ω) = ψ₀ + 2 Re Σ_{k≥1} ψ_k e^{-iωk}
    ///     H(ω₀) = PEAK_GAIN
    ///     H_d(ω) ≈ (ω/ω₀)·H(ω)
    ///
    /// A unit sine at the center frequency yields a unit envelope.
    ///
    /// Upstream owes an `out` at least that long.
    pub fn taps_into(&self, out: &mut [[f32; 4]]) -> usize {
        let (w, rho, k) = (self.wavelet, self.rho, self.k);
        let inv = rho.recip();

        // ψ_T at the upper edge of cell j, zero past the cut
        let edge = |j: usize| {
            if j + 1 < k {
                w.at((j as f64 + 0.5) * rho)
            } else {
                Complex64::default()
            }
        };

        let mut psi = Vec::with_capacity(k);
        let mut d = Vec::with_capacity(k);

        // the center cell is symmetric about u = 0, so the odd parts cancel
        psi.push(Complex64::new(2.0 * inv * w.mass(0.0, 0.5 * rho).re, 0.0));
        d.push(Complex64::new(2.0 * inv / TAU * edge(0).im, 0.0));

        // d_k = −(i/2πρ)·(ψ_T(e_{k+½}) − ψ_T(e_{k−½}))
        let mut lo = edge(0);
        for j in 1..k {
            psi.push(inv * w.mass((j as f64 - 0.5) * rho, (j as f64 + 0.5) * rho));
            let hi = edge(j);
            d.push(-Complex64::i() * inv / TAU * (hi - lo));
            lo = hi;
        }

        // H(ω₀) = ψ₀ + 2 Σ_k Re(ψ_k e^{-2πi u_k}),  u_k = k ρ
        let gain = psi[0].re
            + 2.0
                * psi[1..]
                    .iter()
                    .enumerate()
                    .map(|(j, p)| {
                        let (s, c) = (TAU * (j + 1) as f64 * rho).sin_cos();
                        p.re * c + p.im * s
                    })
                    .sum::<f64>();
        let scale = PEAK_GAIN / gain;

        out[0] = [scale * psi[0].re, 0.0, scale * d[0].re, 0.0].map(|v| v as f32);
        for (o, (p, q)) in out[1..k].iter_mut().zip(psi[1..].iter().zip(&d[1..])) {
            *o = [scale * p.re, scale * p.im, scale * q.re, scale * q.im].map(|v| v as f32);
        }

        k
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
