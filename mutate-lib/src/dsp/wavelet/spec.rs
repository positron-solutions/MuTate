// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spec
//!
//! > We reject: kings, presidents, and voting.
//! > We believe in: rough consensus and running code
//! >
//! > - David "D" Clark
//!
//! The specs only encapsulate the related decisions that go into generating a wavelet family or
//! similar set of bins.  See the parent module for usage.

// NEXT PSL is perhaps a better measure when planning for useful dynamic range because the skirt
// rolls off quite deep for some filters, depending on the truncation performance etc.  After adding
// some spectral cleanup, we'll see where the numbers and shapes land.  Tighter side lobe and
// shaping the garbage on the noise floor is probably something that can be bought.

use core::f64::consts::{FRAC_2_SQRT_PI, LN_10, LN_2, PI, TAU};
use core::ops::Range;

use libm::{erfc, lgamma};
use num_complex::Complex64;

use super::defaults;
use super::generate::{hermite, quadjet::QuadJet};
use super::refine;
use super::restrict;
use super::{Bin, Wavelet, PEAK_GAIN};

/// Lowest stopband target that more taps can buy.
pub const IMAGE_FLOOR_DB: f64 = -140.0;
/// Neper is the natural-log analogue of the decibel.  It is not a beloved unit, but simplifies
/// expressions in some domains.
const DB_PER_NP: f64 = 20.0 / LN_10;
/// Deepest noise floor an `f32` table can reliably express against `PEAK_GAIN`.
pub const NOISE_FLOOR_LIMIT_DB: f64 = -140.0;
/// Truncation sits this far under the noise floor, keeping the fold the binding feature and the
/// delivered −3 dB width faithful to Q.
pub(crate) const TAIL_OVER_FLOOR_DB: f64 = 20.0;
/// Lowest Q whose envelope spans enough carrier periods to hold its shape.  Under it the crest
/// leaves ω₀ and the skirt never reaches a floor.
pub const Q_FLOOR: f64 = 2.5;

/// Controls Q and other critical tradeoffs of the Morse family wavelet parameters.  For exact
/// details, consult [real graphs](https://arxiv.org/pdf/1203.3380).
#[derive(Clone, Copy)]
pub struct Shape {
    /// The 𝛄 value of 3 results in a useful frequency domain uniformity and is the standard choice.
    /// Higher values have slightly different asymmetry but give tighter main lobes.  Both `gamma` and
    /// `beta` cost taps via the `p` factor in downstream calculations, so do your homework.
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
    pub const fn from_q(q: f64, gamma: f64) -> Self {
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

    /// M_k = 0,  Ψ⁽ᵏ⁾(0) = 0,  k < β
    ///
    /// ∫ |t|ᵏ |ψ| is finite over the same range, |ψ(t)| ~ |t|^{−(β+1)}.
    pub fn vanishing_moments(&self) -> Range<u32> {
        0..self.beta.ceil() as u32
    }

    /// ∫ |t|ᵏ |ψ|² finite,  k < 2β + 1
    pub fn energy_moments(&self) -> Range<u32> {
        0..(2.0 * self.beta + 1.0).ceil() as u32
    }

    /// x = ω/ω₀ below the crest where the skirt meets the floor.
    ///
    /// ```text
    /// β ln x + (β/γ)(1 − xᵞ) = floor ln10 / 20
    /// ```
    pub fn skirt(&self, noise_floor: f64) -> f64 {
        let Shape { beta, gamma } = *self;
        let np = -noise_floor.abs() * LN_10 / 20.0;
        let mut x = ((np - beta / gamma) / beta).exp();
        for _ in 0..8 {
            x = ((np - beta / gamma * (1.0 - x.powf(gamma))) / beta).exp();
        }
        x
    }

    /// Nyquist fold of a bin at `rho`, in dB under the passband crest.
    ///
    /// ```text
    /// H(−π) / H(ω₀) = Ψ(ω_p/2ρ) / Ψ(ω_p)
    /// ```
    pub(super) fn image_db(&self, rho: f64) -> f64 {
        -DB_PER_NP * self.beta * fold_cost(0.5 / rho, self.gamma)
    }

    /// Least shape holding the Nyquist fold at or below `noise_floor` for every bin up to
    /// `max_rho`.
    ///
    /// ```text
    /// β D(1/2ρ, γ) = |floor| ln10 / 20
    /// ```
    pub fn from_noise_floor(max_rho: f64, noise_floor: f64, gamma: f64) -> Self {
        let db = noise_floor.abs().min(NOISE_FLOOR_LIMIT_DB.abs());
        let shape = Shape {
            gamma,
            beta: db / (DB_PER_NP * fold_cost(0.5 / max_rho, gamma)),
        };
        debug_assert!(
            shape.q() >= Q_FLOOR,
            "floor {noise_floor:.1} at rho {max_rho} wants Q {:.2}, under the family's {Q_FLOOR}",
            shape.q()
        );
        shape
    }

    /// White noise gain of a bin at `rho`, what a unit variance input reads as |Ψ|².
    ///
    /// ```text
    /// Σ_ν |ψ_ν|² = ρ G² Γ(r) s^{−r} e^s / γ,  s = 2β/γ,  r = s + 1/γ
    /// ```
    #[cfg(test)]
    pub(super) fn noise_gain(&self, rho: f64) -> f64 {
        let s = 2.0 * self.beta / self.gamma;
        let r = s + self.gamma.recip();
        rho * PEAK_GAIN * PEAK_GAIN * (lgamma(r) - r * s.ln() + s).exp() / self.gamma
    }

    /// Model estimate of the truncation point in carrier periods.
    ///
    /// ```text
    /// μ = 10^(-|tail_db| / 10)
    /// 2C t^(-p) = μ
    /// erfc(t ω_p / P) = μ
    /// u = t ω_p / 2π
    /// ```
    pub fn truncation_u(&self, tail_db: f64) -> f64 {
        // NEXT we can absolutely switch over to a pre-baked empirical estimate.  The analytic
        // estimates are just good enough to get off the ground.  Some combination of using the
        // envelope estimation for its math and our high-quality generators can create something
        // accurate to 1% or so pretty handily.  Given how load quantum must work, it's basically
        // meaningless to chase further tail precision except that the noise floor estimates will
        // tighten up.  The actual noise floor will not get any better for a given `tail_db`, but
        // we will estimate it better.

        let Shape { beta, gamma } = *self;
        // 2β + 1, the algebraic decay exponent
        let a = 2.0 * beta + 1.0;
        let l = tail_db.abs() / 10.0 * LN_10;

        // Gaussian bulk about ω_p
        let u_gauss = erfc_inv_exp(l) * self.p() / TAU;

        // algebraic tails from the branch point at ω = 0
        let log_c = LN_2 + gamma.ln() + (a / gamma) * LN_2 + 2.0 * lgamma(beta + 1.0)
            - TAU.ln()
            - a.ln()
            - lgamma(a / gamma);
        let t_alg = ((log_c + l) / a).exp();

        // Watson ratio Γ(β+1+γ) / Γ(β+1) t^γ = 1
        let t_watson = ((lgamma(beta + 1.0 + gamma) - lgamma(beta + 1.0)) / gamma).exp();

        if t_alg > t_watson {
            u_gauss.max(t_alg * self.peak() / TAU)
        } else {
            u_gauss
        }
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape::from_q(defaults::Q, defaults::GAMMA)
    }
}

/// (f^γ − 1)/γ − ln f
fn fold_cost(f: f64, gamma: f64) -> f64 {
    (f.powf(gamma) - 1.0) / gamma - f.ln()
}

/// Inverse of `fold_cost` on f ≥ 1.
fn fold_reach(cost: f64, gamma: f64) -> f64 {
    // γε²/2 near the crest, (f^γ − 1)/γ away from it
    let near = 1.0 + (2.0 * cost / gamma).sqrt();
    let far = (gamma * cost + 1.0).powf(gamma.recip());
    let mut f = near.max(far);
    for _ in 0..6 {
        f -= (fold_cost(f, gamma) - cost) / (f.powf(gamma - 1.0) - f.recip());
    }
    f
}

/// erfc⁻¹(e^(-l))
fn erfc_inv_exp(l: f64) -> f64 {
    // x² + ½ ln(π x²) = l
    let x = (l - 0.5 * (PI * l).ln()).sqrt();

    // Newton on ln erfc
    let le = erfc(x).ln();
    x + (le + l) * (le + x * x).exp() / FRAC_2_SQRT_PI
}

/// Maxima for the family. Every bin served by the bake sits under these.
#[derive(Clone, Copy)]
pub struct WaveletSpec {
    pub(super) shape: Shape,
    pub(super) resolution: usize,
    pub(super) max_load_quantum: usize,
    pub(super) max_noise_floor: f64,
    pub(super) max_rho: Option<f64>,
    pub(super) max_delay: usize,
    pub(super) restriction: restrict::Restriction,
    pub(super) refine: Option<refine::Refine>,
}

impl Default for WaveletSpec {
    fn default() -> Self {
        WaveletSpec {
            shape: Shape::from_q(defaults::Q, defaults::GAMMA),
            resolution: defaults::RESOLUTION,
            max_rho: None,
            max_noise_floor: defaults::NOISE_FLOOR,
            max_load_quantum: defaults::LOAD_QUANTUM,
            max_delay: 0,
            restriction: restrict::Restriction {
                // quadrature: restrict::Quadrature::Weighted,
                quadrature: restrict::Quadrature::Axial,
                taper: restrict::Taper::Knee { curvature: 1.0 },
                // taper: restrict::Taper::Cylinder,
                // taper: restrict::Taper::Rectangle,
                // derivative: restrict::Derivative::Envelope,
                derivative: restrict::Derivative::Folded { sigmas: 6.0 },
                // derivative: restrict::Derivative::Fitted { gate_db: 24.0 },
            },
            refine: Some(refine::Refine::default()),
            // refine: None,
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

    /// Deepest noise floor any bin will ask for.
    pub fn max_noise_floor(mut self, db: f64) -> Self {
        self.max_noise_floor = -db.abs().min(NOISE_FLOOR_LIMIT_DB.abs());
        self
    }

    /// Deepest truncation any bin will ask for, held as the noise floor it lands.
    pub fn max_truncation(mut self, tail_db: f64) -> Self {
        self.max_noise_floor = -tail_db.abs() + TAIL_OVER_FLOOR_DB;
        self
    }

    /// Highest ρ any bin will ask for.  The noise floor holds against the Nyquist fold up to here.
    pub fn max_rho(mut self, rho: f64) -> Self {
        self.max_rho = Some(rho);
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

    /// Set the method for squeezing the high resolution motherlet into `N` taps.
    pub fn with_restriction(mut self, restriction: restrict::Restriction) -> Self {
        self.restriction = restriction;
        self
    }

    /// Set the method for restoring some of the pre-truncation transient response characteristics.
    pub fn with_refine(mut self, refine: Option<refine::Refine>) -> Self {
        self.refine = refine;
        self
    }

    pub fn bake(self) -> Wavelet {
        if let Some(rho) = self.max_rho {
            debug_assert!(
                self.shape.image_db(rho) <= self.max_noise_floor + 1e-9,
                "fold {:.1} dB at rho {rho} breaks the {:.1} dB floor",
                self.shape.image_db(rho),
                self.max_noise_floor
            );
        }

        let du = (self.resolution as f64).recip();
        let u_max = self
            .shape
            .truncation_u(self.max_noise_floor - TAIL_OVER_FLOOR_DB)
            + 0.5 * (self.max_load_quantum + self.max_delay) as f64
            + 0.25;

        let jet = QuadJet::standard(self.shape);
        let (mut psi, mut d): (Vec<Complex64>, Vec<Complex64>) = (0..=(u_max / du).ceil() as usize
            + 1)
            .map(|j| {
                let t = jet.tap_at(j as f64 * du);
                (t.psi, t.d)
            })
            .unzip();

        // Normalize to [0,2) range without touching the mantissas.
        let peak = psi
            .iter()
            .map(|p| p.re.abs().max(p.im.abs()))
            .fold(0.0f64, f64::max);
        debug_assert!(peak.is_normal());
        let exponent = (peak.to_bits() >> 52 & 0x7ff) as i32 - 1023;
        let scale = 2f64.powi(-exponent);
        for (p, q) in psi.iter_mut().zip(&mut d) {
            *p *= scale;
            *q *= scale;
        }

        Wavelet {
            shape: self.shape,
            du,
            psi,
            d,
            limits: BinSpec {
                center: self.max_rho.unwrap_or(0.5),
                rate: 1.0,
                load_quantum: self.max_load_quantum,
                delay: self.max_delay,
                noise_floor: self.max_noise_floor,
                refine: None,
            },
            restriction: self.restriction,
        }
    }
}

/// The choices owned by each [`Bin`] that must only be within [`Wavelet`] limits.  Not yet tied to
/// any `Wavelet`, which is where the [`Shape`] geometry comes from.
#[derive(Clone, Copy)]
pub struct BinSpec {
    pub(super) center: f64,
    pub(super) rate: f64,
    pub(super) load_quantum: usize,
    /// Extra group delay, extra folded taps, padding that will be opportunistically used during
    /// restriction.
    // XXX has not be reconciled with load quantum!
    pub(super) delay: usize,
    pub(super) noise_floor: f64,
    pub(super) refine: Option<refine::Refine>,
}

impl BinSpec {
    pub fn new(center: f64, rate: f64) -> Self {
        debug_assert!(center <= 0.5 * rate);
        Self {
            center,
            rate,
            load_quantum: defaults::LOAD_QUANTUM,
            delay: defaults::DELAY,
            noise_floor: defaults::NOISE_FLOOR,
            refine: Some(refine::Refine::default()),
        }
    }

    pub fn load_quantum(self, quantum: usize) -> Self {
        Self {
            load_quantum: quantum,
            ..self
        }
    }

    pub fn delay(self, delay: usize) -> Self {
        Self { delay, ..self }
    }

    pub fn noise_floor(self, db: f64) -> Self {
        Self {
            noise_floor: -db.abs(),
            ..self
        }
    }

    pub fn truncate(self, tail_db: f64) -> Self {
        self.noise_floor(tail_db.abs() - TAIL_OVER_FLOOR_DB)
    }

    pub fn bin<'w>(self, wavelet: &'w Wavelet) -> Bin<'w> {
        wavelet.from_spec(self)
    }

    pub fn with_refine(self, refine: Option<refine::Refine>) -> Self {
        Self { refine, ..self }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // As mentioned in the main module, we need behaviors that prevent misuse of `Bin` setters
    // without checking Wavelet compatibility.
}
