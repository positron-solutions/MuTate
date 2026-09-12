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

// MAYBE Gamma = 4 is not that wild, but has a flatter top and a steeper main lobe, things we are
// interested in.  It's possibly worth a bit of Q unless reassignment becomes broken.

use core::f64::consts::{FRAC_2_SQRT_PI, LN_10, LN_2, PI, TAU};

use libm::{erfc, lgamma};
use num_complex::Complex64;

use super::defaults;
use super::generate::{hermite, quadjet::QuadJet};
use super::restrict;
use super::{Bin, Wavelet, PEAK_GAIN};

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

    /// Model estimate of the truncation point in carrier periods.
    ///
    ///     μ = 10^(-|tail_db| / 10)
    ///     2C t^(-p) = μ
    ///     erfc(t ω_p / P) = μ
    ///     u = t ω_p / 2π
    pub fn truncation_u(&self, tail_db: f64) -> f64 {
        let Shape { beta, gamma } = *self;
        let p = 2.0 * beta + 1.0;
        let l = tail_db.abs() / 10.0 * LN_10;

        // algebraic tails from the branch point at ω = 0
        let log_c = LN_2 + gamma.ln() + (p / gamma) * LN_2 + 2.0 * lgamma(beta + 1.0)
            - TAU.ln()
            - p.ln()
            - lgamma(p / gamma);
        let u_alg = ((log_c + l) / p).exp() * self.peak() / TAU;

        // Gaussian bulk about ω_p
        let u_gauss = erfc_inv_exp(l) * self.p() / TAU;

        u_alg.max(u_gauss)
    }
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
    pub(super) max_tail_db: f64,
    pub(super) max_load_quantum: usize,
    pub(super) max_delay: usize,
    pub(super) restriction: restrict::Restriction,
}

impl Default for WaveletSpec {
    fn default() -> Self {
        WaveletSpec {
            shape: Shape::from_q(defaults::Q, defaults::GAMMA),
            resolution: defaults::RESOLUTION,
            max_tail_db: defaults::TAIL_DB,
            max_load_quantum: defaults::LOAD_QUANTUM,
            max_delay: 0,
            restriction: restrict::Restriction::default(),
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

    /// Set the method for squeezing the high resolution motherlet into `N` taps.
    pub fn with_restriction(mut self, restriction: restrict::Restriction) -> Self {
        self.restriction = restriction;
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
            restriction: self.restriction,
        }
    }
}

/// The record a runtime `Bin` hydrates from.  No borrow, no realized geometry.
// MAYBE we need to create the setters and the interface to build a bin from a wavelet via spec
// instead of building bins and customizing them in a way that mutates the bins.
#[derive(Clone, Copy)]
pub struct BinSpec {
    pub(super) center: f64,
    pub(super) rate: f64,
    pub(super) load_quantum: usize,
    /// Extra group delay, extra folded taps, padding that will be opportunistically used during
    /// restriction.
    // XXX has not be reconciled with load quantum!
    pub(super) delay: usize,
    pub(super) tail_db: f64,
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

#[cfg(test)]
mod test {
    use super::*;

    // As mentioned in the main module, we need behaviors that prevent misuse of `Bin` setters
    // without checking Wavelet compatibility.
}
