// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spec
//!
//! > We reject: kings, presidents, and voting.
//! > We believe in: rough consensus and running code
//! >
//! > - David "D" Clark
//!
//! Define wavelet families.  The [`Spec`] builds the [`Plan`], handing over the realized
//! configuration choices in the process.  The primary knobs are the [`Truncation`] and the
//! [`Shape`].  You probably want to set truncation with [`tail_db`] and shape with [`q`].  These
//! control the filter length and pitch resolution.  T
//!
//! ```rust
//! # use mutate_lib::dsp::wavelet::Spec;
//!
//! let spec = Spec::default()
//!     .q(3.5)
//!     .truncate(-20.0);
//! ```

// MAYBE Gamma = 4 is not that wild, but has a flatter top and a steeper main lobe, things we are
// interested in.  It's possibly worth a bit of Q unless reassignment becomes broken.

use core::f64::consts::{LN_10, LN_2, PI, TAU};

use libm::lgamma;
use num_complex::Complex64;

use super::generate::{hermite, quadjet::QuadJet};
use super::Plan;
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
    /// `q` is the quality factor on the -3 dB energy width. Higher `q` narrows the band and costs
    /// proportionally more taps at a given center frequency.
    ///
    /// `q` is generally `bandwidth / center frequency`, and this can be used to estimate main lobe
    /// width.  The width at the beginning of the skirt, which must be controlled to avoid
    /// transition bands of downsampled inputs, is usually not more than twice as wide.
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
}

/// Use to build coherent choices for a [`Plan`].  Choices of accuracy tradeoffs and support for
/// different `Q` and `load_quantum` are reconciled before building a `Plan`.  Re-use of these
/// builders to create a range of related plans is a convenient way to sweep across settings.
///
/// ```
/// # use mutate_lib::dsp::wavelet::Spec;
///
/// let spec = Spec::default();
/// ```
// Construction of incoherent choices should not be supported, but warning for incoherence or
// panicking for degenerate choices are both acceptable.
#[derive(Clone, Copy)]
pub struct Spec {
    shape: Shape,
    /// decibels of the truncated tail mass compared to the whole filter mass.
    tail_db: f64,

    /// The maximum load quantum that will be requested.  Load quantum space is used to taper taps
    /// less aggressively, so the mother wavelet needs to include a bit more periods to dilate into
    /// the longer time bought by load quantum.
    max_load_quantum: usize,

    /// Error tolerance for the wavelet grid.
    eps: f64,

    /// A rough sizing estimate to avoid reallocation of scratch space.
    max_taps: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            shape: Shape::from_q(3.0, 3.0),
            eps: 1e-14,
            tail_db: -40.0,
            max_taps: 0,
            max_load_quantum: 1,
        }
    }
}

impl Spec {
    /// Set shape by quality factor, holding gamma.
    pub fn q(mut self, q: f64) -> Self {
        self.shape = Shape::from_q(q, self.shape.gamma);
        self
    }

    /// Set mother wavelet [`Shape`].
    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// `eps` is the spectral truncation floor relative to the peak. It sets how far the baked grid
    /// extends, and through that the tap count, but not the shape. 1e-8 lands near the f32 noise
    /// floor of the output taps.  1e-10 is where measurable effects usually begin appearing in
    /// output filters.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
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

    /// Set error tolerance by envelope geometry.  This will control the selected N taps for each
    /// [`Bin`], so it's one of the most powerful knobs.  Values below 3.0 truncate too hard to
    /// ship until a better numerical solver is available.  Values over 5.5 begin grinding up the
    /// dust of departed f32s.
    pub fn sigmas(mut self, sigmas: f64) -> Self {
        todo!()
    }

    /// Truncate tail mass based on decibels relative to total mass.  -10dB truncates hard.  -80dB
    /// truncates very weakly.
    pub fn truncate(mut self, tail_db: f64) -> Self {
        self.tail_db = -(tail_db.abs());
        self
    }

    pub fn plan(self) -> Plan {
        todo!()
    }

    pub(super) fn shape(&self) -> Shape {
        self.shape
    }

    pub(super) fn load_quantum(&self) -> usize {
        self.max_load_quantum
    }

    /// Truncation point where the omitted tail of one side carries `tail_db` of the total energy.
    ///
    /// The magnitude of `tail_db` is used, since a tail cannot exceed the whole.
    ///
    ///     M(u) = μ,  μ = 10^(-|tail_db| / 10)
    fn truncation_u(&self) -> f64 {
        let Shape { beta, gamma } = self.shape;

        let p = 2.0 * beta + 1.0;

        // log C
        let log_c = gamma.ln() + (p / gamma) * LN_2 + 2.0 * lgamma(beta + 1.0)
            - TAU.ln()
            - p.ln()
            - lgamma(p / gamma);

        // log u = (log C - log μ) / p
        let log_u = (log_c + self.tail_db.abs() / 10.0 * LN_10) / p;

        log_u.exp()
    }

    /// Realize a [`BinPlanner`].  Currently only used to generate testing weights.
    pub fn bin_planner(self, center: f64, rate: f64) -> BinPlanner {
        BinPlanner::new(self, center, rate)
    }
}

/// Grid points per tap.  Cell mass stops moving well before this.
const RESOLUTION: usize = 256;

/// One bin, planned and baked on its own.  Primarily used for testing.  Holds a mother wavelet
/// resolved against this bin's `rho`.
// XXX maximum truncation is a good thing to figure out on the spec.  That with load quantum and max
// group delay can tell us how much *extra* mother wavelet we might need.  For tests where we are
// using extra taps to sweep precision knobs, this will be valuable.
pub struct BinPlanner {
    w0: f64,
    rho: f64,
    half: usize,
    du: f64,
    mother: Vec<Complex64>,
    slope: Vec<Complex64>,
}

impl BinPlanner {
    pub fn new(spec: Spec, center: f64, rate: f64) -> Self {
        let rho = center / rate;
        let quantum = spec.load_quantum();
        let half = ((spec.truncation_u() / rho).ceil() as usize).div_ceil(quantum) * quantum;

        let du = rho / RESOLUTION as f64;
        let jet = QuadJet::standard(spec.shape());

        let (mother, slope) = (0..=RESOLUTION * (half + 1))
            .map(|j| {
                let t = jet.tap_at(j as f64 * du);
                (t.psi, t.d)
            })
            .unzip();

        BinPlanner {
            w0: TAU * rho,
            rho,
            half,
            du,
            mother,
            slope,
        }
    }

    /// Radial velocity.  Radians per input sample at the configured input sample rate.
    pub fn velocity(&self) -> f64 {
        self.w0
    }

    /// Periods `u` per tap.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Number of folded pairs and the center tap.
    pub fn folded_taps(&self) -> usize {
        self.half + 1
    }

    /// Total number of taps.
    pub fn unfolded_taps(&self) -> usize {
        2 * self.half + 1
    }

    /// Writes `folded_taps()` weights and returns that count.
    ///
    /// Upstream owes an `out` at least that long.
    pub fn taps_into(&self, out: &mut [[f32; 4]]) -> usize {
        let k = self.folded_taps();
        let inv = self.rho.recip();

        // psi at the cell edge above tap j
        let edge = |j: usize| self.mother[RESOLUTION / 2 + j * RESOLUTION];

        let mut psi = Vec::with_capacity(k);
        let mut d = Vec::with_capacity(k);

        // the center cell is symmetric about u = 0, so the odd parts cancel
        psi.push(Complex64::new(
            2.0 * inv * self.mass(0.0, 0.5 * self.rho).re,
            0.0,
        ));
        d.push(Complex64::new(2.0 / self.w0 * edge(0).im, 0.0));

        for j in 1..k {
            let (lo, hi) = ((j as f64 - 0.5) * self.rho, (j as f64 + 0.5) * self.rho);
            psi.push(inv * self.mass(lo, hi));
            d.push(-Complex64::i() / self.w0 * (edge(j) - edge(j - 1)));
        }

        // H(w0) = psi_0 + 2 sum_k Re(psi_k e^{-i w0 k}), real by the fold
        let gain = psi[0].re
            + 2.0
                * psi[1..]
                    .iter()
                    .enumerate()
                    .map(|(j, p)| {
                        let (s, c) = (self.w0 * (j + 1) as f64).sin_cos();
                        p.re * c + p.im * s
                    })
                    .sum::<f64>();
        let scale = PEAK_GAIN / gain;

        out[0] = [0.5 * scale * psi[0].re, 0.0, 0.5 * scale * d[0].re, 0.0].map(|v| v as f32);
        for (o, (p, q)) in out[1..k].iter_mut().zip(psi[1..].iter().zip(&d[1..])) {
            *o = [scale * p.re, scale * p.im, scale * q.re, scale * q.im].map(|v| v as f32);
        }

        k
    }

    fn mass(&self, u_beg: f64, u_end: f64) -> Complex64 {
        hermite::integrate(&self.mother, &self.slope, u_beg, u_end, self.du)
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
