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
// rolls off quite deeply for some filters, depending on the truncation performance etc.  After
// adding some spectral cleanup, we'll see where the numbers and shapes land.  Tighter side lobe and
// shaping the garbage on the noise floor is probably something that can be bought.
// MAYBE Contradictory specification goals (eg max delay + min Q) would require some restructuring to
// form a consistent API.  That is mostly a decision for planning a filter bank, not an individual
// filter.  The implementation may be incomplete or the need may be addressed downstream.
// DEBT Noise floor / truncation is among one of the contradictory specification goals.  It should
// be just fine, but currently the units don't match and just got a bit lucky, being pseudolinear
// enough to appear to be working.  As truncation bites harder, the main lobe gets wider and centers
// try to wander.

use core::f64::consts::{FRAC_2_SQRT_PI, LN_10, LN_2, PI, TAU};
use core::ops::Range;

use libm::{erfc, lgamma};
use num_complex::Complex64;

use super::defaults;
use super::generate::quadjet::QuadJet;
use super::inspect::bisect;
use super::refine;
use super::restrict;
use super::{Bin, Wavelet, PEAK_GAIN};

/// Neper is the natural-log analogue of the decibel.  It is not a beloved unit, but simplifies
/// expressions in some domains.
const DB_PER_NP: f64 = 20.0 / LN_10;
/// Deepest noise floor an `f32` table can reliably express against `PEAK_GAIN`.
pub const NOISE_FLOOR_LIMIT_DB: f64 = -160.0;
/// Truncation sits this far under the noise floor, keeping the fold the binding feature and the
/// delivered −3 dB width faithful to Q.
pub(crate) const TAIL_OVER_FLOOR_DB: f64 = -10.0;
/// Lowest Q whose envelope spans enough carrier periods to hold its shape.  Under it the crest
/// leaves ω₀ and the skirt never reaches a floor.
pub const Q_FLOOR: f64 = 2.5;
/// Temporary calibration for Q effective, which is proportionate, not equal to N^{-3/2}.
pub const Q_REASSIGNMENT_CAL: f64 = 1.0;

/// Saddle continuation steps per envelope sigma.
const SADDLE_STEPS: f64 = 4.0;
const SADDLE_NEWTON: usize = 3;

/// |ψ|² at t = 2πu/ω_p as saddle and endpoint parts, logs in the units of Ψ = ω^β e^{−ω^γ}.
struct Parts {
    t: f64,
    /// ω_s
    w: Complex64,
    /// φ''(ω_s)
    c: Complex64,
    /// ln |ψ_s|², scaled to the exact ψ(0)
    saddle: f64,
    /// ln |ψ_a|², −∞ before Watson
    endpoint: f64,
}

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
        // DEBT QuadJet's saddle handling or Stokes table march is the weakness.  We may be able to
        // disable the Jet for non-integral gamma, falling back to pure quadrature, or use feature
        // gating and fall back to IFFT.  To the extent that real gamma would be useful, the refine
        // module can probably pick up the slack.  Technically QuadJet is only valid for gamma >=
        // 3.0, but tell that to the wavelets it successfully produces.
        debug_assert!(gamma.trunc() == gamma, "QuadJet requires integral gamma");

        // p = 2.0 * LN_2.sqrt() * q
        // beta = p * p / gamma
        Shape {
            gamma,
            beta: (4.0 * LN_2) * (q * q / gamma),
        }
    }

    /// Least shape holding the Nyquist fold at or below `noise_floor` for every bin up to
    /// `max_rho`.
    ///
    /// ```text
    /// β D(1/2ρ, γ) = |floor| ln10 / 20
    /// ```
    pub fn from_noise_floor(max_rho: f64, noise_floor: f64, gamma: f64) -> Self {
        // DEBT This method is just overall not that great.  It slipped through smoke tests but
        // doesn't seem very accurate or precise compared to some of the other analytic tools in our
        // box.

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

    /// Nyquist fold of a bin at `rho`, in dB under the passband crest.
    ///
    /// ```text
    /// H(−π) / H(ω₀) = Ψ(ω_p/2ρ) / Ψ(ω_p)
    /// ```
    pub(super) fn image_db(&self, rho: f64) -> f64 {
        -DB_PER_NP * self.beta * fold_cost(0.5 / rho, self.gamma)
    }

    /// Highest ρ whose Nyquist fold holds `noise_floor`, the inverse of `from_noise_floor`.
    ///
    /// ```text
    /// ρ = 1 / 2D⁻¹(|floor| ln10 / 20β, γ)
    /// ```
    pub fn fold_ceiling(&self, noise_floor: f64) -> f64 {
        0.5 / fold_reach(noise_floor.abs() / (DB_PER_NP * self.beta), self.gamma)
    }

    /// White noise gain of a bin at `rho`, what a unit variance input reads as |Ψ|².
    ///
    /// ```text
    /// Σ_ν |ψ_ν|² = ρ G² Γ(r) s^{−r} e^s / γ,  s = 2β/γ,  r = s + 1/γ
    /// ```
    pub fn noise_gain(&self, rho: f64) -> f64 {
        let s = 2.0 * self.beta / self.gamma;
        let r = s + self.gamma.recip();
        rho * PEAK_GAIN * PEAK_GAIN * (lgamma(r) - r * s.ln() + s).exp() / self.gamma
    }

    /// Crest SNR of a complex tone at `snr_db` per sample against white noise.
    ///
    /// ```text
    /// SNR_out = SNR_in G² / Σ|ψ_ν|²
    /// ```
    pub fn snr_out(&self, rho: f64, snr_db: f64) -> f64 {
        10f64.powf(snr_db / 10.0) * PEAK_GAIN * PEAK_GAIN / self.noise_gain(rho)
    }

    /// Reassigned pitch resolution in the half-power width units of `q`.
    ///
    /// ```text
    /// σ_r² = ½ (σ_ω / ω₀)² / SNR_out,  σ_ω = ω_p / √2 P
    /// Qeff = 2 Q √SNR_out
    /// ```
    pub fn q_reassignment(&self, rho: f64, snr_db: f64) -> f64 {
        Q_REASSIGNMENT_CAL * 2.0 * self.q() * self.snr_out(rho, snr_db).sqrt()
    }

    /// φ'' = −β/ω² − γ(γ−1)ω^{γ−2}
    fn curvature(&self, w: Complex64) -> Complex64 {
        let Shape { beta, gamma } = *self;
        -beta / (w * w) - gamma * (gamma - 1.0) * w.powf(gamma - 2.0)
    }

    /// ω_s solving β/ω − γω^{γ−1} + it = 0, continued from ω_p at t = 0.
    fn saddle(&self, t: f64) -> Complex64 {
        let Shape { beta, gamma } = *self;
        // t / σ_t,  σ_t = P / ω_p
        let steps = (SADDLE_STEPS * t * self.peak() / self.p()).ceil().max(1.0) as usize;
        let mut w = Complex64::new(self.peak(), 0.0);
        for s in 1..=steps {
            let it = Complex64::new(0.0, t * s as f64 / steps as f64);
            for _ in 0..SADDLE_NEWTON {
                w -= (beta / w - gamma * w.powf(gamma - 1.0) + it) / self.curvature(w);
            }
        }
        w
    }

    /// ln |ψ(0)|² = 2 ln(Γ((β+1)/γ) / 2πγ)
    fn ln_center(&self) -> f64 {
        2.0 * (lgamma((self.beta + 1.0) / self.gamma) - self.gamma.ln() - TAU.ln())
    }

    fn parts(&self, u: f64) -> Parts {
        let Shape { beta, gamma } = *self;
        let (wp, p) = (self.peak(), self.p());
        let t = TAU * u / wp;

        // ln |ψ(0)|² exact over saddle
        let k0 = self.ln_center()
            - (2.0 * (beta * wp.ln() - beta / gamma) - TAU.ln() - 2.0 * (p / wp).ln());

        let w = self.saddle(t);
        let c = self.curvature(w);
        let phi = beta * w.ln() - w.powf(gamma) + Complex64::new(0.0, t) * w;
        let watson = ((lgamma(beta + 1.0 + gamma) - lgamma(beta + 1.0)) / gamma).exp();

        Parts {
            t,
            w,
            c,
            saddle: 2.0 * phi.re - TAU.ln() - c.norm().ln() + k0,
            // |ψ_a|² = Γ(β+1)² / 4π² t^{2β+2}
            endpoint: match t > watson {
                true => 2.0 * lgamma(beta + 1.0) - 2.0 * TAU.ln() - 2.0 * (beta + 1.0) * t.ln(),
                false => f64::NEG_INFINITY,
            },
        }
    }

    /// ln(a(u)/a(0)) and the elasticity −u a'/a of the envelope.
    ///
    /// ```text
    /// |ψ|² = |ψ_s|² + |ψ_a|²
    /// s = (|ψ_s|² t Im ω_s + |ψ_a|² (β + 1)) / |ψ|²
    /// ```
    pub fn envelope(&self, u: f64) -> (f64, f64) {
        let e = self.parts(u);
        let total = log_add(e.saddle, e.endpoint);
        // endpoint share of |ψ|²
        let share = (e.endpoint - total).exp();
        let s = (1.0 - share) * e.t * e.w.im + share * (self.beta + 1.0);
        (0.5 * (total - self.ln_center()), s)
    }

    /// Energy outside ±u carrier periods as a share of the whole, in dB.
    ///
    /// ```text
    /// f' = −2 Im ω_s,  f'' = 2 Re 1/φ''
    /// ∫_T^∞ |ψ_s|² = |ψ_s(T)|² √(π / 2|f''|) erfcx(|f'| / √(2|f''|))
    /// ∫_T^∞ |ψ_a|² = |ψ_a(T)|² T / (2β + 1)
    /// μ = ∫_T^∞ (|ψ_s|² + |ψ_a|²) / ½∫|ψ|²
    /// ```
    pub fn truncation_tail(&self, u: f64) -> f64 {
        let Shape { beta, gamma } = *self;
        let e = self.parts(u);

        // |f'|, |f''|
        let (a, b) = (2.0 * e.w.im, (-2.0 * e.c.inv().re).max(f64::MIN_POSITIVE));
        let saddle = e.saddle + 0.5 * (PI / (2.0 * b)).ln() + erfcx(a / (2.0 * b).sqrt()).ln();
        // 2β + 1
        let n = 2.0 * beta + 1.0;
        let endpoint = e.endpoint + e.t.ln() - n.ln();

        // ln ½∫|ψ|² = ln Γ(r) − ln γ − r ln 2 − ln 4π,  r = (2β + 1)/γ
        let r = n / gamma;
        let half = lgamma(r) - gamma.ln() - r * LN_2 - (2.0 * TAU).ln();
        0.5 * DB_PER_NP * (log_add(saddle, endpoint) - half)
    }

    /// Least half span in carrier periods whose `truncation_tail` reaches `tail_db`.
    pub fn truncation_u(&self, tail_db: f64) -> f64 {
        let goal = -tail_db.abs();
        // x = 1, doubled past the goal
        let mut hi = self.p() / TAU;
        while self.truncation_tail(hi) > goal {
            hi *= 2.0;
        }
        bisect(|u| self.truncation_tail(u) - goal, 0.0, hi)
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

/// ln(eᵃ + eᵇ)
fn log_add(a: f64, b: f64) -> f64 {
    let (hi, lo) = (a.max(b), a.min(b));
    hi + (lo - hi).exp().ln_1p()
}

/// e^{y²} erfc(y), y ≥ 0
fn erfcx(y: f64) -> f64 {
    match y < 20.0 {
        true => (y * y).exp() * erfc(y),
        // (1/y√π)(1 − 1/2y² + 3/4y⁴)
        false => {
            let z = (y * y).recip();
            0.5 * FRAC_2_SQRT_PI / y * (1.0 - 0.5 * z + 0.75 * z * z)
        }
    }
}

/// The `WaveletSpec` supports
#[derive(Clone, Copy)]
pub struct WaveletSpec {
    pub(super) shape: Shape,
    pub(super) resolution: usize,
    pub(super) max_load_quantum: usize,
    pub(super) max_noise_floor: f64,
    pub(super) max_half_span: Option<f64>,
    pub(super) max_rho: Option<f64>,
    pub(super) max_delay: usize,
    pub(super) restriction: restrict::Restriction,
    pub(super) refinement: Option<refine::Refinement>,
}

impl Default for WaveletSpec {
    fn default() -> Self {
        WaveletSpec {
            shape: Shape::from_q(defaults::Q, defaults::GAMMA),
            resolution: defaults::RESOLUTION,
            max_rho: None,
            max_noise_floor: defaults::NOISE_FLOOR,
            max_half_span: None,
            max_load_quantum: defaults::LOAD_QUANTUM,
            max_delay: 0,
            restriction: restrict::Restriction::default(),
            refinement: Some(refine::Refinement::default()),
        }
    }
}

impl WaveletSpec {
    /// Set shape by quality factor, holding gamma.
    pub fn q(mut self, q: f64) -> Self {
        self.shape = Shape::from_q(q, self.shape.gamma);
        self
    }

    /// Set γ, holding Q.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.shape = Shape::from_q(self.shape.q(), gamma);
        self
    }

    /// Provide a [`Shape`] directly.
    pub fn shape(mut self, shape: Shape) -> Self {
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
        self.max_noise_floor = -(db.abs().min(NOISE_FLOOR_LIMIT_DB.abs()));
        self
    }

    /// Longest half span any bin will reach, in carrier periods.
    pub fn max_half_span(mut self, u: f64) -> Self {
        self.max_half_span = Some(u);
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
    pub fn restriction(mut self, restriction: restrict::Restriction) -> Self {
        self.restriction = restriction;
        self
    }

    /// Set the method for restoring some of the pre-truncation transient response characteristics.
    pub fn refinement(mut self, refine: Option<refine::Refinement>) -> Self {
        self.refinement = refine;
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
        let reach = self.max_half_span.unwrap_or_else(|| {
            self.shape
                .truncation_u(self.max_noise_floor - TAIL_OVER_FLOOR_DB)
        });
        let u_max = reach
            + self.max_rho.unwrap_or(0.5) * ((self.max_load_quantum + self.max_delay) as f64 + 1.5);

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
            },
            restriction: self.restriction,
            refinement: self.refinement,
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
        }
    }

    pub fn load_quantum(self, quantum: usize) -> Self {
        Self {
            load_quantum: quantum,
            ..self
        }
    }

    /// Likely not implemented yet
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
}

#[cfg(test)]
mod test {
    use super::*;

    // As mentioned in the main module, we need behaviors that prevent misuse of `Bin` setters
    // without checking Wavelet compatibility.
}
