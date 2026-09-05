// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Deformed Contour
//!
//! This method is a quadrature integration that is somewhat well behaved (precision in transition
//! region sags but isn't terrible) and serves as a 3rd opinion on the IFFT and QuadJet methods.  The
//! implementation should be compact, simple, and work at any u.
//!
//! Other reference methods tried included Hypergeometric Airy and the Taylor expansion, but both of
//! those only function at small u and worst of all, lose precision in a way that provides no
//! indication of who as wrong in any u sweep.  Small u methods are like the tenth doctor.  They
//! might be right, but since all evidence is against what they say, they can't decide anything
//! alone.

// NEXT support variable precision
// NEXT use a more similar interface as QuadJet with precision knobs.
// NEXT add a self-precison convergence test when available.
// NEXT Check the root selection for the saddle term.  Saddles near zero and in the transition have
// been all over the place, and quadrature is generally being used to make up for this.  If the
// saddle term is bad enough, it might affect the quadrature, and that might affect the method
// agreement.  We don't need something as sophisticated / complex as QuadJet,  but we shouldn't just
// throw away available grip on a known source of badness.
// 🤖 Of course.

use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

use super::super::spec::Shape;
use super::super::whatsleft::Accumulator;

const H: f64 = 1.0 / 64.0;
const TAU_MAX: f64 = 4.0;

pub struct DeformedContourEval {
    pub value: Complex64,
    /// log10(Σ|terms| / |Σ terms|).  Expect well under 1 here.
    pub digits_lost: f64,
    /// Relative change between the h and 2h double-exponential rules.
    pub residual: f64,
}

/// # Deformed Contour Method
///
/// Same integral as the Airy series, on a different path.  `[0, ∞)` becomes `0 → ω_s → ∞·e^{iθ}`,
/// where `ω_s` is the saddle of `β ln ω − ω^γ + iωt` and `θ` is the steepest-descent direction
/// clamped into the sector where `e^{-ω^γ}` still closes the contour at infinity.  The modulus is
/// monotone along both legs, so there is no cancellation to fight and f64 keeps ~14 digits at any
/// `u`.  No asymptotics: the integrand is exact everywhere on the path.
///
/// `u` and the `1/peak` amplitude match the Airy module, so values are directly comparable.
pub fn tap_at(shape: Shape, u: f64) -> DeformedContourEval {
    let peak = (shape.beta / shape.gamma).powf(1.0 / shape.gamma);
    let t = TAU * u / peak;
    let saddle = saddle(shape, t);
    let dir = tail_direction(shape, saddle);

    let f = |w: Complex64| {
        (shape.beta * w.ln() - w.powf(shape.gamma) + Complex64::new(0.0, t) * w).exp()
    };
    let head = |x: f64| f(saddle * x) * saddle;
    let tail = |s: f64| f(saddle + dir * s) * dir;

    let (h0, m0) = tanh_sinh(&head, H);
    let (h1, m1) = exp_sinh(&tail, H);
    let (c0, _) = tanh_sinh(&head, 2.0 * H);
    let (c1, _) = exp_sinh(&tail, 2.0 * H);

    let fine = h0 + h1;
    let coarse = c0 + c1;

    DeformedContourEval {
        value: fine / peak,
        digits_lost: ((m0 + m1) / fine.norm().max(f64::MIN_POSITIVE))
            .log10()
            .max(0.0),
        residual: (fine - coarse).norm() / fine.norm().max(f64::MIN_POSITIVE),
    }
}

/// Root of `φ'(ω) = β/ω − γω^(γ-1) + it`, seeded from the pure-oscillatory saddle offset by the
/// peak frequency so small `t` lands on the real root instead of wandering.
fn saddle(shape: Shape, t: f64) -> Complex64 {
    let Shape { beta, gamma } = shape;
    let mut w = Complex64::from_polar(
        (t / gamma).powf(1.0 / (gamma - 1.0)),
        PI / (2.0 * (gamma - 1.0)),
    ) + Complex64::new((beta / gamma).powf(1.0 / gamma), 0.0);

    for _ in 0..64 {
        let d1 = beta / w - gamma * w.powf(gamma - 1.0) + Complex64::new(0.0, t);
        w -= d1 / curvature(shape, w);
    }
    w
}

fn curvature(shape: Shape, w: Complex64) -> Complex64 {
    -shape.beta / (w * w) - shape.gamma * (shape.gamma - 1.0) * w.powf(shape.gamma - 2.0)
}

/// Steepest descent at the saddle, oriented away from the origin, clamped inside the sector where
/// `Re ω^γ > 0` keeps the closing arc at infinity dead.
fn tail_direction(shape: Shape, w: Complex64) -> Complex64 {
    let descent = 0.5 * (PI - curvature(shape, w).arg());
    let outward = if (descent - w.arg()).cos() > 0.0 {
        descent
    } else {
        descent + PI
    };
    let limit = 0.4 * PI / shape.gamma;
    Complex64::from_polar(1.0, outward.clamp(-limit, limit))
}

/// tanh-sinh over the unit interval; endpoint clustering absorbs the `ω^β` branch point at 0.
fn tanh_sinh<F: Fn(f64) -> Complex64>(g: F, h: f64) -> (Complex64, f64) {
    let n = (TAU_MAX / h) as i32;
    let mut re: Accumulator<f64> = Accumulator::default();
    let mut im: Accumulator<f64> = Accumulator::default();
    let mut mass: Accumulator<f64> = Accumulator::default();

    for k in -n..=n {
        let tau = k as f64 * h;
        let s = 0.5 * PI * tau.sinh();
        let x = 0.5 * (1.0 + s.tanh());
        let w = 0.25 * PI * tau.cosh() / (s.cosh() * s.cosh());
        let term = g(x) * w;
        if term.is_finite() {
            re.add(term.re);
            im.add(term.im);
            mass.add(term.norm());
        }
    }
    (Complex64::new(re.sum(), im.sum()) * h, mass.sum() * h)
}

/// exp-sinh over `[0, ∞)` for the exponentially decaying tail leg.
fn exp_sinh<F: Fn(f64) -> Complex64>(g: F, h: f64) -> (Complex64, f64) {
    let n = (TAU_MAX / h) as i32;
    let mut re: Accumulator<f64> = Accumulator::default();
    let mut im: Accumulator<f64> = Accumulator::default();
    let mut mass: Accumulator<f64> = Accumulator::default();

    for k in -n..=n {
        let tau = k as f64 * h;
        let s = (0.5 * PI * tau.sinh()).exp();
        let w = s * 0.5 * PI * tau.cosh();
        let term = g(s) * w;
        if term.is_finite() {
            re.add(term.re);
            im.add(term.im);
            mass.add(term.norm());
        }
    }
    (Complex64::new(re.sum(), im.sum()) * h, mass.sum() * h)
}
