// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Log-Tilted Saddle Jet
//!
//! `∫₀^∞ ω^β e^{-ω^γ} e^{iωt} dω` by steepest descent in `z = ln(ω/ρ)`.  The log coordinate
//! removes the branch point at the origin, so the saddle condition is the trinomial
//! `u^γ - iτu - 1 = 0` and every derivative of the phase is affine in the single number
//! `B = u^γ - 1`.
//!
//! Which saddles the original contour decomposes into is a membership vector `m`, not a
//! height ranking.  `m` is exact at `τ = 0` (roots of unity, `m = (1,0,…,0)`) and changes
//! only at Stokes crossings, where `Im Δ_jk` changes sign with `Re Δ_jk > 0`.  We march `τ`
//! from zero internally, carrying root identity and `m` together, so the result stays a pure
//! function of `t` and the caller owes us no grid continuity.
//!
//! Membership is integer.  Each contribution is an exact quadrature rather than a truncated
//! asymptotic series, so a flip adds a saddle at `e^{-Re Δ}` of the total and the result is
//! continuous without erfc smoothing — Stokes jumps are an artifact of truncation, not a
//! property of the integral.
//!
//! `γ ≥ 3` assumed.  `γ = 2` has a genuine caustic at `τ = 2` where two saddles coalesce and
//! `Φ''` vanishes; the normal-coordinate map degenerates there and would need a cubic form.

use num_complex::Complex64;
use std::f64::consts::TAU;

use super::spec::Shape;
use super::whatsleft::Accumulator;

const DK_ITERS: usize = 64;
const MARCH_BASE: usize = 24;
const MARCH_PER_TAU: f64 = 8.0;
const MARCH_MAX: usize = 512;
const LIVE: f64 = 0.5;

pub struct JetMorseResult {
    pub psi: Complex64,
    /// dψ/dt, from the same roots and the same paths.
    pub d: Complex64,
    /// Sum of per-saddle decimation estimates.  Absolute, same units as `psi`.
    pub residual: f64,
}

/// Caller owns: `beta > 0`, `gamma` a positive integer ≥ 3.
pub fn jet_morse(shape: Shape, u: f64) -> JetMorseResult {
    let Shape { beta, gamma } = shape;
    let g = gamma as usize;
    let rho = (beta / gamma).powf(1.0 / gamma);
    let t = TAU * u / rho;
    let tau = rho * t.abs() / beta;

    let (roots, weights) = march(g, gamma, beta, tau);
    let saddles: Vec<Saddle> = roots.iter().map(|&u| Saddle::new(u, gamma)).collect();
    let active: Vec<usize> = (0..g).filter(|&i| weights[i].abs() > LIVE).collect();

    let mut psi_re: Accumulator<f64> = Accumulator::default();
    let mut psi_im: Accumulator<f64> = Accumulator::default();
    let mut d_re: Accumulator<f64> = Accumulator::default();
    let mut d_im: Accumulator<f64> = Accumulator::default();
    let mut residual = 0.0_f64;

    for &i in &active {
        let s = &saddles[i];
        let w = weights[i];
        let p = s.quadrature(beta, gamma, rho, 1);
        let q = s.quadrature(beta, gamma, rho, 2);
        residual += (p.residual + q.residual) * w.abs();
        let pv = p.value * w;
        let qv = q.value * w * Complex64::i();
        psi_re.add(pv.re);
        psi_im.add(pv.im);
        d_re.add(qv.re);
        d_im.add(qv.im);
    }

    let psi = Complex64::new(psi_re.sum(), psi_im.sum()) / rho;
    let d = Complex64::new(d_re.sum(), d_im.sum()) / rho;

    if t < 0.0 {
        JetMorseResult {
            psi: psi.conj(),
            d: -d.conj(),
            residual: residual / rho,
        }
    } else {
        JetMorseResult {
            psi,
            d,
            residual: residual / rho,
        }
    }
}

/// Marches `τ` from the exact seed to `tau`, carrying root identity and membership.
/// Returns the roots at `tau` and their integer weights.
fn march(g: usize, gamma: f64, beta: f64, tau: f64) -> (Vec<Complex64>, Vec<f64>) {
    let steps = (MARCH_BASE + (MARCH_PER_TAU * tau) as usize).min(MARCH_MAX);

    let mut roots: Vec<Complex64> = (0..g)
        .map(|k| Complex64::from_polar(1.0, TAU * k as f64 / g as f64))
        .collect();
    let mut m = vec![0.0_f64; g];
    m[0] = 1.0;

    let mut prev_im: Vec<f64> = singulants(&roots, gamma, beta)
        .iter()
        .map(|d| d.im)
        .collect();

    for step in 1..=steps {
        let ti = tau * step as f64 / steps as f64;
        durand_kerner(&mut roots, g, ti);
        let delta = singulants(&roots, gamma, beta);
        let im: Vec<f64> = delta.iter().map(|d| d.im).collect();
        let held = m.clone();

        for j in 0..g {
            for k in 0..g {
                let idx = j * g + k;
                if j == k || delta[idx].re <= 0.0 || prev_im[idx] * im[idx] >= 0.0 {
                    continue;
                }
                m[k] += im[idx].signum() * held[j];
            }
        }

        prev_im = im;
    }

    (roots, m)
}

/// Row-major `Δ_jk = β(Φ_j - Φ_k)`.
fn singulants(roots: &[Complex64], gamma: f64, beta: f64) -> Vec<Complex64> {
    let phi: Vec<Complex64> = roots.iter().map(|&u| Saddle::new(u, gamma).phi).collect();
    let g = roots.len();
    let mut d = vec![Complex64::default(); g * g];
    for j in 0..g {
        for k in 0..g {
            d[j * g + k] = (phi[j] - phi[k]) * beta;
        }
    }
    d
}

struct Saddle {
    u: Complex64,
    b: Complex64,
    phi: Complex64,
}

/// Value plus the difference between the last two refinement levels.
struct Term {
    value: Complex64,
    residual: f64,
}

/// Trapezoid half-length in normal-coordinate units, from `sqrt(2 ln(1/ε))`.
const QUAD_REACH: f64 = 8.9;
/// Node density doublings.  Trapezoid convergence is geometric in the width of the
/// analyticity strip of `v(x)`, whose branch point is the neighboring saddle at distance
/// `~sqrt(2 Re Δ)`.  When saddles are close the strip is narrow and the base density is
/// not enough; each level halves `h`.
const QUAD_LEVELS: usize = 6;
const QUAD_BASE_DENSITY: f64 = 2.0;
const QUAD_TOL: f64 = 1e-13;
const NEWTON_ITERS: usize = 12;

impl Saddle {
    /// `B = u^γ - 1 = iτu`, and `Φ = ln u + B(1 - 1/γ) - 1/γ` follows from the saddle
    /// condition.  Integer power, not `powf`: the principal branch of `u^γ` disagrees with
    /// the polynomial once `γ·arg u` leaves `(-π, π]`, which is most of the root set.
    fn new(u: Complex64, gamma: f64) -> Self {
        let b = u.powi(gamma as i32) - 1.0;
        let phi = u.ln() + b * (1.0 - 1.0 / gamma) - 1.0 / gamma;
        Saddle { u, b, phi }
    }

    /// Refines until the decimation estimate clears tolerance.  Each level halves `h`, and
    /// the previous level's nodes are the even nodes of the next, so the estimate is the
    /// difference between consecutive levels.
    fn quadrature(&self, beta: f64, gamma: f64, rho: f64, m: i32) -> Term {
        let mut density = QUAD_BASE_DENSITY;
        let mut prev: Option<Complex64> = None;
        let mut best = Term {
            value: Complex64::default(),
            residual: f64::INFINITY,
        };

        let exponent = self.phi * beta + (beta + m as f64) * rho.ln();
        let scale = exponent.exp() * self.u.powi(m);

        for _ in 0..QUAD_LEVELS {
            let raw = self.trapezoid(beta, gamma, m, density);
            if let Some(p) = prev {
                let term = Term {
                    value: scale * raw,
                    residual: (scale * (raw - p)).norm(),
                };
                if term.residual < best.residual {
                    best = term;
                }
                if best.residual <= QUAD_TOL * best.value.norm() {
                    return best;
                }
            }
            prev = Some(raw);
            density *= 2.0;
        }
        best
    }

    /// One trapezoid pass at the given node density.  Steepest descent in the normal
    /// coordinate `x`, where `g(v) = -x²/2` exactly, so the Gaussian weight is
    /// `exp(-βx²/2)` by construction and cannot overflow — which a straight segment in `v`
    /// can and does, once the reach is long enough for `e^{γv}` to climb the neighboring
    /// hill.  The path is traced, not assumed: each node solves that equation by Newton
    /// from the previous node, so the contour follows the valley however it bends.
    ///
    /// `dv/dx = -x/g'(v)`, and at the saddle `g'` vanishes with `x`, so the limit `1/s₀`
    /// is used there.  Amplitude `u^m e^{mv}`; `m = 1` is ψ, `m = 2` is the t-derivative
    /// integrand.
    fn trapezoid(&self, beta: f64, gamma: f64, m: i32, density: f64) -> Complex64 {
        let phi2 = self.b * (1.0 - gamma) - gamma;
        let mut s0 = (-phi2).sqrt();
        if s0.re < 0.0 {
            s0 = -s0;
        }

        let bp1 = (self.b + 1.0) / gamma;
        let g = |v: Complex64| v + self.b * (v.exp() - 1.0) - bp1 * ((v * gamma).exp() - 1.0);
        let gp = |v: Complex64| 1.0 + self.b * v.exp() - (self.b + 1.0) * (v * gamma).exp();

        let h = 1.0 / (density * beta.sqrt());
        let n = (QUAD_REACH * density).ceil() as i32;

        let mut acc_re: Accumulator<f64> = Accumulator::default();
        let mut acc_im: Accumulator<f64> = Accumulator::default();

        for side in [1.0_f64, -1.0] {
            let mut v = Complex64::default();
            for j in 0..=n {
                if j == 0 && side < 0.0 {
                    continue;
                }
                let x = side * j as f64 * h;
                let dvdx = if j == 0 {
                    s0.inv()
                } else {
                    v += (side * h) / s0;
                    let target = Complex64::new(-0.5 * x * x, 0.0);
                    for _ in 0..NEWTON_ITERS {
                        let step = (g(v) - target) / gp(v);
                        v -= step;
                        if step.norm() < 1e-15 {
                            break;
                        }
                    }
                    -x / gp(v)
                };
                let f = (-0.5 * beta * x * x).exp() * (v * m as f64).exp() * dvdx;
                acc_re.add(f.re);
                acc_im.add(f.im);
            }
        }

        Complex64::new(acc_re.sum(), acc_im.sum()) * h
    }
}

/// Warm-started in place, so root identity survives the step.
fn durand_kerner(r: &mut [Complex64], gamma: usize, tau: f64) {
    for _ in 0..DK_ITERS {
        let mut moved = 0.0_f64;
        for k in 0..gamma {
            let p = r[k].powi(gamma as i32) - Complex64::i() * tau * r[k] - 1.0;
            let mut den = Complex64::new(1.0, 0.0);
            for j in 0..gamma {
                if j != k {
                    den *= r[k] - r[j];
                }
            }
            let step = p / den;
            r[k] -= step;
            moved = moved.max(step.norm());
        }
        if moved < 1e-15 {
            break;
        }
    }
}
