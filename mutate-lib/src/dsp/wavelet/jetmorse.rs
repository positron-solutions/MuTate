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

use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

use super::spec::Shape;
use super::whatsleft::Accumulator;

const R_MAX: usize = 14;
const DK_ITERS: usize = 64;
const MARCH_BASE: usize = 24;
const MARCH_PER_TAU: f64 = 8.0;
const MARCH_MAX: usize = 512;
const LIVE: f64 = 0.5;

pub struct JetMorseResult {
    pub psi: Complex64,
    /// dψ/dt, from the same roots and the same jets.
    pub d: Complex64,
    /// Sum of per-saddle smallest terms.  Absolute, same units as `psi`.
    pub residual: f64,
}

/// Caller owns: `beta > 0`, `gamma` a positive integer ≥ 2.
pub fn jet_morse(shape: Shape, u: f64) -> JetMorseResult {
    let Shape { beta, gamma } = shape;
    let g = gamma as usize;
    let rho = (beta / gamma).powf(1.0 / gamma);
    let t = TAU * u / rho;
    let tau = rho * t.abs() / beta;

    let (roots, weights) = march(g, gamma, beta, tau);
    let saddles: Vec<Saddle> = roots.iter().map(|&u| Saddle::new(u, gamma)).collect();
    let active: Vec<usize> = (0..g).filter(|&i| weights[i].abs() > LIVE).collect();
    let dominant = *active
        .iter()
        .max_by(|&&a, &&b| saddles[a].phi.re.partial_cmp(&saddles[b].phi.re).unwrap())
        .unwrap();

    // println!(
    //     "u {u:>6.3} | w {:?} | ReΔ {:?}",
    //     weights,
    //     (0..g)
    //         .map(|i| (saddles[dominant].phi - saddles[i].phi).re * beta)
    //         .collect::<Vec<_>>(),
    // );

    let mut psi_re: Accumulator<f64> = Accumulator::default();
    let mut psi_im: Accumulator<f64> = Accumulator::default();
    let mut d_re: Accumulator<f64> = Accumulator::default();
    let mut d_im: Accumulator<f64> = Accumulator::default();
    let mut residual = 0.0_f64;

    for &i in &active {
        let s = &saddles[i];
        let w = weights[i];
        let p = s.pick(beta, gamma, rho, 1);
        let q = s.pick(beta, gamma, rho, 2);
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

    // println!("order: {}", order);

    if t < 0.0 {
        // println!("residual: {}", residual);
        JetMorseResult {
            psi: psi.conj(),
            d: -d.conj(),
            residual: residual / rho,
        }
    } else {
        // println!("residual: {}", residual);
        JetMorseResult {
            psi,
            d,
            residual: residual / rho,
        }
    }
}

/// Last Stokes crossing seen on one ordered pair: the sign of the increment and the
/// dominant saddle's membership at the moment it fired.
#[derive(Clone, Copy)]
struct Crossing {
    s: f64,
    mj: f64,
}

/// Marches `τ` from the exact seed to `tau`, carrying root identity and membership.
/// Returns the roots at `tau` and their real weights.
fn march(g: usize, gamma: f64, beta: f64, tau: f64) -> (Vec<Complex64>, Vec<f64>) {
    let steps = (MARCH_BASE + (MARCH_PER_TAU * tau) as usize).min(MARCH_MAX);

    let mut roots: Vec<Complex64> = (0..g)
        .map(|k| Complex64::from_polar(1.0, TAU * k as f64 / g as f64))
        .collect();
    let mut m = vec![0.0_f64; g];
    m[0] = 1.0;

    let mut delta = singulants(&roots, gamma, beta);
    let mut prev_im: Vec<f64> = delta.iter().map(|d| d.im).collect();
    let mut last: Vec<Option<Crossing>> = vec![None; g * g];

    for step in 1..=steps {
        let ti = tau * step as f64 / steps as f64;
        durand_kerner(&mut roots, g, ti);
        delta = singulants(&roots, gamma, beta);
        let im: Vec<f64> = delta.iter().map(|d| d.im).collect();
        let held = m.clone();

        for j in 0..g {
            for k in 0..g {
                let idx = j * g + k;
                if j == k || delta[idx].re <= 0.0 || prev_im[idx] * im[idx] >= 0.0 {
                    continue;
                }
                let s = im[idx].signum();
                m[k] += s * held[j];
                last[idx] = Some(Crossing { s, mj: held[j] });
            }
        }

        if step < steps {
            prev_im = im;
        }
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

/// Value plus the magnitude of the smallest term, which is the standard error estimate for
/// a divergent asymptotic series truncated at its optimal order.
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

    /// `Φ⁽ᵏ⁾ = B(1 - γ^{k-1}) - γ^{k-1}` for `k ≥ 2`; `g_k = Φ⁽ᵏ⁾/k!`.
    fn shifted_phase(&self, gamma: f64, n: usize) -> Vec<Complex64> {
        let mut g = vec![Complex64::default(); n + 3];
        let mut fact = 1.0;
        for k in 2..g.len() {
            fact *= k as f64;
            let gk = gamma.powi(k as i32 - 1);
            g[k] = (self.b * (1.0 - gk) - gk) / fact;
        }
        g
    }

    /// Amplitude `u^m e^{mv}`; `m = 1` is ψ, `m = 2` is the t-derivative integrand.
    ///
    /// The series in `1/β` diverges; it is summed only until the terms stop shrinking.
    /// `cap` guards the array sizes, it is not a target — at small `β` the turnaround
    /// comes after three or four terms and everything past it is noise.
    fn contribution(&self, beta: f64, gamma: f64, rho: f64, m: i32, cap: usize) -> Term {
        let n = 2 * cap + 1;
        let g = self.shifted_phase(gamma, n);

        let h: Vec<Complex64> = (0..=n).map(|j| g[j + 2] * -2.0).collect();
        let s = series_sqrt(&h);
        let q = series_inv(&s);

        let mut amp = vec![Complex64::default(); n + 1];
        let mut fact = 1.0;
        for (j, a) in amp.iter_mut().enumerate() {
            if j > 0 {
                fact *= j as f64;
            }
            *a = Complex64::new((m as f64).powi(j as i32) / fact, 0.0);
        }

        let mut sum_re: Accumulator<f64> = Accumulator::default();
        let mut sum_im: Accumulator<f64> = Accumulator::default();
        let mut p = q.clone();
        let mut dfact = 1.0;
        let mut smallest = f64::INFINITY;

        for r in 0..=cap {
            if r > 0 {
                p = series_mul(&series_mul(&p, &q), &q);
                dfact *= (2 * r - 1) as f64;
            }
            let mut c = Complex64::default();
            for j in 0..=2 * r {
                c += amp[j] * p[2 * r - j];
            }
            let term = c * dfact / beta.powi(r as i32);
            let mag = term.norm();
            if mag > smallest {
                break;
            }
            smallest = mag;
            sum_re.add(term.re);
            sum_im.add(term.im);
        }

        let series = Complex64::new(sum_re.sum(), sum_im.sum());
        let exponent = self.phi * beta + (beta + m as f64) * rho.ln();
        let prefactor = exponent.exp() * (2.0 * PI / beta).sqrt() * self.u.powi(m);
        Term {
            value: prefactor * series,
            residual: (prefactor * smallest).norm(),
        }
    }

    /// Series when its own error estimate clears tolerance, quadrature otherwise.  The
    /// series wins only in a narrow band; outside it the quadrature is both cheaper and
    /// unbounded in accuracy.
    fn pick(&self, beta: f64, gamma: f64, rho: f64, m: i32) -> Term {
        let quad = self.quadrature(beta, gamma, rho, m);
        if quad.residual <= QUAD_TOL * quad.value.norm() {
            return quad;
        }
        let series = self.contribution(beta, gamma, rho, m, R_MAX);
        if series.residual < quad.residual {
            series
        } else {
            quad
        }
    }

    /// Steepest descent in the normal coordinate `x`, where `g(v) = -x²/2` exactly.
    ///
    /// The path is traced, not assumed: each node solves that equation by Newton from the
    /// previous node, so the contour follows the valley however it bends.  The weight is
    /// then `exp(-βx²/2)` by construction and cannot overflow — which a straight segment
    /// in `v` can and does, once the reach is long enough for `e^{γv}` to climb the
    /// neighboring hill.
    ///
    /// `dv/dx = -x/g'(v)`, and at the saddle `g'` vanishes with `x`, so the limit `1/s₀`
    /// is used there.
    ///
    /// Refines until the decimation estimate clears tolerance.  Each level halves `h`,
    /// and the previous level's nodes are the even nodes of the next, so the estimate is
    /// the difference between consecutive levels.
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
    /// `exp(-βx²/2)` by construction and cannot overflow.  `dv/dx = -x/g'(v)`, with the
    /// limit `1/s₀` at the saddle.
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

fn series_mul(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let n = a.len();
    let mut c = vec![Complex64::default(); n];
    for i in 0..n {
        for j in 0..n - i {
            c[i + j] += a[i] * b[j];
        }
    }
    c
}

/// Branch fixed by `Re(1/s₀) > 0`: the descent contour runs toward increasing `Re z`.
fn series_sqrt(h: &[Complex64]) -> Vec<Complex64> {
    let mut s = vec![Complex64::default(); h.len()];
    s[0] = h[0].sqrt();
    if s[0].re < 0.0 {
        s[0] = -s[0];
    }
    for j in 1..h.len() {
        let mut acc = h[j];
        for i in 1..j {
            acc -= s[i] * s[j - i];
        }
        s[j] = acc / (s[0] * 2.0);
    }
    s
}

fn series_inv(s: &[Complex64]) -> Vec<Complex64> {
    let mut q = vec![Complex64::default(); s.len()];
    q[0] = s[0].inv();
    for j in 1..s.len() {
        let mut acc = Complex64::default();
        for i in 1..=j {
            acc += s[i] * q[j - i];
        }
        q[j] = -acc * q[0];
    }
    q
}
