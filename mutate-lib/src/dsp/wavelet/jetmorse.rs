// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Log-Tilted Saddle Jet
//!
//! `∫₀^∞ ω^β e^{-ω^γ} e^{iωt} dω` by steepest descent in `z = ln(ω/ρ)`.  The log coordinate
//! removes the branch point at the origin, so the saddle condition is the trinomial
//! `u^γ - iτu - 1 = 0` and every derivative of the phase is affine in the single number
//! `B = u^γ - 1`.  Saddle selection is a dominance sort plus an erfc weight; there is no path
//! tracing and no continuation, so the result is a pure function of `t`.

// 🤖 Of course.

use libm::erfc;
use num_complex::Complex64;
use std::f64::consts::{PI, TAU};

use super::spec::Shape;
use super::whatsleft::Accumulator;

const R_MAX: usize = 14;
const DK_ITERS: usize = 200;

pub struct JetMorseResult {
    pub psi: Complex64,
    /// dψ/dt, from the same roots and the same jets.
    pub d: Complex64,
}

/// Caller owns: `beta > 0`, `gamma` a positive integer ≥ 2.
pub fn jet_morse(shape: Shape, u: f64) -> JetMorseResult {
    let Shape { beta, gamma } = shape;
    let rho = (beta / gamma).powf(1.0 / gamma);
    let t = TAU * u / rho;
    let tau = rho * t.abs() / beta;

    let saddles: Vec<Saddle> = durand_kerner(gamma as usize, tau)
        .into_iter()
        .filter(|u| u.im >= -1e-12)
        .map(|u| Saddle::new(u, gamma))
        .collect();

    let primary = saddles
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.phi.re.partial_cmp(&b.1.phi.re).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let singulants: Vec<Complex64> = saddles
        .iter()
        .map(|s| (saddles[primary].phi - s.phi) * beta)
        .collect();

    let order = singulants
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != primary)
        .map(|(_, d)| d.norm())
        .fold(f64::INFINITY, f64::min)
        .clamp(1.0, R_MAX as f64) as usize;

    let mut psi_re: Accumulator<f64> = Accumulator::default();
    let mut psi_im: Accumulator<f64> = Accumulator::default();
    let mut d_re: Accumulator<f64> = Accumulator::default();
    let mut d_im: Accumulator<f64> = Accumulator::default();

    for (i, s) in saddles.iter().enumerate() {
        let w = if i == primary {
            1.0
        } else {
            let delta = singulants[i];
            0.5 * erfc(-delta.im / (2.0 * delta.re).sqrt())
        };
        if w == 0.0 {
            continue;
        }
        let p = s.contribution(beta, gamma, rho, 1, order) * w;
        let q = s.contribution(beta, gamma, rho, 2, order) * w * Complex64::i();
        psi_re.add(p.re);
        psi_im.add(p.im);
        d_re.add(q.re);
        d_im.add(q.im);
    }

    // let psi = Complex64::new(psi_re.sum(), psi_im.sum());
    // let d = Complex64::new(d_re.sum(), d_im.sum());

    let psi = Complex64::new(psi_re.sum(), psi_im.sum()) / rho;
    let d = Complex64::new(d_re.sum(), d_im.sum()) / rho;

    if t < 0.0 {
        JetMorseResult {
            psi: psi.conj(),
            d: -d.conj(),
        }
    } else {
        JetMorseResult { psi, d }
    }
}

struct Saddle {
    u: Complex64,
    b: Complex64,
    phi: Complex64,
}

impl Saddle {
    /// `B = u^γ - 1 = iτu`, and `Φ = ln u + B(1 - 1/γ) - 1/γ` follows from the saddle condition.
    fn new(u: Complex64, gamma: f64) -> Self {
        let b = u.powf(gamma) - 1.0;
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
    fn contribution(&self, beta: f64, gamma: f64, rho: f64, m: i32, order: usize) -> Complex64 {
        let n = 2 * order + 1;
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

        for r in 0..=order {
            if r > 0 {
                p = series_mul(&series_mul(&p, &q), &q);
                dfact *= (2 * r - 1) as f64;
            }
            let mut c = Complex64::default();
            for j in 0..=2 * r {
                c += amp[j] * p[2 * r - j];
            }
            let term = c * dfact / beta.powi(r as i32);
            sum_re.add(term.re);
            sum_im.add(term.im);
        }

        let series = Complex64::new(sum_re.sum(), sum_im.sum());
        let exponent = self.phi * beta + (beta + m as f64) * rho.ln();
        exponent.exp() * (2.0 * PI / beta).sqrt() * self.u.powi(m) * series
    }
}

fn durand_kerner(gamma: usize, tau: f64) -> Vec<Complex64> {
    let radius = (1.0 + tau).powf(1.0 / (gamma as f64 - 1.0));
    let mut r: Vec<Complex64> = (0..gamma)
        .map(|k| Complex64::from_polar(radius, 2.0 * PI * k as f64 / gamma as f64 + 0.3))
        .collect();

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
    r
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
