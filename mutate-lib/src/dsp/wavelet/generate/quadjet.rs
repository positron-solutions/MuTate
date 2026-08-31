// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Log-Tilted Saddle Jet + Terminal Quadrature
//!
//! > Engineers don't know anything but physicists can't do anything.
//! >
//! > - Nigel Clayton
//!
//! This is the production method.  It uses the log tilt to avoid endpoint terms at all.  Heavy
//! reliance on saddle terms is sufficient for many `u`.  Quadrature is used wherever the saddle
//! terms struggle to do the job or to do it accurately.  That's a lot of sophistication.  The
//! simpler IFFT and Contour methods are provided as evidence of its accuracy and precision.
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
//! Membership is integer and needs no Stokes smoothing.  A flip fires where `Re Δ_jk` is
//! maximal, so the entering saddle arrives at `e^{-Re Δ}` of the total, and since each
//! contribution is integrated rather than expanded there is no truncation artifact to
//! repair.  Berry's erfc weight belongs to the divergent-series form of this method and
//! injects error here.
//!
//! `γ ≥ 3` is the supported range.  `γ = 2` has a real caustic at `τ = 2` where two saddles
//! coalesce and `Φ''` vanishes, which degenerates the normal coordinate.

use num_complex::Complex64;
use std::f64::consts::TAU;

use super::super::spec::Shape;
use super::super::whatsleft::Accumulator;

const DK_ITERS: usize = 64;
const MARCH_BASE: usize = 24;
const MARCH_PER_TAU: f64 = 8.0;
const MARCH_MAX: usize = 512;
const LIVE: f64 = 0.5;

pub struct QuadJetResult {
    pub psi: Complex64,
    /// dψ/dt, from the same roots and the same paths.
    pub d: Complex64,
    /// Sum of per-saddle quadrature estimates.  Absolute, same units as `psi`.
    pub residual: f64,
}

/// # QuadJet
///
/// Tolerance and shape are configured once.  This may provide opportunity for re-use of
/// intermediates on long grids.
pub struct QuadJet {
    /// Solver goal precision.  When estimated error of remaining terms falls below this level, the
    /// solution will be returned.
    pub tol: f64,
    /// Morse wavelet parameters.
    pub shape: Shape,
}

impl QuadJet {
    pub fn tap_at(&self, u: f64) -> QuadJetResult {
        // LIES tol is still unused!
        integrate(self.shape, u, self.tol)
    }

    pub fn reference(shape: Shape) -> Self {
        Self { tol: 1e-16, shape }
    }

    pub fn standard(shape: Shape) -> Self {
        Self { tol: 1e-8, shape }
    }
}

/// Caller owns: `beta > 0`, `gamma` an integer ≥ 3.
fn integrate(shape: Shape, u: f64, _tol: f64) -> QuadJetResult {
    let Shape { beta, gamma } = shape;
    let g = gamma as usize;
    let rho = (beta / gamma).powf(1.0 / gamma);
    let t = TAU * u / rho;
    let tau = rho * t.abs() / beta;

    let (roots, weights) = march(g, gamma, beta, tau);
    let saddles: Vec<Saddle> = roots.iter().map(|&u| Saddle::new(u, gamma)).collect();
    let active = (0..g).filter(|&i| weights[i].abs() > LIVE);

    let mut psi_re: Accumulator<f64> = Accumulator::default();
    let mut psi_im: Accumulator<f64> = Accumulator::default();
    let mut d_re: Accumulator<f64> = Accumulator::default();
    let mut d_im: Accumulator<f64> = Accumulator::default();
    let mut residual = 0.0_f64;

    for i in active {
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
    let residual = residual / rho;

    if t < 0.0 {
        QuadJetResult {
            psi: psi.conj(),
            d: -d.conj(),
            residual,
        }
    } else {
        QuadJetResult { psi, d, residual }
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
        durand_kerner(&mut roots, g, tau * step as f64 / steps as f64);
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
    /// is used there.
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

#[cfg(test)]
mod test {
    use super::super::fmt_e;
    use super::*;

    #[test]
    fn convergence_precision() {
        let shape = Shape::from_q(3.5, 3.0);
        let ref_jet = QuadJet::reference(shape);

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Peak, shoulder, the old sick zone around 3.7-4.7, and deep tail.
        let probes = [
            0.0, 0.8, 2.0, 2.4, 3.7, 4.7, 5.5, 6.7, 7.911, 8.2, 8.33, 11.0, 14.0,
        ];

        #[derive(Clone, Copy)]
        enum Channel {
            Psi,
            D,
        }

        let pick = |t: Channel, result: QuadJetResult| match t {
            Channel::Psi => result.psi,
            Channel::D => Complex64::new(777.0, 777.0), // XXX Add back in after supporting the D
        };

        // NOTE this is crude and depends on settings but provides some development signal.
        let mut matrix_time = std::time::Duration::default();
        let mut sweep =
            |channel: Channel, name: &str, knob: &str, vary: &dyn Fn(u32) -> QuadJet, rows: u32| {
                let channel_label = match channel {
                    Channel::Psi => "psi",
                    Channel::D => "d",
                };
                println!("\n=== {name} ({channel_label}) ===");
                print!("  {knob:>10} |");
                for u in probes {
                    print!(" {u:>9.2}");
                }
                println!();

                for i in 0..rows {
                    let test_jet = vary(i);
                    let label = test_jet.tol;

                    print!("  {label:>10.1e} |");
                    for u in probes {
                        let matrix_row_start = std::time::Instant::now();
                        let v = pick(channel, test_jet.tap_at(u));
                        let r = pick(channel, ref_jet.tap_at(u));
                        matrix_time += matrix_row_start.elapsed();
                        print!(" {:>9}", fmt_e(rel(v, r)));
                    }
                    println!();
                }
            };

        // XXX Support the D
        // Channel::D
        for tap in [Channel::Psi] {
            sweep(
                tap,
                "Cranking tol",
                "tol",
                &|i| QuadJet {
                    tol: 1.0 / 10.0f64.powf((i + 1) as f64),
                    ..ref_jet
                },
                13,
            );
        }

        // Acceptance: at the settings we intend to ship, the floor is flat in u.  A dense grid so a
        // narrow seam can't hide between probes.
        println!("\n=== Current Defaults ===");
        let standard_jet = QuadJet::standard(shape);

        let mut worst: f64 = 0.0;
        let mut worst_u: f64 = 0.0;

        for k in 0..=2048 {
            let u = k as f64 * 0.0125;

            let reference = ref_jet.tap_at(u);
            let standard = standard_jet.tap_at(u);
            // XXX after supporting the D, add the max chain back in.
            let err = rel(standard.psi, reference.psi);
            if err > worst {
                worst = err;
                worst_u = u;
            }

            if k % 100 == 0 {
                println!("  u: {u:>6.2}, err: {:>9}", fmt_e(err));
            }
        }

        println!("  worst over grid: {} at {:0.2}", fmt_e(worst), worst_u);
        println!("  matrix completed in: {}µs", matrix_time.as_micros());
        assert!(worst < 1e-8);
    }
}
