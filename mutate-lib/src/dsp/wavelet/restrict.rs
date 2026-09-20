// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Restrict
//!
//! Squeeze the high-resolution mother wavelet grid into `N` taps.
//!
//! > Lab-grown beef is not technology.  The real technology is *grinding* it.  We are bringing to
//! > the world the first product of it's kind.  The narrow strings of beef are chopped, gathered by
//! > real artisans into a flat, circular form, and finally smashed into smithereens.
//! >
//! > *applause*
//! >
//! > - Professor Arima Nayar, MSc in Mathematics
//!
//! Naively, we may sample the motherlet at the point nearest to `k`, and tell our boss that we have
//! done the job.
//!
//! ## That is Not Good Enough
//!
//! At high omega, the motherlet is beginning to alias with N.  Her smooth curves will become
//! reduced to formless points, like pushing Michelangelo's David through a cheese grater.
//!
//! Instead, we want the result of the cheese grater to *look* the same to audio as the ideal
//! wavelet.  Since the audio has gone through roughly the same cheese grater, our job is made a
//! little bit simpler.  In the restriction step, we are mainly concerned with ensuring that the
//! shape of the wavelet we will refine is maximally well preserved every time we shove it through
//! the grater.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use num_complex::Complex64;

use super::{generate::hermite, spec::Shape, Fold, Grid, PEAK_GAIN};

/// How the motherlet lands on a cell.
#[derive(Clone, Copy, Default)]
pub enum Quadrature {
    /// ψ at the tap center.
    ///
    /// ```text
    /// h_k = ψ(kρ)
    /// ```
    Nearest,
    /// Cell mean of ψ.
    ///
    /// ```text
    /// h_k = (1/ρ) ∫_{C_k} ψ du
    /// ```
    ///
    /// The box is real and even about the cell, so the restriction commutes with input phase.
    Complex,
    /// Cell means of |ψ| and of the phase residual against the carrier, recombined.
    ///
    /// ```text
    /// h_k = ā_k · e^{i(2πkρ + θ̄_k)},  θ = arg(ψ e^{-2πiu})
    /// ```
    ///
    /// Nonlinear.  Intra-cell rotation cannot cancel magnitude, so the crest keeps its height.
    #[default]
    Axial,
    /// Trapezoidal footprint of width 3ρ over the cell means.
    ///
    /// ```text
    /// h_k = (B_{k-1} + 2 B_k + B_{k+1}) / 4
    /// ```
    Weighted,
}

/// How the reach ends.
#[derive(Clone, Copy, Default)]
pub enum Taper {
    /// Full amplitude to the last tap.
    Rectangle,
    /// The first discarded magnitude removed proportionally.
    ///
    /// ```text
    /// g_k = 1 − a_K / a_k
    /// ```
    ///
    /// `a_K < a_k` over the reach, so `g_k` is a gain in (0, 1) and no tap can change sign.
    #[default]
    Cylinder,
    /// A profile through `a_K` with curvature `κ`.
    ///
    /// ```text
    /// g_k = 1 − c_k / a_k,  c_k = a_K (1 + (1 − (k/K)^n) / κ)
    /// ```
    Knee { curvature: f64 },
}

#[derive(Clone, Copy, Default)]
pub struct Restriction {
    pub quadrature: Quadrature,
    pub taper: Taper,
    pub derivative: Derivative,
}

impl Restriction {
    /// Writes `out.len()` folded weights, `out[0]` real.
    pub(super) fn psi_into(self, grid: Grid, rho: f64, out: &mut [Complex64]) {
        let reach = out.len();
        let mut cells = vec![Complex64::default(); reach + 1];
        self.quadrature.cells_into(grid, rho, &mut cells);

        let gain = self.taper.gains(&cells);
        for (o, (c, g)) in out.iter_mut().zip(cells.iter().zip(gain)) {
            *o = c * g;
        }
        out[0].im = 0.0;
    }
}

impl Quadrature {
    /// Cell values for `k` in `[0, out.len())`, the last being the first discarded.
    fn cells_into(self, grid: Grid, rho: f64, out: &mut [Complex64]) {
        let inv = rho.recip();

        match self {
            Quadrature::Nearest => {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = grid.at(j as f64 * rho);
                }
            }
            Quadrature::Complex => {
                // the center cell is symmetric about u = 0, so the odd parts cancel
                out[0] = Complex64::new(2.0 * inv * grid.mass(0.0, 0.5 * rho).re, 0.0);
                for (j, o) in out.iter_mut().enumerate().skip(1) {
                    *o = inv * grid.mass((j as f64 - 0.5) * rho, (j as f64 + 0.5) * rho);
                }
            }
            Quadrature::Axial => {
                let (a, da) = magnitude(grid);
                let (t, dt) = residual(grid);
                let mean = |p: &[f64], dp: &[f64], lo: f64, hi: f64| {
                    inv * hermite::integrate_1d(p, dp, lo, hi, grid.du)
                };

                out[0] = Complex64::new(2.0 * mean(&a, &da, 0.0, 0.5 * rho), 0.0);
                for (j, o) in out.iter_mut().enumerate().skip(1) {
                    let (lo, hi) = ((j as f64 - 0.5) * rho, (j as f64 + 0.5) * rho);
                    let arg = TAU * j as f64 * rho + mean(&t, &dt, lo, hi);
                    let (s, c) = arg.sin_cos();
                    *o = mean(&a, &da, lo, hi) * Complex64::new(c, s);
                }
            }
            Quadrature::Weighted => {
                Quadrature::Complex.cells_into(grid, rho, out);

                // B_{-1} = conj(B_1), and one cell past the reach to close the stencil
                let past = inv
                    * grid.mass(
                        (out.len() as f64 - 0.5) * rho,
                        (out.len() as f64 + 0.5) * rho,
                    );
                let at = |j: isize| match j {
                    -1 => out[1].conj(),
                    j if j as usize == out.len() => past,
                    j => out[j as usize],
                };

                const AXIS: [f64; 3] = [0.25, 0.5, 0.25];
                let smooth: Vec<Complex64> = (0..out.len() as isize)
                    .map(|j| {
                        AXIS.iter()
                            .enumerate()
                            .map(|(i, &w)| w * at(j + i as isize - 1))
                            .sum()
                    })
                    .collect();
                out.copy_from_slice(&smooth);
            }
        }
    }
}

impl Taper {
    /// One nonnegative real gain per written tap, from the reach's first discarded magnitude.
    fn gains(self, cells: &[Complex64]) -> Vec<f64> {
        let reach = cells.len() - 1;
        let axis = reach as f64;
        let pedestal = cells[reach].norm();

        let profile: Box<dyn Fn(f64) -> f64> = match self {
            Taper::Rectangle => return vec![1.0; reach],
            Taper::Cylinder => Box::new(move |_| pedestal),
            // s = −K a'_K / a_K, one-sided since K is the last cell held
            Taper::Knee { curvature } => {
                let slope = pedestal - cells[reach - 1].norm();
                let n = curvature * (-axis * slope / pedestal);
                Box::new(move |j: f64| pedestal * (1.0 + (1.0 - (j / axis).powf(n)) / curvature))
            }
        };

        (0..reach)
            .map(|j| 1.0 - profile(j as f64) / cells[j].norm())
            .collect()
    }
}

/// `arg(ψ e^{-2πiu})` and its derivative.  `φ' = 2π Re(conj(ψ)·d)/|ψ|²`, so the residual sheds `2π`.
fn residual(grid: Grid<'_>) -> (Vec<f64>, Vec<f64>) {
    grid.psi
        .iter()
        .zip(grid.d)
        .enumerate()
        .map(|(i, (&psi, &d))| {
            let (s, c) = (TAU * i as f64 * grid.du).sin_cos();
            let turn = Complex64::new(c, -s);
            (
                (psi * turn).arg(),
                TAU * ((psi.re * d.re + psi.im * d.im) / psi.norm_sqr() - 1.0),
            )
        })
        .unzip()
}

/// `|ψ|` and `d|ψ|/du` = `Re(conj(ψ)·ψ')/|ψ| with ψ' = 2πi d`.
fn magnitude(grid: Grid<'_>) -> (Vec<f64>, Vec<f64>) {
    grid.psi
        .iter()
        .zip(grid.d)
        .map(|(&psi, &d)| {
            let a = psi.norm();
            (a, TAU * (psi.im * d.re - psi.re * d.im) / a)
        })
        .unzip()
}

/// How d is drawn from the emitted ψ.
#[derive(Clone, Copy, Default)]
pub enum Derivative {
    /// Stencil on the envelope demodulated at the carrier.
    ///
    /// ```text
    /// ψ_k = a_k e^{2πikρ}
    /// b_k = Σ_m c_m (a_{k+m} − a_{k−m})
    /// d_k = (a_k − (i/ω₀) b_k) e^{2πikρ}
    /// ```
    #[default]
    Envelope,
    /// |ω| smoothed at its kinks, as a real even convolution over the reach.
    ///
    /// ```text
    /// m(ω) = (G_σ ⊛ |ω|) / ω₀,  σ = min(ω̄, π − ω̄) / sigmas`
    /// ```
    ///
    /// Even, so a real tone reads D = m(ω) Ψ at every phase.
    Folded { sigmas: f64 },
    /// Weighted least squares over the reach against |ω| H.
    ///
    /// ```text
    /// min_d Σ_ω W(ω) |H_d(ω) − (|ω|/ω₀) H(ω)|²
    /// W(ω) = 1 / (H(|ω|)² + ε²),  ε = PEAK_GAIN · 10^{gate_db/20}
    /// ```
    ///
    /// W reads bias at +ω and image residual at −ω against the tone's own H(|ω|).
    Fitted { gate_db: f64 },
}

impl Derivative {
    /// Writes `psi.len()` folded weights of d reading ω/ω₀.  `rho` is the envelope carrier.
    pub(super) fn write(self, psi: &[Complex64], rho: f64, w0: f64, out: &mut [Complex64]) {
        match self {
            Derivative::Envelope => envelope(psi, rho, w0, out),
            Derivative::Folded { sigmas } => folded(psi, w0, sigmas, out),
            Derivative::Fitted { gate_db } => fitted(psi, w0, gate_db, out),
        }
    }
}

/// Central first difference, order 2·len.
const STENCIL: [f64; 3] = [0.75, -0.15, 1.0 / 60.0];

/// ```text
/// ψ_k = a_k e^{2πikρ}
/// b_k = Σ_m c_m (a_{k+m} − a_{k−m})
/// d_k = (2πρ a_k − i b_k) e^{2πikρ} / ω₀
/// ```
fn envelope(psi: &[Complex64], rho: f64, w0: f64, out: &mut [Complex64]) {
    let k = psi.len();

    // a_j = ψ_j e^{-2πijρ}, Hermitian
    let at = |j: isize| {
        let m = j.unsigned_abs();
        if m >= k {
            return Complex64::default();
        }
        let (s, c) = (TAU * m as f64 * rho).sin_cos();
        let a = psi[m] * Complex64::new(c, -s);
        if j < 0 {
            a.conj()
        } else {
            a
        }
    };

    let carrier = TAU * rho;
    let inv = w0.recip();

    for (j, o) in out.iter_mut().enumerate() {
        let n = j as isize;
        let b: Complex64 = STENCIL
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let m = i as isize + 1;
                c * (at(n + m) - at(n - m))
            })
            .sum();

        let (s, c) = (TAU * j as f64 * rho).sin_cos();
        *o = (carrier * at(n) - Complex64::i() * b) * inv * Complex64::new(c, s);
    }
}

/// `ψ_j` over the mirror, zero past the reach.
fn unfold(psi: &[Complex64], j: isize) -> Complex64 {
    match psi.get(j.unsigned_abs()) {
        Some(&p) if j < 0 => p.conj(),
        Some(&p) => p,
        None => Complex64::default(),
    }
}

/// `ω̄` = `arg Σ_ν ψ_{ν+1} conj ψ_ν`, the circular mean of `|H|²`
fn centroid(psi: &[Complex64]) -> f64 {
    psi.windows(2)
        .map(|p| p[1] * p[0].conj())
        .sum::<Complex64>()
        .arg()
}

/// `d_k = Σ_n h_n ψ_{k−n} / ω₀`
fn folded(psi: &[Complex64], w0: f64, sigmas: f64, out: &mut [Complex64]) {
    let reach = psi.len() as isize;
    let wc = centroid(psi);
    let sigma = wc.min(PI - wc) / sigmas;

    // G_σ ⊛ |ω| = π/2 − (4/π) Σ_{n odd} e^{−σ²n²/2} cos(nω) / n²
    let h: Vec<f64> = (0..2 * reach - 1)
        .map(|n| match n {
            0 => FRAC_PI_2,
            n if n % 2 == 0 => 0.0,
            n => {
                let n = n as f64;
                -2.0 / (PI * n * n) * (-0.5 * (sigma * n).powi(2)).exp()
            }
        })
        .collect();

    for (k, o) in out.iter_mut().enumerate() {
        let k = k as isize;
        *o = ((k + 1 - reach)..(k + reach))
            .map(|n| h[n.unsigned_abs()] * unfold(psi, k - n))
            .sum::<Complex64>()
            / w0;
    }
    out[0].im = 0.0;
}

/// Frequency samples per unfolded tap in the fit.
const FIT_OVERSAMPLE: usize = 4;

/// Normal equations over ω_j = 2πj/M.
///
/// ```text
/// Σ_μ ŵ(ν − μ) d_μ = b_ν
/// ŵ(n) = Σ_j W_j cos(ω_j n),  b_ν = Σ_j W_j T_j e^{iω_j ν},  T = (|ω|/ω₀) H
/// ```
fn fitted(psi: &[Complex64], w0: f64, gate_db: f64, out: &mut [Complex64]) {
    let reach = psi.len();
    let span = 2 * reach - 1;
    let m = (FIT_OVERSAMPLE * span).next_power_of_two();
    let eps = PEAK_GAIN * 10f64.powf(gate_db / 20.0);

    // H(ω_j)
    let fold = Fold::new(psi);
    let h: Vec<f64> = (0..m)
        .map(|j| fold.dtft(TAU * j as f64 / m as f64))
        .collect();

    let mut r = vec![0.0; span];
    let mut b = vec![Complex64::default(); reach];
    for (j, &hj) in h.iter().enumerate() {
        // |ω_j| on the mirror
        let pos = j.min(m - j);
        let w = (h[pos] * h[pos] + eps * eps).recip();
        let wt = w * hj * TAU * pos as f64 / (m as f64 * w0);

        // e^{iω_j n}
        let step = Complex64::from_polar(1.0, TAU * j as f64 / m as f64);
        let mut rot = Complex64::new(1.0, 0.0);
        for (rn, bn) in r.iter_mut().zip(b.iter_mut()) {
            *rn += w * rot.re;
            *bn += wt * rot;
            rot *= step;
        }
        for rn in &mut r[reach..] {
            *rn += w * rot.re;
            rot *= step;
        }
    }

    // n = ν + K − 1, b_{−ν} = conj b_ν
    let mut rhs = vec![Complex64::default(); span];
    for (v, &bv) in b.iter().enumerate() {
        rhs[reach - 1 + v] = bv;
        rhs[reach - 1 - v] = bv.conj();
    }

    let d = levinson(&r, &rhs);
    out.copy_from_slice(&d[reach - 1..]);
    out[0].im = 0.0;
}

/// `x` with `Σ_j r_{|i−j|} x_j = y_i`, `r` symmetric positive definite Toeplitz.
fn levinson(r: &[f64], y: &[Complex64]) -> Vec<Complex64> {
    let n = y.len();
    let r0 = r[0];
    let t = |i: usize| r[i] / r0;

    let mut x = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    x.push(y[0] / r0);
    v.push(-t(1));
    let (mut alpha, mut beta) = (-t(1), 1.0);

    for k in 1..n {
        beta *= 1.0 - alpha * alpha;

        let mu = (y[k] / r0 - (0..k).map(|i| t(i + 1) * x[k - 1 - i]).sum::<Complex64>()) / beta;
        for i in 0..k {
            x[i] += mu * v[k - 1 - i];
        }
        x.push(mu);

        if k < n - 1 {
            alpha = (-t(k + 1) - (0..k).map(|i| t(i + 1) * v[k - 1 - i]).sum::<f64>()) / beta;
            for i in 0..k / 2 {
                let (a, c) = (v[i], v[k - 1 - i]);
                v[i] = a + alpha * c;
                v[k - 1 - i] = c + alpha * a;
            }
            if k % 2 == 1 {
                v[k / 2] *= 1.0 + alpha;
            }
            v.push(alpha);
        }
    }
    x
}

#[cfg(test)]
mod test {
    use super::super::{Bake, Fold, WaveletSpec, PEAK_GAIN};
    use super::*;

    const TAIL_DB: f64 = -60.0;
    const GAMMAS: [f64; 3] = [2.0, 3.0, 4.0];
    const QS: [f64; 4] = [3.5, 5.0, 8.5, 12.5];
    const RHOS: [f64; 7] = [0.02, 0.06, 0.116, 0.189, 0.25, 0.312, 0.384];

    const QUADRATURES: [(&str, Quadrature); 4] = [
        ("nearest", Quadrature::Nearest),
        ("complex", Quadrature::Complex),
        ("axial", Quadrature::Axial),
        ("weighted", Quadrature::Weighted),
    ];

    const TAPERS: [(&str, Taper); 3] = [
        ("rect", Taper::Rectangle),
        ("cyl", Taper::Cylinder),
        ("knee", Taper::Knee { curvature: 1.0 }),
    ];

    fn methods() -> Vec<(String, Restriction)> {
        QUADRATURES
            .into_iter()
            .flat_map(|(qn, quadrature)| {
                TAPERS.into_iter().map(move |(tn, taper)| {
                    let r = Restriction {
                        quadrature,
                        taper,
                        derivative: Derivative::Envelope,
                    };
                    (format!("{qn}/{tn}"), r)
                })
            })
            .collect()
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// Sanity across every quadrature and taper over γ × Q × ρ.  Differences by design pass.
    /// A lane fails only when it is broken on its own terms and the consensus does not share it.
    ///
    /// ```text
    /// pass   H(ω₀) > ½ PEAK_GAIN
    /// image  |H(−ω₀)| < 0.1 H(ω₀)
    /// body   ‖|h| − ā‖ / ‖ā‖,  ā the per tap median envelope
    /// bump   r_k = ln(|h_{k+1}| / |h_k|) > 0  and  r_k − median r_k > tol
    /// ```
    #[test]
    fn restrictions_are_sane() {
        /// Taps below this fraction of the lane's crest do not vote.
        const FLOOR: f64 = 1e-6;
        const PASS: f64 = 0.5;
        const IMAGE: f64 = 0.1;
        // NOTE initial guesses, recalibrate from the first run.
        const BODY_TOL: f64 = 0.3;
        const RISE_TOL: f64 = 0.05;

        let methods = methods();
        let mut bake = Bake::default();
        let mut failures = Vec::new();

        println!(
            "\n=== RESTRICTION SANITY (tail {TAIL_DB:.0} dB, {} methods) ===",
            methods.len()
        );
        println!(
            "  {:>4} {:>5} {:>6} {:>5} {:>9} {:>16} {:>9} {:>16} {:>4}",
            "γ", "Q", "ρ", "taps", "body", "worst", "bump", "worst", "k"
        );

        for gamma in GAMMAS {
            for q in QS {
                let mut wav = WaveletSpec::default()
                    .with_shape(Shape::from_q(q, gamma))
                    .max_truncation(TAIL_DB)
                    .bake();

                for rho in RHOS {
                    let w0 = TAU * rho;

                    let lanes: Vec<Vec<Complex64>> = methods
                        .iter()
                        .map(|&(_, r)| {
                            wav.restriction = r;
                            wav.at_rho(rho).write(&mut bake);
                            bake.weights.psi.clone()
                        })
                        .collect();
                    let k = lanes[0].len();

                    let mags: Vec<Vec<f64>> = lanes
                        .iter()
                        .map(|l| l.iter().map(|h| h.norm()).collect())
                        .collect();

                    // ā_k
                    let env: Vec<f64> = (0..k)
                        .map(|j| median(mags.iter().map(|m| m[j]).collect()))
                        .collect();
                    // ‖ā‖
                    let env_l2 = env.iter().map(|a| a * a).sum::<f64>().sqrt();

                    // ln(|h_{k+1}| / |h_k|)
                    let rise = |m: &[f64], j: usize| (m[j + 1] / m[j]).ln();
                    // median r_k
                    let consensus: Vec<f64> = (0..k - 1)
                        .map(|j| median(mags.iter().map(|m| rise(m, j)).collect()))
                        .collect();

                    let mut worst_body = (0.0f64, "");
                    let mut worst_bump = (f64::NEG_INFINITY, "", 0usize);

                    for ((name, _), (lane, mag)) in methods.iter().zip(lanes.iter().zip(&mags)) {
                        let tag = format!("γ {gamma} Q {q} ρ {rho} {name}");

                        if !lane.iter().all(|h| h.is_finite()) {
                            failures.push(format!("{tag} non-finite tap"));
                            continue;
                        }

                        // carrier and image
                        let psi = Fold::new(lane);
                        let (pass, image) = (psi.dtft(w0), psi.dtft(-w0).abs());
                        if pass < PASS * PEAK_GAIN {
                            failures.push(format!("{tag} H(ω₀) {pass:.4}"));
                        }
                        if image > IMAGE * pass.abs() {
                            failures.push(format!(
                                "{tag} image {:.2} dB",
                                20.0 * (image / pass.abs()).log10()
                            ));
                        }

                        // ‖|h| − ā‖ / ‖ā‖
                        let body = mag
                            .iter()
                            .zip(&env)
                            .map(|(a, e)| (a - e).powi(2))
                            .sum::<f64>()
                            .sqrt()
                            / env_l2;
                        if body > worst_body.0 {
                            worst_body = (body, name);
                        }
                        if body > BODY_TOL {
                            failures.push(format!("{tag} body {body:.3}"));
                        }

                        // outward rise beyond the consensus
                        let crest = mag.iter().fold(0.0f64, |a, &b| a.max(b));
                        for j in 0..k - 1 {
                            if mag[j + 1] < FLOOR * crest {
                                continue;
                            }
                            let r = rise(mag, j);
                            if r <= 0.0 {
                                continue;
                            }
                            let excess = r - consensus[j];
                            if excess > worst_bump.0 {
                                worst_bump = (excess, name, j + 1);
                            }
                            if excess > RISE_TOL {
                                failures.push(format!(
                                    "{tag} bump at k {} rise {r:.4} consensus {:.4}",
                                    j + 1,
                                    consensus[j]
                                ));
                            }
                        }
                    }

                    println!(
                        "  {gamma:>4.1} {q:>5.1} {rho:>6.3} {:>5} {:>9.2e} {:>16} {:>9.2e} {:>16} {:>4}",
                        2 * k - 1,
                        worst_body.0,
                        worst_body.1,
                        worst_bump.0.max(0.0),
                        worst_bump.1,
                        worst_bump.2,
                    );
                }
            }
        }

        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }
}
