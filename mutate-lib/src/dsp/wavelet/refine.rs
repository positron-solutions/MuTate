// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Refine
//!
//! Polish off the worst of the spectral and transient response artifacts from the imperfect
//! restriction of the finite wavelet.  Write the final weights to `f32` output in a precision and
//! goal-aware manner.
//!
//! > If the state is captive to narrow interests, then why do you wait for the state to organize
//! > what you claim to be an overwhelming mass of your own interests?
//! >
//! > - Major Edward Wuncler III, Halliburton War Veteran
//!
//! ⚠️ This module has been recently re-developed.  Doc comments are design outputs, not
//! necessarilly fully implemented.
//!
//! ## Motivation
//!
//! [`Restriction`](super::restrict) approximates an infinite ideal on a finite support.  Finite
//! support implies truncation.  Taper has been found useful at controlling the damage of
//! truncation.  Spectrally aware tuning of taper application can enhance the overall filter
//! performance while re-establishing characteristics where the virtues they guarantee are just as
//! true for the finite FIR as they were for the infinite ideal wavelet.
//!
//! ## Proper Constraints
//!
//! The dilemma of designing the response with any spectral approach is that improper solver
//! constraints can, for example, pursue an equiripple spectral response to pure tones that is
//! nonetheless extremely fragile to transients, a result that will not perform well at all in
//! practice.  Physically, we can intuit that the pedestals and spiky filter shapes resulting from
//! pure-tone optimization must generate unwanted, very non-smooth transient responses.
//!
//! A reasonable constraint envelope may begin with what dead reckoning solutions for tapering would
//! take away.  The [`Cylinder`](super::restrict::Taper::Cylinder) taper extracts mass equal to
//! the `K + 1` tap across the entire support.  If our deviations land between or not far from the
//! region bracketed by native truncation and our taper, the shape is guaranteed to be quite
//! Morse-like no matter what the solver does.  If we consider distance from the Morse curvature as
//! an extra element of cost, we can balance the intuitive transient robustness of the base Morse
//! shape with further spectral tuning.
//!
//! The ideal Morse wavelet does **not** have a constant angular velocity but instead has a phase
//! curvature that shapes the main lobe. This chirped carrier and its phase curvature show up as a
//! degree of freedom when, in order to balance the moments, we move filter mass along the spiral,
//! changing its phase.  We may also twist the spiral faster or slower, changing the phase curvature
//! and pushing mass around the axis.  Both modifying the spiral and moving mass along the spiral
//! change the filter's net angle and magnitude.
//!
//! ## Solver Goals
//!
//! > Optimal wavelet behavior is then found to be achieved by minimizing third central moments in
//! > both the frequency and the time domains.
//!
//! — Lilly & Olhede, [*Higher-Order Properties of Analytic Wavelets*][lilly-olhede] (2009).
//!
//! Some refinements are aiming to restore analytically mandatory characteristics.  Others are
//! spectrally opportunistic and take advantage of the free real-estate afforded where truncation
//! left no choice but to step into the non-analytic FIR optimization space.  The final goals
//! minimize changes and therefore prevent a walk too far from robust transient response inherent to
//! the basic Morse shape.
//!
//! The first side lobes, main lobe width (especially the transition band), and noise floor shape
//! may all have some low hanging fruit, features that can be directly tied to specific weights.  If
//! polishing these features away can be accomplished within the limited support and the limited
//! freedom of truncation, that's a win.
//!
//! [lilly-olhede]: https://static.aminer.org/pdf/PDF/001/231/038/higher_order_properties_of_analytic_wavelets.pdf

// NOTE solver direction is headed towards a combination of probes from the inspect module, analytic
// conditions like the jet terms and moments creating a null space, and likely some probes against
// reassignment problems.

use core::f64::consts::{FRAC_PI_2, PI, TAU};

use num_complex::Complex64;

use super::Shape;

/// Conditions restored after restriction.
#[derive(Clone, Copy, Debug)]
pub enum Refinement {
    None,
    /// Radial profiles on the spiral over its last `turns`, least norm, spent into the support comb.
    ///
    /// ```text
    /// M_p = 0,  p ∈ moments ∩ [0, β)
    /// δH⁽ᵖ⁾(ω₀) = 0,  p ∈ jet
    /// ```
    Reach {
        moments: &'static [u32],
        jet: &'static [u32],
        turns: f64,
        tangent: bool,
        spare: usize,
    },
}

impl Default for Refinement {
    fn default() -> Self {
        Self::Reach {
            moments: &[0],
            jet: &[0, 1, 2],
            turns: 3.0,
            tangent: false,
            spare: 16,
        }
    }
}

impl Refinement {
    pub(super) fn apply(self, psi: &mut [Complex64], shape: Shape, rho: f64) {
        let Refinement::Reach {
            moments,
            jet,
            turns,
            tangent,
            spare,
        } = self
        else {
            return;
        };
        let len = psi.len();
        let legal = shape.vanishing_moments();
        let w0 = TAU * rho;

        // M_p = 0
        let mut rows: Vec<(Functional, f64)> = moments
            .iter()
            .copied()
            .filter(|p| legal.contains(p))
            .map(|p| (Functional::response(len, 0.0, p), 0.0))
            .collect();
        // H⁽ᵖ⁾(ω₀) held
        rows.extend(jet.iter().map(|&p| {
            let f = Functional::response(len, w0, p);
            let held = f.eval(psi);
            (f, held)
        }));

        frenet(
            psi,
            &Profiles::edge(len, turns / rho, rows.len() + 1 + spare),
            tangent,
            &rows,
        );
    }
}

/// Radial profiles along the spiral's Frenet frame.
///
/// ```text
/// δ_k = Σ_j μ_j b_j(k) e_j(k),  e ∈ {u_k, i u_k},  u_k = ψ_k / |ψ_k|
/// G_ij = Σ_k Re(c̄_ik b_j(k) e_j(k))
/// μ = Gᵀλ,  G Gᵀ λ = t − Cψ
/// ```
pub(super) fn frenet(
    psi: &mut [Complex64],
    profiles: &Profiles,
    tangent: bool,
    rows: &[(Functional, f64)],
) {
    let unit: Vec<Complex64> = psi.iter().map(|p| p / p.norm()).collect();

    // b_j(k) e_j(k)
    let turns: &[Complex64] = match tangent {
        true => &[Complex64::ONE, Complex64::I],
        false => &[Complex64::ONE],
    };

    // Hermitian atoms, the center tap real
    let atoms = profiles.0.iter().flat_map(|b| {
        turns.iter().map(|t| {
            let mut a: Vec<Complex64> = b.iter().zip(&unit).map(|(b, u)| b * t * u).collect();
            a[0].im = 0.0;
            a
        })
    });

    // ⟨a, b⟩ = Σ_k m_k Re(ā_k b_k),  m₀ = 1,  m_k = 2
    let inner = |a: &[Complex64], b: &[Complex64]| -> f64 {
        a.iter()
            .zip(b)
            .enumerate()
            .map(|(k, (a, b))| if k == 0 { 1.0 } else { 2.0 } * (a.conj() * b).re)
            .sum()
    };

    // orthonormal span, least ‖δ‖ under min ‖μ‖
    let mut basis: Vec<Vec<Complex64>> = Vec::new();
    for mut a in atoms {
        let before = inner(&a, &a).sqrt();
        for _ in 0..2 {
            for e in &basis {
                let c = inner(e, &a);
                for (x, y) in a.iter_mut().zip(e) {
                    *x -= c * y;
                }
            }
        }
        let norm = inner(&a, &a).sqrt();
        if norm > 1e-10 * before {
            a.iter_mut().for_each(|x| *x /= norm);
            basis.push(a);
        }
    }
    let atoms = basis;

    let g: Vec<Vec<f64>> = rows
        .iter()
        .map(|(r, _)| {
            atoms
                .iter()
                .map(|a| {
                    r.0.iter()
                        .zip(a)
                        .map(|(c, a)| c.re * a.re + c.im * a.im)
                        .sum()
                })
                .collect()
        })
        .collect();

    let n = rows.len();
    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v: f64 = g[i].iter().zip(&g[j]).map(|(a, b)| a * b).sum();
            gram[i * n + j] = v;
            gram[j * n + i] = v;
        }
    }

    let mut lambda: Vec<f64> = rows.iter().map(|(r, t)| t - r.eval(psi)).collect();
    cholesky(&mut gram, &mut lambda);

    for (j, a) in atoms.iter().enumerate() {
        let mu: f64 = g.iter().zip(&lambda).map(|(gi, l)| gi[j] * l).sum();
        for (p, a) in psi.iter_mut().zip(a) {
            *p += mu * a;
        }
    }
}

/// Real gain profiles over the reach, applied along each tap's own phase.
///
/// ```text
/// δ_k = u_k Σ_j μ_j b_j(k),  u_k = ψ_k / |ψ_k|
/// ```
pub(super) struct Profiles(Vec<Vec<f64>>);

impl Profiles {
    /// b_j(k) = sin²(πx/2) xʲ,  x = clamp((k + 1 − K)/L + 1, 0, 1)
    pub fn edge(len: usize, width: f64, order: usize) -> Self {
        let axis = len as f64;
        let x: Vec<f64> = (0..len)
            .map(|k| ((k as f64 + 1.0 - axis) / width + 1.0).clamp(0.0, 1.0))
            .collect();
        Profiles(
            (0..order as i32)
                .map(|j| {
                    x.iter()
                        .map(|x| (FRAC_PI_2 * x).sin().powi(2) * x.powi(j))
                        .collect()
                })
                .collect(),
        )
    }
}

/// A real linear functional of folded Hermitian taps.
///
/// ```text
/// L(ψ) = Σ_k Re(c̄_k ψ_k)
/// ```
pub(super) struct Functional(Vec<Complex64>);

impl Functional {
    /// H⁽ᵖ⁾(ω),  c_k = 2 (ik)ᵖ e^{iωk}
    pub fn response(len: usize, w: f64, p: u32) -> Self {
        let mut c: Vec<Complex64> = (0..len)
            .map(|k| {
                let k = k as f64;
                Complex64::new(0.0, k).powu(p) * Complex64::from_polar(2.0, w * k)
            })
            .collect();
        // the center tap unfolds once
        c[0] *= 0.5;
        Functional(c)
    }

    pub fn eval(&self, psi: &[Complex64]) -> f64 {
        self.0
            .iter()
            .zip(psi)
            .map(|(c, p)| c.re * p.re + c.im * p.im)
            .sum()
    }
}

/// g x = r in place, g symmetric positive definite and row major.
fn cholesky(g: &mut [f64], r: &mut [f64]) {
    factor(g, r.len());
    solve(g, r);
}

/// g = L Lᵀ, L in the lower triangle.
fn factor(g: &mut [f64], n: usize) {
    for j in 0..n {
        let d = (g[j * n + j] - (0..j).map(|k| g[j * n + k].powi(2)).sum::<f64>()).sqrt();
        g[j * n + j] = d;
        for i in j + 1..n {
            g[i * n + j] =
                (g[i * n + j] - (0..j).map(|k| g[i * n + k] * g[j * n + k]).sum::<f64>()) / d;
        }
    }
}

/// L Lᵀ x = r in place.
fn solve(g: &[f64], r: &mut [f64]) {
    let n = r.len();
    // L y = r
    for i in 0..n {
        r[i] = (r[i] - (0..i).map(|k| g[i * n + k] * r[k]).sum::<f64>()) / g[i * n + i];
    }
    // Lᵀ x = y
    for i in (0..n).rev() {
        r[i] = (r[i] - (i + 1..n).map(|k| g[k * n + i] * r[k]).sum::<f64>()) / g[i * n + i];
    }
}
