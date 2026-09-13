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
//! shape of the wavelet we are refining is maximally well preserved every time we shove it through
//! the grater.
//!
//! ## Restriction Math
//!
//! Each tap integrates the Morse **magnitude** uniformly over its time cell, then restores the
//! exact center-frequency phase, strictly preserving both symmetry and the center frequency omega.
//!
//! `aₖ = (1 / Δt) ∫_{Cₖ} |ψ(t)| dt`
//!
//! `hₖ = aₖ · e⁻ⁱωᶜᵗᵏ`
//!
//! The restriction therefore separates envelope and phase:
//!
//! - `aₖ` represents the Morse envelope contained by the cell.
//! - `e⁻ⁱωᶜᵗᵏ` fixes every tap to the same exact center-frequency rotation.
//! - No envelope weighting can shift a tap's phase or local angular frequency.
//! - Symmetric cells give `a₋ₖ = aₖ`.
//!
//! For a center-frequency input `x(t) = A · eⁱ(ωᶜt + θ)`, the response is
//! phase-covariant:
//!
//! `y(θ) = eⁱθ · y(0)`
//!
//! and therefore its magnitude is independent of input phase:
//!
//! `|y(θ)| = |y(0)|`.
//!
//! The four cardinal input phases consequently remain equivalent in magnitude:
//!
//! `+A`, `+iA`, `−A`, `−iA`.
//!
//! Relative to the center frequency, an input detuning `Δω` sees only the restricted envelope:
//!
//! `H(ωᶜ + Δω) = Σₖ aₖ · e⁻ⁱΔωtₖ`
//!
//! With symmetric real envelope taps, the response is centered and even:
//!
//! `H(ωᶜ + Δω) = H(ωᶜ − Δω)`
//!
//! and the center frequency is stationary:
//!
//! `d|H| / dω | ω=ωᶜ = 0`
//!
//! At `Δω = 0`, all taps contribute coherently. Away from the center, progressive phase rotation
//! produces cancellation according to the envelope's finite length and shape.
//!
//! For transient inputs, the response remains dependent on temporal overlap between the input
//! envelope and the Morse envelope. Only the carrier phase is invariant.
//!
//! The restriction therefore has a single independent design quantity:
//!
//! `aₖ`
//!
//! The tap phase is fixed by the exact center frequency, while the tap magnitude is fixed by
//! uniform integration of the Morse envelope.
//!
//! ## Reference Methods
//!
//! Alternative restrictions are supplied in order maintain a view of the relative effectiveness.

use std::f64::consts::TAU;

use libm::tgamma;
use num_complex::{Complex32, Complex64};

use super::{generate::hermite, spec::Shape, Grid};

/// How the motherlet lands on taps.
#[derive(Clone, Copy, Default)]
pub enum Restriction {
    /// Linear interpolation of the grid at each tap center.  Worst approach.  Leads to unacceptable
    /// aliasing at high omega.  Useful for demonstration...of mediocrity.
    Nearest,
    /// Complex cell average.  A slightly better technique, but not expressly aware of the polar
    /// spiral we are approximating with taps.
    Quadrature,
    /// Cell average of |ψ| carried by the exact center-frequency rotation.
    #[default]
    Magnitude,
}

impl Restriction {
    /// Writes `out.len()` folded weights in cell-average units, `out[0]` real.
    pub(super) fn psi_into(self, grid: Grid, rho: f64, out: &mut [Complex64]) {
        let inv = rho.recip();

        match self {
            Restriction::Nearest => {
                out[0] = Complex64::new(grid.linear(0.0).re, 0.0);
                for (j, o) in out.iter_mut().enumerate().skip(1) {
                    *o = grid.linear(j as f64 * rho);
                }
            }
            Restriction::Quadrature => {
                // the center cell is symmetric about u = 0, so the odd parts cancel
                out[0] = Complex64::new(2.0 * inv * grid.mass(0.0, 0.5 * rho).re, 0.0);
                for (j, o) in out.iter_mut().enumerate().skip(1) {
                    *o = inv * grid.mass((j as f64 - 0.5) * rho, (j as f64 + 0.5) * rho);
                }
            }
            Restriction::Magnitude => {
                // a_k = (1/ρ) ∫_{C_k} |ψ|,  h_k = a_k e^{2πi kρ}
                out[0] = Complex64::new(2.0 * inv * magnitude(grid, 0.0, 0.5 * rho), 0.0);
                for (j, o) in out.iter_mut().enumerate().skip(1) {
                    let a = inv * magnitude(grid, (j as f64 - 0.5) * rho, (j as f64 + 0.5) * rho);
                    let (s, c) = (TAU * j as f64 * rho).sin_cos();
                    *o = a * Complex64::new(c, s);
                }
            }
        }
    }
}

/// ∫_a^b |ψ| by composite Simpson
fn magnitude(grid: Grid, a: f64, b: f64) -> f64 {
    const SUB: usize = 8;

    let h = (b - a) / SUB as f64;
    let f = |i: usize| grid.at(a + i as f64 * h).norm();

    let odd: f64 = (1..SUB).step_by(2).map(f).sum();
    let even: f64 = (2..SUB).step_by(2).map(f).sum();

    h / 3.0 * (f(0) + f(SUB) + 4.0 * odd + 2.0 * even)
}

/// Rotated derivative from the truncated edges.
///
///     d_k = −(i/2πρ)·(ψ_T(e_{k+½}) − ψ_T(e_{k−½}))
// XXX this method has some terrible flaw in the endpoints
pub fn derivative_into(grid: Grid, rho: f64, out: &mut [Complex64]) {
    let k = out.len();
    let inv = rho.recip();

    let mut lo = grid.edge(0, k, rho);
    out[0] = Complex64::new(2.0 * inv / TAU * lo.im, 0.0);

    for (j, o) in out.iter_mut().enumerate().skip(1) {
        let hi = grid.edge(j, k, rho);
        *o = -Complex64::i() * inv / TAU * (hi - lo);
        lo = hi;
    }
}

#[cfg(test)]
mod test {
    use super::super::harness::{dtft, tone_response, unfold};
    use super::super::{Bin, BinSpec, Shape, Wavelet, WaveletSpec};
    use super::*;

    const Q: f64 = 3.5;
    const RHO: f64 = 0.116;
    const TAIL_DB: f64 = -60.0;
    const METHODS: [Restriction; 3] = [
        Restriction::Nearest,
        Restriction::Quadrature,
        Restriction::Magnitude,
    ];

    /// Folded ψ taps under one restriction, `(Re ψ, Im ψ)` from the emitted table.
    fn psi(restriction: Restriction) -> (Vec<Complex64>, f64) {
        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(Q, 3.0))
            .max_truncation(TAIL_DB)
            .with_restriction(restriction)
            .bake();
        let bin = wav.at_rho(RHO);
        let taps = bin
            .taps()
            .iter()
            .map(|t| Complex64::new(t[0] as f64, t[1] as f64))
            .collect();

        (taps, bin.velocity())
    }

    /// Per tap phase advance against the carrier.  arg(h_{k+1} · conj h_k) = ω₀
    #[test]
    fn restriction_holds_omega() {
        const TOL: f64 = 1e-5;

        let cols: Vec<(Vec<Complex64>, f64)> = METHODS.iter().map(|&m| psi(m)).collect();
        let (psi, w0) = &cols[2];
        let k = psi.len();

        println!(
            "\n=== RESTRICTION OMEGA (Q = {Q}, rho {RHO}, tail {TAIL_DB:.0} dB) w0 {w0:.9} ==="
        );
        println!(
            "  {:>4} {:>14} {:>14} {:>14}",
            "k", "nearest", "quad", "mag"
        );

        for j in 0..k - 1 {
            let w = |c: &(Vec<Complex64>, f64)| (c.0[j + 1] * c.0[j].conj()).arg();
            println!(
                "  {j:>4} {:>14.9} {:>14.9} {:>14.9}",
                w(&cols[0]),
                w(&cols[1]),
                w(&cols[2])
            );

            let dev = (w(&cols[2]) - w0).abs();
            assert!(dev < TOL, "tap {j} omega off by {dev:.3e}");
        }
    }

    /// Tap magnitudes, and agreement on the sign of each component.
    #[test]
    fn restriction_holds_sign() {
        /// relative error allowance between all three methods.
        const TOL: f64 = 1e-2;
        /// Below this the tap is truncation ripple and its sign is noise.
        const FLOOR: f64 = 1e-5;

        let cols: Vec<(Vec<Complex64>, f64)> = METHODS.iter().map(|&m| psi(m)).collect();
        let psi = &cols[2].0;
        let peak = psi[0].norm();

        println!("\n=== RESTRICTION MAGNITUDE (Q = {Q}, rho {RHO}, tail {TAIL_DB:.0} dB) ===");
        println!(
            "  {:>4} {:>14} {:>14} {:>14}",
            "k", "nearest", "quad", "mag"
        );

        for j in 0..psi.len() {
            let h = |c: &(Vec<Complex64>, f64)| c.0[j];
            let a = |c: &(Vec<Complex64>, f64)| c.0[j].norm();
            println!(
                "  {j:>4} {:>14.9} {:>14.9} {:>14.9}",
                h(&cols[0]).norm(),
                h(&cols[1]).norm(),
                h(&cols[2]).norm()
            );

            for c in &cols[..2] {
                let dev = (a(c) - a(&cols[2])).abs() / peak;
                assert!(dev < TOL, "tap {j} envelope off by {dev:.3e} of peak");
            }

            if j > 0 {
                for (i, c) in cols.iter().enumerate() {
                    let next = c.0[j - 1].norm();
                    assert!(
                        next >= a(c),
                        "method {i} tap {j} rises {:.9} -> {next:.9}",
                        a(c)
                    );
                }
            }
        }
    }

    /// Whether each restriction answers the same at every input phase.  A tone is driven through
    /// the full 2π of carrier phase and each lane is demodulated, so the reported number is the
    /// worst relative departure from that lane's phase mean.  Zero is phase blind.
    // Relatively slow.  Did detect, modestly, that the Restriciton::Nearest has the least
    // phase-stable response, but its still pretty stable (FIRs amirite?)
    #[ignore]
    #[test]
    fn restriction_holds_phase() {
        const RESOLUTION: f64 = 0.05;
        const STEP: f64 = 100.0;
        const SPAN: isize = 3;

        /// Dense enough to resolve a sawtooth in the carrier quantization of `Nearest`.
        const RHOS: usize = 64;
        /// Spanning an octave from `RHO`, where the tap count roughly halves.
        const RHO_HI: f64 = 2.0 * RHO;

        /// Nearest is coarse, so this only catches a method that has stopped working.
        const SWING_TOL: f64 = 1e-2;

        /// Emitted table and carrier at `rho` under one restriction.
        fn table(restriction: Restriction, rho: f64) -> (Vec<[f32; 4]>, f64) {
            let wav = WaveletSpec::default()
                .with_shape(Shape::from_q(Q, 3.0))
                .max_truncation(TAIL_DB)
                .with_restriction(restriction)
                .bake();
            let bin = wav.at_rho(rho);
            (bin.taps().to_vec(), bin.velocity())
        }

        println!("\n=== RESTRICTION PHASE (Q = {Q}, tail {TAIL_DB:.0} dB) ===");
        println!(
            "  {:>8} {:>6} {:>9} {:>11} {:>11} {:>11}",
            "rho", "taps", "method", "psi", "d", "t"
        );

        for i in 0..RHOS {
            // ρ · (ρ_hi / ρ)^(i / n), geometric so tap count steps evenly
            let rho = RHO * (RHO_HI / RHO).powf(i as f64 / (RHOS - 1) as f64);

            for (name, &m) in ["nearest", "quad", "mag"].iter().zip(&METHODS) {
                let c = table(m, rho);

                // worst over detuning, which carries no trend of its own
                let worst = (-SPAN..=SPAN).fold([0.0f64; 3], |acc, k| {
                    let s = tone_response(&c.0, c.1, k as f64 * STEP, RESOLUTION);
                    core::array::from_fn(|l| acc[l].max(s[l]))
                });

                println!(
                    "  {rho:>8.5} {:>6} {name:>9} {:>11.2e} {:>11.2e} {:>11.2e}",
                    c.0.len(),
                    worst[0],
                    worst[1],
                    worst[2]
                );

                for (lane, s) in ["psi", "d", "t"].iter().zip(worst) {
                    assert!(
                        s < SWING_TOL,
                        "{name} rho {rho:.5} lane {lane} swing {s:.3e}"
                    );
                }
            }
        }
    }
}
