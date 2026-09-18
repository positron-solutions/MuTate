// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Harness Testing
//!
//! Avoiding the complex aspects of our harnesses, we test them against naive implementations that
//! are prohibitively slow for regular testing but can isolate problems with the harnesses
//! themselves.

use core::f64::consts::{PI, TAU};

use super::super::{
    inspect::*,
    spec::{Shape, WaveletSpec},
    PEAK_GAIN,
};
use super::*;

const RATE: f64 = 48_000.0;

/// First nulls of suspicious skirts against a brute-force sign scan of Re H.  Re-run this if
/// there is ever any doubt that our null finder is missing something that dense DTFT scan would
/// find.
#[ignore]
#[test]
fn first_null_matches_dense_scan() {
    const QUANTUM: usize = 4;
    const TAIL_DB: f64 = -40.0;
    /// Naive samples per bin 2π/N.
    const PER_BIN: usize = 32;

    /// (Q, γ, ρ, dir), lower skirt at −1 and upper at +1.
    const CASES: [(f64, f64, f64, f64); 8] = [
        (12.5, 4.0, 0.116, -1.0),
        (12.5, 4.0, 0.116, 1.0),
        (3.5, 3.0, 0.384, 1.0),
        (3.5, 3.0, 0.384, -1.0),
        (3.5, 4.0, 0.384, 1.0),
        (3.5, 3.0, 0.189, -1.0),
        (3.5, 3.0, 0.223, 1.0),
        (5.0, 4.0, 0.189, -1.0),
    ];

    println!(
        "\n=== FIRST NULL vs DENSE (quantum {QUANTUM}, tail {TAIL_DB:.0} dB) ===\n\
         offsets in −3 dB widths from the peak, Δ in naive steps\n"
    );
    println!(
        "  {:>5} {:>4} {:>7} {:>5} {:>5} {:>11} {:>11} {:>10} {:>10} {:>10} {:>10}",
        "Q", "𝛄", "𝛒", "side", "taps", "scanner", "naive", "Δ", "|Im H|", "radius", "Re H"
    );

    let mut failures = Vec::new();
    for (q, gamma, rho, dir) in CASES {
        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(q, gamma))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();
        let bin = wav.at_rho(rho);
        let taps = bin.taps();
        let psi = unfold(&taps, 0);
        let psi64 = lane(&taps, 0);
        let fold = Fold(&psi64);

        let r = characterize(fold, bin.velocity());
        let lobe = r.edges.1 - r.edges.0;
        let side = if dir < 0.0 { "lo" } else { "hi" };
        let tag = format!("Q {q} γ {gamma} ρ {rho} {side}");

        let resp = |w: f64| dtft(&psi, w).re;
        // 16ε Σ|h|
        let null = 16.0 * f64::EPSILON * l1(&psi);

        let from = if dir < 0.0 { r.edges.0 } else { r.edges.1 };
        let stop = from + dir * PI;

        // Naive
        let step = TAU / (PER_BIN * psi.len()) as f64;
        let mut prev = (from, resp(from));
        let naive = (1..)
            .map(|j| from + dir * step * j as f64)
            .take_while(|&w| dir * (stop - w) >= 0.0)
            .find_map(|w| {
                let q = (w, resp(w));
                let hit = (prev.1 < 0.0) != (q.1 < 0.0);
                let pair = (prev.0, q.0);
                prev = q;
                hit.then_some(pair)
            })
            .map(|(a, b)| bisect(&resp, a, b));

        let mut buf = Vec::new();
        let fast = Inspect::new(Fold(&psi64), &mut buf, OVERSAMPLE)
            .null(from, stop)
            .map(|(w, _)| w);

        let (Some(fast), Some(naive)) = (fast, naive) else {
            println!(
                "  {q:>5.1} {gamma:>4.1} {rho:>7.4} {side:>5} {:>5} missing",
                psi.len()
            );
            failures.push(format!("{tag} scanner {fast:?} naive {naive:?}"));
            continue;
        };

        let off = |w: f64| (w - r.peak_w) / lobe;
        let gap = (fast - naive).abs() / step;
        let im = dtft(&psi, naive).im.abs();
        let re_at_fast = resp(fast);

        println!(
            "  {q:>5.1} {gamma:>4.1} {rho:>7.4} {side:>5} {:>5} {:>+11.6} {:>+11.6} {gap:>10.3e} \
             {im:>10.3e} {null:>10.3e} {re_at_fast:>10.3e}",
            psi.len(),
            off(fast),
            off(naive),
        );

        if gap >= 1.0 {
            failures.push(format!(
                "{tag} scanner {:+.4} lobes, naive {:+.4} lobes",
                off(fast),
                off(naive)
            ));
        }
        if im > null {
            failures.push(format!(
                "{tag} Im H {im:.3e} outside null radius {null:.3e}"
            ));
        }
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The frequency domain model against the synthesized tone.  Validating this lets sweeps use
/// `pairing_residual` alone, which needs no tone and no phase loop.
#[ignore]
#[test]
fn reassignment_model_agrees() {
    const Q: f64 = 8.5;
    const TAIL_DB: f64 = -60.0;
    const GATE_DB: f64 = -40.0;
    const RESOLUTION: f64 = 0.05;
    const STEP: f64 = 100.0;
    const SPAN: isize = 12;
    /// The model is linear and the tone is real, so the image sets the agreement floor.
    const TOL_C: f64 = 5e-2;

    let wav = WaveletSpec::default()
        .with_shape(Shape::from_q(Q, 3.0))
        .max_truncation(TAIL_DB)
        .bake();

    for (fc, sr) in [(2_000.0f64, RATE), (12_000.0, RATE)] {
        let bin = wav.at_rho(fc / sr);
        let taps = bin.taps();
        let (psi, d) = (unfold(&taps, 0), unfold(&taps, 2));
        let w0 = bin.velocity();

        for k in -SPAN..=SPAN {
            let cents = k as f64 * STEP;
            let ratio = (cents / 1200.0).exp2();
            let h = dtft(&psi, w0 * ratio).norm();
            if 20.0 * (h / PEAK_GAIN).log10() < GATE_DB {
                continue;
            }

            let ((bias, _), _) = tone_bias(&taps, w0, cents, RESOLUTION);
            let (_, dr) = pairing_residual(&psi, &d, w0, w0 * ratio);
            let pred = 1200.0 / LN_2 * dr / ratio;

            assert!(
                (bias - pred).abs() < TOL_C,
                "fc {fc} detune {cents} model off {:.4}c",
                (bias - pred).abs()
            );
        }
    }
}
