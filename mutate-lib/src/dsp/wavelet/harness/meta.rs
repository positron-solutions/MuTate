// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Harness Testing
//!
//! Avoiding the complex aspects of our harnesses, we test them against naive implementations that
//! are prohibitively slow for regular testing but can isolate problems with the harnesses
//! themselves.

use core::f64::consts::{PI, TAU};

use super::super::spec::{Shape, WaveletSpec};
use super::*;

/// First nulls of suspicious skirts against a brute-force sign scan of Re H.  Re-run this if
/// there is ever any doubt that our null finder is missing something that dense DTFT scan would
/// find.
#[ignore]
#[test]
fn first_null_matches_dense_scan() {
    const QUANTUM: usize = 4;
    const TAIL_DB: f64 = -40.0;
    /// Naive samples per bin 2π/N.
    const PER_BIN: usize = 1024 * 8;

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
        "  {:>5} {:>4} {:>7} {:>5} {:>5} {:>11} {:>11} {:>10} {:>10} {:>10}",
        "Q", "𝛄", "𝛒", "side", "taps", "scanner", "naive", "Δ", "|Im H|", "radius"
    );

    let mut failures = Vec::new();
    for (q, gamma, rho, dir) in CASES {
        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(q, gamma))
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();
        let bin = wav.at_rho(rho);
        let psi = unfold(&bin.taps(), 0);
        let r = characterize(&psi, bin.velocity());
        let lobe = r.edges.1 - r.edges.0;
        let side = if dir < 0.0 { "lo" } else { "hi" };
        let tag = format!("Q {q} γ {gamma} ρ {rho} {side}");

        let resp = |w: f64| dtft(&psi, w).re;
        // 16ε Σ|h|
        let null = 16.0 * f64::EPSILON * l1(&psi);
        let density = 16.0 * psi.len() as f64 / TAU;

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

        let fast = first_null(&psi, from, stop, null, 1.0 / density, density);

        let (Some(fast), Some(naive)) = (fast, naive) else {
            println!(
                "  {q:>5.1} {gamma:>4.1} {rho:>7.4} {side:>5} {:>5} missing",
                psi.len()
            );
            failures.push(format!(
                "{tag} scanner {fast:?} naive {naive:?}",
                fast = fast.map(|p| p.w)
            ));
            continue;
        };

        let off = |w: f64| (w - r.peak_w) / lobe;
        let gap = (fast.w - naive).abs() / step;
        let im = dtft(&psi, naive).im.abs();

        let re_at_fast = resp(fast.w);
        println!(
            "  {q:>5.1} {gamma:>4.1} {rho:>7.4} {side:>5} {:>5} {:>+11.6} {:>+11.6} {gap:>10.3e} \
                    {im:>10.3e} {null:>10.3e} {re_at_fast:>10.3e}",
            psi.len(),
            off(fast.w),
            off(naive),
        );

        if gap >= 1.0 {
            failures.push(format!(
                "{tag} scanner {:+.4} lobes, naive {:+.4} lobes",
                off(fast.w),
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
