// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Print Tests
//!
//! These tests are primarily diagnostic and do not have much in the way of useful assertions.

use core::f64::consts::{LN_2, PI, TAU};

use super::super::harness::*;
use super::super::inspect::*;
use super::super::*;

const RATE: f64 = 48_000.0;

#[ignore]
#[test]
fn print_waveform() {
    const TAIL_DB: f64 = -20.0;
    let wav = WaveletSpec::default()
        .with_shape(Shape {
            beta: 8.5,
            gamma: 3.0,
        })
        .max_truncation(TAIL_DB)
        .bake();

    for (fc, fs) in [(1000.0f64, 8000.0), (300.0, 3000.0), (12_000.0, RATE)] {
        let rho = fc / fs;
        let bin = wav.at_rho(rho);
        let wts = bin.weights();
        let (psi, d) = (wts.psi(), wts.d());
        let w0 = bin.velocity();

        print_wave(
            &format!(
                "BIN fc {fc:.0} fs {fs:.0} w0 {w0:.6} rho {rho:.6} taps {}",
                psi.len_unfolded()
            ),
            psi,
            30,
        );

        let g = psi.dtft(w0).abs();
        let gd = d.dtft(w0).abs();
        let dc = psi.dtft(0.0);
        let neg = psi.dtft(-w0).abs();

        // max over m of | |Ψ(m)| − 1 | against cos(ω₀k)
        let phase_err = (0..8)
            .map(|m| (wts.project(|k| (w0 * k as f64).cos(), m)[0].norm() - 1.0).abs())
            .fold(0.0f64, f64::max);

        println!(
            "  peak {g:.9}  d/psi {:.9}  dc {dc:.3e}  neg {:.2} dB  phase err {phase_err:.3e}",
            gd / g,
            20.0 * (neg / g).log10()
        );
    }
}

/// Rough magnitude response, centered on the measured peak.  The sweep names ρ directly, so
/// a row is a filter and not a sample rate.
#[ignore]
#[test]
fn print_response() {
    const Q: f64 = 16.5;
    const QUANTUM: usize = 4;
    const TAIL_DB: f64 = -20.0;

    const ROWS: usize = 200;
    const COLS: usize = 200;
    const ANTI_ALIAS: usize = 8;
    const FLOOR_DB: f64 = -120.0;
    const LOBES: f64 = 96.0;

    // Periods per tap, sweeping the downsample ladder from 20Hz at 3kHz to 15kHz at 48kHz.
    // Nyquist is 0.5.
    // const RHOS: [f64; 8] = [0.00667, 0.0116, 0.02, 0.035, 0.060, 0.104, 0.180, 0.312];

    const RHOS: [f64; 1] = [0.312];

    let wav = WaveletSpec::default()
        .with_shape(Shape::from_q(Q, 3.0))
        .max_load_quantum(QUANTUM)
        .max_truncation(TAIL_DB)
        .bake();

    for rho in RHOS {
        let bin = wav.at_rho(rho);
        let wts = bin.weights();
        let psi = wts.psi();
        let w0 = bin.velocity();

        // 2π/N against the −3 dB width Q asks for
        let cell = TAU / psi.len_unfolded() as f64;
        println!(
            "rho {rho:.4} taps {} cell {cell:.6} requested width {:.6} ({:.2} cells)",
            psi.len_unfolded(),
            w0 / Q,
            w0 / (Q * cell),
        );

        let r = characterize(psi, w0);

        let lobe = r.edges.1 - r.edges.0;
        let span = LOBES * lobe;
        let lo = (r.peak_w - 0.5 * span).max(-PI);
        let hi = (r.peak_w + 0.5 * span).min(PI);
        let step = (hi - lo) / ROWS as f64;

        println!(
            "\n=== RESPONSE rho {rho:.4} taps {} w0 {w0:.6} \
                 lobe {lobe:.6} ({:.5} w/fs) ===",
            psi.len_unfolded(),
            lobe / TAU
        );
        println!(
            "  columns span {FLOOR_DB:.0} dB (left) to peak (right); \
                 {:.1} of {LOBES:.0} lobes in window",
            (hi - lo) / lobe
        );

        for k in 0..=ROWS {
            let w = lo + step * k as f64;

            // (1/step) ∫ |H(ω)|² dω over [w − step/2, w + step/2]
            let power = (0..ANTI_ALIAS)
                .map(|j| {
                    let u = w + step * ((j as f64 + 0.5) / ANTI_ALIAS as f64 - 0.5);
                    psi.dtft(u).powi(2)
                })
                .sum::<f64>()
                / ANTI_ALIAS as f64;
            let level = 10.0 * (power / (r.gain * r.gain)).log10();

            let cells = ((1.0 - level / FLOOR_DB) * COLS as f64)
                .round()
                .clamp(0.0, COLS as f64) as usize;

            let near = |t: f64| (w - t).abs() <= 0.5 * step;
            let tag = if near(r.peak_w) {
                '<'
            } else if near(-r.peak_w) {
                '>'
            } else if near(r.edges.0) || near(r.edges.1) {
                '-'
            } else {
                ' '
            };

            println!(
                "{:>10.5} {:>9.5} {:>8.2} {tag}{}",
                w,
                w / TAU,
                level,
                "#".repeat(cells)
            );
        }
    }
}

#[ignore]
#[test]
fn print_chirp_transform() {
    // Make a little plot similar to the one show in this paper:
    // https://jmlilly.net/papers/lilly09-itsp-cp.pdf
    //
    // The paper uses a beta and gamma we can't quite draw (betas this low are not generally
    // supported yet), but our outputs are somewhat similar at relevant beta.

    const TAIL_DB: f64 = -40.0;
    const SPAN: isize = 400;
    const SD: f64 = 90.0;
    const ROWS: usize = 40;

    // ρ of the top and bottom rows, the ridge reaching ρ_top at the window edge
    const RHO_TOP: f64 = 0.10;
    const RHO_BOT: f64 = 0.002;

    let wav = WaveletSpec::default()
        .with_shape(Shape {
            beta: 3.0,
            gamma: 3.0,
        })
        .max_truncation(TAIL_DB)
        .bake();

    // ρ_top (ρ_bot/ρ_top)^(r/(ROWS−1))
    let bank: Vec<(f64, Weights)> = (0..ROWS)
        .map(|r| {
            let f = r as f64 / (ROWS - 1) as f64;
            let bin = wav.at_rho(RHO_TOP * (RHO_BOT / RHO_TOP).powf(f));
            (bin.velocity(), bin.weights())
        })
        .collect();

    // ω(t) = a·t
    let a = bank[0].0 / SPAN as f64;
    let x = chirp(a, SD, 0.0);

    print_transform("MORSE", &bank, &x, SPAN, 160, -40.0);
}
