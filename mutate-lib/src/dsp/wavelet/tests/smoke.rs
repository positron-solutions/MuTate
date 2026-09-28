//! # Smoke Tests
//!
//! Tests that fail when some measurable characteristic of the wavelet or the test model is
//! *hypothetically* bad.  Ideally, failures here already show up as failures in the primary
//! characteristics of interest, actual PSL and actual reassignment precision etc.  In practice,
//! wavelets may outperform what these foundation inspections look at, and these tests are
//! considered smoke tests that tell us when to look more deeply and where.
//!
//! > Fast food is bad for you.  Wouldn't you rather wait for healthy food?  Unidentified benevolent
//! > beings are working hard every day to bring us healthy food someday, and if we believe in them
//! > really hard, in ten more years, they will give healthy food to everyone!  If ten years from now,
//! > the healthy food still isn't here, those to blame will be the non-believers, so today we must
//! > socially coerce our information spaces to only be filled with believers, those who will wait,
//! > and we must attack and deny oxygen to the enemies of "free" software, anyone who actually
//! > writes code or drives forward with proactive action.
//! >
//! > - distinguished Redditor u/Wb9VBScxu2uZJHeq2E3W

#![warn(warnings, dead_code, unused_variables)]

use core::f64::consts::{LN_2, PI, TAU};

use super::super::{
    fmt_e,
    harness::*,
    inspect::*,
    refine,
    spec::{Shape, WaveletSpec},
    PEAK_GAIN,
};

const RATE: f64 = 48_000.0;

#[test]
fn gamma_sweep() {
    const QUANTUM: usize = 4;
    const STEP: f64 = 100.0;
    const SPAN: isize = 4;
    /// Worst reassignment bias over the scan, in cents.
    const BIAS_C: f64 = 1.2;
    const TAIL_DB: f64 = 60.0;

    println!("\n=== TAP PROFILE vs GAMMA (Q = 2.4) ===");
    // P² = beta gamma
    let p = 4.0;
    for gamma in [1.0f64, 2.0, 3.0, 6.0] {
        let wav = WaveletSpec::default()
            .shape(Shape {
                gamma,
                beta: p * p / gamma,
            })
            .max_load_quantum(QUANTUM)
            .max_truncation(TAIL_DB)
            .bake();
        let bin = wav.at_rho(1000.0 / 8000.0);
        let w0 = bin.velocity();
        let wts = bin.weights();
        let (psi, d) = (wts.psi(), wts.d());
        let n = psi.len_unfolded();

        // M₁ / M₀
        let delay = (psi.moment(1) / psi.moment(0)).re;

        // worst over detuning of |R| / ‖ψ‖₁ and of (1200/ln 2)·(R/H)/r
        let (floor, worst) = (-SPAN..=SPAN).fold((0.0f64, 0.0f64), |acc, k| {
            let ratio = (k as f64 * STEP / 1200.0).exp2();
            let (res, dr) = pairing_residual(psi, d, w0, w0 * ratio);
            (
                acc.0.max(res / psi.l1()),
                // (1200 / ln 2) · (R/H) / r
                acc.1.max((1200.0 / LN_2 * dr / ratio).abs()),
            )
        });

        println!(
            "\ngamma = {gamma:.1}  weights {}  taps {n}  delay = {delay:+.3e}  \
             floor = {}  bias = {worst:.3}c",
            psi.len_folded(),
            fmt_e(floor),
        );

        let mags: Vec<f64> = psi.mirrored().map(|h| h.norm()).collect();
        let max = mags.iter().fold(0.0f64, |a, &b| a.max(b));
        for (j, &v) in mags.iter().enumerate() {
            println!(
                "{:>4} {}",
                j as isize - (n / 2) as isize,
                "#".repeat((v / max * 40.0).round() as usize)
            );
        }

        assert!(worst < BIAS_C, "gamma {gamma} bias {worst:.3}c");
    }
}

/// Negative frequency energy of each bin, split into the beat against its own passband and
/// leakage through the mirror, beside the positive skirt it competes with.  If leak+ is larger
/// than leak-, the stopband is dominated by the skirt, telling us where to prioritize spectral
/// cleanup work.
#[test]
fn wavelet_is_analytic() {
    const BINS: usize = 6;
    const WORST: usize = 2;

    const QS: Axis = Axis::Levels(&[3.0, 4.0]);
    const GAMMAS: Axis = Axis::Levels(&[3.0, 4.0]);
    const DEEPEST_DB: f64 = -80.0;
    const FLOORS: Axis = Axis::Step(-40.0, DEEPEST_DB, 10.0);
    const RHO: Span = Span::Log(0.004, 0.45);

    const GOOD: f64 = 0.95;
    const EXCESS: f64 = 1e-2;

    let pow_db = |u: f64| 10.0 * u.log10();
    let amp_db = |u: f64| 20.0 * u.log10();
    let med = |v: &mut Vec<(f64, f64)>| weighted_quantile(v, 0.5);
    let mut ledger = Ledger::default();

    println!("\n=== ANALYTICITY ===");
    println!("  shares of white noise energy ∫|H|², dB, medians over served ρ");
    println!("  beat is image inside the passband guard, leak− and leak+ outside it");
    println!("  α is |H(−ω)|/|H(ω)| weighted by |H(ω)|² in band, peak re PEAK_GAIN\n");
    println!(
        "  {:>4} {:>5} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "γ", "Q", "floor", "beat", "leak−", "leak+", "α50", "α99", "peak", "beyond"
    );

    for (wg, gamma) in GAMMAS.levels() {
        for (wq, q) in QS.levels() {
            let wav = WaveletSpec::default()
                .shape(Shape::from_q(q, gamma))
                .max_noise_floor(DEEPEST_DB)
                .bake();
            let s = wav.shape();

            for (wf, floor) in FLOORS.levels() {
                let [mut beat, mut neg, mut pos, mut a50, mut a99, mut peak]: [Vec<(f64, f64)>; 6] =
                    Default::default();
                let mut beyond = 0usize;

                for (wr, [rho]) in survey([RHO], BINS) {
                    // ρ ≤ ρ_fold(floor)
                    if rho > s.fold_ceiling(floor) {
                        beyond += 1;
                        continue;
                    }

                    let bin = wav.at_rho(rho).with_noise_floor(floor);
                    let wts = bin.weights();
                    let psi = wts.psi();
                    let mut img = image(psi, bin.velocity());

                    beat.push((wr, img.beat));
                    neg.push((wr, img.leak_neg));
                    pos.push((wr, img.leak_pos));
                    a50.push((wr, weighted_quantile(&mut img.alpha, 0.5)));
                    a99.push((wr, weighted_quantile(&mut img.alpha, 0.99)));
                    peak.push((wr, img.peak.1 / PEAK_GAIN));

                    // 10^(floor/10)
                    let tol = 10f64.powf(floor / 10.0);
                    let at_w = img.peak.0 / bin.velocity();
                    ledger.record(
                        wg * wq * wf * wr,
                        (img.beat + img.leak_neg) / tol,
                        at!(q, gamma, floor, rho, at_w),
                    );
                }

                println!(
                    "  {gamma:>4.1} {q:>5.1} {floor:>6.0} {:>8.2} {:>8.2} {:>8.2} \
                         {:>8.2} {:>8.2} {:>8.2} {beyond:>7}",
                    pow_db(med(&mut beat)),
                    pow_db(med(&mut neg)),
                    pow_db(med(&mut pos)),
                    amp_db(med(&mut a50)),
                    amp_db(med(&mut a99)),
                    amp_db(med(&mut peak)),
                );
            }
            println!();
        }
    }

    ledger.print_worst(WORST);
    let [good, excess] = ledger.verdict([GOOD, EXCESS]);

    assert!(good > GOOD, "good mass {good:.4}");
    assert!(excess < EXCESS, "excess mass {excess:.2e}");
}

/// Whether the filter answers the same at every input phase.  Each row drives a steady tone
/// through the full 2π of carrier phase and demodulates each lane, so the reported number is
/// the worst relative departure from that lane's phase mean.  Zero is phase blind.
#[test]
fn response_is_phase_independent() {
    const Q: f64 = 8.5;
    const QUANTUM: usize = 4;
    const TAIL_DB: f64 = -60.0;

    const GATE_DB: f64 = -50.0;
    const RESOLUTION: f64 = 0.05;

    /// Well above anything the image alone produces in band.
    const SWING_TOL: f64 = 5e-2;

    const STEP: f64 = 25.0;
    const SPAN: isize = 6;

    let wav = WaveletSpec::default()
        .shape(Shape::from_q(Q, 3.0))
        .max_load_quantum(QUANTUM)
        .max_truncation(TAIL_DB)
        .bake();

    println!("\n=== RESPONSE PHASE DEPENDENCE ===");
    println!("  detune      |H|    img ψ         psi           d           t");

    for (fc, fs) in [
        (2_000.0f64, RATE),
        (200.0, 3000.0),
        (250.0, 3000.0),
        (12_000.0, RATE),
    ] {
        let bin = wav.at_rho(fc / fs);
        let wts = bin.weights();
        let psi = wts.psi();

        let (n, w0) = (psi.len_unfolded(), bin.velocity());
        println!("  fc {fc:.0} fs {fs:.0} taps {n} w0 {w0:.6}");

        for k in -SPAN..=SPAN {
            let cents = k as f64 * STEP;
            let wd = w0 * (cents / 1200.0).exp2();
            let h_db = db(psi.dtft(wd).abs()) - db(PEAK_GAIN);
            if h_db < GATE_DB {
                continue;
            }

            let [sp, sd, st] = tone_response(&wts, w0, cents, RESOLUTION);
            let img = psi.dtft(-wd).abs();

            println!(
                "  {cents:+6.0}c {h_db:>7.1} {:>8.1} {sp:>11.2e} {sd:>11.2e} {st:>11.2e}",
                db(img) - db(PEAK_GAIN),
            );

            for (lane, s) in ["psi", "d", "t"].iter().zip([sp, sd, st]) {
                assert!(
                    s < SWING_TOL,
                    "fc {fc} detune {cents} lane {lane} swing {s:.3e}"
                );
            }
        }
    }
}

/// Whether the reported pitch and quadrature move with the carrier phase.  Each row sweeps
/// the full 2π of input phase at `RESOLUTION` and reports the departure from the phase mean,
/// so a filter that answers the same for every phase reads zero across the board.  The swing
/// is the negative frequency image beating against the signal, so the budget scales with
/// image over signal rather than being flat.
#[test]
fn reassignment_is_phase_independent() {
    const Q: f64 = 8.5;
    const QUANTUM: usize = 4;
    const TAIL_DB: f64 = -60.0;

    const GATE_DB: f64 = -50.0;
    const RESOLUTION: f64 = 0.05;

    /// Swing the image alone implies, within this factor.  Second order in |α|, so tight.
    const SWING_C: f64 = 1.3;

    const STEP: f64 = 25.0;
    const SPAN: isize = 6;

    let wav = WaveletSpec::default()
        .shape(Shape::from_q(Q, 3.0))
        .max_load_quantum(QUANTUM)
        .max_truncation(TAIL_DB)
        .bake();

    println!("\n=== PHASE DEPENDENCE ===");
    println!("  detune      |H|    img ψ    img d     pred     swing     ratio");

    for (fc, fs) in [
        (2_000.0f64, RATE),
        (200.0, 3000.0),
        (250.0, 3000.0),
        (12_000.0, RATE),
    ] {
        let bin = wav.at_rho(fc / fs);
        let wts = bin.weights();
        let (psi, d) = (wts.psi(), wts.d());

        let (n, w0) = (psi.len_unfolded(), bin.velocity());
        println!("  fc {fc:.0} fs {fs:.0} taps {n} w0 {w0:.6}");

        for k in -SPAN..=SPAN {
            let cents = k as f64 * STEP;
            let wd = w0 * (cents / 1200.0).exp2();
            let h = psi.dtft(wd).abs();
            let h_db = 20.0 * (h / PEAK_GAIN).log10();
            if h_db < GATE_DB {
                continue;
            }

            let ((_, swing), (quad, _)) = tone_bias(&wts, w0, cents, RESOLUTION);

            // H_ψ(±ω), H_d(±ω)
            let (p, p_img) = (psi.dtft(wd), psi.dtft(-wd));
            let (q, q_img) = (d.dtft(wd), d.dtft(-wd));
            // α = H_d(−ω)/H_d(ω) − H_ψ(−ω)/H_ψ(ω)
            let alpha = q_img / q - p_img / p;
            let pred = 1200.0 / LN_2 * alpha.abs();

            println!(
                "  {cents:+6.0}c {h_db:>7.1} {:>8.1} {:>8.1} {pred:>8.3}c {swing:>8.3}c {:>9.3}",
                db(p_img.abs()) - db(PEAK_GAIN),
                db(q_img.abs()) - db(PEAK_GAIN),
                swing / pred,
            );

            assert!(
                swing < SWING_C * pred + 1e-3,
                "fc {fc} detune {cents} swing {swing:.4}c over {:.4}c",
                SWING_C * pred + 1e-3
            );
            assert!(
                quad.abs() < 1e-12,
                "fc {fc} detune {cents} quad mean {quad:.3e}"
            );
        }
    }
}

/// A real unit tone at the crest reads |W| = 1 even though |H| = 2: the analytic taps see only
/// the +ω half of the cosine, and the −ω half beats against it by ½|H(−ω)|.
#[test]
fn unit_tone_reads_unity() {
    const EPS: f64 = 1e-6;
    const BINS: usize = 16;
    const WORST: usize = 2;

    const QS: Axis = Axis::Levels(&[3.0, 4.25, 6.0]);
    const GAMMAS: Axis = Axis::Levels(&[3.0, 4.0, 6.0]);
    const RHO: Span = Span::Log(0.004, 0.45);
    const TAIL_DB: Span = Span::Lin(-20.0, -100.0);

    const GOOD: f64 = 0.99;
    const EXCESS: f64 = 5e-3;

    let mut probe = Vec::new();
    let mut all = Ledger::default();

    println!("\n=== UNIT TONE ===");
    println!("  good is the weighted share within tolerance, excess the capped overshoot mass\n");
    println!("  {:>6} {:>6} {:>8} {:>10}", "Q", "γ", "good", "excess");

    for (_, gamma) in GAMMAS.levels() {
        for (_, q) in QS.levels() {
            let wav = WaveletSpec::default()
                .shape(Shape::from_q(q, gamma))
                .max_truncation(TAIL_DB.end())
                .bake();

            let mut shape = Ledger::default();
            for (w, [rho, tail_db]) in survey([RHO, TAIL_DB], BINS) {
                let bin = wav.at_rho(rho).with_truncation(tail_db);
                let wts = bin.weights();
                let psi = wts.psi();

                // ω_peak
                let Some((wp, _)) =
                    Inspect::new(psi, &mut probe, OVERSAMPLE).peak(bin.velocity(), 0.0, PI)
                else {
                    shape.record(w, f64::NAN, at!(q, gamma, rho, tail_db));
                    continue;
                };

                // ½|H(−ω_peak)| + ε
                let tol = 0.5 * psi.dtft(-wp).abs() + EPS;

                // max over m of | |Ψ(m)| − 1 |
                let err = (0..8)
                    .map(|m| (wts.project(|k| (wp * k as f64).cos(), m)[0].norm() - 1.0).abs())
                    .fold(0.0f64, f64::max);

                shape.record(w, err / tol, at!(q, gamma, rho, tail_db));
            }

            println!(
                "  {q:>6.2} {gamma:>6.2} {:>8.4} {:>10.2e}",
                shape.good(),
                shape.excess()
            );
            all.absorb(shape);
        }
    }

    all.print_worst(WORST);
    let [good, excess] = all.verdict([GOOD, EXCESS]);

    assert!(good > GOOD, "good mass {good:.4}");
    assert!(excess < EXCESS, "excess mass {excess:.2e}");
}

/// Peak-normalized constant-Q puts noise gain proportional to ρ: the ψ sum is pinned at
/// PEAK_GAIN, so the envelope's amplitude scales with ρ and its energy with ρ². Dividing that
/// back by the ρ⁻¹ taps per period leaves one power of ρ. White noise therefore floors at a
/// fixed level per bin once ρ is divided out.
#[test]
fn noise_gain_tracks_center() {
    const Q: f64 = 3.0;
    const QUANTUM: usize = 4;
    const TAIL_DB: f64 = -100.0;

    // Tap count is an integer, so envelope truncation loses O(1/N) of the energy, worst at
    // the top of the range. Anchored to split the sweep rather than to any one bin.
    const NOISE_GAIN: f64 = 1.4105;
    const TOL: f64 = 2e-3;

    let w = WaveletSpec::default()
        .shape(Shape::from_q(Q, 3.0))
        .max_load_quantum(QUANTUM)
        .max_truncation(TAIL_DB)
        .bake();

    println!("\n=== NOISE GAIN (Q = {Q}, fs = {RATE}, quantum {QUANTUM}) ===");

    for fc in [500.0f64, 1000.0, 2000.0, 4000.0, 8000.0] {
        let bin = w.bin(fc, RATE);
        let wts = bin.weights();
        let psi = wts.psi();

        let e = psi.energy();
        let ratio = e / bin.rho();

        println!(
            "  fc {:>6.0}  taps {:>5}  rho {:.6}  energy {:.6}  e/rho {:.6}  dev {:+.2e}",
            fc,
            bin.len_unfolded(),
            bin.rho(),
            e,
            ratio,
            ratio / NOISE_GAIN - 1.0
        );

        assert!(
            (ratio / NOISE_GAIN - 1.0).abs() < TOL,
            "fc {fc} noise gain {ratio:.6}"
        );
    }
}

/// Smoke test.  DC-free and correct peak gain, measured with the linear DTFT so a broken fold
/// convention can't agree with itself.  First thing to look at if the bake goes sideways.
#[test]
fn taps_are_conditioned() {
    // Use a rough tail dB so we can verify conditioning under challenging conditions.
    const TAIL_DB: f64 = -80.0;
    const MOMENT_TOL: f64 = 1e-6;

    // NOTE the refinement is explicitly set so that this test doesn't regress whenever the
    // default moment conditioning is updated.
    let wav = WaveletSpec::default()
        .max_truncation(TAIL_DB)
        .refinement(Some(refine::Refinement::Reach {
            moments: &[0, 1, 2, 3],
            jet: &[0, 1, 2, 3],
            tangent: true,
            spare: 16,
            turns: 3.0,
        }))
        .bake();

    for (fc, fs) in [(1000.0f64, 8000.0f64), (250.0, 3000.0), (12_000.0, RATE)] {
        let bin = wav.at_rho(fc / fs);
        let wts = bin.weights();
        let (psi, d) = (wts.psi(), wts.d());

        let w0 = bin.velocity();

        let g = psi.dtft(w0).abs();
        assert!((g - PEAK_GAIN).abs() < 1e-2, "fc {fc} peak gain {g:.6}");

        // Analytic taps: the mirror image is stopband, not signal.
        let neg = psi.dtft(-w0).abs();
        assert!(
            neg < 1e-3 * g,
            "fc {fc} negative-freq leak {:.2} dB",
            20.0 * (neg / g).log10()
        );

        let dc = psi.dtft(0.0);
        assert!(dc.abs() < 1e-2 * g, "fc {fc} dc {dc:.3e}");

        // d/ψ reads ω/ω₀, unity at the carrier.
        let gd = d.dtft(w0).abs();
        assert!((gd / g - 1.0).abs() < 1e-2, "fc {fc} d/psi {:.6}", gd / g);

        // Σ ν^p · (Re ψ if p even, Im ψ if p odd)
        let mom = |p: i32| match p % 2 == 0 {
            true => psi.moment(p).re,
            false => psi.moment(p).im,
        };

        // |M_p| / A_p, the share of each moment's scale left uncancelled.  H^(p)(0) for p = 2, 3
        // are the ones the solve nulls that nothing else measures.
        for p in 0..=3 {
            let residue = mom(p).abs() / psi.abs_moment(p);
            println!("  fc {fc:>6.0}  M{p} residue {residue:.3e}");
            assert!(
                residue < MOMENT_TOL,
                "fc {fc} moment {p} residue {residue:.3e}"
            );
        }
    }
}
