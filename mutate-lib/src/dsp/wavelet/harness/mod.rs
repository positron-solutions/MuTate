// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Testing Harnesses
//!
//! Testing harnesses and meta tests for the harnesses.  We have a lot of test harness code.  Some
//! of it is complex and needs tests against less complex naive implementations that are
//! unacceptable for general use but enable fast iteration against a naive baseline that is easier
//! to inspect and trust.

#[cfg(feature = "validate")]
mod meta;

use core::f64::consts::{LN_2, PI, TAU};

use num_complex::Complex64;

use super::inspect::*;
use super::{Fold, Weights};

/// Max of `f` over [lo, hi] at comb density for `taps` taps.
fn sweep(lo: f64, hi: f64, taps: usize, f: impl Fn(f64) -> f64) -> f64 {
    let n = ((hi - lo) * OVERSAMPLE * taps as f64 / TAU).ceil().max(1.0) as usize;
    (0..=n)
        .map(|k| f(lo + (hi - lo) * k as f64 / n as f64))
        .fold(0.0f64, f64::max)
}

/// max |H| on [−0.05 ω₀, 0.05 ω₀].
pub(super) fn dc_leak(psi: Fold<'_>, w0: f64) -> f64 {
    let e = 0.05 * w0;
    sweep(-e, e, psi.len_unfolded(), |w| psi.dtft(w).abs())
}

/// ∫ h² dω on [lo, hi], h = max(0, 1 − dB/floor_db) with dB relative to `gain`.
pub(super) fn level_moment(psi: Fold<'_>, gain: f64, (lo, hi): (f64, f64), floor_db: f64) -> f64 {
    let n = ((hi - lo) * OVERSAMPLE * psi.len_unfolded() as f64 / TAU)
        .ceil()
        .max(1.0) as usize;
    let dw = (hi - lo) / n as f64;
    (0..n)
        .map(|k| {
            let w = lo + dw * (k as f64 + 0.5);
            let level = db(psi.dtft(w).abs()) - db(gain);
            (1.0 - level / floor_db).max(0.0).powi(2)
        })
        .sum::<f64>()
        * dw
}

/// `|R|` and `R/Ψ` for `R = H_d(ω) − (ω/ω₀)·H_ψ(ω)`.  The ratio is what lands in `r̂`, the
/// magnitude is the floor.
pub(super) fn pairing_residual(psi: Fold<'_>, d: Fold<'_>, w0: f64, w: f64) -> (f64, f64) {
    let h = psi.dtft(w);
    let r = d.dtft(w) - h * (w / w0);
    (r.abs(), r / h)
}

/// Bias of `r̂` in cents and quadrature leak against a real tone detuned `cents` from `w0`,
/// each as its mean over the carrier phase and its worst departure from that mean.  Phases
/// step no coarser than `min_resolution` radians across the full circle.
pub(super) fn tone_bias(
    wts: &Weights,
    w0: f64,
    cents: f64,
    min_resolution: f64,
) -> ((f64, f64), (f64, f64)) {
    // ω₀ · 2^(c/1200)
    let wd = w0 * (cents / 1200.0).exp2();
    let n = (TAU / min_resolution).ceil().max(2.0) as usize;

    let r: Vec<Complex64> = (0..n)
        .map(|j| {
            let phase = TAU * j as f64 / n as f64;
            let tone = |k: isize| (wd * k as f64 + phase).cos();
            let [psi, dee, _] = wts.project(tone, 0);
            // D/Ψ
            dee / psi
        })
        .collect();

    let spread = |f: &dyn Fn(Complex64) -> f64| {
        let mean = r.iter().map(|&v| f(v)).sum::<f64>() / n as f64;
        let worst = r
            .iter()
            .map(|&v| (f(v) - mean).abs())
            .fold(0.0f64, f64::max);
        (mean, worst)
    };

    (spread(&|v| 1200.0 * v.re.log2() - cents), spread(&|v| v.im))
}

/// Departure of each lane from its phase mean, relative, after removing the carrier.
/// Zero where the filter answers identically at every input phase.  Lanes are Ψ, D, T.
pub(super) fn tone_response(wts: &Weights, w0: f64, cents: f64, min_resolution: f64) -> [f64; 3] {
    // ω₀ · 2^(c/1200)
    let wd = w0 * (cents / 1200.0).exp2();
    let n = (TAU / min_resolution).ceil().max(2.0) as usize;

    // Ψ(θ) e^{−iθ}, D(θ) e^{−iθ}, T(θ) e^{−iθ}
    let out: Vec<[Complex64; 3]> = (0..n)
        .map(|j| {
            let theta = TAU * j as f64 / n as f64;
            let tone = |k: isize| (wd * k as f64 + theta).cos();
            let carrier = Complex64::from_polar(1.0, -theta);
            wts.project(tone, 0).map(|v| v * carrier)
        })
        .collect();

    // |Ψ̄|, the scale every lane is read against
    let scale = (out.iter().map(|v| v[0]).sum::<Complex64>() / n as f64).norm();

    core::array::from_fn(|lane| {
        let mean = out.iter().map(|v| v[lane]).sum::<Complex64>() / n as f64;
        out.iter()
            .map(|v| (v[lane] - mean).norm() / scale)
            .fold(0.0f64, f64::max)
    })
}

/// Gaussian tone burst, carrier `w`, envelope sd in samples, centered at `p`.
pub(super) fn burst(w: f64, sd: f64, p: f64) -> impl Fn(isize) -> f64 {
    move |k| {
        let z = (k as f64 - p) / sd;
        (-0.5 * z * z).exp() * (w * (k as f64 - p)).cos()
    }
}

/// Signed bars for `re` and `im` overlaid on one axis, zero between cells `cols / 2 - 1` and
/// `cols / 2`. Caller guarantees `max >= |re|` and `max >= |im|`, which keeps both spans inside
/// the `cols` field.
fn bar(re: f64, im: f64, max: f64, cols: usize) -> String {
    let (cr, ci, cb) = ('●', '·', '•');
    let half = cols / 2;
    let span = |v: f64| {
        let col = ((v / max) * half as f64).round() as isize;
        let end = (half as isize + col) as usize;
        if col >= 0 {
            half..end
        } else {
            end..half
        }
    };
    let (r, i) = (span(re), span(im));
    let cells: String = (0..cols)
        .map(|c| match (r.contains(&c), i.contains(&c)) {
            (true, true) => cb,
            (true, false) => cr,
            (false, true) => ci,
            (false, false) => ' ',
        })
        .collect();
    cells.trim_end().to_string()
}

/// Edges either side of `peak_w` where the response first falls `level_db` below `peak`,
/// searched out to `span`, absent where no crossing lies inside it.
pub(super) fn shoulders(
    psi: Fold<'_>,
    peak: f64,
    peak_w: f64,
    level_db: f64,
    span: f64,
) -> (Option<f64>, Option<f64>) {
    let mut buf = Vec::new();
    let mut insp = Inspect::new(psi, &mut buf, OVERSAMPLE);
    let level = db(peak) + level_db;
    let mut edge = |stop: f64| insp.cross(level, peak_w, stop).map(|(w, _)| w);
    (
        edge((peak_w - span).max(0.0)),
        edge((peak_w + span).min(PI)),
    )
}

/// Extrema of |H| across one stopband, tallest first.
pub(super) struct Skirt {
    /// Largest local maximum, absent where the band holds none.
    pub peak: Option<Sample>,
    /// Median local maximum, the ripple level the peak stands on.
    pub median: f64,
    /// Local maxima found.
    pub lobes: usize,
}

impl Skirt {
    /// 20 log10 (peak / median), how far the tallest lobe clears the ripple.
    pub fn prominence_db(&self) -> Option<f64> {
        self.peak.map(|(_, h)| db(h) - db(self.median))
    }
}

/// Tallest lobes by comb height, pinned before the tallest is chosen.
const SKIRT_PIN: usize = 4;

/// |H| over the stopband between `from` and `stop`, scanned whole.  Empty where no dip bounds
/// the band.
fn skirt(psi: Fold<'_>, from: Option<f64>, stop: f64) -> Skirt {
    let Some(from) = from else {
        return Skirt {
            peak: None,
            median: f64::NAN,
            lobes: 0,
        };
    };

    let mut buf = Vec::new();
    let mut insp = Inspect::new(psi, &mut buf, OVERSAMPLE);

    let mut found = insp.crests(from, stop);
    found.sort_by(|a, b| b.at.1.total_cmp(&a.at.1));

    let peak = found
        .iter()
        .take(SKIRT_PIN)
        .map(|&c| insp.pin_crest(c))
        .max_by(|a, b| a.1.total_cmp(&b.1));

    let median = found.get(found.len() / 2).map_or(f64::NAN, |c| c.at.1);

    Skirt {
        peak,
        median,
        lobes: found.len(),
    }
}

/// Both stopbands outside the first nulls, lower then upper.
pub(super) fn skirts(psi: Fold<'_>, (lo, hi): (Option<f64>, Option<f64>)) -> (Skirt, Skirt) {
    (skirt(psi, lo, 0.0), skirt(psi, hi, PI))
}

/// Root of `resp` in a sign-changing bracket.
pub(super) fn bisect(resp: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let fa = resp(a) < 0.0;
    for _ in 0..64 {
        let m = 0.5 * (a + b);
        if m == a || m == b {
            break;
        }
        let fm = resp(m);
        if fm == 0.0 {
            return m;
        }
        if (fm < 0.0) == fa {
            a = m;
        } else {
            b = m;
        }
    }
    0.5 * (a + b)
}

pub(super) struct Response {
    pub peak_w: f64,
    /// |H(peak_w)|
    pub gain: f64,
    pub edges: (f64, f64),
    pub rel_width: f64,
    /// max |H| on [−π, 0]
    pub image: f64,
    pub floor: f64,
}

/// Peak, -3 dB relative width, image, and the positive-axis floor outside three half-power
/// widths.  `w0` seeds the climb and brackets the edges.
pub(super) fn characterize(psi: Fold<'_>, w0: f64) -> Response {
    let mut buf = Vec::new();
    let mut insp = Inspect::new(psi, &mut buf, super::inspect::OVERSAMPLE);

    let (peak_w, gain) = insp.peak(w0, 0.0, PI).expect("no crest in [0, π]");

    // |H(peak)| / √2
    let half_power = db(gain) - 10.0 * 2.0f64.log10();
    let (lo_stop, hi_stop) = ((peak_w - w0).max(0.0), (peak_w + w0).min(PI));
    let lo = insp.cross(half_power, peak_w, lo_stop);
    let hi = insp.cross(half_power, peak_w, hi_stop);
    let (lo, hi) = (
        lo.map_or(lo_stop, |(w, _)| w),
        hi.map_or(hi_stop, |(w, _)| w),
    );

    let sweep = 32;
    let omega = |k: usize| PI * k as f64 / sweep as f64;
    let guard = 3.0 * (hi - lo);
    let (mut image, mut floor) = (0.0f64, 0.0f64);
    for k in 0..=sweep {
        let w = omega(k);
        image = image.max(psi.dtft(-w).abs());
        if (w - peak_w).abs() > guard {
            floor = floor.max(psi.dtft(w).abs());
        }
    }

    Response {
        peak_w,
        gain,
        edges: (lo, hi),
        rel_width: (hi - lo) / peak_w,
        image,
        floor,
    }
}

/// Gaussian-enveloped chirp, sweep rate `a` rad/sample², envelope sd in samples, centered at
/// `p`.  ω(k) = a(k − p), passing through zero at `p`.
pub(super) fn chirp(a: f64, sd: f64, p: f64) -> impl Fn(isize) -> f64 {
    move |k| {
        let t = k as f64 - p;
        let z = t / sd;
        (-0.5 * z * z).exp() * (0.5 * a * t * t).cos()
    }
}

/// Σ A min(|e|/bin, 1) / Σ A over (A, |e|) readings.  NaN.min(1) is 1, so a lost reading counts
/// as a whole bin.
pub(super) fn garbage(readings: &[(f64, f64)], bin_c: f64) -> f64 {
    let (moved, total) = readings.iter().fold((0.0, 0.0), |(m, t), &(a, e)| {
        (m + a * (e / bin_c).min(1.0), t + a)
    });
    moved / total
}

/// |e| below which fraction `f` of the weight lands.
pub(super) fn weighted_quantile(readings: &mut [(f64, f64)], f: f64) -> f64 {
    readings.sort_by(|a, b| a.1.total_cmp(&b.1));
    let total: f64 = readings.iter().map(|r| r.0).sum();
    let mut acc = 0.0;
    readings
        .iter()
        .find(|r| {
            acc += r.0;
            acc >= f * total
        })
        .map_or(f64::NAN, |r| r.1)
}

/// |e| below which three quarters of the weight lands.
pub(super) fn upper_quartile(readings: &mut [(f64, f64)]) -> f64 {
    weighted_quantile(readings, 0.75)
}

/// Real and imaginary parts of ψ, centered, on a shared scale.
pub(super) fn print_wave(label: &str, psi: Fold<'_>, cols: usize) {
    let n = psi.len_unfolded();
    println!("\n=== {label} ===");
    let max = psi
        .mirrored()
        .map(|h| h.re.abs().max(h.im.abs()))
        .fold(0.0f64, f64::max);

    for (j, h) in psi.mirrored().enumerate() {
        println!(
            "{:>6} {:>12.7} {:>12.7} {}",
            j as isize - (n / 2) as isize,
            h.re,
            h.im,
            bar(h.re, h.im, max, cols)
        );
    }
}

/// Shading ramp, floor to peak.
const SHADE: [char; 5] = [' ', '░', '▒', '▓', '█'];

/// |W(t, ω₀)| over hops and bins, `floor_db` below the loudest cell.  Rows are `(ω₀, table)`
/// in print order, columns are hops [−span, span].
pub(super) fn print_transform(
    label: &str,
    bank: &[(f64, Weights)],
    x: impl Fn(isize) -> f64,
    span: isize,
    cols: usize,
    floor_db: f64,
) {
    let step = 2.0 * span as f64 / (cols - 1) as f64;
    let at = |c: usize| -(span as f64) + step * c as f64;
    let aa = step.ceil() as usize;

    // (1/aa) Σ |Ψ(m)| over the hops a column covers
    let mag: Vec<Vec<f64>> = bank
        .iter()
        .map(|(_, wts)| {
            (0..cols)
                .map(|c| {
                    (0..aa)
                        .map(|j| {
                            let u = at(c) + step * ((j as f64 + 0.5) / aa as f64 - 0.5);
                            wts.project(&x, u.round() as isize)[0].norm()
                        })
                        .sum::<f64>()
                        / aa as f64
                })
                .collect()
        })
        .collect();

    let peak = mag.iter().flatten().fold(0.0f64, |m, &v| m.max(v));

    println!("\n=== {label} ===");
    println!("  {floor_db:.0} dB to peak over {} hops", 2 * span);

    for (&(w0, _), row) in bank.iter().zip(&mag) {
        let cells: String = row
            .iter()
            .map(|&v| {
                let db = 20.0 * (v / peak).log10();
                let q = (1.0 - db / floor_db).clamp(0.0, 1.0);
                SHADE[(q * (SHADE.len() - 1) as f64).round() as usize]
            })
            .collect();
        println!("{w0:>9.6} {:>9.5} |{cells}|", w0 / TAU);
    }
}
