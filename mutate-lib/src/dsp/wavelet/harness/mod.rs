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

use num_complex::{Complex32, Complex64};

/// Folded weights back to centered taps.  Lane `c` selects ψ (0) or d (2).
pub(super) fn unfold(w: &[[f32; 4]], c: usize) -> Vec<Complex32> {
    let k = w.len();
    let mut out = vec![Complex32::default(); 2 * k - 1];
    out[k - 1] = Complex32::new(w[0][c], 0.0);
    for (j, q) in w.iter().enumerate().skip(1) {
        let h = Complex32::new(q[c], q[c + 1]);
        out[k - 1 + j] = h;
        out[k - 1 - j] = h.conj();
    }
    out
}

pub(super) fn widen(taps: &[Complex32]) -> Vec<Complex64> {
    taps.iter()
        .map(|h| Complex64::new(h.re as f64, h.im as f64))
        .collect()
}

/// H(ω) of centered taps, ω in rad/sample.
pub(super) fn dtft(taps: &[Complex32], w: f64) -> Complex64 {
    // Phase drift accumulates proportionate to sqrt(n), and reseed caps the walk.
    // Measured vs full re-seed out to about -270dB of difference, so well below what our
    // eventual storage is losing to f32 truncation already.
    //
    // Set RESEED to 1 for full seeding if this test device is under scrutiny.  **Must be power
    // of two for iteration mask.**
    const RESEED: usize = 512;

    let half = (taps.len() / 2) as f64;
    // e^{−iω}
    let step = Complex64::from_polar(1.0, -w);
    let (mut rot, mut acc) = (Complex64::default(), Complex64::default());

    for (j, h) in taps.iter().enumerate() {
        if j & (RESEED - 1) == 0 {
            // e^{iω(half − j)}
            rot = Complex64::from_polar(1.0, w * (half - j as f64));
        }
        acc += Complex64::new(h.re as f64, h.im as f64) * rot;
        rot *= step;
        rot *= 0.5 * (3.0 - rot.norm_sqr());
    }
    acc
}

/// M_p = Σ_ν ν^p h_ν
pub(super) fn moment(taps: &[Complex32], p: i32) -> Complex64 {
    let half = (taps.len() / 2) as isize;
    taps.iter()
        .enumerate()
        .map(|(j, h)| {
            let nu = (j as isize - half) as f64;
            Complex64::new(h.re as f64, h.im as f64) * nu.powi(p)
        })
        .sum()
}

/// |H_d(ω) − (ω/ω₀)·H_ψ(ω)|.  Absolute, because dividing by H_ψ is exactly what turns a
/// flat floor into a skirt blowup in the cents column.
pub(super) fn pairing_residual(psi: &[Complex32], d: &[Complex32], w0: f64, w: f64) -> f64 {
    (dtft(d, w) - dtft(psi, w) * (w / w0)).norm()
}

/// |H(ω)| and |H(−ω)|, the passband and its image.
fn pair(taps: &[Complex32], w: f64) -> (f64, f64) {
    (dtft(taps, w).norm(), dtft(taps, -w).norm())
}

/// Max of `f` over [lo, hi] at 16 samples per DTFT lobe of `len` taps.
fn sweep(lo: f64, hi: f64, len: usize, f: impl Fn(f64) -> f64) -> f64 {
    let n = ((hi - lo) * (16 * len) as f64 / TAU).ceil().max(1.0) as usize;
    (0..=n)
        .map(|k| f(lo + (hi - lo) * k as f64 / n as f64))
        .fold(0.0f64, f64::max)
}

/// Σ|h|, which bounds every envelope of `taps`.
pub(super) fn l1(taps: &[Complex32]) -> f64 {
    taps.iter().map(|h| h.norm() as f64).sum()
}

/// max |H| on [−0.05 ω₀, 0.05 ω₀].
pub(super) fn dc_leak(taps: &[Complex32], w0: f64) -> f64 {
    let e = 0.05 * w0;
    sweep(-e, e, taps.len(), |w| dtft(taps, w).norm())
}

/// W(m) = sum_j x[m + half - j] * h[j], matching `unit_tone_reads_unity`.
pub(super) fn conv(h: &[Complex64], x: impl Fn(isize) -> f64, m: isize) -> Complex64 {
    let half = (h.len() / 2) as isize;
    h.iter()
        .enumerate()
        .map(|(j, &h)| h * x(m + half - j as isize))
        .sum()
}

/// Ψ, D, and T about center `m`, accumulated as the shader does.
fn project(w: &[[f32; 4]], x: impl Fn(isize) -> f64, m: isize) -> [Complex64; 3] {
    let x0 = x(m);
    let mut psi = Complex64::new(w[0][0] as f64 * x0, 0.0);
    let mut dee = Complex64::new(w[0][2] as f64 * x0, 0.0);
    let mut tee = Complex64::default();

    for (k, c) in w.iter().enumerate().skip(1) {
        let [a, b, p, q] = c.map(f64::from);
        let (hi, lo) = (x(m + k as isize), x(m - k as isize));
        let (sum, dif) = (hi + lo, hi - lo);

        psi += Complex64::new(a * sum, -b * dif);
        dee += Complex64::new(p * sum, -q * dif);
        tee += k as f64 * Complex64::new(a * dif, -b * sum);
    }
    [psi, dee, tee]
}

/// Bias of `r̂` in cents and quadrature leak against a real tone detuned `cents` from `w0`,
/// each as its mean over the carrier phase and its worst departure from that mean.  Phases
/// step no coarser than `min_resolution` radians across the full circle.
pub(super) fn tone_bias(
    table: &[[f32; 4]],
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
            let [psi, dee, _] = project(table, tone, 0);
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
pub(super) fn tone_response(
    table: &[[f32; 4]],
    w0: f64,
    cents: f64,
    min_resolution: f64,
) -> [f64; 3] {
    // ω₀ · 2^(c/1200)
    let wd = w0 * (cents / 1200.0).exp2();
    let n = (TAU / min_resolution).ceil().max(2.0) as usize;

    // Ψ(θ) e^{−iθ}, D(θ) e^{−iθ}, T(θ) e^{−iθ}
    let out: Vec<[Complex64; 3]> = (0..n)
        .map(|j| {
            let theta = TAU * j as f64 / n as f64;
            let tone = |k: isize| (wd * k as f64 + theta).cos();
            let carrier = Complex64::from_polar(1.0, -theta);
            project(table, tone, 0).map(|v| v * carrier)
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

/// Worst |t̂ − t̂_ref| in samples per level bucket, plus the worst imaginary skew in the top
/// bucket.  Cumulative, so the -60 dB entry contains the -20 dB one.  Reassignment only has
/// to hold where the pixel is bright enough to see.
pub(super) fn t_hat_profile(
    table: &[[f32; 4]],
    reference: &[[f32; 4]],
    x: impl Fn(isize) -> f64,
    span: isize,
) -> ([f64; 3], f64) {
    /// Hop levels the buckets accumulate over, dB below the loudest hop.
    const LEVELS: [f64; 3] = [-20.0, -40.0, -60.0];

    let hops: Vec<(f64, f64, f64)> = (-span..=span)
        .map(|m| {
            let [p, _, t] = project(table, &x, m);
            let [rp, _, rt] = project(reference, &x, m);
            // T/Ψ
            let (q, r) = (t / p, rt / rp);
            (p.norm(), (q.re - r.re).abs(), q.im.abs())
        })
        .collect();

    let peak = hops.iter().fold(0.0f64, |a, h| a.max(h.0));
    let (mut worst, mut skew) = ([0.0f64; 3], 0.0f64);
    for (lvl, err, im) in hops {
        let db = 20.0 * (lvl / peak).log10();
        for (w, &l) in worst.iter_mut().zip(&LEVELS) {
            if db >= l {
                *w = w.max(err);
            }
        }
        if db >= LEVELS[0] {
            skew = skew.max(im);
        }
    }
    (worst, skew)
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

/// Real and imaginary parts of `taps`, centered, on a shared scale.
pub(super) fn print_wave(label: &str, taps: &[Complex32], cols: usize) {
    let n = taps.len();
    println!("\n=== {label} ===");
    let max = taps
        .iter()
        .map(|h| h.re.abs().max(h.im.abs()) as f64)
        .fold(0.0, f64::max);

    for (j, h) in taps.iter().enumerate() {
        println!(
            "{:>6} {:>12.7} {:>12.7} {}",
            j as isize - (n / 2) as isize,
            h.re,
            h.im,
            bar(h.re as f64, h.im as f64, max, cols)
        );
    }
}

/// ω and |H| at an extremum.
#[derive(Clone, Copy)]
pub(super) struct Point {
    pub w: f64,
    pub h: f64,
}

/// Extremum of `f` on [a, b], a maximum for `sign` 1 and a minimum for −1.
fn refine(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, sign: f64) -> Point {
    for _ in 0..80 {
        let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
        if sign * f(m1) < sign * f(m2) {
            a = m1;
        } else {
            b = m2;
        }
    }
    let w = 0.5 * (a + b);
    Point { w, h: f(w) }
}

/// ∫ h² dω on [lo, hi], h = max(0, 1 − dB/floor_db) with dB relative to `gain`.
pub(super) fn level_moment(
    taps: &[Complex32],
    gain: f64,
    (lo, hi): (f64, f64),
    floor_db: f64,
) -> f64 {
    let n = ((hi - lo) * (16 * taps.len()) as f64 / TAU).ceil().max(1.0) as usize;
    let dw = (hi - lo) / n as f64;
    (0..n)
        .map(|k| {
            let w = lo + dw * (k as f64 + 0.5);
            let db = 20.0 * (dtft(taps, w).norm() / gain).log10();
            (1.0 - db / floor_db).max(0.0).powi(2)
        })
        .sum::<f64>()
        * dw
}

/// Edges either side of `peak_w` where the response first falls `level_db` below `peak`,
/// searched out to `span`, absent where no crossing lies inside it.
fn shoulders(
    taps: &[Complex32],
    peak: f64,
    peak_w: f64,
    level_db: f64,
    span: f64,
    h: f64,
    density: f64,
) -> (Option<f64>, Option<f64>) {
    let db = |w: f64| 20.0 * dtft(taps, w).norm().log10();
    let rel = |w: f64| db(w) - 20.0 * peak.log10();
    let edge =
        |stop: f64| level_crossing(&rel, level_db, peak_w, stop, 0.0, h, density).map(|p| p.w);
    (
        edge((peak_w - span).max(0.0)),
        edge((peak_w + span).min(PI)),
    )
}

/// −3 dB edges either side of `peak_w`, falling back to the bracket `w0` sets.
pub(super) fn bandwidth(
    taps: &[Complex32],
    peak: f64,
    peak_w: f64,
    w0: f64,
    h: f64,
    density: f64,
) -> (f64, f64) {
    let half_db = -10.0 * 2.0f64.log10();
    let (lo, hi) = shoulders(taps, peak, peak_w, half_db, w0, h, density);
    (
        lo.unwrap_or((peak_w - w0).max(0.0)),
        hi.unwrap_or((peak_w + w0).min(PI)),
    )
}

/// Extrema of |H| across one stopband, tallest first.
pub(super) struct Skirt {
    /// Largest local maximum, absent where the band holds none.
    pub peak: Option<Point>,
    /// Median local maximum, the ripple level the peak stands on.
    pub median: f64,
    /// Local maxima found.
    pub lobes: usize,
}

impl Skirt {
    /// 20 log10 (peak / median), how far the tallest lobe clears the ripple.
    pub fn prominence_db(&self) -> Option<f64> {
        self.peak.map(|p| 20.0 * (p.h / self.median).log10())
    }
}

/// Samples per null spacing 2π/N.  Four resolves every extremum a degree N−1
/// trigonometric polynomial admits.
const SKIRT_OVERSAMPLE: usize = 8;

/// Local maxima refined before the tallest is chosen.
const SKIRT_REFINE: usize = 4;

/// Local maxima of `f` on a uniform grid over [lo, hi], as brackets.
fn crests(f: impl Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64, f64)> {
    let at = |j: usize| lo + (hi - lo) * j as f64 / n as f64;
    let g: Vec<f64> = (0..=n).map(|j| f(at(j))).collect();
    (1..n)
        .filter(|&j| g[j - 1] <= g[j] && g[j] > g[j + 1])
        .map(|j| (at(j - 1), at(j + 1), g[j]))
        .collect()
}

/// |H| over the stopband between `from` and `stop`, scanned whole.
fn skirt(taps: &[Complex32], from: f64, stop: f64, oversample: usize, refine_top: usize) -> Skirt {
    let gain = |w: f64| dtft(taps, w).norm();
    let (lo, hi) = (from.min(stop), from.max(stop));
    let n = ((hi - lo) * (oversample * taps.len()) as f64 / TAU)
        .ceil()
        .max(2.0) as usize;

    let mut found = crests(gain, lo, hi, n);
    found.sort_by(|a, b| b.2.total_cmp(&a.2));

    let peak = found
        .iter()
        .take(refine_top)
        .map(|&(a, b, _)| refine(gain, a, b, 1.0))
        .max_by(|a, b| a.h.total_cmp(&b.h));

    let median = found.get(found.len() / 2).map_or(f64::NAN, |&(_, _, h)| h);

    Skirt {
        peak,
        median,
        lobes: found.len(),
    }
}

/// Both stopbands outside the first nulls, lower then upper.
pub(super) fn skirts(taps: &[Complex32], (lo, hi): (Point, Point)) -> (Skirt, Skirt) {
    (
        skirt(taps, lo.w, 0.0, SKIRT_OVERSAMPLE, SKIRT_REFINE),
        skirt(taps, hi.w, PI, SKIRT_OVERSAMPLE, SKIRT_REFINE),
    )
}

/// Shared stencil walk toward a level crossing of `resp − level`, from `from` toward `stop`,
/// within `tol`.
fn level_crossing(
    resp: impl Fn(f64) -> f64,
    level: f64,
    from: f64,
    stop: f64,
    tol: f64,
    h: f64,
    density: f64,
) -> Option<Point> {
    let shifted = |w: f64| resp(w) - level;
    let dir = (stop - from).signum();
    let clamp = |w: f64| if dir * (w - stop) > 0.0 { stop } else { w };
    let cap = 8.0 / density;

    let root = |a: f64, b: f64| {
        let w = bisect(&shifted, a, b);
        Point {
            w,
            h: shifted(w).abs(),
        }
    };
    let cross = |p: &[Sample]| first_pair(p, |a, b| (a.1 < 0.0) != (b.1 < 0.0));
    let toward_dir = |l: &Local| l.toward(dir);
    let scan = |a: f64, b: f64| {
        let n = ((b - a).abs() * density).ceil().max(1.0) as usize;
        let at = |j: usize| a + (b - a) * j as f64 / n as f64;
        let p: Vec<Sample> = (0..=n).map(|j| (at(j), shifted(at(j)))).collect();
        cross(&p).map_or(b, |(x, y)| root(x.0, y.0).w)
    };

    let old = Local::at(&shifted, from, h);
    if old.g[1].abs() <= tol {
        return Some(Point {
            w: from,
            h: old.g[1].abs(),
        });
    }

    let mut old = old;
    let mut jump = 2.0 * h;
    loop {
        jump = toward_dir(&old)
            .map_or(2.0 * jump, |t| (t - old.w).abs())
            .clamp(2.0 * h, cap);
        let land = clamp(old.w + dir * jump);
        let new = Local::at(&shifted, land, h);

        if new.g[1].abs() <= tol && (old.g[1] < 0.0) != (new.g[1] < 0.0) {
            return Some(Point {
                w: land,
                h: new.g[1].abs(),
            });
        }

        if let Some(t) = cross(&order([old.points(), new.points()].concat(), dir)) {
            let w = tighten(&shifted, t, old, h, density, toward_dir, cross, scan);
            return Some(root(w - h, w + h));
        }

        let (lo, hi) = (old.w.min(land), old.w.max(land));
        if new.toward(-dir).is_some_and(|t| t > lo && t < hi) {
            let n = ((land - old.w).abs() * density).ceil().max(1.0) as usize;
            let at = |j: usize| old.w + (land - old.w) * j as f64 / n as f64;
            let p: Vec<Sample> = (0..=n).map(|j| (at(j), shifted(at(j)))).collect();
            if let Some((a, b)) = cross(&p) {
                return Some(root(a.0, b.0));
            }
        }

        if land == stop {
            return None;
        }
        old = new;
    }
}

/// First zero crossing of the real response H walking from `from` toward `stop`.
pub(super) fn first_null(
    taps: &[Complex32],
    from: f64,
    stop: f64,
    null: f64,
    h: f64,
    density: f64,
) -> Option<Point> {
    let response = |w| dtft(taps, w).re;
    level_crossing(response, 0.0, from, stop, null, h, density)
}

/// τ solving y + sτ + ½kτ² = r, non-finite where absent.
fn roots(y: f64, s: f64, k: f64, r: f64) -> [f64; 2] {
    let c = r - y;
    if k.abs() < f64::EPSILON {
        return [c / s, f64::NAN];
    }
    // √(s² + 2kc)
    let disc = (s * s + 2.0 * k * c).sqrt();
    [(-s - disc) / k, (-s + disc) / k]
}

/// ω and a value there.
type Sample = (f64, f64);

/// A bracket around a feature, outer samples in walking order.
type Bracket = (Sample, Sample);

/// DTFTs one stencil costs, used to price scanning against tightening.
const STENCIL: f64 = 3.0;

/// Samples ordered along +dir.
fn order(mut p: Vec<Sample>, dir: f64) -> Vec<Sample> {
    p.sort_by(|a, b| (dir * a.0).total_cmp(&(dir * b.0)));
    p
}

/// First adjacent pair satisfying `hit`.
fn first_pair(p: &[Sample], hit: impl Fn(Sample, Sample) -> bool) -> Option<Bracket> {
    p.windows(2).map(|w| (w[0], w[1])).find(|&(a, b)| hit(a, b))
}

/// First triple satisfying `hit`, reduced to its outer pair.
fn first_triple(p: &[Sample], hit: impl Fn(Sample, Sample, Sample) -> bool) -> Option<Bracket> {
    p.windows(3)
        .map(|w| (w[0], w[1], w[2]))
        .find(|&(a, b, c)| hit(a, b, c))
        .map(|(a, _, c)| (a, c))
}

/// Shrink `t` by aiming stencils with `predict`, re-witnessing with `witness` after each
/// landing, until a step stops saving more DTFTs than it costs, then hand the remainder
/// to `scan`.
fn tighten(
    resp: impl Fn(f64) -> f64,
    mut t: Bracket,
    mut aim: Local,
    h: f64,
    density: f64,
    predict: impl Fn(&Local) -> Option<f64>,
    witness: impl Fn(&[Sample]) -> Option<Bracket>,
    scan: impl Fn(f64, f64) -> f64,
) -> f64 {
    loop {
        let width = (t.1 .0 - t.0 .0).abs() * density;
        let inside = |v: f64| {
            let (a, b) = (t.0 .0.min(t.1 .0) + h, t.0 .0.max(t.1 .0) - h);
            v > a && v < b
        };

        let Some(v) = predict(&aim).filter(|&v| inside(v)) else {
            return scan(t.0 .0, t.1 .0);
        };
        if width <= STENCIL {
            return scan(t.0 .0, t.1 .0);
        }

        let next = Local::at(&resp, v, h);
        let dir = (t.1 .0 - t.0 .0).signum();
        let mut p = vec![t.0, t.1];
        p.extend(next.points());
        let Some(u) = witness(&order(p, dir)) else {
            return scan(t.0 .0, t.1 .0);
        };

        if (u.1 .0 - u.0 .0).abs() * density > width - STENCIL {
            return scan(u.0 .0, u.1 .0);
        }
        (t, aim) = (u, next);
    }
}

/// ω of the highest `db` in the basin around `from`, within [lo, hi].
fn climb(taps: &[Complex32], from: f64, lo: f64, hi: f64, h: f64, density: f64) -> f64 {
    let db = |w: f64| 20.0 * dtft(taps, w).norm().log10();
    let cap = 8.0 / density;
    let scan = |a: f64, b: f64| {
        let n = ((b - a).abs() * density).ceil().max(2.0) as usize;
        let at = |j: usize| (a + (b - a) * j as f64 / n as f64).clamp(lo, hi);
        let p: Vec<Sample> = (0..=n).map(|j| (at(j), db(at(j)))).collect();
        let (x, y) = first_triple(&p, |a, b, c| a.1 <= b.1 && b.1 > c.1).unwrap_or((p[0], p[n]));
        refine(&db, x.0.min(y.0), x.0.max(y.0), 1.0).w
    };
    let crest = |p: &[Sample]| first_triple(p, |a, b, c| a.1 <= b.1 && b.1 > c.1);
    let concave_vertex = |l: &Local| (l.k < 0.0).then(|| l.vertex());

    let old = Local::at(&db, from, h);
    let dir = old.s.signum();
    let stop = if dir > 0.0 { hi } else { lo };
    let clamp = |w: f64| if dir * (w - stop) > 0.0 { stop } else { w };

    if let Some(t) = crest(&order(old.points().to_vec(), dir)) {
        return tighten(&db, t, old, h, density, concave_vertex, crest, scan);
    }

    let mut old = old;
    let mut jump = 2.0 * h;
    loop {
        jump = concave_vertex(&old)
            .map_or(2.0 * jump, |v| (v - old.w).abs())
            .clamp(2.0 * h, cap);
        let land = clamp(old.w + dir * jump);
        let new = Local::at(&db, land, h);

        if let Some(t) = crest(&order([old.points(), new.points()].concat(), dir)) {
            let aim = if new.k < 0.0 { new } else { old };
            return tighten(&db, t, aim, h, density, concave_vertex, crest, scan);
        }

        let (a, b) = (old.w.min(land), old.w.max(land));
        let v = new.vertex();
        if new.k > 0.0 && v > a && v < b {
            return scan(old.w - dir * h, land);
        }

        if land == stop {
            return stop;
        }
        old = new;
    }
}

/// Quadratic model of H on [w − h, w + h], slope and curvature along +ω.
#[derive(Clone, Copy)]
struct Local {
    w: f64,
    h: f64,
    g: [f64; 3],
    s: f64,
    k: f64,
}

impl Local {
    fn at(resp: impl Fn(f64) -> f64, w: f64, h: f64) -> Self {
        let g = [resp(w - h), resp(w), resp(w + h)];
        Local {
            w,
            h,
            g,
            // (g₊ − g₋) / 2h
            s: (g[2] - g[0]) / (2.0 * h),
            // (g₊ − 2g₀ + g₋) / h²
            k: (g[2] - 2.0 * g[1] + g[0]) / (h * h),
        }
    }

    /// Stencil samples along +ω.
    fn points(&self) -> [Sample; 3] {
        [
            (self.w - self.h, self.g[0]),
            (self.w, self.g[1]),
            (self.w + self.h, self.g[2]),
        ]
    }

    /// Vertex of the model.
    fn vertex(&self) -> f64 {
        // w − s/k
        self.w - self.s / self.k
    }

    /// Nearest root of the model on the `dir` side of center, if any.
    fn toward(&self, dir: f64) -> Option<f64> {
        let [a, b] = roots(self.g[1], self.s, self.k, 0.0);
        [a, b]
            .into_iter()
            .filter(|t| t.is_finite() && dir * *t > 0.0)
            .map(|t| self.w + t)
            .min_by(|a, b| (a - self.w).abs().total_cmp(&(b - self.w).abs()))
    }
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

/// Peak, -3 dB relative width, image, and the positive-axis floor outside three half-power widths.
/// `w0` only brackets the edges.
pub(super) fn characterize(taps: &[Complex32], w0: f64) -> Response {
    let sweep = (16 * taps.len()).next_power_of_two();
    let omega = |k: usize| PI * k as f64 / sweep as f64;
    let gain = |w: f64| dtft(taps, w).norm();

    let resp: Vec<(f64, f64)> = (0..=sweep).map(|k| pair(taps, omega(k))).collect();

    let image = resp.iter().fold(0.0f64, |m, &(_, i)| m.max(i));

    let density = 16.0 * taps.len() as f64 / TAU;
    let h = 4.0 / density;

    let peak_w = climb(taps, w0, 0.0, PI, h, density);
    let peak = dtft(taps, peak_w).norm();
    let (lo, hi) = bandwidth(taps, peak, peak_w, w0, h, density);

    let guard = 3.0 * (hi - lo);
    let floor = resp
        .iter()
        .enumerate()
        .filter(|&(k, _)| (omega(k) - peak_w).abs() > guard)
        .fold(0.0f64, |f, (_, &(g, _))| f.max(g));

    Response {
        peak_w,
        gain: peak,
        edges: (lo, hi),
        rel_width: (hi - lo) / peak_w,
        image,
        floor,
    }
}
