// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Calibrate
//!
//! Measure filter behavior versus configuration parameters.  Predict [`Bin`] configurations needed
//! to achieve goal behaviors without baking any [`Wavelet`]s.
//!
//! > Poca agua bendita sale de un dique reventado.
//! >
//! > - Ernesto Guaraná
//!
//! ## Motivation
//!
//! We want to plan filter banks.  This requires control over filter behavior at each bin.  However,
//! even at a constant Q and truncation setting:
//!
//! - The noise floor rises with ρ (our filters are L1 normalized by default)
//! - Wide main lobes and skirts begin to fold after the Nyquist limit, adding mirror response to at
//!   negative frequency, corrupting reassignment signal.
//! - High truncation settings begin eating the filter body, broadening the main lobes.
//! - Mild analytic deviations exist across γ, Q, and
//!
//! All of this means that the filter the user asks for can never be the filter they get without
//! using a complete model of the related behaviors.
//!
//! Furthermore, modeling the critical `Q_reassign`, the effective bandwidth after including the
//! precision of pitch reassignment, can only be derived from an accurate model of signal-to-noise
//! ratio in response to musically relevant pitch offsets.  Inaccurate, purely analytic
//! understanding of main lobe widths and skirts would compound errors in those estimates, leading
//! to banded response patterns where parts of the bank perform as advertised while other parts are
//! frustrated by the local bias patterns.
//!
//! Finally, measuring the intended response for every single wavelet through trial-and-error would
//! require building a [`Wavelet`] for each trial, which gets expensive fast.  Both banks and tests
//! need many wavelets.  Persisting a calibrated map of the responses for a whole family of wavelets
//! enables cheap predictions.  Because each relation carries the analytic shape of what it
//! measures, a fit is a handful of coefficients.
//!
//! ## Primary and Derived Quantities
//!
//! The tail at the cut and the Nyquist fold are analytic.  PSL and main lobe width are measured and
//! tight.  Floor is the skirt read at the antipode, so it fits only the decay out from the PSL.
//! The fold is the main lobe at Nyquist and matters only where it clears the PSL.  Center stays
//! under a cent and is held as a contract rather than predicted.
//!
//! Width alone knees.  A shallow cut eats the main lobe in proportion to the envelope left at the
//! cut, which is gone by x ≈ 3.5.  The knee is fit in log, where it is linear, so its deep tail
//! weighs as much as its shallow crest.  PSL crosses its shoulder handoff inside its margin.
//!
//! ## Key Relations
//!
//! ```text
//! u        ρ(K − 1), realized half span in carrier periods
//! x        2πu / P, envelope sigmas at the cut
//! t        truncation_tail(u)
//! R        20 log10(Q/ρ), twice the main lobe widths out to the antipode
//!
//! psl      = t (1 + p_slope/P) + p_0 + ρ (p_rho + p_lobe/Q)
//! floor    = psl + f_0 + f_tail t − f_reach R
//! fold     = image_db(ρ)
//! width·Q  = 1 + w_knee xⁿ e^{−m x²/2}
//! Q_real    = Q / width·Q
//! ```
//!
//! Predictions are envelopes.  Levels rise by a margin and the knee scales by one, each covering
//! `ENVELOPE` of measured bins, so plans over deliver rather than miss.  Levels on table rounding
//! and cells whose fold breaks the shallowest plan stay out of the fits.
//!
//! Predictions exist only at realized halves.  Measurement records every half at unit quantum, so
//! each fit point carries just the taps its length needs.  A plan takes the least half on its
//! quantum whose envelope holds, and taps rounding adds past that only deepen what it delivers.

// 🤖 Mostly generated so far.  Working on the user API.  Keep it working, but don't worry about
// stepping on anyone's pet lines of code.

use core::f64::consts::{PI, TAU};

use super::inspect::{bisect, characterize, db, skirts, Inspect, OVERSAMPLE};
use super::spec::Shape;
use super::{refine, restrict, Bin, WaveletSpec};

/// Lowest delivered Q.  Shorter filters have a main lobe too wide to place a pitch.
pub const Q_REAL_MIN: f64 = 1.0;
/// Deepest floor a plan promises.
pub const DEEPEST_DB: f64 = -120.0;
/// Shallowest floor a plan targets.  Cells whose fold breaks it are never planned.
const SHALLOWEST_DB: f64 = -40.0;
/// Levels on f32 table rounding, where they plateau whatever the cut.
const QUANTIZED_DB: f64 = -135.0;
/// Share of measured bins each envelope covers.
const ENVELOPE: f64 = 0.9;
/// Knee excess the width fit and its envelope resolve, clear of bulk offsets.
const KNEE_GATE: f64 = 0.02;

/// Everything shaping a bake besides Q and its maxima.
#[derive(Clone, Copy)]
pub struct Settings {
    pub gamma: f64,
    pub resolution: usize,
    pub restriction: restrict::Restriction,
    pub refinement: Option<refine::Refinement>,
}

impl Default for Settings {
    fn default() -> Self {
        let s = WaveletSpec::default();
        Settings {
            gamma: s.shape.gamma,
            resolution: s.resolution,
            restriction: s.restriction,
            refinement: s.refinement,
        }
    }
}

impl Settings {
    /// Default restriction and refinement at γ 3.
    pub fn knee() -> Self {
        Settings {
            gamma: 3.0,
            ..Settings::default()
        }
    }

    pub fn shape(&self, q: f64) -> Shape {
        Shape::from_q(q, self.gamma)
    }

    /// A family at `q`, maxima left to the caller.
    pub fn wavelet(&self, q: f64) -> WaveletSpec {
        WaveletSpec::default()
            .shape(self.shape(q))
            .resolution(self.resolution)
            .restriction(self.restriction)
            .refinement(self.refinement)
    }
}

/// Grid a calibration measures.
#[derive(Clone, Copy)]
pub struct Sweep {
    pub qs: &'static [f64],
    pub rhos: &'static [f64],
    /// (lo, hi, step) of x
    pub xs: (f64, f64, f64),
}

impl Default for Sweep {
    fn default() -> Self {
        Sweep {
            qs: &[3.5, 6.0, 12.0, 20.0],
            rhos: &[0.05, 0.15, 0.2, 0.25, 0.3, 0.4],
            xs: (2.0, 5.25, 0.1),
        }
    }
}

/// Lifts from each fitted relation to the envelope `ENVELOPE` of measured bins sit under.
#[derive(Clone, Copy, Debug)]
pub struct Margin {
    /// dB
    pub psl: f64,
    /// dB
    pub floor: f64,
    /// dB over the analytic fold
    pub fold: f64,
    /// Scale on the width knee
    pub knee: f64,
}

impl Margin {
    /// The fitted relations themselves.
    const NONE: Margin = Margin {
        psl: 0.0,
        floor: 0.0,
        fold: 0.0,
        knee: 1.0,
    };
}

/// Skirt level re the PSL at fπ from the crest.
///
/// ```text
/// L − psl = base + tail t − reach R(f)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Decay {
    pub base: f64,
    /// Clear lost per dB of cut
    pub tail: f64,
    /// |H| ∝ d^{−reach}
    pub reach: f64,
}

impl Decay {
    fn at(&self, psl: f64, tail: f64, q: f64, rho: f64, f: f64) -> f64 {
        psl + self.base + self.tail * tail - self.reach * reach_db(q, rho, f)
    }

    /// Fit over (obs, L, f).
    fn fit<'a>(rows: impl Iterator<Item = (&'a Obs, f64, f64)>) -> Self {
        let rows: Vec<_> = rows
            .map(|(o, level, f)| ([1.0, o.tail, -reach_db(o.q, o.rho, f)], level - o.psl))
            .collect();
        let [base, tail, reach] = lstsq(&rows);
        Decay { base, tail, reach }
    }
}

/// Coefficients of the module's relations.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub p_0: f64,
    /// Tail slope excess in units of 1/P
    pub p_slope: f64,
    pub p_rho: f64,
    pub p_lobe: f64,
    /// Skirt at the antipode
    pub floor: Decay,
    pub w_knee: f64,
    pub w_n: f64,
    pub w_m: f64,
    /// x the fit was measured over.
    pub span: (f64, f64),
    pub margin: Margin,
}

impl Fit {
    /// Priors.  Rectangular Gibbs lobe under the tail, C¹ skirt from a lobe 2.7 widths out,
    /// Gaussian edge knee.
    pub const ANALYTIC: Fit = Fit {
        p_0: -13.26,
        p_slope: 0.0,
        p_rho: 0.0,
        p_lobe: 0.0,
        floor: Decay {
            base: 29.3,
            tail: 0.0,
            reach: 2.0,
        },
        w_knee: 0.7,
        w_n: 2.0,
        w_m: 1.0,
        span: (1.7, 5.25),
        margin: Margin {
            psl: 3.0,
            floor: 10.0,
            fold: 3.0,
            knee: 1.2,
        },
    };

    /// Knee settings, printed by `calibration_is_fit` once accepted.
    pub const CALIBRATED: Fit = Fit {
        p_0: -13.07015008941905,
        p_slope: -0.5883677857968512,
        p_rho: 2.548070102227145,
        p_lobe: 9.0243837366645,
        floor: Decay {
            base: 37.24895110631419,
            tail: -0.12981152500147797,
            reach: 2.6614525619662492,
        },
        w_knee: 1.033828933140709,
        w_n: 1.1611723927192519,
        w_m: 0.8620850505812259,
        span: (2.009355381156725, 5.498437260308141),
        margin: Margin {
            psl: 2.641687032523194,
            floor: 2.298835040603791,
            fold: 3.6338203145165835,
            knee: 1.0400243351581635,
        },
    };

    /// Relations fit to `obs` at `gamma`, each independently of the others.
    pub(super) fn new(gamma: f64, obs: &[Obs]) -> Self {
        let lobed: Vec<&Obs> = obs.iter().filter(|o| resolved(o.psl)).collect();

        // psl − t
        let rows: Vec<_> = lobed
            .iter()
            .map(|o| {
                let p = Shape::from_q(o.q, gamma).p();
                ([o.tail / p, 1.0, o.rho, o.rho / o.q], o.psl - o.tail)
            })
            .collect();
        let [p_slope, p_0, p_rho, p_lobe] = lstsq(&rows);

        let floor = Decay::fit(
            lobed
                .iter()
                .filter(|o| resolved(o.floor))
                .map(|o| (*o, o.floor, 1.0)),
        );

        // ln(width·Q − 1) = ln w_knee + n ln x − m x²/2
        let rows: Vec<_> = obs
            .iter()
            .filter(|o| o.width_q - 1.0 >= KNEE_GATE)
            .map(|o| ([1.0, o.x.ln(), -0.5 * o.x * o.x], (o.width_q - 1.0).ln()))
            .collect();
        let [ln_w, w_n, w_m] = lstsq(&rows);

        let xs = obs.iter().map(|o| o.x);
        let span = (
            xs.clone().fold(f64::INFINITY, f64::min),
            xs.fold(f64::NEG_INFINITY, f64::max),
        );

        let fit = Fit {
            p_0,
            p_slope,
            p_rho,
            p_lobe,
            floor,
            w_knee: ln_w.exp(),
            w_n,
            w_m,
            span,
            margin: Margin::NONE,
        };
        Fit {
            margin: fit.margin_over(gamma, obs),
            ..fit
        }
    }

    /// Envelope at `shape` and `rho` with `half` folded weights past the center.
    pub fn predict(&self, shape: Shape, rho: f64, half: usize) -> Prediction {
        let (q, p) = (shape.q(), shape.p());
        let u = rho * (half + 1) as f64;
        let (tail, x) = (shape.truncation_tail(u), TAU * u / p);
        let m = self.margin;

        let psl = tail * (1.0 + self.p_slope / p) + self.p_0 + rho * (self.p_rho + self.p_lobe / q);
        let floor = self.floor.at(psl, tail, q, rho, 1.0);

        Prediction {
            q,
            rho,
            half,
            u,
            x,
            tail,
            psl: psl + m.psl,
            floor: floor + m.floor,
            fold: shape.image_db(rho) + m.fold,
            width_q: 1.0 + m.knee * self.w_knee * knee(x, self.w_n, self.w_m),
        }
    }

    /// Margins lifting these relations over `ENVELOPE` of `obs`.
    fn margin_over(&self, gamma: f64, obs: &[Obs]) -> Margin {
        let pairs: Vec<(&Obs, Prediction)> = obs
            .iter()
            .map(|o| {
                (
                    o,
                    self.predict(Shape::from_q(o.q, gamma), o.rho, o.taps / 2),
                )
            })
            .collect();

        // q_ENVELOPE of seen − fit over resolved levels
        let lift = |seen: fn(&Obs) -> f64, fit: fn(&Prediction) -> f64| {
            let v = pairs
                .iter()
                .filter(|(o, _)| resolved(seen(o)))
                .map(|(o, p)| seen(o) - fit(p))
                .collect();
            quantile(v, ENVELOPE)
        };

        // q_ENVELOPE of seen − analytic fold where the main lobe clears the PSL at Nyquist
        let fold = pairs
            .iter()
            .filter(|(o, p)| resolved(o.fold) && p.fold > o.psl)
            .map(|(o, p)| o.fold - p.fold)
            .collect();

        // q_ENVELOPE of seen over fitted knee excess where the measured knee resolves
        let knee = pairs
            .iter()
            .filter(|(o, _)| o.width_q - 1.0 >= KNEE_GATE)
            .map(|(o, p)| (o.width_q - 1.0) / (p.width_q - 1.0))
            .collect();

        Margin {
            psl: lift(|o| o.psl, |p| p.psl),
            floor: lift(|o| o.floor, |p| p.floor),
            fold: quantile(fold, ENVELOPE),
            knee: quantile(knee, ENVELOPE),
        }
    }

    /// Tail whose PSL envelope lands on `psl` at `rho`.
    ///
    /// ```text
    /// t = (psl − m_psl − p_0 − ρ(p_rho + p_lobe/Q)) / (1 + p_slope/P)
    /// ```
    pub fn tail_for_psl(&self, shape: Shape, rho: f64, psl: f64) -> f64 {
        let (q, p) = (shape.q(), shape.p());
        (psl - self.margin.psl - self.p_0 - rho * (self.p_rho + self.p_lobe / q))
            / (1.0 + self.p_slope / p)
    }
}

/// Response a calibration promises at a realized length.  Levels in dB re the crest, each an
/// envelope.
#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub q: f64,
    pub rho: f64,
    /// Folded weights past the center.
    pub half: usize,
    /// ρ half
    pub u: f64,
    pub x: f64,
    /// Feeds `with_truncation`.
    pub tail: f64,
    pub psl: f64,
    pub floor: f64,
    /// Main lobe at Nyquist
    pub fold: f64,
    pub width_q: f64,
}

impl Prediction {
    /// Q / width·Q
    pub fn q_real(&self) -> f64 {
        self.q / self.width_q
    }
}

#[derive(Clone, Copy)]
pub struct Calibration {
    pub settings: Settings,
    pub fit: Fit,
}

impl Calibration {
    /// Measure `sweep` under `settings` and fit.
    pub fn new(settings: Settings, sweep: Sweep) -> Self {
        let obs = measure(settings, sweep);
        Calibration {
            settings,
            fit: Fit::new(settings.gamma, &obs),
        }
    }

    pub fn calibrated() -> Self {
        Calibration {
            settings: Settings::knee(),
            fit: Fit::CALIBRATED,
        }
    }

    /// Envelope at `q` and `rho` with `half` folded weights past the center.
    pub fn predict(&self, q: f64, rho: f64, half: usize) -> Prediction {
        self.fit.predict(self.settings.shape(q), rho, half)
    }

    /// Least half on `quantum` whose floor envelope holds `floor`, absent past `DEEPEST_DB`.
    pub fn half_for_floor(&self, q: f64, rho: f64, floor: f64, quantum: usize) -> Option<usize> {
        self.least_half(q, rho, quantum, floor, |p| p.floor)
    }

    /// Least half on `quantum` whose PSL envelope holds `psl`, absent past `DEEPEST_DB`.
    pub fn half_for_psl(&self, q: f64, rho: f64, psl: f64, quantum: usize) -> Option<usize> {
        self.least_half(q, rho, quantum, psl, |p| p.psl)
    }

    /// Least half on `quantum` whose `level` envelope holds `level_db`.
    fn least_half(
        &self,
        q: f64,
        rho: f64,
        quantum: usize,
        level_db: f64,
        level: fn(&Prediction) -> f64,
    ) -> Option<usize> {
        let goal = -level_db.abs();
        if goal < DEEPEST_DB {
            return None;
        }
        let (lo, hi) = self.reach(q, rho, quantum)?;
        least(
            |n| level(&self.predict(q, rho, n * quantum)) <= goal,
            lo,
            hi,
        )
        .map(|n| n * quantum)
    }

    /// Envelope of a realized bin baked under these settings.
    pub fn predict_bin(&self, bin: Bin<'_>) -> Prediction {
        self.predict(bin.wavelet.shape.q(), bin.rho(), bin.len_folded() - 1)
    }

    /// Multiples of `quantum` whose halves the fit measured and whose Q_real holds `Q_REAL_MIN`.
    fn reach(&self, q: f64, rho: f64, quantum: usize) -> Option<(usize, usize)> {
        // (x P / 2πρ − 1) / quantum, the halves of K spanning x
        let scale = self.settings.shape(q).p() / (TAU * rho);
        let multiple = |x: f64| (x * scale - 1.0) / quantum as f64;
        let (lo, hi) = (
            multiple(self.fit.span.0).ceil() as usize,
            multiple(self.fit.span.1).floor() as usize,
        );
        let lo = least(
            |n| self.predict(q, rho, n * quantum).q_real() >= Q_REAL_MIN,
            lo,
            hi,
        )?;
        Some((lo, hi))
    }

    /// Q at least `q` whose fold envelope holds `floor` at `rho`, at the least half on `quantum`
    /// holding it past the fold.
    pub fn plan(&self, q: f64, rho: f64, floor: f64, quantum: usize) -> Option<Prediction> {
        // goal − fold margin
        let limit = -floor.abs() - self.fit.margin.fold;
        let fold = self.settings.shape(q).image_db(rho);
        // fold ∝ β ∝ Q²
        let q = match fold > limit {
            true => q * (limit / fold).sqrt(),
            false => q,
        };
        self.half_for_floor(q, rho, floor, quantum)
            .map(|half| self.predict(q, rho, half))
    }
}

/// One realized bin as delivered.  Levels in dB re the crest, center in cents.
#[derive(Clone, Copy)]
pub(super) struct Obs {
    pub q: f64,
    pub rho: f64,
    pub u: f64,
    pub x: f64,
    pub taps: usize,
    pub tail: f64,
    pub psl: f64,
    pub floor: f64,
    pub fold: f64,
    pub width_q: f64,
    pub center: f64,
}

/// Absent where the bin has no crest or delivers Q under `Q_REAL_MIN`.
pub(super) fn observe(bin: Bin<'_>) -> Option<Obs> {
    let shape = bin.wavelet.shape;
    let (q, rho, w0) = (shape.q(), bin.rho(), bin.velocity());
    let u = rho * bin.len_folded() as f64;
    let wts = bin.weights();
    let psi = wts.psi();

    let mut buf = Vec::new();
    Inspect::new(psi, &mut buf, OVERSAMPLE).peak(w0, 0.0, PI)?;
    let r = characterize(psi, w0);
    // Q_real = 1 / rel_width
    if r.rel_width * Q_REAL_MIN > 1.0 {
        return None;
    }
    let rel = |v: f64| db(v) - db(r.gain);

    let (lo, hi) = skirts(psi, &r);
    let psl = [lo.peak, hi.peak]
        .into_iter()
        .flatten()
        .map(|(_, h)| rel(h))
        .fold(f64::NAN, f64::max);

    Some(Obs {
        q,
        rho,
        u,
        x: TAU * u / shape.p(),
        taps: bin.len_unfolded(),
        tail: shape.truncation_tail(u),
        psl,
        floor: rel(r.floor),
        fold: rel(psi.dtft(PI).abs()),
        width_q: r.rel_width * q,
        center: 1200.0 * (r.peak_w / w0).log2(),
    })
}

/// Level above table rounding, false where absent.
fn resolved(level: f64) -> bool {
    level > QUANTIZED_DB
}

/// 20 log10(fQ/ρ), twice the main lobe widths out to fπ from the crest
fn reach_db(q: f64, rho: f64, f: f64) -> f64 {
    20.0 * (f * q / rho).log10()
}

/// xⁿ e^{−m x²/2}
fn knee(x: f64, n: f64, m: f64) -> f64 {
    x.powf(n) * (-0.5 * m * x * x).exp()
}

/// Every distinct realized bin of `sweep` at unit quantum, starting a step inside the taper's
/// domain, over cells whose fold holds the shallowest plan.
pub(super) fn measure(settings: Settings, sweep: Sweep) -> Vec<Obs> {
    let (lo, hi, step) = sweep.xs;
    let xs: Vec<f64> =
        grid((lo.max(settings.restriction.taper.x_min() + step), hi, step)).collect();
    let deepest = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out = Vec::new();

    for &q in sweep.qs {
        let shape = settings.shape(q);
        // u = x P / 2π
        let u = |x: f64| x * shape.p() / TAU;
        let wav = settings
            .wavelet(q)
            .max_load_quantum(1)
            .max_half_span(u(deepest))
            .bake();

        for &rho in sweep.rhos {
            if shape.image_db(rho) > SHALLOWEST_DB {
                continue;
            }
            let mut last = 0;
            for &x in &xs {
                let half = (u(x) / rho).ceil() as usize - 1;
                if half != last {
                    last = half;
                    out.extend(observe(wav.at_reach(rho, half)));
                }
            }
        }
    }
    out
}

/// Value under which share `f` of `v` falls, NaN over nothing.
fn quantile(mut v: Vec<f64>, f: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * f).round() as usize]
}

/// lo + i·step up to hi.
fn grid((lo, hi, step): (f64, f64, f64)) -> impl Iterator<Item = f64> + Clone {
    let n = ((hi - lo) / step + 1e-6).floor() as usize;
    (0..=n).map(move |i| lo + step * i as f64)
}

/// Least n of [lo, hi] where a monotone `holds` turns true.
fn least(holds: impl Fn(usize) -> bool, mut lo: usize, mut hi: usize) -> Option<usize> {
    if lo > hi || !holds(hi) {
        return None;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match holds(mid) {
            true => hi = mid,
            false => lo = mid + 1,
        }
    }
    Some(lo)
}

/// x minimizing Σ (a·x − y)².
fn lstsq<const M: usize>(rows: &[([f64; M], f64)]) -> [f64; M] {
    // AᵀA x = Aᵀy
    let (mut n, mut b) = ([[0.0; M]; M], [0.0; M]);
    for (a, y) in rows {
        for i in 0..M {
            b[i] += a[i] * y;
            for j in 0..M {
                n[i][j] += a[i] * a[j];
            }
        }
    }

    // Gauss Jordan with partial pivoting
    for c in 0..M {
        let p = (c..M)
            .max_by(|&i, &j| n[i][c].abs().total_cmp(&n[j][c].abs()))
            .unwrap();
        n.swap(c, p);
        b.swap(c, p);
        let (pivot, pb) = (n[c], b[c]);
        for r in (0..M).filter(|&r| r != c) {
            let f = n[r][c] / pivot[c];
            for j in 0..M {
                n[r][j] -= f * pivot[j];
            }
            b[r] -= f * pb;
        }
    }

    core::array::from_fn(|i| b[i] / n[i][i])
}

#[cfg(test)]
mod test {
    use super::*;

    // NOTE The tests have two parts:
    //
    // - Run a fresh calibration via `make_calibration`
    // - Test the `Calibration::CALIBRATED` via `calibration_is_accepted`

    /// (IQM |e|, median e, worst e) over rows with an error
    fn scale(e: &[f64]) -> [f64; 3] {
        let e: Vec<f64> = e.iter().copied().filter(|e| !e.is_nan()).collect();
        let mut abs: Vec<f64> = e.iter().map(|e| e.abs()).collect();
        abs.sort_by(f64::total_cmp);
        let q = abs.len() / 4;
        let mid = &abs[q..abs.len() - q];
        [
            mid.iter().sum::<f64>() / mid.len() as f64,
            quantile(e.clone(), 0.5),
            e.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ]
    }

    /// Plans off the calibration ladder deliver their PSL and sit under the calibration's own
    /// envelopes.  Each realized bin answers once, and the PSL envelope answers where it chose the
    /// length.  Margins must sit under limits contaminated fits cross.
    fn accept(cal: &Calibration) -> bool {
        // Target margins, the over delivery each envelope costs.
        const PSL_DB: f64 = 2.5;
        const FLOOR_DB: f64 = 6.0;
        const FOLD_DB: f64 = 3.0;
        const KNEE: f64 = 1.1;
        // Margins contaminated fits cross, well clear of the in-domain bands.
        const PSL_CEIL_DB: f64 = 5.0;
        const FLOOR_CEIL_DB: f64 = 12.0;
        const FOLD_CEIL_DB: f64 = 10.0;
        const KNEE_CEIL: f64 = 1.3;

        const QUANTA: [usize; 2] = [1, 4];
        const QS: [f64; 3] = [4.5, 8.5, 16.0];
        const RHOS: [f64; 4] = [0.08, 0.2, 0.25, 0.35];
        const PSLS: [f64; 4] = [-40.0, -60.0, -80.0, -100.0];
        /// `ENVELOPE` less slack for bins off the ladder.
        const ENVELOPED: f64 = 0.85;
        /// Width·Q a Q_real promise ignores.
        const WIDTH_SLACK: f64 = 2e-3;

        let m = cal.fit.margin;
        let mut accepted = true;

        println!("\n=== ACCEPTANCE (γ {}) ===", cal.settings.gamma);
        println!(
            "\n  fit measured over x {:.2} to {:.2}",
            cal.fit.span.0, cal.fit.span.1
        );
        println!("\n  margin is how far each envelope sits over its fitted relation");
        println!("  target is the margin we would like, limit is where the fit is contaminated\n");
        println!("  {:>6} {:>9} {:>9} {:>9}", "", "margin", "target", "limit");
        for (label, v, w, c, unit) in [
            ("psl", m.psl, PSL_DB, PSL_CEIL_DB, "dB"),
            ("floor", m.floor, FLOOR_DB, FLOOR_CEIL_DB, "dB"),
            ("fold", m.fold, FOLD_DB, FOLD_CEIL_DB, "dB"),
            ("knee", m.knee, KNEE, KNEE_CEIL, "×"),
        ] {
            let mark = match (v < w, v < c) {
                (true, _) => "",
                (false, true) => "over target",
                (false, false) => "FAIL over limit",
            };
            println!(
                "  {label:>6} {:>9} {:>9} {:>9}  {mark}",
                format!("{v:.2} {unit:<2}"),
                format!("{w:.2} {unit:<2}"),
                format!("{c:.2} {unit:<2}"),
            );
            accepted &= v < c;
        }

        // (label, unit, (measured − fit, envelope − measured, within the envelope))
        let mut ledgers = [
            ("psl", "dB", Vec::new()),
            ("floor", "dB", Vec::new()),
            ("fold", "dB", Vec::new()),
            ("width", "%", Vec::new()),
        ];
        let bare = Fit {
            margin: Margin::NONE,
            ..cal.fit
        };
        // (measured − fit, envelope − measured, envelope holds)
        let level = |seen: f64, fit: f64, env: f64| {
            resolved(seen).then(|| (seen - fit, env - seen, seen <= env))
        };
        let (mut kept, mut plans) = (0usize, 0usize);

        // (Q, ρ, half) of bins already held to their envelopes
        let mut held = Vec::new();
        // target − psl over plans whose PSL chose the length
        let mut waste = Vec::new();

        println!("\n  plans off the ladder by PSL target, levels in dB re gain");
        println!("  error is |measured − fit|, the fit's miss before any margin");
        println!("  by names what set the length, the requested PSL or the fit span edge\n");
        println!(
            "  {:>3} {:>5} {:>6} {:>6} {:>5} {:>5} {:>5} {:>8} {:>7} {:>8} {:>7} {:>7} {:>7}",
            "n",
            "Q",
            "rho",
            "target",
            "taps",
            "x",
            "by",
            "psl",
            "error",
            "floor",
            "error",
            "width·Q",
            "error %"
        );

        for quantum in QUANTA {
            for q in QS {
                for rho in RHOS {
                    for target in PSLS {
                        let Some(half) = cal.half_for_psl(q, rho, target, quantum) else {
                            println!(
                                "  {quantum:>3} {q:>5.1} {rho:>6.3} {target:>6.0}  unreachable"
                            );
                            continue;
                        };
                        let plan = cal.predict(q, rho, half);
                        let wav = cal
                            .settings
                            .wavelet(q)
                            .max_load_quantum(1)
                            .max_half_span(plan.u)
                            .bake();
                        let bin = wav.at_reach(rho, half);

                        // longer than the fit span edge
                        let bound = cal
                            .reach(q, rho, quantum)
                            .is_some_and(|(lo, _)| half > lo * quantum);
                        let key = (q.to_bits(), rho.to_bits(), half);
                        let fresh = !held.contains(&key);
                        if fresh {
                            held.push(key);
                        }

                        plans += 1;
                        let Some(seen) = observe(bin) else {
                            println!(
                                "  {quantum:>3} {q:>5.1} {rho:>6.3} {target:>6.0}  degenerate"
                            );
                            continue;
                        };
                        let fit = bare.predict(cal.settings.shape(q), rho, half);

                        println!(
                            "  {quantum:>3} {q:>5.1} {rho:>6.3} {target:>6.0} {:>5} {:>5.2} \
                             {:>5} {:>8.2} {:>7.2} {:>8.2} {:>7.2} {:>7.4} {:>7.2}",
                            seen.taps,
                            seen.x,
                            if bound { "psl" } else { "span" },
                            seen.psl,
                            (seen.psl - fit.psl).abs(),
                            seen.floor,
                            (seen.floor - fit.floor).abs(),
                            seen.width_q,
                            100.0 * (seen.width_q / fit.width_q - 1.0).abs(),
                        );

                        kept += (seen.psl <= target) as usize;
                        if bound && seen.psl <= target {
                            waste.push(target - seen.psl);
                        }

                        if fresh {
                            let rows = [
                                level(seen.psl, fit.psl, plan.psl).filter(|_| bound),
                                level(seen.floor, fit.floor, plan.floor),
                                level(seen.fold, fit.fold, plan.fold)
                                    .filter(|_| plan.fold > plan.psl),
                                // 100 (W_seen / W − 1), 100 (W_env − W_seen) / W_seen
                                Some((
                                    100.0 * (seen.width_q / fit.width_q - 1.0),
                                    100.0 * (plan.width_q - seen.width_q) / seen.width_q,
                                    seen.width_q - plan.width_q <= WIDTH_SLACK,
                                )),
                            ];
                            for ((_, _, errs), row) in ledgers.iter_mut().zip(rows) {
                                errs.extend(row);
                            }
                        }
                    }
                }
            }
        }

        println!(
            "\n  requested PSL delivered on {kept} of {plans} plans, {:.1}%",
            100.0 * kept as f64 / plans as f64
        );
        if !waste.is_empty() {
            println!(
                "  over delivery where the PSL set the length, median {:.1} dB, q90 {:.1} dB",
                quantile(waste.clone(), 0.5),
                quantile(waste, 0.9)
            );
        }
        accepted &= kept as f64 >= ENVELOPED * plans as f64;

        let rule = "=".repeat(90);
        println!("\n  {rule}");
        println!("  bias is the median of measured − fit, spread is how far bins land from the fit either way");
        println!("  headroom is envelope − measured, crossed is the share of bins where it went negative");
        println!("  {rule}");
        println!(
            "  {:>6} {:>5} {:>13} {:>13} {:>13} {:>8} {:>15}",
            "", "bins", "bias", "typical ±", "90% within ±", "crossed", "worst headroom"
        );
        for (label, unit, errs) in &ledgers {
            let abs: Vec<f64> = errs.iter().map(|v| v.0.abs()).collect();
            let crossed = errs.iter().filter(|v| !v.2).count() as f64 / errs.len() as f64;
            let worst = errs.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
            println!(
                "  {label:>6} {:>5} {:>13} {:>13} {:>13} {:>7.1}% {:>15}",
                errs.len(),
                format!(
                    "{:+.2} {unit:<2}",
                    quantile(errs.iter().map(|v| v.0).collect(), 0.5)
                ),
                format!("{:.2} {unit:<2}", scale(&abs)[0]),
                format!("{:.2} {unit:<2}", quantile(abs.clone(), 0.9)),
                100.0 * crossed,
                format!("{worst:+.2} {unit:<2}"),
            );
            accepted &= crossed <= 1.0 - ENVELOPED;
        }
        println!("  {rule}");

        accepted
    }

    /// Measure and fit the knee settings, print headroom by cell and by cut, and print the literal
    /// to paste as `Fit::CALIBRATED` once accepted.
    #[test]
    #[ignore]
    fn make_calibration() {
        /// x band edges from the knee's domain, shallow to deep.
        const BANDS: [f64; 6] = [1.5, 2.25, 3.0, 3.5, 4.0, 6.0];
        const UNITS: [&str; 4] = ["dB", "dB", "dB", "%"];

        let settings = Settings::knee();
        let obs = measure(settings, Sweep::default());
        let cal = Calibration {
            settings,
            fit: Fit::new(settings.gamma, &obs),
        };

        // (median, share under zero) of envelope − measured for psl, floor, fold in dB and width in % of the lobe
        let headroom = |rows: &[Obs]| -> [(f64, f64); 4] {
            let h: Vec<[f64; 4]> = rows
                .iter()
                .map(|o| {
                    let p = cal.predict(o.q, o.rho, o.taps / 2);
                    let level = |want: f64, seen: f64| match resolved(seen) {
                        true => want - seen,
                        false => f64::NAN,
                    };
                    [
                        level(p.psl, o.psl),
                        level(p.floor, o.floor),
                        match p.fold > o.psl {
                            true => level(p.fold, o.fold),
                            false => f64::NAN,
                        },
                        100.0 * (p.width_q - o.width_q) / o.width_q,
                    ]
                })
                .collect();
            core::array::from_fn(|i| {
                let v: Vec<f64> = h.iter().map(|h| h[i]).filter(|h| h.is_finite()).collect();
                let under = v.iter().filter(|&&h| h < 0.0).count() as f64 / v.len() as f64;
                (quantile(v, 0.5), under)
            })
        };
        let cols = |stats: [(f64, f64); 4]| {
            stats
                .into_iter()
                .zip(UNITS)
                .map(|((h, under), unit)| match h.is_finite() {
                    true => format!(
                        "{:>10} {:>7}",
                        format!("{h:+.2} {unit:<2}"),
                        format!("{:.0}%", 100.0 * under)
                    ),
                    false => format!("{:>10} {:>7}", "—", "—"),
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let head = format!(
            "{:>10} {:>7} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7}",
            "psl", "missed", "floor", "missed", "fold", "missed", "width", "missed"
        );

        println!(
            "\n=== CALIBRATION (γ {}, quantum 1, {} bins) ===",
            settings.gamma,
            obs.len()
        );
        println!("  headroom is envelope − measured, median per group, negative means the envelope was crossed");
        println!("  missed is the share of bins that crossed it");
        println!("  fold only where the main lobe clears the PSL at Nyquist, dash where no bin qualifies");
        println!("  levels under {QUANTIZED_DB:.0} dB sit on table rounding and are left out\n");
        println!("  {:>5} {:>6} {:>5} {head}", "Q", "rho", "bins");
        for cell in obs.chunk_by(|a, b| a.q == b.q && a.rho == b.rho) {
            println!(
                "  {:>5.1} {:>6.3} {:>5} {}",
                cell[0].q,
                cell[0].rho,
                cell.len(),
                cols(headroom(cell))
            );
        }

        println!("\n  {:>12} {:>8} {:>5} {head}", "x band", "tail dB", "bins");
        for w in BANDS.windows(2) {
            let rows: Vec<Obs> = obs
                .iter()
                .copied()
                .filter(|o| o.x >= w[0] && o.x < w[1])
                .collect();
            let tail = rows.iter().map(|o| o.tail).sum::<f64>() / rows.len() as f64;
            println!(
                "  {:>12} {tail:>8.1} {:>5} {}",
                format!("{:.2} to {:.2}", w[0], w[1]),
                rows.len(),
                cols(headroom(&rows))
            );
        }
        println!(
            "  {:>12} {:>8} {:>5} {}",
            "pooled",
            "",
            obs.len(),
            cols(headroom(&obs))
        );

        assert!(accept(&cal), "calibration not accepted, literal withheld");
        println!("\n  paste as Fit::CALIBRATED\n{:#?}", cal.fit);
    }

    /// The pasted calibration holds under the current settings.
    #[test]
    fn calibration_is_accepted() {
        assert!(
            accept(&Calibration::calibrated()),
            "calibration not accepted, see crossed and worst headroom per relation"
        );
    }
}
