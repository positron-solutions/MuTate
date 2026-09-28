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
        p_0: -11.666820569063718,
        p_slope: -0.2935918566464073,
        p_rho: -3.1763379447535125,
        p_lobe: -107.69384308936678,
        floor: Decay {
            base: 38.41209779698655,
            tail: 0.01848431251887577,
            reach: 2.6494544373459683,
        },
        w_knee: 2.05022891556142,
        w_n: -0.43711836348271854,
        w_m: 0.6776895798425826,
        span: (2.009355381156725, 5.498437260308141),
        margin: Margin {
            psl: 2.748700258153704,
            floor: 9.86477496637184,
            fold: 2.983412775955429,
            knee: 1.1323489524574923,
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

    /// Envelope at `shape` and `rho` with half span `u`, taps continuous.
    pub fn predict(&self, shape: Shape, rho: f64, u: f64) -> Prediction {
        let (q, p) = (shape.q(), shape.p());
        let (tail, x) = (shape.truncation_tail(u), TAU * u / p);
        let m = self.margin;

        let psl = tail * (1.0 + self.p_slope / p) + self.p_0 + rho * (self.p_rho + self.p_lobe / q);
        let floor = self.floor.at(psl, tail, q, rho, 1.0);

        Prediction {
            q,
            rho,
            u,
            x,
            tail,
            taps: 2 * (u / rho).ceil() as usize + 1,
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
            .map(|o| (o, self.predict(Shape::from_q(o.q, gamma), o.rho, o.u)))
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

/// Response a calibration promises.  Levels in dB re the crest, each an envelope.
#[derive(Clone, Copy, Debug)]
pub struct Prediction {
    pub q: f64,
    pub rho: f64,
    pub u: f64,
    pub x: f64,
    /// Feeds `with_truncation`.
    pub tail: f64,
    /// Unfolded at unit quantum.
    pub taps: usize,
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

    /// Envelope at `q` and `rho` with half span `u`, taps continuous.
    pub fn predict(&self, q: f64, rho: f64, u: f64) -> Prediction {
        self.fit.predict(self.settings.shape(q), rho, u)
    }

    /// Least half span whose floor envelope holds `floor`, absent past `DEEPEST_DB`.
    pub fn u_for_floor(&self, q: f64, rho: f64, floor: f64) -> Option<f64> {
        self.least_u(q, rho, floor, |p| p.floor)
    }

    /// Least half span whose PSL envelope holds `psl`, absent past `DEEPEST_DB`.
    pub fn u_for_psl(&self, q: f64, rho: f64, psl: f64) -> Option<f64> {
        self.least_u(q, rho, psl, |p| p.psl)
    }

    /// Least half span whose `level` envelope holds `level_db`.
    fn least_u(
        &self,
        q: f64,
        rho: f64,
        level_db: f64,
        level: fn(&Prediction) -> f64,
    ) -> Option<f64> {
        let goal = -level_db.abs();
        if goal < DEEPEST_DB {
            return None;
        }
        let (lo, hi) = self.reach(q, rho)?;
        least(|u| level(&self.predict(q, rho, u)) - goal, lo, hi)
    }

    /// Envelope of a realized bin baked under these settings.
    pub fn predict_bin(&self, bin: Bin<'_>) -> Prediction {
        let u = bin.rho() * (bin.len_folded() - 1) as f64;
        self.predict(bin.wavelet.shape.q(), bin.rho(), u)
    }

    /// Half spans measured by the fit where Q_real holds `Q_REAL_MIN`.
    fn reach(&self, q: f64, rho: f64) -> Option<(f64, f64)> {
        // u = x P / 2π
        let scale = self.settings.shape(q).p() / TAU;
        let (lo, hi) = (self.fit.span.0 * scale, self.fit.span.1 * scale);
        let lo = least(|u| Q_REAL_MIN - self.predict(q, rho, u).q_real(), lo, hi)?;
        Some((lo, hi))
    }

    /// Q at least `q` whose fold envelope holds `floor` at `rho`, and the half span holding it
    /// past the fold.
    pub fn plan(&self, q: f64, rho: f64, floor: f64) -> Option<Prediction> {
        // goal − fold margin
        let limit = -floor.abs() - self.fit.margin.fold;
        let fold = self.settings.shape(q).image_db(rho);
        // fold ∝ β ∝ Q²
        let q = match fold > limit {
            true => q * (limit / fold).sqrt(),
            false => q,
        };
        self.u_for_floor(q, rho, floor)
            .map(|u| self.predict(q, rho, u))
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
    let u = rho * (bin.len_folded() - 1) as f64;
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
                let half = (u(x) / rho).ceil() as usize;
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

/// Least point of [lo, hi] where a decreasing `f` reaches zero.
fn least(f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> Option<f64> {
    match (f(lo) <= 0.0, f(hi) <= 0.0) {
        (true, _) => Some(lo),
        (false, true) => Some(bisect(f, lo, hi)),
        (false, false) => None,
    }
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
    use super::super::harness::{at, Ledger};
    use super::*;

    // NOTE The tests have two parts:
    //
    // - Run a fresh calibration via `make_calibration`
    // - Test the `Calibration::CALIBRATED` via

    /// Plans off the calibration ladder deliver their floors and sit under the calibration's own
    /// envelopes.  Margins must sit under ceilings contaminated fits cross.  Wishes and over
    /// delivery are reported, not judged.
    fn accept(cal: &Calibration) -> bool {
        // Wished margins, the over delivery each envelope costs.
        const PSL_DB: f64 = 2.5;
        const FLOOR_DB: f64 = 6.0;
        const FOLD_DB: f64 = 3.0;
        const KNEE: f64 = 1.1;
        // Margins contaminated fits cross, well clear of the in-domain bands.
        const PSL_CEIL_DB: f64 = 5.0;
        const FLOOR_CEIL_DB: f64 = 12.0;
        const FOLD_CEIL_DB: f64 = 10.0;
        const KNEE_CEIL: f64 = 1.3;

        const QS: [f64; 3] = [4.5, 8.5, 16.0];
        const RHOS: [f64; 4] = [0.08, 0.2, 0.25, 0.35];
        const FLOORS: [f64; 4] = [-40.0, -60.0, -90.0, -120.0];
        /// `ENVELOPE` less slack for bins off the ladder.
        const ENVELOPED: f64 = 0.85;
        /// Center misses a pitch estimate ignores.
        const CENTER_C: f64 = 0.5;
        /// Width·Q a Q_real promise ignores.
        const WIDTH_SLACK: f64 = 2e-3;
        const WORST: usize = 3;

        let m = cal.fit.margin;
        let mut held = true;

        println!("\n=== ACCEPTANCE (γ {}) ===", cal.settings.gamma);
        println!("\n  span x {:.3} to {:.3}", cal.fit.span.0, cal.fit.span.1);
        println!(
            "\n  {:>9} {:>9} {:>9} {:>9}",
            "margin", "fit", "wish", "ceiling"
        );
        for (label, v, w, c) in [
            ("psl dB", m.psl, PSL_DB, PSL_CEIL_DB),
            ("floor dB", m.floor, FLOOR_DB, FLOOR_CEIL_DB),
            ("fold dB", m.fold, FOLD_DB, FOLD_CEIL_DB),
            ("knee", m.knee, KNEE, KNEE_CEIL),
        ] {
            let mark = match (v < w, v < c) {
                (true, _) => "",
                (false, true) => "short",
                (false, false) => "FAIL",
            };
            println!("  {label:>9} {v:>9.4} {w:>9.4} {c:>9.4} {mark}");
            held &= v < c;
        }

        let mut ledgers = [
            ("delivered", ENVELOPED, Ledger::default()),
            ("floor", ENVELOPED, Ledger::default()),
            ("psl", ENVELOPED, Ledger::default()),
            ("fold", ENVELOPED, Ledger::default()),
            ("width", ENVELOPED, Ledger::default()),
            ("center", 1.0, Ledger::default()),
            ("q_real", 1.0, Ledger::default()),
        ];
        // target − floor over delivered plans
        let mut waste = Vec::new();
        // 10^((seen − want)/20), holding where the seen level is absent or on table rounding
        let over = |seen: f64, want: f64| match resolved(seen) {
            true => 10f64.powf((seen - want) / 20.0),
            false => 0.0,
        };

        println!("\n  plans off the ladder, seen / envelope, dB re gain and cents\n");
        println!(
            "  {:>5} {:>6} {:>6} {:>6} {:>5} {:>6} {:>17} {:>17} {:>17} {:>15} {:>7}",
            "Q",
            "rho",
            "target",
            "plan Q",
            "taps",
            "x",
            "floor",
            "psl",
            "fold",
            "width·Q",
            "center"
        );

        for q in QS {
            for rho in RHOS {
                for target in FLOORS {
                    let Some(plan) = cal.plan(q, rho, target) else {
                        println!("  {q:>5.1} {rho:>6.3} {target:>6.0}  unreachable");
                        continue;
                    };
                    let wav = cal
                        .settings
                        .wavelet(plan.q)
                        .max_load_quantum(1)
                        .max_half_span(plan.u)
                        .bake();
                    let bin = wav.at_reach(rho, (plan.u / rho).ceil() as usize);
                    let at = at!(q, rho, target);

                    let Some(seen) = observe(bin) else {
                        println!("  {q:>5.1} {rho:>6.3} {target:>6.0}  degenerate");
                        let (_, _, q_real) = ledgers.last_mut().unwrap();
                        q_real.record(1.0, f64::INFINITY, at);
                        continue;
                    };
                    let want = cal.predict_bin(bin);

                    println!(
                        "  {q:>5.1} {rho:>6.3} {target:>6.0} {:>6.2} {:>5} {:>6.2} \
                         {:>8.2}/{:<8.2} {:>8.2}/{:<8.2} {:>8.2}/{:<8.2} {:>7.4}/{:<7.4} {:>7.3}",
                        plan.q,
                        seen.taps,
                        seen.x,
                        seen.floor,
                        want.floor,
                        seen.psl,
                        want.psl,
                        seen.fold,
                        want.fold,
                        seen.width_q,
                        want.width_q,
                        seen.center,
                    );

                    if seen.floor <= target {
                        waste.push(target - seen.floor);
                    }
                    let ratios = [
                        over(seen.floor, target),
                        over(seen.floor, want.floor),
                        over(seen.psl, want.psl),
                        // fold envelope where the main lobe clears the PSL at Nyquist
                        match want.fold > want.psl {
                            true => over(seen.fold, want.fold),
                            false => 0.0,
                        },
                        // (seen − want) / slack
                        (seen.width_q - want.width_q) / WIDTH_SLACK,
                        seen.center.abs() / CENTER_C,
                        // Q_REAL_MIN / Q_real
                        Q_REAL_MIN * seen.width_q / seen.q,
                    ];
                    for ((_, _, ledger), r) in ledgers.iter_mut().zip(ratios) {
                        ledger.record(1.0, r, at.clone());
                    }
                }
            }
        }

        if !waste.is_empty() {
            println!(
                "\n  over delivery, median {:.1} dB, q90 {:.1} dB",
                quantile(waste.clone(), 0.5),
                quantile(waste, 0.9)
            );
        }

        for (label, share, mut ledger) in ledgers {
            println!("\n  {label}  good {:.4}", ledger.good());
            ledger.print_worst(WORST);
            held &= ledger.good() >= share;
        }
        held
    }

    /// Measure and fit the knee settings, print headroom by cell and by cut, and print the literal
    /// to paste as `Fit::CALIBRATED` once accepted.
    #[test]
    #[ignore]
    fn make_calibration() {
        /// x band edges from the knee's domain, shallow to deep.
        const BANDS: [f64; 6] = [1.5, 2.25, 3.0, 3.5, 4.0, 6.0];

        let settings = Settings::knee();
        let obs = measure(settings, Sweep::default());
        let cal = Calibration {
            settings,
            fit: Fit::new(settings.gamma, &obs),
        };

        // (median, share under zero) of envelope − seen for psl, floor, fold, width·Q
        let headroom = |rows: &[Obs]| -> [(f64, f64); 4] {
            let h: Vec<[f64; 4]> = rows
                .iter()
                .map(|o| {
                    let p = cal.predict(o.q, o.rho, o.u);
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
                        p.width_q - o.width_q,
                    ]
                })
                .collect();
            core::array::from_fn(|i| {
                let v: Vec<f64> = h.iter().map(|h| h[i]).filter(|h| h.is_finite()).collect();
                let under = v.iter().filter(|&&h| h < 0.0).count() as f64 / v.len() as f64;
                (quantile(v, 0.5), under)
            })
        };
        // max |center|
        let center = |rows: &[Obs]| rows.iter().map(|o| o.center.abs()).fold(0.0, f64::max);
        let cols = |[p, f, d, w]: [(f64, f64); 4], c: f64| {
            format!(
                "{:>+7.2} {:>6.1}% {:>+8.2} {:>6.1}% {:>+8.2} {:>6.1}% {:>+8.4} {:>6.1}% {c:>6.3}",
                p.0,
                100.0 * p.1,
                f.0,
                100.0 * f.1,
                d.0,
                100.0 * d.1,
                w.0,
                100.0 * w.1,
            )
        };
        let head = format!(
            "{:>7} {:>7} {:>8} {:>7} {:>8} {:>7} {:>8} {:>7} {:>6}",
            "psl h", "under", "floor h", "under", "fold h", "under", "width h", "under", "|c|"
        );

        println!(
            "\n=== CALIBRATION (γ {}, quantum 1, {} bins) ===",
            settings.gamma,
            obs.len()
        );
        println!("  headroom h is median envelope − seen, under the share delivered worse");
        println!("  psl, floor, fold in dB, width in width·Q, |c| the worst center in cents");
        println!("  fold only where its envelope clears the PSL, dashes where no bin does");
        println!("  levels under {QUANTIZED_DB:.0} dB sit on table rounding and are left out\n");
        println!("  {:>5} {:>6} {:>5} {head}", "Q", "rho", "bins");
        for cell in obs.chunk_by(|a, b| a.q == b.q && a.rho == b.rho) {
            println!(
                "  {:>5.1} {:>6.3} {:>5} {}",
                cell[0].q,
                cell[0].rho,
                cell.len(),
                cols(headroom(cell), center(cell))
            );
        }

        println!("\n  {:>11} {:>7} {:>5} {head}", "x band", "tail", "bins");
        for w in BANDS.windows(2) {
            let rows: Vec<Obs> = obs
                .iter()
                .copied()
                .filter(|o| o.x >= w[0] && o.x < w[1])
                .collect();
            let tail = rows.iter().map(|o| o.tail).sum::<f64>() / rows.len() as f64;
            println!(
                "  {:>11} {tail:>7.1} {:>5} {}",
                format!("{:.2} to {:.2}", w[0], w[1]),
                rows.len(),
                cols(headroom(&rows), center(&rows))
            );
        }
        println!(
            "  {:>11} {:>7} {:>5} {}",
            "pooled",
            "",
            obs.len(),
            cols(headroom(&obs), center(&obs))
        );

        assert!(accept(&cal), "calibration not accepted, literal withheld");
        println!("\n  paste as Fit::CALIBRATED\n{:#?}", cal.fit);
    }

    /// The pasted calibration holds under the current settings.
    #[test]
    fn calibration_is_accepted() {
        assert!(
            accept(&Calibration::calibrated()),
            "calibration not accepted, worst listed per ledger"
        );
    }
}
