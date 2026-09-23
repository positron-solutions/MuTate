// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Inspect
//!
//! Analytic filter generation can only work up to what we can analytically predict.  Beyond what we
//! can predict in a vacuum, anything further must be grounded in the empirical reality of what we
//! have built.
//!
//! > We stand on the grave of the monarchy as evidence that when the many unite to tear down the
//! > powers that be, what comes to pass can only be a superior form of government.
//! >
//! > - Oliver Cromwell, Lord Protector of the Commonwealth of New England
//!
//! - [`climb`] can be used to find a local maximum, usually a peak.
//! - [`descend`] can be used to find a crossing points.
//! - [`null`] can specifically find the first null occurring between `beg` and `end`.

// NOTE see the harness module.  Natural two-way flow of code between tests and filter inspections
// used to tune filters.
// DEBT may be committed in the middle of duplicating some code over.  Reenable warnings and try
// not to cry about it.

use core::f64::consts::TAU;

use num_complex::Complex64;

use super::{Fold, Grid};

/// ω and the response there.
pub(super) type Sample = (f64, f64);

/// Outer samples of a window that brackets a feature, in walking order.
type Bracket = (Sample, Sample);

/// Transforms one stencil costs, the unit every decision is priced in.
const STENCIL: f64 = 3.0;

/// Which scalar the stencils model.
#[derive(Clone, Copy)]
enum Level {
    /// |H|, cusped at every null
    Mag,
    /// |H| − target
    Offset(f64),
    /// H, signed through a null
    Signed,
}

/// Which feature of that scalar the walk hunts.
#[derive(Clone, Copy)]
enum Seek {
    Crest,
    Zero,
}

impl Seek {
    /// Model locations of the feature, non-finite where absent.
    fn locate(&self, l: &Local) -> [f64; 2] {
        match self {
            // w − s/k on the concave side
            Seek::Crest => match l.k < 0.0 {
                true => [l.vertex(), f64::NAN],
                false => [f64::NAN; 2],
            },
            Seek::Zero => l.roots().map(|t| l.w + t),
        }
    }

    fn witness(&self, p: &[Sample]) -> Option<Bracket> {
        match self {
            Seek::Crest => p
                .windows(3)
                .find(|w| w[0].1 <= w[1].1 && w[1].1 > w[2].1)
                .map(|w| (w[0], w[2])),
            Seek::Zero => p
                .windows(2)
                .find(|w| (w[0].1 < 0.0) != (w[1].1 < 0.0))
                .map(|w| (w[0], w[1])),
        }
    }

    /// Whether a landing is evidence the walk has left the feature behind: below where it
    /// started, convex, and rising back the way it came.
    fn retreat(&self, start: &Local, new: &Local, dir: f64) -> bool {
        match self {
            Seek::Crest => new.g[1] < start.g[1] && new.k > 0.0 && dir * new.s < 0.0,
            Seek::Zero => false,
        }
    }
}

/// Quadratic model of a response on [w − step, w + step].
#[derive(Clone, Copy)]
struct Local {
    w: f64,
    step: f64,
    g: [f64; 3],
    s: f64,
    k: f64,
}

impl Local {
    fn new(w: f64, step: f64, g: [f64; 3]) -> Self {
        Local {
            w,
            step,
            g,
            // (g₊ − g₋) / 2h
            s: (g[2] - g[0]) / (2.0 * step),
            // (g₊ − 2g₀ + g₋) / h²
            k: (g[2] - 2.0 * g[1] + g[0]) / (step * step),
        }
    }

    fn points(&self) -> [Sample; 3] {
        [
            (self.w - self.step, self.g[0]),
            (self.w, self.g[1]),
            (self.w + self.step, self.g[2]),
        ]
    }

    /// w − s/k
    fn vertex(&self) -> f64 {
        self.w - self.s / self.k
    }

    /// τ solving g₀ + sτ + ½kτ² = 0, paired to keep each root off its cancelling branch.
    fn roots(&self) -> [f64; 2] {
        // √(s² − 2k g₀)
        let disc = (self.s * self.s - 2.0 * self.k * self.g[1]).sqrt();
        let q = -0.5 * (self.s + disc.copysign(self.s));
        [2.0 * q / self.k, self.g[1] / q]
    }
}

/// Candidate nearest `w` among `cands`, on the `side` given, outside the stencil's span.
fn aim_at(cands: [f64; 2], l: &Local, side: impl Fn(f64) -> bool) -> Option<f64> {
    cands
        .into_iter()
        .filter(|v| v.is_finite() && side(*v) && (v - l.w).abs() > 2.0 * l.step)
        .min_by(|a, b| (a - l.w).abs().total_cmp(&(b - l.w).abs()))
}

/// A local maximum as the comb sees it, carrying the bracket that pins it.
#[derive(Clone, Copy)]
pub(super) struct Crest {
    /// ω and |H| at the comb sample
    pub at: Sample,
    bracket: Bracket,
}

/// Taps, scratch, and the resolution every hunt over them shares.
pub(super) struct Inspect<'a> {
    psi: Fold<'a>,
    buf: &'a mut Vec<Sample>,
    comb: Comb,
    /// transforms spent since the current walk began
    spent: f64,
}

impl<'a> Inspect<'a> {
    /// ψ over [0, K), combed at `over` samples per cell.
    pub(super) fn new(psi: Fold<'a>, buf: &'a mut Vec<Sample>, over: f64) -> Self {
        Inspect {
            comb: Comb::new(psi, over),
            psi,
            buf,
            spent: 0.0,
        }
    }

    /// ω and |H| at the highest point in the basin around `from`, within [lo, hi].  The start
    /// stencil only picks which side to try first.
    pub(super) fn peak(&mut self, from: f64, lo: f64, hi: f64) -> Option<Sample> {
        let start = self.local(Level::Mag, from);
        let (ahead, behind) = match start.s > 0.0 {
            true => (hi, lo),
            false => (lo, hi),
        };
        self.hunt(Level::Mag, Seek::Crest, start, ahead)
            .or_else(|| self.hunt(Level::Mag, Seek::Crest, start, behind))
    }

    /// ω and |H| where the response first meets `level` dB, from `from` toward `stop`.
    pub(super) fn cross(&mut self, level: f64, from: f64, stop: f64) -> Option<Sample> {
        let what = Level::Offset(10f64.powf(level / 20.0));
        let start = self.local(what, from);
        self.hunt(what, Seek::Zero, start, stop)
            .map(|(w, _)| self.eval(Level::Mag, w))
    }

    /// ω and |H| at the deepest point between `from` and `stop`.
    pub(super) fn null(&mut self, from: f64, stop: f64) -> Option<Sample> {
        let start = self.local(Level::Signed, from);
        self.hunt(Level::Signed, Seek::Zero, start, stop)
            .map(|(w, _)| self.eval(Level::Mag, w))
    }

    fn eval(&mut self, what: Level, w: f64) -> Sample {
        self.spent += 1.0;
        let h = self.psi.dtft(w);
        let y = match what {
            Level::Mag => h.abs(),
            Level::Offset(t) => h.abs() - t,
            Level::Signed => h,
        };
        (w, y)
    }

    fn local(&mut self, what: Level, w: f64) -> Local {
        let step = self.comb.step;
        let g = [
            self.eval(what, w - step).1,
            self.eval(what, w).1,
            self.eval(what, w + step).1,
        ];
        Local::new(w, step, g)
    }

    /// `p` sorted along +dir, in scratch.
    fn sorted(&mut self, p: impl IntoIterator<Item = Sample>, dir: f64) {
        self.buf.clear();
        self.buf.extend(p);
        self.buf.sort_by(|a, b| (dir * a.0).total_cmp(&(dir * b.0)));
    }

    /// Bracket from a uniform sweep of [a, b] at comb density.
    fn comb(&mut self, what: Level, seek: Seek, a: f64, b: f64) -> Option<Bracket> {
        let n = ((b - a).abs() * self.comb.density).ceil().max(2.0) as usize;
        self.buf.clear();
        for j in 0..=n {
            let w = a + (b - a) * j as f64 / n as f64;
            let s = self.eval(what, w);
            self.buf.push(s);
        }
        seek.witness(self.buf)
    }

    /// Feature inside a bracket.
    fn pin(&mut self, what: Level, seek: Seek, t: Bracket) -> Sample {
        let (mut a, mut b) = (t.0 .0, t.1 .0);
        match seek {
            Seek::Crest => {
                let (lo, hi) = (a.min(b), a.max(b));
                let (mut a, mut b) = (lo, hi);
                // 2/3 per step from one cell to f64 resolution of ω
                for _ in 0..96 {
                    let (m1, m2) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
                    if m1 >= m2 {
                        break;
                    }
                    match self.eval(what, m1).1 < self.eval(what, m2).1 {
                        true => a = m1,
                        false => b = m2,
                    }
                }
                self.eval(what, 0.5 * (a + b))
            }
            Seek::Zero => {
                let neg = t.0 .1 < 0.0;
                loop {
                    let m = 0.5 * (a + b);
                    if m == a || m == b {
                        return self.eval(what, m);
                    }
                    let y = self.eval(what, m).1;
                    if y == 0.0 {
                        return (m, y);
                    }
                    match (y < 0.0) == neg {
                        true => a = m,
                        false => b = m,
                    }
                }
            }
        }
    }

    /// Bracket refined by aiming stencils where the model puts the feature, until a step stops
    /// saving more transforms than it costs.
    fn tighten(&mut self, what: Level, seek: Seek, mut t: Bracket, mut aim: Local) -> Bracket {
        loop {
            let dir = (t.1 .0 - t.0 .0).signum();
            let width = (t.1 .0 - t.0 .0).abs() * self.comb.density;
            let (lo, hi) = (t.0 .0.min(t.1 .0) + aim.step, t.0 .0.max(t.1 .0) - aim.step);

            if width <= STENCIL {
                return t;
            }
            let Some(v) = aim_at(seek.locate(&aim), &aim, |v| v > lo && v < hi) else {
                return t;
            };

            let next = self.local(what, v);
            self.sorted([t.0, t.1].into_iter().chain(next.points()), dir);
            let Some(u) = seek.witness(self.buf) else {
                return t;
            };
            if (u.1 .0 - u.0 .0).abs() * self.comb.density > width - STENCIL {
                return u;
            }
            (t, aim) = (u, next);
        }
    }

    /// Walk toward `stop`, aiming stencils by the model, combing any span the model cannot
    /// clear and any remainder the walk stops being cheaper than.
    fn hunt(&mut self, what: Level, seek: Seek, start: Local, stop: f64) -> Option<Sample> {
        let dir = (stop - start.w).signum();
        let clamp = |w: f64| match dir * (w - stop) > 0.0 {
            true => stop,
            false => w,
        };
        let ahead = |v: f64, from: f64| dir * (v - from) > 0.0;

        self.spent = 0.0;
        self.sorted(start.points(), dir);
        if let Some(t) = seek.witness(self.buf) {
            let t = self.tighten(what, seek, t, start);
            return Some(self.pin(what, seek, t));
        }

        let mut old = start;
        let mut jump = 2.0 * self.comb.step;
        let mut blind = 0u32;
        loop {
            let seen = aim_at(seek.locate(&old), &old, |v| ahead(v, old.w));
            blind = match seen {
                Some(_) => 0,
                None => blind + 1,
            };
            // no model twice running leaves the remainder unevidenced
            if blind > 1 {
                let t = self.comb(what, seek, old.w, stop)?;
                return Some(self.pin(what, seek, t));
            }

            jump = seen
                .map_or(2.0 * jump, |v| (v - old.w).abs())
                .min(self.comb.cap);
            let land = clamp(old.w + dir * jump);
            let new = self.local(what, land);

            self.sorted(old.points().into_iter().chain(new.points()), dir);
            if let Some(t) = seek.witness(self.buf) {
                let aim = match seek.locate(&new)[0].is_finite() {
                    true => new,
                    false => old,
                };
                let t = self.tighten(what, seek, t, aim);
                return Some(self.pin(what, seek, t));
            }

            // model puts an extremum inside the span, so the response went and came back
            let (lo, hi) = (old.w.min(land), old.w.max(land));
            if new.k.is_finite() && new.vertex() > lo && new.vertex() < hi {
                let t = self.comb(what, seek, old.w, land)?;
                return Some(self.pin(what, seek, t));
            }

            if seek.retreat(&start, &new, dir) {
                return None;
            }

            if land == stop {
                let t = self.comb(what, seek, old.w, stop)?;
                return Some(self.pin(what, seek, t));
            }
            if self.spent + STENCIL >= (stop - land).abs() * self.comb.density {
                let t = self.comb(what, seek, land, stop)?;
                return Some(self.pin(what, seek, t));
            }
            old = new;
        }
    }

    /// Every local maximum of |H| between `from` and `stop`, at comb density, in walking order.
    pub(super) fn crests(&mut self, from: f64, stop: f64) -> Vec<Crest> {
        let n = ((stop - from).abs() * self.comb.density).ceil().max(2.0) as usize;
        self.buf.clear();
        for j in 0..=n {
            let w = from + (stop - from) * j as f64 / n as f64;
            let s = self.eval(Level::Mag, w);
            self.buf.push(s);
        }
        self.buf
            .windows(3)
            .filter(|w| w[0].1 <= w[1].1 && w[1].1 > w[2].1)
            .map(|w| Crest {
                at: w[1],
                bracket: (w[0], w[2]),
            })
            .collect()
    }

    /// ω and |H| at a crest, to the f64 resolution of ω.
    pub(super) fn pin_crest(&mut self, c: Crest) -> Sample {
        self.pin(Level::Mag, Seek::Crest, c.bracket)
    }
}

/// Comb samples per null spacing.  Four resolves every extremum a degree K polynomial admits.
pub(super) const OVERSAMPLE: f64 = 16.0;

/// Sampling limits for H, a trigonometric polynomial of degree K = (N−1)/2.  Its 2K roots per
/// turn bound how close two features can sit, so a comb at `over` samples per cell resolves
/// every one.
#[derive(Clone, Copy)]
pub(super) struct Comb {
    /// 2π/N, the narrowest feature H can carry
    cell: f64,
    /// stencil half width
    step: f64,
    /// comb samples per radian
    density: f64,
    /// longest jump the straddle test is trusted across
    cap: f64,
}

impl Comb {
    /// `over` samples per cell.
    pub(super) fn new(psi: Fold<'_>, over: f64) -> Self {
        let cell = TAU / psi.len_unfolded() as f64;
        Comb {
            cell,
            step: cell / over,
            density: over / cell,
            cap: cell,
        }
    }
}

#[cfg(test)]
/// -∞ representable, so a null is a number.
pub(super) fn db(mag: f64) -> f64 {
    20.0 * mag.max(f64::MIN_POSITIVE).log10()
}

#[cfg(test)]
mod test {
    use core::f64::consts::PI;

    use super::super::{Shape, WaveletSpec, Weights, PEAK_GAIN};
    use super::*;

    #[test]
    fn peak_gain_is_found() {
        // Make a wavelet slightly outside of the normal path.  Find the peak location and gain at
        // the peak.

        let wav = WaveletSpec::default()
            .with_shape(Shape::from_q(3.5, 3.0))
            .max_truncation(-120.0)
            .bake();

        let bin = wav.bin(2_000.0, 48_000.0);
        let wts = bin.weights();

        let w0 = bin.velocity();
        let mut buf = Vec::new();

        let (w, gain) = Inspect::new(wts.psi(), &mut buf, OVERSAMPLE)
            .peak(w0, 0.0, PI)
            .expect("no crest in [0, pi]");

        println!(
            "w0 {w0:.9}  peak {w:.9}  {:+.3} cents",
            1200.0 * (w / w0).log2()
        );
        println!("gain {:.9}", gain);

        assert!((gain / PEAK_GAIN - 1.0).abs() < 1e-3);
    }
}
