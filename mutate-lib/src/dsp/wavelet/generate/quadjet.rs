// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Log-Tilted Saddle Jet + Terminal Quadrature
//!
//! > Engineers don't know anything but physicists can't do anything.
//! >
//! > - Nigel Clayton
//!
//! This is the production method.  It uses the log tilt to avoid endpoint terms at all.  Heavy
//! reliance on saddle terms is sufficient for many `u` and avoids expensive quadrature.  Quadrature
//! is used wherever the saddle terms struggle to do the job or to do it accurately.  That's a lot
//! of sophistication.  The simpler IFFT and Contour methods are provided as evidence of this
//! implementation's accuracy and precision.
//!
//! `∫₀^∞ ω^β e^{-ω^γ} e^{iωt} dω` by steepest descent in `z = ln(ω/ρ)`.  Working in the log
//! coordinate carries the branch point at the origin out to infinity, so the saddle condition
//! becomes the trinomial `u^γ - iτu - 1 = 0` and every derivative of the phase comes out affine
//! in the single number `B = u^γ - 1`.
//!
//! The original contour decomposes into some subset of those saddles, and which subset is a
//! matter of membership rather than height.  At `τ = 0` the roots are the roots of unity and
//! only one of them belongs.  Membership changes as `τ` grows only where two saddles exchange
//! dominance, so it has to be marched from that seed rather than evaluated at a point, and the
//! march happens once.  A tap is then a pure function of `t` and the caller owes us no grid
//! continuity.
//!
//! Every saddle that survives is approached two ways.  A Watson jet expands around it and costs
//! almost nothing, and a trapezoid traced along its steepest-descent path costs a great deal
//! more.  The jet runs everywhere and reports what it could not resolve, and the path is spent
//! only where that report says the series cannot reach the accuracy the caller asked for.
//!
//! `γ ≥ 3` is the supported range.  At `γ = 2` two saddles collide on the real `τ` axis at
//! `τ = 2` and `Φ''` vanishes with them, taking the normal coordinate the whole method is built
//! on with it.

// 🤖 This module exemplifies a principled approach to wrapping engineering around heuristically
// generated code.  The approach is similar to supplier quality assurance.  Our shop might not build
// widgets, but we can test that a widget behaves as expected, generating a wavelet to a degree of
// accuracy that cannot be expected of a degenerate, malformed widget.
//
// First to verify that the approach had any hope at all, a reference implementation of the
// quadrature, the deformed contour, was created.  It has long agreed with a well understood
// approach, the IFFT.  Where the IFFT approach has self-converging precision (and the lucky
// butterflies removed from this consideration), the deformed contour approach appears equally
// precise.
//
// The deformed contour was selected via an analytic design canvas, a survey of well-formalized
// approaches.  Such a survey does not intend to understand each option up to the point of
// fine-tuning a concrete implementation.  Instead, it only aims to understand their qualities and
// to identify *evidence* that the approach has some advantage over others.  The output is an
// ensemble of design choices that are believed likely to work well together.  This is one phase of
// risk management among many.
//
// The same process identified important design considerations, such as the rotated log coordinates,
// as having desirable properties.  Several simpler but similar implementations were built up and
// checked against the IFFT and contour methods.  The testing methodology was built up.  The proper
// signals were identified and built up.
//
// The degree of method-independent corroboration can be made so overwhelming that any other
// evidence leaning solely on a prior is made vanishingly small.  That doesn't constitute a formal
// proof.  Code usually does not (outside of safety critical applications with explicit formal
// verification within the toolchain).  What the corroboration becomes is a sufficient degree of
// demonstrated reliability to be competitive with any other option.
//
// The IFFT approach, the COTS standard, the low-risk entry into the race, is not actually a low
// risk approach without the same corroborating evaluation process.  It was the lack of agreement
// with the other methods that identified the IFFT's need for padding and how much.  A low risk tool
// used too eagerly without engineering acceptance testing is not a low risk tool.  Advocating use
// of such libraries and endlessly encouraging authors to make them infallible with the goal of
// turning one's own brain off while using them is a familiar kind of laziness we should discourage.
//
// What remains worthy of emphasis is the importance of encouraging open discussion such of methods.
// Code such as this is welcome in this project.  It should be welcome in more projects, and it
// should not be discussed dishonestly and lazily from the shaky foundations of repeatedly refuted,
// outdated, and increasingly belying priors.
//
// Formally defined methods and the tools created from them do not often kind-of succeed.
// Heuristics approximating implementation of formal methods are strongly attracted to grooves of
// consistency.  The coherence goes up.  The attracting manifold attracts.  Upon intersection with a
// correct approach, at the intersection of a correct implementation, the result ceases to be a
// heuristic output and has crossed into being an algorithm, a formally consistent approach that
// always obtains a correct result.  If a descent can be coaxed along a line that will fall into a
// groove, a heuristic an be made to behave in a deterministic manner, and to pretend otherwise is
// perhaps an instance of motivated failure to comprehend.

use num_complex::Complex64;
use std::f64::consts::TAU;

use super::super::spec::Shape;
use super::super::whatsleft::Accumulator;

const EPS: f64 = f64::EPSILON;

// Stokes table
const DK_ITERS: usize = 36;
const DK_TOL: f64 = 1e-14;
const TABLE_NEWTON_TOL: f64 = 1e-14;
const TABLE_NEWTON_ITERS: usize = 32;
const TABLE_NODES: usize = 2048;
const TABLE_QUIET_SPAN: f64 = 4.0;
const TABLE_TAU_MAX: f64 = 256.0;

// Saddles & Jets
const ADJ_CONE: f64 = 0.9;
const JET_ORDER: usize = 96;
const JET_SLOTS: usize = JET_ORDER + 2;
const JET_MIN_ORDER: usize = 6;
const LIVE: f64 = 0.5;
const MAX_G: usize = 8;

// Quadrature
const GAP_FLOOR: f64 = 1e-6;
const LIFT_CREDIT: f64 = 0.5;
const NEWTON_ITERS: usize = 6;
const NEWTON_TOL: f64 = 1e-12;
const QUAD_MARGIN: f64 = 1.2;
const QUAD_MAX_DENSITY: f64 = QUAD_MAX_NODES as f64 / (2.0 * REACH_MAX);
const QUAD_MAX_NODES: usize = 512;
const QUAD_MIN_DENSITY: f64 = 0.1;
const QUAD_SLOTS: usize = QUAD_MAX_NODES + 2;
const REACH_MAX: f64 = 12.0;
const TRACE_MAX_SPLIT: u32 = 6;

/// The shape, together with the forms of it the rest of the file actually reads.  `ρ` is where
/// the saddle sits at `τ = 0` and sets the scale `ω` is measured in, and the two integer casts
/// of `γ` are what the loop bounds and `powi` want.
#[derive(Clone, Copy)]
struct Frame {
    beta: f64,
    gamma: f64,
    g: usize,
    gi: i32,
    rho: f64,
}

impl Frame {
    fn new(shape: Shape) -> Self {
        let Shape { beta, gamma } = shape;
        Self {
            beta,
            gamma,
            g: gamma as usize,
            gi: gamma as i32,
            rho: (beta / gamma).powf(1.0 / gamma),
        }
    }
}

pub struct QuadJetResult {
    pub psi: Complex64,
    /// dψ/dt, from the same roots and the same paths.
    pub d: Complex64,
    /// What each saddle's own method could say about its own error, summed over the
    /// decomposition, in the units of the channel it describes.
    pub residual: [f64; 2],
    /// Approximate spend for this tap.
    pub cost: Cost,
}

/// Crude spend, for development signal only.  Every column sums over the call tree of one tap,
/// so the numbers are read against themselves across a change rather than against each other.
#[derive(Clone, Copy, Default)]
pub struct Cost {
    /// `jet_pass` invocations.  More than one per saddle means the predicted order fell short.
    pub jet_passes: u32,
    /// Summed order actually reached.
    pub jet_orders: u32,
    /// `trapezoid` invocations.
    pub quad_paths: u32,
    /// Nodes a `walk` was actually spent on; a refinement reuses its even neighbors.
    pub quad_nodes: u32,
    /// How far the traced path moved the jet's answer, measured against the bar that asked for
    /// it.  Under one, the series had already arrived wherever a path was spent.
    pub quad_gain: f64,
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        self.jet_passes += rhs.jet_passes;
        self.jet_orders += rhs.jet_orders;
        self.quad_paths += rhs.quad_paths;
        self.quad_nodes += rhs.quad_nodes;
        self.quad_gain = self.quad_gain.max(rhs.quad_gain);
    }
}

/// A jet at one order, with whether the series ended on the mathematics or on the max order.
struct Pass {
    terms: [Term; 2],
    raw: [Complex64; 2],
    reached: [usize; 2],
    settled: bool,
    cost: Cost,
}

/// What a jet had to say, together with where each channel's series stopped improving.  That
/// order is the degree of the polynomial a traced path would subtract, and the point past which
/// the same polynomial turns and must not be trusted.
struct Expansion {
    terms: [Term; 2],
    raw: [Complex64; 2],
    reached: [usize; 2],
    cost: Cost,
}

/// Both channels plus the spend that produced them.
struct Priced {
    terms: [Term; 2],
    cost: Cost,
}

/// # QuadJet
///
/// Everything that depends on `γ` alone.  The Stokes march is the expensive part and the same
/// for every tap, so a tap reads this and writes only its own stack.
pub struct QuadJet {
    /// The accuracy being asked for, which sets both the bar a saddle's contribution is
    /// measured against and the point at which the jet's residual buys a traced path.
    pub tol: f64,
    /// Morse wavelet parameters.
    pub shape: Shape,
    /// Skip the jet's verdict and trace every saddle, which is the mode the IFFT and contour
    /// methods are compared against.
    pub quadrature_only: bool,

    frame: Frame,
    table: StokesTable,
}

impl QuadJet {
    /// Caller is responsible that `beta > 0`, `gamma` an integer in `3..=MAX_G`.
    pub fn new(shape: Shape, tol: f64, quadrature_only: bool) -> Self {
        let frame = Frame::new(shape);
        let table = StokesTable::build(&frame);
        Self {
            tol,
            shape,
            quadrature_only,
            frame,
            table,
        }
    }

    pub fn reference(shape: Shape) -> Self {
        Self::new(shape, 0.0, true)
    }

    pub fn standard(shape: Shape) -> Self {
        Self::new(shape, 1e-8, false)
    }

    pub fn tap_at(&self, u: f64) -> QuadJetResult {
        self.integrate(u)
    }

    fn integrate(&self, u: f64) -> QuadJetResult {
        let frame = &self.frame;
        let Frame { beta, g, rho, .. } = *frame;
        let t = TAU * u / rho;
        let tau = rho * t.abs() / beta;
        let rel = relative(self.tol);

        let (roots, weights) = self.table.roots_at(tau);

        let mut saddles: [Saddle; MAX_G] = [Saddle::default(); MAX_G];
        for i in 0..g {
            saddles[i] = Saddle::new(roots[i], frame);
        }
        let mut active = [0usize; MAX_G];
        let mut live = 0;
        for i in 0..g {
            if weights[i].abs() > LIVE {
                active[live] = i;
                live += 1;
            }
        }
        let active = &active[..live];

        // Every scale carries `e^{βΦ}`, whose real part runs to hundreds before anything else in
        // the tap goes wrong.  Dividing the dominant saddle's exponent out leaves ratios near
        // unity for everything downstream to compare, and it goes back on once at the end.
        let log_scale = active
            .iter()
            .map(|&i| beta * saddles[i].phi.re)
            .fold(f64::NEG_INFINITY, f64::max)
            + beta * rho.ln();

        let mut scales = [[Complex64::default(); 2]; MAX_G];
        for &i in active {
            scales[i] = saddles[i].scales(frame, log_scale);
        }

        let singulants = Singulants::new(&saddles, frame);
        let adj = singulants.adjacent();

        let mut amps = [[0.0_f64; 2]; MAX_G];
        for &i in active {
            for m in 0..2 {
                amps[i][m] = saddles[i].amplitude(frame, scales[i][m]);
            }
        }

        let dominant = [0usize, 1].map(|m| {
            active
                .iter()
                .map(|&i| amps[i][m] * weights[i].abs())
                .fold(0.0_f64, f64::max)
        });

        let mut psi_re: Accumulator<f64> = Accumulator::default();
        let mut psi_im: Accumulator<f64> = Accumulator::default();
        let mut d_re: Accumulator<f64> = Accumulator::default();
        let mut d_im: Accumulator<f64> = Accumulator::default();
        let mut residual = [0.0_f64; 2];

        let mut nodes = [(Complex64::default(), Complex64::default()); QUAD_SLOTS];
        let mut normal = Normal::default();
        let mut cost = Cost::default();

        for &i in active {
            let s = &saddles[i];
            let w = weights[i];
            let seen = singulants.from(i);

            // The bar is `rel` of the largest contribution any saddle makes to the channel,
            // divided back through this saddle's weight because everything in here is still
            // pre-weight.
            let ch = [0usize, 1].map(|m| Channel {
                scale: scales[i][m],
                target: rel * dominant[m] / w.abs(),
            });

            // The nearest singulant that can actually reach saddle `i` is both where its series
            // turns and the floor `e^{-r*}` it can never go below.  A shadowed saddle sits
            // closer in `|ΔΦ|` without being the singularity the expansion runs into.
            let mut r_star = f64::INFINITY;
            for j in 0..g {
                if adj[i] & (1 << j) != 0 {
                    r_star = r_star.min(seen[j].norm());
                }
            }

            // The bar belongs to the dominant saddle, and a suppressed one meets it with fewer
            // digits of its own.  This is what saddle `i` owes in its own units, the wider
            // channel governing since it is the one still unmet when the other has stopped.
            let local_bar = (0..2)
                .map(|m| (amps[i][m] / ch[m].target).ln())
                .fold(f64::NEG_INFINITY, f64::max);

            // Charge based on the leading Watson term being discarded
            if local_bar <= 1.0 {
                residual[0] += amps[i][0] * w.abs();
                residual[1] += amps[i][1] * w.abs();
                continue;
            }

            let jet = s.jet(frame, ch, r_star, local_bar, &mut normal);
            cost += jet.cost;

            let worth_tracing = |m: usize| jet.terms[m].residual > ch[m].target;

            // Fall back on the path where the series floor cannot reach the bar, and where the
            // series reports that it stalled before getting there.
            let [p, q] = if self.quadrature_only
                || r_star <= local_bar
                || worth_tracing(0)
                || worth_tracing(1)
            {
                // How much of the distance from full amplitude down to target the jet already covered,
                // in the same nats `bar` is measured in. A jet that settled near its own floor leaves
                // this near zero; one that stalled at n_cap leaves most of it standing.
                let jet_progress = (0..2)
                    .map(|m| ((amps[i][m] / jet.terms[m].residual.max(EPS)).ln()).max(0.0))
                    .fold(f64::INFINITY, f64::min);

                let handoff = Handoff {
                    normal: &normal,
                    degree: [2 * jet.reached[0], 2 * jet.reached[1]],
                    moments: jet.raw,
                };

                let quad = s.quadrature(
                    frame,
                    local_bar,
                    jet_progress,
                    ch,
                    seen,
                    &handoff,
                    &mut nodes,
                );
                cost += quad.cost;

                // The jet ran either way, so what the path moved is free to measure against the
                // bar that asked for it.
                for m in 0..2 {
                    let moved = (quad.terms[m].value - jet.terms[m].value).norm();
                    cost.quad_gain = cost.quad_gain.max(moved / ch[m].target);
                }
                quad.terms
            } else {
                jet.terms
            };

            residual[0] += p.residual * w.abs();
            residual[1] += q.residual * w.abs();
            let pv = p.value * w;
            let qv = q.value * w * Complex64::i();
            psi_re.add(pv.re);
            psi_im.add(pv.im);
            d_re.add(qv.re);
            d_im.add(qv.im);
        }

        let unscale = log_scale.exp() / rho;
        let psi = Complex64::new(psi_re.sum(), psi_im.sum()) * unscale;
        let d = Complex64::new(d_re.sum(), d_im.sum()) * unscale;
        let residual = [residual[0] * unscale, residual[1] * unscale];

        if t < 0.0 {
            QuadJetResult {
                psi: psi.conj(),
                d: -d.conj(),
                residual,
                cost,
            }
        } else {
            QuadJetResult {
                psi,
                d,
                residual,
                cost,
            }
        }
    }
}

/// One of the two things a saddle is asked for, `ψ` and `dψ/dt`, which travel together because
/// they differ only in the `e^{mv}` the moments read.
#[derive(Clone, Copy, Default)]
struct Channel {
    /// The prefactor, with the dominant `e^{βΦ}` already divided out.
    scale: Complex64,
    /// The bar this channel has to clear, absolute and pre-weight.
    target: f64,
}

/// Value, and whatever the producing method could say about its own error.
struct Term {
    value: Complex64,
    residual: f64,
}

/// A root of the saddle condition, carrying the numbers every later step reads off it.
#[derive(Clone, Copy, Default)]
struct Saddle {
    u: Complex64,
    b: Complex64,
    phi: Complex64,
    /// The curvature at the saddle, branch fixed so that `x` runs up the valley the contour
    /// arrives on.
    s0: Complex64,
    /// `(B+1)/γ`, the coefficient the `e^{γv}` term of the phase carries.
    bp1: Complex64,
    gi: i32,
}

impl Saddle {
    /// The phase `Φ = ln u + B(1 - 1/γ) - 1/γ` at a root, where the saddle condition collapses
    /// `u^γ - 1` into `B = iτu`.  Integer power rather than `powf`, since the principal branch
    /// of `u^γ` parts company with the polynomial over most of the root set.
    fn new(u: Complex64, frame: &Frame) -> Self {
        let Frame { gamma, gi, .. } = *frame;
        let b = u.powi(gi) - 1.0;
        let phi = u.ln() + b * (1.0 - 1.0 / gamma) - 1.0 / gamma;

        let mut s0 = (-(b * (1.0 - gamma) - gamma)).sqrt();
        if s0.re < 0.0 {
            s0 = -s0;
        }

        Saddle {
            u,
            b,
            phi,
            s0,
            bp1: (b + 1.0) / gamma,
            gi,
        }
    }

    /// The cheap approach to a saddle, Watson's lemma on the series `Normal` carries.
    ///
    /// The order is predicted rather than searched.  The singulant `r_star` says where the
    /// series turns and the bar says where its terms fall under notice, and both are known
    /// before the first coefficient exists.  A prediction that runs short is corrected upward,
    /// and correcting it costs only the orders it did not already have.
    fn jet(
        &self,
        frame: &Frame,
        ch: [Channel; 2],
        r_star: f64,
        bar: f64,
        normal: &mut Normal,
    ) -> Expansion {
        normal.seed(self, frame);

        let n_cap = (r_star.ceil() as usize).clamp(1, JET_ORDER / 2);
        let mut n = bar.max(1.0);
        for _ in 0..2 {
            n = bar / ((r_star / n).max(1.0).ln() + 1.0);
        }
        let mut n_max = (n.ceil() as usize)
            .clamp(1, n_cap)
            .max(JET_MIN_ORDER.min(n_cap));
        let mut cost = Cost::default();
        loop {
            normal.extend(self, frame, 2 * n_max);
            let pass = self.jet_sum(frame, ch, r_star, n_max, normal);
            cost += pass.cost;
            if pass.settled || n_max == n_cap {
                return Expansion {
                    terms: pass.terms,
                    raw: pass.raw,
                    reached: pass.reached,
                    cost,
                };
            }
            n_max = (2 * n_max).min(n_cap);
        }
    }

    /// The series at one order.  Each even coefficient meets its Gaussian moment and the two
    /// channels differ only in which `h` they read.
    ///
    /// A channel that ended on the mathematics, having diverged or crossed under the bar or run
    /// into roundoff, has said everything it is going to say, and only a channel that ran out of
    /// order is worth asking again.
    fn jet_sum(
        &self,
        frame: &Frame,
        ch: [Channel; 2],
        r_star: f64,
        n_max: usize,
        normal: &Normal,
    ) -> Pass {
        let beta = frame.beta;
        let scale_abs = [ch[0].scale.norm(), ch[1].scale.norm()];

        let mut moment = (TAU / beta).sqrt();
        let mut value = [Complex64::default(); 2];
        let mut prev = [f64::INFINITY; 2];
        let mut last = [0.0_f64; 2];
        let mut residual = [f64::INFINITY; 2];
        let mut settled = [false; 2];
        let mut reached = [0usize; 2];
        for n in 0..=n_max {
            for m in 0..2 {
                if settled[m] {
                    continue;
                }
                let term = normal.h[m][2 * n] * moment;
                let size = scale_abs[m] * term.norm();

                // A conjugate pair of singulants makes the terms oscillate, which is forced at
                // `τ = 0` where every `Δ` is imaginary, so a single small term is a zero of the
                // oscillation rather than the remainder.  The envelope is what decays.
                let env = size.max(last[m]);
                last[m] = size;

                if env > prev[m] {
                    settled[m] = true;
                    continue;
                }

                prev[m] = env;
                value[m] += term;
                reached[m] = n;
                residual[m] = env * (2.0 * n as f64 / r_star).max(1.0);

                // Meeting the bar is reason enough to stop reporting, but not to stop building.
                // The path downstream walks whatever shape this series hands it, so a channel
                // that arrived early keeps going until it has some to hand over.
                let done = env <= ch[m].target || env <= EPS * scale_abs[m] * value[m].norm();
                if done && n >= JET_MIN_ORDER {
                    settled[m] = true;
                }
            }

            if settled[0] && settled[1] {
                break;
            }

            moment *= (2 * n + 1) as f64 / beta;
        }

        Pass {
            terms: [
                Term {
                    value: ch[0].scale * value[0],
                    residual: residual[0],
                },
                Term {
                    value: ch[1].scale * value[1],
                    residual: residual[1],
                },
            ],
            reached,
            raw: value,
            settled: settled[0] && settled[1],
            cost: Cost {
                jet_passes: 1,
                jet_orders: reached[0].max(reached[1]) as u32,
                ..Default::default()
            },
        }
    }

    /// The expensive approach to a saddle, a trapezoid along its traced steepest-descent path.
    ///
    /// Node density is a question about the neighbors and nothing else.  Aliasing folds in
    /// whatever feature sits `gap` off the path carrying the size that feature already has, so a
    /// neighbor suppressed below the bar asks for nothing and the closest survivor asks for
    /// everything.  Distance decides which one binds.  The trapezoid buries that neighbor as
    /// `e^{-2π·gap·density}`, so the density that clears the bar is a closed form and the pass
    /// lands on it directly.  The ladder below survives to catch a gap that was not the whole
    /// story, not to find the answer by climbing.
    fn quadrature(
        &self,
        frame: &Frame,
        bar: f64,
        jet_progress: f64,
        ch: [Channel; 2],
        seen: &[Complex64],
        hand: &Handoff,
        nodes: &mut [(Complex64, Complex64)],
    ) -> Priced {
        // The neighbor that binds is the one whose own size, carried onto the path at its own
        // distance, is hardest to bury.  One standing above this saddle enters amplified and
        // charges full freight, while one already suppressed is credited only partly, since the
        // fold point sits nearer than the neighbor's own center and sees less of that decay.
        let mut gap = 1.0_f64;
        let mut lift = 0.0_f64;
        let mut want = 0.0_f64;
        for &d in seen {
            let t = (-2.0 * d).sqrt().im.abs();
            if t <= GAP_FLOOR {
                continue;
            }
            let l = if d.re > 0.0 { d.re } else { LIFT_CREDIT * d.re };
            let need = (bar + l).max(0.0) / t;
            if need > want {
                want = need;
                gap = t;
                lift = l;
            }
        }

        let reach = (2.0 * (bar + 3.0)).sqrt().min(REACH_MAX);
        let tail = (-0.5 * reach * reach).exp() * (TAU / frame.beta).sqrt() / self.s0.norm();

        let density = (QUAD_MARGIN * (bar + lift) / (TAU * (gap - jet_progress)))
            .clamp(QUAD_MIN_DENSITY, QUAD_MAX_DENSITY);

        let settle =
            |terms: &mut [Term; 2], resid: [Complex64; 2], diff: [f64; 2], contraction: f64| {
                for m in 0..2 {
                    let value = ch[m].scale * (resid[m] + hand.moments[m]);
                    let noise = EPS * value.norm();
                    let endpoint = tail * ch[m].scale.norm();
                    terms[m] = Term {
                        value,
                        residual: (diff[m] * contraction).max(noise) + endpoint,
                    };
                }
            };

        let converged = |terms: &[Term; 2]| {
            (0..2).all(|m| terms[m].residual <= ch[m].target.max(EPS * terms[m].value.norm()))
        };

        // One pass at the predicted density.  With no second level there is no diff to measure,
        // so the model's suppression is read against what the path actually integrated.  That is
        // the residual the series could not reach rather than the answer itself.
        let mut cost = Cost::default();
        let (mut resid, seed_cost) = self.trapezoid(frame, reach, density, hand, nodes, false);
        cost += seed_cost;

        let modeled = (lift - TAU * gap * density).exp().max(EPS);

        let mut terms = [0, 1].map(|m| Term {
            value: ch[m].scale * (resid[m] + hand.moments[m]),
            residual: f64::INFINITY,
        });
        let guess = [0, 1].map(|m| (ch[m].scale * resid[m]).norm());
        settle(&mut terms, resid, guess, modeled);

        // The prediction is an ETA.  If the closed form was reading the wrong neighbor, the
        // ladder buys density back a doubling at a time, now with a rate it can measure.
        let mut density = density;
        let mut prev_diff = [f64::INFINITY; 2];
        while !converged(&terms) && 2.0 * density <= QUAD_MAX_DENSITY {
            density *= 2.0;
            let (next, step_cost) = self.trapezoid(frame, reach, density, hand, nodes, true);
            cost += step_cost;
            let diff = [0, 1].map(|m| (ch[m].scale * (next[m] - resid[m])).norm());
            let measured = (0..2)
                .map(|m| {
                    if prev_diff[m].is_finite() && prev_diff[m] > 0.0 {
                        diff[m] / prev_diff[m]
                    } else {
                        1.0
                    }
                })
                .fold(0.0_f64, f64::max);
            let modeled = (lift - TAU * gap * density).exp().max(EPS);
            settle(&mut terms, next, diff, modeled.max(measured));
            prev_diff = diff;
            resid = next;
        }
        Priced { terms, cost }
    }

    /// One level of the ladder.  In the normal coordinate the phase is exactly `-x²/2`, so the
    /// weight is Gaussian by construction and cannot overflow the way a straight segment in `v`
    /// does once the reach is long enough for `e^{γv}` to climb the neighboring hill.  The path
    /// carries no `m`, so one trace serves both channels.
    ///
    /// Nodes are held at the finest density this ladder will reach, so a refinement finds half
    /// its work already done beside it.
    fn trapezoid(
        &self,
        frame: &Frame,
        reach: f64,
        density: f64,
        hand: &Handoff,
        nodes: &mut [(Complex64, Complex64)],
        refine: bool,
    ) -> ([Complex64; 2], Cost) {
        let beta = frame.beta;

        let h = 1.0 / (density * beta.sqrt());
        let n = (reach * density).floor() as usize;

        // At the saddle `x` and `g'` vanish together and the slope `-x/g'(v)` goes to `1/s0`.
        let apex = (Complex64::default(), self.s0.inv());
        nodes[0] = apex;
        nodes[1] = apex;

        if refine {
            for si in 0..2 {
                for j in (1..=n / 2).rev() {
                    nodes[4 * j + si] = nodes[2 * j + si];
                }
            }
        }

        let mut acc = [
            (Accumulator::<f64>::default(), Accumulator::<f64>::default()),
            (Accumulator::<f64>::default(), Accumulator::<f64>::default()),
        ];

        let mut tally = |x: f64, v: Complex64, slope: Complex64| {
            let ev = v.exp();
            let gauss = (-0.5 * beta * x * x).exp();
            let f1 = gauss * (ev * slope - hand.poly(0, x));
            let f2 = gauss * (ev * ev * slope - hand.poly(1, x));
            acc[0].0.add(f1.re);
            acc[0].1.add(f1.im);
            acc[1].0.add(f2.re);
            acc[1].1.add(f2.im);
        };

        tally(0.0, apex.0, apex.1);

        let mut walked = 0u32;

        for (si, side) in [1.0_f64, -1.0].into_iter().enumerate() {
            for j in 1..=n {
                let slot = 2 * j + si;
                let x = side * j as f64 * h;

                if !refine || j % 2 == 1 {
                    let x_prev = side * (j - 1) as f64 * h;
                    nodes[slot] = self.walk(nodes[2 * (j - 1) + si], x_prev, x);
                    walked += 1;
                }

                let (v, slope) = nodes[slot];
                tally(x, v, slope);
            }
        }

        drop(tally);

        (
            [
                Complex64::new(acc[0].0.sum(), acc[0].1.sum()) * h,
                Complex64::new(acc[1].0.sum(), acc[1].1.sum()) * h,
            ],
            Cost {
                quad_paths: 1,
                quad_nodes: walked,
                ..Default::default()
            },
        )
    }

    /// One step along the path that `dv/dx = -x/g'(v)` predicts, closed by Newton on
    /// `g(v) = -x²/2`.
    ///
    /// The valley runs close by neighboring saddles, where `g'` is small enough that the
    /// prediction overshoots and Newton lands on the neighbor's sheet.  Refusing a step that
    /// wanders leaves the caller to approach the neighbor more slowly.
    fn advance(
        &self,
        v: Complex64,
        slope: Complex64,
        x0: f64,
        x1: f64,
    ) -> Option<(Complex64, Complex64)> {
        let h = x1 - x0;
        let mut w = v + slope * h;
        let target = Complex64::new(-0.5 * x1 * x1, 0.0);

        let mut converged = false;
        let mut gp = Complex64::default();
        for _ in 0..NEWTON_ITERS {
            let (gv, gpv) = self.g(w);
            let step = (gv - target) / gpv;
            w -= step;
            gp = gpv;
            if step.norm() < NEWTON_TOL {
                converged = true;
                break;
            }
        }

        if !converged || (w - v).norm() > 4.0 * slope.norm() * h.abs() + 0.25 {
            return None;
        }
        Some((w, -x1 / gp))
    }

    /// Subdivides a step until the path stays in its own valley.
    fn walk(&self, from: (Complex64, Complex64), x0: f64, x1: f64) -> (Complex64, Complex64) {
        let mut splits = 1u32;
        loop {
            let sub = (x1 - x0) / splits as f64;
            let mut at = from;
            let mut ok = true;
            for k in 1..=splits {
                let a = x0 + sub * (k - 1) as f64;
                let b = x0 + sub * k as f64;
                match self.advance(at.0, at.1, a, b) {
                    Some(next) => at = next,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok || splits > 1 << TRACE_MAX_SPLIT {
                return at;
            }
            splits *= 2;
        }
    }

    /// This saddle's prefactor for each channel, with the dominant `e^{βΦ}` already divided out
    /// so that the caller can restore it once at the end.
    fn scales(&self, frame: &Frame, log_scale: f64) -> [Complex64; 2] {
        let Frame { beta, rho, .. } = *frame;
        let base = (self.phi * beta + beta * rho.ln() - log_scale).exp();
        [base * rho * self.u, base * rho * rho * self.u * self.u]
    }

    /// The phase `g(v) = v + B(e^v - 1) - ((B+1)/γ)(e^{γv} - 1)` measured from the saddle, along
    /// with the derivative every Newton iteration of every node asks for.  `e^{γv}` is an
    /// integer power of `e^v`, so one transcendental serves both.
    fn g(&self, v: Complex64) -> (Complex64, Complex64) {
        let ev = v.exp();
        let egv = ev.powi(self.gi);
        (
            v + self.b * (ev - 1.0) - self.bp1 * (egv - 1.0),
            1.0 + self.b * ev - (self.b + 1.0) * egv,
        )
    }

    /// The leading Watson term `|scale| √(2π/β)/|s₀|`, which is what this saddle contributes
    /// before any correction and what its bar is measured against.  The Gaussian width belongs
    /// in it, since a saddle with small `Φ''` is wide and outweighs its prefactor.
    fn amplitude(&self, frame: &Frame, scale: Complex64) -> f64 {
        scale.norm() * (TAU / frame.beta).sqrt() / self.s0.norm()
    }
}

/// What the jet leaves behind for the path to stand on.  The series carries the shape the
/// integrand has near the saddle, so the trapezoid walks only what is left once that shape comes
/// out, and the jet's own moments put it back exactly.
///
/// The polynomial turns somewhere past the order it reached, but every one of its terms peaks at
/// `√k` in the standardized coordinate and the reach always sits beyond `√(2·reached)`, so the
/// Gaussian has buried the turn before the path arrives at it.
struct Handoff<'a> {
    normal: &'a Normal,
    /// Degree in `x`, twice the order each channel reached.
    degree: [usize; 2],
    /// The Gaussian integral of that polynomial, pre-scale, which is the sum the jet already
    /// formed.
    moments: [Complex64; 2],
}

impl Handoff<'_> {
    fn poly(&self, m: usize, x: f64) -> Complex64 {
        let c = &self.normal.h[m];
        let mut acc = c[self.degree[m]];
        for k in (0..self.degree[m]).rev() {
            acc = acc * x + c[k];
        }
        acc
    }
}

/// The integrand in the normal coordinate, carried as a series in `x`.
///
/// The saddle is where `g'` vanishes, so `P = g'` has no constant term and `P = x·P̃`.  That
/// missing term is what makes the whole scheme go.  A convolution that would otherwise close on
/// itself at order `k` instead drops its own leading term, and the order-`k` coefficient falls
/// out against lower ones.
///
/// Every recurrence below reads only what sits beneath it, so a jet that guessed its order too
/// low extends into the series it already has.  Growing to twice the order costs the difference
/// and nothing more, which is the whole reason for the coordinate change.
///
/// `h` is the pair the moments read and the pair a traced path will subtract, held to the full
/// order rather than the even half, since the path samples both signs of `x` and wants the odd
/// coefficients the Gaussian throws away.
struct Normal {
    /// `e^{v(x)}`.
    y: [Complex64; JET_SLOTS],
    /// `y^γ`, kept because `P` reads it at every order and rebuilding it is a convolution.
    yg: [Complex64; JET_SLOTS],
    /// `g'` along the path, `P_0 = 0` by construction.
    p: [Complex64; JET_SLOTS],
    /// `dv/dx`, the same slope `trapezoid` walks.
    w: [Complex64; JET_SLOTS],
    /// `e^{v}·dv/dx` and `e^{2v}·dv/dx`, one per channel.
    h: [[Complex64; JET_SLOTS]; 2],
    /// The valid prefix of `w` and `h`.  `y`, `yg`, and `p` run one order further.
    width: usize,
}

impl Default for Normal {
    fn default() -> Self {
        Self {
            y: [Complex64::default(); JET_SLOTS],
            yg: [Complex64::default(); JET_SLOTS],
            p: [Complex64::default(); JET_SLOTS],
            w: [Complex64::default(); JET_SLOTS],
            h: [[Complex64::default(); JET_SLOTS]; 2],
            width: 0,
        }
    }
}

impl Normal {
    /// Plants the series at the saddle.  Order one is the quadratic root of `y₁P₁ = -1`, whose
    /// branch is the one `s0` already committed to, and everything above it is linear.
    fn seed(&mut self, s: &Saddle, frame: &Frame) {
        let y1 = s.s0.inv();
        self.y[0] = Complex64::new(1.0, 0.0);
        self.y[1] = y1;
        self.yg[0] = Complex64::new(1.0, 0.0);
        self.yg[1] = y1 * frame.gamma;
        self.p[0] = Complex64::default();
        self.p[1] = -s.s0;
        self.w[0] = y1;
        self.h[0][0] = y1;
        self.h[1][0] = y1;
        self.width = 0;
    }

    /// Grows the series to `target`, picking up wherever the last call stopped.
    fn extend(&mut self, s: &Saddle, frame: &Frame, target: usize) {
        if target <= self.width {
            return;
        }
        let gamma = frame.gamma;
        let bp1 = s.b + 1.0;
        let y1 = self.y[1];

        // Walk `y` up first, since `P` is a function of it and the slope reads `P` one order
        // above its own.  `y^γ` splits into the part that carries `y_k` and the part that does
        // not, and only the second half is known when the step begins.
        for k in (self.width + 2)..=(target + 1) {
            let mut tail = Complex64::default();
            for i in 1..k {
                tail += (gamma * i as f64 - (k - i) as f64) * self.y[i] * self.yg[k - i];
            }
            tail /= k as f64;

            let mut mid = Complex64::default();
            for j in 2..k {
                mid += (k - j + 1) as f64 * self.y[k - j + 1] * self.p[j];
            }

            self.y[k] = (self.y[k - 1] + mid - y1 * bp1 * tail) / ((k + 1) as f64 * s.s0);
            self.yg[k] = self.y[k] * gamma + tail;
            self.p[k] = self.y[k] * s.b - self.yg[k] * bp1;
        }

        // The slope inverts `P̃w = -1`, and the two channels follow as products against `y`.
        for k in (self.width + 1)..=target {
            let mut acc = Complex64::default();
            for i in 0..k {
                acc += self.w[i] * self.p[k - i + 1];
            }
            self.w[k] = -acc / self.p[1];

            let mut h0 = Complex64::default();
            let mut h1 = Complex64::default();
            for i in 0..=k {
                h0 += self.y[i] * self.w[k - i];
            }
            self.h[0][k] = h0;
            for i in 0..=k {
                h1 += self.y[i] * self.h[0][k - i];
            }
            self.h[1][k] = h1;
        }

        self.width = target;
    }
}

/// `Δ` between every pair of saddles, stored so that `from(i)` is the row saddle `i` reads:
/// `Δ_i[j] = β(Φ_j - Φ_i)`, the exponent a neighbor `j` enters at when `i` is the one being
/// evaluated.  The self entry is zero, which every consumer screens out by size.
struct Singulants {
    d: [Complex64; MAX_G * MAX_G],
    g: usize,
}

impl Singulants {
    fn new(saddles: &[Saddle; MAX_G], frame: &Frame) -> Self {
        let g = frame.g;
        let mut d = [Complex64::default(); MAX_G * MAX_G];
        for i in 0..g {
            for j in 0..g {
                d[i * g + j] = (saddles[j].phi - saddles[i].phi) * frame.beta;
            }
        }
        Self { d, g }
    }

    fn from(&self, i: usize) -> &[Complex64] {
        &self.d[i * self.g..(i + 1) * self.g]
    }

    /// Screens the singulants that could set each saddle's truncation order.  A saddle lying
    /// behind a nearer one in the same direction is not the singularity the series runs into,
    /// and screening too loosely only lowers `r_star`.
    fn adjacent(&self) -> [u8; MAX_G] {
        let mut adj = [0u8; MAX_G];
        for i in 0..self.g {
            let seen = self.from(i);
            for k in 0..self.g {
                if k == i {
                    continue;
                }
                let dk = seen[k];
                let mut shadowed = false;
                for j in 0..self.g {
                    if j == i || j == k {
                        continue;
                    }
                    let dj = seen[j];
                    if dj.norm() < dk.norm()
                        && (dk * dj.conj()).re > ADJ_CONE * dk.norm() * dj.norm()
                    {
                        shadowed = true;
                        break;
                    }
                }
                if !shadowed {
                    adj[i] |= 1 << k;
                }
            }
        }
        adj
    }
}

/// The saddle geometry that depends on `τ` alone, held in the units `Φ` comes in so that a tap
/// applies its own `β` on the way out.  The transverse reach carries `√β` and the turn carries
/// a full `β`.
#[derive(Clone, Copy, Default)]
struct Geometry {
    /// Distance off the path to the nearest neighbor that can alias onto it.
    gap: f64,
    /// The same for the next one out, which is where a residual's aliasing moves once the
    /// nearest has been subtracted away.  The ratio against `gap` is what says whether one
    /// subtraction is worth attempting.
    gap_next: f64,
    /// `|ΔΦ|` to the nearest unshadowed neighbor, where the series turns.
    turn: f64,
    /// Which saddle owns `gap`.
    binding: u8,
    /// The unshadowed neighbors, screened as in `Singulants::adjacent`.
    adj: u8,
}

/// Reads the whole neighbor picture off one node's phases.  The shadow cone and the norm
/// comparisons are ratios, so `β` cancels out of the screen entirely and only the distances
/// carry it.
fn geometry(p: &[Complex64; MAX_G], g: usize) -> [Geometry; MAX_G] {
    let mut out = [Geometry::default(); MAX_G];
    for i in 0..g {
        let mut d = [Complex64::default(); MAX_G];
        for k in 0..g {
            d[k] = p[k] - p[i];
        }

        let mut adj = 0u8;
        for k in 0..g {
            if k == i {
                continue;
            }
            let dk = d[k];
            let mut shadowed = false;
            for j in 0..g {
                if j == i || j == k {
                    continue;
                }
                let dj = d[j];
                if dj.norm() < dk.norm() && (dk * dj.conj()).re > ADJ_CONE * dk.norm() * dj.norm() {
                    shadowed = true;
                    break;
                }
            }
            if !shadowed {
                adj |= 1 << k;
            }
        }

        let mut gap = f64::INFINITY;
        let mut gap_next = f64::INFINITY;
        let mut binding = i as u8;
        let mut turn = f64::INFINITY;
        for k in 0..g {
            if k == i {
                continue;
            }
            let t = (-2.0 * d[k]).sqrt().im.abs();
            if t > GAP_FLOOR {
                if t < gap {
                    gap_next = gap;
                    gap = t;
                    binding = k as u8;
                } else if t < gap_next {
                    gap_next = t;
                }
            }
            if adj & (1 << k) != 0 {
                turn = turn.min(d[k].norm());
            }
        }

        out[i] = Geometry {
            gap,
            gap_next,
            turn,
            binding,
            adj,
        };
    }
    out
}

/// Root identity and membership as functions of `τ`, resolved once.
///
/// Membership changes only where two saddles exchange dominance, so it has to be marched from
/// the seed at `τ = 0` rather than evaluated at a point.  Nothing in the march depends on `β`,
/// so it happens at construction and a tap brackets into the result and polishes.
///
/// Nodes sit in `ln(1 + τ)`, where the roots move slowly.  `du/dτ = iu/(γu^{γ-1} - iτ)` carries
/// the double-root condition in its denominator, which for `γ ≥ 3` never comes due on the real
/// `τ` axis, so a node's roots stay within reach of Newton across its interval.
struct StokesTable {
    g: usize,
    /// `ln(1 + τ)` at each node, ascending.
    s: Box<[f64]>,
    /// Row-major `[node][root]`, labeled by continuation from the roots of unity.
    roots: Box<[Complex64]>,
    /// Row-major `[node][root]`.
    m: Box<[f64]>,
    /// Row-major `[node][saddle]`.
    geom: Box<[Geometry]>,
}

impl StokesTable {
    fn build(frame: &Frame) -> Self {
        let g = frame.g;

        let mut roots = vec![Complex64::default(); TABLE_NODES * g].into_boxed_slice();
        let mut geom = vec![Geometry::default(); TABLE_NODES * g].into_boxed_slice();
        let mut m = vec![0.0_f64; TABLE_NODES * g].into_boxed_slice();
        let mut s = vec![0.0_f64; TABLE_NODES].into_boxed_slice();

        let mut r: [Complex64; MAX_G] = [Complex64::default(); MAX_G];
        for k in 0..g {
            r[k] = Complex64::from_polar(1.0, TAU * k as f64 / frame.gamma);
        }
        let mut weights = [0.0_f64; MAX_G];
        weights[0] = 1.0;

        let phis = |r: &[Complex64; MAX_G]| -> [Complex64; MAX_G] {
            let mut p = [Complex64::default(); MAX_G];
            for k in 0..g {
                p[k] = Saddle::new(r[k], frame).phi;
            }
            p
        };

        let mut prev_im = [0.0_f64; MAX_G * MAX_G];
        let p = phis(&r);
        for j in 0..g {
            for k in 0..g {
                prev_im[j * g + k] = (p[j] - p[k]).im;
            }
        }

        let mut last_flip_s = 0.0_f64;
        let mut used = 0usize;

        let s_max = (1.0 + TABLE_TAU_MAX).ln();
        for n in 0..TABLE_NODES {
            let sn = s_max * n as f64 / (TABLE_NODES - 1) as f64;
            let tau = sn.exp() - 1.0;

            durand_kerner(&mut r, g, tau);
            let p = phis(&r);

            // A crossing is where `Im Δ` changes sign while `Re Δ > 0`, at which point the
            // dominant saddle hands its membership across to the one it dominates.
            let held = weights;
            for j in 0..g {
                for k in 0..g {
                    if j == k {
                        continue;
                    }
                    let d = p[j] - p[k];
                    let idx = j * g + k;
                    let im = d.im;
                    if d.re > 0.0 && prev_im[idx] * im < 0.0 {
                        weights[k] += im.signum() * held[j];
                        last_flip_s = sn;
                    }
                    prev_im[idx] = im;
                }
            }

            s[n] = sn;
            roots[n * g..(n + 1) * g].copy_from_slice(&r[..g]);
            m[n * g..(n + 1) * g].copy_from_slice(&weights[..g]);
            geom[n * g..(n + 1) * g].copy_from_slice(&geometry(&p, g)[..g]);
            used = n + 1;

            // Once the crossings have stopped coming the decomposition is settled and there is
            // nothing left for the march to discover.
            if sn - last_flip_s > TABLE_QUIET_SPAN && n > 8 {
                break;
            }
        }

        Self {
            g,
            s: s[..used].to_vec().into_boxed_slice(),
            roots: roots[..used * g].to_vec().into_boxed_slice(),
            m: m[..used * g].to_vec().into_boxed_slice(),
            geom: geom[..used * g].to_vec().into_boxed_slice(),
        }
    }

    /// The roots at `tau` with the membership they carry.  Bracket into the march, take that
    /// node's roots as a seed, and let Newton close the rest.  Past the last node the
    /// membership is settled and the roots follow their asymptotics, so the final node goes on
    /// serving as a seed.
    fn roots_at(&self, tau: f64) -> ([Complex64; MAX_G], [f64; MAX_G]) {
        let g = self.g;
        let target = (1.0 + tau).ln();

        let n = match self
            .s
            .binary_search_by(|probe| probe.partial_cmp(&target).unwrap())
        {
            Ok(hit) => hit,
            Err(above) => above.min(self.s.len() - 1),
        };

        let mut r = [Complex64::default(); MAX_G];
        r[..g].copy_from_slice(&self.roots[n * g..(n + 1) * g]);

        let mut w = [0.0_f64; MAX_G];
        w[..g].copy_from_slice(&self.m[n * g..(n + 1) * g]);

        for k in 0..g {
            newton_trinomial(&mut r[k], g, tau);
        }

        (r, w)
    }

    /// The neighbor picture at `tau`, read off the bracketing node without polishing.  The
    /// roots move slowly in `ln(1 + τ)` and this only ever feeds a decision about which method
    /// to spend, so the node's own geometry stands in for the exact one.
    fn geometry_at(&self, tau: f64) -> &[Geometry] {
        let target = (1.0 + tau).ln();
        let n = match self
            .s
            .binary_search_by(|probe| probe.partial_cmp(&target).unwrap())
        {
            Ok(hit) => hit,
            Err(above) => above.min(self.s.len() - 1),
        };
        &self.geom[n * self.g..(n + 1) * self.g]
    }
}

/// Polishes a single root of `u^γ - iτu - 1` in place, quadratic from a table seed.
fn newton_trinomial(u: &mut Complex64, g: usize, tau: f64) {
    let gi = g as i32;
    let it = Complex64::i() * tau;
    for _ in 0..TABLE_NEWTON_ITERS {
        let p = u.powi(gi) - it * *u - 1.0;
        let dp = u.powi(gi - 1) * g as f64 - it;
        let step = p / dp;
        *u -= step;
        if step.norm() < TABLE_NEWTON_TOL {
            break;
        }
    }
}

/// Solves the trinomial for all `γ` roots at once during the march, warm-started in place so
/// that root identity survives the step.
fn durand_kerner(r: &mut [Complex64; MAX_G], g: usize, tau: f64) {
    for _ in 0..DK_ITERS {
        let mut moved = 0.0_f64;
        for k in 0..g {
            let p = r[k].powi(g as i32) - Complex64::i() * tau * r[k] - 1.0;
            let mut den = Complex64::new(1.0, 0.0);
            for j in 0..g {
                if j != k {
                    den *= r[k] - r[j];
                }
            }
            let step = p / den;
            r[k] -= step;
            moved = moved.max(step.norm());
        }
        if moved < DK_TOL {
            break;
        }
    }
}

/// The bar, as a fraction of the dominant contribution.  Half of `tol` leaves the other half for
/// the sum over saddles, and the floor is a few ulp of a partial sum that has already lost bits
/// to cancellation.
fn relative(tol: f64) -> f64 {
    0.5 * tol.max(8.0 * EPS)
}

#[cfg(test)]
mod test {
    use super::super::fmt_e;
    use super::*;

    #[test]
    fn convergence_precision() {
        let shape = Shape::from_q(3.5, 3.0);
        let ref_jet = QuadJet::reference(shape);

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Origin, transition, the band where the jet's verdict has gone wrong before, and deep
        // tail.  Suspiciously precise probes are old sore spots.
        let probes = [
            0.0, 0.8, 2.0, 2.9375, 3.7, 4.7, 5.5, 6.7, 7.911, 8.2, 8.33, 11.0, 14.0,
        ];

        #[derive(Clone, Copy)]
        enum Channel {
            Psi,
            D,
        }

        let pick = |t: Channel, result: QuadJetResult| match t {
            Channel::Psi => result.psi,
            Channel::D => Complex64::new(777.0, 777.0), // XXX Add back in after supporting the D
        };

        let sweep =
            |channel: Channel, name: &str, knob: &str, vary: &dyn Fn(u32) -> QuadJet, rows: u32| {
                let channel_label = match channel {
                    Channel::Psi => "psi",
                    Channel::D => "d",
                };
                println!("\n=== {name} ({channel_label}) ===");
                print!("  {knob:>10} |");
                for u in probes {
                    print!(" {u:>9.2}");
                }
                println!();

                for i in 0..rows {
                    let test_jet = vary(i);
                    let label = test_jet.tol;

                    print!("  {label:>10.1e} |");
                    for u in probes {
                        let v = pick(channel, test_jet.tap_at(u));
                        let r = pick(channel, ref_jet.tap_at(u));
                        print!(" {:>9}", fmt_e(rel(v, r)));
                    }
                    println!();
                }
            };

        // XXX Support the D
        // Channel::D
        for tap in [Channel::Psi] {
            sweep(
                tap,
                "Cranking tol",
                "tol",
                &|i| QuadJet::new(shape, 1.0 / 10.0f64.powf((i + 1) as f64), false),
                12,
            );
        }

        // Dense grid so a narrow seam can't hide between probes.
        println!("\n=== Current Defaults ===");
        let standard_jet = QuadJet::standard(shape);

        let mut worst: f64 = 0.0;
        let mut worst_u: f64 = 0.0;

        let mut worst_time: std::time::Duration = std::time::Duration::default();
        let mut worst_time_u: f64 = 0.0;

        // NOTE this is crude and depends on settings but provides some development signal.
        let mut matrix_time = std::time::Duration::default();

        for k in 0..=4096 {
            let u = k as f64 * 0.00625 * 0.5;

            let reference = ref_jet.tap_at(u);
            let matrix_row_start = std::time::Instant::now();
            let standard = standard_jet.tap_at(u);
            let elapsed = matrix_row_start.elapsed();
            matrix_time += elapsed;

            if elapsed > worst_time && k != 0 {
                worst_time = elapsed;
                worst_time_u = u;
            }

            // XXX after supporting the D, add the max chain back in.
            let err = rel(standard.psi, reference.psi);
            if err > worst {
                worst = err;
                worst_u = u;
            }

            if k % 100 == 0 {
                let c = standard.cost;
                println!(
                    "  u: {u:>6.2}, err: {:>9}, res: {:>9}, time: {:>6}µs, jet: {:>3}p/{:>4}o, quad: {:>2}p/{:>4}n, gain: {:>9}",
                    fmt_e(err),
                    fmt_e(standard.residual[0] / standard.psi.norm()),
                    elapsed.as_micros(),
                    c.jet_passes,
                    c.jet_orders,
                    c.quad_paths,
                    c.quad_nodes,
                    fmt_e(c.quad_gain),
                );
            }
        }

        println!(
            "  worst precision over grid: {} at {:0.5}",
            fmt_e(worst),
            worst_u
        );
        println!(
            "  worst time over grid: {}µs at {:0.5}",
            worst_time.as_micros(),
            worst_time_u
        );
        println!("  matrix completed in: {}µs", matrix_time.as_micros());
        assert!(worst < 1e-7);
    }
}
