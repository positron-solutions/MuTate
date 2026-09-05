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
const DK_ITERS: usize = 32;
const DK_TOL: f64 = 1e-16;
const TABLE_NEWTON_TOL: f64 = 1e-16;
const TABLE_NEWTON_ITERS: usize = 32;
const TABLE_NODES: usize = 1024;
const TABLE_QUIET_SPAN: f64 = 4.0;
const TABLE_TAU_MAX: f64 = 128.0;

// Saddle orientation
const DESCENT_AXIS_TIE: f64 = 1e-12;

// Jets
const ADJ_CONE: f64 = 0.5;
const HANDOVER_SPLITS: u32 = 2;
const JET_ORDER: usize = 128;
const JET_SLOTS: usize = JET_ORDER + 2;
const JET_SETTLE_FLOOR: usize = 4;
const JET_OVERSHOOT: f64 = 2.5;
const LIVE: f64 = 0.5;
const MAX_G: usize = 8;
const TURN_CONFIRM: u32 = 8;

// Quadrature
const CALIBRATE_NODES: f64 = 2.0;
const ANCHOR_EFOLDS: f64 = 3.0;
const CERTIFY_DERATE: f64 = 0.35;
const GAP_FLOOR: f64 = 1e-6;
const LIFT_CREDIT: f64 = 0.5;
const NEWTON_ITERS: usize = 8;
const NEWTON_STALL: f64 = 1e-11;
const NEWTON_TOL_FACTOR: f64 = 0.1;
const NEWTON_TOL_FLOOR: f64 = 1e-13;
const PLACE_MARGIN: f64 = 1.2;
const PLACE_ROUNDS: u32 = 3;
const QUAD_MAX_NODES: usize = 2048;
const QUAD_MIN_DENSITY: f64 = 1e-3;
const RATE_CREDIBLE: f64 = 4.0;
const REACH_MARGIN: f64 = 3.0;
const REACH_MAX: f64 = 16.0;
const SERIES_TRUST: f64 = 0.5;
const TRACE_MAX_SPLIT: u32 = 8;

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
    pub residual: f64,
    /// Approximate spend for this tap.
    pub cost: Cost,
}

/// Crude spend, for development signal only.  Every column sums over the call tree of one tap,
/// so the numbers are read against themselves across a change rather than against each other.
#[derive(Clone, Copy, Default)]
pub struct Cost {
    /// `jet_pass` invocations.
    pub jet_passes: u32,
    /// Summed order reached.
    pub jet_orders: u32,
    /// The deepest order any single pass reached.
    pub jet_depth: u32,
    /// Saddles the jet was run on at all.
    pub jet_saddles: u32,
    /// `trapezoid` invocations.
    pub quad_paths: u32,
    /// Nodes a `walk` was actually spent on, summed over every level traced.
    pub quad_nodes: u32,
    /// Nodes the map's own series placed instead, which cost a Horner pass apiece.
    pub quad_series: u32,
    /// How far the traced path moved the jet's answer, measured against the bar that asked for
    /// it.  Under one, the series had already arrived wherever a path was spent.
    pub quad_gain: f64,
    /// A trace ran out of valley before the reach was spent, so part of the contour is carried
    /// by the endpoint charge rather than by nodes.
    pub quad_truncated: bool,
    /// The placement stopped with the residual still above the bar it was given.
    pub quad_short: bool,
    pub quad_density_ratio: f64,
    pub quad_rate_ratio: f64,
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, rhs: Self) {
        self.jet_passes += rhs.jet_passes;
        self.jet_orders += rhs.jet_orders;
        self.jet_depth = self.jet_depth.max(rhs.jet_depth);
        self.jet_saddles += rhs.jet_saddles;
        self.quad_paths += rhs.quad_paths;
        self.quad_nodes += rhs.quad_nodes;
        self.quad_series += rhs.quad_series;
        self.quad_gain = self.quad_gain.max(rhs.quad_gain);
        self.quad_truncated |= rhs.quad_truncated;
        self.quad_short |= rhs.quad_short;
        self.quad_density_ratio = self.quad_density_ratio.max(rhs.quad_density_ratio);
        self.quad_rate_ratio = self.quad_rate_ratio.max(rhs.quad_rate_ratio);
    }
}

/// Both channels, together with whatever the producing method could say about its own error in
/// `ψ`.  Convergence is a question about `ψ` alone, so `dψ/dt` carries a value and no verdict.
struct Terms {
    value: [Complex64; 2],
    residual: f64,
}

/// A jet at one order, with whether the series ended on the mathematics or on the max order.
struct Pass {
    terms: Terms,
    reached: usize,
    spent: bool,
    /// Met the bar it needed to cross.
    settled: bool,
    cost: Cost,
}

/// What a jet had to say, together with the last order that improved the sum.  Everything above
/// that order is the divergent tail and belongs to nobody.
struct Expansion {
    terms: Terms,
    reached: usize,
    cost: Cost,
    /// Nothing left to expand
    exhausted: bool,
}

/// Both channels plus the spend that produced them.
struct Priced {
    terms: Terms,
    cost: Cost,
}

/// One traced level.  The sum it came to, how far along the path it actually got, and whether
/// that was the whole reach it was asked for.  A walk that ran out of valley early owes an
/// endpoint charge at the span it reached rather than the reach it was handed.
struct Level {
    resid: [Complex64; 2],
    spans: [f64; 2],
    full: bool,
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
    /// What each saddle's own method could say about its error in `ψ`, summed over the
    /// decomposition.
    pub residual: f64,
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
            residual: 0.0,
        }
    }

    pub fn reference(shape: Shape) -> Self {
        Self::new(shape, 0.0, true)
    }

    pub fn standard(shape: Shape) -> Self {
        Self::new(shape, 1e-7, false)
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

        // Sizes are read off `ψ` alone, since `ψ` is the channel the accuracy is owed to.
        let mut amps = [0.0_f64; MAX_G];
        for &i in active {
            amps[i] = saddles[i].amplitude(frame, scales[i][0]);
        }

        let dominant = active
            .iter()
            .map(|&i| amps[i] * weights[i].abs())
            .fold(0.0_f64, f64::max);

        let mut psi_re: Accumulator<f64> = Accumulator::default();
        let mut psi_im: Accumulator<f64> = Accumulator::default();
        let mut d_re: Accumulator<f64> = Accumulator::default();
        let mut d_im: Accumulator<f64> = Accumulator::default();
        let mut residual = 0.0_f64;

        let mut normal = Normal::default();
        let mut cost = Cost::default();

        for &i in active {
            let s = &saddles[i];
            let w = weights[i];
            let seen = singulants.from(i);
            let scale = scales[i];

            // The bar is `rel` of the largest contribution any saddle makes to `ψ`, divided back
            // through this saddle's weight because everything in here is still pre-weight.
            let target = rel * dominant / w.abs();

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
            // digits of its own.  This is what saddle `i` owes in its own units.
            let local_bar = (amps[i] / target).ln();

            // The skip below is really the jet's own opinion that a saddle is too small to
            // bother with, read off its leading Watson term.  Reference mode has no jet, so it
            // owes that opinion nothing: every active saddle gets integrated, cold, by
            // quadrature, and stands or falls on what quadrature itself finds.
            if !self.quadrature_only && local_bar <= 1.0 {
                residual += amps[i] * w.abs();
                continue;
            }

            let terms = if self.quadrature_only {
                let handoff = Handoff {
                    normal: None,
                    degree: 0,
                    moments: [Complex64::default(); 2],
                    inner: 0.0,
                };
                // Nothing was subtracted, so the path owes the whole saddle in both the density
                // it needs and the distance it walks.
                let quad = s.quadrature(frame, local_bar, rel, scale, target, seen, &handoff);
                cost += quad.cost;
                quad.terms
            } else {
                let mut jet = s.jet(frame, scale, target, r_star, local_bar, &mut normal);
                cost += jet.cost;

                // A missed bar buys more series before it buys any nodes, but only where the
                // first pass stopped on the search rather than on its own turn.  A pass that
                // watched its terms turn or already reached the cap has nothing further to sell,
                // and asking again reproduces it exactly.
                if jet.terms.residual > target && !jet.exhausted {
                    let deeper = s.deepen(frame, scale, target, r_star, &mut normal);
                    cost += deeper.cost;
                    jet = deeper;
                }

                if jet.terms.residual > target {
                    // Every term of the polynomial peaks at `√k` in the standardized coordinate,
                    // so a walk that stops short of `√degree` subtracts a shape whose own mass
                    // sits outside it while the moments add that mass back over the whole line.
                    // The degree stops where the path's footprint stops and the moments follow
                    // the same truncation.
                    let reach = reach_for(local_bar);
                    let degree_cap = (reach * reach) as usize;
                    // `degree` counts coefficients of `x` and the moments count orders, so the
                    // even half is what puts back exactly the polynomial the path took out.
                    let degree = (2 * jet.reached).min(degree_cap) & !1;

                    let width = trust_width(degree, rel);
                    normal.extend(s, frame, width);
                    let inner = if width >= trust_orders(rel) + degree {
                        SERIES_TRUST * branch_wall(seen, beta)
                    } else {
                        0.0
                    };

                    let handoff = Handoff {
                        normal: Some(&normal),
                        degree,
                        moments: normal.moments(frame.beta, degree / 2),
                        inner,
                    };

                    let quad = s.quadrature(frame, local_bar, rel, scale, target, seen, &handoff);
                    cost += quad.cost;

                    let moved = (quad.terms.value[0] - jet.terms.value[0]).norm();
                    cost.quad_gain = cost.quad_gain.max(moved / target);

                    quad.terms
                } else {
                    jet.terms
                }
            };

            residual += terms.residual * w.abs();
            let pv = terms.value[0] * w;
            let qv = terms.value[1] * w * Complex64::i();
            psi_re.add(pv.re);
            psi_im.add(pv.im);
            d_re.add(qv.re);
            d_im.add(qv.im);
        }

        let unscale = log_scale.exp() / rho;
        let psi = Complex64::new(psi_re.sum(), psi_im.sum()) * unscale;
        let d = Complex64::new(d_re.sum(), d_im.sum()) * unscale;
        let residual = residual * unscale;

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

/// The steepest-descent curvature at a saddle, carrying the orientation the traced contour is
/// walked in.
#[derive(Clone, Copy, Default)]
struct Descent(Complex64);

impl Descent {
    fn new(raw: Complex64) -> Self {
        let flip = if raw.re.abs() > DESCENT_AXIS_TIE * raw.norm() {
            raw.re < 0.0
        } else {
            raw.im < 0.0
        };
        Self(if flip { -raw } else { raw })
    }
}

/// A root of the saddle condition, carrying the numbers every later step reads off it.
#[derive(Clone, Copy, Default)]
struct Saddle {
    u: Complex64,
    b: Complex64,
    phi: Complex64,
    /// The curvature at the saddle, branch fixed so that `x` runs up the valley the contour
    /// arrives on.
    s0: Descent,
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

        Saddle {
            u,
            b,
            phi,
            s0: Descent::new((-(b * (1.0 - gamma) - gamma)).sqrt()),
            bp1: (b + 1.0) / gamma,
            gi,
        }
    }

    /// The cheap approach to a saddle, Watson's lemma on the series `Normal` carries.
    ///
    /// The order is predicted before the first coefficient exists, since the singulant `r_star`
    /// says where the series turns and the bar says where its terms fall under notice.  A
    /// prediction that runs short is doubled until it holds or reaches the cap, and each retry
    /// costs only the orders the series did not already have.
    fn jet(
        &self,
        frame: &Frame,
        scales: [Complex64; 2],
        target: f64,
        r_star: f64,
        bar: f64,
        normal: &mut Normal,
    ) -> Expansion {
        normal.seed(self, frame);

        let n_cap = cap_for(r_star);
        let mut cost = Cost {
            jet_saddles: 1,
            ..Default::default()
        };

        // A series whose own floor sits under its own bar is handing its polynomial to
        // quadrature no matter how far it climbs.  There is nothing to search for, so build it
        // once at the cap and let the caller trace the rest.
        let mut n_max = if r_star <= bar {
            n_cap
        } else {
            predict_order(bar, r_star).clamp(JET_SETTLE_FLOOR, n_cap)
        };

        loop {
            normal.extend(self, frame, 2 * n_max);
            let pass = self.jet_sum(frame, scales, target, n_max, true, normal);
            cost += pass.cost;
            if pass.settled || pass.spent || n_max == n_cap {
                return Expansion {
                    terms: pass.terms,
                    reached: pass.reached,
                    exhausted: pass.spent || n_max == n_cap,
                    cost,
                };
            }
            n_max = (2 * n_max).min(n_cap);
        }
    }

    /// Everything the series still has, spent before any node is.  An order is one convolution
    /// against coefficients that already exist, while a node is a Newton solve down a valley
    /// that may have to be walked twice, so a saddle that missed its bar asks the series for the
    /// rest before it asks the path for anything.  Often the rest is enough and no path is
    /// traced.  When it is not, the deeper polynomial is what the path gets to stand on and the
    /// shortfall it has to cover is smaller by whatever those orders bought.
    fn deepen(
        &self,
        frame: &Frame,
        scales: [Complex64; 2],
        target: f64,
        r_star: f64,
        normal: &mut Normal,
    ) -> Expansion {
        let n_cap = cap_for(r_star);
        normal.extend(self, frame, 2 * n_cap);
        let pass = self.jet_sum(frame, scales, target, n_cap, false, normal);
        Expansion {
            terms: pass.terms,
            reached: pass.reached,
            cost: pass.cost,
            exhausted: true,
        }
    }

    /// The series at one order.  Each even coefficient meets its Gaussian moment, and the two
    /// channels differ only in which `h` they read.
    ///
    /// The envelope of the `ψ` terms is what decides where the sum ends.  `dψ/dt` is accumulated
    /// over exactly the orders `ψ` accepted, so it neither prolongs the sum nor cuts it short.
    /// Ending on the mathematics, having diverged or crossed under the bar or run into roundoff,
    /// means the series has said everything it is going to say.
    fn jet_sum(
        &self,
        frame: &Frame,
        scales: [Complex64; 2],
        target: f64,
        n_max: usize,
        stop_at_bar: bool,
        normal: &Normal,
    ) -> Pass {
        let beta = frame.beta;
        let scale_abs = scales[0].norm();

        let mut moment = (TAU / beta).sqrt();
        let mut running = [Complex64::default(); 2];
        let mut last = 0.0_f64;

        // The envelope decides where the series turned and the accepted term is what the series
        // still owes.  An order only becomes a truncation when the envelope improved, which
        // means both it and its predecessor are small, so the accepted term is never a zero of
        // the oscillation standing in for the remainder.
        let mut best = [Complex64::default(); 2];
        let mut best_env = f64::INFINITY;
        let mut best_term = f64::INFINITY;
        let mut reached = 0usize;
        let mut descents = 0usize;

        let mut spent = false;
        let mut met = false;
        let mut rises = 0u32;

        for n in 0..=n_max {
            let term = [normal.h[0][2 * n] * moment, normal.h[1][2 * n] * moment];
            let size = scale_abs * term[0].norm();

            // A conjugate pair of singulants makes the terms oscillate, which is forced at
            // `τ = 0` where every `Δ` is imaginary, so a single small term is a zero of the
            // oscillation rather than the remainder.  The envelope is what decays.
            let env = size.max(last);
            last = size;

            running[0] += term[0];
            running[1] += term[1];

            if env < best_env {
                best_env = env;
                best_term = size;
                best = running;
                reached = n;
                rises = 0;
                descents += 1;

                // Meeting the bar ends a pass that was sent to find the bar, once the envelope
                // has descended far enough to be an envelope rather than a slow zero of the
                // oscillation.  A pass building the shape a path will stand on notes the
                // crossing and keeps climbing, since the orders above are ones the path does not
                // have to resolve.  Roundoff ends both and answers to no floor, since a sum that
                // has lost its bits gains nothing from more terms.
                if env <= target && descents >= JET_SETTLE_FLOOR {
                    met = true;
                    if stop_at_bar {
                        break;
                    }
                }

                if env <= EPS * scale_abs * running[0].norm() {
                    spent = true;
                    met = true;
                    break;
                }
            } else {
                rises += 1;
                // Terms oscillate and alias within an envelope, so the turn is confirmed over
                // several orders before the tail is believed.
                if rises >= TURN_CONFIRM {
                    spent = true;
                    break;
                }
            }

            moment *= (2 * n + 1) as f64 / beta;
        }

        let value = [scales[0] * best[0], scales[1] * best[1]];
        Pass {
            terms: Terms {
                // No channel can claim more digits than the partial sum carries, and the traced
                // path is held to the same floor.
                residual: best_term.max(EPS * value[0].norm()),
                value,
            },
            reached,
            settled: met,
            spent,
            cost: Cost {
                jet_passes: 1,
                jet_orders: reached as u32,
                jet_depth: reached as u32,
                ..Default::default()
            },
        }
    }

    /// The expensive approach to a saddle, a trapezoid along its traced steepest-descent path.
    ///
    /// Node density is a question about the neighbors and nothing else.  Aliasing folds in
    /// whatever feature sits `gap` off the path, carrying the size that feature already has, and
    /// the trapezoid buries it as `e^{-2π·gap·density}`.  The rate in that exponent is geometry
    /// and the model knows it.  The amplitude in front of it is the thing the model only guesses
    /// at, and it is the guess that goes wrong wherever two saddles are close to trading places.
    ///
    /// So the amplitude gets measured instead.  A coarse anchor level is placed where the feature
    /// is still standing in the answer, and the difference between it and a finer level names the
    /// amplitude the model was standing in for.  Each further level is placed where the last
    /// difference says the bar is met, and once two differences exist the rate becomes measured
    /// too.  Any of those levels may already be the whole answer, so the passes that calibrate
    /// the model are the same passes that can retire it.
    ///
    /// The path is one contour and it carries no `m`, so the placement is decided by what `ψ`
    /// still owes and `dψ/dt` is read off the same nodes.
    ///
    /// How far the path walks is a different question with a different answer.  The reach belongs
    /// to the bar alone, and where a neighboring saddle sits on this path the trace runs out of
    /// valley and stops on its own.  What it never reached is charged to the endpoint and
    /// reported, never handed to the placement, since no density buys back a piece of contour
    /// that was never walked.
    fn quadrature(
        &self,
        frame: &Frame,
        bar: f64,
        rel: f64,
        scales: [Complex64; 2],
        target: f64,
        seen: &[Complex64],
        hand: &Handoff,
    ) -> Priced {
        // Each neighbor is a square-root branch point of the inverse map at `x_c = √(-2Δ/β)`.
        // Its distance off the path is the aliasing gap the density answers to, and its modulus
        // is where the map's own series stops converging.  A neighbor sitting below this one
        // with equal `Im Φ` puts that point on the path itself, which is the configuration a
        // trace runs out of valley in.

        // Each neighbor is a square-root branch point of the inverse map at `x_c = √(-2Δ/β)`,
        // and what the density answers to is how far that point sits off the path.  One standing
        // above this saddle enters amplified and charges full freight, while one already
        // suppressed is credited only partly, since the fold point sits nearer than the
        // neighbor's own center and sees less of that decay.
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
        // Aliasing folds the neighbor onto the path carrying the size the neighbor already has,
        // and a polynomial is entire, so subtracting one leaves that fold exactly where it was.
        // Density answers to the whole saddle no matter how well the series did.  So does the
        // distance, since the moments put the polynomial back over the whole line and a walk
        // that stops early owes the difference.
        let reach = reach_for(bar);
        let model_rate = TAU * gap;
        let ceiling = (QUAD_MAX_NODES as f64 / (2.0 * reach)).max(QUAD_MIN_DENSITY);
        let ideal =
            (PLACE_MARGIN * (bar + lift).max(0.0) / model_rate).clamp(QUAD_MIN_DENSITY, ceiling);

        let value = |resid: [Complex64; 2], m: usize| scales[m] * (resid[m] + hand.moments[m]);
        let spread = |a: [Complex64; 2], b: [Complex64; 2]| (scales[0] * (a[0] - b[0])).norm();

        let mut cost = Cost::default();

        // The anchor is placed by the suppression and not by the answer.  A few e-folds of decay
        // is what makes the gap between two levels a measurement of the neighbor rather than a
        // measurement of the anchor's own coarseness, and an extrapolation off the flat stretch
        // below onset always lands short.  Where the answer itself sits below onset the anchor
        // keeps its place and the finer level is lifted to clear it.
        let onset = ANCHOR_EFOLDS / model_rate;
        let anchor_density = onset
            .max(CALIBRATE_NODES / reach)
            .clamp(QUAD_MIN_DENSITY, ceiling);
        let fine_density = ideal.max(PLACE_MARGIN * anchor_density).min(ceiling);

        let anchor = self.trapezoid(frame, reach, anchor_density, rel, hand);
        cost += anchor.cost;

        let mut level = self.trapezoid(frame, reach, fine_density, rel, hand);
        cost += level.cost;

        let mut lo = anchor_density;
        let mut hi = fine_density;
        let mut err_lo = spread(level.resid, anchor.resid);
        let mut rate = model_rate;
        let mut carried = f64::INFINITY;

        for _ in 0..PLACE_ROUNDS {
            // A difference between two levels bounds the error of the coarser one, so what the
            // finer still carries is that difference decayed across the step between them.  The
            // decay is derated, since the rate is the model's until a second difference exists
            // and a model that flatters the neighbor should not be allowed to certify on the
            // strength of that flattery.
            let fade = (-CERTIFY_DERATE * rate * (hi - lo)).exp();
            let bar_abs = target.max(EPS * value(level.resid, 0).norm());
            carried = err_lo * fade;
            let short = carried > bar_abs;
            let place = lo + (err_lo / bar_abs).max(1.0).ln() / rate;

            // A pair that has not yet reached onset is measuring itself, so its difference is no
            // evidence about either level and the loop buys another regardless of what the
            // projection claims.
            if !short && lo >= onset {
                break;
            }

            let next = (hi + PLACE_MARGIN * (place - hi)).min(ceiling);
            if next <= hi {
                break;
            }

            let step = self.trapezoid(frame, reach, next, rel, hand);
            cost += step.cost;
            let err_hi = spread(step.resid, level.resid);

            // Two consecutive levels name the suppression the path actually gets, which is the
            // number the model was standing in for.  A pair that failed to shrink has hit
            // roundoff and has nothing left to say, and a rate far from the model's has been
            // read off terrain the model does not describe, so both leave the job with the model
            // rather than let one measurement steer the whole placement.
            if err_hi > 0.0 && err_lo > err_hi {
                let measured = (err_lo / err_hi).ln() / (hi - lo);
                rate = measured.clamp(model_rate / RATE_CREDIBLE, model_rate * RATE_CREDIBLE);
            }

            lo = hi;
            hi = next;
            err_lo = err_hi;
            level = step;
        }

        cost.quad_density_ratio = hi / ideal;
        cost.quad_rate_ratio = rate / model_rate;

        let alias = carried;

        // Each side stopped where its own valley gave out, so the Gaussian is charged twice at
        // two different edges rather than once at the shorter of them.
        let width = (TAU / frame.beta).sqrt() / self.s0().norm();
        let tail = 0.5
            * width
            * level
                .spans
                .iter()
                .map(|s| (-0.5 * s * s).exp())
                .sum::<f64>();

        let value = [value(level.resid, 0), value(level.resid, 1)];
        let residual = (alias + tail * scales[0].norm()).max(EPS * value[0].norm());

        cost.quad_truncated = !level.full;
        cost.quad_short = residual > target.max(EPS * value[0].norm());

        Priced {
            terms: Terms { value, residual },
            cost,
        }
    }

    /// One level.  In the normal coordinate the phase is exactly `-x²/2`, so the weight is
    /// Gaussian by construction and cannot overflow the way a straight segment in `v` does once
    /// the reach is long enough for `e^{γv}` to climb the neighboring hill.  The path carries no
    /// `m`, so one trace serves both channels.
    ///
    /// Each level stands alone at its own density.  The inner nodes come from the map's own
    /// series and the outer ones are traced, and either side may run out of valley before the
    /// reach is spent, so the sum stops there and reports the span it actually covered, which is
    /// what the endpoint charge is built on.
    fn trapezoid(
        &self,
        frame: &Frame,
        reach: f64,
        density: f64,
        rel: f64,
        hand: &Handoff,
    ) -> Level {
        let beta = frame.beta;
        let h = 1.0 / (density * beta.sqrt());
        let n = (reach * density).round().max(1.0) as usize;

        // At the saddle `x` and `g'` vanish together and the slope `-x/g'(v)` goes to `1/s0`.
        let apex = (Complex64::default(), self.s0().inv());

        let mut acc = [
            (Accumulator::<f64>::default(), Accumulator::<f64>::default()),
            (Accumulator::<f64>::default(), Accumulator::<f64>::default()),
        ];

        let mut tally = |x: f64, f: [Complex64; 2]| {
            let gauss = (-0.5 * beta * x * x).exp();
            let f1 = gauss * f[0];
            let f2 = gauss * f[1];
            acc[0].0.add(f1.re);
            acc[0].1.add(f1.im);
            acc[1].0.add(f2.re);
            acc[1].1.add(f2.im);
        };

        // The node as a traced point gives it, the whole integrand less the whole subtracted
        // polynomial.  Past the trust radius this is the only form available.
        let traced = |x: f64, v: Complex64, slope: Complex64| {
            let ev = v.exp();
            [
                ev * slope - hand.poly(0, x),
                ev * ev * slope - hand.poly(1, x),
            ]
        };

        tally(
            0.0,
            hand.tail(0.0)
                .unwrap_or_else(|| traced(0.0, apex.0, apex.1)),
        );

        let mut walked = 0u32;
        let mut summed = 0u32;
        let mut spans = [0.0_f64; 2];
        let mut full = true;

        // Each side runs the summed stretch first and hands the path to Newton once, at the last
        // node the series placed.  Difficulty grows outward, so the handover sits where the walk
        // would have started climbing its split count anyway.
        for (which, side) in [1.0_f64, -1.0].into_iter().enumerate() {
            let mut at = apex;
            let mut splits = 1u32;
            let mut last = 0usize;
            let mut summing = true;
            for j in 1..=n {
                let x0 = side * (j - 1) as f64 * h;
                let x1 = side * j as f64 * h;

                if summing {
                    if let Some(f) = hand.tail(x1) {
                        tally(x1, f);
                        summed += 1;
                        last = j;
                        continue;
                    }
                    if let Some(state) = hand.seed(x0) {
                        at = state;
                        splits = HANDOVER_SPLITS;
                    }
                    summing = false;
                }

                match self.walk(at, x0, x1, rel, &mut splits) {
                    Some(next) => at = next,
                    None => break,
                }
                walked += 1;
                tally(x1, traced(x1, at.0, at.1));
                last = j;
            }
            full &= last == n;
            spans[which] = last as f64 / density;
        }
        drop(tally);

        Level {
            resid: [
                Complex64::new(acc[0].0.sum(), acc[0].1.sum()) * h,
                Complex64::new(acc[1].0.sum(), acc[1].1.sum()) * h,
            ],
            spans,
            full,
            cost: Cost {
                quad_paths: 1,
                quad_nodes: walked,
                quad_series: summed,
                ..Default::default()
            },
        }
    }

    /// One step along the path that `dv/dx = -x/g'(v)` predicts, closed by Newton on
    /// `g(v) = -x²/2`.
    ///
    /// Newton doubles its digits every pass, so a step that stops shrinking has found the floor
    /// the arithmetic allows rather than failing to arrive.  Both endings count as arrival while
    /// the last step is small.  A step that is still large means the prediction overshot into a
    /// neighboring valley and Newton landed on that saddle's sheet, and the caller is left to
    /// approach more slowly or to stop.
    fn advance(
        &self,
        v: Complex64,
        slope: Complex64,
        x0: f64,
        x1: f64,
        rel: f64,
    ) -> Option<(Complex64, Complex64)> {
        let h = x1 - x0;
        let mut w = v + slope * h;
        let target = Complex64::new(-0.5 * x1 * x1, 0.0);
        let step_tol = (NEWTON_TOL_FACTOR * rel).max(NEWTON_TOL_FLOOR);

        let mut arrived = false;
        let mut gp = Complex64::default();
        let mut prev = f64::INFINITY;
        for _ in 0..NEWTON_ITERS {
            let (gv, gpv) = self.g(w);
            let step = (gv - target) / gpv;
            w -= step;
            gp = gpv;

            let size = step.norm();
            if size < step_tol {
                arrived = true;
                break;
            }
            if size > 0.5 * prev {
                arrived = size < NEWTON_STALL;
                break;
            }
            prev = size;
        }

        if !arrived || (w - v).norm() > 4.0 * slope.norm() * h.abs() + 0.25 {
            return None;
        }
        Some((w, -x1 / gp))
    }

    /// Subdivides a step until the path stays in its own valley, and reports the give-up rather
    /// than handing back where it got to.  A node quietly placed short of its own `x` is worse
    /// than no node.
    ///
    /// Difficulty here is a property of the terrain and not of the node, and it only grows as
    /// the walk leaves the saddle and closes on a neighbor.  So the split count that worked last
    /// time is where the next node starts.  Climbing from one at every node pays the whole climb
    /// again along the stretch where the valley is tightest, which is exactly the stretch
    /// carrying the most nodes.
    fn walk(
        &self,
        from: (Complex64, Complex64),
        x0: f64,
        x1: f64,
        rel: f64,
        splits: &mut u32,
    ) -> Option<(Complex64, Complex64)> {
        loop {
            let sub = (x1 - x0) / *splits as f64;
            let mut at = from;
            let mut ok = true;
            for k in 1..=*splits {
                let a = x0 + sub * (k - 1) as f64;
                let b = x0 + sub * k as f64;
                match self.advance(at.0, at.1, a, b, rel) {
                    Some(next) => at = next,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return Some(at);
            }
            if *splits > 1 << TRACE_MAX_SPLIT {
                return None;
            }
            *splits *= 2;
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
        scale.norm() * (TAU / frame.beta).sqrt() / self.s0().norm()
    }

    /// The resolved curvature, the one number the whole normal coordinate stands on.
    fn s0(&self) -> Complex64 {
        self.s0.0
    }
}

/// The order the series is predicted to need, in the index the moments actually carry.  Terms go
/// like `Γ(n)/r*^n`, so a bar of `bar` e-folds asks for the `n` solving `n(1 + ln(r*/n)) = bar`.
/// The map is increasing and the seed sits above the root, so the iteration walks down onto it
/// and stopping early lands on the generous side.
fn predict_order(bar: f64, r_star: f64) -> usize {
    let mut n = bar.max(1.0);
    for _ in 0..3 {
        n = bar / ((r_star / n).max(1.0).ln() + 1.0);
    }
    n.ceil() as usize
}

/// Where the search is allowed to stop asking.  The singulant says where the terms turn, the
/// settle floor keeps a short singulant from capping the search under the order a bar can be
/// read at, and `JET_SLOTS` is the hard wall.
fn cap_for(r_star: f64) -> usize {
    ((JET_OVERSHOOT * r_star).ceil() as usize)
        .max(JET_SETTLE_FLOOR)
        .min(JET_ORDER / 2)
}

/// What the jet leaves behind for the path to stand on.  The series carries the shape the
/// integrand has near the saddle, so the trapezoid walks only what is left once that shape comes
/// out, and the moments of the same truncation put it back exactly.
///
/// The caller owes us a degree the path's own footprint covers, since a term of degree `k` peaks
/// at `√k` and one peaking outside the walk would be subtracted where the path is not and added
/// back over the whole line.
///
/// One degree serves both channels, since the truncation is a property of how far the path
/// walked and the path is the same curve for both.
struct Handoff<'a> {
    /// `None` means no series ran for this saddle at all, and the trapezoid should integrate the
    /// bare integrand with nothing subtracted, which is what reference mode asks for on every
    /// saddle.
    normal: Option<&'a Normal>,
    /// Degree in `x`, twice the order the series reached.  Unused when `normal` is `None`.
    degree: usize,
    /// The Gaussian integral of that polynomial, pre-scale.  Zero when `normal` is `None`.
    moments: [Complex64; 2],
    /// How far out the jet series places nodes on its own.
    inner: f64,
}

impl Handoff<'_> {
    fn poly(&self, m: usize, x: f64) -> Complex64 {
        let Some(normal) = self.normal else {
            return Complex64::default();
        };
        let c = &normal.h[m];
        let mut acc = c[self.degree];
        for k in (0..self.degree).rev() {
            acc = acc * x + c[k];
        }
        acc
    }

    /// Both channels inside the trust radius.  The integrand and the subtracted polynomial are
    /// the same series truncated at two places, so their difference is the coefficient tail and
    /// is formed as one.  The leading `x^{degree+1}` is the whole reason that difference is
    /// small, so it multiplies in once rather than being discovered by cancellation.
    fn tail(&self, x: f64) -> Option<[Complex64; 2]> {
        let normal = self.normal?;
        if x.abs() >= self.inner {
            return None;
        }
        let lead = x.powi(self.degree as i32 + 1);
        let mut out = [Complex64::default(); 2];
        for m in 0..2 {
            let c = &normal.h[m];
            let mut acc = c[normal.width];
            for k in (self.degree + 1..normal.width).rev() {
                acc = acc * x + c[k];
            }
            out[m] = acc * lead;
        }
        Some(out)
    }

    /// The state a walk needs to carry on from where the summed stretch stopped, the point on
    /// the path and the slope the next step is predicted with.
    fn seed(&self, x: f64) -> Option<(Complex64, Complex64)> {
        let normal = self.normal?;
        let mut v = normal.v[normal.width];
        let mut w = normal.w[normal.width];
        for k in (0..normal.width).rev() {
            v = v * x + normal.v[k];
            w = w * x + normal.w[k];
        }
        Some((v, w))
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
    /// `v(x)` itself, the antiderivative of `w` anchored at the saddle.
    v: [Complex64; JET_SLOTS],
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
            v: [Complex64::default(); JET_SLOTS],
            h: [[Complex64::default(); JET_SLOTS]; 2],
            width: 0,
        }
    }
}

impl Normal {
    /// Plants the series at the saddle.  Order one is the quadratic root of `y₁P₁ = -1`, whose
    /// branch is the one `s0` already committed to, and everything above it is linear.
    fn seed(&mut self, s: &Saddle, frame: &Frame) {
        let y1 = s.s0().inv();
        self.y[0] = Complex64::new(1.0, 0.0);
        self.y[1] = y1;
        self.yg[0] = Complex64::new(1.0, 0.0);
        self.yg[1] = y1 * frame.gamma;
        self.p[0] = Complex64::default();
        self.p[1] = -s.s0();
        self.w[0] = y1;
        self.v[0] = Complex64::default();
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

            self.y[k] = (self.y[k - 1] + mid - y1 * bp1 * tail) / ((k + 1) as f64 * s.s0());
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
            self.v[k] = self.w[k - 1] / k as f64;

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

    /// The Gaussian moments of the series truncated at order `n`, one per channel.  A path whose
    /// reach cannot cover the whole polynomial subtracts fewer orders than the jet reached, and
    /// what it adds back has to be the same truncation it took away.
    fn moments(&self, beta: f64, n: usize) -> [Complex64; 2] {
        let mut out = [Complex64::default(); 2];
        for m in 0..2 {
            let mut moment = (TAU / beta).sqrt();
            for k in 0..=n {
                out[m] += self.h[m][2 * k] * moment;
                moment *= (2 * k + 1) as f64 / beta;
            }
        }
        out
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
}

impl StokesTable {
    fn build(frame: &Frame) -> Self {
        let g = frame.g;

        let mut roots = vec![Complex64::default(); TABLE_NODES * g].into_boxed_slice();
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

    /// Prints the march's own record across a span of `τ`, every root and not just the live
    /// ones, since which roots count as live is read off the very weights in question.  The
    /// integrality flag is the cheap tell.  Membership is a count of steepest-descent paths and
    /// has no business being fractional, so a row that fails it has found something the
    /// crossing test does not understand.
    #[cfg(test)]
    fn dump(&self, frame: &Frame, from: f64, to: f64) {
        println!("\n=== stokes march, tau {from:.3} to {to:.3} ===");
        print!("  {:>6} {:>8} |", "node", "tau");
        for k in 0..self.g {
            print!(" {:>18}", format!("root {k}"));
        }
        println!("  {:>7}", "weights");

        for n in 0..self.s.len() {
            let tau = self.s[n].exp() - 1.0;
            if tau < from || tau > to {
                continue;
            }

            let row = &self.m[n * self.g..(n + 1) * self.g];
            let ragged = row.iter().any(|w| (w - w.round()).abs() > 1e-9);

            print!("  {n:>6} {tau:>8.4} |");
            for k in 0..self.g {
                let u = self.roots[n * self.g + k];
                let phi = Saddle::new(u, frame).phi;
                print!(" {:>8.4}{:+8.4}i", phi.re * frame.beta, phi.im * frame.beta);
            }
            print!("  ");
            for w in row {
                let live = if w.abs() > LIVE { '*' } else { ' ' };
                print!("{w:>5.2}{live}");
            }
            if ragged {
                print!("  RAGGED");
            }
            println!();
        }
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

/// How far a traced path walks, in the coordinate where the weight is exactly `e^{-x²/2}`.  Far
/// enough that the Gaussian has buried whatever the bar still cares about, and no farther, since
/// every node past that point is spent on digits nobody reads.
fn reach_for(bar: f64) -> f64 {
    (2.0 * (bar + REACH_MARGIN).max(0.0)).sqrt().min(REACH_MAX)
}

/// Where the inverse map `x ↦ v` stops converging.  Each neighbor puts a square-root branch
/// point at `x_c = √(-2Δ/β)`, since that is the `x` whose target depth `-x²/2` is the neighbor's
/// own, and the nearest of those moduli bounds the disk the series speaks for.
fn branch_wall(seen: &[Complex64], beta: f64) -> f64 {
    let mut wall = f64::INFINITY;
    for &d in seen {
        let r = (-2.0 * d / beta).sqrt().norm();
        if r > GAP_FLOOR {
            wall = wall.min(r);
        }
    }
    wall
}

/// How far past the subtracted polynomial the coefficient tail has to run.  Inside the trust
/// radius each further order buys a factor `SERIES_TRUST`, so the count is what drives that
/// geometric under the bar.
fn trust_orders(rel: f64) -> usize {
    (rel.recip().ln() / SERIES_TRUST.recip().ln()).ceil() as usize
}

/// The width the tail wants, clamped to the slots the series has.  A clamp that bites is the
/// caller's signal that the stretch has no evidence behind it.
fn trust_width(degree: usize, rel: f64) -> usize {
    (degree + trust_orders(rel)).min(JET_ORDER)
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
                18,
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

        // for k in 0..=4096 {
        //    let u = k as f64 * 0.00625 * 0.5;
        for k in 0..=32 {
            let u = k as f64 * 0.05 + 2.5;

            let reference = ref_jet.tap_at(u);
            let rc = reference.cost;
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

            // if k % 100 == 0 {
            {
                let c = standard.cost;
                // `T` is a trace that ran out of valley, `S` is a bar the placement never met.
                let flags = |truncated: bool, short: bool| match (truncated, short) {
                    (true, true) => "TS",
                    (true, false) => "T ",
                    (false, true) => " S",
                    (false, false) => "  ",
                };
                println!(
                    "  u: {u:>6.2}, err: {:>9}, res: {:>9}, time: {:>6}µs, jet: {:>2}s/{:>2}p/{:>3}d, quad: {:>2}p/{:>5}n+{:>5}s {}, ref: {:>2}p/{:>5}n {}, gain: {:>9}",
                    fmt_e(err),
                    fmt_e(standard.residual / standard.psi.norm()),
                    elapsed.as_micros(),
                    c.jet_saddles,
                    c.jet_passes,
                    c.jet_depth,
                    c.quad_paths,
                    c.quad_nodes,
                    c.quad_series,
                    flags(c.quad_truncated, c.quad_short),
                    rc.quad_paths,
                    rc.quad_nodes,
                    flags(rc.quad_truncated, rc.quad_short),
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
        assert!(worst < 1e-6);
    }

    #[test]
    fn stokes_membership_band() {
        let shape = Shape::from_q(3.5, 3.0);
        let frame = Frame::new(shape);
        let jet = QuadJet::standard(shape);

        // The band shows up in `u`, but the march lives in `τ`, and the two are related by
        // `τ = 2πu/β` once `ρ` cancels.  Widening past the observed edges catches a flip that
        // fires early and gets corrected late.
        let tau_of = |u: f64| TAU * u / frame.beta;
        jet.table.dump(&frame, tau_of(1.0), tau_of(4.0));
    }
}
