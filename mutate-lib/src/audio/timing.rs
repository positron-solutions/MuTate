//! # Timing
//!
//! Treating incoming discrete data as a virtual continuous input stream enables all upstreams and
//! downstreams to track as if following smooth processes.  Knowledge of the phase and jitter of new
//! data is necessary for this treatment to avoid underruns.  When done correctly, the linear
//! interpretation calculates a phase-aware target for where consumers should track in the input
//! data, enabling correctly smooth consumption with allowances for jitter and input quanta.
//!
//! Audio server period is usually a fixed quantity set by the quantum size and sample rate.  An
//! estimation of audio server phase completes the picture and enables frontends to tightly track
//! the incoming audio.  Instinct expects video to precede the audio, so reading with a phase-aware
//! helps achieve the desired happens-before relationship.
//!
//! By filtering chunk timings from the audio server, we can maintain a robust sense of phase.  This
//! module takes audio process callback timings and converts them into a robust phase estimation.
//! The implementation uses a Kalman filter and Gaussian approximation, which is modestly truthful
//! but most importantly yields a usefully stable phase estimation.

// NOTE This module really deserves some standardization if a crate doesn't already do this.  The
// ring-to-ring time alignment, phase tracking, and buffering goal math will all see a lot of use.
// NOTE Stability!  Stability!  Stability!  These estimators MUST not diverge under any input
// conditions or else the program may go nuts during a live show while the operators are reluctant
// to restart the program.  If the instability re-occurs on startup because of some particular
// initialization situation, game over man, game over!  All changes to work on stability, especially
// demonstrating stability or codifying it into types, are welcome changes!
// LIES Until the vanilla Kalman filter is augmented for the true distribution shape, this module
// expresses a Gaussian approximation.  Observed timing data on Linux was put on a histogram and the
// module produces a pretty stable phase grid, meaning outliers quickly revert to near normally
// distributed without upsetting the filter's phase estimate.  P99 is about 300micros.  Other
// platforms may bite us more.
// NEXT testing should include cases with big discontinuities, sudden bursts of noise, and
// multimodal timing distributions that will stress the Gaussian approximation (and prove the
// improvement of a follow-on scenting or particle solution).
// DEBT Audio rates
// NEXT audio servers can send us different chunk sizes, and supporting this requires changing the
// physical model for state update.  See blame.

use std::time::{Duration, Instant};

use ringbuf::traits::{Consumer, Observer, RingBuffer}; // Producer,

// ♻️ this period also shows up in requesting the pipewire latency
const PERIOD_BYTES: usize = 512 * 4 * 2;
const PERIOD_SAMPLES: f64 = 512.0;
const PERIOD_NS: f64 = PERIOD_SAMPLES * 1e9 / 48000.0;
const Q_DELTA: f64 = 1.0; // ns², period error diffusion per callback

const PHASE_PRIOR_NS: f64 = PERIOD_NS / 2.0; // diffuse phase prior std (±T/2)
const DRIFT_PRIOR_NS: f64 = 1000.0; // drift prior std, ns/callback
const R_PRIOR_NS: f64 = 40_000.0; // 40µs jitter prior in ns

// Re-seat gate: a squared-Mahalanobis miss this large means the prior can't explain the arrival —
// an underrun, pause/resume, or discontinuity. `nis` divides by `s`, which already folds in `p00`,
// so a relaxed (high-covariance) filter needs a *larger* raw miss to trip the gate. That's the
// reset-resistance: once phase starts locking and `p00` shrinks, the gate tightens; right after a
// re-seat `p00` is huge and the gate is deliberately hard to re-trip.
const RESEAT_NIS: f64 = 25.0; // about 5σ

// NOTE The phase offset μ is collapsed out of this type: `reference` absorbs the correction every
// cycle, so the stored phase is a structural zero and `reference` *is* the phase.  Regime changes
// don't reintroduce it — a re-seat writes the new phase straight into `reference` and blows the
// covariance back out, so a persisted phase field is never needed.
#[derive(Clone, Copy)]
struct Estimate {
    /// δ - period error / drift, ns per callback.
    period_error: f64,
    /// P - the error matrix, which is symmetric 2x2, so we can store instead as (0,0), (0,1), and
    /// (1,1) instead.
    covariance: [f64; 3],
}

impl Estimate {
    /// Project one cycle forward under the constant-drift model.  F = [[1, 1], [0, 1]]; δ⁻ = δ and
    /// the covariance telescopes and gains Q on the drift term.
    // NOTE The phase row of F (μ⁻ = μ + δ) is applied at the re-seat in `observe`, where the
    // projected correction `k0*nu + period_error` is folded into `reference` directly.  The phase
    // never lives here because it never persists across cycles — steady state re-seats a small
    // correction, a regime change re-seats a large one, and neither needs a stored phase row.
    fn project(self) -> Self {
        let [p00, p01, p11] = self.covariance;
        Self {
            period_error: self.period_error,
            covariance: [p00 + 2.0 * p01 + p11, p01 + p11, p11 + Q_DELTA],
        }
    }
}

/// Integrates successive audio chunk timings to predict phase alignment of deadlines downstream.
/// The implementation is a very basic Kalman filter using the
pub(crate) struct TimingFilter {
    /// Predicted arrival instance for the next chunk.  Advanced by one step each `observe`, folded
    /// toward each measurement by the Kalman correction.
    reference: Instant,
    /// R, a scalar due to the simple nature of the audio problem.
    observation_covariance: f64,
    /// The Kalman *prior*: the one-step projection for the cycle the next chunk will fall in.
    /// Produced once per `observe` from the freshly updated posterior, handed to the consumer as
    /// that call's `AudioTiming`, and reused verbatim as the prior on the next `observe`.  The
    /// projection is therefore computed exactly once per chunk instead of twice.
    prediction: Estimate,

    /// History of server chunk callback arrival times.  Only process calls that receive data will
    /// push measurements.
    // NEXT Pipewire can send different chunk sizes; supporting that means restoring a per-chunk
    // sample count here and scaling the physical model per callback instead of by PERIOD_SAMPLES.
    measurements: ringbuf::LocalRb<ringbuf::storage::Heap<Instant>>,
    /// Ring buffer of innovations for Mehra adaptive R estimation.
    innovations: ringbuf::LocalRb<ringbuf::storage::Heap<f64>>,
}

impl TimingFilter {
    pub(crate) fn new() -> Self {
        Self {
            reference: Instant::now() + Duration::from_nanos(PERIOD_NS as u64),
            observation_covariance: R_PRIOR_NS.powi(2),
            // Seed the prior by projecting the diffuse initial estimate one step, exactly as the
            // first observe() used to do at its top.
            prediction: Estimate {
                period_error: 0.0,
                covariance: [PHASE_PRIOR_NS.powi(2), 0.0, DRIFT_PRIOR_NS.powi(2)],
            }
            .project(),
            measurements: ringbuf::LocalRb::<ringbuf::storage::Heap<Instant>>::new(32),
            innovations: ringbuf::LocalRb::new(32),
        }
    }

    /// Re-seat the phase onto a fresh measurement after a discontinuity. The observed arrival
    /// *becomes* the phase (reference), covariance blows back out to the diffuse prior, drift is
    /// dropped (a discontinuity carries no trustworthy drift), and the innovation window is purged
    /// so dead-regime residuals can't poison Mehra R. This is the same shape as steady-state
    /// re-seating — write `reference`, don't persist a correction — just at regime-change scale.
    fn relatch(&mut self, arrived: Instant) -> AudioTiming {
        self.reference = arrived + Duration::from_nanos(PERIOD_NS as u64);
        self.observation_covariance = R_PRIOR_NS * R_PRIOR_NS;
        self.prediction = Estimate {
            period_error: 0.0,
            covariance: [PHASE_PRIOR_NS.powi(2), 0.0, DRIFT_PRIOR_NS.powi(2)],
        }
        .project();
        self.innovations.clear();

        let variance = self.prediction.covariance[0] + self.observation_covariance;
        AudioTiming {
            next: self.reference,
            variance,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        }
    }

    /// `arrived` is the instant recorded at the start of the process callback in pipewire.
    /// `written` is bytes written only used for a debug assert.
    pub(crate) fn observe(&mut self, arrived: Instant, written: usize) -> AudioTiming {
        debug_assert_eq!(written, PERIOD_BYTES);
        let r = signed_delta_ns(arrived, self.reference);

        // Prior for this cycle: the projection stored at the end of the previous observe.
        let prior = self.prediction;
        let [p00, p01, p11] = prior.covariance;

        // Innovation against the prior.  The stored phase is a structural zero (the correction is
        // re-seated into `reference` each cycle), so the innovation is the raw offset from the
        // reference — wrapped into (-T/2, +T/2] so a wrong-cycle prediction reads as the small
        // residual it really is.  A miss too large to wrap is handled upstream by the re-seat gate,
        // never here.
        let nu = r;
        let s = p00 + self.observation_covariance;

        // Re-seat gate. `nis` is the squared Mahalanobis distance of this raw (unwrapped) miss
        // under the prior. `s` carries `p00`, so confidence *is* the gate: only a certain filter
        // (small `s`) turns a large miss into a large `nis` and re-seats; an uncertain filter
        // (large `s` — freshly started or freshly relatched) sees the same miss as modest and stays
        // in the linear regime, letting Kalman/Mehra pull it in. So a confident filter that sees
        // the stream jump a full period declares a discontinuity and relatches, while an unsure
        // filter never does — the anti-thrash property, with no wrap logic and no `T/2` fold point.
        //
        // Consequence, accepted by design: in a locked filter a genuine single-cycle slip is
        // indistinguishable from a discontinuity, and both re-seat. In this domain a confident
        // cycle-slip *is* a discontinuity — the stream moved — and the relatch is cheap (diffuse
        // variance out), so a false re-seat costs one relaxed cycle, never corruption.
        //
        // Gated *before* any state mutation so a bad cycle leaves no trace in the innovation window
        // or covariance. Right after a relatch `s` is diffuse, so the gate is hard to re-trip for a
        // few cycles — the same confidence mechanism, now buying reset-resistance for free.
        if nu * nu / s > RESEAT_NIS {
            self.measurements.push_overwrite(arrived);
            return self.relatch(arrived);
        }

        // XXX why are we recording these timings?
        self.measurements.push_overwrite(arrived);
        // Mehra adaptive R from the innovation window.
        self.innovations.push_overwrite(nu);
        let n = self.innovations.occupied_len() as f64;

        // C₀ must be the innovation variance about its mean. E[ν²] folds in phase bias
        // ν̄², and a lagging filter has ν̄ ≠ 0 — that bias² would inflate R, shrink the
        // gain, deepen the lag, inflate R again. Centering makes R invariant to bias, so
        // tracking error can no longer masquerade as observation noise.
        let mean = self.innovations.iter().sum::<f64>() / n;
        let innovation_variance = self
            .innovations
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;

        // Mehra needs a populated window, and R must stay in a band where the gain can
        // neither die (blow-up) nor vanish (collapse-to-zero). Bounds are the stability
        // floor until the particle scheme lands.
        const R_MIN: f64 = R_PRIOR_NS * R_PRIOR_NS * 0.01;
        const R_MAX: f64 = R_PRIOR_NS * R_PRIOR_NS * 100.0;

        // Pre-gate vestige: `nis > 9.0` was the *implicit* outlier guard before the re-seat gate
        // existed.  Now it only freezes Mehra R adaptation on a bad cycle; the outlier role moved
        // upstream to the gate.  Goes away once the gate fully owns anomaly handling.
        let nis = nu * nu / s;
        self.observation_covariance = if n < 8.0 || nis > 9.0 {
            R_PRIOR_NS * R_PRIOR_NS
        } else {
            (innovation_variance - p00).clamp(R_MIN, R_MAX)
        };

        // Kalman gain (uses the pre-update R via `s`).
        let k0 = p00 / s;
        let k1 = p01 / s;

        // Posterior phase correction for this cycle.  Born from this innovation, consumed at the
        // re-seat below, never stored — which is why it's a local and not an `Estimate` field.  The
        // gate's reset branch is the same shape at larger scale: a re-seat of `reference`, not a
        // persisted correction — so it lives in `relatch`, not here.
        let phase_correction = k0 * nu;

        // Posterior for this cycle.
        let posterior = Estimate {
            period_error: prior.period_error + k1 * nu,
            covariance: [(1.0 - k0) * p00, (1.0 - k0) * p01, p11 - k1 * p01],
        };

        // Project once.  This single result is both the prior for the next observe and this
        // call's outgoing timing — no second projection anywhere.
        self.prediction = posterior.project();

        let variance = self.prediction.covariance[0] + self.observation_covariance;

        // Advance the running reference one step under the current drift model, then absorb the
        // fresh phase correction.  Re-seating the correction into the reference is what keeps the
        // state near zero and removes any fixed anchor: `reference` *is* the predicted arrival.
        // The projected phase advance μ⁻ = μ + δ is reconstructed here from the local correction
        // plus drift, since `project()` no longer carries the phase row.
        let step = PERIOD_NS + self.prediction.period_error + phase_correction;
        self.reference = add_signed_ns(self.reference, step);

        let out = AudioTiming {
            next: self.reference,
            variance,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        };

        // DEBT tracing!!!
        // `reference` is the predicted next arrival; measure the lead from this chunk's arrival.
        // Signed: a late chunk can leave the next prediction already behind us.
        let until_next_us = signed_delta_ns(self.reference, arrived) / 1000.0;
        println!(
            "until_next={until_next_us:+.0}µs stddev={:?}",
            Duration::from_nanos(variance.sqrt().round() as u64),
        );
        out
    }
}

/// Data about a connection's phase, period, jitter, and time between chunks.
#[derive(Clone, Copy, Debug)]
pub struct AudioTiming {
    /// Process time of the next mean prediction.  Using shape and scale parameters of the
    /// uncertainty distribution, consumers can use this to predict how much tracking leeway will
    /// provide a desired probability of underrun.
    pub next: Instant,
    /// The uncertainty around the next prediction in ns².  This folds up both process uncertainty
    /// and our uncertainty about the true phase.  If uncertainty is high, the consumer should track
    /// behind farther.
    pub variance: f64,
    /// Duration of each phase in nanoseconds.
    pub period_ns: f64,
    /// Data samples per period.  Use this to estimate the data velocity in time.
    pub period_samples: f64,
}

impl AudioTiming {
    pub(crate) fn new() -> Self {
        AudioTiming {
            next: Instant::now() + Duration::from_nanos(PERIOD_NS as u64),
            variance: PHASE_PRIOR_NS.powi(2) + DRIFT_PRIOR_NS.powi(2) + R_PRIOR_NS.powi(2),
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        }
    }
}

// Signed nanosecond delta between two Instants, underflow-safe (Instant subtraction panics on
// negative).  Sign is load-bearing: a late chunk yields a negative lead in the trace, and the
// re-seat gate keys off the magnitude of the signed miss.
// XXX this is probably some sloppy bullshit, but will be dealt with by reconciling the overall type
// graph.
fn signed_delta_ns(a: Instant, b: Instant) -> f64 {
    if a >= b {
        (a - b).as_nanos() as f64
    } else {
        -((b - a).as_nanos() as f64)
    }
}

// XXX type graph
fn add_signed_ns(t: Instant, ns: f64) -> Instant {
    let ns = ns.round();
    if ns >= 0.0 {
        t + Duration::from_nanos(ns as u64)
    } else {
        t - Duration::from_nanos((-ns) as u64)
    }
}
