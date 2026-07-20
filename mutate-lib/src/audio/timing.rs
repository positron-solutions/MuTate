//! # Timing
//!
//! Estimates the phase of an audio server's periodic process callbacks so downstreams can
//! track the input as a continuous stream without underrunning. Callback arrival instants are
//! filtered into a phase estimate that predicts the next chunk's arrival plus an uncertainty
//! band for jitter allowance.
//!
//! Implementation: a 2-state Kalman filter (phase, drift) under a constant-drift model, with
//! Mehra adaptive measurement-noise (R) estimation and a Mahalanobis gate that re-seats phase
//! on discontinuities. Noise is treated as Gaussian.
//!
//! ## Motivation
//!
//! Treating incoming discrete data as a virtual continuous input stream enables all upstreams and
//! downstreams to track as if following smooth processes.  Knowledge of the phase and jitter of new
//! data is necessary to decide how closely downstreams can follow the linear interpretation while
//! still avoiding underruns when chunks are late
//!
//! Audio server period is usually a fixed quantity set by the quantum size and sample rate.  An
//! estimation of audio server phase completes the picture and enables frontends to tightly track
//! the incoming audio.  Instinct expects video to precede the audio, so reading with a phase-aware
//! helps achieve the desired happens-before relationship.

// NOTE This module really deserves some standardization if a crate doesn't already do this.  The
// ring-to-ring time alignment, phase tracking, and buffering goal math will all see a lot of use.
// NOTE Stability!  Stability!  Stability!  These estimators MUST not diverge under any input
// conditions or else the program may go nuts during a live show while the operators are reluctant
// to restart the program.  If the instability re-occurs on startup because of some particular
// initialization situation, game over man, game over!  All changes to work on stability, especially
// demonstrating stability or codifying it into types, are welcome changes!
// LIES This module implements a Gaussian approximation, but real timing data can sharply diverge
// and this is platform dependent!  A histogram of observed timing on a desktop Linux system
// demonstrated that a Gaussian approximation is very reasonable.  The estimated phase grid is
// pretty stable.  Outliers revert usually on the next sample.  P99 on Linux ≈ 300µs.
// NEXT testing should include cases with big discontinuities, sudden bursts of noise, and
// multimodal timing distributions that will stress the Gaussian approximation (and prove the
// improvement of a follow-on scenting or particle solution).
// DEBT Sample rates and channels.
// NEXT Assumes fixed chunk size, PERIOD_SAMPLES.  Other servers may send us variable or edge case
// chunk sizes, and handling this requires changing the physical model for state update.

use std::time::{Duration, Instant};

use ringbuf::traits::{Consumer, Observer, RingBuffer}; // Producer,

/// Published to consumers.  Includes right information for consumers to slew their tracking on the
/// input stream with the desired protection from underruns.
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

// ♻️ this period also shows up in requesting the pipewire latency
const PERIOD_BYTES: usize = 512 * 4 * 2;
const PERIOD_SAMPLES: f64 = 512.0;
const PERIOD_NS: f64 = PERIOD_SAMPLES * 1e9 / 48000.0;

const PHASE_PRIOR_NS: f64 = PERIOD_NS / 2.0; // diffuse phase prior std (±T/2)
const DRIFT_PRIOR_NS: f64 = 1000.0; // drift prior std, ns/callback
const R_PRIOR_NS: f64 = 40_000.0; // 40µs jitter prior in ns
const R_MIN: f64 = R_PRIOR_NS * R_PRIOR_NS * 0.01;
const R_MAX: f64 = R_PRIOR_NS * R_PRIOR_NS * 100.0;
const Q_DELTA: f64 = 1.0; // ns², period error diffusion per callback

// Re-seat gate threshold on squared Mahalanobis distance (~5σ). Larger miss is interpreted as
// underrun / pause / discontinuity).  Because NIS divides by s (which folds in p00), the gate
// self-tightens as phase locks and loosens right after a re-seat.
const RESEAT_NIS: f64 = 25.0; // about 5σ
const MEHRA_TRUST_NIS: f64 = 9.0; // ~3σ: admit to R-estimation window

/// Kalman prior.  Estimates are published for consumers, and we re-use them on each observation, so
/// they are persisted explicitly.
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
    fn project(self) -> Self {
        let [p00, p01, p11] = self.covariance;
        Self {
            period_error: self.period_error,
            covariance: [p00 + 2.0 * p01 + p11, p01 + p11, p11 + Q_DELTA],
        }
    }
}

/// Integrates successive audio chunk timings to provide phase estimations necessary for downstreams
/// to slew their read reads.
pub(crate) struct TimingFilter {
    /// Predicted arrival instance for the next chunk.  Advanced by one step each `observe`, folded
    /// toward each measurement by the Kalman correction.
    reference: Instant,
    /// R, a scalar due to the simple nature of the audio problem.
    observation_covariance: f64,
    /// Kalman prior for each next chunk.  Published for downstreams and then used verbatium on
    /// updates in `observer`.
    prediction: Estimate,
    /// Ring buffer of innovations for Mehra adaptive R estimation.
    innovations: ringbuf::LocalRb<ringbuf::storage::Heap<f64>>,
}

impl TimingFilter {
    pub(crate) fn new() -> Self {
        Self {
            reference: Instant::now() + Duration::from_nanos(PERIOD_NS as u64),
            observation_covariance: R_PRIOR_NS.powi(2),
            // XXX should we combine the seed logic?
            // Seed the prior by projecting the diffuse initial estimate one step, exactly as the
            // first observe() used to do at its top.
            prediction: Estimate {
                period_error: 0.0,
                covariance: [PHASE_PRIOR_NS.powi(2), 0.0, DRIFT_PRIOR_NS.powi(2)],
            }
            .project(),
            innovations: ringbuf::LocalRb::new(32),
        }
    }

    /// Handle playback discontinuities such as pause-resume cycles or delivery disruptions.
    /// Re-seat the phase onto a fresh measurement and relax the filter:
    ///
    /// - Observed arrival *becomes* the phase (reference).
    /// - Covariance blows back out to the diffuse prior.
    /// - Drift is dropped (a discontinuity carries no trustworthy drift).
    /// - The innovation window is purged so dead-regime residuals can't poison Mehra R.
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
        // Unpack previous cycle's projection.
        let prior = self.prediction;
        let [p00, p01, p11] = prior.covariance;

        // NOTE Innovation against the prior.  The stored phase is a structural zero (the correction is
        // re-seated into `reference` each cycle), so the residual is the innovationn for each
        // update.  Implausibly large misses are handled later by the re-seat gate.
        let nu = signed_delta_ns(arrived, self.reference);
        let s = p00 + self.observation_covariance;

        // NOTE If innovation is implausibly large, re-seat the grid and relax the filter.  The relaxed
        // `p00` will feed into `s`, preventing reset thrashing as the relaxed filter re-latches to
        // the new phase grid.  The gate is checked *before* any state mutation so a bad cycle
        // leaves no trace in the innovation window or covariance.
        // NOTE the innovation calculation above does *not* wrap to the nearest cycle.
        // Consequently, the filter will usually any tick closer to another expected tick as a
        // discontinuity and completely re-latch rather than presuming a tick appeared or
        // disappeared.
        if nu * nu / s > RESEAT_NIS {
            return self.relatch(arrived);
        }

        // Mehra adaptive R from the innovation window. Only innovations inside the 3σ trust band
        // admit to the window; 3σ–5σ residuals are folded into phase/ drift below but excluded here
        // so one fat sample can't set the noise floor for the next window-length cycles.
        let nis = nu * nu / s;
        if nis <= MEHRA_TRUST_NIS {
            self.innovations.push_overwrite(nu);
        }
        let n = self.innovations.occupied_len() as f64;

        // C₀ must be the innovation variance about its mean. E[ν²] folds in phase bias ν̄², and a
        // lagging filter has ν̄ ≠ 0 — that bias² would inflate R, shrink the gain, deepen the lag,
        // inflate R again. Centering makes R invariant to bias, so tracking error can no longer
        // masquerade as observation noise.
        let mean = self.innovations.iter().sum::<f64>() / n;
        let innovation_variance = self
            .innovations
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;

        // freeze Mehra R adaptation on a bad cycle.
        self.observation_covariance = if n < 8.0 || nis > MEHRA_TRUST_NIS {
            R_PRIOR_NS * R_PRIOR_NS
        } else {
            // R must stay in a band where the gain can neither blow up nor vanish.
            (innovation_variance - p00).clamp(R_MIN, R_MAX)
        };

        // Kalman gain (uses the pre-update R via `s`).
        let k0 = p00 / s;
        let k1 = p01 / s;

        // Posterior phase correction, k0·ν.  Phase isn't a stored state — it's re-seated into
        // `reference` every cycle — so this correction is a local, consumed at the `reference`
        // advance below and never persisted.  The drift correction k1·ν *does* persist, into
        // `period_error`.  relatch() is the same move at larger scale: re-seat `reference`, don't
        // store a correction.
        let phase_correction = k0 * nu;

        // Posterior for this cycle.
        let posterior = Estimate {
            period_error: prior.period_error + k1 * nu,
            covariance: [(1.0 - k0) * p00, (1.0 - k0) * p01, p11 - k1 * p01],
        };

        // Projection for next cycle and for publishing for downstream consumers.
        self.prediction = posterior.project();

        // Predictive variance of the next arrival: projected phase uncertainty (p00⁻) plus
        // observation noise (R).  Same shape as `s` above, one cycle ahead — this is the band
        // downstreams size their read leeway against.
        let variance = self.prediction.covariance[0] + self.observation_covariance;
        let step = PERIOD_NS + self.prediction.period_error + phase_correction;
        self.reference = add_signed_ns(self.reference, step);

        let out = AudioTiming {
            next: self.reference,
            variance,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        };

        // DEBT tracing!!!
        // `reference` is the predicted next arrival.  `Arrived` can be noisy.
        // let until_next_micros = signed_delta_ns(self.reference, arrived) / 1000.0;
        // println!(
        //     "until_next={until_next_micros:+.0}µs stddev={:?}",
        //     Duration::from_nanos(variance.sqrt().round() as u64),
        // );
        out
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
