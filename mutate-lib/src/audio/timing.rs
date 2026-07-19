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
// NEXT whenever playback stalls or jumps, we would prefer to generate several new particles to
// attempt to lock onto the new phase faster than the old filter can loosen up.  The new particles
// should be selected whenever the Bayes ratio suggests that their estimations are more likely
// accurate and precise than the old filter.  The old filter starts off looking relatively accurate,
// but after several predictions, the new filters will will have tightened up their covariance
// matrix closer to the true phase and they will be much more reliable than the old filter.
// DEBT Audio rates

use std::time::{Duration, Instant};

use ringbuf::traits::{Consumer, Observer, RingBuffer}; // Producer,

// ♻️ this period also shows up in requesting the pipewire latency
const PERIOD_SAMPLES: f64 = 512.0;
const PERIOD_NS: f64 = PERIOD_SAMPLES * 1e9 / 48000.0;
const Q_DELTA: f64 = 1.0; // ns², period error diffusion per callback

const PHASE_PRIOR_NS: f64 = PERIOD_NS / 2.0; // diffuse phase prior std (±T/2)
const DRIFT_PRIOR_NS: f64 = 1000.0; // drift prior std, ns/callback
const R_PRIOR_NS: f64 = 40_000.0; // 40µs jitter prior in ns

/// Integrates successive audio chunk timings to predict phase alignment of deadlines downstream.
/// The implementation is a very basic Kalman filter using the
pub(crate) struct TimingFilter {
    /// The epoch for counting cycles.  Cycles never go backwards in time, so we just set the epoch
    /// to begin when we create the filter.
    t0: Instant,
    /// Cycle count.  This can jump to account for missing chunks and always presumes that new
    /// chunks arrive strictly in order.
    k: u64,
    // NEXT this single-filter setup will be changed to a particle style where new particles are
    // spawned when the existing filter seems to go crazy.  If the new particle log-prob and
    // innovation drop below the old particle, we switch to the new particles.  Each particle will
    // need to be able to keep a short history of log probs.  When the Bayes ratio becomes
    // overwhelming, we switch particles and lock the new timing.
    /// Server's phase offset
    phase_offset: f64,
    /// Amount of drift
    period_error: f64,
    /// R, a scalar due to the simple nature of the audio problem.
    observation_covariance: f64,
    /// P, the error matrix, which is symmetric 2x2, so we can store instead as (0,0), (0,1), and
    /// (1,1) instead.
    error_covariance: [f64; 3],

    /// History of server chunk callback timing and chunk size.  Only process calls that receive
    /// data will push data.
    // XXX discontinuities are a particle spawn signal.
    // NEXT Pipewire can send us different chunk sizes, and supporting this requires changing the
    // physical model for state update.
    measurements: ringbuf::LocalRb<ringbuf::storage::Heap<(Instant, usize)>>,
    /// Ring buffer of innovations for Mehra adaptive R estimation.
    innovations: ringbuf::LocalRb<ringbuf::storage::Heap<f64>>,
}

impl TimingFilter {
    pub(crate) fn new() -> Self {
        Self {
            t0: Instant::now(),
            k: 0,
            phase_offset: 0.0,
            period_error: 0.0,
            observation_covariance: R_PRIOR_NS.powi(2),
            // [p00, p01, p11]
            error_covariance: [PHASE_PRIOR_NS.powi(2), 0.0, DRIFT_PRIOR_NS.powi(2)],
            measurements: ringbuf::LocalRb::<ringbuf::storage::Heap<(Instant, usize)>>::new(32),
            innovations: ringbuf::LocalRb::new(32),
        }
    }

    pub(crate) fn observe(&mut self, arrived: Instant, written: usize) -> AudioTiming {
        self.measurements.push_overwrite((arrived, written));

        let [p00, p01, p11] = self.error_covariance;

        self.k += 1;
        let r = (arrived - self.t0).as_nanos() as f64 - self.k as f64 * PERIOD_NS;
        let mu_pred = self.phase_offset + self.period_error;
        let delta_pred = self.period_error;

        // Prediction step
        let p00_pred = p00 + 2.0 * p01 + p11;
        let p01_pred = p01 + p11;
        let p11_pred = p11 + Q_DELTA;

        // Measure innovation
        let nu = r - mu_pred;
        let s = p00_pred + self.observation_covariance;

        self.innovations.push_overwrite(nu);
        let n = self.innovations.occupied_len() as f64;
        let innovation_variance = self.innovations.iter().map(|v| v * v).sum::<f64>() / n;
        // XXX can this become unstable or collapse to zero?
        self.observation_covariance = (innovation_variance - p00_pred).max(0.0);

        // Kalman gain
        let k0 = p00_pred / s;
        let k1 = p01_pred / s;

        // State updates including measurements
        self.phase_offset = mu_pred + k0 * nu;
        self.period_error = delta_pred + k1 * nu;

        // Covariance update
        self.error_covariance = [
            (1.0 - k0) * p00_pred,
            (1.0 - k0) * p01_pred,
            p11_pred - k1 * p01_pred,
        ];

        // NEXT we need to shift the prediction into state so that we can send the next prediction
        // to the user and then re-use that prediction when updating state on the next chunk.
        // Duplicate the prediction step, now from the *posterior*, to fill the outgoing
        // timing.  Deliberately not cached — the next observe() recomputes the identical
        // projection at its top from this same state.
        let [q00, q01, q11] = self.error_covariance;
        let mu_next = self.phase_offset + self.period_error;
        let p00_next = q00 + 2.0 * q01 + q11;
        let variance = p00_next + self.observation_covariance;

        // Absolute arrival predicted for cycle k+1, anchored to the same t0 grid as r.
        let next_ns = ((self.k + 1) as f64 * PERIOD_NS + mu_next).max(0.0).round() as u64;

        AudioTiming {
            next: self.t0 + Duration::from_nanos(next_ns),
            variance,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        }
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
