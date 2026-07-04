//! # Timing
//!
//! An estimation of audio server delivery phase and duration enables reading audio closer to the
//! real time in which it will be played over the physical interface.  It is instinctual to expect
//! visual to precede audio, so reading closer to real time helps achieve the desired happens-before
//! relationship.
//!
//! This module takes audio process callback timings and converts them into a persistent phase and
//! duration estimation.  The implementation uses a Kalman filter.  The timing grid must match the
//! audio streaming rate, but its phase is slightly uncertain.  By filtering, we can maintain a
//! robust sense of phase.

// NEXT whenever playback stalls or jumps, we would prefer to generate several new particles to
// attempt to lock onto the new phase faster than the old filter can loosen up.  The new particles
// should be selected whenever the Bayes ratio suggests that their estimations are more likely
// accurate and precise than the old filter.  The old filter starts off looking relatively accurate,
// but after several predictions, the new filters will will have tightened up their covariance
// matrix closer to the true phase and they will be much more reliable than the old filter.

use std::time::Instant;

use ringbuf::traits::{Consumer, Observer, RingBuffer}; // Producer,

// ♻️ this period also shows up in requesting the pipewire latency
const NOMINAL_PERIOD_NS: f64 = 512.0 / 48000.0 * 1e9;
const Q_DELTA: f64 = 1.0; // ns², period error diffusion per callback

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

            // Start with large diagonal — diffuse prior.
            // p00: phase uncertainty ±T/2, p11: drift uncertainty ±100ns/callback
            observation_covariance: (40_000.0f64).powi(2), // 40µs jitter prior in ns²
            error_covariance: [
                (NOMINAL_PERIOD_NS / 2.0).powi(2), // p00
                0.0,                               // p01
                (100.0f64).powi(2),                // p11
            ],
            measurements: ringbuf::LocalRb::<ringbuf::storage::Heap<(Instant, usize)>>::new(32),
            innovations: ringbuf::LocalRb::new(32),
        }
    }

    pub(crate) fn observe(&mut self, arrived: Instant, written: usize) -> AudioTiming {
        self.measurements.push_overwrite((arrived, written));

        let [p00, p01, p11] = self.error_covariance;

        self.k += 1;
        let r = (arrived - self.t0).as_nanos() as f64 - self.k as f64 * NOMINAL_PERIOD_NS;
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

        AudioTiming {
            count: self.k,
            last: arrived,
        }
    }
}

/// Data about a connection's phase, period, jitter, and time between chunks.
#[derive(Clone, Copy)]
pub struct AudioTiming {
    /// Number of samples this connection has seen.
    pub count: u64,
    /// Last chunk received timing.  Decided at the beginning of the process callback and only
    /// updated if we actually manage to write data.
    pub last: Instant,
}

impl AudioTiming {
    pub(crate) fn new() -> Self {
        AudioTiming {
            count: 0,
            last: Instant::now(),
        }
    }
}
