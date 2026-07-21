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
//! downstreams to track as if following smooth processes **even if they tick on independent
//! clocks**.  Knowledge of the phase and jitter of new data is necessary to decide how closely
//! downstreams can follow the linear interpretation while still avoiding underruns when chunks are
//! late.
//!
//! Audio server period is usually a fixed quantity set by the quantum size and sample rate.  An
//! estimation of audio server phase completes the picture and enables frontends to tightly track
//! the incoming audio.  Instinct expects video to precede the audio, so reading in a phase-aware
//! manner helps achieve the desired happens-before relationship.

// NOTE This module really deserves some standardization if a crate doesn't already do this.  The
// ring-to-ring time alignment, phase tracking, and buffering goal math will all see a lot of use.
// Expect advantages such as flexibility for read-write heads and less coercion (and bugs) by the
// consumers.  Combined variance strategies may also develop.
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

use ringbuf::traits::{Consumer, Observer, RingBuffer};

// ♻️ this period also shows up in requesting the pipewire latency
const PERIOD_BYTES: u64 = 512 * 4 * 2;
const PERIOD_SAMPLES: f64 = 512.0;
const PERIOD_NS: f64 = PERIOD_SAMPLES * 1e9 / 48000.0;

const PHASE_PRIOR_NS: f64 = PERIOD_NS / 2.0; // diffuse phase prior std (±T/2)
const DRIFT_PRIOR_NS: f64 = 10_000.0; // drift prior std, 10µs/callback
const R_PRIOR_NS: f64 = 1_000_000.0; // 1ms jitter prior in ns

/// Timing snapshot published to consumers.  Includes enough information for consumers to slew their
/// tracking on the input stream with the desired protection from underruns.
///
/// Use by first obtaining the expected write head less safety margin at some moment in time when
/// read should be configured.
///
/// ```ignore
/// let now = Instant::now();
/// let goal_now = timing.virtual_read_goal(now); // write head + virtually received - safety margin
/// ```
///
/// Now just interpret `goal_now` back to the physical ring being read:
///
/// - The goal will usually be ahead by one read quantum size.  The consumer should slew towards
///   this relationship.  Slew according to the underrun or producer blocking risk.
/// - If the goal is ahead of what the ring physically holds, an underrun is in progress and you should choose a fallback
///   advance method such as half of the remaining data.
/// - If the data has lapped the goal, a burst is in progress and you should drop approximately half
///   of the stored data to unblock the producer and begin reading again.
///
/// Tune these decisions to the physical buffer this timing data describes and the pipeline depth of
/// your consumers.
// XXX very little of this needs to be public...
#[derive(Clone, Copy, Debug)]
pub struct AudioTiming {
    /// Process time of the next mean prediction.  Used to anchor the clock grid and its predictions
    /// for this snapshot.
    pub next: Instant,
    /// Samples written for this timing snapshot.  The writer may diverge near chunk delivery, but
    /// this timing data will still provide an accurate prediction for how many samples are
    /// virtually received at any moment in time.
    pub written: u64,
    /// Duration of each phase in nanoseconds.
    pub period_ns: f64,
    /// Data samples per period.  Used to estimate the data velocity in time.
    pub period_samples: f64,
    /// Combined uncertainty in the next tick, including process, measurement, and state
    /// uncertainty.
    pub variance: f64,
}

impl AudioTiming {
    pub(crate) fn new() -> Self {
        Self {
            next: Instant::now() + Duration::from_nanos(PERIOD_NS as u64),
            written: 0,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
            variance: PHASE_PRIOR_NS.powi(2) + DRIFT_PRIOR_NS.powi(2) + R_PRIOR_NS.powi(2),
        }
    }

    /// Returns the maximum smooth advance for the consumer.  `on_time` is the probability that
    /// jitter will not result in underrun.  Use values such as 0.999 for a three nines on-time
    /// rate.  The consumer should understand their own linearized read rate and slew their reads
    /// towards the read goal.
    pub fn virtual_read_goal(&self, read_instant: Instant, on_time: f64) -> u64 {
        let sample_velocity = self.period_samples / self.period_ns;
        let lead_ns = signed_delta_ns(read_instant, self.next);
        let head_delta = lead_ns * sample_velocity;
        // NOTE here's where the normal approximation and on_time tolerance convert to extra buffer
        // time.  If the normal distribution is swapped out, it must be swapped out here too.
        let offset_samples =
            self.period_samples + z_one_sided(on_time) * self.variance.sqrt() * sample_velocity;
        let adjust = (head_delta - offset_samples).floor();
        self.written.saturating_add_signed(adjust as i64)
    }
}

const R_MIN: f64 = R_PRIOR_NS * R_PRIOR_NS * 0.01; // floor at 100µs stddev
const R_MAX: f64 = R_PRIOR_NS * R_PRIOR_NS * 100.0;
const Q_DELTA: f64 = 1.0; // ns², period error diffusion per callback

// Re-seat gate threshold on squared Mahalanobis distance (~5σ). Larger miss is interpreted as
// underrun / pause / discontinuity).  Because NIS divides by s (which folds in p00), the gate
// self-tightens as phase locks and loosens right after a re-seat.
const RESEAT_NIS: f64 = 36.0; // about 6σ
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
    /// Number of samples written.  Used to align expected writes with timing snapshots.
    written: u64,
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
            written: 0,
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
            written: self.written,
            next: self.reference,
            variance,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        }
    }

    /// `arrived` is the instant recorded at the start of the process callback in pipewire.
    /// `written` is bytes written only used for a debug assert.
    pub(crate) fn observe(&mut self, arrived: Instant, written: u64) -> AudioTiming {
        debug_assert_eq!(written, PERIOD_BYTES);
        self.written += written / (4 * 2);
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
            // println!("innovation gate triggered!");
            return self.relatch(arrived);
        }

        // Trust-band decision against the same floored scale the gate uses, not the tight
        // pre-update s.  After a regime change s is stale-tight, which inflates nis and freezes R
        // for the whole transition — the filter then carries stale variance for a full window
        // length.  Ensuring the noise scale can't collapse below tolerance lets a sustained shift
        // admit to the window promptly instead of being read as a run of outliers.
        let trust_s = s.max(R_PRIOR_NS * R_PRIOR_NS);
        let nis = nu * nu / trust_s;
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
            written: self.written,
            next: self.reference,
            variance,
            period_ns: PERIOD_NS,
            period_samples: PERIOD_SAMPLES,
        };

        // DEBT tracing!!!
        // `reference` is the predicted next arrival.  `Arrived` can be noisy.
        // let until_next_micros = signed_delta_ns(self.reference, arrived) / 1000.0;
        // println!(
        //     "R={:.2}µs nu={:+.0}µs until_next={until_next_micros:+.0}µs iv={:.0}µs gate={:.2} ",
        //     self.observation_covariance.sqrt() / 1000.0,
        //     nu / 1000.0,
        //     innovation_variance.sqrt() / 1000.0,
        //     nu * nu / s.max(R_PRIOR_NS * R_PRIOR_NS),
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

/// One-sided normal quantile: z such that Φ(z) = p. Producer lateness is one-tailed, so this is
/// the number of stddevs of margin for an `p` probability the data has arrived.
// XXX Pure 🤖.  Would prefer to lean on a crate implementation since we don't care enough to check
// this.  While thinking about it, the lack of a clean inverse CDF is why the normal distribution is
// only a good choice when it is a true account of the uncertainty.  For timing data, it's not.  All
// my homies hate the normal distribution.
fn z_one_sided(p: f64) -> f64 {
    // Acklam's rational approximation to the inverse normal CDF.
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    let pl = 0.02425;
    if p < pl {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - pl {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_converges_across_discontinuity() {
        // This tests contains several accounting and sanity checks.  It contains a discontinuity in
        // order to trigger re-latch and verify that we still get a clean grid estimation after the
        // discontinuity (no filter explosions etc).
        fn drive(
            filter: &mut TimingFilter,
            clock: Instant,
            written: u64,
            remaining: u64,
        ) -> (AudioTiming, Instant, u64) {
            let timing = filter.observe(clock, PERIOD_BYTES);
            let next = clock + Duration::from_nanos(PERIOD_NS as u64);
            if remaining == 0 {
                (timing, next, written + PERIOD_BYTES)
            } else {
                drive(filter, next, written + PERIOD_BYTES, remaining - 1)
            }
        }

        let mut filter = TimingFilter::new();
        let start = Instant::now() + Duration::from_nanos(PERIOD_NS as u64);

        let (before, resume_clock, written) = drive(&mut filter, start, 0, 200);
        let (settled, discontinuity_clock, written) = drive(&mut filter, resume_clock, written, 20);

        let jumped = discontinuity_clock + Duration::from_nanos((PERIOD_NS * 50.0) as u64);
        let relatched = filter.observe(jumped, PERIOD_BYTES);
        let written = written + PERIOD_BYTES; // count the discontinuity observe, test-side
        let recover_clock = jumped + Duration::from_nanos(PERIOD_NS as u64);

        let (after, final_clock, total_written) = drive(&mut filter, recover_clock, written, 200);

        assert_eq!(after.written, total_written / (4 * 2));
        assert!(settled.variance < before.variance);
        assert!(relatched.variance > settled.variance);
        assert!(after.variance < relatched.variance);

        let goal = after.virtual_read_goal(final_clock, 0.999);
        let stddev_samples = after.variance.sqrt() * (PERIOD_SAMPLES / PERIOD_NS);
        let slack = (z_one_sided(0.999) * stddev_samples).ceil() as u64 + PERIOD_SAMPLES as u64;

        println!("goal: {}, slack: {}", goal, slack);
        assert!(goal + slack >= total_written / (4 * 2));
        assert!(goal <= total_written / (4 * 2));
    }

    #[test]
    fn variance_latches_tighter_under_lower_noise() {
        // Deterministic sub-gate jitter for noise-response tests.  splitmix64-ish; we only need
        // reproducible, zero-mean-ish perturbations, not statistical quality.
        fn jitter_ns(seed: &mut u64, amplitude_ns: f64) -> f64 {
            *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // map to [-1, 1)
            let unit = (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            unit * amplitude_ns
        }

        // Drive on-grid arrivals perturbed by bounded, zero-centered jitter.  Each step advances
        // the nominal clock by exactly PERIOD_NS; the jitter is applied only to the *arrival*
        // instant handed to observe(), never accumulated into the grid, so the filter sees pure
        // measurement noise and no drift.  Amplitudes stay inside the reseat gate so this exercises
        // Mehra R-adaptation, not the discontinuity path.
        fn drive_noisy(
            filter: &mut TimingFilter,
            mut clock: Instant,
            seed: &mut u64,
            amplitude_ns: f64,
            steps: u64,
        ) -> AudioTiming {
            let mut last = filter.observe(clock, PERIOD_BYTES);
            for _ in 0..steps {
                clock += Duration::from_nanos(PERIOD_NS as u64);
                let arrived = add_signed_ns(clock, jitter_ns(seed, amplitude_ns));
                last = filter.observe(arrived, PERIOD_BYTES);
            }
            last
        }

        let mut filter = TimingFilter::new();
        let mut seed = 0x1234_5678_9ABC_DEF0;
        let start = Instant::now() + Duration::from_nanos(PERIOD_NS as u64);

        // High-noise region: ~900µs jitter, near the 1ms tolerance ceiling but inside the reseat
        // gate.  Long enough to fill the 32-wide innovation window several times so R settles to
        // the observed noise rather than the prior.
        let high = drive_noisy(&mut filter, start, &mut seed, 900_000.0, 400);
        let high_clock = start + Duration::from_nanos((PERIOD_NS as u64) * 401);

        // Low-noise region: ~90µs jitter, same dwell — a well-scheduled system.  R should collapse
        // roughly 10× toward the smaller floor.
        let low = drive_noisy(&mut filter, high_clock, &mut seed, 90_000.0, 400);

        println!(
            "high.variance={:.3e} (stddev {:.0}ns), low.variance={:.3e} (stddev {:.0}ns)",
            high.variance,
            high.variance.sqrt(),
            low.variance,
            low.variance.sqrt(),
        );

        // The core claim: the published uncertainty band tracks input noise.  When jitter drops
        // 10×, the filter must re-latch to a materially tighter variance.
        assert!(
            low.variance < high.variance,
            "low-noise variance {:.3e} should be below high-noise {:.3e}",
            low.variance,
            high.variance,
        );

        // The consumer-facing slack must track the same pattern.  virtual_read_goal holds back
        // z·σ·velocity samples below the write head; evaluating each region's goal at its own
        // `next` instant zeroes the lead term (head_delta ≈ 0), leaving the offset as pure
        // variance-driven margin.  Louder input → larger held-back slack.
        let on_time = 0.999;
        let velocity = PERIOD_SAMPLES / PERIOD_NS;

        let high_slack = high.written - high.virtual_read_goal(high.next, on_time);
        let low_slack = low.written - low.virtual_read_goal(low.next, on_time);

        println!("high_slack={high_slack} samples, low_slack={low_slack} samples");

        // Same ordering as the variance itself: tighter latch ⇒ smaller protective margin.
        assert!(
            low_slack < high_slack,
            "low-noise slack {low_slack} should be below high-noise {high_slack}",
        );
    }
}
