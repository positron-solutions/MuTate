// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Generate Wavelets
//!
//! > I am incensed that the features I require have been so thoroughly neglected only so that
//! > this utter slop could be rushed into the hands of mere others who inconsiderately do not share
//! > my specific requirements, and I will hold accountable the selfish cretins responsible for what
//! > yesterday I did not know that I must have.
//! >
//! > - Ttang Kong
//!
//! High resolution mother wavelets from which our highly tuned but fundamentally rough N-tap
//! filters are made.  The moments and shape that the stencil and tapering solution attempts to
//! restore are first measured from these high-fidelity inputs, so we try to make them good.
//!
//! Ideally both quick and accurate, but perhaps split between debug and release or offline baking
//! where tradeoffs must be made.  Downstream uses `f64` for several steps, but the stencil itself
//! washes away *unbiased* noise during the reduction to `N` taps.  The final shape-aware rounding
//! to `f32` forgets any inaccuracy accumulated below its own precision.  The final word is **avoid
//! bias.**
//!
//! ## The Implementations
//!
//! Three implementations are contained.  We use them to corroborate the wavelet engineering and
//! verify the precision convergence of our approaches.  The requirement is to establish convergence
//! over the `f32` filter performance within the application.  If numerical precision no longer
//! limits our filter, we can focus on improving the filter we are approximating in N taps.
//!
//! - **IFFT** - the standard baseline.  Precision is flat and relative error climbs at high `u`.
//! - **Deformed Contour** - a reference implementation
//! - **QuadJet** - the production implementation, precise, fast, and accurate, choose three.
//!
//! All generators evaluate the same one-sided inverse transform in the same
//! units. `samples_per_period` (`S`) resolves the carrier into samples and *is* the 𝛚 in carrier
//! radians per sample.  The output taps are `ψ/S` and `d/S²`.  The dilation step, where the
//! daughter wavelet sample omega is known, applies both.
//!
//! ## Conventions
//!
//! - As seen elsewhere, `𝐮` is the normalized time coordinate, `𝛚𝐭`, periods at the wavelet's
//!   angular velocity.
//! - peak-2 normalization, which affects the IFFT evaluation.
//! - Every generator differentiates in `u` and reports the derivative channels with a quarter turn
//!   divided out, so `d = -i dψ/du` and `dd = -i dd(d)/du`.
//!
//! Because the mother wavelet will only be sampled over a fixed number of periods, there is one
//! normalization.  The expression `1.0 / (2 * periods)` appears in several places and normalizes
//! sample magnitude for the higher pitch of the wavelet necessary to fit the specified `periods`
//! within the IFFT output.  The inverse Fourier integral matches this normalization to create
//! regularity across the module.
//!
//! Rotating `d` (the `-i` factor) allows folded storage.  `ψ(-u) = conj ψ(u)`, and the honest
//! derivative carries a sign instead, so removing the turn puts every channel under a single mirror
//! rule `f(-u) = conj f(u)` and one half-length record with one reflection serves all three.  It
//! also keeps the spectral weights real, since each successive channel picks up `ζ` rather than
//! `iζ` and rides the same cosine and sine the channel below it did.  Consumers wanting a true
//! slope for Hermite interpolation supply the `i`, which on the device is a lane swap and a negate.

#[cfg(feature = "validate")]
pub(crate) mod contour;
#[cfg(feature = "validate")]
pub(crate) mod ifft;

pub(crate) mod hermite;
pub(crate) mod quadjet;

use num_complex::Complex64;

use crate::dsp::wavelet::whatsleft::Accumulator;
use crate::dsp::wavelet::Shape;

// NOTE The doc comments are getting really dense.  While it's fun to play educator, I'd like to
// focus on getting the nomenclature to presume the reader is familiar with a **good** model for
// these problems and focus on getting across the choices in place for key conventions where most
// people choose one or the other formalism.
// NEXT  Self-evaluation precision convergence tests are eacty that.  DFT tests are the ultimate
// discriminator and only DFT can verify accurate wavelet generation.   One may yet bet an
// imposter. 🥷🏿
// NOTE We're interested in bias, not extra precision.  We can barely measure bias any more
// accurately with extra precision, but a large bias, we have plenty of precision to measure.  Bias
// will show up even after we squeeze the result through a stencil.  Noise will just get washed away
// in the stencil and f32 truncation.
// NEXT Did not compare any other FFT libraries, just went with stock standard.

fn fmt_e(x: f64) -> String {
    let s = format!("{x:+.2e}");
    // split "±m.mme±dd" into mantissa and exponent, then zero-pad the exponent
    let (mantissa, exp) = s.split_once('e').unwrap_or(("999", "999"));
    let exp: i32 = exp.parse().unwrap();
    format!("{mantissa}e{exp:+03}")
}

#[cfg(test)]
mod test {
    use super::*;

    const TRUST_DIGITS: f64 = 5.5;

    // If bad > worst, worst = bad, location = u.
    struct Extreme {
        value: f64,
        u: f64,
        keep_min: bool,
    }

    impl Extreme {
        fn min() -> Self {
            Self {
                value: f64::INFINITY,
                u: 0.0,
                keep_min: true,
            }
        }
        fn max() -> Self {
            Self {
                value: f64::NEG_INFINITY,
                u: 0.0,
                keep_min: false,
            }
        }
        fn see(&mut self, value: f64, u: f64) {
            let better = if self.keep_min {
                value < self.value
            } else {
                value > self.value
            };
            if better {
                self.value = value;
                self.u = u;
            }
        }
    }

    fn digits(x: f64) -> f64 {
        if x <= 1e-17 {
            17.0
        } else {
            -x.log10()
        }
    }

    fn digits_complex(a: Complex64, b: Complex64, scale: f64) -> f64 {
        digits((a - b).norm() / scale)
    }

    #[cfg(feature = "validate")]
    #[test]
    fn multi_cross_reference_psi() {
        // We compare the standard jet to its quadrature-only reference behavior.  The reference is
        // then compared to contour and ifft.  Whichever of the three non-jet methods agrees better
        // is taken as the goal.  Standard jet sags versus the goal will be tracked as a precision
        // deficit and asserted on.  Worst pairs are reported, but it's known that IFFT at high u
        // sags and mid-u Contour sags vs IFFT and reference QuadJet.

        use std::time::{Duration, Instant};

        let shape = Shape::from_q(3.5, 3.0);
        let beta = shape.beta;

        let settings = ifft::IfftSettings {
            periods: 10,
            ..ifft::IfftSettings::default()
        };

        let ifft_nominal_taps = settings.periods * settings.resolution;
        let ifft_start = Instant::now();
        let (psi, _, _) = ifft::morse_half_taps(shape, settings);
        let ifft_us_per_tap = ifft_start.elapsed().as_secs_f64() * 1e6 / ifft_nominal_taps as f64;

        let ref_jet = quadjet::QuadJet::reference(shape);
        let std_jet = quadjet::QuadJet::standard(shape);

        let res = settings.resolution as f64;
        let stride = settings.resolution / 4;
        let last = (10.0 * res) as usize;

        let mut worst_qs_qr = Extreme::min();
        let mut worst_qr_c = Extreme::min();
        let mut worst_qr_i = Extreme::min();
        let mut worst_c_i = Extreme::min();
        let mut worst_qs_deficit = Extreme::max();

        let mut contour_time = Duration::ZERO;
        let mut ref_time = Duration::ZERO;
        let mut std_time = Duration::ZERO;
        let mut taps: u64 = 0;

        println!("\n=== pairwise agreement, beta {beta:3.2} ===");
        println!(
            "  {:>6} | {:>9} | {:>5} {:>5} {:>5} {:>5} | {:>5}",
            "u", "scale", "qs-qr", "qr-c", "qr-i", "c-i", "defct"
        );

        for k in (0..=last).step_by(stride) {
            let u = k as f64 / res;

            let ifft = psi[k];

            let t = Instant::now();
            let contour = contour::tap_at(shape, u);
            contour_time += t.elapsed();

            let t = Instant::now();
            let reference = ref_jet.tap_at(u);
            ref_time += t.elapsed();

            let t = Instant::now();
            let standard = std_jet.tap_at(u);
            std_time += t.elapsed();

            taps += 1;

            let scale = ifft.norm().max(contour.value.norm());

            let qs_qr = digits_complex(standard.psi, reference.psi, scale);
            let qr_c = digits_complex(reference.psi, contour.value, scale);
            let qr_i = digits_complex(reference.psi, ifft, scale);
            let c_i = digits_complex(contour.value, ifft, scale);

            worst_qs_qr.see(qs_qr, u);
            worst_qr_c.see(qr_c, u);
            worst_qr_i.see(qr_i, u);
            worst_c_i.see(c_i, u);

            // The best-agreeing pair among the three non-jet comparisons sets the bar. Below
            // TRUST_DIGITS none of the three is corroborated enough to trip, so the assertion on
            // jet's agreement with quadrature-only is leaned on in a separate assert.
            let goal = qr_c.max(qr_i).max(c_i);
            let deficit = if goal > TRUST_DIGITS {
                (goal - qs_qr).max(0.0)
            } else {
                0.0
            };
            worst_qs_deficit.see(deficit, u);

            println!(
                "  {u:>6.3} | {:>9} | {qs_qr:>5.1} {qr_c:>5.1} {qr_i:>5.1} {c_i:>5.1} | {deficit:>5.1}",
                fmt_e(scale),
            );
        }

        println!(
            "  worst qs-qr: {:.1} digits at u {:.3}",
            worst_qs_qr.value, worst_qs_qr.u
        );
        println!(
            "  worst qr-c:  {:.1} digits at u {:.3}",
            worst_qr_c.value, worst_qr_c.u
        );
        println!(
            "  worst qr-i:  {:.1} digits at u {:.3}",
            worst_qr_i.value, worst_qr_i.u
        );
        println!(
            "  worst c-i:   {:.1} digits at u {:.3}",
            worst_c_i.value, worst_c_i.u
        );
        println!(
            "  worst qs deficit: {:.1} digits at u {:.3}",
            worst_qs_deficit.value, worst_qs_deficit.u
        );

        let contour_us_per_tap = contour_time.as_secs_f64() * 1e6 / taps as f64;
        let ref_us_per_tap = ref_time.as_secs_f64() * 1e6 / taps as f64;
        let std_us_per_tap = std_time.as_secs_f64() * 1e6 / taps as f64;

        println!("\n=== per-tap cost ===");
        println!("  {:>22} | {:>8}", "method", "us/tap");
        println!(
            "  {:>22} | {:>8.3}",
            format!("ifft ({ifft_nominal_taps})"),
            ifft_us_per_tap
        );
        println!("  {:>22} | {:>8.3}", "contour", contour_us_per_tap);
        println!("  {:>22} | {:>8.3}", "quadjet (reference)", ref_us_per_tap);
        println!("  {:>22} | {:>8.3}", "quadjet (standard)", std_us_per_tap);

        assert!(worst_qr_c.value.max(worst_qr_i.value).max(worst_c_i.value) > 5.0);
        assert!(worst_qs_qr.value > 6.0);
        // Pull this up as the remaining dips go away.
        assert!(worst_qs_deficit.value < 8.0);
    }

    #[cfg(feature = "validate")]
    #[test]
    fn multi_cross_reference_d() {
        // `d` channel check.  The quadrature-only jet is the reference and IFFT is its independent
        // corroboration.  Where the two agree, the standard jet is expected to follow, and its sag
        // is the deficit.

        // use a slightly different q to smoke test without sweep.
        let shape = Shape::from_q(4.5, 3.0);
        let beta = shape.beta;

        let settings = ifft::IfftSettings {
            periods: 10,
            ..ifft::IfftSettings::default()
        };

        let (_, d, _) = ifft::morse_half_taps(shape, settings);

        let ref_jet = quadjet::QuadJet::reference(shape);
        let std_jet = quadjet::QuadJet::standard(shape);

        let res = settings.resolution as f64;
        let stride = settings.resolution / 4;
        // NOTE IFFT precision in `d` is likely falling off even faster than `psi`, so we just cap
        // the test over a reasonable range to smoke test the conventions.  To investigate more
        // thoroughly, implement `d` for the contour and add it to the test.
        let last = (5.0 * res) as usize;

        let mut worst_qs_qr = Extreme::min();
        let mut worst_qr_i = Extreme::min();
        let mut worst_qs_deficit = Extreme::max();

        println!("\n=== pairwise agreement on d, beta {beta:3.2} ===");
        println!(
            "  {:>6} | {:>9} | {:>5} {:>5} | {:>5}",
            "u", "scale", "qs-qr", "qr-i", "defct"
        );

        for k in (0..=last).step_by(stride) {
            let u = k as f64 / res;

            let ifft = d[k];
            let reference = ref_jet.tap_at(u);
            let standard = std_jet.tap_at(u);

            let scale = ifft.norm().max(reference.d.norm());

            let qs_qr = digits_complex(standard.d, reference.d, scale);
            let qr_i = digits_complex(reference.d, ifft, scale);

            worst_qs_qr.see(qs_qr, u);
            worst_qr_i.see(qr_i, u);

            // IFFT corroborating the reference sets the bar, and below TRUST_DIGITS nothing is
            // corroborated well enough to trip.
            let deficit = if qr_i > TRUST_DIGITS {
                (qr_i - qs_qr).max(0.0)
            } else {
                0.0
            };
            worst_qs_deficit.see(deficit, u);

            println!(
                "  {u:>6.3} | {:>9} | {qs_qr:>5.1} {qr_i:>5.1} | {deficit:>5.1}",
                fmt_e(scale),
            );
        }

        println!(
            "  worst qs-qr: {:.1} digits at u {:.3}",
            worst_qs_qr.value, worst_qs_qr.u
        );
        println!(
            "  worst qr-i:  {:.1} digits at u {:.3}",
            worst_qr_i.value, worst_qr_i.u
        );
        println!(
            "  worst qs deficit: {:.1} digits at u {:.3}",
            worst_qs_deficit.value, worst_qs_deficit.u
        );

        assert!(worst_qr_i.value > 9.0);
        assert!(worst_qs_qr.value > 7.0);
        // Pull this up as the remaining dips go away.
        assert!(worst_qs_deficit.value < 10.0);
    }

    #[cfg(feature = "validate")]
    #[test]
    fn omega_check() {
        const TAPS: usize = 128;

        let shape = Shape::from_q(3.5, 3.0);
        let beta = shape.beta;

        let settings = ifft::IfftSettings {
            periods: 10,
            ..ifft::IfftSettings::default()
        };
        let (psi, _, _) = ifft::morse_half_taps(shape, settings);

        let std_jet = quadjet::QuadJet::standard(shape);
        let res = settings.resolution as f64;

        // Both methods are read on the ifft grid so the two omegas share an abscissa.
        let taps: Vec<(Complex64, Complex64)> = (0..TAPS)
            .map(|t| (psi[t], std_jet.tap_at(t as f64 / res).psi))
            .collect();

        println!("\n=== omega, ifft versus standard jet ===");
        println!(
            "  {:>3} {:>7} | {:>13} {:>10} | {:>13} {:>10} | {:>10}",
            "t", "u", "omega i", "|psi| i", "omega qs", "|psi| qs", "d omega"
        );

        for t in 1..TAPS {
            let (ifft, ifft_prev) = (taps[t].0, taps[t - 1].0);
            let (std, std_prev) = (taps[t].1, taps[t - 1].1);

            // Modulus cancels out of the argument, so it is reported only as a conditioning number.
            let omega_i = (ifft * ifft_prev.conj()).arg();
            let omega_s = (std * std_prev.conj()).arg();

            println!(
                "  {t:>3} {:>7.4} | {omega_i:>+13.9} {:>10} | {omega_s:>+13.9} {:>10} | {:>+10.2e}",
                t as f64 / res,
                fmt_e(ifft.norm()),
                fmt_e(std.norm()),
                omega_s - omega_i,
            );
        }
    }
}
