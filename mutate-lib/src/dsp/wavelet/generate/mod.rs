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
//!
//! Because the mother wavelet will only be sampled over a fixed number of periods, there is one
//! normalization.  The expression `1.0 / (2 * periods)` appears in several places and normalizes
//! sample magnitude for the higher pitch of the wavelet necessary to fit the specified `periods`
//! within the IFFT output.  The inverse Fourier integral matches this normalization to create
//! regularity across the module.

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

    #[cfg(feature = "validate")]
    #[test]
    fn multi_cross_reference() {
        use std::time::{Duration, Instant};

        let shape = Shape::from_q(3.5, 3.0);
        let beta = shape.beta;

        let settings = ifft::IfftSettings {
            periods: 12,
            ..ifft::IfftSettings::default()
        };

        let ifft_nominal_taps = 1000;
        let ifft_start = Instant::now();
        let (psi, _, _) = ifft::morse_half_taps(shape, settings);
        let ifft_us_per_tap = ifft_start.elapsed().as_secs_f64() * 1e6 / ifft_nominal_taps as f64;

        // let ref_jet = quadjet::QuadJet::reference(shape);
        let ref_jet = quadjet::QuadJet::standard(shape);

        let res = settings.resolution as f64;
        let stride = settings.resolution / 4;
        let last = (10.0 * res) as usize;

        let digits = |x: f64| if x <= 1e-17 { 17.0 } else { -x.log10() };

        // Every pair, so no method sits in the denominator of its own score.  `f` is IFFT, `c` is
        // deformed contour, `j` is the jet.  If the two references agree with each other better
        // than any of them agrees with `j`, the jet is the outlier; if `j-c` tracks `f-c`, the jet
        // is inside the reference noise.
        println!("\n=== pairwise agreement, beta {beta:3.2} ===");
        println!(
            "  {:>6} | {:>9} | {:>5} {:>5} {:>5} | {:>5} | {:>5}",
            "u", "scale", "f-c", "j-c", "j-f", "pred", "jpred"
        );

        let mut worst_jc: f64 = 17.0;
        let mut worst_jc_u: f64 = 0.0;
        let mut worst_jf: f64 = 17.0;
        let mut worst_jf_u: f64 = 0.0;
        let mut worst_gap: f64 = 17.0;
        let mut worst_gap_u: f64 = 0.0;

        let mut contour_time = Duration::ZERO;
        let mut jet_time = Duration::ZERO;
        let mut taps: u64 = 0;

        for k in (0..=last).step_by(stride) {
            let u = k as f64 / res;

            let ifft = psi[k];

            let t = Instant::now();
            let contour = contour::tap_at(shape, u);
            contour_time += t.elapsed();

            let t = Instant::now();
            let jet = ref_jet.tap_at(u);
            jet_time += t.elapsed();

            taps += 1;

            let scale = ifft.norm().max(contour.value.norm());
            let d = |a: Complex64, b: Complex64| digits((a - b).norm() / scale);

            let fc = d(ifft, contour.value);
            let jc = d(jet.psi, contour.value);
            let jf = d(jet.psi, ifft);
            let pred = digits(contour.residual / scale);
            let jpred = digits(jet.residual[0] / scale);

            // How much worse the jet's best pairing is than the integrators' best pairing.
            let gap = jc.max(jf) - fc;

            if jc < worst_jc {
                worst_jc = jc;
                worst_jc_u = u;
            }

            if jf < worst_jf {
                worst_jf = jf;
                worst_jf_u = u;
            }

            if gap > 0.0 && 17.0 - gap < worst_gap {
                worst_gap = 17.0 - gap;
                worst_gap_u = u;
            }

            println!(
                "  {u:>6.3} | {:>9} | {fc:>5.1} {jc:>5.1} {jf:>5.1} \
                | {pred:>5.1} | {jpred:>5.1}",
                fmt_e(scale),
            );
        }

        println!("  worst jet/ifft: {worst_jf:.1} digits at u {worst_jf_u:.3}");
        println!("  worst jet/contour: {worst_jc:.1} digits at u {worst_jc_u:.3}");
        println!(
            "  largest jet deficit: {:.1} digits at u {worst_gap_u:.3}",
            17.0 - worst_gap
        );

        let contour_us_per_tap = contour_time.as_secs_f64() * 1e6 / taps as f64;
        let jet_us_per_tap = jet_time.as_secs_f64() * 1e6 / taps as f64;

        println!("\n=== per-tap cost ===");
        println!("  {:>18} | {:>8}", "method", "us/tap");
        println!(
            "  {:>18} | {:>8.3}",
            format!("ifft ({ifft_nominal_taps})"),
            ifft_us_per_tap
        );
        println!("  {:>18} | {:>8.3}", "contour", contour_us_per_tap);
        println!("  {:>18} | {:>8.3}", "jet", jet_us_per_tap);

        assert!(worst_jc > 5.5);
    }
}
