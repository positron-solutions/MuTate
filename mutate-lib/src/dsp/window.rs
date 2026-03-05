// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Window Functions
//!
//! This module contains window functions used in many filters, especially DFTs.

use std::f64::consts::{PI as PI64, TAU as TAU64};

use num_complex::Complex;

use crate::dsp;

/// ## Window Choice
///
/// There are many deceptive ideas an engineer may have about window selection.  The Dolph-Chebyshev
/// or its generalizations are the right answer, but let's explore why.
///
/// When we look at graphs of windows, we will note that windows like the Hamming and Triangle will
/// not reach peak output until the first samples of a pure tone get to the middle of the window and
/// begin to reach the higher weights.  We may longingly gaze at the high initial weights of the
/// `Boxcar`.  A high-Q DFT already has slow attack.  Why make it slower?
///
/// The answer is that waiting until the middle is actually faster than the `Boxcar`, which must
/// wait until samples begin reaching the very tail of the window due to its equal weighting
/// everywhere.  A continuous pure tone will only saturate a fraction of a `Boxcar`'s weights while
/// it will be making a lump-in-the-middle filter hum strongly.  Those narrow-lump filters will also
/// peak higher for tones shorter than the window length while the `Boxcar` will peak low and then
/// continue smearing the blip for the entire length of the window!
///
/// Worse yet, while we see more weight given to early samples in the `Boxcar`, we don't know if
/// they are signal or noise yet.  The response is only faster because we are willing to admit
/// terrible noise from the huge side lobes.  Not only are we blurring responses, but we don't even
/// know if they are the right pitch!  We might be smearing pure noise.
///
/// The engineer desperate to register short blips is tempted to lament the length and slow rise in
/// weights of many high-performance windows, but the blip will eventually enter the fat part of the
/// window, twice as fast as the `Boxcar`, and we will know that it is the target pitch, not some
/// unwanted side lobe  Because of the very high quality of the main lobes, we can listen to much
/// weaker tones.  A small light on a dark night appears twice as bright.  They were right about
/// choosing Shakuras.
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    /// Also known as the Rectangle.  Very bad, not good at all, -13.3dB first side-lobe.  When you
    /// account for needing to fill the entire window, response time is about double and in many
    /// cases the window never fills.  Smears everything.  Avoid.
    BoxCar,
    /// In theory, the high weights means you see a response from the first samples sooner, but just
    /// like the box car, the window takes forever to fill, and we don't know if the samples are
    /// noise because the first side lobes are still -21.3dB.
    Welch,
    /// Aka the Triangle.  When better meets uncomplicated.  -26.5dB PSLL.
    Bartlett,
    /// The hamming cancels its first side lobe to a modestly usable -42.7dB.  Possibly less precise
    /// in time resolution than some choices.
    Hamming,
    /// Optimal window for tuning noise floors and bandwidth tradeoffs in every way.  Combined with
    /// window length, this window offers the engineer maximum control and performance at every goal
    /// except possibly handling extremely high noise at extremely distant points, which can be
    /// readily suppressed with other simple zero-delay filters.
    DolphChebyshev { attenuation_db: f64 },
    // NEXT Kaiser, another tunable window with slightly better noise decay at extreme pitch
    // differences.  This can save an un-filtered DFT from registering a sudden super-loud pitch at
    // an unexpected, distant frequency.
    // Kaiser,
    // NEXT Ultraspherical, yet another tunable window.  This may offer narrower main lobes when
    // configured to have a positive slope of side lobes as we get farther form the main lobe.
    // Paired with another filter to suppress the side lobe growth, this may offer a more precise
    // main lobe.  Parameterized to 1, it *is* the Dolph-Chebyshev window.
    // Ultraspherical,
}

impl WindowFunction {
    pub fn make_window(&self, size: usize) -> Vec<f64> {
        match self {
            Self::BoxCar => bin_weights(&boxcar, size),
            Self::Bartlett => bin_weights(&bartlett, size),
            Self::Welch => bin_weights(&welch, size),
            Self::Hamming => bin_weights(&hamming, size),
            Self::DolphChebyshev { attenuation_db } => {
                dolph_chebyshev_window(size, *attenuation_db)
            }
        }
    }

    /// Windows are calculated in f64 for accuracy, but on the GPU, we use f32, so we have to use
    /// the truncated window to reproduce outcomes *in situ*.
    pub fn make_window_32(&self, size: usize) -> Vec<f32> {
        self.make_window(size)
            .into_iter()
            .map(|w| w as f32)
            .collect()
    }

    /// How often should this window be applied to yield a fresh result to satisfy COLA?  The  lower
    /// the repeat, the more often we will apply the window.  the more windows overlap, the more
    /// equally we measure any data.
    // Much was learned from this PDF:
    // https://holometer.fnal.gov/GH_FFT.pdf
    pub fn repeat(&self, length: usize) -> u32 {
        match self {
            Self::BoxCar => (length as f64 / 2.0).ceil() as u32,
            Self::Bartlett => (length as f64 / 2.0).ceil() as u32,
            Self::Welch => (length as f64 * 0.293).ceil() as u32,
            Self::Hamming => (length as f64 / 2.0).ceil() as u32,
            // MAYBE COLA values for Dolph Chebyshev need some experimental tuning.  Higher
            // attenuation was said to demand more overlap.  Workbench!
            Self::DolphChebyshev { attenuation_db } => (length as f64 / 4.0).ceil() as u32,
        }
    }

    /// Bandwidth normalization factor.  DFT effective Q is *proportionate* to N.  The window
    /// choice, including boxcar, has an effect on any way we measure the resulting bandwidth.  The
    /// goal of the BNF is to make most Q settings near 1.0 and to allow faster engineering when
    /// changing window choices.
    // XXX apply to results
    pub fn bandwidth_norm_factor(&self) -> f32 {
        match self {
            Self::BoxCar => 1.0,
            Self::Welch => 1.0,
            Self::Bartlett => 1.0,
            Self::Hamming => 1.0,
            Self::DolphChebyshev { attenuation_db } => 1.0,
        }
    }

    /// Gain normalization factor
    // XXX apply to results
    pub fn amplitude_norm_factor(&self) -> f32 {
        match self {
            Self::BoxCar => 1.0,
            Self::Welch => 1.0,
            Self::Bartlett => 1.0,
            Self::Hamming => 1.0,
            Self::DolphChebyshev { attenuation_db } => 1.0,
        }
    }
}

impl Default for WindowFunction {
    fn default() -> Self {
        Self::DolphChebyshev {
            attenuation_db: 40.0,
        }
    }
}

fn boxcar(_x: f64) -> f64 {
    1.0
}

fn welch(x: f64) -> f64 {
    let t = 2.0 * x - 1.0;
    1.0 - t * t
}

fn bartlett(x: f64) -> f64 {
    if x < 0.5 {
        2.0 * x
    } else {
        2.0 - (2.0 * x)
    }
}

// Our integration samples really close to both endpoints, but technically we're supposed to reach a
// specific toe value.  Anyway.  Better than boxcar.
fn hamming(x: f64) -> f64 {
    // Source Wikipedia and a robot 🤖
    const A0: f64 = 25.0 / 46.0;
    A0 - (1.0 - A0) * (2.0 * PI64 * x).cos()
}

/// Integrates discrete bin weights given a window_fn.  Will automatically normalize windows where
/// normalization in the window_fn is hard.
pub fn bin_weights(window_fn: &impl Fn(f64) -> f64, bins: usize) -> Vec<f64> {
    let samples_per_bin = 512;
    let mut weights = Vec::with_capacity(bins);

    for bin in 0..bins {
        let bin_start = bin as f64 / bins as f64;
        let bin_end = (bin + 1) as f64 / bins as f64;

        let mut sum = 0.0;
        let step = (bin_end - bin_start) / samples_per_bin as f64;
        for s in 0..samples_per_bin {
            let t = bin_start + (s as f64 + 0.5) * step;
            sum += window_fn(t);
        }
        weights.push(sum / samples_per_bin as f64);
    }
    if !weights.iter().find(|x| **x == 1.0).is_some() {
        let max = weights.iter().fold(0.0f64, |max, x| max.max(*x));
        let norm = 1.0 / max;
        weights.iter_mut().for_each(|x| {
            *x = *x * norm;
        });
        weights
    } else {
        weights
    }
}

// # The Dolph-Chebyshev Window
//
// Chebyshev created the equiripple polynomials and Dolph applied them to finding the minimum main
// lobe width.
//
// The specific value of this window cannot be overstated.  We can make an optimal window to only
// respond to one main lobe with all side lobes being equal.  The flatness of that floor enables us
// to treat everything above it as as true tone at the target frequency.  Together with the window
// length and parameterization of the window, we can choose the noise floor, Q, and time resolution,
// right up to the Gabor limit in every single case.
//
// The limit cases so far seem actually useful.  If the noise floor is chosen to be high, the
// weights can begin to resemble the Hamming window, demonstrating the same principled side lobe
// suppression, increasing fast attack characteristics without jeopardizing the noise floor
// behavior.
//
// The tradeoffs enable much additional information to be gained via interpretation.  If we are
// listening to a signal with 70dB peaks and we use a narrow, short, -20dB filter, the last 20dB are
// all usable signal.  If we use a longer window with -70dB side lobes, all of the 70dB are usable.
// We can look for extremely quiet sounds with some delay and uncertainty around pitch, and we can
// look for extremely loud tones at a specific pitch with the minimum window and maximum
// selectivity.  It is truly the Dolph Lundgren of windows.
//
// ## Avoid Crappy Interpolations! 🙅‍♂️
//
// The goal of solving for the window at each bin is to find the exact solutions for each window
// location instead of numerically integrating the shape and praying for the best.  The true
// Chebyshev window cannot be usefully calculated in that way and will have unpredictable side-lobe
// errors, either curving upward (leading to frequency responses at a distance!) or having
// unexpectedly high first side lobes.  Different parameters lead to shapes that look nothing like
// the smooth graph approximation.  This is a discrete problem!
//
// ## Specific Credits
//
// Fist thanks to Practical Cryptography for leaving a post up for posterity.  Th C implementation
// found here was translated to Rust to first establish grounding with a working reference
// implementation:
//
// http://practicalcryptography.com/miscellaneous/machine-learning/implementing-dolph-chebyshev-window/
//
// After conversion to Rust, opportunities for more precision were taken to minimize floating point
// errors in summation and multiplications.  Nonetheless, the window edges were quite unstable for
// window sizes relevant to our mission, and so the IDFT route was vibe coded together and verified
// against reference implementations.
//
// Second thanks to Richard Lyons for posting this a while back:
//
// https://www.dsprelated.com/showarticle/42.php
//
// Their explicit procedure allowed zooming in on a very important detail that deserves microscopic
// attention: the first (and last) index.  A proper Chebyshev window has "pedestals" in many
// solutions, the first point being larger than the others.  This isn't crazy since we can imagine
// an elaborate set of diffraction slits creating a flat interference pattern, exactly the kind of
// flatness of side lobes we want in the Chebyshev window.  Because the farthest points in the
// window are the first sample that will cancel very nearby waves that have only just begun to cycle
// out of phase, this pedestal is a critical element, not to be treated as a mere artifact that
// should be thrown away.
//
// The technique that Lyon outlines includes dividing the first index by two.  It is related to the
// chosen method.  The Cosine Summation formula is said to have an asymmetry that the toy IDFT
// method here does not.  Because of this, we do not divide the first index by two.
//
// ## Contributing to Theory via Practice
//
// Ultimately, the final word is owned by practice.  Show us a flatter side lobe on real data in
// f32, and we will adapt our practice until the model and theory can catch up.

#[deny(deprecated)]
/// Never use this except to compare the IDFT implementation to the naive one.
fn dolph_chebyshev_toy(x: f64, attenuation_db: f64) -> f64 {
    let r = 10f64.powf(attenuation_db / 20.0);
    let xc = x - 0.5;
    ((r + (r.powi(2) - 1.0).sqrt()).ln() * (PI64 * xc).cos()).cosh()
}

/// A specific treatment of the Chebyshev polynomial calculation said to be more stable between -1.0
/// and 1.0.
fn chebyshev_t_clenshaw(n: usize, x: f64) -> f64 {
    let mut b_kplus1 = 0.0;
    let mut b_kplus2 = 0.0;
    let two_x = 2.0 * x;
    for k in (1..=n).rev() {
        let b_k = (two_x).mul_add(b_kplus1, -b_kplus2 + if k == n { 1.0 } else { 0.0 });
        b_kplus2 = b_kplus1;
        b_kplus1 = b_k;
    }
    x * b_kplus1 - b_kplus2
}

/// The combined Chebyshev polynomial calculation, switching implementaitons based on accuracy and
/// resulting stability over various domains.
fn chebyshev_t(n: usize, x: f64) -> f64 {
    if x.abs() <= 1.0 {
        chebyshev_t_clenshaw(n, x)
    } else if x >= 1.0 {
        ((n as f64) * x.acosh()).cosh()
    } else {
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sign * ((n as f64) * (-x).acosh()).cosh()
    }
}

/// Generates the frequency domain data for feeding into the IDFT.
fn dolph_chebyshev_spectrum(n: usize, attenuation_db: f64) -> Vec<Complex<f64>> {
    let m = (n - 1) as f64;
    let tg = 10f64.powf(attenuation_db / 20.0);
    let beta = (tg.acosh() / m).cosh();

    // Pre-calculate the denominator T_m(beta)
    let denom = chebyshev_t(n - 1, beta);

    (0..n)
        .map(|k| {
            // We sample the circle at 2*PI*k/N, then divide by 2 to get the cosine argument.
            // This ensures the samples are perfectly symmetric around the Nyquist point.
            let theta = (TAU64 * k as f64) / (2 * n) as f64;
            let x = beta * theta.cos();

            let poly = chebyshev_t(n - 1, x);
            let weight = poly / denom;

            // Apply the centering phase shift.
            // NOTE Use (n-1) specifically to center the window across the N samples.
            let shift = (n as f64 - 1.0) / 2.0;
            let angle = -TAU64 * k as f64 * shift / n as f64;

            Complex::from_polar(weight, angle)
        })
        .collect()
}

/// Inverse discrete Fourier transform. Convert frequency domain into time domain, and windows are
/// weights over the time domain.
fn idft(x: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = x.len();
    let n_f64 = n as f64;
    (0..n)
        .map(|m| {
            let sum = x
                .iter()
                .enumerate()
                .fold(Complex::new(0.0, 0.0), |acc, (k, &xk)| {
                    let angle = TAU64 * k as f64 * m as f64 / n_f64;
                    acc + xk * Complex::from_polar(1.0, angle)
                });
            sum / n_f64
        })
        .collect()
}

/// Generate the Dolph Lundgren of window functions.  Set `attenuation_db` and then as long as you
/// approximately expect the correct peak volume levels in your input, everything from peak to the
/// attenuation dB is usable signal.  The peak is narrower with less attenuation.  This has
/// interplay with the Q vs window length relationship.
///
/// The Dolph-Chebyshev window in this module deserves special mention.  While all windows mitigate
/// the worse of DFT side lobe and noise floor problems, the Dolph-Chebyshev brings us into tightly
/// controllable engineering.  The human ear is responsive over a range of about 100dB while music
/// is often listened to with 70dB peaks.  If we want to know the difference between barely audible
/// tone at the target frequency and noise from a neighboring side lobe or loud crashing cymbals at
/// some other frequency, we need to suppress 60-80dB of noise at all other frequencies.  The
/// Dolph-Chebyshev window lets us do that without stretching the window length to unacceptably slow
/// filling lengths that would smear sounds in time.
pub fn dolph_chebyshev_window(n: usize, attenuation_db: f64) -> Vec<f64> {
    assert!(n >= 2, "Window lengths below 3 cannot suppress side lobes");
    assert!(
        attenuation_db > 0.0,
        "Valid attenuation levels must be positive."
    );
    let spectrum = dolph_chebyshev_spectrum(n, attenuation_db);
    let mut out: Vec<f64> = idft(&spectrum).iter().map(|c| c.re).collect();

    // enforce symmetry before normalization
    for i in 0..n / 2 {
        let avg = 0.5 * (out[i] + out[n - 1 - i]);
        out[i] = avg;
        out[n - 1 - i] = avg;
    }

    // normalize by max
    let max_val = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    out.iter_mut().for_each(|v| *v /= max_val);
    out
}

#[cfg(test)]
mod test {
    use crate::dsp::Filter;

    use super::*;

    // We're just demonstrating that the less trivial bins are sane.  These have been eyeballed.
    // NEXT anyone want to clean these up?  Valid windows must get pretty close to 1.0.  The
    // printing and everything could be checked via macro, but make printing vs not printing easy to
    // toggle.  The workbench is the best place for real testing.

    #[test]
    fn test_window_function_hamming() {
        let weights = bin_weights(&(hamming as fn(f64) -> f64), 25);
        // weights.iter().enumerate().for_each(|(i, b)| {
        //     println!("Hamming: {:3}: {:0.8}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));
    }

    #[test]
    fn test_window_function_welch() {
        let weights = bin_weights(&(welch as fn(f64) -> f64), 25);
        // weights.iter().enumerate().for_each(|(i, b)| {
        //   println!("Welch: {:3}: {:0.8}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));
    }

    #[test]
    fn test_window_function_dolph_window() {
        let weights = dolph_chebyshev_window(10, 60.0);
        // println!("\nn = 10");
        // weights.iter().enumerate().for_each(|(i, b)| {
        //     println!("Chebyshev: {:5}: {:0.8}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));

        let weights = dolph_chebyshev_window(11, 60.0);
        // println!("\nn = 11");
        // weights.iter().enumerate().for_each(|(i, b)| {
        //     println!("Chebyshev: {:5}: {:0.8}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));

        // Long window with deep attenuation calculation for more pitch-precise recognition of a
        // very low frequency.
        let weights = dolph_chebyshev_window(200, 80.0);
        // println!("\nn = 200");
        // weights.iter().enumerate().for_each(|(i, b)| {
        //     println!("Chebyshev: {:5}: {:0.19}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));

        // Look at the neat little steps!  Exactly like Matlab!
        let weights = dolph_chebyshev_window(31, 40.0);
        // println!("\nn = 31, attenuation = 40.0");
        // weights.iter().enumerate().for_each(|(i, b)| {
        //     println!("Chebyshev: {:5}: {:0.19}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));

        // Weird settings obtain weird windows because the physical reality we are judging is weird.
        let weights = dolph_chebyshev_window(12, 15.0);
        // println!("\nn = 31, attenuation = 5.0");
        // weights.iter().enumerate().for_each(|(i, b)| {
        //     println!("Chebyshev: {:5}: {:0.19}", i, b);
        // });
        assert!(weights.iter().all(|b| *b > 0.0 && *b <= 1.0));
    }
}
