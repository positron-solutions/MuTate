// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused, dead_code)]

//! # Finite Impulse Response (FIR)
//!
//! This module is used for engineering purposes, but may be useful for general DSP if you want to
//! watch specific bands or frequencies.
//!
//! We use FIRs to perform linear phase filtering to avoid distortions that would shift impulses in
//! time.  When rendering at high sampling rates, these shifts would cause misalignment across
//! frequencies, violating the hear-something-see-something condition in a way that a delay line
//! cannot fix.  All-pass phase shifts may restore phase linearity, but then we are adding back the
//! same delay that we were trying to avoid by using an IIR.  Just use an FIR and be done with it.
//!
//! ## Tradeoffs
//!
//! Note the use of windows that are very aggressive and filters that are quite short.  This is
//! because **we are trading transition band for shorter filters and deeper cuts** in almost every
//! case.  We can boost back up signals that are not badly distorted, so we really don't even worry
//! too much about pass band droop.  Loss of phase information, delay, and aliasing of higher
//! frequencies are the real enemies.
//!
//! ## Low Pass
//!
//! The low pass filter in this crate is designed to facilitate downsampling.  We need to vaporize
//! the stop bands so that they don't fold back into our target frequencies.  We can tolerate some
//! leakage into the portion of the band we won't use (higher bands were already measured from the
//! pre-decimated stream ).  Correspondingly, we can dump the ripple into regions that will fold
//! back into the part of the new signal we don't care about, giving us a really large transition
//! band to work with (about a quarter of the old Nyquist limit) and use really short FIRs.

use std::f64::consts::PI as PI64;

use mutate_slide::SlidingWindow;

use crate::tree::WindowedTreeSum;

use crate::dsp::{
    self, window::WindowFunction, Filter, FilterArgs, IntegerDownsampler, Resampler, ResamplerArgs,
};

pub struct FirLowpass<const N: usize> {
    states: SlidingWindow<[f32; N]>,
    window: [f32; N],
    gain: f32,
}

impl<const N: usize> Filter for FirLowpass<N> {
    fn process(&mut self, input: f32) -> f32 {
        self.states.push(input);
        self.window
            .iter()
            .copied()
            .zip(self.states.iter().copied())
            .windowed_tree_sum()
    }

    fn from_args(args: &FilterArgs) -> Self {
        // XXX pretty shaky "from args" so far.
        FirLowpass::new(args.window_choice, args.center / args.fs)
    }
}

impl<const N: usize> FirLowpass<N> {
    pub fn new(window_function: WindowFunction, cutoff: f64) -> Self {
        let coefficients: Vec<f32> = window_function
            .make_windowed_sinc(N, cutoff)
            .iter()
            .map(|c| *c as f32)
            .collect();
        Self::with_coefficients(coefficients)
    }

    pub fn with_coefficients(coeffs: impl IntoIterator<Item = f32>) -> Self {
        let mut window = [0.0f32; N];
        let mut count = 0;
        for (slot, coeff) in window.iter_mut().zip(coeffs) {
            *slot = coeff;
            count += 1;
        }
        assert_eq!(count, N, "expected {N} coefficients, got {count}");
        // NOTE seems PMR weights are pretty far off of this condition, so it had to be relaxed.
        assert!((window.iter().sum::<f32>() - 1.0).abs() < 0.1);
        // dbg!(window);
        Self {
            states: SlidingWindow::from_storage([0f32; N]),
            window,
            gain: 1.0,
        }
    }
}

/// Dynamic variant of the `FirLowpass` for use in CLI tools etc where we don't know the size until
/// runtime.
// NEXT
pub struct DynamicFirLowpass {
    states: SlidingWindow<Vec<f32>>,
    window: Vec<f32>,
    gain: f32,
}

impl DynamicFirLowpass {
    pub fn new(n: usize, window_function: WindowFunction, cutoff: f64) -> Self {
        let coefficients: Vec<f32> = window_function
            .make_windowed_sinc(n, cutoff)
            .iter()
            .map(|c| *c as f32)
            .collect();
        Self::with_coefficients(coefficients)
    }

    pub fn with_coefficients(coeffs: impl IntoIterator<Item = f32>) -> Self {
        let mut window: Vec<f32> = coeffs.into_iter().collect();
        let n = window.len();
        assert!(n != 0, "Window must not be empty");
        assert!(n % 2 == 1, "Window length must be odd");
        Self {
            states: SlidingWindow::from_storage(vec![0f32; n]),
            window,
            gain: 1.0,
        }
    }
}

impl Filter for DynamicFirLowpass {
    fn process(&mut self, input: f32) -> f32 {
        self.states.push(input);
        self.window
            .iter()
            .copied()
            .zip(self.states.iter().copied())
            .windowed_tree_sum()
    }

    fn from_args(args: &FilterArgs) -> Self {
        // XXX size!
        DynamicFirLowpass::new(23, args.window_choice, args.center / args.fs)
    }
}

// XXX This implementation has been designed and this is a sketch of the code, left private for now.
// The implementation makes sense and will show up at engineering time to be sure that our combined
// filters are gain normalized at each bin.  Same result will be implemented on the GPU.
struct Polyphase<const D: usize, const H: usize> {
    states: [SlidingWindow<[f32; H]>; D],
    coeffs: [[f32; H]; D],
    phase: usize,
}

impl<const D: usize, const H: usize> Polyphase<D, H> {
    pub fn from_prototype(h: &[f32]) -> Self {
        assert_eq!(
            h.len(),
            D * H,
            "prototype must have exactly D * H = {} taps, got {}",
            D * H,
            h.len()
        );

        let mut coeffs = [[0f32; H]; D];
        for (k, &coeff) in h.iter().enumerate() {
            coeffs[k % D][k / D] = coeff;
        }

        Self {
            states: [SlidingWindow::from_storage([0f32; H]); D],
            coeffs,
            phase: 0,
        }
    }
}

impl<const D: usize, const H: usize> Resampler for Polyphase<D, H> {
    type Input = [f32; D];
    type Output = f32;

    fn process(&mut self, input: [f32; D]) -> f32 {
        // Each input sample goes to exactly its own sub-filter — no overlap.
        for (i, &sample) in input.iter().enumerate() {
            self.states[i].push(sample);
        }

        // Sum across all D sub-filters.
        self.coeffs
            .iter()
            .zip(self.states.iter())
            .map(|(c, s)| {
                c.iter()
                    .zip(s.iter())
                    .map(|(&coeff, &state)| coeff * state)
                    .sum::<f32>()
            })
            .sum()
    }

    fn from_args(args: &ResamplerArgs) -> Self {
        todo!("caller should supply prototype via from_prototype; use ResamplerArgs to derive tap count and cutoff externally")
    }
}

impl<const D: usize, const H: usize> IntegerDownsampler<f32, D> for Polyphase<D, H> {}

#[cfg(test)]
mod test {
    use super::*;

    const TEST_FILTER: [f32; 23] = [
        f32::from_bits(0x39799e4d), // +0.00023805
        f32::from_bits(0x3a3d579e), // +0.00072228
        f32::from_bits(0xba529089), // -0.00080324
        f32::from_bits(0xbbae524a), // -0.00531987
        f32::from_bits(0xbb1a1145), // -0.00235088
        f32::from_bits(0x3c858aec), // +0.01630159
        f32::from_bits(0x3c9ae446), // +0.01890768
        f32::from_bits(0xbd0690a2), // -0.03285278
        f32::from_bits(0xbd91603f), // -0.07098436
        f32::from_bits(0x3d46e164), // +0.04855479
        f32::from_bits(0x3e9c1674), // +0.30485880
        f32::from_bits(0x3ee3cc87), // +0.44491979
        f32::from_bits(0x3e9c1674), // +0.30485880
        f32::from_bits(0x3d46e164), // +0.04855479
        f32::from_bits(0xbd91603f), // -0.07098436
        f32::from_bits(0xbd0690a2), // -0.03285278
        f32::from_bits(0x3c9ae446), // +0.01890768
        f32::from_bits(0x3c858aec), // +0.01630159
        f32::from_bits(0xbb1a1145), // -0.00235088
        f32::from_bits(0xbbae524a), // -0.00531987
        f32::from_bits(0xba529089), // -0.00080324
        f32::from_bits(0x3a3d579e), // +0.00072228
        f32::from_bits(0x39799e4d), // +0.00023805
    ];

    #[test]
    fn fir_windowed_sinc() {
        let window_fns = [
            WindowFunction::BoxCar,
            WindowFunction::Welch,
            WindowFunction::Bartlett,
            WindowFunction::Hamming,
            WindowFunction::DolphChebyshev {
                attenuation_db: 80.0,
            },
            WindowFunction::Literal {
                weights: &TEST_FILTER,
            },
        ];

        for wf in window_fns.iter() {
            // Input generator is Halfway up the new Nyquist limit.
            let cutoff = 0.125; // 0.25 Nyquist, 6kHz
            let f_sample = 48_000.0;
            let f_pass = 2000.0;
            let f_stop = 18_000.0;
            // ⚠️ Be sure to adjust all locations with N!  Window length is not well compile-time
            let n = 23;
            let mut input = dsp::SineSweeper::new(f_pass, f_sample);

            // Make a windowed sinc with exactly a new Nyquist limit cutoff
            // checked yet.
            let mut coeffs: Vec<f32> = match wf {
                WindowFunction::Literal { weights } => weights.iter().copied().collect(),
                _ => wf
                    .make_windowed_sinc(n, cutoff)
                    .iter()
                    .map(|w| *w as f32)
                    .collect(),
            };
            // N sized filters
            let mut filter = FirLowpass::<23>::with_coefficients(coeffs);

            input.set_frequency(f_pass);
            for _ in 0..(n * 2) {
                filter.process(input.next().unwrap());
            }
            let measure = input.nsamples(128.0);
            let mut peak: f32 = 0.0;
            for _ in 0..measure {
                peak = peak.max(filter.process(input.next().unwrap()).abs());
            }
            // In the pass runs, we're looking for
            println!("Peak pass {:<20} {peak:2.8}", format!("{:}:", wf));
            assert!(peak > 0.7);

            input.set_frequency(f_stop);
            for _ in 0..(2 * n) {
                filter.process(input.next().unwrap());
            }
            let measure = input.nsamples(128.0);
            let mut peak: f32 = 0.0;
            for _ in 0..measure {
                peak = peak.max(filter.process(input.next().unwrap()).abs());
            }
            /// NOTE the "winners" can be a little unpredictable due to ripple in the stop bands.
            /// Whenever the test frequency lands right on one of the sweet spots, that filter will
            /// obliterate every last trace of the input signal.
            println!("Peak stop {:<20} {peak:2.8}", format!("{:}:", wf));
            assert!(peak < 0.1);
        }
    }
}
