// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Discrete Fourier Transform
//!
//! This module contains a basic CPU implementation for a single DFT and several window functions
//! for engineering the bins of filter banks for implementation on the GPU.

use std::f64::consts::{PI as PI64, TAU as TAU64};

use num_complex::Complex;
use num_traits::Zero;

// DEBT switch to the better ring buffer already!!!!
use ringbuf::traits::{Consumer, Observer, Producer};
use ringbuf::{storage::Heap, LocalRb};

use crate::dsp::{self, window};

/// ## Discrete Fourier Transform
///
/// A single Goertzel-based, 2nd order finite impulse response (FIR) filter.  Its characteristics
/// are very sharp frequency resolution but unavoidably delayed attack due to window shaping and the
/// measurement depending on a sum of the window.  When moving configurations to the GPU, multiple
/// DFTs can be efficiently parallelized at different Q and noise tolerances by re-summing the same
/// ring buffer of raw Goertzel outputs, which are only bound by center frequency, with a variety of
/// window shapes and lengths, enabling excellent pitch, time, and amplitude resolution.
///
/// DFTs fundamentally do not yield a distinct output for every input sample unless we re-sum the
/// window on every output.  Every window has a recommended COLA length (we use recommendations
/// found [here](https://holometer.fnal.gov/GH_FFT.pdf)) beyond which we are just over-calculating
/// without yielding any increased power or amplitude flatness.
///
/// In order to implement `dft::Filter` and work use the same diagnostic routines as other filters
/// in the workbench, `process` will always yield an output, but it is a repeated value unless the
/// window has been re-summed.  We only re-sum the window enough to satisfy COLA power flatness.
/// This is the behavior of short-time Fourier Transforms (STFT).
///
/// The output is effectively an amplitude, as if we have seen a constant tone for the duration
/// between updates.  This will produce roughly usable peaks, RMS, and derived measurements like
/// rise time etc.
pub struct Dft {
    /// Do the right thing and choose either Dolph-Chebyshev or write a new window and combine it
    /// with a a pre-filter.
    window_choice: window::WindowFunction,
    /// Multiplying the ring buffer by the window yields an output.  This CPU DFT only implements
    /// one window.  Parallelization takes place on the GPU implementation.
    window_factors: Vec<f32>,
    /// Normalize output by the weight sum of the window.
    window_norm: f32,
    /// The window of complex numbers resulting from the Goertzel algorithm application.  This must
    /// be windowed and summed to produce an output.
    goertzel_terms: LocalRb<Heap<Complex<f32>>>,
    /// The center frequency for this filter
    center: f32,
    /// Pre-calculated gain to apply.  Set this to 1.0 when calibrating.
    gain: f32,
    /// The rotation angle that is updated with each sample processed.
    phase: Complex<f32>,
    /// Rate of rotation per sample.
    velocity: Complex<f32>,
    /// The number of samples remaining until we update our output by re-summing while applying the
    /// window.  This value paces the window overlap and controls COLA.
    window_repeat: u32,
    /// The number of times we have repeated a sample.
    repeated: u32,
    /// Previous output, used when repeating.
    last_output: f32,
}

impl dsp::Filter for Dft {
    /// Remember,the DFT yields identical results for each window_repeat samples!
    fn process(&mut self, sample: f32) -> f32 {
        // DEBT The ring buffer really feels like it has two-sided semantics even in the local case.
        // If we know we are literally just interested in keeping a fixed size window, it's a little
        // hacky and the semantics don't really feel right.  Either we need our own ring buffer or
        // to make this one express the semantics that we want.
        let mut last_added = self.goertzel_terms.first_mut();
        let inner = last_added.as_mut().unwrap();

        // Goertzel
        let term = Complex {
            re: sample * self.phase.re,
            im: -sample * self.phase.im,
        };

        **inner = term;
        unsafe {
            self.goertzel_terms.advance_read_index(1);
        }
        unsafe {
            self.goertzel_terms.advance_write_index(1);
        }
        assert_eq!(
            self.goertzel_terms.occupied_len(),
            self.window_factors.len()
        );

        // Rotate phase by velocity
        self.phase = Complex {
            re: self.phase.re * self.velocity.re - self.phase.im * self.velocity.im,
            im: self.phase.re * self.velocity.im + self.phase.im * self.velocity.re,
        };

        if self.repeated == self.window_repeat {
            self.repeated = 0;
            let sum: Complex<f32> = self
                .goertzel_terms
                .iter()
                .zip(self.window_factors.iter())
                .fold(Complex::zero(), |accum, (g, window_factor)| {
                    accum + (g * window_factor)
                });
            self.last_output = 2.0 * sum.norm() / self.window_norm;

            // Normalize the phase to prevent drift over time.
            let norm = (self.phase.re * self.phase.re + self.phase.im * self.phase.im).sqrt();
            self.phase.re /= norm;
            self.phase.im /= norm;
        }
        self.repeated += 1;
        self.last_output
    }

    fn from_args(args: &dsp::FilterArgs) -> Self {
        let length = (args.q * args.fs / args.center).ceil() as usize;

        Dft::new(args.center, args.fs, length, args.window_choice)
    }
}

impl Dft {
    pub fn new(
        center: f64,
        sample_rate: f64,
        length: usize,
        window_choice: window::WindowFunction,
    ) -> Self {
        let window_factors = window_choice.make_window_32(length);
        let window_repeat = window_choice.repeat(length);
        let mut goertzel_terms = LocalRb::new(length);
        unsafe { goertzel_terms.advance_write_index(length) };
        goertzel_terms.iter_mut().for_each(|a| *a = Complex::zero());
        let scalar_velocity = (TAU64 * center) / sample_rate;
        let (sin, cos) = scalar_velocity.sin_cos();
        let velocity = Complex {
            re: cos as f32,
            im: sin as f32,
        };
        let window_norm = window_factors.iter().sum();
        Self {
            center: center as f32,
            window_choice,
            window_norm,
            goertzel_terms,
            window_factors,
            gain: 1.0,
            velocity,
            window_repeat,
            // MAYBE If I ever knew why we initialize it this way, I forgot.
            phase: Complex { re: 1.0, im: 0.0 },
            repeated: 0,
            last_output: 0.0,
        }
    }

    /// Reutrn the number of samples that must be processed to completely saturate the window.
    pub fn length(&self) -> usize {
        self.window_factors.len()
    }
}

#[cfg(test)]
mod test {
    use crate::dsp::Filter;

    use super::*;

    #[test]
    fn test_dft_sanity() {
        let mut args = dsp::FilterArgs::default();
        args.window_choice = window::WindowFunction::DolphChebyshev {
            attenuation_db: 80.0,
        };
        args.q = 32.0;
        args.center = 400.0;
        let mut sg = args.sine_gen();
        let mut dft = Dft::from_args(&args);
        let nsamples = dft.length() * 4;
        let mut peak: f32 = 0.0;
        for n in 0..nsamples {
            peak = peak.max(dft.process(sg.next().unwrap()));
        }
        println!("bin-centered peak: {peak}");

        sg.set_frequency(400.0 + 40.0);

        // Empty the window
        let mut last: f32 = 0.0;
        for _ in 0..nsamples * 8 {
            let sample = sg.next().unwrap();
            let this = dft.process(sample);
            if this != last {
                // println!("emptying window: {this}");
                last = this;
            }
        }

        let mut off_center_peak: f32 = 0.0;
        for _ in 0..nsamples * 32 {
            off_center_peak = off_center_peak.max(dft.process(sg.next().unwrap()).abs());
        }
        println!("off-center peak: {off_center_peak}");

        assert!(off_center_peak < peak);
    }
}
