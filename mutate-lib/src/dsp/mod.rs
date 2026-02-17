// Copyright 2026 The MuTate Contributorjs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Digital Signal Processing
//!
//! MuTate needs a very high quality awareness of what's going on inside the music.  We need:
//!
//! - precision resolution of high-pitch blips shorter than presentation times (under 16ms for 60Hz
//!   video).
//! - fast rise, faster than presentation times, in order to avoid causality violation for the
//!   viewer.  Real physical causes are always seen before they are heard.
//! - fine pitch resolution for 4k and higher, without lumpy response or gaps when sweeping in
//!   pitch, all the way down to barely audible 20Hz ranges.
//! - perceptually accurate responses that appear flat to viewers with human gain curves in both
//!   visual and audio.
//!
//! ## Physics vs Design
//!
//! The main enemy is the Gabor limit.  We cannot simultaneously obtain pitch, time, and amplitude
//! resolution with one kind of filter and for every signal.  In spite of the limits of pure physics
//! and information theory, we have hard requirements that force us into tradeoffs:
//!
//! - If the viewer perceives a sound, it has corresponding visual.
//! - If a sound ends, the visual ends.
//! - If the sound moves in pitch, the visual movement is smooth, not banded.
//! - If something sounds different, it looks different.
//!
//! The last point captures the physical reality that changes in waves or strangely shaped waves
//! like sawtooth or square might look the same to a single-frequency filter but contain a
//! superposition of many transient waves at other frequencies that add into the final shape.
//!
//! ## The Target is the GPU
//!
//! Rust implementations are primarily for engineering tuning before generating static tables of
//! filter bank values that will be run by GPU programs.  Because we usually need the output on the
//! GPU for rendering video, and because the calculations are fairly heavy in order to obtain the
//! desired resolutions in time and pitch, it makes sense to always target the GPU and use Rust
//! primarily to develop new filter bank strategies.
//!
//! ## Usage
//!
//! This crate consists of two primary parts, the workbench and the spectrogram.  The workbench is
//! used to create tables for generating spectrograms or full spectrogram definitions.
//!
//! ### Workbench
//!
//! Filters are finicky, especially when pushed to the limit, and we don't know where the limit is.
//! The workbench bin program is a CLI tool for testing various filters for sanity, bandwidth, gain,
//! response speed, and noise rejection.  In addition to testing single filters, you can re-tune the
//! entire spectrogram for another resolution.
//!
//! ### Spectrogram
//!
//! Running a filter bank on the GPU requires a lot of control data to decide which shaders will run
//! what filters on which input data.  The outputs must be written to the correct addresses for
//! reading inside drawing or compute-to-present flows.  The spectrogram includes the shaders for
//! the filters in the bank, the bank control data, the audio upload, and the output buffer types
//! and geometries.

use std::f64::consts::{SQRT_2 as SQRT_2_64, TAU as TAU64};

use num_complex::Complex;

pub mod bank;
pub mod dft;
pub mod iir;
pub mod iso226; // XXX remove from use in visualizer
pub mod spectrogram;

/// Old people and rock stars cannot hear above certain frequencies.  Even if the sampling rate will
/// allow us to resolve higher frequencies, there is little visually interesting above them, and
/// only trouble makers who carry on and talk back seem to respond to them anyway.  🦕🦕🦕🦕
// NEXT user setting
// XXX remove from CQT
pub const MAX_FREQ_OLD_PEOPLE: f64 = 12_333.0;
/// Unless you have some $2000 headphones or a room built to collect energy at 20Hz, there is little
/// to perceive and thus little to draw below this frequency.  It is also very difficult to measure
/// very slow waves since they are almost entirely smooth DC that will always take a while for any
/// detector to phase-lock on while high-cutting literally everything else.
// NEXT user setting
// XXX remove from CQT
pub const MIN_FREQ_CHEAP_DRIVERS: f64 = 24.0;

// /// Compute peak amplitude of a sine wave for a given SPL (in dB)
// /// relative to 0 dB = 20 µPa (threshold of hearing)
// pub const fn peak_amplitude_from_db(spl_db: f64) -> f64 {
//     const P0: f64 = 20e-6; // reference pressure in Pa
//     let rms = P0 * 10f64.powf(spl_db / 20.0);
//     SQRT_2_64 * rms
// }

// const PEAK_0DB: f64 = peak_amplitude_from_db(0.0);
// const PEAK_80DB: f64 = peak_amplitude_from_db(80.0);
// const PEAK_40DB: f64 = peak_amplitude_from_db(40.0);

#[derive(Clone, Copy)]
pub struct FilterArgs {
    /// Quality factor equal to `center` / `bandwidth`.
    pub q: f64,
    /// Frequency where the peak gain is located.
    pub center: f64,
    /// Frequency of the sample rate.
    pub fs: f64,
    /// A final gain factor.  This is applied to the output of an individual filter or to the final
    /// output of any cascade of filters.  Individual filters should be gain normalized where
    /// possible to make gain levelling easier for banks of filters.
    pub gain_factor: f64,

    /// Butterworth
    pub butterworth: bool,
    /// Stagger frequencies
    pub stagger: Option<f64>,
    /// Stages.  12dB per octave per stage.  80dB per octave, meaning a max comfortable signal at
    /// twice the frequency will become visible, is about 7 stages, or a 14th order IIR.
    pub stages: usize,

    /// For DFT based filters, the weights that will be used to sum the window.
    pub window_choice: dft::WindowFunction,
}

impl FilterArgs {
    /// Return a `SineSweeper` for the center frequency.  You can modulate the sine wave before
    /// reading if you want another center frequency.
    pub fn sine_gen(&self) -> SineSweeper {
        // DEBT we are throwing around Center and Sample frequencies a bit too haphazardly, and it's
        // goint to bite us.
        SineSweeper::new(self.center, self.fs)
    }

    /// Return the number of samples required to complete `nwaves` cycles at the center frequency.
    pub fn nsamples(&self, nwaves: f64) -> usize {
        (self.fs / self.center * nwaves).ceil() as usize
    }
}

impl Default for FilterArgs {
    fn default() -> Self {
        FilterArgs {
            q: 10.0,
            center: 1000.0,
            fs: 48_000.0,
            gain_factor: 1.0,
            butterworth: false,
            stagger: None,
            stages: 4,
            window_choice: dft::WindowFunction::DolphChebyshev {
                attenuation_db: 40.0,
            },
        }
    }
}

pub trait Filter {
    /// Process a single amplitude sample.
    fn process(&mut self, sample: f32) -> f32;

    /// Create the filter from generic arguments.
    fn from_args(args: &FilterArgs) -> Self
    where
        Self: Sized;
}

/// Fixed sine wave generator.  Truncates to f32.
// NEXT replace all usages with the more versatile SineSweeper.
pub fn sine_gen(f0: f64, fs: f64) -> impl Iterator<Item = f32> {
    let omega = TAU64 * f0 / fs;
    let mut re = 1.0;
    let mut im = 0.0;
    let cos = omega.cos();
    let sin = omega.sin();

    std::iter::from_fn(move || {
        let out = im as f32;
        let new_re = re * cos - im * sin;
        let new_im = re * sin + im * cos;
        re = new_re;
        im = new_im;
        Some(out)
    })
}

/// 48k sample rate sine wave generator
pub fn sine_gen_48k(f0: f64) -> impl Iterator<Item = f32> {
    sine_gen(f0, 48_000.0)
}

/// Sine wave generator with frequency modulation.  Use to generate rough chirps to quickly look for
/// changes in filter response.
pub struct SineSweeper {
    re: f64,
    im: f64,
    omega: f64,
    cos: f64,
    sin: f64,
    fs: f64,
    f0: f64,
}

impl SineSweeper {
    pub fn new(f0: f64, fs: f64) -> Self {
        let omega = TAU64 * f0 / fs;
        Self {
            re: 1.0,
            im: 0.0,
            omega,
            cos: omega.cos(),
            sin: omega.sin(),
            fs,
            f0,
        }
    }

    /// Update the frequency on the fly
    pub fn set_frequency(&mut self, f0: f64) {
        self.omega = TAU64 * f0 / self.fs;
        self.cos = self.omega.cos();
        self.sin = self.omega.sin();
    }

    /// Read the current center frequency.
    pub fn center(&self) -> f64 {
        self.f0
    }

    /// Return the number of samples required to cover `nwaves` full cycles.
    pub fn nsamples(&self, nwaves: f64) -> usize {
        (self.fs / self.f0 * nwaves).ceil() as usize
    }
}

impl Iterator for SineSweeper {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let out = self.im as f32;
        let new_re = self.re * self.cos - self.im * self.sin;
        let new_im = self.re * self.sin + self.im * self.cos;
        self.re = new_re;
        self.im = new_im;
        Some(out)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sine_gen_phase_amplitude() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let mut s = sine_gen(f0, fs);

        // NOTE we burn a sample when initializing last, so n should be zero, but because we
        // estimate the peak and trough from zero, not 1, it will still take n omegas to hit
        // extrema.
        let n_peak = (fs / (4.0 * f0)).ceil() as usize;
        let n_trough = (fs * 3.0 / (4.0 * f0)).ceil() as usize;

        let mut last = s.next().unwrap();
        let mut n = 0;

        // scan for first decrease
        loop {
            let next = s.next().unwrap();
            if next < last {
                assert_eq!(n, n_peak);
                assert!((next - 1.0).abs() < 0.05);

                // consistent counting for next loop
                n += 1;
                break;
            }
            last = next;
            n += 1;
        }

        // scan for first increase
        loop {
            let next = s.next().unwrap();
            if next > last {
                assert_eq!(n, n_trough);
                assert!((next + 1.0).abs() < 0.05);
                break;
            }
            last = next;
            n += 1;
        }
    }

    #[test]
    fn test_sine_gen_cycles() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let mut s = sine_gen(f0, fs);

        let mut last = 0.0;
        let mut n = 0;

        let target_cycles = 1000usize;

        let expected_n = (target_cycles as f64 * fs / f0).round() as usize;

        let mut last = s.next().unwrap();
        let mut n: usize = 1; // n + 1 from first sample above
        let mut crossings = 0;

        loop {
            let current = s.next().unwrap();

            if last < 0.0 && current >= 0.0 {
                crossings += 1;

                if crossings == target_cycles {
                    assert_eq!(n, expected_n);
                    break;
                }
            }

            last = current;
            n += 1;
        }
    }

    #[test]
    fn test_sine_gen_rms() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let amplitude: f64 = 1.0;

        let mut s = sine_gen(f0, fs);

        let target_cycles = 7777usize;
        let expected_samples = (target_cycles as f64 * fs / f0).ceil() as usize;

        let mut sum_sq: f64 = 0.0;
        let mut n = 0usize;

        let mut last = s.next().unwrap();

        for _ in (0..expected_samples) {
            let current = s.next().unwrap() as f64;
            sum_sq += current.powi(2);
            n += 1;
        }

        let rms = (sum_sq / n as f64).sqrt() as f64;
        let expected_rms = amplitude / std::f64::consts::SQRT_2;

        // rms = 0.7071067612059638, expected = 0.7071067811865475
        let tolerance = 0.0000001;

        assert!(
            (rms - expected_rms).abs() < tolerance,
            "rms = {}, expected = {}",
            rms,
            expected_rms
        );
    }

    #[test]
    fn test_sine_sweep_gen_rms() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let amplitude: f64 = 1.0;

        let mut s = SineSweeper::new(f0, fs);

        let target_cycles = 7777usize;
        let expected_samples = (target_cycles as f64 * fs / f0).ceil() as usize;

        let mut sum_sq: f64 = 0.0;
        let mut n = 0usize;

        let mut last = s.next().unwrap();

        for _ in (0..expected_samples) {
            let current = s.next().unwrap() as f64;
            sum_sq += current.powi(2);
            n += 1;
        }

        let rms = (sum_sq / n as f64).sqrt() as f64;
        let expected_rms = amplitude / std::f64::consts::SQRT_2;

        // rms = 0.7071067612059638, expected = 0.7071067811865475
        let tolerance = 0.0000001;

        assert!(
            (rms - expected_rms).abs() < tolerance,
            "rms = {}, expected = {}",
            rms,
            expected_rms
        );
    }

    #[test]
    fn test_sine_sweep_gen_smooth() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let amplitude: f64 = 1.0;

        let mut s = SineSweeper::new(f0, fs);

        // 100 of the initial waves
        let samples = (fs / f0 * 100.0) as usize;

        // NEXT log sweep would be better...
        let alpha = f0 * 9.0 / samples as f64;

        // Check that we never have any sudden steps larger than the expected omega
        let tolerance = 1.01;
        let calc_max_delta = |f| (TAU64 * f / fs * tolerance) as f32;
        let mut delta_allow = calc_max_delta(f0);
        let mut last = s.next().unwrap();

        // Taking the RMS because it should be pretty close
        let mut sum_sq = 0.0;
        for n in 0..samples {
            let current = s.next().unwrap();

            let delta = current - last;
            // println!("delta: {:0.8}, delta_allow: {:0.8}", delta, delta_allow);
            assert!(delta.abs() < delta_allow);
            last = current;

            sum_sq += (current as f64).powi(2);

            let f_next = f0 + alpha * (n as f64);
            s.set_frequency(f_next);
            delta_allow = calc_max_delta(f_next);
        }

        let rms = (sum_sq / samples as f64).sqrt() as f64;
        let expected_rms = amplitude / std::f64::consts::SQRT_2;

        // NOTE it's a bit less accurate because we are changing stuff a lot
        // rms = 0.7071260099558259, expected = 0.7071067811865475
        let tolerance = 0.001;

        assert!(
            (rms - expected_rms).abs() < tolerance,
            "rms = {}, expected = {}",
            rms,
            expected_rms
        );
    }
}
