// Copyright 2026 The MuTate Contributors
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
pub mod window;

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

/// Many filters can be operated in several modes with essentially no change in architecture.
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum FilterMode {
    HighPass,
    LowPass,
    BandPass,
    Notch,
    AllPass,
}

#[derive(Clone, Copy)]
/// To enable testing many filters side by side, all filters can be constructed from a relatively
/// common set of arguments.
pub struct FilterArgs {
    /// Quality factor equal to `center` / `bandwidth`.
    pub q: f64,
    /// Frequency where the peak gain is located.
    pub center: f64,
    /// Frequency of the sample rate.
    pub fs: f64,
    /// A final gain factor.  This is applied to the output of an individual filter or to the final
    /// output of any cascade of filters.  Individual filters should be gain normalized where
    /// possible to make gain leveling easier for banks of filters.
    pub gain_factor: f64,
    /// Bandpass, lowpass, highpass, and notch usually.  Not all filters support all modes.
    pub mode: FilterMode,

    /// Use butterworth Q ratios (highpass and lowpass only) for maximally flat pass band.
    // DEBT this is very much not enforced and was better supported than it should have been in the
    // first pass.  Please clean this up.  Only highpass and lowpass need to support this.  The
    // question is when and how to tell the user / compiler.
    pub butterworth: bool,
    /// Stagger frequencies to "dull" ringing.  With normalized gains, dull filters can "ring" like
    /// hitting a steel plate buried in the ground.  It might have a sharp pass band for a preferred
    /// frequency, but it can't store energy.  **See implementation.  Not a lot of control for
    /// stagger ratios yet.**
    pub stagger: Option<f64>,
    /// Stages.  12dB per octave per stage.  80dB per octave, meaning a max comfortable signal at
    /// twice the frequency will become visible, is about 7 stages, or a 14th order IIR.
    pub stages: usize,

    /// For DFT based filters, the weights that will be used to sum the window.
    pub window_choice: window::WindowFunction,
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
            mode: FilterMode::BandPass,
            stagger: None,
            stages: 4,
            window_choice: window::WindowFunction::DolphChebyshev {
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

#[derive(Clone, Copy, Debug)]
pub enum ResampleRatio {
    /// Create an `IntegerDownsampler`.
    IntegerDownsample { input: usize },
    /// Create a `RationalDownsampler`.
    RationalDownsample { input: usize, output: usize },
    /// Create a `RealDownsampler`.
    RealDownsample(f64),
}

#[derive(Clone, Copy)]
/// Generic arguments for testing multiple re-samplers side by side.  Because we only need
/// downsampling to simplify other kinds of filters (like DFTs), upsampling is not actually
/// supported.
///
/// The cutoff frequency is controlled by the input `sample_rate` and the `resample_ratio`.  Once we
/// figure out the new sample rate, we know the new Nyquist limit.  The stop band will be set to
/// reach the `attenuation` right at the new Nyquist limit / stop band.  Everything up to
/// the `cutoff_ratio` fraction of the new stop band will be useable pass band.
pub struct ResamplerArgs {
    /// What fraction of the new Nyquist limit will be passband.  The remainder will be transition
    /// band.  The stop band will begin at the new Nyquist limit, which is always half of the new
    /// sample rate.
    cutoff_ratio: f64,
    /// How much to reduce signal in the stop band to control aliasing.
    attenuation: f64,
    /// Rate of input to output.
    resample_ratio: ResampleRatio,
    /// Sample rate of inputs, per second.
    input_rate: usize,
}

impl ResamplerArgs {
    /// The frequency where we should be in the stop band, near the target attenuation.
    pub fn stop(&self) -> f64 {
        // Find the stop by using the Nyquist of the new output rate.
        match self.resample_ratio {
            ResampleRatio::IntegerDownsample { input: rate } => {
                let f_output = self.input_rate as f64 / rate as f64;
                f_output / 2.0
            }
            _ => todo!(),
        }
    }

    /// The frequency where the transition begins.  Gain begins to ripple or roll off and phase
    /// begins to distort (more).
    pub fn cutoff(&self) -> f64 {
        // Cutoff is a fraction of the new Nyquist limit, which is also where the stop band begins
        self.stop() * self.cutoff_ratio
    }

    /// Return sine generator centered at the beginning of the new stop band frequency, emitting
    /// samples at the input sample rate.  Use when generating stop band signal to test for folding
    /// into pass band.
    pub fn sinegen_stop(&self) -> SineSweeper {
        SineSweeper::new(self.stop(), self.input_rate as f64)
    }

    /// Return sine generator centered at half of the cutoff frequency, emitting samples at the
    /// input rate.  Use to verify integrity and delay of signals at the cutoff frequency.  Gain is
    /// likely slightly below unity.  Phase is likely to begin distorting.  Modulate down to find an
    /// acceptable practical cutoff.
    pub fn sinegen_cutoff(&self) -> SineSweeper {
        SineSweeper::new(self.cutoff(), self.input_rate as f64)
    }

    /// Return sine generator centered at half of the cutoff frequency, emitting samples at the
    /// input rate.  Use to verify integrity of signals within the passband and delay
    /// characteristics (by modulating the sine generator's wave amplitude in time).
    pub fn sinegen_pass(&self) -> SineSweeper {
        SineSweeper::new(self.cutoff() * 0.5, self.input_rate as f64)
    }
}

impl Default for ResamplerArgs {
    fn default() -> Self {
        ResamplerArgs {
            cutoff_ratio: 0.8,
            attenuation: 60.0,
            input_rate: 48000,
            resample_ratio: ResampleRatio::IntegerDownsample { input: 2 },
        }
    }
}

/// Without input and output bounds, `process` is basically a generic function.
pub trait Resampler {
    type Input;
    type Output;

    fn process(&mut self, input: Self::Input) -> Self::Output;

    /// Create the resampler from generic arguments.
    fn from_args(args: &ResamplerArgs) -> Self
    where
        Self: Sized;
}

/// A Downsampler that converts every Nth input into an output and only consumes N sized input chunks.
pub trait IntegerDownsampler<T, const N: usize>: Resampler<Input = [T; N], Output = T> {}

/// A Downsampler that converts N inputs into M outputs at a fixed ratio and only operates on N sized
/// chunks.  It is named for the rational N / M ratio.
pub trait RationalDownsampler<T, const N: usize, const M: usize>:
    Resampler<Input = [T; N], Output = [T; M]>
{
    const ASSERT_DOWNSAMPLE: () = assert!(M < N, "Rational downsampling requires M < N");
}

/// If M to N is real rather than rational, we must accept that we cannot always anticipate a new
/// output sample and instead return an Option.
pub trait RealDownsampler<T>: Resampler<Input = T, Output = Option<T>> {}

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

    /// Update the frequency on the fly.  Does not modify the current phase, only the angular
    /// velocity.
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
    fn test_sine_sweeper_phase_amplitude() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let mut s = SineSweeper::new(f0, fs);

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
    fn test_sine_sweeper_cycles() {
        let f0: f64 = 123.0;
        let fs: f64 = 48_000.0;
        let mut s = SineSweeper::new(f0, fs);

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
