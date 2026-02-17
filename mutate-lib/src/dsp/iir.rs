// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused, dead_code)]

//! # Infinite Impulse Response
//!
//! This module is used for engineering purposes, but may be useful for general DSP if you want to
//! watch specific bands or frequencies.
//!
//! ## Included Resonators
//!
//! - `Complex`: A simple two-pole complex resonator.
//! - `Biquad`: The very common Biquad.
//! - `Svf`: State Vector Filters (SVF) for low-frequency filters for use where biquads would have poles
//!   very close to one and encounter numerical stability issues.
//! - `CtyomicSvf`: Cytomic derivation of the SVF with zero delay and more focus on numerical stability in extreme
//!   Q and low frequency.
//! - `Cascade` Second-order-section (SoS) cascades of SVF or Biquads etc to steepen roll-off
//!    outside the pass bands.
//!
//! The `Cascade` implementation is generic over SoS and supports Butterworth Q ratios and detuning
//! the center frequency to reduce ringing.
//!
//! 32bit precision is preferred since this is what is available on GPUs, but 64bit variants may be
//! used to quickly determine the presence or nature of numerical stability issues in GPU-bound
//! implementations.  The initialization is 64bit and truncates after calculating constants.
//!
//! Tests for this crate merely check for sanity, NaN errors on on-bin input or excessive noise at
//! off-center pitches.  Use the workbench bin for any real tuning or evaluation.  So far most
//! filters seem well-behaved at pretty aggressive settings, but the off-bin gains of biquads have
//! been observed to be larger than expected at low frequencies and high Q.

use std::f64::consts::{PI as PI64, TAU as TAU64};

use num_complex::{Complex, Complex64};
use num_traits::Zero;

use super::{Filter, FilterArgs};

/// First order complex resonator, one of the simplest IIRs
pub struct ComplexResonator<T> {
    pole: Complex<T>,
    state: Complex<T>,
    gain: T,
}

// FIXME new is always f64!
// Cannot use this filter (for comparison, it's just a toy implementation) until the constructor is
// fixed.  Good place to make the floating point generic while creating 1st order section.
impl<T: num_traits::Float> ComplexResonator<T> {
    pub fn new(f0: T, fs: T, q: T) -> Self {
        let pi = T::from(PI64).unwrap();
        let k = pi * f0 / fs;

        Self {
            pole: Complex::from_polar(T::one() - k / q, T::from(2.0).unwrap() * k),
            state: Complex::zero(),
            gain: k / q,
        }
    }

    #[inline]
    pub fn process(&mut self, x: T) -> T {
        self.state = self.pole * self.state + self.gain * x;
        self.state.norm()
    }
}

/// DF2T 2nd order biquad.
pub struct Biquad {
    s1: f32,
    s2: f32,
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
}

impl Biquad {
    pub fn new(f0: f64, fs: f64, q: f64) -> Self {
        let w0 = TAU64 * f0 / fs;
        let alpha = w0.sin() / (2.0 * q);

        let b0 = alpha;
        let b1 = 0.0;
        let b2 = -alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * w0.cos();
        let a2 = 1.0 - alpha;

        Self {
            s1: 0.0,
            s2: 0.0,
            b0: (b0 / a0) as f32,
            b1: (b1 / a0) as f32,
            b2: (b2 / a0) as f32,
            a1: (a1 / a0) as f32,
            a2: (a2 / a0) as f32,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        // NEXT clean up f32 math while preserving structure
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Simple state variable filter, using the Toplogy Preserving Transform
pub struct Svf {
    s1: f32, // Integrator 1 state
    s2: f32, // Integrator 2 state
    g: f32,  // Alpha (tuned frequency)
    r: f32,  // 1/Q (damping)
    h: f32,  // Feedback gain
}

impl Svf {
    pub fn new(f0: f64, fs: f64, q: f64) -> Self {
        let g = (PI64 * f0 / fs).tan();
        let r = 1.0 / q;
        let h = 1.0 / (1.0 + r * g + g * g);

        Self {
            s1: 0.0,
            s2: 0.0,
            g: g as f32,
            r: r as f32,
            h: h as f32,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        // NEXT tighten up f32 precision without changing the filter itself.
        let hp = (x - (self.r + self.g) * self.s1 - self.s2) * self.h;

        let v1 = self.g * hp;
        let bp = v1 + self.s1;
        self.s1 = bp + v1;

        let v2 = self.g * bp;
        let lp = v2 + self.s2;
        self.s2 = lp + v2;

        bp * self.r
    }
}

/// Cytomic derivation of the SVF is said to be very precise even at high Qs and low frequencies.
pub struct CytomicSvf {
    s1: f32, // Integrator 1 state
    s2: f32, // Integrator 2 state
    a1: f32, // Pre-calculated multiplier 1
    a2: f32, // Pre-calculated multiplier 2
    a3: f32, // Pre-calculated multiplier 3

    norm: f32,
}

impl CytomicSvf {
    pub fn new(f0: f64, fs: f64, q: f64) -> Self {
        let g = (PI64 * f0 / fs).tan();
        let k = 1.0 / q;
        let denom = 1.0 + g * (g + k);
        let norm = 1.0_f64 / q;

        Self {
            s1: 0.0,
            s2: 0.0,
            a1: (1.0 / denom) as f32,
            a2: (g / denom) as f32,
            a3: (g * g / denom) as f32,
            norm: norm as f32,
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        // High-precision math (under development)
        let a2s1 = self.a2 * self.s1;

        let tmp = self.a2.mul_add(self.s1, self.s2);
        let v3 = x - tmp + a2s1;

        let v1 = self.a1.mul_add(self.s1, self.a2 * v3);
        let v2 = self.a3.mul_add(v3, tmp);

        self.s1 = 2.0f32.mul_add(v1, -self.s1);
        self.s2 = 2.0f32.mul_add(v2, -self.s2);

        v1 * self.norm
    }
}

/// Second-order Sections, filters that can be cascaded.
pub trait SoS: Filter {
    fn new(center: f64, fs: f64, q: f64) -> Self;
}

/// Cascade of Second-order Sections, a filter made out of filters.
pub struct Cascade<T: SoS> {
    stages: Vec<T>,
    args: FilterArgs,
    post_gain: f32,
}

impl<T: SoS> Filter for Cascade<T> {
    fn from_args(args: &FilterArgs) -> Self {
        // NEXT Not implemented, but a First order section like the regular complex resonator will
        // use arg.stages * 1.
        let bqfs = butterworth_q_factors(args.stages * 2);
        let mut staggers = args
            .stagger
            .map(|scale| stagger_factors(args.stages, scale));
        let stages: Vec<T> = (0..args.stages)
            .map(|i| {
                let f0 = if staggers.is_some() {
                    if let Some(stagger) = staggers.as_mut().unwrap().pop() {
                        args.center * stagger
                    } else {
                        args.center
                    }
                } else {
                    args.center
                };

                let q_norm = (1.0 / args.stages as f64).sqrt();
                let q = if args.butterworth {
                    args.q * (bqfs[i] * q_norm)
                } else {
                    args.q * q_norm
                };
                T::new(f0, args.fs, q)
            })
            .collect();

        Self {
            stages,
            args: *args,
            post_gain: args.gain_factor as f32,
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        let mut out = sample;
        for stage in self.stages.iter_mut() {
            out = stage.process(out);
        }
        out * self.post_gain
    }
}

macro_rules! impl_sos {
    ($t:ty) => {
        impl SoS for $t {
            fn new(center: f64, fs: f64, q: f64) -> Self {
                <$t>::new(center, fs, q)
            }
        }
    };
}

impl_sos!(Biquad);
impl_sos!(Svf);
impl_sos!(CytomicSvf);

macro_rules! impl_filter {
    ($ty:ty) => {
        impl Filter for $ty {
            #[inline]
            fn process(&mut self, sample: f32) -> f32 {
                <$ty>::process(self, sample)
            }

            fn from_args(args: &FilterArgs) -> Self {
                <$ty>::new(args.center, args.fs, args.q)
            }
        }
    };
}

impl_filter!(Biquad);
impl_filter!(Svf);
impl_filter!(CytomicSvf);

/// Use order, not number of stages, usually 2 * stages.
fn butterworth_q_factors(order: usize) -> Vec<f64> {
    assert!(order % 2 == 0, "Order must be even");
    let n_biquads = order / 2;

    (0..n_biquads)
        .rev()
        .map(|k| {
            let theta = (2.0 * k as f64 + 1.0) * std::f64::consts::PI / (2.0 * order as f64);
            1.0 / (2.0 * theta.sin())
        })
        .collect()
}

/// Stagger factors to slightly reduce perfect ringing.  The "scale" represents the total amount of
/// frequency twiddling that will occur.  Butterworth scaling will be used for no other reason than
/// we don't have another obvious choice.  The effect is that staggers begin large and get small.
/// The final SoS will have the true center frequency.
fn stagger_factors(stages: usize, scale: f64) -> Vec<f64> {
    if stages == 1 {
        return Vec::with_capacity(0);
    }
    let log_scale = scale.log2();
    let butters = butterworth_q_factors((stages - 1) * 2);
    let butter_norm: f64 = 1.0 / butters.iter().sum::<f64>();
    let mut factors = Vec::with_capacity(stages - 1);
    for i in 0..(stages - 1) {
        let b: f64 = butters[i] * butter_norm;
        let even = butters.len() % 2 == 0;
        if ((i % 2 == 0) && even) || ((i % 2 == 1) && !even) {
            factors.push((-b * log_scale).exp2());
        } else {
            factors.push((b * log_scale).exp2());
        }
    }
    factors.reverse();
    factors
}

#[cfg(test)]
mod test {
    use super::*;

    const TOL: f64 = 0.01;

    #[test]
    fn test_stagger_factors() {
        let staggers = stagger_factors(8, 1.07);
        //println!("staggers: {staggers:?}");
        let tv = vec![
            1.0332039355225888,
            0.9889877991404793,
            1.006897841709276,
            0.9948411863395327,
            1.0043286629924968,
            0.9961327873483008,
            1.0036871965774472,
        ];
        for (t, s) in staggers.iter().zip(tv.iter()) {
            assert!(((t / s) - 1.0).abs() < 0.0001);
        }

        let staggers = stagger_factors(2, 1.01);
        assert!((staggers[0] / 1.01 - 1.0) < 0.001);

        let staggers = stagger_factors(3, 1.07);
        println!("staggers: {staggers:?}");
        let tv = vec![1.0490047831651481, 0.9803783020235028];
        for (t, s) in staggers.iter().zip(tv.iter()) {
            assert!(((t / s) - 1.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_iir_butterworth_factors() {
        let facs = butterworth_q_factors(8);
        // println!("butterworth factors for 8th order: {:?}", facs);
        let tv = vec![
            0.5097955791041592,
            0.6013448869350453,
            0.8999762231364158,
            2.5629154477415064,
        ];
        assert!(facs.len() == tv.len());
        assert!(facs
            .iter()
            .zip(tv.iter())
            .all(|(x, y)| (*x - *y).abs() <= TOL));

        let facs = butterworth_q_factors(6);
        // println!("butterworth factors for 6th order: {:?}", facs);
        let tv = vec![0.5176380902050415, 0.7071067811865476, 1.9318516525781368];
        assert!(facs.len() == tv.len());
        assert!(facs
            .iter()
            .zip(tv.iter())
            .all(|(x, y)| (*x - *y).abs() <= TOL));
    }

    // NEXT an RMS accumulation test may prove even more accurate
    #[test]
    fn test_iir_cytonic_gain_vs_q() {
        let f0 = 1024.0;
        let fs = 48000.0;
        for q in [0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0] {
            let mut f = CytomicSvf::new(f0, fs, q);
            let mut peak = 0.0f32;
            let mut sine_gen = crate::dsp::sine_gen_48k(f0);

            // Scan for peak amplitude for 2s
            for x in sine_gen.take((fs * 2.0) as usize) {
                let y = f.process(x);
                peak = peak.max(y.abs());
            }
            println!("Q={} peak={}", q, peak);
        }
    }

    #[test]
    fn test_iir_cascading_sos() {
        let f0 = 32.0;
        let fs = 48000.0;
        for q in [0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0] {
            let args = FilterArgs {
                q,
                center: f0,
                fs,
                gain_factor: 1.0,
                butterworth: true,
                stagger: None,
                stages: 2,
                ..Default::default()
            };

            let mut f = Cascade::<CytomicSvf>::from_args(&args);
            let mut peak = 0.0f32;
            let mut sine_gen = crate::dsp::sine_gen_48k(f0);

            // Scan for peak amplitude for 1s
            for x in sine_gen.take((fs * 1.0) as usize) {
                let y = f.process(x);
                peak = peak.max(y.abs());
            }
            println!("Q={} peak={}", q, peak);
        }
    }

    // This test is a prototype
    #[test]
    fn test_iir_bandwidth() {
        let f0 = 24.0;
        let fs = 48_000.0;
        let q = 20.0;

        // let mut res = CytomicSvf::new(f0, fs, q);

        let args = FilterArgs {
            q: q,
            center: f0,
            fs,
            gain_factor: 1.0,
            butterworth: true,
            stagger: Some(1.02), // Some(1.0005),
            stages: 8,
            ..Default::default()
        };

        let mut res = Cascade::<CytomicSvf>::from_args(&args);

        let mut sine = args.sine_gen();

        // let warmup_waves = 8 * 4096;
        let warmup_waves = 32;
        let mut warmup_samples = (fs / f0 * warmup_waves as f64).ceil() as usize;
        let mut peak = 0.0f64;

        // Step 1: warm-up at center frequency to find the peak
        for _ in 0..warmup_samples {
            let y = res.process(sine.next().unwrap());
            peak = peak.max(y.abs() as f64);
        }
        println!("Peak amplitude = {:.6}", peak);

        // Step 2: fast sweep up, looking for loss of peaks over -5dB in last 5 waves
        let threshold_minus5db = peak * 10f64.powf(-5.0 / 20.0);
        println!("-5dB threshold = {:.6}", threshold_minus5db);

        let f_max_ratio: f64 = 4.0; // sweep 2 octaves
        let sweep_resolution = 4096;
        let log_f_step = f_max_ratio.log2() / sweep_resolution as f64;
        let next_freq = |s| f0 * (log_f_step * s as f64).exp2();

        let mut found = false;
        let mut last_seen_freq = f0;
        let mut last_seen = 0;

        // NOTE initial wave is at center frequency, so we step + 1.
        for s in 0..(sweep_resolution + 1) {
            let freq = next_freq(s);
            sine.set_frequency(freq);
            let wave_samples = (fs * 2.0 / freq).round() as usize;
            let mut wave_peak: f64 = 0.0;

            for w in 0..wave_samples {
                let y = res.process(sine.next().unwrap()) as f64;
                if y.abs() > threshold_minus5db {
                    last_seen = s;
                    last_seen_freq = freq;
                }
                wave_peak = wave_peak.max(y.abs());
            }

            // If we haven't seen a peak in five waves, we have passed the cutoff.
            if s - last_seen > 5 {
                found = true;
                println!("-5dB not seen after {:.3} Hz", last_seen_freq);
                println!("Last wave peak amplitude: {:.6}", wave_peak);
                break;
            }
        }

        if !found {
            println!("Warning: -5dB loss not detected within sweep range.");
            last_seen_freq = f0 * f_max_ratio;
        }

        // Step 3: slower log sweep back down until we see a peak over -3dB
        let threshold_minus3db = peak / 2.0f64.sqrt();
        println!("-3dB threshold = {:.6}", threshold_minus3db);
        let slow_factor = 4;
        let log_f_step = (f0 / last_seen_freq).log2() / (sweep_resolution * slow_factor) as f64;
        assert!(log_f_step < 0.0); // we're going down
        let next_freq = move |s| -> f64 { last_seen_freq * (log_f_step * s as f64).exp2() };
        let mut recovered_freq = last_seen_freq;
        let mut found = false;
        'sweep: for s in 0..(sweep_resolution * slow_factor + 1) {
            let freq = next_freq(s);
            assert!(freq > f0 * 0.99); // we don't scan below the target.
            sine.set_frequency(freq);

            let wave_samples = (fs / freq).round() as usize;
            for w in 0..wave_samples {
                let y = res.process(sine.next().unwrap()) as f64;
                if y.abs() > threshold_minus3db {
                    recovered_freq = freq;
                    found = true;
                    println!("Recovered -3dB at {:.3} Hz", freq);
                    break 'sweep;
                }
            }
        }
        if found {
            let bandwidth = 2.0 * (recovered_freq - f0);
            let q = f0 / bandwidth;
            println!("Bandwidth: {bandwidth:6.2} Hz");
            println!("Estimated q: {q:6.2}")
        } else {
            println!("Warning: -3dB not recovered before center frequency.");
        }
    }
}
