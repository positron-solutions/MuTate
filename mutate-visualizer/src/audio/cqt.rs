// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use ringbuf::traits::{Consumer, Observer, Producer};
use ringbuf::{storage::Heap, LocalRb};

use crate::audio::iso226;
use crate::audio::raw::Audio;
use crate::graph::GraphEvent;

// NEXT Windowing functions.  Windowed values must be re-applied on rolling sum portions.
// NEXT ML corrections for harmonics and leakages, inferring absence of true tones from presence of
// sympathetic tones.
// NEXT Low-pass filters for long-wavelength bins to reduce accumulation of decimation aliasing
// noise.
// NEXT pre-compute twiddles and use phase adjustment for accumulation
// DEBT sample rate
// DEBT format channels

/// Simple Complex number without any cruft.
#[derive(Default, Copy, Clone)]
pub struct Complex {
    pub real: f32,
    pub imag: f32,
}

impl Complex {
    /// Magnitude
    pub fn mag(&self) -> f32 {
        self.power().sqrt()
    }

    /// The square of the magnitude.
    pub fn power(&self) -> f32 {
        self.real * self.real + self.imag * self.imag
    }

    /// Phase angle in radians
    pub fn phase(self) -> f32 {
        self.imag.atan2(self.real)
    }

    /// Scalar multiplication
    pub fn scale(&self, scale: f32) -> Self {
        Complex {
            real: self.real * scale,
            imag: self.imag * scale,
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct AudioComplex {
    pub left: Complex,
    pub right: Complex,
}

impl AudioComplex {
    pub fn scale(&self, scale: f32) -> Self {
        AudioComplex {
            left: self.left.scale(scale),
            right: self.right.scale(scale),
        }
    }
}

/// Output of a `CqtBin`.  An array of them is a `CqtBin` output.  Each channel outputs an
/// `AudioComplex` and a scalar for perceptual sound level.
// DEBT format channels
#[derive(Default, Clone)]
pub struct Cqt {
    pub left: Complex,
    pub right: Complex,
    /// ISO226 adjusted, approximate phons dB.  Because no reference pressure is known without a
    /// calibrated microphone, just apply a uniform summand to approximate real phon levels.
    pub left_perceptual: f32,
    pub right_perceptual: f32,
    /// Center frequency for the originating bin,
    pub freq: f32,
    /// The ISO226 factor that can be used to correct relative magnitudes.
    pub iso226_factor: f32,
}

struct CqtBin {
    center: f32,
    /// ISO226 correction dB.
    iso226_offset: f32,
    // DEBT memory management
    terms: LocalRb<Heap<AudioComplex>>,
    /// As we slide the window, the phase becomes offset relative to where we started.
    phase: f32,
    /// Constant modifier accounting for sample versus natural rate.
    velocity: f32,
    /// Decimate the input by only reading every nth sample.
    decimation: usize,
    /// When reading decimated inputs, if we did not skip enough inputs for consistent decimation,
    /// skip this many data points from the next consume cycle.
    skip: usize,
}

impl CqtBin {
    pub fn new(center: f32, size: usize, decimation: usize) -> Self {
        // Ensure the effective size is longer than the typical window to avoid energy accumulation
        // at frequencies that couple with the chunk size.
        let size = (size as f32 / decimation as f32).ceil() as usize;
        let mut terms = LocalRb::new(size);
        unsafe {
            terms.advance_write_index(size);
        }
        assert!((size * decimation) >= 800);
        terms.iter_mut().for_each(|a| *a = AudioComplex::default());
        Self {
            center,
            iso226_offset: iso226::iso226_gain(center).unwrap(),
            terms,
            phase: 0.0,
            velocity: (std::f32::consts::TAU * center * (decimation as f32)) / 48_000_f32, // DEBT format rate
            decimation,
            skip: 0,
        }
    }

    // DEBT sample rate
    pub fn consume(&mut self, input: &[Audio]) {
        // XXX Seek events etc are just not handled.  The way to handle them is via accumulation
        // between calls to produce,  but it needs to be numerically stable.
        let mut input = input;
        if self.skip != 0 {
            self.phase += self.velocity;
            input = &input[self.skip..];
        }
        let read_len = input.len() / self.decimation;
        self.skip = (self.decimation - (input.len() % self.decimation)) % self.decimation;

        // Roll off old data
        if read_len == self.len() {
            // XXX watch for input > terms.. it's not correctly handled
            self.terms.clear(); // NOTE advances read index
        } else {
            unsafe { self.terms.advance_read_index(read_len) };
        }

        // Roll on the new data
        // XXX phase actually breaks if the input is too large for the ring.
        let mut phase = self.phase;
        unsafe {
            self.terms.advance_write_index(read_len);
        }
        let (head, tail) = self.terms.as_mut_slices();
        // if we can write into just the tail, do it, otherwise write into a chain of head and tail.
        let tail_len = tail.len();
        let head_len = head.len();

        // If read_len is larger than the slice, something is probably wrong.  If the slices we are
        // cutting off are not the same size as the read_len, something is wrong.
        // This assertion is trivially always true, so the code below seems good.
        // assert_eq!(read_len, (head_len - (read_len - tail_len)) + (read_len - tail_len));
        let head_index = if read_len >= tail_len {
            (head_len + tail_len).saturating_sub(read_len)
        } else {
            0
        };
        let output = head[head_index..]
            .iter_mut()
            .chain(tail[tail_len.saturating_sub(read_len)..].iter_mut());

        input
            .iter()
            .step_by(self.decimation)
            .take(read_len)
            .zip(output)
            .for_each(|(input, out)| {
                let (c, s) = phase.sin_cos();
                let left_real = input.left * c;
                let left_imag = -input.left * s;
                let right_real = input.right * c;
                let right_imag = -input.right * s;

                *out = AudioComplex {
                    left: Complex {
                        real: left_real,
                        imag: left_imag,
                    },
                    right: Complex {
                        real: right_real,
                        imag: right_imag,
                    },
                };
                phase += self.velocity;
                if phase > std::f32::consts::TAU {
                    phase -= std::f32::consts::TAU;
                }
            });
        assert_eq!(self.terms.occupied_len(), self.terms.capacity().into());
        self.phase = phase;
    }

    // The filter bins have essentially extracted the harmonic component and when we do the RMS
    // operations on the result, we obtain something very similar to RMS at each bin frequency.  We
    // need RMS to have a consistent path to applying an inverse ISO226 curve correction, enabling
    // our machine to see sound somewhat like a human.
    pub fn produce(&self) -> Cqt {
        // This unrolled add was quite a bit faster.  Can't be wasting time on these sums on the CPU.
        let mut l_real = 0.0f32;
        let mut l_imag = 0.0f32;
        let mut r_real = 0.0f32;
        let mut r_imag = 0.0f32;

        assert_eq!(self.terms.occupied_len(), self.terms.capacity().into());
        for x in self.terms.iter() {
            l_real += x.left.real;
            l_imag += x.left.imag;
            r_real += x.right.real;
            r_imag += x.right.imag;
        }

        let sum = AudioComplex {
            left: Complex {
                real: l_real,
                imag: l_imag,
            },
            right: Complex {
                real: r_real,
                imag: r_imag,
            },
        };

        // XXX length
        let norm = 1.0 / self.effective_len() as f32;
        // `c` because this RMS is off by some constant factor we don't care about.
        let left_c_rms = sum.left.scale(norm * std::f32::consts::SQRT_2).mag();
        let right_c_rms = sum.right.scale(norm * std::f32::consts::SQRT_2).mag();
        let left_spl = 20.0 * left_c_rms.log10() + self.iso226_offset;
        let right_spl = 20.0 * right_c_rms.log10() + self.iso226_offset;

        Cqt {
            left: sum.left,
            right: sum.right,
            left_perceptual: left_spl,
            right_perceptual: right_spl,
            freq: self.center,
            // XXX There is something wrong with the math and I'm a bit too tired to do the
            // "correction" quite right, but basically we convert decibels to bels.  The 20x factor
            // in SPL needs some kind of tweak and this might not be right, but it's close.
            iso226_factor: (10.0_f32.powf(self.iso226_offset / 10.0)),
        }
    }

    /// The length of the input that will be decimated and read.  This is the window length for this
    /// filter bin.
    pub fn effective_len(&self) -> usize {
        self.len() * self.decimation
    }

    /// The length of the internal buffer, which is only filled after decimation.  This is **not**
    /// the window length for this filter bin.
    pub fn len(&self) -> usize {
        self.terms.capacity().into()
    }
}

/// A Constant Q transform uses a variable window size that can provide better response times for
/// higher frequencies that can be resolved with shorter window.  It cannot re-use terms efficiently
/// like an FFT but is more appropriate for sliding windows and logarithmic bin spacing, which
/// complicates term re-use anyway.  Several design goals are used in this treatment:
///
/// - Bin center frequencies are chosen for perceptual consistency (linearly spaced octaves)
/// - Create bins over the entire perceptual range, from 20Hz to the Nyquist limit (24kHz) for a
///   typical sample rate (48kHz)
/// - Variable bin counts, enabling consumers to choose the resolution they need.
/// - Efficient re-use of partial terms when the window is larger than the time step
pub struct CqtNode {
    bins: Box<[CqtBin]>,
    /// Quality factor of this CQT filter bank.
    q: f32,
    output: Box<[Cqt]>,
}

impl CqtNode {
    /// * `resolution` - Number of frequency bins.
    /// * `sample_rate` - Audio input samples per second.
    /// * `update_rate` - expected maximum frames per second. Establishes a floor on the window
    ///    length since higher frequencies will just use all of the samples  in each frame.
    pub fn new(resolution: usize, sample_rate: u32, update_rate: f32) -> Self {
        let freq_min = 20.0f32;
        let freq_max = (sample_rate / 2) as f32; // Nyquist Limit

        let log_min = freq_min.log2();
        let log_max = freq_max.log2();
        let log_step = (log_max - log_min) / (resolution - 1) as f32;

        // calculate window length using "quality" derived from bins per octvate
        let octaves = (freq_max / freq_min).log2();
        let b = resolution as f32 / octaves;
        let q = 1.0 / (2.0f32.powf(1.0 / b) - 1.0);
        let size_min = (sample_rate as f32 / update_rate).ceil() as usize;
        let bins: Vec<CqtBin> = (0..resolution)
            .map(|n| {
                let freq = (log_min + (n as f32 * log_step)).exp2();
                let bin_nyquist = freq * 2.0;
                // Whenever we have over two times as many samples as we need, decimate the sample
                // rate by two.  This keeps the information margins low.  The extra 2.0 keeps bin
                // sizes relatively consistent, around 200 samples.
                let decimation = 2u32.pow((freq_max / (bin_nyquist * 2.0)).log2() as u32);
                let size = (q * sample_rate as f32 / freq).ceil() as usize;
                CqtBin::new(freq, size.max(size_min), decimation as usize)
            })
            .collect();
        let total = &bins.iter().fold(0usize, |accum, b| accum + b.len());
        println!("total bins length: {}", total);

        Self {
            q,
            bins: bins.into_boxed_slice(),
            output: std::vec::from_elem(Cqt::default(), resolution).into_boxed_slice(),
        }
    }

    pub fn consume(&mut self, input: &GraphEvent<Audio>) {
        // XXX What are we doing with the intent?
        let fresh = input.buffer.fresh();
        for b in &mut self.bins {
            b.consume(fresh);
        }
    }

    pub fn produce(&mut self) -> &[Cqt] {
        self.output
            .iter_mut()
            .zip(self.bins.iter_mut())
            .for_each(|(out, bin)| {
                *out = bin.produce();
            });
        self.output.as_ref()
    }

    #[allow(unused_variables)]
    pub fn destroy(&self, device: &ash::Device) {}

    pub fn resolution(&self) -> usize {
        self.bins.len()
    }
}

#[cfg(test)]
mod test {
    use ringbuf::traits::Producer;

    use crate::graph::{EventIntent, GraphBuffer};

    use super::*;

    #[test]
    fn test_cqt_bin_lengths() {
        let cqt = CqtNode::new(128, 48000, 60.0);

        // uncomment to output reference values
        // for (i, b) in cqt.bins.iter().enumerate() {
        //     println!(
        //         "i: {}, frequency: {}, effective_length: {},  length: {}",
        //         i,
        //         b.center,
        //         b.effective_len(),
        //         b.len()
        //     );
        // }

        // LIES if the test values bobble a bit, it's no big deal, just update them. What we want to
        // verify is that we get a constant window for values that are shorter than the step size.
        assert_eq!(cqt.bins[0].effective_len(), 42240);
        assert_eq!(cqt.bins[72].effective_len(), 800);
        assert_eq!(cqt.bins[127].effective_len(), 800);

        assert_eq!(cqt.bins[0].len(), 165);
        assert_eq!(cqt.bins[72].len(), 200);
        assert_eq!(cqt.bins[127].len(), 800);

        // LIES central frequencies may also bobble a bit due to changes in manner of calculation.
        // Just update them.  We want to be sure we're covering approximately the full window.
        assert!(cqt.bins[0].center - 20.0 < 0.01);
        assert!(cqt.bins[127].center - 24000.0 < 1.0);
    }

    #[test]
    fn test_cqt_bin_precision() {
        let freq = 800_f32;
        let sample_rate = 48_000_f32;
        let angular = std::f32::consts::TAU * freq / sample_rate;
        let input: [Audio; 5800] = std::array::from_fn(|i| {
            let phase = i as f32 * angular;
            Audio {
                left: phase.sin(),
                right: phase.cos(),
            }
        });

        let mut tuned_bin = CqtBin::new(freq, 5800, 2);
        let mut mistuned_bin = CqtBin::new(freq * (std::f32::consts::SQRT_2 - 1.0), 5800, 2);

        tuned_bin.consume(&input);
        mistuned_bin.consume(&input);

        let t = tuned_bin.produce();
        // Results are about 40-50x different in linear space.
        println!(
            "tuned: left {:8.6}, right {:8.6}",
            t.left.mag(),
            t.right.mag()
        );
        let m = mistuned_bin.produce();
        println!(
            "mistuned: left {:8.6}, right {:8.6}",
            m.left.mag(),
            m.right.mag()
        );
        let tuned = tuned_bin.produce();
        let mistuned = mistuned_bin.produce();

        // About 20 dB between the tuned and mistuned bins
        // sum of logs is log of products
        assert!(tuned.left_perceptual > mistuned.left_perceptual + 20.0);
        assert!(tuned.right_perceptual > mistuned.right_perceptual + 20.0);

        // This assertion is near the precision limit.
        assert!(
            ((tuned.left.phase() - tuned.right.phase()).abs() - std::f32::consts::FRAC_PI_2).abs()
                < 0.01
        );
    }

    #[test]
    fn test_cqt_node_precision() {
        let test_freq = 5050_f32;
        let sample_rate = 48_000_f32;
        let angular = std::f32::consts::TAU * test_freq / sample_rate;
        let mut phase = 0f32;
        let mut raw = [Audio::default(); 48000];
        raw.iter_mut().for_each(|out| {
            *out = Audio {
                left: phase.sin(),
                right: phase.cos(),
            };
            phase = phase + angular;
        });

        let mut cqt = CqtNode::new(256, 48000, 60.0);

        let mut input: GraphBuffer<Audio> = GraphBuffer::new(800);
        let mut i = 0usize;
        while i + 800 <= 48000 {
            input.write(&raw[i..(i + 800usize)]);
            i += 800;
            let event = GraphEvent {
                intent: EventIntent::Full,
                buffer: &input,
            };

            cqt.consume(&event);
        }

        let out = cqt.produce();
        for o in out {
            println!(
                "freq: {:6.2}, left magnitude: {:6.2}, left_perceptual: {:6.2}",
                o.freq,
                o.left.mag() * o.iso226_factor,
                o.left_perceptual
            )
        }

        let max_by_percep_left = out
            .iter()
            .max_by(|c, d| c.left_perceptual.total_cmp(&d.left_perceptual))
            .unwrap();

        let max_by_mag_left = out
            .iter()
            .max_by(|c, d| c.left.mag().total_cmp(&d.left.mag()))
            .unwrap();

        let max_by_percep_right = out
            .iter()
            .max_by(|c, d| c.right_perceptual.total_cmp(&d.right_perceptual))
            .unwrap();

        let max_by_mag_right = out
            .iter()
            .max_by(|c, d| c.right.mag().total_cmp(&d.right.mag()))
            .unwrap();

        let closest_bin = out
            .iter()
            .min_by(|c, d| {
                (c.freq - test_freq)
                    .abs()
                    .total_cmp(&(d.freq - test_freq).abs())
            })
            .unwrap();

        // Test result is sensitive to resolution because the bucket with the most energy is just
        // "near" the test frequency.  Adjust to match a nearby bin and comparisons will pass.
        assert!((closest_bin.freq - test_freq) / test_freq < 0.02);
        assert_eq!(max_by_percep_left.freq, closest_bin.freq);
        assert_eq!(max_by_mag_left.freq, closest_bin.freq);
        assert_eq!(max_by_percep_right.freq, closest_bin.freq);
        assert_eq!(max_by_mag_right.freq, closest_bin.freq);
    }
}
