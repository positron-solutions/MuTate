// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Bank
//!
//! *A filter banks combines filters in order to comb the desert for fugitives and their stolen
//! robots.*
//!
//! Certain design goals will always pull us towards multiple sub-banks and multiple output time
//! slots.  This module contains the data structures necessary to describe our bank so that it may
//! be hardcoded into GPU control logic for execution.

use super::iso226;

pub struct Bin {
    /// Minimum frequency
    pub min: f64,
    /// Maximum frequency
    pub max: f64,
    /// Center frequency
    pub center: f64,
    /// iso226 gain correction summand (dB).  Add this to the bin measured dB for an
    /// iso-loud perceptually corrected dB.
    pub iso226_gain: f64,
}

impl Bin {
    /// Difference between minimum and maximum frequency.
    pub fn bandwidth(&self) -> f64 {
        self.max - self.min
    }

    /// Quality factor for this bin presuming we have a perfect filter.
    pub fn q(&self) -> f64 {
        self.center / self.bandwidth()
    }
}

/// Return a list of bin spacings.  We use logarithmic spacing because it matches music and pretty
/// closely matches human senses of tones.
pub fn bins(min: f64, max: f64, count: usize) -> Vec<Bin> {
    assert!(max > min);
    assert!(count > 1);

    // Once again, I shall remind myself how to walk the log.  Just a terrible morning and I haven't
    // yet digested the coffee. ☕
    //
    // The sum of logs is the log of a product.  The product of all of the factors for all steps is
    // equal to max / min.  We are chopping up a ratio into equal ratios rather than chopping up a
    // sum into equal summands.
    //
    // The first bin and last bin are actually half-step inside.  Bins must know their width, so
    // it's actually easier if we walk the log in half steps.  For each bin, we will walk two half
    // steps, beginning at min and ending at max.  Thus there are 2 * count steps.
    let ratio = max / min;
    let steps = 2 * count;

    // method 1
    // max_ratio: 1.0000000000000004441
    let log_step = ratio.log2() / steps as f64;
    let freq = |i: usize| min * (log_step * i as f64).exp2();

    // Method 2
    // max_ratio: 0.9999999999997188915
    // sum of logs is log of product, and the product of steps is equal to the ratio.
    // let step = (ratio.log2() / (steps as f64)).exp2();
    // let freq = |i: usize| min * step.powi(i as i32);

    // Method 3
    // max_ratio: 0.9999999999994294564
    // The ith root to the ith power is unity
    // let step = ratio.powf(1.0 / steps as f64); //  * 0.99999989418598;
    // let freq = |i: usize| min * step.powf(i as f64);

    // Method 2
    (0..count)
        .map(|i| {
            let i0 = i * 2;
            let center = freq(i0 + 1);
            Bin {
                min: freq(i0),
                center,
                max: freq(i0 + 2),
                iso226_gain: iso226::iso226_gain(center).unwrap(),
            }
        })
        .collect()
}

pub fn bin_lookup(min: f64, max: f64, count: usize, center: f64) -> Bin {
    let mut bins = bins(min, max, count);
    let mut closest = 0usize;
    let mut min_dist = f64::MAX;
    for (i, b) in bins.iter().enumerate() {
        let ratio = center / b.center;
        let dist = (1.0 - ratio).abs();
        if dist < min_dist {
            closest = i;
            min_dist = dist;
        }
    }
    bins.remove(closest)
}

#[cfg(test)]
mod test {

    use crate::dsp;

    use super::*;

    #[test]
    fn test_bins_range() {
        let count = dsp::spectrogram::RESOLUTION_4K_WIDTH;
        let freq_min = dsp::MIN_FREQ_CHEAP_DRIVERS;
        let freq_max = dsp::MAX_FREQ_OLD_PEOPLE;
        let bins = bins(freq_min, freq_max, count);

        assert_eq!(bins.len(), count);

        let min = bins[0].min;
        let max = bins.last().unwrap().max;

        // Check that we covered the range by verifying that our interpolation hits the same
        // beginning and end points.
        let min_ratio = min / dsp::MIN_FREQ_CHEAP_DRIVERS;
        let max_ratio = max / dsp::MAX_FREQ_OLD_PEOPLE;

        // println!("min ratio: {:10.19}", min_ratio);
        // println!("max ratio: {:10.19}", max_ratio);

        // So basically we are testing for more accuracy than what f32 can reliably represent.  Nice
        // bins for hardcoding ;-)
        assert!((min_ratio - 1.0).abs() < 0.000000000000001);
        assert!((max_ratio - 1.0).abs() < 0.000000000000001);
        assert_eq!(bins.len(), count);

        // Well ordered
        for b in bins.iter() {
            assert!(b.min < b.center);
            assert!(b.center < b.max);
        }

        // Bandwidth sum matches target spectrum
        let mut sum = 0.0;
        for b in bins.iter() {
            sum += b.bandwidth();
        }
        assert!(((sum / (freq_max - freq_min)) - 1.0).abs() < 0.000000000000001);
    }

    #[test]
    fn test_bins_bin_lookup() {
        let target = 4000.0;
        let closest = bin_lookup(
            dsp::MIN_FREQ_CHEAP_DRIVERS,
            dsp::MAX_FREQ_OLD_PEOPLE,
            dsp::spectrogram::RESOLUTION_4K_WIDTH,
            target,
        );

        // NOTE bins do not have infinite resolution

        let ratio = closest.center / target;
        assert!((ratio - 1.0).abs() < 0.01);

        // First bin will have the minimum frequency
        let closest = bin_lookup(
            dsp::MIN_FREQ_CHEAP_DRIVERS,
            dsp::MAX_FREQ_OLD_PEOPLE,
            dsp::spectrogram::RESOLUTION_4K_WIDTH,
            0.0,
        );
        let ratio = closest.min / dsp::MIN_FREQ_CHEAP_DRIVERS;
        assert!((ratio - 1.0).abs() < 0.0000001);

        // Last bin will have the maximum frequency
        let closest = bin_lookup(
            dsp::MIN_FREQ_CHEAP_DRIVERS,
            dsp::MAX_FREQ_OLD_PEOPLE,
            dsp::spectrogram::RESOLUTION_4K_WIDTH,
            100_000.0,
        );
        let ratio = closest.max / dsp::MAX_FREQ_OLD_PEOPLE;
        assert!((ratio - 1.0).abs() < 0.0000001);
    }
}
