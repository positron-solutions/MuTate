// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Warp & Workgroup Planning Analysis
//!
//! A record of how recent GPU workload leveling was planned.

// NOTE A lot of modules need to start migrating back into lib.  Until then, the imports were
// inconvenient.
// DEBT slang libraries so we can put this and associated code into lib and import them into the
// visualizer.

use super::{BINS, COLA, MAX_FREQ, SAMPLE_RATE};

#[cfg(test)]
mod test {

    use super::*;

    use mutate_lib::dsp::{self, bank, MIN_FREQ_CHEAP_DRIVERS};

    use crate::audio::{
        cwt::wavelet::{Spec, Taper},
        downsample::FILTERS,
    };

    const LOAD_QUANTUM: usize = 8;

    fn spec() -> Spec {
        Spec::default()
            .taper(Taper {
                eps_time: 5e-4,
                rho: 0.05,
            })
            .max_load_quantum(4)
    }

    fn bins() -> Vec<bank::Bin> {
        bank::bins(dsp::MIN_FREQ_CHEAP_DRIVERS, MAX_FREQ, BINS as usize)
    }

    // NOTE Whole module assumes that filters are in order of decreasing cutoffs, but alternative
    // filter designs with higher frequency cutoffs for a given decimation have been proposed.  If
    // the sorting of filters and cutoffs changes, some assumptions will need revisiting.
    fn cutoffs() -> Vec<f32> {
        FILTERS
            .iter()
            .map(|f| f.cutoff(SAMPLE_RATE))
            .collect::<Vec<f32>>()
    }

    #[test]
    fn downsampling_rate_bins_histogram() {
        // For this test, we just want to know the sample rates that our bins wind up using.  It's a
        // decent check for how often we need each input rate loaded.  Workgroups cannot work on a
        // bin without the audio in the LDS, so before anything else happens, we have to balance the
        // workgroups to the downsampling rates.

        let bins = bins();
        let cutoffs = cutoffs();

        // Bins above all cutoffs use the full sampling rate.
        let mut counts = vec![0usize; cutoffs.len() + 1];
        for b in bins.iter() {
            counts[cutoffs.partition_point(|&c| c > b.center as f32)] += 1;
        }

        for (i, n) in counts.iter().enumerate() {
            let cutoff = if i == 0 {
                0.5 * SAMPLE_RATE
            } else {
                cutoffs[i - 1]
            };
            println!("cutoff {cutoff}: {n} bins");
        }
    }

    #[test]
    fn downsampling_rate_taps_histogram() {
        // This test enables us to see the memory sizes of taps and estimate both total taps we need
        // in L2 / LDS and the taps memory demand and how sensitive we are to further downsampling.
        // We can also estimate the taps sizes for individual workgroup LDS and assign warps more
        // effectively.

        let bins = bins();
        let cutoffs = cutoffs();

        let p = spec().plan();

        // Bins above all cutoffs use the full sampling rate.
        let mut buckets = vec![0usize; usize::BITS as usize];
        let mut total = 0usize;

        for b in bins.iter() {
            let level = cutoffs.partition_point(|&c| c > b.center as f32);
            let decimation = if level == 0 {
                1
            } else {
                FILTERS[level - 1].decimation
            };
            let rate = SAMPLE_RATE / decimation as f32;
            let taps = p.bin(b.center, rate as f64).unfolded_taps(LOAD_QUANTUM) as usize;

            buckets[usize::BITS as usize - 1 - taps.leading_zeros() as usize] += 1;
            total += taps;
        }

        for (lg, n) in buckets.iter().enumerate() {
            if *n != 0 {
                println!("{:>7}..{:<7} {n} bins", 1usize << lg, (2usize << lg) - 1);
            }
        }
        println!("total taps {total}");
    }

    #[test]
    fn downsampling_rate_macs_histogram() {
        // This test attempts to expose the MAC rates and show us the need for MAC leveling across
        // the sample rates.  We need approximately proportionate amounts of lanes that can reach
        // MAC dense regions of the workload.  The memory requirements of those workgroups are known
        // from the taps histogram while the assignment density is more reflected in the bin
        // histogram.  We will need to naturally (without too much atomic contention and flow
        // control) balance all of these concerns when deciding our workgroup plan.

        let bins = bins();
        let cutoffs = cutoffs();

        let p = spec().plan();

        // Bins above all cutoffs use the full sampling rate.
        let mut buckets = vec![0usize; usize::BITS as usize];
        let mut levels = vec![(0usize, 0.0f64, 0usize); cutoffs.len() + 1];
        let mut mac_s = 0.0f64;

        for b in bins.iter() {
            let level = cutoffs.partition_point(|&c| c > b.center as f32);
            let decimation = if level == 0 {
                1
            } else {
                FILTERS[level - 1].decimation
            };
            let rate = SAMPLE_RATE / decimation as f32;
            let taps = p.bin(b.center, rate as f64).unfolded_taps(LOAD_QUANTUM) as usize;

            // length cancels: (rate * COLA / taps) outputs/s * taps MAC/output
            let bin_mac_s = COLA * rate as f64;
            mac_s += bin_mac_s;

            let l = &mut levels[level];
            l.0 += 1;
            l.1 += bin_mac_s;
            l.2 = l.2.max(taps);

            buckets[usize::BITS as usize - 1 - taps.leading_zeros() as usize] += 1;
        }

        println!("=== TAP LENGTHS (ring size / latency, not cost) ===");
        for (lg, n) in buckets.iter().enumerate() {
            if *n != 0 {
                println!("{:>7}..{:<7} {n} bins", 1usize << lg, (2usize << lg) - 1);
            }
        }

        println!("\n=== COST (COLA {COLA:.0}x) ===");
        for (i, &(n, c, max_taps)) in levels.iter().enumerate() {
            if n == 0 {
                continue;
            }
            let decimation = if i == 0 { 1 } else { FILTERS[i - 1].decimation };
            let rate = SAMPLE_RATE / decimation as f32;
            println!(
                "  /{decimation:<3} rate {rate:>8.0}  {n:>4} bins  \
             {:>9.2} MMAC/s  worst window {:>6} taps = {:>7.1} ms",
                c / 1e6,
                max_taps,
                1e3 * max_taps as f64 / rate as f64,
            );
        }
        println!("\n  total {:.2} MMAC/s", mac_s / 1e6);
    }
}
