// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # High-Precision, Destruction Tolerant Accumulator
//!
//! > When I'm done, half will still remain.
//! >
//! > - Ivo Nahak
//!
//! Calculating **vanishing** residuals of oscillating functions can provide marvelous opportunities
//! for transient loss of precision to obliterate many whole terms.
//!
//! This implementation descended from using a stack array to perform a tree sum as each term is
//! emitted, so no pre-generation or sorting would be required.  Further review lead to the
//! realization that the stack slots could be used to store compensator terms, retaining more
//! precision of each intermediate term.  The result is similar to a Møller–Knuth two-sum style
//! bidirectional carry.  The benefits for long vanishing sums are demonstrated.  Add as you like.
//! Destructively cancel as you will.
//!
//! There has been a lot of convergent evolution over the years:
//!
//! - Wolfe (1964) took the first step toward bucket summation.
//! - Malcolm's (1974) contribution was reducing the number of buckets by associating each with `d`
//!   consecutive exponents.
//!
//! Additionally, these contributions were pointed out as "roughly similar" in ways that are likely
//! to at least inspire improvements if not land on the formalization:
//!
//! - Zhu & Hayes, HybridSum (2009) and OnlineExactSum (2010)
//! - Lange & Rump aaaSum (2022)
//!
//! This is a toy implementation used in some tests and does not use SIMD or any kind of fast
//! routing decisions, and that strongly limits the performance.  It does better than Kahan 64-bit
//! summation while using 32bit operands. If they were fast 32bit operands, it might actually be
//! worth it!

// NEXT We really need a way to return one compensator-residual pair so that intermediate sums
// retain acquired precision across intermediate calls to sum.
// NEXT Assemble some pathological test cases to at least think about what functions this can't be
// used to sum, especially ones where the other methods are better.
// NEXT Binning oracle trait so that any type with a magnitude signal can be added
// NEXT SIMD.... probably based on some kind of pre-calculated greater-than and scale masks that
// enable routing decisions to be pre-calculated (behind the actual sums) and enable fast2sum and
// other such things.  Overlapping magnitude ranges may actually some kind of compensator-summand
// spectrum that we can economize over a structure to limit the number of scalar operations.  The
// operand load pipeline for SIMD is just extra space we can epilogue away and taken all together,
// would hide a shit ton of latency.  Zero padding.  Integrated design approach necessary, but this
// can probably be done with SIMD and variable load with to support flexible scalar packing.
// NEXT Complex support as some kind of adapter trait for n-floats structures as long as they are
// commutative and associative, basically as long as we can treat n-floats as n-sums.
// NOTE If the SIMD or generalized compensator structure get done, it will be time for this to
// become a published crate.  Numerical integrations have to contend with finer resolution vs higher
// error all the time, and residual sums are just one pathological case.
// MAYBE Peek sum vs consuming finalizer sum.  Depends on if the stores can be skipped efficiently
// and if registers pressure / shuffling is different for consume compared to peek.  My bet is that
// SIMD will later want a consume method that uses the fast path while peek has to avoid destroying
// any compensators.  Peek is extremely useful for prefix summing.

use std::ops::{Add, Sub};

use aligned::{Aligned, A64};
use num_traits::Float;

/// A bank of Kahan accumulators, one per magnitude band.
///
/// Incoming values route to the band matching their magnitude.  When a slot's value leaves its band
/// it is carried into the band it now belongs to.  That carry is what keeps each accumulator small
/// relative to its own compensator, alleviating the worst case magnitude differences that can
/// frustrate simple Kahan summation.
#[derive(Clone, Copy)]
pub struct Accumulator<F, const N: usize = 32> {
    slots: Aligned<A64, [(F, F); N]>,
    min_exp: i32,
}

impl<F: Float, const N: usize> Default for Accumulator<F, N> {
    fn default() -> Self {
        let (_, min_exp, _) = F::min_positive_value().integer_decode();
        Self {
            slots: Aligned([(F::zero(), F::zero()); N]),
            min_exp: min_exp as i32,
        }
    }
}

fn two_sum<F: Float>(a: F, b: F) -> (F, F) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

impl<F: Float, const N: usize> Accumulator<F, N> {
    #[inline(always)]
    fn bin(&self, x: F) -> usize {
        let (_, e, _) = x.integer_decode();
        (((e as i32 - self.min_exp) / 8) as usize).min(N - 1)
    }

    fn insert(&mut self, x: F) {
        let mut pending = x;
        while !pending.is_zero() {
            let i = self.bin(pending);
            let (v, c) = self.slots[i];

            let (v2, e) = two_sum(v, pending);
            let (c2, _third_order) = two_sum(c, e);

            if v2.is_zero() || self.bin(v2) == i {
                self.slots[i] = (v2, c2);
                return;
            }

            self.slots[i] = (F::zero(), F::zero());
            if !c2.is_zero() {
                self.insert(c2);
            }
            pending = v2;
        }
    }

    pub fn add(&mut self, value: F) {
        self.insert(value);
    }

    pub fn sub(&mut self, value: F) {
        self.insert(-value);
    }

    /// Compensated fold from the smallest band upward, taking each slot's compensation
    /// before its value so the single unavoidable rounding lands once, at the top.
    pub fn sum(&self) -> F {
        let (v, c) = self
            .slots
            .iter()
            .fold((F::zero(), F::zero()), |(acc, comp), &(v, c)| {
                let (t1, e1) = two_sum(acc, c);
                let (t2, e2) = two_sum(t1, v);
                (t2, comp + e1 + e2)
            });
        v + c
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Neumaier's improvement over Kahan: the correction is chosen by comparing magnitudes, so it
    /// stays correct when the incoming term dominates the running sum.  Its weakness is that `c`
    /// is itself a plain accumulator — corrections spanning many magnitudes fall off its bottom.
    fn neumaier_step<F: Float>(x: F, v: &mut F, c: &mut F) {
        let t = *v + x;
        *c = *c
            + if v.abs() >= x.abs() {
                (*v - t) + x
            } else {
                (x - t) + *v
            };
        *v = t;
    }

    #[test]
    fn all_zeroes() {
        let mut a: Accumulator<f32> = Accumulator::default();
        for v in vec![0.0_f32; 100] {
            a.add(v);
        }
        assert_eq!(a.sum(), 0.0);
    }

    #[test]
    fn beats_naive() {
        // small amplitude phasor, rotated through some cycles.  Exactly predictable residual.
        // Accumulator should be bracketed by expected and naive results.

        let cycles = 119_usize;
        let samples_per_cycle = 320001_usize; // odd, not a power of two
        let n = cycles * samples_per_cycle;

        let amplitude = 1.0e-2_f64;
        let theta = 2.0 * std::f64::consts::PI / samples_per_cycle as f64;
        let (sin_t, cos_t) = theta.sin_cos();

        // unit phasor, rotated in place each step
        let (mut re, mut im) = (1.0_f64, 0.0_f64);

        let mut accum = Accumulator::<f32>::default();
        let mut naive = 0.0_f32;

        // Go ahead and destroy the precision scales
        accum.add(77.0);
        naive += 77.0;

        for _ in 0..n {
            let sample = (amplitude * im) as f32;
            accum.add(sample);
            naive += sample;

            let next_re = re * cos_t - im * sin_t;
            let next_im = re * sin_t + im * cos_t;
            re = next_re;
            im = next_im;

            // Newton step to keep the phasor from spiraling.
            let mag_sq = re * re + im * im;
            let k = 1.5 - 0.5 * mag_sq;
            re *= k;
            im *= k;
        }

        // let expected = 1.0e4;
        let expected = 77.0;
        let accum_result = accum.sum();

        let accum_err = (accum_result as f64 - expected).abs();
        let naive_err = (naive as f64 - expected).abs();

        // Several orders better on the accumulator
        println!(
            "expected: {}, accumulator error: {:+0.4e}, naive error: {:+0.4e}",
            expected, accum_err, naive_err
        );

        assert!(
            accum_err < naive_err,
            "accum ({accum_result:+0.4e}, err {accum_err:+0.4e}) should beat naive ({naive:+0.4e}, err {naive_err:+0.4e})"
        );
        assert!(
            (naive_err > 1e-7),
            "naive was unexpectedly exact — test isn't stressing accumulation"
        );
    }

    #[test]
    fn beats_kahan() {
        // Sum a chirp, then sum the negation.  Result should be zero, but Kahan alone can't let you
        // do that.
        let n = 4_000_000_usize;
        let theta = 2.0 * std::f64::consts::PI / 977.0; // odd period, no exact cancellation
        let log_start = 1.0e18_f64.ln();
        let log_end = 1.0e-18_f64.ln();

        // The offset's job is to upset the residual at any given time and then become the residual
        // at the end.
        let offset = 0.5_f32;

        let sample = |i: usize| -> f32 {
            let t = i as f64 / (n - 1) as f64;
            let amp = (log_start + (log_end - log_start) * t).exp();
            (amp * (i as f64 * theta).sin()) as f32
        };

        let mut kahan32_v = offset;
        let mut kahan32_c = 0.0_f32;
        let mut kahan64_v = offset as f64;
        let mut kahan64_c = 0.0_f64;
        let mut accum = Accumulator::<f32>::default();
        accum.add(offset);

        fn kahan_step<F>(x: F, v: &mut F, c: &mut F)
        where
            F: Copy + Add<Output = F> + Sub<Output = F>,
        {
            let y = x - *c;
            let sum = *v + y;
            *c = (sum - *v) - y;
            *v = sum;
        }

        for i in 0..n {
            let x = sample(i);
            kahan_step(x, &mut kahan32_v, &mut kahan32_c);
            kahan_step(x as f64, &mut kahan64_v, &mut kahan64_c);
            accum.add(x);
        }
        for i in (0..n).rev() {
            let x = -sample(i);
            kahan_step(x, &mut kahan32_v, &mut kahan32_c);
            kahan_step(x as f64, &mut kahan64_v, &mut kahan64_c);
            accum.add(x);
        }

        let expected = offset as f64;
        let kahan32_err = (kahan32_v as f64 - expected).abs();
        let kahan64_err = (kahan64_v - expected).abs();
        let accum_err = (accum.sum() as f64 - expected).abs();

        println!(
            "expected: {expected}, kahan32 error: {kahan32_err:+0.16e}, kahan64 error: {kahan64_err:+0.16e}, accumulator error: {accum_err:+0.16e}"
        );

        assert!(
            accum_err < kahan32_err,
            "accumulator ({accum_err:+0.4e}) should beat kahan32 ({kahan32_err:+0.4e})"
        );
    }

    #[test]
    fn smoke_test_f64() {
        let mut a: Accumulator<f64> = Accumulator::default();
        for i in 0..16_000_000 {
            a.add(1e-6f64);
        }
        let result = a.sum();

        // NOTE didn't think too long about this one, but the rounding may have been in our favor.
        // Should remain so on any platform compliant with the same floating point spec.
        assert!(
            (result - 16.0).abs() < f64::MIN_POSITIVE,
            "f64 sum was not sixteen: {}",
            result
        );
    }

    #[test]
    fn meets_neumaier_on_uniform_terms() {
        // Uniform small terms into a growing sum: the case Neumaier handles essentially exactly in
        // f32.  We only need to confirm we are not worse.
        let n = 1_000_000_usize;
        let x = 1.0e-6_f32;

        let mut neumaier_v = 0.0_f32;
        let mut neumaier_c = 0.0_f32;
        let mut accum = Accumulator::<f32>::default();

        for _ in 0..n {
            neumaier_step(x, &mut neumaier_v, &mut neumaier_c);
            accum.add(x);
        }

        let expected = n as f64 * x as f64;
        let neumaier_err = ((neumaier_v + neumaier_c) as f64 - expected).abs();
        let accum_err = (accum.sum() as f64 - expected).abs();

        println!(
            "expected: {expected}, neumaier error: {neumaier_err:+0.4e}, accumulator error: {accum_err:+0.4e}"
        );

        assert!(
            accum_err <= neumaier_err,
            "accumulator ({accum_err:+0.4e}) should match or beat neumaier ({neumaier_err:+0.4e})"
        );
    }

    #[test]
    fn beats_neumaier() {
        // Stress the single-compensator assumption directly.  The chirp sweeps 36 decades, so the
        // per-step corrections themselves span a range no single f32 can hold: early corrections
        // near 1e11 pin the exponent of `c` and every later correction rounds straight out of it.
        // The anchor offset survives as the whole answer after cancellation.
        let n = 4_000_000_usize;
        let theta = 2.0 * std::f64::consts::PI / 977.0;
        let log_start = 1.0e18_f64.ln();
        let log_end = 1.0e-18_f64.ln();

        let offset = 0.5_f32;

        let sample = |i: usize| -> f32 {
            let t = i as f64 / (n - 1) as f64;
            let amp = (log_start + (log_end - log_start) * t).exp();
            (amp * (i as f64 * theta).sin()) as f32
        };

        let mut neumaier_v = offset;
        let mut neumaier_c = 0.0_f32;
        let mut accum = Accumulator::<f32>::default();
        accum.add(offset);

        for i in 0..n {
            let x = sample(i);
            neumaier_step(x, &mut neumaier_v, &mut neumaier_c);
            accum.add(x);
        }
        for i in (0..n).rev() {
            let x = -sample(i);
            neumaier_step(x, &mut neumaier_v, &mut neumaier_c);
            accum.add(x);
        }

        let expected = offset as f64;
        let neumaier_err = ((neumaier_v + neumaier_c) as f64 - expected).abs();
        let accum_err = (accum.sum() as f64 - expected).abs();

        println!(
            "expected: {expected}, neumaier error: {neumaier_err:+0.16e}, accumulator error: {accum_err:+0.16e}"
        );

        assert!(
            neumaier_err > 1e-7,
            "neumaier was unexpectedly exact — test isn't stressing the compensator"
        );
        assert!(
            accum_err < neumaier_err,
            "accumulator ({accum_err:+0.4e}) should beat neumaier ({neumaier_err:+0.4e})"
        );
    }
}
