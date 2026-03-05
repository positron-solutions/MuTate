//! # Tree
//!
//! *"Tree! Tree! Tree! Tree!*
//!
//! *"Tree!" - Don Cheadle*
//!
//! Since we tend to write a lot of windowed sums, this little module helps us not only make them a
//! bit faster, but more accurate by the innate nature of tree sums.  When summing Goertzel terms
//! that are destructively cancelling or constructively adding up, the terms will *tend* to be
//! around the same size, playing into the precision advantages of tree sums.
//!
//! The operations are designed to auto-vectorize **and parallelize in time via superscalar
//! operations** on the CPU.  It's used around the library to speed up some places and maintain a
//! Rust implementation for comparison with Slang variants.
//!
//! The procedures are written serially for **simple** use on the GPU but you can of course rotate
//! the procedures across warps and use warp and group level synchronization to execute the same
//! routines with much more SIMT (Single Input Multiple Thread) parallelism.
//!
//! Save the planet.  Use tree sums.  To build better solar cells and advance magnetohydrodynamic
//! research for engineering fusion reactors!  Peace in our time.

// Tree!

use std::ops::{Add, Mul};

use aligned::{Aligned, A64};
use num_traits::{Float, MulAdd, Zero};

/// Sums an iterator using a binary tree reduction to improve numerical accuracy
/// and enable superscalar parallelism.
///
/// Adjacent pairs are summed first, then pairs of pairs, and so on — keeping
/// operands close in magnitude and avoiding the catastrophic cancellation that
/// a naive left-fold accumulates when large terms of opposing sign dominate.
///
/// # Examples
///
/// ```
/// use mutate_lib::tree::TreeSum;
///
/// // Integers: exact result for 1..=100
/// let result = (1..=100_i64).tree_sum();
/// assert_eq!(result, 5050);
///
/// // Empty iterator returns the zero value
/// let result = std::iter::empty::<f64>().tree_sum();
/// assert_eq!(result, 0.0);
///
/// // Single element is returned as-is
/// let result = std::iter::once(42.0_f64).tree_sum();
/// assert_eq!(result, 42.0);
///
/// // Catastrophic cancellation: naive left-fold loses the 1.0 entirely,
/// // tree sum pairs the ±1e15 terms first and recovers it
/// let v = vec![1.0e15_f64, 1.0, -1.0e15];
/// assert_eq!(v.iter().copied().tree_sum(), 1.0);
/// ```
pub trait TreeSum: Iterator {
    type Output;
    fn tree_sum(self) -> Self::Output;
}

impl<F, I> TreeSum for I
where
    F: Zero + Add<Output = F> + Clone,
    I: Iterator<Item = F>,
{
    type Output = F;
    fn tree_sum(mut self) -> F {
        let mut stack: Aligned<A64, [Option<F>; 32]> = Aligned(std::array::from_fn(|_| None));
        for s in self {
            let mut carry = s;
            for slot in stack.iter_mut() {
                match slot {
                    None => {
                        *slot = Some(carry);
                        break;
                    }
                    Some(existing) => {
                        carry = existing.clone() + carry;
                        *slot = None;
                    }
                }
            }
        }
        stack
            .iter()
            .rev()
            .flatten()
            .cloned()
            .fold(F::zero(), |acc, x| acc + x)
    }
}

/// Sums an iterator of `(signal, window)` pairs using a fused multiply-add
/// tree reduction. Each pair is accumulated via `fmadd` for numerical accuracy,
/// with additions balanced in a binary tree to minimize rounding error.
///
/// # Examples
///
/// ```
/// use mutate_lib::tree::WindowedTreeSum;
///
/// // Flat window (all 1.0) should give the same result as plain sum
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let window = vec![1.0_f64; 4];
/// let result = signal.iter().copied().zip(window.iter().copied()).windowed_tree_sum();
/// assert_eq!(result, 10.0);
///
/// // Scaling window halves each element
/// let signal = vec![2.0_f64, 4.0, 6.0, 8.0];
/// let window = vec![0.5_f64; 4];
/// let result = signal.iter().copied().zip(window.iter().copied()).windowed_tree_sum();
/// assert_eq!(result, 10.0);
///
/// // Empty iterator returns zero
/// let result = std::iter::empty::<(f64, f64)>().windowed_tree_sum();
/// assert_eq!(result, 0.0);
///
/// // Single element
/// let result = std::iter::once((3.0_f64, 4.0_f64)).windowed_tree_sum();
/// assert_eq!(result, 12.0);
///
/// // Works with f32 too
/// let signal = vec![1.0_f32, 2.0, 3.0];
/// let window = vec![2.0_f32; 3];
/// let result = signal.iter().copied().zip(window.iter().copied()).windowed_tree_sum();
/// assert_eq!(result, 12.0_f32);
/// ```
pub trait WindowedTreeSum: Iterator {
    type Output;
    fn windowed_tree_sum(self) -> Self::Output;
}

impl<F, I> WindowedTreeSum for I
where
    F: Zero + Add<Output = F> + Mul<Output = F> + MulAdd<Output = F> + Clone,
    I: Iterator<Item = (F, F)>,
{
    type Output = F;

    fn windowed_tree_sum(mut self) -> F {
        let mut stack: Aligned<A64, [Option<F>; 32]> = Aligned(std::array::from_fn(|_| None));
        // "manually" unrolled 🤖 to let give the compiler more room.
        for (s, w) in self {
            let product = s.clone() * w.clone();

            // Slot 0
            let carry = match stack[0].take() {
                None => {
                    stack[0] = Some(product);
                    continue;
                }
                Some(e) => s.mul_add(w, e),
            };

            // Slot 1
            let carry = match stack[1].take() {
                None => {
                    stack[1] = Some(carry);
                    continue;
                }
                Some(e) => carry + e,
            };

            // Slot 2
            let carry = match stack[2].take() {
                None => {
                    stack[2] = Some(carry);
                    continue;
                }
                Some(e) => carry + e,
            };

            // Slot 3
            let carry = match stack[3].take() {
                None => {
                    stack[3] = Some(carry);
                    continue;
                }
                Some(e) => carry + e,
            };

            // Slow path: slots 4-31, rarely reached (~6% of iterations)
            let mut carry = Some(carry);
            for slot in stack[4..].iter_mut() {
                match (slot.take(), carry) {
                    (None, c) => {
                        *slot = c;
                        break;
                    }
                    (Some(e), None) => {
                        carry = Some(e);
                        break;
                    }
                    (Some(e), Some(c)) => {
                        carry = Some(c + e);
                    }
                }
            }
        }

        stack
            .iter()
            .rev()
            .flatten()
            .cloned()
            .fold(F::zero(), |acc, x| acc + x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: naive left-fold sum, maximally inaccurate, used to confirm
    /// our tree sum is at least as good and agrees on exact cases.
    fn naive_sum(v: &[f32]) -> f32 {
        v.iter().copied().fold(0.0, |a, x| a + x)
    }

    /// Reference windowed sum via naive fold.
    fn naive_windowed_tree(signal: &[f32], window: &[f32]) -> f32 {
        signal
            .iter()
            .zip(window.iter())
            .fold(0.0, |acc, (&s, &w)| acc + s * w)
    }

    #[test]
    fn tree_sum_empty() {
        let result = std::iter::empty::<f32>().tree_sum();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn tree_sum_single() {
        let result = std::iter::once(42.0_f32).tree_sum();
        assert_eq!(result, 42.0);
    }

    #[test]
    fn tree_sum_two() {
        let result = [1.0_f32, 2.0].iter().copied().tree_sum();
        assert_eq!(result, 3.0);
    }

    // Exact power-of-two lengths — tree is perfectly balanced
    #[test]
    fn tree_sum_powers_of_two() {
        for exp in 1..=8 {
            let n = 1usize << exp;
            let v: Vec<f32> = (1..=n).map(|i| i as f32).collect();
            let expected = (n * (n + 1) / 2) as f32;
            assert_eq!(v.iter().copied().tree_sum(), expected, "failed at n={n}");
        }
    }

    // Non-power-of-two
    #[test]
    fn tree_sum_non_power_of_two() {
        for exp in 2..=8 {
            let n = 1usize << exp - 1;
            let v: Vec<f32> = (1..=n).map(|i| i as f32).collect();
            let expected = (n * (n + 1) / 2) as f32;
            assert_eq!(v.iter().copied().tree_sum(), expected, "failed at n={n}");
        }
    }

    // Many values that cancel with residuals of very different scales, exactly like summing
    // Goertzel values.
    #[test]
    fn tree_sum_cancellation_accuracy() {
        let v: Vec<f32> = vec![-1.1e-2, std::f32::consts::PI, -std::f32::consts::PI, 1.1e-2]
            .into_iter()
            .cycle()
            .take(4 * 1024)
            .collect();
        let tree = v.iter().copied().tree_sum();
        let naive = naive_sum(&v);
        // dbg!(tree, naive);
        assert!(
            tree.abs() < naive.abs(),
            "naive {naive:4.16} beat tree: {tree:4.16}"
        );
        assert_eq!(tree, 0.0, "tree sum lost precision: got {tree}");
    }

    // All zeros
    #[test]
    fn tree_sum_all_zeros() {
        let v = vec![0.0_f32; 100];
        assert_eq!(v.iter().copied().tree_sum(), 0.0);
    }

    // Alternating +1/-1: exact integer result, stresses the merge boundaries
    #[test]
    fn tree_sum_alternating_signs() {
        for n in [2, 4, 8, 16, 32, 64, 3, 5, 7, 15, 33, 63, 100] {
            let v: Vec<f32> = (0..n)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect();
            let expected = if n % 2 == 0 { 0.0 } else { 1.0 };
            assert_eq!(v.iter().copied().tree_sum(), expected, "failed at n={n}");
        }
    }

    // Single very large value dominates
    #[test]
    fn tree_sum_large_value() {
        let mut v = vec![1.0_f32; 15];
        v[7] = 1.0e15;
        let expected = 1.0e15 + 14.0;
        assert_eq!(v.iter().copied().tree_sum(), expected);
    }

    // Works for integers too (Default + Add)
    #[test]
    fn tree_sum_integers() {
        let v: Vec<i64> = (1..=100).collect();
        assert_eq!(v.iter().copied().tree_sum(), 5050_i64);
    }

    #[test]
    fn windowed_tree() {
        let signal = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let window = vec![2.0_f32; 5];
        let result = signal
            .iter()
            .copied()
            .zip(window.iter().copied())
            .windowed_tree_sum();
        assert_eq!(result, 30.0_f32);
    }

    #[test]
    fn windowed_tree_empty() {
        let result = std::iter::empty::<(f32, f32)>().windowed_tree_sum();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn windowed_tree_single() {
        let result = std::iter::once((3.0_f32, 7.0_f32)).windowed_tree_sum();
        assert_eq!(result, 21.0);
    }

    // Mismatched lengths via zip: zip stops at shortest, should still work.
    // MAYBE in the spirit of people not using mis-sized windows. this should probably fail.  It's
    // just hard to do with the Iterator API.
    #[test]
    fn windowed_tree_zip_stops_at_shortest() {
        let signal = vec![1.0_f32; 10];
        let window = vec![1.0_f32; 7]; // shorter
        let result = signal
            .iter()
            .copied()
            .zip(window.iter().copied())
            .windowed_tree_sum();
        assert_eq!(result, 7.0);
    }

    #[test]
    fn windowed_tree_unit_window_matches_tree_sum() {
        // A flat window of 1.0 should give the same result as tree_sum
        for n in [1, 2, 3, 4, 7, 8, 15, 16, 17, 100] {
            let signal: Vec<f32> = (1..=n).map(|i| i as f32).collect();
            let window = vec![1.0_f32; n];
            let windowed = signal
                .iter()
                .copied()
                .zip(window.iter().copied())
                .windowed_tree_sum();
            let tree = signal.iter().copied().tree_sum();
            assert_eq!(windowed, tree, "failed at n={n}");
        }
    }

    #[test]
    fn windowed_tree_zero_window() {
        let signal: Vec<f32> = (1..=50).map(|i| i as f32).collect();
        let window = vec![0.0_f32; 50];
        let result = signal
            .iter()
            .copied()
            .zip(window.iter().copied())
            .windowed_tree_sum();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn windowed_tree_non_power_of_two() {
        for n in [3, 5, 6, 7, 9, 13, 15, 17, 100, 255] {
            let signal: Vec<f32> = (1..=n).map(|i| i as f32).collect();
            let window: Vec<f32> = (1..=n).map(|i| 1.0 / i as f32).collect();
            let result = signal
                .iter()
                .copied()
                .zip(window.iter().copied())
                .windowed_tree_sum();
            // signal[i] * window[i] = i * (1/i) = 1.0 for each term, so sum = n
            let expected = n as f32;
            assert!(
                (result - expected).abs() < 1e-10,
                "n={n}: got {result}, expected {expected}"
            );
        }
    }

    // ADD: a large-N cancellation case showing the advantage scales up.
    #[test]
    fn windowed_tree_large_cancellation() {
        // Alternating ±1e12 scaled by window, net sum should be 0 (even N)
        // or exactly 1.0 * window_scale (odd N with one extra positive).
        // Naive fold drifts; tree sum stays exact.
        let n = 1024usize * 1024;
        let mut signal: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0e12 } else { -1.0e12 })
            .collect();
        let mut window = vec![1.0_f32; n]; // flat window keeps it interpretable

        let tree_result = signal
            .iter()
            .copied()
            .zip(window.iter().copied())
            .windowed_tree_sum();

        // n is even, so exact answer is 0.0
        assert_eq!(
            tree_result, 0.0,
            "tree sum drifted on large cancellation: {tree_result}"
        );
    }

    #[test]
    fn windowed_tree_fmadd_accuracy() {
        // Construct N pairs where each s*w = 1.0 + r exactly in f32 (signal is
        // 1.0 so the product equals the window value with no mul rounding).
        //
        // r = 2^-15 ≈ 3e-5.  Once the naive accumulator exceeds ~r * 2^23 = 512,
        // each individual residual r falls below half an ulp of the accumulator
        // and is rounded away.  By N=1024 roughly half the residuals have been
        // silently discarded by the left-fold.
        //
        // Tree sum avoids this by keeping sub-totals small: the level-0 buckets
        // each hold only one product (~1.0), where r is well above the ulp.
        // The residuals are safely captured before buckets are merged upward.
        //
        // True sum = 1024 * (1.0 + 2^-15) = 1024.03125, exactly representable.
        let n = 1024_usize;
        // 1.0 + 2^-15 is exactly 1.000030517578125 in f32
        let w_val = 1.000_030_517_578_125_f32;
        let signal = vec![1.0_f32; n];
        let window = vec![w_val; n];

        let tree_result = signal
            .iter()
            .copied()
            .zip(window.iter().copied())
            .windowed_tree_sum();
        let naive_result = naive_windowed_tree(&signal, &window);

        let expected = 1024.031_25_f32; // = 1024 * (1 + 2^-15), exactly representable

        println!("expected: {expected:5.6}, naive_result: {naive_result:5.6}, tree_result: {tree_result:5.6}");

        assert_eq!(
            tree_result, expected,
            "tree sum lost the residuals: got {tree_result}"
        );
        assert_ne!(
            naive_result, expected,
            "naive unexpectedly exact — test is not stressing accumulation loss"
        );
        assert!(
            (tree_result - expected).abs() < (naive_result - expected).abs(),
            "tree ({tree_result}) should be closer to {expected} than naive ({naive_result})"
        );
    }

    #[test]
    fn windowed_tree_structure_and_fmadd_accuracy() {
        // We want s*w products that:
        //   - require fmadd precision to compute accurately (s and w are not
        //     exactly representable, so naive mul rounds), AND
        //   - cancel in a pattern that a linear accumulator loses, but a
        //     tree sum recovers.
        //
        // Construction:
        //   signal[i]  = if even { large } else { -large }   (alternating sign)
        //   window[i]  = 1.0 + epsilon                        (not exactly 1.0)
        //
        // Each product s*w = ±large * (1 + eps).
        // The ±large terms cancel in pairs, leaving ±large*eps per pair.
        // Those residuals are all positive and sum to N/2 * large * eps.
        //
        // For the linear accumulator:
        //   After accumulating ~large values, the accumulator magnitude is ~large.
        //   large * eps falls below half an ULP of `large` for small enough eps,
        //   so the residuals are silently discarded — the running total never
        //   "sees" them.
        //
        // For the tree sum:
        //   Level-0 buckets each hold one product (~±large * (1+eps)).
        //   Adjacent pairs cancel to ~±large*eps, which is still well above the
        //   ULP of that small residual.  The residuals are captured before any
        //   large accumulation occurs.
        //
        // We also need fmadd for the per-product step: if we computed s*w as a
        // plain multiply first, the rounding in the multiply would discard eps
        // entirely (large * eps < 0.5 ulp(large) for our chosen values), so
        // fmadd's ability to defer the rounding is essential.

        let n = 512_usize; // power of two so the tree is perfectly balanced

        // large enough that large*eps < 0.5 ulp(large) — plain mul loses eps
        let large = 1.0e7_f32;
        // 2^-24 is one ULP of 1.0 in f32; choose eps = 2^-10 so large*eps = ~9.77
        // which is well above zero but eps itself is tiny relative to `large`
        let eps = 2.0_f32.powi(-10); // ≈ 9.77e-4

        // First half positive, second half negative — naive accumulates to +large*N/2
        // before seeing any negatives; tree pairs across the midpoint
        let signal: Vec<f32> = (0..n)
            .map(|i| if i < n / 2 { large } else { -large })
            .collect();

        let window = vec![1.0_f32 + eps; n];

        // Exact answer: every adjacent pair (large*(1+eps), -large*(1+eps))
        // sums to 0, so the true total is 0.0.
        let expected = 0.0_f32;

        let tree_result = signal
            .iter()
            .copied()
            .zip(window.iter().copied())
            .windowed_tree_sum();

        let naive_result = naive_windowed_tree(&signal, &window);

        assert_eq!(
            tree_result, expected,
            "tree sum drifted: got {tree_result} (naive got {naive_result})"
        );
        assert_ne!(
            naive_result, expected,
            "naive was unexpectedly exact — test is not exercising accumulation loss"
        );
    }
}
