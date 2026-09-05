// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Hermite Interpolation
//!
//! > Control! Control!
//! >
//! > - Darth Jar Jar
//!
//! Just some basic functions developed to approximately test IFFT convergence without requiring
//! perfect grid point alignment.  Of course interpolation adds error.  We slapped some compensation
//! on top to *mitigate*.  Since most wavelet generation methods obtain `d` trivially, this allows a
//! lower resolution `psi` to do the job much more cheaply.  Hermite interpolation vs evaluating
//! more grid points is a win.  The anchors are over-precise.

// NEXT Some characterization of the error would be appreciated.  If we ask for 1e-9 but the Hermite
// points are 1e-5, we're losing.  Only if we can avoid creating more 1e-11 points achieve 1e-9 is
// the trade worth it, and we need control!

use num_complex::Complex64;

use super::Accumulator;

// NEXT healthy dose of renaming

#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let s = a - b;
    let bv = s - a;
    (s, (a - (s - bv)) + (-b - bv))
}

/// Hermite basis in delta form, anchored on the nearer endpoint.
///
/// `a` and `b` are the one-sided second differences; they are the only place cancellation
/// occurs, and the two-sum residuals are folded back before they reach the Horner chain.
#[inline]
pub fn hermite_1d(p0: f64, p1: f64, m0: f64, m1: f64, f: f64) -> f64 {
    let (delta, delta_err) = two_diff(p1, p0);
    let (a, a_err) = two_diff(delta, m0);
    let (b, b_err) = two_diff(m1, delta);
    let a = a + (a_err + delta_err);
    let b = b + (b_err - delta_err);

    if f <= 0.5 {
        p0 + f * (b - a).mul_add(f, 2.0 * a - b).mul_add(f, m0)
    } else {
        let g = 1.0 - f;
        p1 + g * (a - b).mul_add(g, 2.0 * b - a).mul_add(g, -m1)
    }
}

/// Cubic Hermite reconstruction from a tap and its derivative.
///
/// `d` is the `u`-derivative up to a quarter turn (`dpsi/du == i * d`), so the slopes are exact
/// rather than estimated and the stencil stays at two taps.  Error is O(dt^4 |psi''''|).
///
/// `t` is in tap units; `resolution` converts the exact `u`-derivatives to that spacing.
pub fn resample_hermite(
    taps: &[Complex64],
    d: &[Complex64],
    t: f64,
    resolution: usize,
) -> Complex64 {
    let floor = t.floor();
    let f = t - floor;
    let i = floor as usize;

    let res = resolution as f64;
    let m0 = Complex64::new(-d[i].im / res, d[i].re / res);
    let m1 = Complex64::new(-d[i + 1].im / res, d[i + 1].re / res);

    let p0 = taps[i];
    let p1 = taps[i + 1];

    Complex64::new(
        hermite_1d(p0.re, p1.re, m0.re, m1.re, f),
        hermite_1d(p0.im, p1.im, m0.im, m1.im, f),
    )
}

/// Exact integral of the cubic Hermite reconstruction over `[i0, i1]`, in `u`.
///
/// Same stencil as `resample_hermite`, so this measures the area under the curve consumers
/// will actually see rather than the area under an independent quadrature rule.
pub fn hermite_integral(
    vals: &[Complex64],
    d: &[Complex64],
    i0: usize,
    i1: usize,
    resolution: usize,
) -> Complex64 {
    let dt = 1.0 / resolution as f64;
    let mut real: Accumulator<f64> = Accumulator::default();
    let mut imag: Accumulator<f64> = Accumulator::default();

    for i in i0..i1 {
        let m0 = Complex64::I * d[i] * dt;
        let m1 = Complex64::I * d[i + 1] * dt;
        let seg = (vals[i] + vals[i + 1]) * 0.5 + (m0 - m1) / 12.0;
        real.add(seg.re);
        imag.add(seg.im);
    }

    Complex64::new(real.sum(), imag.sum()) * dt
}
