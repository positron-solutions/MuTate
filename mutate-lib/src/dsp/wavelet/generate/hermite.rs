// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Hermite Interpolation
//!
//! > Control! Control!
//! >
//! > - Darth Jar Jar
//!
//! Just some basic functions developed to interpolate and integrate with curvature awareness we
//! already have.  Only intended for our usage, with rotated `dψ/du` and `resolution` describing the
//! fineness of grid points between periods `u`.
//!
//! ## Motivations
//!
//! At first, we just needed a way to test IFFT convergence without requiring perfect grid point
//! alignment.  Since most wavelet generation methods obtain `dψ/du` trivially, this allows a lower
//! resolution `ψ` to do the job much more cheaply.  The anchors are usually already over-precise,
//! so Hermite interpolation vs evaluating more grid points is probably a win.
//!
//! Of course interpolation adds error.  We slapped some compensation on top to *mitigate*.
//!
//! ## Lexicon
//!
//! See parent conventions for shared definitions.
//!
//! | symbol | Rust | object |
//! |---|---|---|
//! | `s` | `s` | position along the grid, `u·resolution`, so the cell is `⌊s⌋` |
//! | `f` | `f` | fractional position within the cell, always on `[0.0, 1.0)` |
//! | `Δu` | `delta_u` | grid spacing in `u`, `1/resolution` |
//! | `p₀`, `p₁` | `p0`, `p1` | the bracketing tap values |
//! | `m₀`, `m₁` | `m0`, `m1` | the tangents at those taps, scaled into cell units by `Δu` |
//!
//! `m` here is a Hermite tangent and not the crate's center sample.  A tangent is `2π·i·d·Δu`,
//! the storage convention reverted before the spacing goes on.
//!
//! `hermite_integral` integrates against `du`, so its result is `ψ` times periods.  The same sum
//! against `dν` is larger by `1/ρ`, which is what a consumer working in samples wants.

// NEXT Some characterization of the error would be appreciated.  If we ask for 1e-9 but the Hermite
// points are 1e-5, we're losing.  Only if we can avoid creating more 1e-11 points achieve 1e-9 is
// the trade worth it, and we need control!
// NEXT healthy dose of renaming
// MAYBE a newtype to protect rotated from de-rotated `d` from the storage channel?  Would affect all
// users, but caller that know the storage situation would be tempted to manually derotate before
// calling.

use std::f64::consts::TAU;

use num_complex::Complex64;

use super::Accumulator;

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
#[inline(always)]
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
/// Exact slopes hold the stencil at two taps.  Error is `O(Δu⁴ |ψ''''|)`.
///
/// `s` indexes the grid, so `s = u·resolution`, and `Δu` is the spacing the exact `u`-derivatives
/// are scaled into.
#[inline(always)]
pub fn resample_hermite(
    taps: &[Complex64],
    d: &[Complex64],
    s: f64,
    resolution: usize,
) -> Complex64 {
    let cell = s.floor();
    let f = s - cell;
    let i = cell as usize;

    let delta_u = 1.0 / resolution as f64;
    let m0 = tangent(d[i], delta_u);
    let m1 = tangent(d[i + 1], delta_u);

    let p0 = taps[i];
    let p1 = taps[i + 1];

    Complex64::new(
        hermite_1d(p0.re, p1.re, m0.re, m1.re, f),
        hermite_1d(p0.im, p1.im, m0.im, m1.im, f),
    )
}

/// Exact integral of the cubic Hermite reconstruction over the cells `i0..i1`, with `du` as the
/// measure, so the result carries one power of periods.
///
/// Same stencil as `resample_hermite`, so this measures the area under the curve consumers
/// will actually see rather than the area under an independent quadrature rule.
#[inline(always)]
pub fn hermite_integral(
    vals: &[Complex64],
    d: &[Complex64],
    i0: usize,
    i1: usize,
    resolution: usize,
) -> Complex64 {
    let delta_u = 1.0 / resolution as f64;
    let mut real = Accumulator::default();
    let mut imag = Accumulator::default();

    // The trapezoid plus the cubic's own correction, which the endpoint slopes supply exactly.
    for i in i0..i1 {
        let m0 = tangent(d[i], delta_u);
        let m1 = tangent(d[i + 1], delta_u);
        let seg = (vals[i] + vals[i + 1]) * 0.5 + (m0 - m1) / 12.0;
        real.add(seg.re);
        imag.add(seg.im);
    }

    Complex64::new(real.sum(), imag.sum()) * delta_u
}

/// Antiderivative of the cell's cubic in delta form, `∫₀^f p`, in cell units.
///
/// `p = p₀ + m₀f + (2a − b)f² + (b − a)f³` with `a = Δ − m₀`, `b = m₁ − Δ`, `Δ = p₁ − p₀`.
#[inline(always)]
fn cell_integral(p0: Complex64, p1: Complex64, m0: Complex64, m1: Complex64, f: f64) -> Complex64 {
    let delta = p1 - p0;
    let a = delta - m0;
    let b = m1 - delta;
    let f2 = f * f;
    p0 * f + m0 * (f2 * 0.5) + (a * 2.0 - b) * (f2 * f / 3.0) + (b - a) * (f2 * f2 * 0.25)
}

/// Integral of the cubic Hermite reconstruction over `[u_beg, u_end]`, with `du` as the measure.
///
/// `rho_grid` is the tap spacing in periods, which places the endpoints in the grid.  Whole cells
/// contribute their closed form and the two end cells contribute a fraction of theirs, so the sum
/// is the exact area under the same stencil `resample_hermite` reconstructs.
///
/// Caller is responsible that `u_beg <= u_end` and that both lie within the taps' reach.
pub fn integrate(
    taps: &[Complex64],
    d: &[Complex64],
    u_beg: f64,
    u_end: f64,
    rho_grid: f64,
) -> Complex64 {
    let s_beg = u_beg / rho_grid;
    let s_end = u_end / rho_grid;

    let last = taps.len() - 2;
    let i_beg = (s_beg.floor() as usize).min(last);
    let i_end = (s_end.floor() as usize).min(last);

    let cell = |i: usize| {
        (
            taps[i],
            taps[i + 1],
            tangent(d[i], rho_grid),
            tangent(d[i + 1], rho_grid),
        )
    };

    if i_beg == i_end {
        let (p0, p1, m0, m1) = cell(i_beg);
        let f0 = s_beg - i_beg as f64;
        let f1 = s_end - i_beg as f64;
        return (cell_integral(p0, p1, m0, m1, f1) - cell_integral(p0, p1, m0, m1, f0)) * rho_grid;
    }

    let mut real = Accumulator::default();
    let mut imag = Accumulator::default();
    let mut add = |seg: Complex64| {
        real.add(seg.re);
        imag.add(seg.im);
    };

    // Opening fraction.
    let (p0, p1, m0, m1) = cell(i_beg);
    let f0 = s_beg - i_beg as f64;
    add(cell_integral(p0, p1, m0, m1, 1.0) - cell_integral(p0, p1, m0, m1, f0));

    // Whole cells.
    for i in (i_beg + 1)..i_end {
        let (p0, p1, m0, m1) = cell(i);
        add((p0 + p1) * 0.5 + (m0 - m1) / 12.0);
    }

    // Closing fraction.
    let (p0, p1, m0, m1) = cell(i_end);
    add(cell_integral(p0, p1, m0, m1, s_end - i_end as f64));

    Complex64::new(real.sum(), imag.sum()) * rho_grid
}

/// The slope across one cell.  The stored channel is `−(i/2π) dψ/du`, so the quarter turn goes back on
/// before the spacing does.
#[inline(always)]
fn tangent(d: Complex64, delta_u: f64) -> Complex64 {
    Complex64::I * TAU * d * delta_u
}
