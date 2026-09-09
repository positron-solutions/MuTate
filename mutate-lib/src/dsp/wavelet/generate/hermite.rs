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

/// One-sided second differences of the cell, with the two-sum residuals folded back.
///
/// `a = Δ − m₀`, `b = m₁ − Δ`, `Δ = p₁ − p₀`.
#[inline(always)]
fn deltas(p0: f64, p1: f64, m0: f64, m1: f64) -> (f64, f64) {
    let (delta, delta_err) = two_diff(p1, p0);
    let (a, a_err) = two_diff(delta, m0);
    let (b, b_err) = two_diff(m1, delta);
    (a + (a_err + delta_err), b + (b_err - delta_err))
}

/// Hermite basis in delta form, anchored on the nearer endpoint.
///
/// `a` and `b` are the one-sided second differences; they are the only place cancellation
/// occurs, and the two-sum residuals are folded back before they reach the Horner chain.
#[inline(always)]
pub fn hermite_1d(p0: f64, p1: f64, m0: f64, m1: f64, f: f64) -> f64 {
    let (a, b) = deltas(p0, p1, m0, m1);

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

/// The slope across one cell.  The stored channel is `−(i/2π) dψ/du`, so the quarter turn goes back on
/// before the spacing does.
#[inline(always)]
fn tangent(d: Complex64, delta_u: f64) -> Complex64 {
    Complex64::I * TAU * d * delta_u
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

/// `∫_{f₀}^{f₁} p` in cell units, accumulated term by term.
///
/// `p = p₀ + m₀f + (2a − b)f² + (b − a)f³`, integrated with the power differences factored,
/// `f₁ᵏ⁺¹ − f₀ᵏ⁺¹ = h·Sₖ₊₁`.
#[inline(always)]
fn cell_span_1d(
    acc: &mut Accumulator<f64>,
    p0: f64,
    m0: f64,
    a: f64,
    b: f64,
    f0: f64,
    f1: f64,
    h: f64,
) {
    let f0_2 = f0 * f0;
    let f1_2 = f1 * f1;

    let s2 = f1 + f0;
    let s3 = f1_2 + f1 * f0 + f0_2;
    let s4 = s2 * (f1_2 + f0_2);

    let q3 = h * s3 / 3.0;
    let q4 = h * s4 * 0.25;

    acc.add(h * p0);
    acc.add(h * m0 * s2 * 0.5);
    acc.add(2.0 * a * q3);
    acc.add(-b * q3);
    acc.add(b * q4);
    acc.add(-a * q4);
}

#[inline(always)]
fn cell_span(
    real: &mut Accumulator<f64>,
    imag: &mut Accumulator<f64>,
    p0: Complex64,
    p1: Complex64,
    m0: Complex64,
    m1: Complex64,
    f0: f64,
    f1: f64,
    h: f64,
) {
    let (a_re, b_re) = deltas(p0.re, p1.re, m0.re, m1.re);
    let (a_im, b_im) = deltas(p0.im, p1.im, m0.im, m1.im);
    cell_span_1d(real, p0.re, m0.re, a_re, b_re, f0, f1, h);
    cell_span_1d(imag, p0.im, m0.im, a_im, b_im, f0, f1, h);
}

/// A whole cell, trapezoid and cubic correction kept apart.
#[inline(always)]
fn whole_cell(
    real: &mut Accumulator<f64>,
    imag: &mut Accumulator<f64>,
    p0: Complex64,
    p1: Complex64,
    m0: Complex64,
    m1: Complex64,
) {
    real.add(p0.re * 0.5);
    real.add(p1.re * 0.5);
    real.add(m0.re / 12.0);
    real.add(-m1.re / 12.0);
    imag.add(p0.im * 0.5);
    imag.add(p1.im * 0.5);
    imag.add(m0.im / 12.0);
    imag.add(-m1.im / 12.0);
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
    let last = taps.len() - 2;
    let i_beg = ((u_beg / rho_grid).floor() as usize).min(last);
    let i_end = ((u_end / rho_grid).floor() as usize).min(last);

    // Fractions reduced against `u` so they carry no error from the grid position.
    let f_beg = (-(i_beg as f64)).mul_add(rho_grid, u_beg) / rho_grid;
    let f_end = (-(i_end as f64)).mul_add(rho_grid, u_end) / rho_grid;

    let cell = |i: usize| {
        (
            taps[i],
            taps[i + 1],
            tangent(d[i], rho_grid),
            tangent(d[i + 1], rho_grid),
        )
    };

    let mut real = Accumulator::default();
    let mut imag = Accumulator::default();

    if i_beg == i_end {
        let (p0, p1, m0, m1) = cell(i_beg);
        let h = (u_end - u_beg) / rho_grid;
        cell_span(&mut real, &mut imag, p0, p1, m0, m1, f_beg, f_end, h);
        return Complex64::new(real.sum(), imag.sum()) * rho_grid;
    }

    // Opening fraction.
    let (p0, p1, m0, m1) = cell(i_beg);
    cell_span(
        &mut real,
        &mut imag,
        p0,
        p1,
        m0,
        m1,
        f_beg,
        1.0,
        1.0 - f_beg,
    );

    // Whole cells.
    for i in (i_beg + 1)..i_end {
        let (p0, p1, m0, m1) = cell(i);
        whole_cell(&mut real, &mut imag, p0, p1, m0, m1);
    }

    // Closing fraction.
    let (p0, p1, m0, m1) = cell(i_end);
    cell_span(&mut real, &mut imag, p0, p1, m0, m1, 0.0, f_end, f_end);

    Complex64::new(real.sum(), imag.sum()) * rho_grid
}
