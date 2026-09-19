// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Refine
//!
//! Polish off the worst of the spectral artifacts from the restriction.  Write the final weights to
//! `f32` output in a truncation and goal-aware manner.
//!
//! > If the state is captive to narrow interests, then why do you wait for the state to organize
//! > what you claim to be an overwhelming mass of your own interests?
//! >
//! > - Major Edward Wuncler III, Haliburton War Veteran
//!
//! ⚠️ This module is currently in full retreat.  Functionality drifted out from under it before it
//! shipped.  There is an intended role, but that role is only just taking shape.
//!
//! ## Motivation
//!
//! Restriction provides a shape.  The goal of the shape is to create robust transient response.
//! Restriction is intentionally not spectrally aware because unbounded spectral optimizations
//! compromise shape and transient response.
//!
//! All restrictions leave behind various kinds of spectral damage.  Finite support says we must
//! truncate that shape, at a minimum creating strong Gibbs ringing in the floor.  The truncated
//! filter mass is, with respect to the shape, free real-estate.  What the restriction's spectrally
//! blind truncation would simply remove, spectrally aware refinement can shape into something more
//! useful.
//!
//! The first side lobes, main lobe width (especially the transition band), and noise floor shape
//! may all have some low hanging fruit, features that can be directly tied to magnitude of some
//! specific weights.  If polishing these features away can be accomplished within the limited
//! support and the limited freedom of truncation, that's a win.
//!
//! In conclusion, the goal of this module is to only condition the worst spectral characteristics
//! so that unlucky truncation, load quantum, and omega interactions can be made more dependable.
//! The transient response is protected by limiting the scope of the polish to the filter mass that
//! is slated for removal by truncation.

// The replacement module is beginning to show us some requirements.  We want to know how much mass
// we have to work with.  Polar vs complex f64 is, in either case, just two f64 values, and we can
// make this a runtime adaptation to support polar vs non-polar... (go check )

use core::f64::consts::{FRAC_1_SQRT_2, TAU};

use super::{Shape, PEAK_GAIN};

/// Scale the cut envelope to unit center gain.
#[derive(Clone, Copy, Default)]
pub struct Refine;

pub struct Report {
    /// `ℓ² = Σ b_k u_k² / Σ b_k`
    pub locality: f64,
    /// Half power half width over the ideal `ρ/2Q`.
    pub width: f64,
    /// Deepest negative excursion of the response, `-inf` where it stays nonnegative.
    pub neg_db: f64,
    /// Tallest lobe past the main lobe.
    pub floor_db: f64,
}

impl Refine {
    /// Writes `out.len()` envelope taps normalized to `PEAK_GAIN`.  `env` is the restricted
    /// envelope over the window, of which the leading `out.len()` taps are read.
    pub fn refine_into(&self, shape: Shape, env: &[f64], rho: f64, out: &mut [f64]) -> Report {
        let eps = |j: usize| if j == 0 { 1.0 } else { 2.0 };

        let env = &env[..out.len()];
        let norm = PEAK_GAIN / env.iter().enumerate().map(|(j, a)| eps(j) * a).sum::<f64>();

        for (o, a) in out.iter_mut().zip(env) {
            *o = a * norm;
        }

        let (mass, spread) = out.iter().enumerate().fold((0.0, 0.0), |acc, (k, b)| {
            let u = k as f64 * rho;
            (acc.0 + eps(k) * b, acc.1 + eps(k) * b * u * u)
        });

        let (width, neg_db, floor_db, _) = response(shape, rho, out);

        Report {
            locality: (spread / mass).sqrt(),
            width,
            neg_db,
            floor_db,
        }
    }
}

/// Half power half width over the ideal, deepest negative excursion, lobe floor, and the image
/// about `−ω₀` arriving at detuning `2ω₀`.
///
///     T(f) = b_0 + 2 Σ_{k≥1} b_k cos(2π f k)
fn response(shape: Shape, rho: f64, b: &[f64]) -> (f64, f64, f64, f64) {
    let m = 8 * b.len();

    let at = |f: f64| {
        let sum: f64 = b[1..]
            .iter()
            .enumerate()
            .map(|(j, b)| b * (TAU * f * (j + 1) as f64).cos())
            .sum();
        b[0] + 2.0 * sum
    };

    let t: Vec<f64> = (0..=m).map(|i| at(0.5 * i as f64 / m as f64)).collect();

    let peak = t[0];
    let lobe = (1..=m).find(|&i| t[i] >= t[i - 1]).unwrap_or(m);
    let half = (1..=m).find(|&i| t[i] <= peak * FRAC_1_SQRT_2).unwrap_or(m);

    let (hi, lo) = (t[half - 1], t[half]);
    let frac = (hi - peak * FRAC_1_SQRT_2) / (hi - lo);
    let f_half = 0.5 * (half as f64 - 1.0 + frac) / m as f64;

    let floor = t[lobe..].iter().fold(0.0f64, |a, t| a.max(t.abs()));
    let sag = t.iter().fold(0.0f64, |a, &t| a.min(t));

    (
        f_half * 2.0 * shape.q() / rho,
        20.0 * (-sag / peak).log10(),
        20.0 * (floor / peak).log10(),
        20.0 * (at(2.0 * rho).abs() / peak).log10(),
    )
}

impl Report {
    /// Reads `b` as folded envelope taps already at `PEAK_GAIN`.
    pub fn measure(shape: Shape, rho: f64, b: &[f64]) -> Report {
        let (mass, spread) = b.iter().enumerate().fold((0.0, 0.0), |acc, (k, b)| {
            let (w, u) = (if k == 0 { 1.0 } else { 2.0 } * b, k as f64 * rho);
            (acc.0 + w, acc.1 + w * u * u)
        });

        let (width, neg_db, floor_db, _) = response(shape, rho, b);

        Report {
            locality: (spread / mass).sqrt(),
            width,
            neg_db,
            floor_db,
        }
    }
}

// mod.rs
