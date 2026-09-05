// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Inverse "Fast" Fourier Transform
//!
//! > One does not discover new lands without consenting to lose sight, for a very long time, of the
//! > shore.
//! >
//! > - Krishnan Kanthavel
//!
//! This is the plain Jane, the easy to trust, hard to get wrong baseline.  It suffers mainly from
//! issues that affect precision convergence at high `u`.  The reference methods developed to
//! confirm and characterize the precision and accuracy grew to replace Jane, so this poor module
//! has been demoted behind the `validation` feature.

use core::f64::consts::{LN_2, TAU};

use std::ops::Div;

use super::super::spec::Shape;

use num_complex::Complex64;
use rustfft::FftPlanner;

// NEXT `periods` is not a good way to determine truncation of the result.  Q determines how many
// points of any wavelet we will need, and the accuracy and precision track higher Q out to higher
// `u`.  It's the skinny tail where the IFFT begins to struggle.  We don't use the tail, but we
// don't use the IFFT either, so same same 이라고.

/// Grid for the IFFT generator.
///
/// `periods` is the reach of the returned half-taps in carrier cycles; `pad` extends the record
/// past that reach so the periodic wrap lands in the decayed tail; `resolution` is samples per
/// carrier cycle.  Only `record` and `n_fft` are derived, so no call site recomputes them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IfftSettings {
    pub periods: usize,
    pub pad: usize,
    pub resolution: usize,
}

impl IfftSettings {
    /// Periods + padding periods
    pub fn record(self) -> usize {
        self.periods + self.pad
    }

    /// Dimensions of the IFFT that will result
    pub fn n_fft(self) -> usize {
        self.record() * self.resolution
    }

    /// Folded legth in taps
    pub fn half_len(self) -> usize {
        self.periods * self.resolution + 1
    }

    /// Reach of the returned taps in `u`, i.e. the largest `u` a resample can index.
    pub fn reach(self) -> f64 {
        (self.half_len() - 1) as f64 / self.resolution as f64
    }

    /// Post-elbow reference settings.  Additional precision will buy only noise or regression.
    pub fn reference() -> Self {
        Self {
            periods: 8,
            pad: 64,
            resolution: 256,
        }
    }
}

impl Default for IfftSettings {
    /// Past both elbows in `ifft_precision_convergence` and `ifft_quadrature_precision`: pad is
    /// into the flat rows around 1e-14 at the near lobes, and resolution 1024 puts Hermite
    /// interpolation at ~2e-12 where the truncation floor no longer binds.
    fn default() -> Self {
        Self {
            periods: 8,
            pad: 32,
            resolution: 64,
        }
    }
}

/// Use an IFFT to generate time-domain solutions for psi, d, and dd.
///
/// The result covers `settings.periods` carrier cycles at `settings.resolution` samples per cycle,
/// so `psi.len() == settings.half_len()`.
///
/// `settings.n_fft()` is the transform size.
///
/// Each successive array is the `u`-derivative of the one before it up to a quarter turn, so
/// `dpsi/du == i * d` and `dd/du == i * dd`.  Consumers interpolating `psi` or `d` have exact
/// slopes available.  This enables both `psi` and `d` to use Hermitian interpolation.
pub(crate) fn morse_half_taps(
    shape: Shape,
    settings: IfftSettings,
) -> (Vec<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let record = settings.record() as f64;
    let half_len = settings.half_len();
    let resolution = settings.resolution as f64;

    let s_per_bin = shape.peak() / record;
    let zeta_per_bin = TAU / record;

    // Integer binade shift placing the spectral peak in [1, 2).  Exponent-only, so folding it
    // back out through `norm` restores the mantissa exactly.
    let peak = shape.peak();
    let shift = -((shape.beta * peak.ln() - peak.powf(shape.gamma)) / LN_2).floor();
    let norm = (-shift).exp2() / record;

    // The envelope underflows well before the Nyquist bin, so the significant support is far
    // shorter than the transform would be.  Collect it once and reuse across all samples.
    let bins: Vec<[f64; 4]> = (1..settings.n_fft() / 2)
        .map(|k| {
            let s = k as f64 * s_per_bin;
            let mag = ((shape.beta * s.ln() - s.powf(shape.gamma)) / LN_2 + shift).exp2();
            let zeta = k as f64 * zeta_per_bin;
            [k as f64, mag, mag * zeta, mag * zeta * zeta]
        })
        .filter(|b| b[1] > 0.0)
        .collect();

    let mut psi = Vec::with_capacity(half_len);
    let mut d = Vec::with_capacity(half_len);
    let mut dd = Vec::with_capacity(half_len);

    for i in 0..half_len {
        let u = i as f64 / resolution;
        let mut acc = [0.0f64; 6];
        let mut comp = [0.0f64; 6];

        for &[k, w_psi, w_d, w_dd] in &bins {
            let (sin, cos) = (TAU * frac_turns(k, u, record)).sin_cos();
            for (j, (w, trig)) in [
                (w_psi, cos),
                (w_psi, sin),
                (w_d, cos),
                (w_d, sin),
                (w_dd, cos),
                (w_dd, sin),
            ]
            .into_iter()
            .enumerate()
            {
                let x = w * trig;
                let x_err = w.mul_add(trig, -x);
                let t = acc[j] + x;
                comp[j] += (acc[j] - (t - x)) + (x - (t - acc[j])) + x_err;
                acc[j] = t;
            }
        }

        let mut out =
            |j: usize| Complex64::new((acc[j] + comp[j]) * norm, (acc[j + 1] + comp[j + 1]) * norm);
        psi.push(out(0));
        d.push(out(2));
        dd.push(out(4));
    }

    (psi, d, dd)
}

/// Fractional turns of `k*u/record`, carrying the low half of the product.
///
/// The phase reaches tens of thousands of radians at high `u`; reducing in turns before
/// scaling by TAU keeps the argument reduction out of `sin_cos`.
#[inline]
fn frac_turns(k: f64, u: f64, record: f64) -> f64 {
    let p_hi = k * u;
    let p_lo = k.mul_add(u, -p_hi);
    let q_hi = p_hi / record;
    let q_lo = ((-q_hi).mul_add(record, p_hi) + p_lo) / record;
    (q_hi - q_hi.floor()) + q_lo
}

#[inline]
fn two_sum_into(acc: &mut f64, comp: &mut f64, x: f64) {
    let t = *acc + x;
    *comp += (*acc - (t - x)) + (x - (t - *acc));
    *acc = t;
}

#[cfg(test)]
mod test {
    use super::super::{fmt_e, hermite, quadjet};
    use super::*;

    #[test]
    fn convergence_precision() {
        // Find the grid and padding elbows.  Remaining disagreement in other tests is disagreement
        // about the wavelet rather than disagreement with self. Tests show that padding has the
        // largest effect on accuracy.  Grid size has relatively little influence relative to padding.

        let shape = Shape::from_q(3.5, 3.0);

        let reference = IfftSettings::reference();

        let (ref_psi, ref_d, _) = morse_half_taps(shape, reference);

        let rel = |v: Complex64, r: Complex64| (v - r).norm() / r.norm();

        // Deliberately off-grid at every resolution in the sweep.
        let probes = [
            0.0, 0.123, 0.37, 0.89, 1.31, 2.07, 2.66, 3.42, 4.19, 5.88, 6.42, 7.911,
        ];

        let refs: Vec<_> = probes
            .iter()
            .map(|&u| {
                hermite::resample_hermite(
                    &ref_psi,
                    &ref_d,
                    u * reference.resolution as f64,
                    reference.resolution,
                )
            })
            .collect();

        let sweep =
            |name: &str, knob: &str, vary: &dyn Fn(u32) -> (usize, IfftSettings), rows: u32| {
                println!("\n=== {name} ===");
                print!("  {knob:>10} |");
                for u in probes {
                    print!(" {u:>9.2}");
                }
                println!();

                for i in 0..rows {
                    let (label, settings) = vary(i);
                    let (psi, d, _) = morse_half_taps(shape, settings);

                    print!("  {label:>10} |");
                    for (&u, &psi_ref) in probes.iter().zip(&refs) {
                        let t = u * settings.resolution as f64;
                        let e = rel(
                            hermite::resample_hermite(&psi, &d, t, settings.resolution),
                            psi_ref,
                        );
                        print!(" {:>9}", fmt_e(e));
                    }
                    println!();
                }
            };

        sweep(
            "Cranking pad",
            "pad",
            &|i| {
                let pad = (i as usize + 1) * 2;
                (pad, IfftSettings { pad, ..reference })
            },
            32,
        );
        sweep(
            "Cranking resolution",
            "resolution",
            &|i| {
                let resolution = (i as usize + 1) * 4;
                (
                    resolution,
                    IfftSettings {
                        resolution,
                        ..reference
                    },
                )
            },
            32,
        );

        println!("\n=== Shipping Grid ===");
        let shipping = IfftSettings::default();
        let (psi, d, _) = morse_half_taps(shape, shipping);

        let mut worst: f64 = 0.0;
        let mut worst_u: f64 = 0.0;

        for k in 0..=150 {
            let u = k as f64 * 0.05 + 0.011;
            let e = rel(
                hermite::resample_hermite(
                    &psi,
                    &d,
                    u * shipping.resolution as f64,
                    shipping.resolution,
                ),
                hermite::resample_hermite(
                    &ref_psi,
                    &ref_d,
                    u * reference.resolution as f64,
                    reference.resolution,
                ),
            );
            if e > worst {
                worst = e;
                worst_u = u;
            }

            if k % 10 == 0 {
                println!("  u: {u:>6.2}, err: {:>9}", fmt_e(e));
            }
        }

        println!("  worst over grid: {} at {:0.2}", fmt_e(worst), worst_u);
        assert!(worst < 1e-5);
    }

    #[test]
    fn quadrature_precision() {
        // Hermite quadrature of psi over half periods, against an over-provisioned instance of
        // itself.  This is the same stencil `resample_hermite` uses, so a column reports when the
        // stored psi stops carrying the area downstream will reconstruct.
        //
        // Bounds are quarter-period fractions straddling the carrier's zero crossings, so each
        // column is one signed lobe and successive columns alternate sign.  A whole-period window
        // would cancel and report on the cancellation instead of the grid.

        let shape = Shape::from_q(3.5, 3.0);

        let reference = IfftSettings::reference();

        // Half periods 0..10, covering the first five carrier periods.
        const LOBES: usize = 10;

        let (ref_psi, ref_d, _) = morse_half_taps(shape, reference);

        let bounds = |k: usize, resolution: usize| {
            ((1 + 2 * k) * resolution / 4, (3 + 2 * k) * resolution / 4)
        };

        let refs: Vec<Complex64> = (0..LOBES)
            .map(|k| {
                let (i0, i1) = bounds(k, reference.resolution);
                hermite::hermite_integral(&ref_psi, &ref_d, i0, i1, reference.resolution)
            })
            .collect();

        let sweep = |name: &str, knob: &str, vary: &dyn Fn(u32) -> IfftSettings, rows: u32| {
            println!("\n=== {name} ===");
            print!("  {knob:>10} |");
            for k in 0..LOBES {
                print!(" {:>9.2}", 0.25 + k as f64 * 0.5);
            }
            println!();

            for i in 0..rows {
                let settings = vary(i);
                let label = if knob == "pad" {
                    settings.pad
                } else {
                    settings.resolution
                };
                let (psi, d, _) = morse_half_taps(shape, settings);

                print!("  {label:>10} |");
                for (k, &area_ref) in refs.iter().enumerate() {
                    let (i0, i1) = bounds(k, settings.resolution);
                    let area = hermite::hermite_integral(&psi, &d, i0, i1, settings.resolution);
                    print!(" {:>9}", fmt_e((area - area_ref).norm() / area_ref.norm()));
                }
                println!();
            }
        };

        sweep(
            "Cranking pad",
            "pad",
            &|i| IfftSettings {
                pad: 2 * i as usize + 2,
                resolution: 1 << 12,
                ..reference
            },
            12,
        );
        // Truncation floor has to sit below the interpolation error at every row, so pad tracks the
        // resolution rather than sitting at a fixed over-provision.
        sweep(
            "Cranking resolution",
            "resolution",
            &|i| IfftSettings {
                pad: 32,
                resolution: 4 + (i as usize * 4),
                ..reference
            },
            12,
        );

        // Acceptance: the shipping grid sits past both elbows, so the area it carries differs from
        // the reference only by floor.
        println!("\n=== Shipping Grid ===");
        let shipping = IfftSettings::default();
        let (psi, d, _) = morse_half_taps(shape, shipping);
        let mut worst: f64 = 0.0;

        for (k, &area_ref) in refs.iter().enumerate() {
            let (i0, i1) = bounds(k, shipping.resolution);
            let area = hermite::hermite_integral(&psi, &d, i0, i1, shipping.resolution);
            let e = (area - area_ref).norm() / area_ref.norm();
            worst = worst.max(e);

            println!(
                "  u: {:>6.2}, area: {:+9.7}, err: {:>9}",
                0.25 + k as f64 * 0.5,
                area,
                fmt_e(e)
            );
        }

        println!("  worst over lobes: {}", fmt_e(worst));
        assert!(worst < 1e-6);
    }

    // MAYBE sigh... what to do with you.  Use of the production method as an... "oracle" to verify
    // the IFFT accuracy convergence basically gives us a smoke test that confirms or at least
    // demonstrates the error floor. Testing on-grid vs off-grid points (with and without Hermite
    // interpolation error) may be informative at some point.  An exercise left for the reader.
    #[test]
    fn accuracy_convergence() {
        // Reconstruct taps at arbitrary `u` by Hermite resampling against the QuadJet as an oracle.
        // Errors are normalized to the oracle's amplitude at the nearest half-phase, so a column
        // tracks the local extrema.

        // Amplitude has to clear the roundoff floor by this much before a cell reports.
        const LIVE: f64 = 16.0;
        const TOL: f64 = 1e-4; // Around u = 7, IFFT and QuadJet start to disagree enough to trip.

        let shape = Shape::from_q(3.5, 3.0);
        let oracle = quadjet::QuadJet::reference(shape);

        let base = IfftSettings {
            periods: 8,
            pad: 56,
            resolution: 1 << 8,
        };

        let rel = |v: Complex64, r: Complex64, scale: f64| (v - r).norm() / scale;

        // Snap to a half period so a probe is scored against the previous extremum it sits just
        // after, not the local error of a zero crossing which would produce an unstable ruler.
        let local_scale = |u: f64| {
            let anchor = (u * 2.0).floor() / 2.0;
            // XXX add back the D
            let reference = oracle.tap_at(anchor);
            reference.psi.norm()
        };

        // Deliberately off-grid at every resolution in the sweep.
        let probes = [
            0.37, 1.31, 2.66, 3.42, 4.19, 4.77, 5.31, 5.88, 6.42, 6.94, 7.54,
        ];

        let refs: Vec<_> = probes
            .iter()
            .map(|&u| (oracle.tap_at(u), local_scale(u)))
            .collect();

        let peak = refs[0].1;

        print!("\n  {:>10} |", "decay");
        for &(_, ps) in &refs {
            print!(" {:>9}", fmt_e(ps / peak));
        }
        println!();

        #[derive(Clone, Copy)]
        enum Tap {
            Psi,
            D,
        }

        let sweep =
            |tap: Tap, name: &str, knob: &str, vary: &dyn Fn(u32) -> IfftSettings, rows: u32| {
                let label_tap = match tap {
                    Tap::Psi => "psi",
                    Tap::D => "d",
                };
                println!("\n=== {name} ({label_tap}) ===");
                print!("  {knob:>10} |");
                for u in probes {
                    print!(" {u:>9.2}");
                }
                println!();

                for i in 0..rows {
                    let settings = vary(i);
                    let label = match knob {
                        "periods" => settings.periods,
                        "pad" => settings.pad,
                        "record" => settings.record(),
                        _ => settings.resolution,
                    };
                    let (psi, d, dd) = morse_half_taps(shape, settings);
                    let (value, slope) = match tap {
                        Tap::Psi => (&psi, &d),
                        Tap::D => (&d, &dd),
                    };
                    let reach = settings.reach();
                    let floor = (settings.n_fft() as f64).log2().sqrt() * f64::EPSILON * peak;

                    print!("  {label:>10} |");
                    for (&u, &(ref psi_ref, ps)) in probes.iter().zip(&refs) {
                        let (reference, scale) = match tap {
                            Tap::Psi => (psi_ref.psi, ps),
                            Tap::D => (Complex64::new(0.0, 0.0), 0.0), // XXX D Support
                        };

                        // Skip cells that won't have an index due to being too short.
                        if u >= reach - 1.0 / settings.resolution as f64 {
                            print!(" {:>9}", "-");
                            continue;
                        }

                        let t = u * settings.resolution as f64;
                        let cell = format!(
                            "{:>9}",
                            fmt_e(rel(
                                hermite::resample_hermite(value, slope, t, settings.resolution),
                                reference,
                                scale
                            ))
                        );

                        // LIES When the predicted IFFT noise floor is higher than the scale of the
                        // features we are attempting to draw.  The prediction may be loose or we may
                        // just be truly above the noise floor in average cases.  Leaving this here in
                        // case this failure mode is encountered.
                        if scale < floor * LIVE {
                            print!(" \x1b[2;90m{cell}\x1b[0m");
                        } else {
                            print!(" {cell}");
                        }
                    }

                    println!();
                }
            };

        // Each sweep is over-provisioned in the knobs it isn't varying, so the only thing that can
        // bind is the one in the label.
        // XXX Add back the D
        for tap in [Tap::Psi] {
            // Drive up the FFT grid points per period, resulting in finer sampling of the wavelet
            // in the time domain.
            sweep(
                tap,
                "Cranking resolution",
                "resolution",
                &|i| IfftSettings {
                    resolution: 4 + (i as usize * 4),
                    ..base
                },
                8,
            );

            // Add padding to constant period count.
            sweep(
                tap,
                "Cranking pad",
                "pad",
                &|i| IfftSettings {
                    pad: (1usize << i) - 1,
                    ..base
                },
                8,
            );

            // Fixed reach with a growing transform.  Once pad clears truncation the rows go flat
            // and stay flat across seven doublings of `n_fft`, so the floor doesn't scale with
            // transform size the way a coherent-sum bound would predict.  Extra record past that
            // point is free and useless.
            sweep(
                tap,
                "Cranking record",
                "record",
                &|i| IfftSettings {
                    periods: 6,
                    pad: (1usize << (i + 3)) - 6,
                    ..base
                },
                8,
            );
        }

        println!("\n=== Shipping Grid ===");

        let shipping = IfftSettings::default();
        let (psi, d, _) = morse_half_taps(shape, shipping);

        let noise = (shipping.n_fft() as f64).log2().sqrt() * f64::EPSILON * peak;
        let floor = noise.max(oracle.tol * peak);

        let reach = shipping.reach();

        let mut worst = (0.0f64, 0.0f64);
        let mut drift = (0.0f64, 0.0f64);
        let mut live_to = 0.0f64;

        println!(
            "  {:>6} | {:>9} | {:>9} | {:>9}",
            "u", "psi", "ifft-psi", "decay"
        );

        let steps = ((reach - 0.011) / 0.05) as u32;

        for k in 0..steps {
            let u = k as f64 * 0.05 + 0.011;
            let t = u * shipping.resolution as f64;
            let scale = local_scale(u);

            let e = rel(
                hermite::resample_hermite(&psi, &d, t, shipping.resolution),
                oracle.tap_at(u).psi,
                scale,
            );

            // Nearest grid index; the oracle is exact at any `u`, so evaluating it *there*
            // isolates solver-vs-solver drift from Hermite error.
            let idx = t.round() as usize;
            let u_grid = idx as f64 / shipping.resolution as f64;
            let g = rel(psi[idx], oracle.tap_at(u_grid).psi, scale);

            if k % 10 == 0 {
                println!(
                    "  {u:>6.2} | {:>9} | {:>9} | {:>9}",
                    fmt_e(e),
                    fmt_e(g),
                    fmt_e(scale / peak)
                );
            }

            if scale <= floor * LIVE {
                continue;
            }

            live_to = u;
            if g > drift.0 {
                drift = (g, u);
            }
            if e > worst.0 {
                worst = (e, u);
            }
        }

        println!(
            "\n  floor {} at peak scale (ifft {}, oracle {})",
            fmt_e(floor / peak),
            fmt_e(noise / peak),
            fmt_e(oracle.tol)
        );
        println!(
            "  psi: worst {} at u {:.3}, oracle drift {} at u {:.3}, live to u {:.2}",
            fmt_e(worst.0),
            worst.1,
            fmt_e(drift.0),
            drift.1,
            live_to
        );

        assert!(worst.0 < TOL);
        assert!(live_to > 5.5);
    }

    // XXX get this tested and actually just make it a speed test, who can get us psi and d for n
    // taps faster at a given error tolerance (where the IFFT is somewhat helpeless to control error
    // beyond a certain u, but let's be generous)
    #[ignore]
    #[test]
    fn full_resolution() {
        let shape = Shape::from_q(3.5, 3.0);
        let now = std::time::Instant::now();
        let _ = morse_half_taps(shape, IfftSettings::default());

        let elapsed = now.elapsed().as_micros();
        println!("elapsed: {:?}", elapsed);

        const SLOW_MICROS: u128 = 512000;
        assert!(elapsed < SLOW_MICROS, "FFT slow: {}µs ", elapsed);
    }
}
