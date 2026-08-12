// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # The Wavelet
//!
//! > The traveler who fears the unknown road will eventually learn that known roads return
//! > to where they began.
//! >
//! > - Anthony L. Ray
//!
//!               #
//!               #
//!               #
//!
//!                ###
//!                ######
//!                ######
//!
//!       #########
//! ###############
//!    ############
//!
//!                ##############
//!                ####################
//!                ##############
//!
//!    ############
//! ###############
//!       #########
//!
//!                ######
//!                ######
//!                ###
//!
//!               #
//!               #
//!               #
//!
//! This module generates our wavelet tables. Morse wavelet is the first chosen implementation.
//!
//! - Easy to generate without dependencies
//! - Regarded as nice for time and frequency reassignment
//! - Parameterized
//!
//! None of these things matter unless people can see pretty pixels, so get to waving!
//!
//! ## Usage
//!
//! One [`Plan`] per Q. It holds the spectrum in a normalized frequency variable, so it is
//! independent of center frequency and sample rate, and every voice sharing that Q reuses it.
//!
//! ```
//! # use mutate::wavelet::Plan;
//! let plan = Plan::new(2.4, 3.0, 1e-8, 1.0);
//!
//! // size scratch for the lowest bin; every higher one fits.
//! let mut scratch = plan.scratch(plan.bin(50.0, 8000.0));
//!
//! let bin = plan.bin(1000.0, 8000.0);
//! let mut taps = vec![(0.0, 0.0); bin.len()];
//! plan.taps_into(bin, &mut scratch, &mut taps);
//! ```
//!
//! Reassignment consumes the plan and adds the two derivative spectra. The `d` and `t` taps
//! carry psi's normalization, so all three convolve against the same signal scale.
//!
//! ```
//! # use mutate::wavelet::Plan;
//! let plan = Plan::new(2.4, 3.0, 1e-8, 1.0).with_reassignment();
//!
//! // size scratch for the lowest bin; every higher one fits.
//! let mut scratch = plan.scratch(plan.bin(50.0, 8000.0));
//!
//! let bin = plan.bin(1000.0, 8000.0);
//! let n = bin.len();
//! let (mut psi, mut d, mut t) = (vec![(0.0, 0.0); n], vec![(0.0, 0.0); n], vec![(0.0, 0.0); n]);
//!
//! // d and t are multiplied by i at use.
//! plan.taps_into(bin, &mut scratch, &mut psi, &mut d, &mut t);
//! ```

// NOTE We have logarithmic bin spacings, but the cutoff frequencies that determine which downsample
// will be used are not particularly aware, so it's not expected that we can re-use exact bins in
// any kind of octave structure.  Mel scaling etc also defeats this, so there's no point.
// NEXT Run time of the bin generation test (not reflective of actual sample rates and Q) is about
// 50ms on a Zen2+ part.  This affects CWT startup time.

// 🤖 Heavy generation.  Should be pretty standard academic stuff, so not expecting a lot of
// surprises.  We will, for the most part, swiftly and knowingly eat shit if the wavelet is busted.
// Well-formalized work doesn't have a lot of wiggle room to violate the consistency of the
// formalism.

/// A center frequency resolved against a plan and a sample rate.
#[derive(Clone, Copy)]
pub struct Bin {
    // rad/sample at the decimated rate
    w0: f64,
    len: usize,
}

impl Bin {
    /// Number of taps.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Rotational velocity ദ്ദി(•̀ω-)✧ in radians.
    pub fn velocity(&self) -> f64 {
        self.w0
    }
}

/// Baking scratch for cleanups before writing out f32s.
pub struct Scratch {
    buf: Box<[(f64, f64)]>,
    max: usize,
}

impl Scratch {
    fn with_len(max: usize) -> Self {
        Scratch {
            buf: vec![(0.0, 0.0); 3 * max].into_boxed_slice(),
            max,
        }
    }

    /// Longest bin this scratch can bake.
    pub fn capacity(&self) -> usize {
        self.max
    }

    fn one(&mut self, n: usize) -> &mut [(f64, f64)] {
        &mut self.buf[..n]
    }

    fn three(&mut self, n: usize) -> [&mut [(f64, f64)]; 3] {
        let (a, rest) = self.buf.split_at_mut(self.max);
        let (b, c) = rest.split_at_mut(self.max);
        [&mut a[..n], &mut b[..n], &mut c[..n]]
    }
}

/// Shape-only. Everything here is independent of center frequency and rate.
pub struct Plan {
    shape: Shape,
    peak: f64,     // argmax of w^beta e^{-w^gamma}
    c: f64,        // half_width_scaled; taps = 2*ceil(c/omega0)+1
    du: f64,       // uniform step in u = w/w_peak
    psi: Vec<f64>, // psi at u_j = j*du, already L2-normalized
}

impl Plan {
    pub fn new(q: f64, gamma: f64, eps: f64, tail_a: f64) -> Self {
        let shape = Shape::from_q(q, gamma);
        let peak = shape.peak();
        let c = half_width_scaled(shape, eps, tail_a);

        // Aliasing in t occurs at tau/(du*omega0); taps ~ 2c/omega0, so du < pi/c.
        // At 0.8 the replica's own eps-edge clears ours by half a window half-width;
        // the core is Gaussian there, so leakage is far under eps at every Q we bake.
        let du = 0.8 * core::f64::consts::PI / c;

        // g(u) = beta*ln(u) - (beta/gamma)*u^gamma, g(1) = -beta/gamma.
        // Truncate where g(u) - g(1) < ln(eps), on both flanks.
        let (beta, bg) = (shape.beta, shape.beta / gamma);
        let le = eps.ln();
        let g = |u: f64| beta * u.ln() - bg * u.powf(gamma) + bg;

        let mut j = 1usize;
        while g(j as f64 * du) < le {
            j += 1;
        }
        // j is now the first grid point inside the band; walk past the peak to the far edge.
        while g(j as f64 * du) >= le {
            j += 1;
        }
        let m = j + 1;

        let mut psi = vec![0.0; m];
        fill_grid(&shape, du, &mut psi);
        let inv = psi.iter().map(|v| v * v).sum::<f64>().sqrt().recip();
        for v in psi.iter_mut() {
            *v *= inv;
        }

        Plan {
            shape,
            peak,
            c,
            du,
            psi,
        }
    }

    /// Adds the reassignment spectra. Costs two more grids of plan memory.
    pub fn with_reassignment(self) -> ReassignPlan {
        let m = self.psi.len();
        let (beta, gamma) = (self.shape.beta, self.shape.gamma);
        let mut spec = vec![[0.0; 3]; m];

        for j in 1..m {
            let w = self.peak * j as f64 * self.du;
            let wg = w.powf(gamma);
            let p = self.psi[j];
            spec[j] = [p, p * w, p * (beta / w - gamma * wg / w)];
        }
        ReassignPlan { plan: self, spec }
    }

    /// Calculate angular velocity for a bin at `center`, sampled at `rate`.  Ties together center
    /// radial velocity and tap count.
    ///
    /// `center` below `rate` and its Nyquist limit.
    pub fn bin(&self, center: f64, rate: f64) -> Bin {
        let nyquist = rate / 2.0;
        debug_assert!(
            (center < nyquist),
            "Center: {center:.1}Hz must be less than sample rate: {rate:.0}Hz and its Nyquist:
            {nyquist:.0}"
        );
        let w0 = core::f64::consts::TAU * center / rate;
        Bin {
            w0,
            len: 2 * (self.c / w0).ceil() as usize + 1,
        }
    }

    /// Writes `bin.len()` complex taps, centered, unit L2, DC removed.
    pub fn taps_into(&self, bin: Bin, scratch: &mut Scratch, out: &mut [(f32, f32)]) {
        let buf = scratch.one(bin.len);
        self.transform(bin, buf);
        Self::center_l2(buf);

        // NOTE this fiddly code was generated while scanning for low-hanging sources of error.  The
        let n = bin.len;
        let half = n / 2;
        let mut err = 0.0f64;
        for i in half..n {
            let (r, im) = buf[i];
            let w = if i == half { 1.0 } else { 2.0 };
            let rr = (r + err / w) as f32;
            err += w * (r - rr as f64);
            let ii = im as f32;
            out[i] = (rr, ii);
            out[n - 1 - i] = (rr, -ii);
        }
    }

    /// Single-spectrum rotor walk over psi.  Takes advantage of odd symmetry.
    fn transform(&self, bin: Bin, out: &mut [(f64, f64)]) {
        let n = out.len();
        let half = n / 2;
        let step = self.du * bin.w0;

        for i in half..n {
            let d = step * (i - half) as f64;
            let (ds, dc) = d.sin_cos();
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for &p in &self.psi[1..] {
                let (nr, ni) = (cr * dc - ci * ds, cr * ds + ci * dc);
                cr = nr;
                ci = ni;
                re += p * cr;
                im += p * ci;
            }
            out[i] = (re, im);
            out[n - 1 - i] = (re, -im);
        }
    }

    /// DC removal, then unit L2. Returns the scale applied, which the
    /// reassignment taps need in order to share psi's normalization.
    fn center_l2(out: &mut [(f64, f64)]) -> f64 {
        let n = out.len();
        let half = n / 2;
        let len = n as f64;

        // Hermitian: pairs contribute 2*re, center once, imag cancels.
        let mr = (2.0 * out[half + 1..].iter().map(|s| s.0).sum::<f64>() + out[half].0) / len;

        let mut e = 0.0;
        for s in out.iter_mut() {
            s.0 -= mr;
            e += s.0 * s.0 + s.1 * s.1;
        }

        let inv = e.sqrt().recip();
        for s in out.iter_mut() {
            s.0 *= inv;
            s.1 *= inv;
        }
        inv
    }

    /// DC removal on psi's scale.
    fn center_scale(out: &mut [(f64, f64)], norm: f64) {
        let n = out.len();
        let half = n / 2;
        let mr = (2.0 * out[half + 1..].iter().map(|s| s.0).sum::<f64>() + out[half].0) / n as f64;
        for s in out.iter_mut() {
            s.0 = (s.0 - mr) * norm;
            s.1 *= norm;
        }
    }

    /// Scratch sized for `bin` and every smaller one. Pass the lowest center
    /// frequency you intend to bake at the same rate.
    pub fn scratch(&self, bin: Bin) -> Scratch {
        Scratch::with_len(bin.len)
    }
}

/// A [`Plan`] carrying the two reassignment spectra.
pub struct ReassignPlan {
    plan: Plan,
    spec: Vec<[f64; 3]>, // [psi, d, t] per grid point
}

impl ReassignPlan {
    /// Writes psi, pitch weights, and time weights for `bin`. All three output
    /// slices must be `bin.len()`.
    ///
    /// `d` and `t` are to be multiplied by `i` at use.
    pub fn taps_into(
        &self,
        bin: Bin,
        scratch: &mut Scratch,
        psi: &mut [(f32, f32)],
        d: &mut [(f32, f32)],
        t: &mut [(f32, f32)],
    ) {
        let [bp, bd, bt] = scratch.three(bin.len);
        self.transform3(bin, bp, bd, bt);

        let norm = Plan::center_l2(bp);
        Plan::center_scale(bd, norm);
        Plan::center_scale(bt, norm);

        for (out, src) in [(&mut *psi, &*bp), (&mut *d, &*bd), (&mut *t, &*bt)] {
            for (o, &(r, i)) in out.iter_mut().zip(src.iter()) {
                *o = (r as f32, i as f32);
            }
        }
    }

    /// Rotor walk over the interleaved spectra, one pass for all three outputs.
    /// Takes advantage of odd symmetry.
    fn transform3(
        &self,
        bin: Bin,
        psi: &mut [(f64, f64)],
        d: &mut [(f64, f64)],
        t: &mut [(f64, f64)],
    ) {
        let n = psi.len();
        let half = n / 2;
        let step = self.plan.du * bin.w0;

        for i in half..n {
            let dt = step * (i - half) as f64;
            let (ds, dc) = dt.sin_cos();
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            let (mut a0, mut a1, mut a2) = ((0.0f64, 0.0f64), (0.0f64, 0.0f64), (0.0f64, 0.0f64));

            for &[sp, sd, st] in &self.spec[1..] {
                let (nr, ni) = (cr * dc - ci * ds, cr * ds + ci * dc);
                cr = nr;
                ci = ni;
                a0 = (a0.0 + sp * cr, a0.1 + sp * ci);
                a1 = (a1.0 + sd * cr, a1.1 + sd * ci);
                a2 = (a2.0 + st * cr, a2.1 + st * ci);
            }

            for (out, acc) in [(&mut *psi, a0), (&mut *d, a1), (&mut *t, a2)] {
                out[i] = acc;
                out[n - 1 - i] = (acc.0, -acc.1);
            }
        }
    }
}

/// ReassignPlan is basically also a Plan and has access to its methods.
impl core::ops::Deref for ReassignPlan {
    type Target = Plan;
    fn deref(&self) -> &Plan {
        &self.plan
    }
}

#[derive(Clone, Copy)]
pub struct Shape {
    pub gamma: f64,
    pub beta: f64,
}

impl Shape {
    /// Q by the -3 dB energy width: dF/F = 2*sqrt(ln2)/P.
    pub fn from_q(q: f64, gamma: f64) -> Self {
        let p = 1.6651 * q;
        Shape {
            gamma,
            beta: p * p / gamma,
        }
    }
    pub fn p(&self) -> f64 {
        (self.beta * self.gamma).sqrt()
    }

    pub fn peak(&self) -> f64 {
        if self.gamma == 3.0 {
            (self.beta / 3.0).cbrt()
        } else {
            (self.beta / self.gamma).powf(1.0 / self.gamma)
        }
    }
}

fn fill_grid(shape: &Shape, du: f64, lo: usize, psi: &mut [f64]) {
    let (beta, gamma) = (shape.beta, shape.gamma);
    let bg = beta / gamma;
    for (j, p) in psi.iter_mut().enumerate().skip(lo) {
        let u = j as f64 * du;
        *p = (beta * u.ln() - bg * u.powf(gamma) + bg).exp();
    }
}

/// Half width in samples, times omega0. Pure function of shape and leakage.
pub fn half_width_scaled(s: Shape, eps: f64, tail_a: f64) -> f64 {
    let core = (2.0 * eps.recip().ln()).sqrt() * s.p();
    let tail = (tail_a / eps).powf(1.0 / (2.0 * s.beta + 1.0));
    core.max(tail)
}

#[cfg(test)]
mod test {
    use super::*;

    const BINS: usize = 1024;
    const RATE: f64 = 48_000.0;

    fn plan(q: f64, eps: f64) -> Plan {
        Plan::new(q, 3.0, eps, 1.0)
    }

    fn taps(plan: &Plan, bin: Bin) -> Vec<(f32, f32)> {
        let mut s = plan.scratch(bin);
        let mut t = vec![(0.0, 0.0); bin.len()];
        plan.taps_into(bin, &mut s, &mut t);
        t
    }

    fn mag((re, im): (f32, f32)) -> f64 {
        let (re, im) = (re as f64, im as f64);
        (re * re + im * im).sqrt()
    }

    #[test]
    fn print_tap_energy() {
        for eps in [1e-4f64, 1e-6, 1e-8] {
            let plan = plan(3.0, eps);
            for (fc, sr) in [(1000.0f64, 8000.0f64), (500.0, 8000.0), (250.0, 8000.0)] {
                let bin = plan.bin(fc, sr);
                let t = taps(&plan, bin);

                // Edge magnitude relative to center: what the truncation actually costs.
                println!(
                    "eps {:.0e}  fc {:>6.0}  omega0 {:.4}  m {:>4}  taps {:>4}  edge/peak {:.3e}",
                    eps,
                    fc,
                    bin.velocity(),
                    plan.psi.len(),
                    bin.len(),
                    mag(t[0]) / mag(t[bin.len() / 2])
                );
            }
        }
    }

    #[test]
    fn print_taps() {
        let plan = plan(3.0, 1e-8);
        let bin = plan.bin(1000.0, 8000.0);
        let t = taps(&plan, bin);
        let n = bin.len();

        let max = t.iter().copied().map(mag).fold(0.0f64, f64::max);

        println!(
            "\n=== TAPS ({} complex, omega0 {:.4}) ===",
            n,
            bin.velocity()
        );
        println!(
            "{:>4} {:>10} {:>10} {:>10}  real",
            "n", "real", "imag", "|w|"
        );
        for (j, &(re, im)) in t.iter().enumerate() {
            let col = ((re as f64 / max) * 20.0).round() as isize;
            let bar = if col >= 0 {
                format!("{}{}", " ".repeat(20), "#".repeat(col as usize))
            } else {
                format!(
                    "{}{}",
                    " ".repeat((20 + col) as usize),
                    "#".repeat((-col) as usize)
                )
            };
            println!(
                "{:>4} {:>10.5} {:>10.5} {:>10.5}  {}",
                j as isize - (n / 2) as isize,
                re,
                im,
                mag((re, im)),
                bar
            );
        }

        let e: f64 = t
            .iter()
            .map(|&(r, i)| (r as f64).powi(2) + (i as f64).powi(2))
            .sum();
        let (sr, si) = t.iter().fold((0.0f64, 0.0f64), |a, s| {
            (a.0 + s.0 as f64, a.1 + s.1 as f64)
        });
        println!("energy {:.7}  sum ({:.3e}, {:.3e})", e, sr, si);
    }

    #[test]
    fn print_gamma_sweep() {
        println!("\n=== ENVELOPE vs GAMMA (Q = 2.4) ===");
        for gamma in [1.0f64, 2.0, 3.0, 6.0] {
            let plan = Plan::new(2.4, gamma, 1e-8, 1.0);
            let bin = plan.bin(1000.0, 8000.0);
            let t = taps(&plan, bin);
            let n = bin.len();

            let mags: Vec<f64> = t.iter().copied().map(mag).collect();
            let max = mags.iter().fold(0.0f64, |a, &b| a.max(b));
            let ctr = (n / 2) as f64;
            let m: f64 = mags
                .iter()
                .enumerate()
                .map(|(j, &v)| (j as f64 - ctr) * v * v)
                .sum();
            let e: f64 = mags.iter().map(|v| v * v).sum();

            println!(
                "\ngamma = {:.1}  taps {}  centroid offset = {:+.3}",
                gamma,
                n,
                m / e
            );
            for (j, &v) in mags.iter().enumerate() {
                println!(
                    "{:>4} {}",
                    j as isize - (n / 2) as isize,
                    "#".repeat((v / max * 40.0).round() as usize)
                );
            }
        }
    }

    #[test]
    fn print_spectrum() {
        let plan = plan(3.0, 1e-6).with_reassignment();

        println!(
            "\n=== SPECTRUM ({} grid points, u = w/w_peak) ===",
            plan.spec.len()
        );
        println!(
            "{:>4} {:>8} {:>8} {:>12} {:>12} {:>12}",
            "j", "u", "omega", "psi", "dpsi", "tpsi"
        );
        for (j, &[p, pd, pt]) in plan.spec.iter().enumerate() {
            if p < 1e-7 {
                continue;
            }
            let u = j as f64 * plan.du;
            println!(
                "{:>4} {:>8.4} {:>8.4} {:>12.6} {:>12.6} {:>12.6}",
                j,
                u,
                plan.peak * u,
                p,
                pd,
                pt
            );
        }

        let pk = (1..plan.spec.len())
            .max_by(|&a, &b| plan.spec[a][0].total_cmp(&plan.spec[b][0]))
            .unwrap();
        println!("peak at u = {:.4}, wanted 1.0", pk as f64 * plan.du);
    }

    #[test]
    fn print_reassignment() {
        let plan = plan(3.0, 1e-6).with_reassignment();
        let bin = plan.bin(1000.0, 8000.0);
        let mut s = plan.scratch(bin);
        let n = bin.len();
        let (mut psi, mut d, mut t) = (
            vec![(0.0, 0.0); n],
            vec![(0.0, 0.0); n],
            vec![(0.0, 0.0); n],
        );
        plan.taps_into(bin, &mut s, &mut psi, &mut d, &mut t);

        println!("\n=== REASSIGNMENT ({} taps) ===", n);
        println!("{:>4} {:>12} {:>12} {:>12}", "n", "|psi|", "|d|", "|t|");
        for j in 0..n {
            println!(
                "{:>4} {:>12.6} {:>12.6} {:>12.6}",
                j as isize - (n / 2) as isize,
                mag(psi[j]),
                mag(d[j]),
                mag(t[j])
            );
        }
    }

    /// Bakes the full bank at production scale. Ignored by default; run with
    ///
    /// ```text
    /// cargo test --release wavelet::test::bake_bank -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn bake_bank() {
        use mutate_lib::dsp::bank;

        let start = std::time::Instant::now();
        let plan = Plan::new(20.0, 3.0, 1e-8, 1.0);
        let bins = bank::bins(2_000.0, 20_000.0, BINS);
        println!("planning time: {:?}µs", start.elapsed().as_micros());

        let voices: Vec<Bin> = bins
            .iter()
            .map(|b| b.center)
            .map(|c| plan.bin(c, RATE))
            .collect();

        // bins are ascending, so the first voice is the longest.
        let mut scratch = plan.scratch(voices[0]);

        let total: usize = voices.iter().map(|b| b.len()).sum();
        let mut taps = vec![(0.0f32, 0.0f32); total];
        let mut offsets = Vec::with_capacity(voices.len());

        let mut cursor = 0;

        let start = std::time::Instant::now();
        for &bin in &voices {
            let n = bin.len();
            offsets.push(cursor);
            plan.taps_into(bin, &mut scratch, &mut taps[cursor..cursor + n]);
            cursor += n;
        }

        let e: f64 = taps
            .iter()
            .map(|&(r, i)| (r as f64) * (r as f64) + (i as f64) * (i as f64))
            .sum();
        let elapsed = start.elapsed();

        let n = voices[0].len();
        let lo = &taps[..n];
        let max = lo
            .iter()
            .map(|&(r, _)| (r as f64).abs())
            .fold(0.0, f64::max);

        println!(
            "\n=== LOWEST BIN ({:.0}Hz, {} taps, omega0 {:.5}) ===",
            bins[0].center,
            n,
            voices[0].velocity()
        );
        for (j, &(re, _)) in lo.iter().enumerate() {
            let col = ((re as f64 / max) * 30.0).round() as isize;
            let bar = if col >= 0 {
                format!("{}{}", " ".repeat(30), "#".repeat(col as usize))
            } else {
                format!(
                    "{}{}",
                    " ".repeat((30 + col) as usize),
                    "#".repeat((-col) as usize)
                )
            };
            println!("{:>6} {:>12.7} {}", j as isize - (n / 2) as isize, re, bar);
        }

        println!(
            "voices {} of {}  taps {}  longest {}  shortest {}  mean energy {:.6}",
            voices.len(),
            BINS,
            total,
            voices[0].len(),
            voices[voices.len() - 1].len(),
            e / voices.len() as f64,
        );

        println!("bin filling time: {:?}µs", elapsed.as_micros());
    }
}
