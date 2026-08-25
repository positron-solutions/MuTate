// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spec
//!
//! > We reject: kings, presidents, and voting.
//! > We believe in: rough consensus and running code
//! >
//! > - David "D" Clark
//!
//! Define wavelet families.  The [`Spec`] builds the [`Plan`], handing over the realized
//! configuration choices in the process.  The primary knobs are the [`Truncation`] and the
//! [`Shape`].  You probably want to set truncation with `tail_db` and shape with `q`.
//!
//! ```rust
//! # use mutate_lib::dsp::wavelet::Spec;
//!
//! let spec = Spec::default()
//!     .q(3.5)
//!     .tail_db(-20.0);
//! ```

// MAYBE Gamma = 4 is not that wild, but has a flatter top and a steeper main lobe, things we are
// interested in.  It's possibly worth a bit of Q unless reassignment becomes broken.

use super::Plan;

use core::f64::consts::{LN_10, LN_2, PI, TAU};

use num_complex;

/// Controls Q and other critical tradeoffs of the Morse family wavelet parameters.  For exact
/// details, consult [real graphs](https://arxiv.org/pdf/1203.3380).
#[derive(Clone, Copy)]
pub struct Shape {
    /// The 𝛄 value of 3 results in a useful frequency domain uniformity and is the standard choice.
    pub gamma: f64,
    /// Adjusting beta at fixed gamma is adjusting Q.  The same envelope shape will be dilated over
    /// more carrier periods, resulting in more taps required to approximate the wavelet, the
    /// familiar time vs pitch resolution tradeoff.
    pub beta: f64,
}

impl Shape {
    /// `q` is the quality factor on the -3 dB energy width. Higher `q` narrows the band and costs
    /// proportionally more taps at a given center frequency.
    ///
    /// `q` is generally `bandwidth / center frequency`, and this can be used to estimate main lobe
    /// width.  The width at the beginning of the skirt, which must be controlled to avoid
    /// transition bands of downsampled inputs, is usually not more than twice as wide.
    pub fn from_q(q: f64, gamma: f64) -> Self {
        // p = 2.0 * LN_2.sqrt() * q
        // beta = p * p / gamma
        Shape {
            gamma,
            beta: (4.0 * LN_2) * (q * q / gamma),
        }
    }

    /// Morse wavelet time-bandwidth product (P = √(βγ)).  It can be interpreted as a surrogate for
    /// Q.
    pub fn p(&self) -> f64 {
        (self.beta * self.gamma).sqrt()
    }

    /// Return the **estimated** `q` for this shape.  Realized `q` will be close.
    pub fn q(&self) -> f64 {
        self.p() / (2.0 * LN_2.sqrt())
    }

    /// Argmax of the spectral envelope, in rad/sample.
    pub fn peak(&self) -> f64 {
        if self.gamma == 3.0 {
            // cbrt is just the fast path
            (self.beta / 3.0).cbrt()
        } else {
            (self.beta / self.gamma).powf(1.0 / self.gamma)
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Truncation {
    Sigmas(f64),
    FloorDb(f64),
}

/// Truncation leakage is bounded by the envelope tail, `exp(-n²/2)` at `n` sigmas.
/// `10/ln(10)` is the dB conversion, so a floor request and a sigma request are the
/// same number in two units.
impl Truncation {
    fn sigmas(&self) -> f64 {
        match *self {
            Truncation::Sigmas(n) => n,
            Truncation::FloorDb(db) => (db.abs() * LN_10 / 10.0).sqrt(),
        }
    }
}

/// Use to build coherent choices for a [`Plan`].  Choices of accuracy tradeoffs and support for
/// different `Q` and `load_quantum` are reconciled before building a `Plan`.  Re-use of these
/// builders to create a range of related plans is a convenient way to sweep across settings.
///
/// ```
/// # use mutate_lib::dsp::wavelet::Spec;
///
/// let spec = Spec::default();
/// ```
// Construction of incoherent choices should not be supported, but warning for incoherence or
// panicking for degenerate choices are both acceptable.
#[derive(Clone, Copy)]
pub struct Spec {
    shape: Shape,
    /// Error tolerance for the spectrum.
    grid_eps: f64,
    /// Error tolerance for filter length truncation
    truncation: Truncation,
    max_taps: usize,
    max_load_quantum: usize,
}

impl Default for Spec {
    fn default() -> Self {
        Spec {
            shape: Shape::from_q(3.0, 3.0),
            grid_eps: 1e-14,
            truncation: Truncation::Sigmas(4.0),
            max_taps: 0,
            max_load_quantum: 1,
        }
    }
}

impl Spec {
    /// Set mother wavelet [`Shape`].
    pub fn shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    /// `eps` is the spectral truncation floor relative to the peak. It sets how far the baked grid
    /// extends, and through that the tap count, but not the shape. 1e-8 lands near the f32 noise
    /// floor of the output taps.  1e-10 has measurable effects at 5.5sigmas.
    pub fn grid_eps(mut self, eps: f64) -> Self {
        self.grid_eps = eps;
        self
    }

    /// `max_taps` is a **hint** to allocate a larger scratch `Vec`, which will of course resize if
    /// necessary.  Uses [`Vec::with_capacity`](std::vec::Vec::with_capacity).  No effect on quality.
    pub fn max_taps(mut self, max_taps: usize) -> Self {
        self.max_taps = max_taps;
        self
    }

    /// Largest load quantum any bake will pass in. Sizes the rotor grid so the emitted span stays
    /// clear of the time-domain replica.
    pub fn max_load_quantum(mut self, q: usize) -> Self {
        self.max_load_quantum = q;
        self
    }

    /// Set error tolerance by envelope geometry.  This will control the selected N taps for each
    /// [`Bin`], so it's one of the most powerful knobs.  Values below 3.0 truncate too hard to
    /// ship until a better numerical solver is available.  Values over 5.5 begin grinding up the
    /// dust of departed f32s.
    pub fn sigmas(mut self, sigmas: f64) -> Self {
        self.truncation = Truncation::Sigmas(sigmas);
        self
    }

    /// Set error tolerance by desired noise floor, which is **estimated** to an envelope geometry
    /// and ultimately used to select a value for [`sigmas`].  A different way to attempt to say the
    /// same thing.  State dB if you do not measure sigmas.
    pub fn floor_db(mut self, db: f64) -> Self {
        self.truncation = Truncation::FloorDb(db);
        self
    }

    /// Half-span that sets tap count, and the grid extent that feeds it.
    fn spans(&self) -> (f64, f64) {
        let taps = self.truncation.sigmas() * self.shape.p();
        let grid = half_width_scaled(self.shape, self.grid_eps);

        // Quantum rounding plus the center tap extend the emitted span by up to 2q+1 samples,
        // worst at w0 = PI in scaled units.
        let pad = (2 * self.max_load_quantum + 1) as f64 * PI;
        // Rectangle rule on a uniform grid aliases rather than truncates: the error is the
        // time-domain replica at period TAU/du. Placing it a full eps half-width past the
        // emitted span puts the fold-back at eps.
        (taps, TAU / (taps + pad + grid.max(taps)))
    }

    pub fn plan(self) -> Plan {
        let (c, du) = self.spans();
        let du = snap(du);
        let (lo, m) = support(self.shape, du, self.grid_eps);

        // d = w*psi shares psi's support, so lo bounds both.
        let env = log_env(self.shape);
        let mut spec = vec![[0.0; 2]; m];
        for (j, s) in spec.iter_mut().enumerate().skip(lo) {
            let u = j as f64 * du;
            let p = env(u).exp();
            *s = [p, p * u];
        }

        Plan {
            shape: self.shape,
            c,
            du,
            spec,
            lo,
            buf: Vec::with_capacity(2 * (self.max_taps / 2 + self.max_load_quantum + 1)),
            max_load_quantum: self.max_load_quantum,
            floor: 0.0,
        }
    }
}

// XXX Envelop is going to die, so probably will this pretty soon
/// g(u) = beta*ln(u) - (beta/gamma)*u^gamma, normalized so g(1) = 0.
pub fn log_env(s: Shape) -> impl Fn(f64) -> f64 {
    let bg = s.beta / s.gamma;
    move |u| s.beta * u.ln() - bg * u.powf(s.gamma) + bg
}

// XXX Does not deserve to be free
/// Largest power-of-two step at or below `du`. Grids at different steps are then nested, so a
/// change to quantum or eps that doesn't cross a dyadic boundary leaves every u_j where it was.
pub fn snap(du: f64) -> f64 {
    du.log2().floor().exp2()
}

// XXX huh?
/// Roots of g(u) = ln(eps). g rises to 0 at u = 1 and falls after, so Newton from each
/// asymptotic branch converges monotonically inward.
fn roots(s: Shape, eps: f64) -> (f64, f64) {
    let g = log_env(s);
    let le = eps.ln();
    let dg = |u: f64| s.beta / u - s.beta * u.powf(s.gamma - 1.0);

    let solve = |mut u: f64| {
        for _ in 0..40 {
            u -= (g(u) - le) / dg(u);
        }
        u
    };
    (
        solve(eps.powf(1.0 / s.beta)),
        solve(1.0 + (2.0 * -le / s.beta).sqrt()),
    )
}

// XXX  Um.. dilate & truncate will kill this.
/// Grid indices bracketing the spectrum above `eps`: `[lo, m)`.
pub fn support(shape: Shape, du: f64, eps: f64) -> (usize, usize) {
    let (u_lo, u_hi) = roots(shape, eps);
    let lo = ((u_lo / du).ceil() as usize).max(1);
    (lo, (u_hi / du).floor() as usize + 1)
}

// XXX inline this if it does anything.
/// Half width in samples, times omega0. Pure function of shape and leakage.
fn half_width_scaled(s: Shape, eps: f64) -> f64 {
    let core = (2.0 * eps.recip().ln()).sqrt() * s.p();
    let tail = eps.recip().powf(1.0 / (2.0 * s.beta + 1.0));
    core.max(tail)
}
