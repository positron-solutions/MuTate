// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Parks McClellen Remez (PMR)
//!
//! This utility will output FIR filter weights for downsampling. The `pm_remez` crate was pretty
//! heavy, so this small binary is the only thing that depends on it.
//!
//! We have a neat trick on our side:
//!
//! - We will have already analyzed high frequency bins using the normal input stream.
//! - We can begin our transition in bins we won't read in the output anyway.
//! - That region is consecutive with the region that will fold into it after downsampling.
//! - We use both already read bins and their folding mirror as an ultra-wide transition band.
//!
//! We're cheating and letting some noise fold in, but are attenuating it enough that it's
//! definitely not growing.  Anything that remains will be further obliterated by our other filters
//! and The important part is that **no noise folds into bands we will be looking at.**
//!
//! After exploiting that design trick, we make some tradeoffs based on what we're doing.
//!
//! - Noise floor in the stop, which would fold into our new bands of interest, must be crushed to
//!   pieces.  Pieces!
//! - Noise from the fold in the transition should at least be attenuated.  We don't want our pass
//!   band filters to be working harder due to folded noise that is both more intense and nearer in
//!   pitch.
//! - Acceptable ripple in the pass (signal we are definitely looking at).  A few dB of droop in the
//!   pass will be nothing compared to being able to use 70dB dynamic range by crushing the stop
//!   band.
//! - If a filter is not terminal, we can allow it to ripple more in the early pass band because a
//!   lower rate will be read by those bins in the bank.
//! - **Short FIR length and very acceptable delays.**
//! - Phase linearity and all the good stuff that FIRs bring.
//!
//! ## Usage
//!
//! Within this repo:
//!
//! ```sh
//! cargo pmr lowpass --taps 25
//! ```
//!
//! Fully flat pass band down to DC (the last, lowest downsampler)
//!
//! ```sh
//! cargo pmr lowpass --taps 25 --terminal
//! ```
//!
//! Wider margin for main lobes in the pass band to not get scalloped by the transition.
//!
//! ```sh
//! cargo pmr lowpass --taps 25 --guard 0.25
//! ```

// NEXT it would be welcome to just reduce the weight of the pm_remez crate.  The calculation seems
// super fast if only it didn't bring in a bunch of dependencies.
// NEXT not appropriate for calibration.  We keep finding better weight combinations.  Well, why not
// automate some of the search for our actual priorities?
// NEXT DFT the outputs and quit "sine seweeeeping".
// NOTE No wonder this crate seemed like slop to some.  Note, it was sloppy, not slop.  Slop has a
// floor.  Humans can go lower.  💩
// NEXT Just get rid of length estimation.  Our designs don't work with it at all.

use clap::{Parser, Subcommand, ValueEnum};
use pm_remez::{
    constant, linear, order_estimates::ichige, pm_parameters, pm_remez, BandSetting, PMParameters,
    ParametersBuilder, Symmetry,
};

use mutate_lib as utate;
use utate::dsp::{fir::DynamicFirLowpass, Filter, SineSweeper};

// XXX LOL.  Good thinking!
#[derive(Debug, thiserror::Error)]
enum PmrError {
    #[error("Unhandled error: {0}")]
    Unhandled(#[from] utate::MutateError),
}

#[derive(Parser, Debug)]
#[command(name = "Parks McClellen Remez")]
#[command(about = "Design optimal FIR weights for downsampling lowpass filters.", long_about = None)]
#[command(arg_required_else_help = true)]
struct EntryPoint {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Output optimal FIR lowpass filter weights.
    Lowpass(LowpassArgs),
}

fn ranged_f64(s: &str, lo: f64, hi: f64) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("`{s}` is not a number"))?;
    if (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        Err(format!("must be in {lo}..={hi}"))
    }
}

fn read_fraction(s: &str) -> Result<f64, String> {
    ranged_f64(s, 0.05, 0.95)
}

fn guard_fraction(s: &str) -> Result<f64, String> {
    ranged_f64(s, 0.0, 0.9)
}

fn decimate_factor(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|_| format!("`{s}` is not an integer"))?;
    if v >= 2 && v.is_power_of_two() {
        Ok(v)
    } else {
        Err("must be a power of two, at least 2".into())
    }
}

#[derive(Debug, clap::Args)]
struct LowpassArgs {
    /// Decimation factor applied after this filter.  Sets the target Nyquist.
    #[arg(long, default_value_t = 2, value_parser = decimate_factor)]
    pub decimate: usize,
    /// Number of samples that influence each output.
    #[arg(long)]
    pub taps: Option<usize>,
    /// Fraction of the *target* Nyquist that is actually read (bottom half by default).
    #[arg(long, default_value_t = 0.5, value_parser = read_fraction)]
    pub read: f64,
    /// This is the last filter in the chain; hold the deep pass band flat too.
    #[arg(long)]
    pub terminal: bool,
    /// Flat margin above the read edge, as a fraction of the read edge.
    #[arg(long, default_value_t = 0.10, value_parser = guard_fraction)]
    pub guard: f64,
}

fn main() -> Result<(), PmrError> {
    let args = EntryPoint::parse();

    match args.command {
        None => unreachable!(),
        Some(Command::Lowpass(a)) => cmd_lowpass(a),
    }

    Ok(())
}

// ♻️ Copied from workbench
const INDENT: usize = 2;
const LABEL_W: usize = 32; // includes colon
const VALUE_W: usize = 22;

macro_rules! header {
    ($($arg:tt)*) => {{
        const WIDTH: usize = INDENT + LABEL_W + 1 + VALUE_W;
        let title = format!($($arg)*);
        println!("\n{title}");
        println!("{}", "=".repeat(WIDTH));
    }};
}

fn cmd_lowpass(args: LowpassArgs) {
    header!("Optimal Lowpass FIR Weights:");
    let taps = args.taps.unwrap_or(21);
    assert!(taps % 2 == 1, "--taps must be odd");
    println!("Filter length: {taps}");

    let d = args.decimate as f64;
    let f_out = 1.0 / d;
    let f_n_out = 0.5 * f_out;
    let pass = args.read * f_n_out;

    // Raw transition is everything besides pass band and its reflection.
    let raw_transition = f_out - (2.0 * pass);
    let guard = args.guard;
    let pass_edge = pass * (1.0 + guard);
    let stop_edge = f_out - pass_edge;

    let guard_ratio = pass_edge / pass;
    let read_low = pass * 0.5;
    let deep_edge = read_low / guard_ratio;

    println!("decimation: {}x", args.decimate);
    println!("output rate:    {f_out:.4}  (target Nyquist {f_n_out:.4})");

    println!("\nAll fractions in cycles/sample of the input rate (Nyquist = 0.5).");
    println!("passband:   0.0-{pass:.4} (guarded through {pass_edge:.4})");
    println!("shoulder:   {pass_edge:.4}-{f_n_out:.4} (lands in unread band)");
    println!("fold:       {f_n_out:.4}-{stop_edge:.4} (folds into unread band)");
    println!("stopband:   {stop_edge:.4}-0.5 (folds into passband)");
    println!("transition: {:.4} wide", stop_edge - pass_edge);
    println!(
        "weighting:  {}",
        if args.terminal {
            "terminal (flat down to DC)"
        } else {
            "non-terminal (DC ripple relaxed)"
        }
    );

    // NOTE my memory could be failing me, but in at least one library, I found that linear weights
    // were implemented correctly while some constant weights were not.  "Failures of imagination."

    // Below the next stage's Nyquist.  Read at a lower rate downstream, so ripple
    // here is someone else's problem.  Empty when terminal.
    let deep_pass_weight = if args.terminal { 100.0 } else { 1.0 };
    let deep_pass =
        BandSetting::with_weight(0.0, deep_edge, constant(1.0), constant(deep_pass_weight))
            .unwrap();

    // The read band plus guards on both sides.  Flat weight, flat target.
    let pass_band =
        BandSetting::with_weight(deep_edge, pass_edge, constant(1.0), linear(100.0, 100.0))
            .unwrap();

    // Junk that stays put in a band we never read.  Weakly constrained to allow ripple to flourish.
    let shoulder =
        BandSetting::with_weight(pass_edge, f_n_out, linear(1.0, 0.0), linear(0.00, 1.0)).unwrap();

    // Junk that folds down onto the shoulder.
    let fold =
        BandSetting::with_weight(f_n_out, stop_edge, constant(0.0), linear(1.0, 1.0)).unwrap();

    // Everything past here folds into what we read.  Content near 0.5 folds to near
    // DC, so the weight stays high across the whole band.
    let stop_band =
        BandSetting::with_weight(stop_edge, 0.5, constant(0.0), linear(10000.0, 10000.0)).unwrap();

    let bands = [deep_pass, pass_band, shoulder, fold, stop_band];
    let mut parameters = pm_parameters(taps, &bands).unwrap();

    // rarely exceeds 6 or so.
    parameters.set_max_iterations(64);
    // We always use odd symmetry.  Even symmetry is for some other use case according to pm_remez
    // docs.
    // parameters.set_symmetry(Symmetry::Even);
    parameters.set_flatness_threshold(1e-8);
    // NOTE Setting this generally made filters worse.
    // Use Chebyshev degree N = L - 1 where L is filter length and L - 1 is filter order, N.
    // parameters.set_chebyshev_proxy_degree((taps - 1).min(126));

    // Boom!
    let design = pm_remez(&parameters).unwrap();
    header!("Design Results");
    println!("  weighted error: {:.8}", design.weighted_error);
    println!("  flatness: {:.8}", design.flatness);
    println!("  iterations: {:.8}", design.num_iterations);

    // Test the max gain at various frequencies of interest.
    let f_sample = 48_000f64;

    let f_pass = pass * f_sample;
    let f_pass_deep = 0.5 * pass * f_sample;
    // Source frequency whose first image lands exactly on the pass band edge.
    // Content here folds *directly* into the band we still care about.
    let f_pass_mirror = (f_out - pass) * f_sample;
    // Source frequency whose image lands on the deep pass edge.  Worst case.
    let f_pass_deep_mirror = (f_out - 0.5 * pass) * f_sample;
    // Down-sampled Nyquist, the frequency at which we must see at least -6dB attenuation.
    let f_n_down = f_n_out * f_sample;
    let f_stop = stop_edge * f_sample;
    let f_stop_deep = ((0.5 - stop_edge) * 0.5 + stop_edge) * f_sample;

    let mut input = SineSweeper::new(777.0, f_sample as f64);
    // NOTE normalization seems to violate some of the ideas behind the theory.
    // let sum: f64 = design.impulse_response.iter().sum();
    // let norm = 1.0 / sum;
    let coefficients: Vec<f64> = design.impulse_response.clone();
    let coefficients_f32: Vec<f32> = coefficients.iter().map(|&c| c as f32).collect();
    let mut filter = DynamicFirLowpass::with_coefficients(coefficients_f32.clone());

    // Is there a band named Pass? 🚿
    header!("Gain Testing");
    for (f_test, band_name) in [
        (f_pass_deep, "deep pass band"),
        (f_pass, "pass band"),
        (f_n_down, "new Nyquist"),
        (f_pass_mirror, "upper passband fold mirror"),
        (f_pass_deep_mirror, "deep passband fold mirror"),
        (f_stop, "stop band"),
        (f_stop_deep, "deep stop band"),
    ] {
        input.set_frequency(f_test);
        let warmup = input.nsamples(512.0);
        for _ in 0..warmup {
            filter.process(input.next().unwrap());
        }

        let measure = input.nsamples(256.0);
        // Ratio of RMS, so the input's own partial-cycle bias mostly cancels and we don't
        // care what amplitude SineSweeper hands us.
        let mut energy_in = 0.0f64;
        let mut energy_out = 0.0f64;
        for _ in 0..measure {
            let x = input.next().unwrap();
            let y = filter.process(x);
            energy_in += (x as f64) * (x as f64);
            energy_out += (y as f64) * (y as f64);
        }
        let gain = (energy_out / energy_in).sqrt();
        let db = 20.0 * gain.log10();

        println!("  {:<42} {gain:.8}  {db:+8.2} dB", format!("{band_name}:"));
    }

    header!("64-bit Weights");
    println!("const FILTER: [f64; {taps}] = [");
    for w in &coefficients {
        let bits = w.to_bits();
        debug_assert_eq!(f64::from_bits(bits), *w);
        println!("  f64::from_bits(0x{bits:016x}), // {w:+.17e}");
    }
    println!("];");

    header!("32-bit Weights");
    println!("const FILTER: [f32; {taps}] = [");
    for w in &coefficients_f32 {
        let bits = w.to_bits();
        debug_assert_eq!(f32::from_bits(bits), *w);
        println!("  f32::from_bits(0x{bits:08x}), // {w:+.9e}");
    }
    println!("];");
}
