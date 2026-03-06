// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Parks McClellen Remez (PMR)
//!
//! This utility will output static PMR windows weights.
//!
//! The `pm_remez` crate was pretty heavy, so this small binary is the only thing that depends on it.
//! We are currently only using PMR to generate low pass filters for downsampling.  We have a neat
//! trick on our side:
//!
//! - We have already analyzed high frequency bins using the normal input stream.
//! - We can begin our transition in bins we won't read in the output anyway.
//! - That region is consecutive with the region that will fold into it after downsampling.
//! - We use both already read bins and their folding mirror as an ultra-wide transition band.
//!
//! We're cheating and letting some noise fold in, but are attenuating it enough that it's
//! definitely not growing.  Anything that remains will be further obliterated by our DFT bins and
//! IIR pre-conditioning.  The important part is that **no noise folds into bands we will be looking
//! at with DFTs.**
//!
//! This leaves us some tradeoffs to shoot for:
//!
//! - Noise floor in the stop (which would fold into our new bands of interest!)
//! - Acceptable folding noise in the transition.
//! - **Short FIR length and very acceptable delays.**
//! - Phase linearity and all the good stuff that FIRs bring.
//!
//! ## Usage
//!
//! Within this repo:
//!
//! ```sh
//! cargo pmr lowpass --taps 23
//! ```
//!
//! As a standalone binary:
//!
//! ```sh
//! pmr estimate --attenuation 60.0
//! pmr --help
//! ```

// NEXT it would be welcome to just reduce the weight of the pm_remez crate.  The calculation seems
// super fast if only it didn't bring in a bunch of dependencies.

use clap::{Parser, Subcommand, ValueEnum};
use pm_remez::{
    constant, linear, order_estimates::ichige, pm_parameters, pm_remez, BandSetting, PMParameters,
    ParametersBuilder, Symmetry,
};

use mutate_lib as utate;
use utate::dsp::{fir::DynamicFirLowpass, Filter, SineSweeper};

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
    /// Estimate the required FIR length from qualitative goals.
    EstimateBins(EstimateArgs),
}

#[derive(Debug, clap::Args)]
struct LowpassArgs {
    #[arg(long)]
    /// Number of samples that influence each output.
    pub taps: Option<usize>,
    #[arg(long)]
    /// Beginning of pass band as a fraction of the input sample rate.
    pub pass: Option<f64>,
    #[arg(long)]
    /// Beginning of the stop band as a fraction of the input sample rate.
    pub stop: Option<f64>,
    // NEXT adjust attenuation vs pass goals.  The scale of weights in the stop band trades ripple
    // in the pass (and transition) for depth of cut in the stop band, which is most important for
    // us.
    #[arg(long)]
    /// Use just a little extra pass band to fight ripple / droop at the edge.
    pub pass_guard: Option<f64>,
    #[arg(long)]
    /// Use just a little extra stop band to fight ripple at the edge.
    pub stop_guard: Option<f64>,
}

#[derive(Debug, clap::Args)]
struct EstimateArgs {
    #[arg(long)]
    /// Beginning of passband as a fraction of the input sample rate
    pub passband_edge: Option<f64>,
    #[arg(long)]
    /// Width of transition band as a fraction of the input sample rate.
    pub transition_width: Option<f64>,
    #[arg(long)]
    /// How much deviation to tolerate in the passband (hint, low!)
    pub passband_ripple: Option<f64>,
    #[arg(long)]
    /// How much deviation to tolerate in the stop band (higher can be acceptable)
    pub stopband_ripple: Option<f64>,
}

fn main() -> Result<(), PmrError> {
    let args = EntryPoint::parse();

    match args.command {
        None => unreachable!(),
        Some(Command::Lowpass(a)) => cmd_lowpass(a),
        Some(Command::EstimateBins(a)) => cmd_estimate_taps(a),
    }

    Ok(())
}

// ♻️ Copied from workbenc
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

fn cmd_estimate_taps(args: EstimateArgs) {
    header!("Ichige PMR Length Estimator");
    println!("Output the number of taps needed.\n\n");
    println!("Frequencies are normalized to sample rate.\n");

    let f_n = 1.0 / 2.0;

    // XXX check docs and implement things
    let fc = 0.250 * f_n;
    let fs = 0.750 * f_n;

    println!("cutoff band: 0.0-{}", fc);
    println!("transition width: {}", fs - fc);
    println!("stop band: {}-1.0", fs);

    let pass_ripple = args.passband_ripple.unwrap_or(0.01);
    let stop_ripple = args.stopband_ripple.unwrap_or(0.05);

    // https://docs.rs/pm-remez/latest/pm_remez/order_estimates/fn.ichige.html
    let n_taps = ichige(fc, fs - fc, pass_ripple, stop_ripple);
    println!("Estimated taps: {}", n_taps);
}

fn cmd_lowpass(args: LowpassArgs) {
    header!("Optimal Lowpass FIR Weights:");
    let taps = args.taps.unwrap_or(21);
    assert!(taps % 2 == 1, "--taps must be odd");
    println!("Filter length: {taps}");

    println!("Band frequencies are normalized to sample rate of 1.0\n");

    let pass = args.pass.unwrap_or(0.125);
    let stop = args.stop.unwrap_or(0.375);
    assert!(pass < 0.5, "Pass band must end below input Nyquist");
    assert!(stop < 0.5, "Stop band must end below input Nyquist");
    println!("passband: 0.0-{pass:.3}");
    println!("stopband: {stop:.3}-0.5");
    let mid = (pass + stop) / 2.0;
    let nyquist_input = 0.5;

    let pass_guard = args.pass_guard.unwrap_or(0.02);
    let stop_guard = args.stop_guard.unwrap_or(0.02);
    // XXX ensure guards not too large!

    // Bands are normalized to sample rate of 1.

    // Strongly weighted pass band.
    let pass_band = BandSetting::with_weight(
        0.0,
        pass * (1.0 + pass_guard),
        constant(1.0),
        linear(10.0, 10.0),
    )
    .unwrap();

    // Center band with weak weights to give solver a place we don't care about to to cram in all
    // the ripple.
    let linear_transition = BandSetting::with_weight(
        (pass + mid) * 0.5,
        (mid + stop) * 0.5,
        constant(0.0),
        linear(0.0, 0.5),
    )
    .unwrap();

    // Strongly weighted stop band
    let stop_band = BandSetting::with_weight(
        (stop * (1.0 - stop_guard)),
        nyquist_input,
        constant(0.0),
        linear(14000.0, 18000.0),
    )
    .unwrap();

    // early_transition, late_transition,
    let bands = [pass_band, stop_band]; // early_transition, late_transition,
    let mut parameters = pm_parameters(taps, &bands).unwrap();

    // rarely exceeds 6 or so.
    parameters.set_max_iterations(64);
    // We always use odd symmetry.  Even symmetry is for some other use case according to pm_remez
    // docs.
    // parameters.set_symmetry(Symmetry::Even);
    parameters.set_flatness_threshold(0.00000001);
    // Use Chebyshev degree N = L - 1 where L is filter length and L - 1 is filter order, N.
    parameters.set_chebyshev_proxy_degree(taps - 1);

    // Boom!
    let design = pm_remez(&parameters).unwrap();
    header!("Design Results");
    println!("  weighted error: {:.8}", design.weighted_error);
    println!("  flatness: {:.8}", design.flatness);
    println!("  iterations: {:.8}", design.num_iterations);

    // Test the max gain at various frequencies of interest.
    let f_sample = 48_000f64;
    let f_pass_deep = 0.5 * pass * f_sample;
    let f_pass = pass * f_sample;
    // Down-sampled Nyquist, the frequency at which we must see at least -6dB attenuation.
    let f_n_down = f_sample * 0.25;
    let f_stop = stop * f_sample;
    let f_stop_deep = ((0.5 - stop) * 0.5 + stop) * f_sample;

    let mut input = SineSweeper::new(777.0, f_sample as f64);
    // NOTE normalization seems to violate some of the ideas behind the theory.
    // let sum: f64 = design.impulse_response.iter().sum();
    // let norm = 1.0 / sum;
    let coefficients: Vec<f32> = design.impulse_response.iter().map(|c| *c as f32).collect();
    let mut filter = DynamicFirLowpass::with_coefficients(coefficients.clone());

    // Is there a band named pass? 🚿
    header!("Gain Testing");
    for (f_test, band_name) in [
        (f_pass_deep, "deep pass band"),
        (f_pass, "pass band"),
        (f_n_down, "downsample Nyquist"),
        (f_stop, "stop band"),
        (f_stop_deep, "deep stop band"),
    ] {
        input.set_frequency(f_test);
        let warmup = input.nsamples(64.0);
        for _ in 0..warmup {
            filter.process(input.next().unwrap());
        }
        let measure = input.nsamples(128.0);
        let mut peak: f32 = 0.0;
        for _ in 0..measure {
            peak = peak.max(filter.process(input.next().unwrap()).abs());
        }

        println!("  {:<20} {peak:2.8}", format!("{band_name}:"));
    }

    header!("Weights");
    println!("const FILTER: [f32;{taps}] = [");
    for w in coefficients {
        let bits = w.to_bits();
        println!("  f32::from_bits(0x{bits:08x}), // {w:+.8}");
    }
    println!("];");

    // XXX Test the frequency folding with the new weights :-)
}
