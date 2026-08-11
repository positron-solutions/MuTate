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
//! pmr estimate-bins --passband-edge 0.125
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
    /// Number of samples that influence each output.
    #[arg(long)]
    pub taps: Option<usize>,
    /// Beginning of pass band as a fraction of the input sample rate.
    #[arg(long)]
    pub pass: Option<f64>,
    /// Beginning of the stop band as a fraction of the input sample rate.
    #[arg(long)]
    pub stop: Option<f64>,
    // NEXT adjust attenuation vs pass goals.  The scale of weights in the stop band trades ripple
    // in the pass (and transition) for depth of cut in the stop band, which is most important for
    // us.
    /// Use just a little extra pass band to fight ripple / droop at the edge.
    #[arg(long)]
    pub pass_guard: Option<f64>,
    /// Use just a little extra stop band to fight ripple at the edge.
    #[arg(long)]
    pub stop_guard: Option<f64>,
    /// Decimation factor applied after this filter.
    #[arg(long, default_value_t = 2)]
    pub decimate: usize,
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

    let pass_ripple = args.passband_ripple.unwrap_or(0.001);
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

    let transition = stop - pass;
    assert!(pass < 0.5, "Pass band must end below input Nyquist");
    assert!(stop < 0.5, "Stop band must end below input Nyquist");
    assert!(transition > 0.0, "--pass must be below --stop");

    let pass_guard = args.pass_guard.unwrap_or(0.02);
    let stop_guard = args.stop_guard.unwrap_or(0.02);
    assert!(
        (0.0..0.5).contains(&pass_guard),
        "--pass-guard must be in [0, 0.5)"
    );
    assert!(
        (0.0..0.5).contains(&stop_guard),
        "--stop-guard must be in [0, 0.5)"
    );

    let pass_edge = pass + transition * pass_guard;
    let stop_edge = stop - transition * stop_guard;
    assert!(
        pass_edge < stop_edge,
        "guards consumed the entire transition band"
    );

    println!("passband:   0.0-{pass:.4} (guarded to {pass_edge:.4})");
    println!("stopband:   {stop:.4}-0.5 (guarded from {stop_edge:.4})");
    println!(
        "transition: {:.4} wide, {:.4} after guards",
        transition,
        stop_edge - pass_edge
    );

    let mid = (pass + stop) / 2.0;
    let nyquist_input = 0.5;

    // Strongly weighted pass band.
    let pass_band =
        BandSetting::with_weight(0.0, pass_edge, constant(1.0), linear(10.0, 100.0)).unwrap();

    // Center band with weak weights to give solver a place we don't care about to to cram in all
    // the ripple.  Less signal in transition is better, but we care more about flatness in the pass
    // and stop bands.
    let linear_transition = BandSetting::with_weight(
        // begin definition between
        pass_edge,
        stop_edge,
        constant(0.0),
        linear(0.1, 100.0),
    )
    .unwrap();

    // Strongly weighted stop band, favoring the deep folding region that will map directly into the
    // useful region.
    let stop_band = BandSetting::with_weight(
        stop_edge,
        nyquist_input,
        constant(0.0),
        linear(100.0, 400.0),
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
    parameters.set_flatness_threshold(0.0001);
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
    let f_pass_deep = 0.5 * pass * f_sample;
    let f_pass = pass * f_sample;

    let decimate = args.decimate;
    assert!(decimate >= 2, "--decimate must be at least 2");
    let d = decimate as f64;

    // Down-sampled Nyquist, the frequency at which we must see at least -6dB attenuation.
    let f_n_down = 0.5 / d * f_sample;
    // Source frequency whose first image lands exactly on the pass band edge.
    // Content here folds *directly* into the band we still care about.
    let f_pass_mirror = (1.0 / d - pass) * f_sample;
    // Source frequency whose image lands on the deep pass edge.  Worst case.
    let f_pass_deep_mirror = (1.0 / d - 0.5 * pass) * f_sample;

    let f_stop = stop * f_sample;
    let f_stop_deep = ((0.5 - stop) * 0.5 + stop) * f_sample;

    let mut input = SineSweeper::new(777.0, f_sample as f64);
    // NOTE normalization seems to violate some of the ideas behind the theory.
    // let sum: f64 = design.impulse_response.iter().sum();
    // let norm = 1.0 / sum;
    let coefficients: Vec<f64> = design.impulse_response.clone();
    let coefficients_f32: Vec<f32> = coefficients.iter().map(|&c| c as f32).collect();
    let mut filter = DynamicFirLowpass::with_coefficients(coefficients_f32.clone());

    // Is there a band named pass? 🚿
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
        let measure = input.nsamples(512.0);
        let mut peak: f32 = 0.0;
        for _ in 0..measure {
            peak = peak.max(filter.process(input.next().unwrap()).abs());
        }

        println!("  {:<42} {peak:2.8}", format!("{band_name}:"));
    }

    header!("64-bit Weights");
    println!("const FILTER: [f64; {taps}] = [");
    for w in &coefficients {
        let bits = w.to_bits();
        debug_assert_eq!(f64::from_bits(bits), *w);
        println!("  f64::from_bits(0x{bits:016x}), // {w:+.17e}");
    }
    println!("];");

    // XXX filter truncating!
    header!("32-bit Weights");
    println!("const FILTER: [f32; {taps}] = [");
    for w in &coefficients_f32 {
        let bits = w.to_bits();
        debug_assert_eq!(f32::from_bits(bits), *w);
        println!("  f32::from_bits(0x{bits:08x}), // {w:+.9e}");
    }
    println!("];");

    // XXX Test the frequency folding with the new weights :-)
}
