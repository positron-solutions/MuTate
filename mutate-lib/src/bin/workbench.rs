// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused)]
#![allow(dead_code)]

//! # Workbench
//!
//! The workbench is used for engineering tuning of the filters that make up the filter bank used by
//! the spectrogram.
//!
//! ## Usage
//!
//! Remember that dog in Star Fox that said, "Good luck!"?  That's what you get.
//!
//! (Try --help)

use clap::{Parser, Subcommand, ValueEnum};

use mutate_lib::{
    self as utate,
    dsp::{
        self, dft,
        iir::{self, Biquad, Cascade, CytomicSvf, Svf},
        Filter, FilterArgs, SineSweeper,
    },
    prelude::*,
};

#[derive(Parser, Debug)]
#[command(name = "workbench")]
#[command(about = "Engineering tuning for filters and spectrogram filter bank.", long_about = None)]
#[command(arg_required_else_help = true)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, thiserror::Error)]
enum WorkbenchError {
    #[error("Unhandled error: {0}")]
    Unhandled(#[from] utate::MutateError),
}

fn main() -> Result<(), WorkbenchError> {
    let args = Args::parse();

    match args.command {
        None => unreachable!(),
        Some(Command::List(_)) => cmd_list(),
        Some(Command::Config(_)) => cmd_config(),
        Some(Command::Optimize(_)) => cmd_optimize(),
        Some(Command::Sanity(a)) => cmd_sanity(a),
        Some(Command::Stress(a)) => cmd_stress(a),
        Some(Command::Rise(a)) => cmd_rise(a),
        Some(Command::Decay(a)) => cmd_decay(a),
        Some(Command::Bandwidth(a)) => cmd_bandwidth(a),
        Some(Command::Noise(a)) => cmd_noise(a),
        Some(Command::Gain(a)) => cmd_gain(a),
        Some(Command::Bin(a)) => cmd_bin(a.center),
    }

    Ok(())
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum FilterChoice {
    /// Simple 1st order complex resonator
    Complex,
    /// Biquad filters
    Biquad,
    /// State Vector Filter with Topology Preserving Transform
    Svf,
    /// Cytomic SVF, a high-stability variant
    Cytomic,
    /// Discrete Fourier Transform
    Dft,
}

impl FilterChoice {
    fn instantiate(&self, args: &FilterArgs) -> Box<dyn Filter> {
        match self {
            Self::Biquad => Box::new(Cascade::<Biquad>::from_args(args)),
            Self::Svf => Box::new(Cascade::<Svf>::from_args(args)),
            Self::Cytomic => Box::new(Cascade::<CytomicSvf>::from_args(args)),
            Self::Dft => Box::new(dft::Dft::from_args(args)),
            // Vanilla complex not supported yet.
            _ => todo!(),
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
#[value(rename_all = "kebab_case")]
enum FilterSelector {
    All,

    Complex,
    Biquad,
    Svf,
    Cytomic,
    Dft,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List all filters
    List(ListArgs),
    /// Show default filter and bank settings
    Config(ConfigArgs),
    /// Validate filter stability in forgiving situations
    Sanity(SanityArgs),
    /// Test filters in extreme situations
    Stress(StressArgs),
    /// Measure time to rise (fast attack)
    Rise(RiseArgs),
    /// Measure time to decay.
    Decay(DecayArgs),
    /// Measure peak gain and gain linearity
    Gain(GainArgs),
    /// Measure width of the pass band
    Bandwidth(BandwidthArgs),
    /// Sweep for unexpected resonance frequencies
    Noise(NoiseArgs),
    /// Calibrate bank and generate table
    Optimize(OptimizeArgs),
    /// Locate the visual bin for a frequency
    Bin(BinArgs),
}

#[derive(clap::Args, Debug)]
struct GainArgs {
    /// Filters to test, or `all`
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,
}

#[derive(clap::Args, Debug)]
struct SanityArgs {
    /// Filters to test, or `all`
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,
}

#[derive(clap::Args, Debug)]
struct StressArgs {
    /// Filters to test, or `all`
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,
}

#[derive(clap::Args, Debug)]
struct RiseArgs {
    /// Filters to test, or `all`
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,
}

#[derive(clap::Args, Debug)]
struct DecayArgs {
    /// Filters to test, or `all`
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,
}

#[derive(clap::Args, Debug)]
struct BandwidthArgs {
    /// Filters to test
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,

    /// Threshold in dB (default -3dB)
    #[arg(long)]
    threshold: Option<f32>,

    /// Center frequency in Hz
    #[arg(long)]
    center: Option<f32>,

    /// Q factor override
    #[arg(long = "q-factor")]
    q_factor: Option<f32>,
}

#[derive(clap::Args, Debug)]
struct ListArgs {}

#[derive(clap::Args, Debug)]
struct ConfigArgs {}

#[derive(clap::Args, Debug)]
struct NoiseArgs {
    /// Filters, separated by comma
    #[arg(index = 1, value_delimiter = ',')]
    filters: Vec<FilterSelector>,

    /// Center frequency in Hz (required)
    #[arg(long)]
    center: f32,

    /// Optional flags for noise behavior
    #[arg(long)]
    flags: Option<String>,
}

#[derive(clap::Args, Debug)]
struct OptimizeArgs {}

#[derive(clap::Args, Debug)]
struct BinArgs {
    #[arg(index = 1, required = true)]
    center: f64,
}

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

// DEBT This row macro is pretty pathetic and I don't really have a strong feeling about table
// crates or what we need, so just keep going for now.
macro_rules! row {
    ($label:expr, $fmt:expr, $value:expr) => {{
        let value = format!($fmt, $value);
        println!(
            "{:indent$}{label:<label_w$} {:>value_w$}",
            "",
            value,
            indent = INDENT,
            label = format!("{}:", $label),
            label_w = LABEL_W,
            value_w = VALUE_W,
        );
    }};
}

fn cmd_config() {
    let defaults = WorkbenchConfig::defaults();

    header!("µTate Workbench Configured Defaults");
    row!("Min frequency", "{:3.2} Hz", dsp::MIN_FREQ_CHEAP_DRIVERS);
    row!("Max frequency", "{:3.2} Hz", dsp::MAX_FREQ_OLD_PEOPLE);
    row!(
        "Filter bank bins",
        "{:4}",
        dsp::spectrogram::RESOLUTION_4K_WIDTH
    );
    row!("Q", "{:3.2}", defaults.q());
    row!("Center frequency", "{:} Hz", defaults.center());
    row!("Sample frequency", "{:} Hz", defaults.sample_rate());
    // XXX change "stagger" to detune
    row!("Cascade Detune", "{:?}", defaults.cascade_detune());
    row!("Cascade Stages", "{}", defaults.cascade_stages());
    row!(
        "Cascade Butterworth:",
        "{:?}",
        defaults.cascade_butterworth()
    );
    row!("DFT Window", "{:?}", defaults.dft_window());
    row!(
        "Default bandwidth threshold",
        "{:4.2}dB",
        defaults.bandwidth_db_threshold()
    );
}

fn cmd_list() {
    println!("Available filters:");

    for variant in FilterChoice::value_variants() {
        let pv = variant.to_possible_value().expect("ValueEnum invariant");

        let name = pv.get_name();

        let help = pv.get_help().map(|h| h.to_string()).unwrap_or_default();

        if help.is_empty() {
            println!("  {}", name);
        } else {
            println!("  {:<10} {}", name, help);
        }
    }
}

fn cmd_sanity(args: SanityArgs) {
    let filter_choices = expand_filter_choices(args.filters);
    let cfg = WorkbenchConfig::defaults().args();

    for fc in filter_choices.iter() {
        let mut filter = fc.instantiate(&cfg);
        let mut sg = cfg.sine_gen();
        let nsamples = sg.nsamples(64.0);

        // measure on-center peak gain
        let mut center_peak: f32 = 0.0;
        for _ in 0..nsamples {
            center_peak = center_peak.max(filter.process(sg.next().unwrap()).abs());
        }

        // Drain for off-center measurement
        sg.set_frequency(cfg.center * 7.77);
        for _ in 0..nsamples {
            filter.process(sg.next().unwrap());
        }

        // measure off-center peak gain
        let mut off_center_peak: f32 = 0.0;
        for _ in 0..nsamples {
            off_center_peak = off_center_peak.max(filter.process(sg.next().unwrap()).abs());
        }

        if center_peak <= off_center_peak {
            eprintln!("warning: {fc:?} off-center gain exceeded center: {off_center_peak:4.2} > {center_peak:4.2}");
        } else if !(0.95 < center_peak && off_center_peak < 1.05) {
            eprintln!("warning: {fc:?} gains appear un-normalized: {center_peak:4.2}");
        } else {
            println!("  {fc:?}: sane");
        }
    }
}

fn cmd_stress(_filters: StressArgs) {
    // Gain at low frequency
    // Off-center +/- gain at low frequency
    // Gain at high frequency
    // Off-center +/- gain at high frequency
}

// XXX rename to "attack!"
fn cmd_rise(args: RiseArgs) {
    let filter_choices = expand_filter_choices(args.filters);
    let args = WorkbenchConfig::defaults().args();

    header!("Normalizing Max Gains");
    let gains = normalized_gains(&filter_choices, &args);
    for (fc, gain) in filter_choices.iter().zip(gains.iter()) {
        println!("  {fc:?}: {gain:3.2}");
    }

    let f0 = args.center;
    let fs = args.fs;
    let q = args.q;

    header!("Rise Test");
    for goal in [0.1, 0.25, 0.5, 0.75, 0.9] {
        let mut sg = dsp::sine_gen_48k(f0);
        let mut filters: Vec<Box<dyn Filter>> = filter_choices
            .iter()
            .map(|fc| fc.instantiate(&args))
            .collect();

        // NEXT change how we report time, cycles vs seconds
        println!("time to {goal:2.1}");
        for (fc, (filter, max_gain)) in filter_choices
            .iter()
            .zip(filters.iter_mut().zip(gains.iter()))
        {
            let max_samples = (4096.0 * fs / f0) as usize;
            let mut peak: f32 = 0.0;
            let mut found = false;
            for s in 0..max_samples {
                peak = peak.max(filter.process(sg.next().unwrap()));
                let waves = s as f64 / (fs / f0);
                if peak.abs() > goal * max_gain {
                    println!("  {fc:?}: {waves:7.2} cycles");
                    found = true;
                    break;
                };
            }
            if !found {
                eprintln!("  warning: {fc:?} did not reach the target gain!");
            }
        }
    }
}

fn cmd_decay(args: DecayArgs) {
    // Find the gains manually by spinning them up.  Taper off one half wave.  Then measure waves
    // until signal passes below some threshold.
    let filter_choices = expand_filter_choices(args.filters);
    let args = WorkbenchConfig::defaults().args();

    // NEXT if the filters have non-linear max, we will register their decay early!  We need to
    // normalize at the target, not the maximum!  For now, just start each filter at the max and
    // only vary the decay gaol.
    let gains = normalized_gains(&filter_choices, &args);

    let f0 = args.center;
    let fs = args.fs;
    let q = args.q;

    let goals = [0.75, 0.5, 0.25, 0.1, 0.05];
    for goal in goals {
        println!("time from 1.0 to {goal:2.1}");
        for (fc, gain) in filter_choices.iter().zip(gains.iter()) {
            let mut sg = args.sine_gen();
            let mut filter = fc.instantiate(&args);

            // Peak the filter
            let mut peak: f32 = 0.0;
            let mut max_peak_samples = 1_000_000;
            let mut peaked = false;
            loop {
                peak = peak.max(filter.process(sg.next().unwrap()).abs());
                max_peak_samples -= 1;
                if peak * 1.01 > *gain {
                    peaked = true;
                    break;
                }
                if max_peak_samples < 0 {
                    break;
                }
            }
            if !peaked {
                eprintln!("warning: could not peak filter {:?}", fc);
            }
            // Decay one half wave
            let half_wave = sg.nsamples(0.5);
            for n in 0..half_wave {
                let decaying_gain = n as f32 / half_wave as f32;
                filter.process(sg.next().unwrap() * decaying_gain);
            }

            // Measure time to decay
            let threshold = gain * goal;
            let max_decay_cycles = 1_000_000;
            let mut decayed = false;
            let mut decay_cycles = 0;
            let mut since_exceed = 0;
            // MAYBE we can look at the half wavelength, but this does open us up to dynamic
            // interactions with the decaying filter.
            let wave = sg.nsamples(1.0);
            loop {
                let out = filter.process(0.0).abs();
                if decay_cycles > max_decay_cycles {
                    break;
                }

                if out > threshold {
                    since_exceed = 0;
                } else {
                    since_exceed += 1;
                    if since_exceed > wave {
                        decayed = true;
                        break;
                    }
                }
                decay_cycles += 1;
            }
            if !decayed {
                eprintln!("warning: {fc:?} did not reach {goal:3.2}");
            }
            // DEBT We need to type waves and samples to prevent their mixing.
            let waves = decay_cycles as f64 / (wave as f64);
            println!("  {fc:?}: {waves:7.2} cycles");
        }
    }
}

fn cmd_noise(_args: NoiseArgs) {
    // Find the +/- shoulder and sweep out to edges, looking for secondary modes.
}

fn cmd_gain(args: GainArgs) {
    let filter_choices = expand_filter_choices(args.filters);
    header!("Gain Test");

    let args = WorkbenchConfig::defaults().args();

    let f0 = args.center;
    let fs = args.fs;
    let q = args.q;

    for scale in [1.0, 0.5, 0.25, 0.05] {
        println!("test volume = {scale:2.1}");
        for fc in filter_choices.iter() {
            let mut filter = fc.instantiate(&args);
            let mut sg = dsp::sine_gen_48k(f0);
            let samples = (128.0 * fs / f0) as usize;
            let mut max: f32 = 0.0;
            for s in 0..samples {
                max = max.max(filter.process(sg.next().unwrap() * scale));
            }
            println!("  gain for {fc:?}: {max:7.5}");
        }
    }
}

// NEXT sweep both sides
// NEXT track peak to detect off-center main lobe
fn cmd_bandwidth(args: BandwidthArgs) {
    let filter_choices = expand_filter_choices(args.filters);
    header!("Bandwidth Test");

    let args = WorkbenchConfig::defaults().args();
    let gains = normalized_gains(&filter_choices, &args);

    let cfg = WorkbenchConfig::defaults();
    let threshold_db_find = -(cfg.bandwidth_db_threshold().abs());
    let threshold_db_lose = threshold_db_find - 5.0;

    // NOTE at very low frequencies, 128 Q results in a really long DFTs that become quite slow.  In
    // the GPU this is not a problem.
    for q in [3.0, 5.0, 8.0, 16.0, 32.0, 42.0, 64.0, 128.0, 256.0, 512.0] {
        println!("Goal Q: {q:4.2}");
        for (fc, gain) in filter_choices.iter().zip(gains.iter()) {
            let mut args = args.clone();
            args.q = q;

            let mut filter = fc.instantiate(&args);
            let mut sg = args.sine_gen();

            // Sweep up
            let start_freq = args.center;
            let limit_freq = args.center * 8.0; // three octave outward sweep
            let gain_threshold = power_db_to_amplitude(threshold_db_lose, *gain as f64);
            let lost_freq =
                sweep_outward(&mut filter, &mut sg, start_freq, limit_freq, gain_threshold);
            if let Some(lost) = lost_freq {
                // Sweep back up
                let gain_threshold = power_db_to_amplitude(threshold_db_find, *gain as f64);
                let regained = sweep_inward(&mut filter, &mut sg, lost, start_freq, gain_threshold);
                if let Some(found) = regained {
                    let bandwidth = ((start_freq - found) * 2.0).abs();
                    println!("  {fc:?}: {:8.2} Hz", bandwidth);
                    println!("    measured Q: {:6.2}", args.center / bandwidth);
                } else {
                    eprintln!("warning: {fc:?} did not reach threshold while sweeping inward.");
                    continue;
                }
            } else {
                eprintln!("warning: {fc:?} did not decay while sweeping outward");
                continue;
            }
        }
    }
}

fn cmd_bin(center: f64) {
    let bin = dsp::bank::bin_lookup(
        dsp::MIN_FREQ_CHEAP_DRIVERS,
        dsp::MAX_FREQ_OLD_PEOPLE,
        dsp::spectrogram::RESOLUTION_4K_WIDTH,
        center,
    );
    header!("Bin centered at {:6.1}Hz", bin.center);

    row!("min", "{:.2} Hz", bin.min);
    row!("max", "{:.2} Hz", bin.max);
    row!("bandwidth", "{:.2} Hz", bin.bandwidth());
    row!("quality", "{:.1}", bin.q());
}
fn cmd_optimize() {}

// Just convert the choices.  Don't instantiate filters yet!
fn expand_filter_choices(selectors: Vec<FilterSelector>) -> Vec<FilterChoice> {
    if selectors.iter().any(|f| matches!(f, FilterSelector::All)) {
        vec![
            FilterChoice::Complex,
            FilterChoice::Svf,
            FilterChoice::Biquad,
            FilterChoice::Cytomic,
            FilterChoice::Dft,
        ]
    } else {
        selectors
            .into_iter()
            .map(|f| match f {
                FilterSelector::Svf => FilterChoice::Svf,
                FilterSelector::Biquad => FilterChoice::Biquad,
                FilterSelector::Complex => FilterChoice::Complex,
                FilterSelector::Cytomic => FilterChoice::Cytomic,
                FilterSelector::Dft => FilterChoice::Dft,

                FilterSelector::All => unreachable!(),
            })
            .collect()
    }
}

/// A stub for an actual configuration that has merged flags and defaults and can produce a set of
/// FilterArgs.
/// NEXT instantiate this by reading the config file
struct WorkbenchConfig {}

impl WorkbenchConfig {
    /// XXX not truly implemented stubbed out.  Go read code.
    fn defaults() -> Self {
        WorkbenchConfig {}
    }

    /// Return the default arguments used to build a filter.
    fn args(&self) -> FilterArgs {
        FilterArgs {
            q: self.q(),
            center: self.center(),
            fs: self.sample_rate(),
            gain_factor: 1.00,
            butterworth: self.cascade_butterworth(),
            stagger: self.cascade_detune(),
            stages: self.cascade_stages(),
            window_choice: self.dft_window(),
        }
    }

    /// Build a filter with the default arguments.
    fn filter(&self, filter_choice: FilterChoice) -> Box<dyn Filter> {
        filter_choice.instantiate(&self.args())
    }

    /// Default target bandwidth ratio.
    fn q(&self) -> f64 {
        8.0
    }

    /// Default gain threshold for bandwidth estimation.
    fn bandwidth_db_threshold(&self) -> f64 {
        -10.0
    }

    /// Default sampling frequency
    fn sample_rate(&self) -> f64 {
        48000.0
    }

    /// Default center frequency
    fn center(&self) -> f64 {
        1000.0
    }

    /// Default stages for cascaded filters.
    fn cascade_stages(&self) -> usize {
        1
    }

    /// Toggle for Butterworth Q distribution in cascaded filters.
    fn cascade_butterworth(&self) -> bool {
        false
    }

    /// Default de-tuning for cascaded filters.
    fn cascade_detune(&self) -> Option<f64> {
        Some(1.01)
        // None
    }

    /// DFT window choice
    fn dft_window(&self) -> dft::WindowChoice {
        dft::WindowChoice::DolphChebyshev {
            attenuation_db: 22.5,
        }
        // dft::WindowChoice::BoxCar
        // dft::WindowChoice::Bartlett
        // dft::WindowChoice::Hamming
        // dft::WindowChoice::Welch
    }
}

/// Find maximum gain for each filter choice.
fn normalized_gains(choices: &[FilterChoice], args: &FilterArgs) -> Vec<f32> {
    choices.iter().map(|fc| normalized_gain(fc, args)).collect()
}

/// Find maximum gain.
fn normalized_gain(choice: &FilterChoice, args: &FilterArgs) -> f32 {
    let mut sg = dsp::sine_gen(args.center, args.fs);
    let mut filter = choice.instantiate(args);

    // NEXT dynamic max gain detection.  No new peaks in n samples etc.
    let gain_samples = (args.fs / args.center * 512.0) as usize;

    let mut peak: f32 = 0.0;
    for _ in 0..gain_samples {
        peak = peak.max(filter.process(sg.next().unwrap()).abs());
    }
    peak
}

/// Sweep from `start` to `end` until `threshold_amplitude` is no longer observed for several waves.
fn sweep_outward(
    filter: &mut Box<dyn Filter>,
    sg: &mut SineSweeper,
    start: f64,
    end: f64,
    threshold_amplitude: f64,
) -> Option<f64> {
    let warmup_samples = sg.nsamples(128.0);
    for _ in 0..warmup_samples {
        filter.process(sg.next().unwrap());
    }

    let sweep_resolution = 4096 * 64;
    let log_f_step = (end / start).log2() / sweep_resolution as f64;
    let next_freq = |s| start * (log_f_step * s as f64).exp2();
    let mut threshold_samples = sg.nsamples(16.0);

    let mut found = false;
    let mut last_peak_freq = start;
    let mut last_peak_samples = 0;

    'sweep: for s in 0..(sweep_resolution + 1) {
        let freq = next_freq(s);
        sg.set_frequency(freq);
        let threshold_samples = sg.nsamples(16.0);

        let wave_samples = sg.nsamples(1.0);
        let mut wave_peak: f64 = 0.0;

        for w in 0..wave_samples {
            let y = filter.process(sg.next().unwrap()) as f64;
            wave_peak = wave_peak.max(y.abs());

            if y.abs() > threshold_amplitude {
                last_peak_samples = 0;
                last_peak_freq = freq;
            } else {
                last_peak_samples += 1;
                if last_peak_samples > threshold_samples {
                    found = true;
                    break 'sweep;
                }
            }
        }
    }

    if found {
        Some(last_peak_freq)
    } else {
        None
    }
}

/// Sweep from `start` to `end` until `threshold_amplitude` is first observed.
fn sweep_inward(
    filter: &mut Box<dyn Filter>,
    sg: &mut SineSweeper,
    start: f64,
    end: f64,
    threshold_amplitude: f64,
) -> Option<f64> {
    // NOTE when sweeping inward, we have already found the shoulder, so inward steps tend to be
    // smaller unless the bandwidth is very near three octaves anyway
    let sweep_resolution = 2048 * 8;
    let log_f_step = (end / start).log2() / sweep_resolution as f64;
    let next_freq = |s| start * (log_f_step * s as f64).exp2();

    for s in 0..(sweep_resolution + 1) {
        let freq = next_freq(s);
        sg.set_frequency(freq);
        let wave_samples = sg.nsamples(1.0);
        for w in 0..wave_samples {
            let y = filter.process(sg.next().unwrap()) as f64;
            if y.abs() > threshold_amplitude {
                return Some(freq);
            }
        }
    }
    return None;
}

/// Bandwidth is usually is interpreted as a dB threshold, but we must look for a fraction of
/// normalized gain, which is already expressed as a peak amplitude value.
fn power_db_to_amplitude(db: f64, gain: f64) -> f64 {
    10.0_f64.powf(db / 20.0) * gain
}
