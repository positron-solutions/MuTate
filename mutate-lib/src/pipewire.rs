// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Write;
use std::sync::{atomic, atomic::Ordering, Arc};

use ringbuf::traits::*;

use mutate_lib::audio::{AudioChoice, AudioContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = AudioContext::new()?;
    let mut first_choices = Vec::new();
    let check = |choices: &[AudioChoice]| {
        first_choices.extend_from_slice(choices);
    };
    context.with_choices(check).unwrap();
    assert!(first_choices.len() == 0 as usize);

    let check = |choices: &[AudioChoice]| {
        first_choices.extend_from_slice(choices);
    };

    println!("Choose the audio source:");
    context.with_choices_blocking(check).unwrap();
    first_choices.iter().enumerate().for_each(|(i, c)| {
        println!("[{}] {} AudioChoice: {:?}", i, c.id(), c.name());
    });

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let choice_idx = input.trim().parse().unwrap();
    let choice = first_choices.remove(choice_idx);

    let mut rx = context.connect(&choice, "mutate-test".to_owned()).unwrap();

    let running = Arc::new(atomic::AtomicBool::new(true));

    let handle = std::thread::spawn(move || {
        let mut window_buffer = [0u8; 6400];
        let mut window_index = 0usize; // Windex
        let window_size = 6400; // one 60FPS frame at 48kHz and 8 bytes per frame

        let mut wrote = false;

        // NEXT deadline tracking and numerical stability enhancements
        let mut prev_frame_finish = std::time::Instant::now();
        let mut last_frame_start = std::time::Instant::now();
        let mut last_frame_finish = std::time::Instant::now();
        let mut last_update_duration = std::time::Duration::from_nanos(0);
        let mut last_present_duration = std::time::Duration::from_nanos(0);
        let mut last_pre_present_duration = std::time::Duration::from_nanos(0);
        let mut last_spins = 0;
        // 1ms OS jitters reportedly not uncommon without assumptions about the scheduler
        // configuration.
        // MAYBE spin_for and update_slack can be combined.
        // NOTE on slower OSs, these slack values are likely not enough.  My Kernel is configured
        // with somewhat more aggressive scheduling than a default Linux kernel.
        let spin_for = std::time::Duration::from_micros(150);
        let frame_time = std::time::Duration::from_micros(16_667);
        let update_slack = std::time::Duration::from_micros(50);

        // NEXT react to actual terminal dimensions
        let mut display_left = String::with_capacity(100);
        let mut display_right = String::with_capacity(100);

        let mut stdout = std::io::stdout();

        let inner_running = running.clone();
        ctrlc::set_handler(move || {
            inner_running.store(false, Ordering::Relaxed);
            reset_terminal();
        })
        .unwrap();

        // Alternate mode, hide cursor
        // NOTE hides output from the context's audio thread.  May make debugging a pain.  Output
        // code works either with alternate mode or without it, so you can comment this if audio
        // thread output is interesting.  Consider stderr for more first-class debugging.
        write!(stdout, "\x1B[?1049h\x1B[?25l").unwrap();
        stdout.flush().unwrap();
        while running.load(Ordering::Relaxed) {
            let avail = rx.occupied();
            if avail > 0 {
                let slice = &mut window_buffer[window_index..window_size];
                window_index += rx.read(slice).unwrap();

                // If we filled up the entire slice, we can "display" a visual frame.
                if window_index == window_size {
                    window_index = 0;

                    let (mut last_l, mut last_r) = (0.0, 0.0);
                    let (left_sum, right_sum) = window_buffer
                        .chunks_exact(8) // 2 samples per frame × 4 bytes = 8 bytes per frame
                        .map(|frame| {
                            let left = f32::from_le_bytes(frame[0..4].try_into().unwrap());
                            let right = f32::from_le_bytes(frame[4..8].try_into().unwrap());
                            (left, right)
                        })
                        .fold((0.0f32, 0.0f32), |(acc_l, acc_r), (l, r)| {
                            // absolute delta + absolute amplitude
                            let accum = (
                                acc_l + (l - last_l).abs() + l.abs(),
                                acc_r + (r - last_r).abs() + r.abs(),
                            );
                            last_l = l;
                            last_r = r;
                            accum
                        });

                    if wrote {
                        write!(stdout, "\x1B[{}A", 8).unwrap();
                    } else {
                        wrote = true;
                    }

                    // Throttle the presentation to 60Hz
                    // NEXT track wake jitter from sleeping
                    // FIXME there can be negative diffs.
                    // FIXME update_time might become unstable.
                    // Wake up when we have approximately just enough time to finish the frame
                    let wake_target = last_frame_finish + frame_time
                        - last_update_duration.mul_f32(0.9) // Decay update durations
                        - update_slack;
                    let mut diff = wake_target - std::time::Instant::now();
                    while diff > std::time::Duration::from_micros(1000) {
                        std::thread::sleep(diff - spin_for);
                        diff = wake_target - std::time::Instant::now();
                    }
                    let frame_start = std::time::Instant::now();

                    // Basic processing to provide some interpretable visual feedback
                    let left = hill_function(left_sum, 150.0, 100.0, 1.5).floor() as u32;
                    let right = hill_function(right_sum, 150.0, 100.0, 1.5).floor() as u32;

                    display_left.clear();
                    for _ in 0..left {
                        display_left.push('#');
                    }
                    display_right.clear();
                    for _ in 0..right {
                        display_right.push('#');
                    }

                    write!(stdout, "\x1B[2K\rl: {}\n", display_left).unwrap();
                    write!(stdout, "\x1B[2K\rr: {}\n", display_right).unwrap();
                    write!(
                        stdout,
                        "last frame update time: {:10.4} ms\n",
                        (last_update_duration.as_micros() as f32 / 1_000.0)
                    )
                    .unwrap();
                    write!(
                        stdout,
                        "last present duration:  {:10.4} ms\n",
                        (last_present_duration.as_nanos() as f32 / 1_000_000.0)
                    )
                    .unwrap();
                    write!(
                        stdout,
                        "last pre-present duration: {:7.4} ms\n",
                        (last_pre_present_duration.as_nanos() as f32 / 1_000_000.0)
                    )
                    .unwrap();
                    write!(stdout, "last pre-present spins: {:7} spins\n", last_spins).unwrap();
                    write!(
                        stdout,
                        "last frame duration:    {:10.4} ms\n",
                        ((last_frame_finish - prev_frame_finish).as_micros() as f32 / 1_000.0)
                    )
                    .unwrap();
                    write!(
                        stdout,
                        "frame wakeup time:      {:10.4} ms\n",
                        ((frame_start - last_frame_start).as_micros() as f32 / 1_000.0)
                    )
                    .unwrap();
                    let update_finish = std::time::Instant::now();

                    // Present when there is just enough time to flush.  Since there is less
                    // variable time, we can wait more precisely here.
                    let present_target = last_frame_finish + frame_time - last_present_duration;
                    let mut diff = present_target - std::time::Instant::now();
                    last_pre_present_duration = diff;
                    let mut spins = 0;
                    while diff > std::time::Duration::from_micros(1) {
                        std::hint::spin_loop();
                        spins += 1;
                        diff = present_target - std::time::Instant::now();
                    }

                    let present_start = std::time::Instant::now();
                    stdout.flush().unwrap();
                    let frame_finish = std::time::Instant::now();
                    last_update_duration = update_finish - frame_start;
                    last_frame_start = frame_start;
                    prev_frame_finish = last_frame_finish;
                    last_frame_finish = frame_finish;
                    last_present_duration = frame_finish - present_start;
                    last_spins = spins;
                }
            } else {
                // Underfed, either we can pad with "empty" data or wait for new data.  Let's wait.
                match rx.wait() {
                    Ok(_) => {
                        // eprintln!("waited ⏰");
                    }
                    Err(e) => {
                        write!(stdout, "listening aborted: {}", e).unwrap();
                        break;
                    }
                }
            }
        }
    });

    handle.join().unwrap();

    Ok(())
}

/// Hill function starts at zero, has a controllable halfway point, asymptote, and shape.
///
/// - `x` the variable input.  Should be on the scale.
///
/// - `half_x` select which x values will reach half of the asymptote.
///
/// - `max` the asymptote.
///
/// - `c_hill` Hill coeffeciant.  Choose > 1.0 for double inflection shapes.
///
fn hill_function(x: f32, half_x: f32, max: f32, c_hill: f32) -> f32 {
    let t_n = x.powf(c_hill);
    max * (t_n / (t_n + half_x.powf(c_hill)))
}

/// Revert cursor and alternate mode settings.
fn reset_terminal() {
    let mut stdout = std::io::stdout();
    write!(stdout, "\x1B[?1049l\x1B[?25h").unwrap();
    stdout.flush().unwrap();
}
