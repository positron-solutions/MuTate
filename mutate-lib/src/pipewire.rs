// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use ringbuf::traits::*;

use mutate_lib::{AudioChoice, AudioContext};

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

    let rx = context.connect(&choice, "mutate-test".to_owned()).unwrap();

    let handle = std::thread::spawn(move || {
        let mut window_buffer = [0u8; 6400];
        let window_size = 6400; // one 60FPS frame at 48kHz and 8 bytes per frame
        let read_behind = 7680; // ~20ms read-behind

        let mut conn = std::mem::ManuallyDrop::new(unsafe { Box::from_raw(rx.conn) });
        let mut frames = 0;
        let mut wrote = false;

        while frames < 60 * 600 {
            let avail = conn.buffer.occupied_len();
            if avail >= window_size {
                let read = conn.buffer.peek_slice(&mut window_buffer);
                assert!(read == window_size);

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
                    print!("\x1B[{}A", 2);
                } else {
                    wrote = true;
                }
                println!("left: {}", left_sum);
                println!("right: {}", right_sum);

                frames += 1;

                // before read_behind, we just presume that the writer stay caught up and therefore
                // we should move forward by a single frame amount and discard it.  within
                // read-behind, we need to begin throttling forward movement to allow for teh writer
                // to catch up.
                if avail >= (window_size * 2) + read_behind {
                    conn.buffer.skip(window_size);
                } else {
                    // eprintln!("warning!  underrun! ⚠️");
                }

                // XXX some kind of back-pressure solution
                std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 60.0));
            } else {
                // Underfed, either we can pad with "empty" data or wait for new data.  Let's wait.
                match rx.wait() {
                    Ok(_) => {
                        // eprintln!("waited ⏰");
                    }
                    Err(e) => {
                        eprintln!("listening aborted: {}", e);
                        break;
                    }
                }
            }
        }

        // XXX something something lifetime
        drop(rx);
    });

    handle.join().unwrap();

    Ok(())
}
