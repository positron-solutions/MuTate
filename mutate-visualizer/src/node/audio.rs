// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::CString;

use ash::vk;
use palette::convert::FromColorUnclamped;
use ringbuf::traits::*;

use mutate_lib as utate;

// Output type for our rudimentary audio -> color node
pub struct AudioColors {
    pub clear: vk::ClearValue,
    pub color: palette::Srgb<f32>,
    pub scale: f32,
}

// This is a first pass extraction of the original node.  It will be refined into a more
// render-graph like construction, an audio input node with separate processing before feeding into
// the graphics node.
pub struct AudioNode {
    audio_events: ringbuf::wrap::caching::Caching<
        std::sync::Arc<ringbuf::SharedRb<ringbuf::storage::Heap<(f32, f32, f32, f32)>>>,
        false,
        true,
    >,
    hue: f32,
    value: f32,
    handle: std::thread::JoinHandle<()>,
    context: utate::AudioContext,
}

impl AudioNode {
    pub fn new() -> Result<Self, utate::MutateError> {
        let context = utate::AudioContext::new()?;
        println!("Choose the audio source:");

        let mut first_choices = Vec::new();
        let check = |choices: &[utate::AudioChoice]| {
            first_choices.extend_from_slice(choices);
        };

        context.with_choices_blocking(check).unwrap();
        first_choices.iter().enumerate().for_each(|(i, c)| {
            println!("[{}] {} AudioChoice: {:?}", i, c.id(), c.name());
        });

        // Of course this needs retry and a default.  Also, the stream in use does not seem to be
        // respecting our choice of stream anyway.  That should be fixed for cases where multiple output
        // streams are valid.
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let choice_idx = input.trim().parse().unwrap();
        let choice = first_choices.remove(choice_idx);

        let rx = context.connect(&choice, "mutate".to_owned()).unwrap();

        // audio events, processed results of the buffer, using an independent ring to provide some
        // buffering, synchronized communication, and back pressure support.
        let ae_ring = ringbuf::HeapRb::new(3);
        let (mut ae_tx, ae_rx) = ae_ring.split();

        // NEXT Package this into a node.  The node will store a thread handle.  The node is used to
        // yield inputs to the visual node in draw_frame.  Treat each output as some independent
        // transformation so that we may begin creating the kinds of tension that our later render graph
        // architecture will have to solve.
        let audio_thread = std::thread::spawn(move || {
            // This thread continuously emits events.  The scheme is a sliding window with a 120Hz width
            // and sliding in 240Hz increments.  The production of events is faster than the frame rate,
            // and balanced back pressure is accomplished by looking at the ring buffer size.

            // To subtract the noise floor, we track the moving average with a 240 sample exponential
            // moving average.
            let mut window_buffer = [0u8; 3200];
            let window_size = 3200; // one 240FPS frame at 48kHz and 8 bytes per frame
            let read_behind = 3200; // one frame of read-behind
            let mut left_max = 0f32;
            let mut right_max = 0f32;
            let mut left_noise = 0f32;
            let mut right_noise = 0f32;

            let alpha = 2.0 / (240.0 + 1.0);
            let alpha_resid = 1.0 - alpha;

            let mut left_fast_accum = 0f32;
            let mut right_fast_accum = 0f32;
            let mut left_fast = 0f32;
            let mut right_fast = 0f32;
            let alpha_f = 2.0 / (8.0 + 1.0);
            let alpha_f_resid = 1.0 - alpha_f;

            // FIXME Ah yes, the user friendly API for real Gs
            let mut conn = std::mem::ManuallyDrop::new(unsafe { Box::from_raw(rx.conn) });

            while ae_tx.read_is_held() {
                let avail = conn.buffer.occupied_len();
                if avail >= window_size {
                    let read = conn.buffer.peek_slice(&mut window_buffer);
                    assert!(read == window_size);

                    // Estimate the energy by absolute delta.  IIRC not only is this physically wrong
                    // but also doesn't map to perceptual very well.
                    let (mut last_l, mut last_r) = (0.0, 0.0);
                    let (left_sum, right_sum) = window_buffer
                        .chunks_exact(8) // 2 samples per frame × 4 bytes = 8 bytes per frame
                        .map(|frame| {
                            let left = f32::from_le_bytes(frame[0..4].try_into().unwrap());
                            let right = f32::from_le_bytes(frame[4..8].try_into().unwrap());
                            (left, right)
                        })
                        .fold((0f32, 0f32), |(acc_l, acc_r), (l, r)| {
                            // absolute delta + absolute amplitude
                            let accum = (
                                acc_l + (l - last_l).abs() + l.abs(),
                                acc_r + (r - last_r).abs() + r.abs(),
                            );
                            last_l = l;
                            last_r = r;
                            accum
                        });

                    left_noise = (alpha * left_sum) + (alpha_resid * left_noise);
                    right_noise = (alpha * right_sum) + (alpha_resid * right_noise);

                    // Cut noise and normalize remaining to noise
                    let left_excess = (left_sum - (left_noise * 1.3)) / left_noise.max(0.000001);
                    let right_excess =
                        (right_sum - (right_noise * 1.3)) / right_noise.max(0.000001);

                    // Fast EMA of the cleaned signal for beats
                    left_fast = (alpha_f * left_excess) + (alpha_f_resid * left_fast);
                    right_fast = (alpha_f * right_excess) + (alpha_f_resid * right_fast);

                    // Instantaneous response on climb
                    if left_fast < left_excess {
                        left_fast = left_excess;
                    }
                    if right_fast < right_excess {
                        right_fast = right_excess;
                    }

                    left_fast_accum = left_fast + left_fast_accum;
                    right_fast_accum = right_fast + right_fast_accum;

                    left_max = left_max.max(left_excess);
                    right_max = right_max.max(right_excess);

                    // Backoff using queue size
                    if ae_tx.vacant_len() > 1 {
                        match ae_tx.try_push((
                            left_max,
                            right_max,
                            left_fast_accum,
                            right_fast_accum,
                        )) {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("sending audio event failed: {:?}", e);
                                if ae_tx.is_full() {
                                    eprintln!("audio event consumer is falling behind");
                                }
                            }
                        }
                        left_max = 0.0;
                        right_max = 0.0;
                        left_fast_accum = 0.0;
                        right_fast_accum = 0.0;
                    }

                    if avail >= (window_size * 2) + read_behind {
                        conn.buffer.skip(window_size / 2 + 200); // LIES +200 🤔
                    }

                    std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 240.0));
                } else {
                    // Underfed, either we can pad with "empty" data or wait for new data.  Let's wait.
                    match rx.wait() {
                        Ok(_) => {
                            eprintln!("audio buffered ⏰");
                        }
                        Err(e) => {
                            eprintln!("listening aborted: {}", e);
                            break;
                        }
                    }
                }
            }
        });

        Ok(Self {
            hue: rand::random::<f32>(),
            value: 0.0,
            handle: audio_thread,
            context,
            audio_events: ae_rx,
        })
    }

    pub fn process(&mut self) -> AudioColors {
        // NEXT extract this audio event -> color stream as nodes
        if self.audio_events.is_full() {
            eprintln!("audio event backpressure drop");
            self.audio_events.skip(1);
        }
        let (_slow, fast) = match self.audio_events.try_pop() {
            Some(got) => ((got.0 + got.1), (got.2 + got.3)),
            None => {
                eprintln!("No audio event was ready");
                (0.1, 0.1)
            }
        };

        self.value = fast;
        self.hue += 0.002 * fast;
        if self.hue > 1.0 || self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        } else if self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        }

        // Extract audio -> color stream
        let tweaked = self.value * 0.02 + 0.3;
        let value = tweaked.clamp(0.0, 1.0);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(self.hue * 360.0, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        // XXX Transitioning the image to get ready for drawing performs the clear, but the clear
        // color selection on the output is a node behavior.  This creates some coupling between the
        // node and target that either requires the node to provide the clear color early or to
        // perform the entire image layout transition, which is not bad, but adds a function call to
        // each node.  We can enforce the correct behavior by passing the untransitioned target and
        // then transitioning it with a clear color as an argument.
        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [rgb.red, rgb.green, rgb.blue, 1.0],
            },
        };

        let mut trie_hue = self.hue * 360.0 + 180.0;
        if trie_hue > 360.0 {
            trie_hue -= 360.0;
        }
        let scale = 0.8 + (0.2 * self.value);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(trie_hue, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        AudioColors {
            clear,
            color: rgb,
            scale,
        }
    }

    pub fn destroy(self) {
        // Note, dropping the rx will tell the tx thread to break
        drop(self.audio_events);
        self.handle.join().unwrap()
    }
}
