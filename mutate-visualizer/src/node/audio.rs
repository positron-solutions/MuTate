// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

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

// NEXT these are the kinds of fields we want to expose as simple inputs for other nodes, but the
// targets will need just one field per input.
pub struct ScalarAudioEvent {
    /// left channel RMS
    pub left: f32,
    /// right channel RMS
    pub right: f32,
}

// This is a first pass extraction of the original node.  It will be refined into a more
// render-graph like construction, an audio input node with separate processing before feeding into
// the graphics node.
pub struct AudioNode {
    // MAYBE is everything in this type necessary?
    audio_events: ringbuf::wrap::caching::Caching<
        std::sync::Arc<ringbuf::SharedRb<ringbuf::storage::Heap<ScalarAudioEvent>>>,
        false,
        true,
    >,

    hue: f32,

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

            let mut window_buffer = [0u8; 3200];
            let window_size = 3200; // one 240FPS frame at 48kHz and 8 bytes per frame
            let read_behind = 3200; // one frame of read-behind

            // Noise is calculated with a 2.0s window of RMS samples (averaged left and right)
            let mut rmss = [0f32; 480];
            let mut rmss_i = 0;

            // FIXME Ah yes, the user friendly API for real Gs
            let mut conn = std::mem::ManuallyDrop::new(unsafe { Box::from_raw(rx.conn) });

            while ae_tx.read_is_held() {
                let avail = conn.buffer.occupied_len();
                if avail >= window_size {
                    let read = conn.buffer.peek_slice(&mut window_buffer);
                    assert!(read == window_size);

                    // Square sums
                    let (left_sum_sq, right_sum_sq) = window_buffer
                        .chunks_exact(8) // 2 samples per frame × 4 bytes = 8 bytes per frame
                        .map(|frame| {
                            let left = f32::from_le_bytes(frame[0..4].try_into().unwrap());
                            let right = f32::from_le_bytes(frame[4..8].try_into().unwrap());
                            (left, right)
                        })
                        .fold((0f32, 0f32), |(acc_l, acc_r), (l, r)| {
                            (acc_l + (l * l), acc_r + (r * r))
                        });

                    // RMS
                    let (left_rms, right_rms) = (left_sum_sq.sqrt(), right_sum_sq.sqrt());

                    // Store total RMS for noise floor
                    rmss[rmss_i] = (left_rms + right_rms) * 0.5;
                    rmss_i += 1;
                    if !(rmss_i < rmss.len()) {
                        rmss_i = 0;
                    }

                    // Backoff using queue size
                    if ae_tx.vacant_len() > 1 {
                        match ae_tx.try_push(ScalarAudioEvent {
                            left: left_rms,
                            right: right_rms,
                        }) {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("sending audio event failed");
                            }
                        }
                    } else if ae_tx.is_full() {
                        eprintln!("audio event consumer is falling behind");
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
            handle: audio_thread,
            context,
            audio_events: ae_rx,
        })
    }

    pub fn process(&mut self) -> AudioColors {
        if self.audio_events.is_full() {
            eprintln!("audio event backpressure drop");
            self.audio_events.skip(1);
        }
        let rms = match self.audio_events.try_pop() {
            Some(event) => event.left + event.right,
            None => {
                eprintln!("No audio event was ready");
                0.1
            }
        };

        self.hue += 0.0005 * rms;
        if self.hue > 1.0 || self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        }

        // Extract audio -> color stream
        let tweaked = rms * 0.05 + 0.1;
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
        let scale = rms * 0.25 - 0.5;
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
