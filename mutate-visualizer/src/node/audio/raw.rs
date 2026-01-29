// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Raw Audio Node
//!
//! The raw audio crate exposes an upstream node interface onto the AudioContext for downstream
//! consumer nodes.  It provides the minimal interface so that other nodes may handle different
//! throttling or processing requirements on their own.
//!
//! ## Buffering and Windowing
//!
//! Different consumers may have different input requirements.  Some nodes may benefit from
//! extremely fast updates of small sizes.  Some may prefer longer windows.  These differing
//! requirements should be handled by downstream nodes, either dedicated nodes or internal behaviors
//! of nodes.

use mutate_lib as utate;

// NEXT prelude & module structure
use crate::{
    graph::{self, EventIntent, GraphEvent},
    node::SeekState,
};

pub struct RawAudioNode {
    context: utate::AudioContext,
    rx: utate::AudioConsumer,
    buffer: graph::GraphBuffer<Audio>,
    // NEXT GraphBuffer will need the Read trait or something to work better with the upstream ring
    // buffer.
    read_buffer: [u8; 6400],
    state: SeekState,
    // Monotonically record the upstream buffer read down.  Whenever it grows instead, the buffer
    // has read a chunk and we can record a new bottom.  If that bottom is too large, we can read
    // ahead a bit to get closer to the range of jitter.
    upstream_bottom: i32,
    upstream_min: i32,
    upstream_allowance: i32,

    // Debug stats
    bytes_read: usize,
    begin: std::time::Instant,
    on_time: bool,
    frames: usize,
}

// DEBT audio format
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Audio {
    pub left: f32,
    pub right: f32,
}

impl std::ops::Add for Audio {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Audio {
            left: self.left + other.left,
            right: self.right + other.right,
        }
    }
}

impl RawAudioNode {
    pub fn new() -> Result<Self, utate::MutateError> {
        // NEXT choice is a dependency required by the node to be created.  Handle via config, then
        // defaults, user input if necessary / specified on command line.
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
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        // FIXME handle invalid choices.
        // FIXME enforce the choice on the pipewire side (pipewire seems do do whatever it wants)
        let choice_idx = input.trim().parse().unwrap();
        let choice = first_choices.remove(choice_idx);

        // Connect to stream and hold onto the consumer.
        let rx = context.connect(&choice, "mutate".to_owned()).unwrap();

        Ok(Self {
            context,
            rx,
            buffer: graph::GraphBuffer::new(800),
            read_buffer: [0u8; 6400],

            state: SeekState::OnTime,

            upstream_bottom: 0,
            upstream_min: 51200,
            upstream_allowance: 9600,

            bytes_read: 0,
            begin: std::time::Instant::now(), // LIES initialized on first ON TIME
            on_time: false,
            frames: 0,
        })
    }

    // Consumes no input.  Just decides the alignment state for now.  This is rather disjoint and
    // demonstrates the tradeoff between separating consume / produce without getting control
    // signals in the right places.
    pub fn consume(&mut self) -> Result<SeekState, utate::MutateError> {
        // NEXT record upstream jitter and solve for the correct allowance for smooth playback.
        let avail = self.rx.occupied() as i32; // DEBT rate
        if avail < self.upstream_min || avail == 0 {
            // Still on the decreasing side.  If we don't have enough for one frame, we already read
            // too much and need to slow down.
            self.upstream_min = avail;
            if avail > 6400 {
                // DEBT rate
                self.state = SeekState::OnTime;
            } else {
                self.state = SeekState::OverProduced;
            }
        } else {
            // Upstream read a chunk.  Record the bottom, which is the previous mininimum available
            // less one frame.  If it is larger than the allowance, we will seek forward a bit so
            // that the next bottom is closer to the allowance.
            self.upstream_bottom = (self.upstream_min - 6400).max(0);
            self.upstream_min = avail;
            if self.upstream_bottom > self.upstream_allowance {
                self.state = SeekState::UnderProduced;
            } else {
                self.state = SeekState::OnTime;
            }
        }
        Ok(self.state)
    }

    pub fn produce(&mut self) -> Result<GraphEvent<Audio>, utate::MutateError> {
        self.frames += 1;
        let out = match self.state {
            SeekState::OverProduced => {
                // XXX over produced was breaking something
                let occupied = self.rx.occupied().min(6400);

                let read = self.rx.read(&mut self.read_buffer[0..occupied])?;
                assert!(read % 8 == 0);

                self.bytes_read += read;

                let coerced = unsafe {
                    std::slice::from_raw_parts(self.read_buffer.as_ptr() as *const Audio, read / 8)
                };
                self.buffer.write(coerced);

                Ok(GraphEvent {
                    intent: EventIntent::Partial,
                    buffer: &self.buffer,
                })
            }
            SeekState::UnderProduced => {
                // Yielding a seek event to try and catch up.  We target the middle of the allowance
                // but only approach it at half of the error.
                assert!(self.upstream_bottom > self.upstream_allowance);
                // NEXT use an integral controller to make the error rate smaller but accounting for
                // observed and experimentally observed (prior) jitter.
                // NOTE do some algebra.  ((b - a) + a/2) / 2
                let error = (self.upstream_bottom / 2) - (self.upstream_allowance / 4);
                // Error might be larger than a whole frame.
                let mut clamped = error.clamp(0, 6400) as usize;
                if clamped % 8 != 0 {
                    clamped = (clamped / 8) * 8;
                }

                let read = self.rx.read(&mut self.read_buffer[0..clamped])?;
                assert!(read % 8 == 0);

                self.bytes_read += read;

                let coerced = unsafe {
                    std::slice::from_raw_parts(self.read_buffer.as_ptr() as *const Audio, read / 8)
                };
                self.buffer.write(coerced);
                self.state = SeekState::OnTime;

                Ok(GraphEvent {
                    intent: EventIntent::Seek,
                    buffer: &self.buffer,
                })
            }
            SeekState::OnTime => {
                if !self.on_time {
                    self.on_time = true;
                    self.begin = std::time::Instant::now();
                }

                let read = self.rx.read(&mut self.read_buffer[0..6400])?;
                assert!(read % 8 == 0);

                self.bytes_read += read;

                // XXX use zerocopy for this style of transmutation.
                let coerced = unsafe {
                    std::slice::from_raw_parts(self.read_buffer.as_ptr() as *const Audio, read / 8)
                };
                self.buffer.write(coerced);

                Ok(GraphEvent {
                    intent: EventIntent::Full,
                    buffer: &self.buffer,
                })
            }
        };

        // NEXT hold tracking information on the context and pass context in Node interface
        if rand::random_range(0..100) > 98 {
            let running = (std::time::Instant::now() - self.begin).as_millis() as f64;
            let read = self.bytes_read as f64;
            let expected_by_time = running / 1000.0 * 48000.0 * 8.0;
            let expected_by_frames = self.frames as f64 * 6400.0;
            let expected_frames_by_time = running / 1000.0 * 60.0;
            println!(
                "frames: {}, expected_frames: {:.2}, expected_by_frames: {:.0}, expected_by_time: {:.0}, actuallly read {:.0}",
                self.frames, expected_frames_by_time, expected_by_frames, expected_by_time, read
            )
        }

        out
    }

    pub fn destroy(self) {
        // NOTE explicit drop just indicates where the behavior is living right now.
        drop(self.rx);
    }
}
