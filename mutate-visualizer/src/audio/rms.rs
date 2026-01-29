// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Root Mean Squared
//!
//! RMS is not very useful on its own.  It over-represents bass tones that humans can't even
//! perceive, below 40Hz.  It does not sufficiently emphasize high frequency tones.  To correct
//! this, use a perceived loudness curve such as K-weighting or A-weighting etc.

use crate::audio::raw::Audio;
use crate::graph::GraphEvent;

pub struct Rms {
    pub left: f32,
    pub right: f32,
}

/// A very simple node.
pub struct RmsNode {
    left_sum_sq: f32,
    right_sum_sq: f32,
    n: usize,
}

impl RmsNode {
    pub fn new() -> Self {
        RmsNode {
            left_sum_sq: 0.0,
            right_sum_sq: 0.0,
            n: 0,
        }
    }

    pub fn consume(&mut self, input: &GraphEvent<Audio>) {
        // Tracking frames is suddenly necessary
        if !(input.intent == crate::graph::EventIntent::Seek) {
            self.n = 0;
            self.left_sum_sq = 0.0;
            self.right_sum_sq = 0.0;
        }

        let new = &input.buffer.data;

        if new.len() > 0 {
            // Sum of squares
            let (left_sum_sq, right_sum_sq) =
                new.iter().fold((0f32, 0f32), |(acc_l, acc_r), frame| {
                    (
                        acc_l + (frame.left * frame.left),
                        acc_r + (frame.right * frame.right),
                    )
                });

            self.n += new.len();

            self.left_sum_sq += left_sum_sq;
            self.right_sum_sq += right_sum_sq;
        }
    }

    pub fn produce(&mut self) -> Rms {
        Rms {
            left: (self.left_sum_sq / (self.n as f32).max(1.0)).sqrt(),
            right: (self.right_sum_sq / (self.n as f32).max(1.0)).sqrt(),
        }
    }
}
