// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # K-weight curve
//!
//! Applies a perceptual filter on amplitude signals.  Filters are defined by EU broadcasting
//! standard, [ITU-R BS.1770-5](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf).
//! A-weighting is another popular choice, but was developed for pure tones.  ISO226 is perhaps
//! better for music but harder to apply directly to an amplitude signal and more useful for
//! weighting frequency buckets for filter banks.

use crate::graph::{EventIntent, GraphBuffer, GraphEvent};
use crate::node::audio::raw::Audio;

#[derive(Default, Clone)]
pub struct Biquad {
    // feedback (a0 assumed = 1.0)
    a1: f32,
    a2: f32,

    // feed forward
    b0: f32,
    b1: f32,
    b2: f32,

    // state
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    // 🤖 generated.  Corroborate against similar implementations.
    // NOTE private for inline
    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;

        y
    }
}

// DEBT audio channels
pub struct KWeightsNode {
    left_shelf: Biquad,
    left_highpass: Biquad,

    right_shelf: Biquad,
    right_highpass: Biquad,

    output: GraphBuffer<Audio>,
}

impl KWeightsNode {
    /// Create a K-weighting filter for 48 kHz sample rate
    pub fn new() -> Self {
        // NOTE Constants from spec:
        // https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf

        // Stage 1: shelving filter
        let shelf = Biquad {
            a1: -1.69065929318241,
            a2: 0.73248077421585,

            b0: 1.53512485958697,
            b1: -2.69169618940638,
            b2: 1.19839281085285,

            ..Default::default()
        };

        // Stage 2: RLB high-pass
        let highpass = Biquad {
            a1: -1.99004745483398,
            a2: 0.99007225036621,

            b0: 1.0,
            b1: -2.0,
            b2: 1.0,

            ..Default::default()
        };

        Self {
            left_shelf: shelf.clone(),
            left_highpass: highpass.clone(),

            right_shelf: shelf,
            right_highpass: highpass,

            output: GraphBuffer::new(800), // DEBT audio rate
        }
    }

    pub fn consume(&mut self, input: &GraphEvent<Audio>) {
        // NEXT would like to iterate from input into output
        let new = input.buffer.fresh();
        if new.len() > 0 {
            let out = self.output.writeable_slice(new.len());
            for (i, n) in new.iter().enumerate() {
                let left = self.left_shelf.process(n.left);
                let right = self.right_shelf.process(n.right);
                out[i] = Audio {
                    left: self.left_highpass.process(left),
                    right: self.right_highpass.process(right),
                }
            }
        }
    }

    pub fn produce(&mut self) -> GraphEvent<Audio> {
        GraphEvent {
            intent: EventIntent::Full,
            buffer: &self.output,
        }
    }
}
