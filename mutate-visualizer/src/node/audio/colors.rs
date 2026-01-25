// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Audio Colors
//!
//! Convert scalar volume into some color values.  This node is just a refactoring of the original
//! triangle demo to suit the emerging graph interface.

use ash::vk;
use palette::convert::FromColorUnclamped;

use crate::graph::GraphEvent;
use crate::node::audio::rms::Rms;

// Output type for our rudimentary audio -> color node
pub struct AudioColors {
    pub clear: vk::ClearValue,
    pub color: palette::Srgb<f32>,
    pub scale: f32,
}

pub struct AudioColorsNode {
    hue: f32,
    amplitude: f32,
}

impl AudioColorsNode {
    pub fn new() -> Self {
        Self {
            hue: rand::random::<f32>(),
            amplitude: 0.0,
        }
    }

    // XXX not using the node interface here at all until scalar and array types both work.
    pub fn consume(&mut self, input: &Rms) {
        let avg_rms = (input.left + input.right) / 2.0;
        let tweaked_rms = hill_function(avg_rms, 0.08, 1.5, 1.2);

        self.hue += 0.01 * (tweaked_rms * 0.2 - 0.5);
        if self.hue > 1.0 || self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        }
        self.amplitude = tweaked_rms;
    }

    pub fn produce(&mut self) -> AudioColors {
        // NOTE we want 0 to 1.0, but hill function max was 1.5
        let value = (self.amplitude * 0.666667).clamp(0.0, 1.0);

        let hsv: palette::Hsv = palette::Hsv::new_srgb(self.hue * 360.0, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        // NOTE Transitioning the image to get ready for drawing performs the clear, but the clear
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

        // NEXT bring back decay based negative values as its own node
        let scale = self.amplitude * 2.5 - 0.5;
        let hsv: palette::Hsv = palette::Hsv::new_srgb(trie_hue, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        AudioColors {
            clear,
            color: rgb,
            scale,
        }
    }
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
