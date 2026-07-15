// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Audio
//!
//! Select a device.  Set up stream from server to device.  Run a callback on each audio tick.

use mutate_lib::{self as utate, audio, prelude::*, vulkan::prelude::*};

pub struct Audio {
    context: audio::AudioContext,
    pub consumer: audio::import::Consumer<2>,
}

impl Audio {
    pub fn new(device: &Device) -> Result<Self, utate::MutateError> {
        // NEXT choice is a dependency required by the node to be created.  Handle via config, then
        // defaults, user input if necessary / specified on command line.
        let context = audio::AudioContext::new()?;
        println!("Choose the audio source:");
        let mut first_choices = Vec::new();
        let check = |choices: &[audio::AudioChoice]| {
            first_choices.extend_from_slice(choices);
        };
        context.with_choices_blocking(check).unwrap();
        let max_name_width = first_choices
            .iter()
            .map(|c| c.name().len())
            .max()
            .unwrap_or(0);
        first_choices.iter().enumerate().for_each(|(i, c)| {
            println!("[{}] {:<max_name_width$}  [{}]", i, c.name(), c.kind());
        });
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        // FIXME handle invalid input choices.
        let choice_idx = input.trim().parse().unwrap();
        let choice = first_choices.remove(choice_idx);

        let consumer = context.import_to_device(device, &choice, 6400, "µTate")?;

        Ok(Self { context, consumer })
    }

    pub fn destroy(&mut self, device: &Device) -> Result<(), MutateError> {
        self.consumer.destroy(device)?;
        // context has no vulkan resources and may just drop.
        Ok(())
    }
}
