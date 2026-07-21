// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//! # Audio
//!
//! Select a device.  Set up stream from server to device.  Run a callback on each audio tick.

use mutate_lib::{self as utate, audio, prelude::*, vulkan::prelude::*};

#[compute_pipeline(
    compute = stage!("audio/rms", Compute, c"main"),
    push = push!(RmsConstants{
        pub input: DeviceAddress,
        pub ouput: DeviceAddress,
        pub head: UInt,
    }),
)]
struct RmsComputePipeline;

pub struct CallbackResources {
    // Raw buffer
    // output: vanilla buffer here
    // XXX are we holding onto something?  stuff all that stuff here.
    resources: Box<()>,
}

pub struct Audio {
    context: audio::AudioContext,
    pub consumer: audio::import::Consumer<2>,
    resources: CallbackResources,
}

impl Audio {
    // NOTE all of the audio processing can happen in compute queues, but the resources for hand-off
    // to graphics for presentation will need concurrent access since the ring buffers cannot be
    // QFOT and we otherwise have to do awkward window copying.
    pub fn new(device: &Device, queue: &QueueRef<Graphics>) -> Result<Self, utate::MutateError> {
        // NEXT Handle audio choice via deafult + config so that user input is only necessary where
        // explicitly requested or updated at runtime.
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

        // XXX Device reference, command pool ring, and queue for queue submission might all be
        // packaged up so that this closure may use but does not own the necessary gear.  Then use
        // some unsafe.  The Consumer is joined prior to destruction, so we usually won't worry

        let buffer = utate::vulkan::resource::buffer::DeviceBuffer::new(device, 128)?;

        buffer.destroy(device);

        // let ring = PoolRing::new(device, queue)?;
        let pipeline = ComputePipeline::<RmsComputePipeline>::new(device);

        let on_flush = move |state: &utate::audio::import::DeviceRingView<2>| {
            // acquire buffer & semaphore from ring.
            // dispatch a shader named...p

            // Returns all of the data to allow it all to be reclaimed
            state.occupied_len()
        };
        let consumer = context.import_to_device(device, &choice, 6400, "µTate", on_flush)?;

        let resources = CallbackResources {
            resources: Box::new(()),
        };

        Ok(Self {
            context,
            consumer,
            resources,
        })
    }

    pub fn destroy(&mut self, device: &Device) -> Result<(), MutateError> {
        self.consumer.destroy(device)?;
        // context has no vulkan resources and may just drop.
        Ok(())
    }
}
