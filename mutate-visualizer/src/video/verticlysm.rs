// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Verticlysm
//!
//! Draws DFT with bins mapped horizontally, creating a vertical pattern.  Left interpolates from
//! bottom to top.  Right interpolates from top to bottom.  Intensity controls both initial
//! brightness and length of interpolation along the screen's vertical axis.  Interpolates bins and
//! the 240Hz output with phase-aware tracking of the input stream.

use ash::vk;
use mutate_lib::{self as utate, prelude::*, vulkan::resource::buffer};

use crate::audio;

#[compute_pipeline(
    compute = stage!("verticlysm/verticlysm", Compute, c"main"),
    push = push!(VerticlysmPushConstants {
        // Input
        pub left_channel: DeviceAddress,
        pub right_channel: DeviceAddress,
        // Output
        pub output: DeviceAddress,

        pub window_width: UInt,
        pub window_height: UInt,

        /// number of DFT bins (rows of the ring)
        pub input_height: UInt,
        /// ring width in columns; PoT
        pub input_width: UInt,

        pub beg_col: UInt,
        pub beg_phase: Float,
        pub span: Float,

        // pub gain: DeviceAddress,
    }),
)]
struct VerticlysmPipeline;

pub struct Verticlysm {
    pipeline: ComputePipeline<VerticlysmPipeline>,
    output: Option<MappedAllocation<rgb::Rgba<u8>>>,
    output_address: DeviceAddress,
}

impl Verticlysm {
    pub fn new(device: &Device) -> Self {
        Self {
            pipeline: ComputePipeline::<VerticlysmPipeline>::new(device).unwrap(),
            output: None,
            output_address: DeviceAddress::NULL,
        }
    }

    pub fn provision(
        &mut self,
        device: &Device,
        size: vk::Extent2D,
    ) -> Result<(), utate::MutateError> {
        if let Some(existing) = self.output.take() {
            unsafe {
                existing.destroy(device)?;
            }
        }
        let output = MappedAllocation::new(device, (size.width * size.height) as usize)?;
        self.output_address = output.device_address(device)?.into();
        self.output = Some(output);
        Ok(())
    }

    pub fn draw(
        &mut self,
        device: &Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        acquired_image: &AcquiredImage,
        audio_outputs: &audio::AudioOutputs,
    ) {
        let extent = acquired_image.extent;
        let dft = &audio_outputs.dft;

        // XXX argument order (reverse cb & device)
        self.output
            .as_ref()
            .unwrap()
            .barrier_compute_pre(&cb, device);

        let width = dft.ring_width;
        debug_assert!(width.is_power_of_two(), "shader masks with input_width - 1");

        // Leading edge in fractional logical columns: closed columns plus the
        // fraction of the open column the device has already accumulated.
        let open_frac = dft.column_ticks_beg as f64 / dft.ticks_per_column as f64;
        let end = dft.write_head as f64 + open_frac;

        // Never integrate columns that were never written, and never lap the ring.
        const TARGET_SPAN: f64 = 6.0;

        let width = dft.ring_width;
        let span = TARGET_SPAN.min(end).min(width as f64 - 1.0);
        if span <= 0.0 {
            return; // nothing published yet
        }

        let beg = end - span;
        let beg_col = (beg.floor() as u64) & (width as u64 - 1);
        let beg_phase = (beg - beg.floor()) as f32;

        let push = VerticlysmPushConstants {
            left_channel: dft.channels[0].into(),
            right_channel: dft.channels[1].into(),
            output: self.output_address,

            window_width: extent.width.into(),
            window_height: extent.height.into(),

            input_height: dft.ring_height.into(),
            input_width: width.into(),
            beg_col: (beg_col as u32).into(),
            beg_phase: beg_phase.into(),
            span: (span as f32).into(),
        };
        self.pipeline.push(device, **cb, &push);

        const LANE_WIDTH: u32 = 32;
        const LANE_ROWS: u32 = 128;
        let dispatch_x = extent.width.div_ceil(LANE_WIDTH);
        let dispatch_y = extent.height.div_ceil(LANE_ROWS);
        self.pipeline
            .dispatch(device, **cb, dispatch_x, dispatch_y, 1);

        self.output
            .as_ref()
            .unwrap()
            .barrier_compute_post(&cb, device);

        let region = buffer::buffer_image_copy_full(extent);
        unsafe {
            device.cmd_copy_buffer_to_image(
                **cb,
                self.output.as_ref().unwrap().buffer,
                acquired_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // XXX we have no way to pass semaphores to the render gear
        // Yeah... we have a submission builder that cannot be given any extra semaphores...
        // Seems like we will need to build our own submission harness.  That sucks, but it will
        // work better.  During the multithreaded switcharoo we can get that done.  Just hope it's
        // on time =D
        let ready = &dft.data_ready;
    }

    pub fn destroy(self, device: &Device) -> Result<(), utate::MutateError> {
        unsafe {
            self.pipeline.destroy(device);
            if let Some(allocated) = self.output {
                allocated.destroy(&device)?;
            }
        }
        Ok(())
    }
}
