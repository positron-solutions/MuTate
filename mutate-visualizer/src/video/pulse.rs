// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Pulse
//!
//! Stub for the upcoming three-channel thing.  Just a copy of hello draw to get the visualizer
//! rearrangement done.

use ash::vk;
use mutate_lib::{self as utate, prelude::*};
use utate::vulkan::resource::{buffer, image};

#[compute_pipeline(
    compute = stage!("pulse/compute", Compute, c"main"),
    push = push!(HelloConstants {
        pub counter: UInt,
        pub window_width: Float,
        pub window_height: Float,
        pub output_idx: SsboIdx,
    }),
)]
pub struct HelloPipeline;

pub struct HelloDraw {
    pipeline: ComputePipeline<HelloPipeline>,
    counter: u32,

    output_buffer: Option<buffer::MappedAllocation<rgb::Rgba<u8>>>,
    output_idx: SsboIdx,
}

impl HelloDraw {
    pub fn new(device: &Device) -> Self {
        Self {
            pipeline: ComputePipeline::<HelloPipeline>::new(device).unwrap(),
            counter: 0,
            output_buffer: None,
            output_idx: SsboIdx::INVALID,
        }
    }

    pub fn provision(
        &mut self,
        device: &Device,
        size: vk::Extent2D,
    ) -> Result<(), utate::MutateError> {
        if let Some(existing) = self.output_buffer.take() {
            unsafe {
                existing.destroy(device)?;
                device.descriptors.unbind_ssbo(self.output_idx);
            }
            self.output_idx = SsboIdx::INVALID;
        }

        let output_buffer =
            buffer::MappedAllocation::new((size.width * size.height) as usize, device)?;

        self.output_idx = output_buffer.bound(device);
        self.output_buffer = Some(output_buffer);

        Ok(())
    }

    pub fn draw(
        &mut self,
        device: &Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        acquired_image: &AcquiredImage,
    ) {
        let extent = acquired_image.extent;

        // XXX argument order
        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_pre(&cb, device);

        let push = HelloConstants {
            counter: self.counter.into(),
            window_width: (extent.width as f32).into(),
            window_height: (extent.height as f32).into(),
            output_idx: self.output_idx,
        };
        // XXX allow pushing to wrapped buffers
        self.pipeline.push(device, **cb, &push);
        self.counter += 1;

        // This dispatch math needs to respect the compute stage's declared dimensions.  We can make
        // that adjustable with specialization constants during the pipeline compilation.  This math
        // can be abstracted with some reflection data as the general pattern is that we need the
        // dispatch geometry to exceed the invocation area necessary for the output.  The slang code
        // checks bounds and masks off lanes that are unnecessary.
        let wg_x = 8;
        let wg_y = 4;
        let dispatch_x = (extent.width + wg_x - 1) / wg_x;
        let dispatch_y = (extent.height + wg_y - 1) / wg_y;
        self.pipeline
            .dispatch(device, **cb, dispatch_x, dispatch_y, 1);

        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_post(&cb, device);

        let region = buffer::buffer_image_copy_full(extent);
        unsafe {
            device.as_raw().cmd_copy_buffer_to_image(
                **cb,
                self.output_buffer.as_ref().unwrap().buffer,
                acquired_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    pub fn destroy(self, device: &Device) -> Result<(), utate::MutateError> {
        unsafe {
            self.pipeline.destroy(device);
            if let Some(allocated) = self.output_buffer {
                allocated.destroy(&device)?;
                device.descriptors.unbind_ssbo(self.output_idx);
            }
        }
        Ok(())
    }
}
