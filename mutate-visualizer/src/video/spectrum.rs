// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Spectrum
//!
//! Show the world the power of the Constant-Q transform.  Let them see through the eyes of the
//! machine!
//!

use ash::vk;

use mutate_assets as assets;

use mutate_lib::{self as utate, prelude::*};
use utate::vulkan::{
    resource::{buffer, image},
    util,
};

#[repr(C)]
#[derive(Clone, Copy)]
/// Processed inputs for the shader.
struct SpectrumSample {
    left_decibels: f32,
    left_mag: f32,
    right_decibels: f32,
    right_mag: f32,
}

#[compute_pipeline(
    compute = stage!("spectrum/compute", Compute, c"main"),
    push = push!(SpectrumConstants {
        pub window_width: Float,
        pub window_height: Float,
        pub spectrum_idx: SsboIdx,
        pub output_idx: SsboIdx,
    }),
)]
pub struct SpectrumPipeline;

// This will be an interface after more nodes exist
pub struct SpectrumNode {
    pipeline: ComputePipeline<SpectrumPipeline>,

    spectrum_buffer: Option<buffer::MappedAllocation<SpectrumSample>>,
    output_buffer: Option<buffer::MappedAllocation<rgb::Rgba<u8>>>,

    spectrum_idx: SsboIdx,
    output_idx: SsboIdx,
}

impl SpectrumNode {
    pub fn new(device: &Device) -> Self {
        let assets = assets::AssetDirs::new();
        let pipeline = ComputePipeline::<SpectrumPipeline>::new(&device).unwrap();

        Self {
            pipeline,
            spectrum_buffer: None,
            output_buffer: None,

            // XXX in resources API, these will be provided in a totally different way.
            spectrum_idx: SsboIdx::INVALID,
            output_idx: SsboIdx::INVALID,
        }
    }

    // NEXT Provisioning and re-provisioning reactively to the upstream render target size is a very
    // important problem to work on!
    pub fn provision(
        &mut self,
        device: &mut Device,
        size: vk::Extent2D,
    ) -> Result<(), utate::MutateError> {
        if let Some(existing) = self.spectrum_buffer.take() {
            unsafe {
                existing.destroy(device)?;
                device.descriptors.unbind_ssbo(self.spectrum_idx);
            }
            self.spectrum_idx = SsboIdx::INVALID;
        }
        if let Some(existing) = self.output_buffer.take() {
            unsafe {
                existing.destroy(device)?;
                device.descriptors.unbind_ssbo(self.output_idx);
            }
            self.output_idx = SsboIdx::INVALID;
        }

        let spectrum_buffer = buffer::MappedAllocation::new(size.height as usize, device)?;
        let output_buffer =
            buffer::MappedAllocation::new((size.width * size.height) as usize, device)?;

        self.spectrum_idx = spectrum_buffer.bound(device);
        self.output_idx = output_buffer.bound(device);

        self.spectrum_buffer = Some(spectrum_buffer);
        self.output_buffer = Some(output_buffer);

        Ok(())
    }

    /// The draw will first update and flush the spectrum buffer with the latest CQT data from
    /// upstream.  It then needs to run a compute shader to draw the spectrum onto the output
    /// Buffer.  The output buffer is transitioned for buffer copy and then written to the render
    /// target, an Image, which can be presented normally (de-coupled downstream concern).

    /// XXX In order to initiate feedback rendering, we will need to sample the previous output into
    /// a new input.  This will have the new base output drawn over it.  We will then dynamically
    /// composite the output and stuff.
    pub fn draw(
        &mut self,
        device: &Device,
        cb: &RecordingBuffer<Graphics, OneTime>,
        acquired_image: &AcquiredImage,
        input: &[crate::audio::cqt::Cqt],
    ) {
        // Transition the output layout for writing.
        let out_image = acquired_image.image;

        let mut i = 0usize;
        let spectrum = self.spectrum_buffer.as_mut().unwrap();
        spectrum.as_mut_slice().iter_mut().for_each(|o| {
            let bin = &input[i];
            i += 1;
            i = i % 599; // XXX Index wut?
            *o = SpectrumSample {
                left_decibels: bin.left_perceptual,
                left_mag: bin.left.mag(),
                right_decibels: bin.right_perceptual,
                right_mag: bin.right.mag(),
            }
        });
        spectrum.flush(device).unwrap();

        // Barrier between flush and use in shader
        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_pre(&cb, device);

        // Tell shader the location and geometry of the input
        let extent = acquired_image.extent;
        let push_constants = SpectrumConstants {
            window_width: (extent.width as f32).into(),
            window_height: (extent.height as f32).into(),
            spectrum_idx: self.spectrum_idx,
            output_idx: self.output_idx,
        };
        self.pipeline.push(device, **cb, &push_constants);

        // We will do a lot of these full-buffer shaders.  These workgroup dispatch sizes need to be
        // declared in one place and then propagated into slang, such as with a `slang!` macro.
        let workgroup_size_x = 4;
        let workgroup_size_y = 8;

        let dispatch_x = (extent.width + workgroup_size_x - 1) / workgroup_size_x;
        let dispatch_y = (extent.height + workgroup_size_y - 1) / workgroup_size_y;

        self.pipeline
            .dispatch(device, **cb, dispatch_x, dispatch_y, 1);

        // Insert a buffer barrier so we can write it to the target
        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_post(&cb, device);

        // Copy the buffer to the image.
        let region = buffer::buffer_image_copy_full(extent);
        unsafe {
            device.as_raw().cmd_copy_buffer_to_image(
                **cb,
                self.output_buffer.as_ref().unwrap().buffer,
                out_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    pub fn destroy(self, device: &mut Device) -> Result<(), utate::MutateError> {
        let Self {
            pipeline,
            spectrum_buffer,
            output_buffer,
            ..
        } = self;
        unsafe {
            pipeline.destroy(device);
            if let Some(allocated) = spectrum_buffer {
                allocated.destroy(&device)?;
                device.descriptors.unbind_ssbo(self.spectrum_idx);
            }
            if let Some(allocated) = output_buffer {
                allocated.destroy(&device)?;
                device.descriptors.unbind_ssbo(self.output_idx);
            }
        }
        Ok(())
    }
}
