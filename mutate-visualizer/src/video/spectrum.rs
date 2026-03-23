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
    context::descriptors::SsboIdx,
    dispatch::command::{CommandPool, RecordingSlot},
    present::swapchain::AcquiredImage,
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

#[repr(C)]
pub struct PushConstants {
    pub window_size: [f32; 2],
    pub spectrum_idx: SsboIdx,
    pub output_idx: SsboIdx,
}

// This will be an interface after more nodes exist
pub struct SpectrumNode {
    pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,

    spectrum_buffer: Option<buffer::MappedAllocation<SpectrumSample>>,
    output_buffer: Option<buffer::MappedAllocation<rgb::Rgba<u8>>>,

    spectrum_idx: SsboIdx,
    output_idx: SsboIdx,
}

const ENTRY_POINT: &[u8] = b"main\0";

impl SpectrumNode {
    pub fn new(context: &DeviceContext) -> Self {
        let device = context.device();
        let assets = assets::AssetDirs::new();

        let compute_spv = assets.find_shader("spectrum/compute").unwrap();
        let compute_module_ci = vk::ShaderModuleCreateInfo::default().code(&compute_spv);
        let compute_shader_module = unsafe {
            device
                .create_shader_module(&compute_module_ci, None)
                .unwrap()
        };
        let shader_stage = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: compute_shader_module,
            p_name: ENTRY_POINT.as_ptr() as *const std::os::raw::c_char,
            ..Default::default()
        };

        let push_constant_range = vk::PushConstantRange {
            // NEXT the usage of push constants must be reflected upstream in the `stage_flags`.
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            // Doing the math this way to make the tedious nature of manually accounting this stuff
            // obvious.
            size: std::mem::size_of::<PushConstants>() as u32,
        };

        // NOTE Since PipelineLayoutCI has descriptor counts, either we take away the ability to do
        // extra sets or the API has to absorb the tax.  Probably another case for bon to give us
        // y-not-both advantages.
        let layout = context.descriptors.layout();
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            set_layout_count: 1,
            p_set_layouts: layout.as_ptr(),
            ..Default::default()
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .unwrap()
        };
        let compute_pipeline_ci = vk::ComputePipelineCreateInfo {
            stage: shader_stage,
            layout: pipeline_layout,
            ..Default::default()
        };
        let compute_pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[compute_pipeline_ci], None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(compute_shader_module, None);
        }

        Self {
            compute_pipeline,
            pipeline_layout,

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
        context: &mut DeviceContext,
        size: vk::Extent2D,
    ) -> Result<(), utate::MutateError> {
        let spectrum_buffer = buffer::MappedAllocation::new(size.height as usize, context)?;
        let output_buffer =
            buffer::MappedAllocation::new((size.width * size.height) as usize, context)?;

        self.spectrum_idx = spectrum_buffer.bound(context);
        self.output_idx = output_buffer.bound(context);

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
        recording_slot: &RecordingSlot,
        acquired_image: &AcquiredImage,
        input: &[crate::audio::cqt::Cqt],
        context: &crate::DeviceContext,
        // NEXT This extent is basically part of the target
        extent: vk::Extent2D,
    ) {
        let cb = recording_slot.command_buffer;
        let device = context.device();

        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
        }

        // Transition the output layout for writing.
        let out_image = acquired_image.image;
        let range = image::range();
        image::transition_layout(
            out_image,
            &cb,
            range,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            context,
        );

        let mut i = 0usize;
        let spectrum = self.spectrum_buffer.as_mut().unwrap();
        // XXX write some real CQT data in
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
        spectrum.flush(context).unwrap();

        // Barrier between flush and use in shader
        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_pre(&cb, context);

        // Tell shader the location and geometry of the input
        // XXX doing alchemy here to illustrate the silliness of doing this
        let push_constants = PushConstants {
            window_size: [extent.width as f32, extent.height as f32],
            spectrum_idx: self.spectrum_idx,
            output_idx: self.output_idx,
        };
        unsafe {
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &push_constants as *const PushConstants as *const u8,
                    std::mem::size_of::<PushConstants>(),
                ),
            );
        }

        // We will do a lot of these full-buffer shaders.  These workgroup dispatch sizes need to be
        // declared in one place and then propagated into slang, such as with a `slang!` macro.
        let workgroup_size_x = 8;
        let workgroup_size_y = 16;

        let dispatch_x = (extent.width + workgroup_size_x - 1) / workgroup_size_x;
        let dispatch_y = (extent.height + workgroup_size_y - 1) / workgroup_size_y;

        // Draw into the output buffer :-)
        unsafe {
            // XXX descriptor set binds necessary for all graphics pipelines 🤮.  Any time push
            // constants (or sets, which we avoid ever changing) change, we need to rebind
            // descriptors before the pipeline.
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[context.descriptors.set()],
                &[],
            );
            device.cmd_dispatch(cb, dispatch_x, dispatch_y, 1);
        }

        // Insert a buffer barrier so we can write it to the target
        self.output_buffer
            .as_ref()
            .unwrap()
            .barrier_compute_post(&cb, context);

        // Copy the buffer to the image.
        let region = buffer::buffer_image_copy_full(extent);
        unsafe {
            device.cmd_copy_buffer_to_image(
                cb,
                self.output_buffer.as_ref().unwrap().buffer,
                out_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // Transition the output back for presentation
        // NEXT make unknown transitions fail instead!
        image::transition_layout(
            out_image,
            &cb,
            range,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            context,
        );
    }

    pub fn destroy(&self, context: &mut DeviceContext) -> Result<(), utate::MutateError> {
        let device = context.device();
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);

            if let Some(allocated) = &self.spectrum_buffer {
                allocated.destroy(&context)?;
                context.descriptors.unbind_ssbo(self.spectrum_idx);
            }
            if let Some(allocated) = &self.output_buffer {
                allocated.destroy(&context)?;
                context.descriptors.unbind_ssbo(self.output_idx);
            }
        }
        Ok(())
    }
}
