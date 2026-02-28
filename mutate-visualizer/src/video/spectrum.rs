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
use utate::vulkan::{buffer, image, util};

#[repr(C)]
#[derive(Clone, Copy)]
/// Processed inputs for the shader.
struct SpectrumSample {
    left_decibels: f32,
    left_mag: f32,
    right_decibels: f32,
    right_mag: f32,
}

// This will be an interface after more nodes exist
pub struct SpectrumNode {
    pipeline_layout: vk::PipelineLayout,
    compute_pipeline: vk::Pipeline,

    spectrum_buffer: Option<buffer::MappedAllocation<SpectrumSample>>,
    output_buffer: Option<buffer::MappedAllocation<rgb::Rgba<u8>>>,

    // XXX the bindless style will need to take over.
    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
}

const ENTRY_POINT: &[u8] = b"main\0";

impl SpectrumNode {
    // NEXT what is format doing here?  why must it be passed with new?  Seems like presentation
    // details leaking into new, but it's reasonable drawing nodes need to learn about their
    // presentation targets.  This could also be considered provision time?
    pub fn new(context: &VkContext, format: vk::Format) -> Self {
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
            size: std::mem::size_of::<[f32; 2]>() as u32,
        };

        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 2,
        }];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        let layout = util::descriptor_set_layout(device).unwrap();
        let layouts = [layout];

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            set_layout_count: 1,
            p_set_layouts: layouts.as_ptr(),
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
            descriptor_set,

            spectrum_buffer: None,
            output_buffer: None,

            descriptor_pool: pool,
            layout: layout,
        }
    }

    // NEXT Provisioning and re-provisioning reactively to the upstream render target size is a very
    // important problem to work on!
    pub fn provision(
        &mut self,
        context: &VkContext,
        size: vk::Extent2D,
    ) -> Result<(), utate::MutateError> {
        let spectrum_buffer = buffer::MappedAllocation::new(size.height as usize, context)?;
        let output_buffer =
            buffer::MappedAllocation::new((size.width * size.height) as usize, context)?;

        let spectrum_buffer_info = vk::DescriptorBufferInfo {
            buffer: spectrum_buffer.buffer,
            offset: 0,
            range: spectrum_buffer.size_bytes,
        };

        let read = vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &spectrum_buffer_info,
            ..Default::default()
        };

        let output_buffer_info = vk::DescriptorBufferInfo {
            buffer: output_buffer.buffer,
            offset: 0,
            range: output_buffer.size_bytes,
        };

        let write = vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 1,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &output_buffer_info,
            ..Default::default()
        };

        unsafe {
            context.device().update_descriptor_sets(&[read, write], &[]);
        }

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
        target: &crate::video::present::DrawTarget,
        input: &[crate::audio::cqt::Cqt],
        context: &crate::VkContext,
        // NEXT This extent is basically part of the target
        extent: &vk::Extent2D,
    ) {
        let cb = target.command_buffer;
        let device = context.device();

        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
        }

        // Transition the output layout for writing.
        let out_image = target.image;
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

        // Use compute shader to write
        // MAYBE Push constants are a place where we really need automation
        //
        // - If I push data in the draw call, that should propagate into declaring the push constant
        //   range.
        // - When I push in the draw call, that should declare inputs in the slang, perhaps as a
        //   prepend to a Leptos style inline macro template, `slang!` or something.
        // - The data that I push should determine the type of the push constants everywhere.
        //
        // That might be some complex macro code but completely worth it because push constants are
        // a useful way to update control data that can be used downstream everywhere.  Storage
        // buffers might be a way to avoid some constants.
        let combined_push: [f32; 2] = [extent.width as f32, extent.height as f32];
        unsafe {
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    combined_push.as_ptr() as *const u8,
                    std::mem::size_of::<[f32; 2]>(),
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
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
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
        let region = buffer::buffer_image_copy_full(*extent);
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

    pub fn destroy(&self, context: &VkContext) -> Result<(), utate::MutateError> {
        let device = context.device();
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);

            if let Some(allocated) = &self.spectrum_buffer {
                allocated.destroy(&context)?;
            }

            if let Some(allocated) = &self.output_buffer {
                allocated.destroy(&context)?;
            }
        }
        Ok(())
    }
}
