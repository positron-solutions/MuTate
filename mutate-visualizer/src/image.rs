// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Image
//!
//! The `Image` and `ImageView` gather up related Vulkan functionality.  We don't do everything that
//! the spec does, so as duplication emerges, attempt to generalize the functionality and
//! boilerplate down to a sub-language that does everything we need.
//!
//! This treatment does not use any kind of RAII.  You have validation layers and other Vulkan
//! debugging tools to spot lifecycle issues.

use ash::vk;

use mutate_lib::{self as utate, prelude::*};

use crate::util;

/// The memory and dimensions for an allocated Vulkan Image.
pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Image {
    pub fn new(
        context: &VkContext,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, utate::MutateError> {
        let device = context.device();

        let image_ci = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let image = unsafe { device.create_image(&image_ci, None)? };
        let mem_req = unsafe { device.get_image_memory_requirements(image) };

        let mem_props = unsafe {
            context
                .instance
                .get_physical_device_memory_properties(context.physical_device)
        };

        let memory_type_index = util::find_memory_type_index(
            &mem_req,
            &mem_props,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(utate::MutateError::Ash(
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ))?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index,
            ..Default::default()
        };

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        Ok(Self {
            image,
            memory,
            format,
            extent,
        })
    }

    pub fn destroy(self, context: &VkContext) -> Result<(), utate::MutateError> {
        let device = context.device();
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
        Ok(())
    }

    pub fn view(
        &self,
        context: &VkContext,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Result<ImageView, utate::MutateError> {
        let device = context.device();

        let view_ci = vk::ImageViewCreateInfo {
            image: self.image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: self.format,
            subresource_range: subresource_range,
            ..Default::default()
        };

        let view = unsafe { device.create_image_view(&view_ci, None)? };

        Ok(ImageView {
            view,
            image: self.image,
            format: self.format,
            subresource_range,
        })
    }

    pub fn default_view(&self, context: &VkContext) -> Result<ImageView, utate::MutateError> {
        let subresource_range = range();
        self.view(context, subresource_range)
    }

    /// Forwards to `transition_layout` function for regular `vk::image`.
    pub fn transition_layout(
        &self,
        cmd_buffer: vk::CommandBuffer,
        subresource_range: vk::ImageSubresourceRange,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        context: &VkContext,
    ) {
        transition_layout(
            self.image,
            &cmd_buffer,
            subresource_range,
            old_layout,
            new_layout,
            context,
        );
    }

    /// Transition from UNDEFINED → TRANSFER_DST_OPTIMAL for uploading data.
    pub fn transition_to_transfer_dst(&self, cmd_buffer: vk::CommandBuffer, context: &VkContext) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            context,
        );
    }

    /// Transition from TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL for sampling in shaders.
    pub fn transition_to_shader_read(&self, cmd_buffer: vk::CommandBuffer, context: &VkContext) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            context,
        );
    }

    /// Transition from UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL for depth/stencil attachments.
    // XXX This one is known to be pretty incomplete but not in use yet, so fix it when you need it.
    pub fn transition_to_depth_attachment(
        &self,
        cmd_buffer: vk::CommandBuffer,
        context: &VkContext,
    ) {
        self.transition_layout(
            cmd_buffer,
            range_stencil(), // full depth/stencil range
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            context,
        );
    }

    /// Transition from COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR for presenting swapchain images.
    pub fn transition_to_present(&self, cmd_buffer: vk::CommandBuffer, context: &VkContext) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            context,
        );
    }

    /// Transition from PRESENT_SRC_KHR → COLOR_ATTACHMENT_OPTIMAL for rendering to swapchain images.
    pub fn transition_from_present(&self, cmd_buffer: vk::CommandBuffer, context: &VkContext) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            context,
        );
    }
}

/// View and layout transition functionality for a Vulkan Image.
pub struct ImageView {
    pub view: vk::ImageView,
    pub image: vk::Image,
    pub format: vk::Format,
    pub subresource_range: vk::ImageSubresourceRange,
}

impl ImageView {
    pub fn destroy(self, context: &VkContext) -> Result<(), utate::MutateError> {
        unsafe {
            context.device().destroy_image_view(self.view, None);
        }
        Ok(())
    }
}

/// Full color range, the most common.
pub fn range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Full depth/stencil range.
pub fn range_stencil() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Transition image layout with an appropriate barrier.  Automatically infers src/dst masks and
/// pipeline stages for common usage patterns.  `Image` uses this implementation.
pub fn transition_layout(
    image: vk::Image,
    cmd_buffer: &vk::CommandBuffer,
    subresource_range: vk::ImageSubresourceRange,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    context: &VkContext,
) {
    // Infer barrier settings based on old/new layout
    let (src_stage, dst_stage, src_access, dst_access) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
        (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::MEMORY_READ,
        ),
        (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::MEMORY_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),
        // From color attachment to shader read (offscreen render → sampling)
        (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),

        // From shader read back to color attachment (rare, but valid)
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        ),

        // Transfer SRC / DST for image blits or copies
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        ),
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        ),

        // Sample depth in shader
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ) => (
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        ),

        // Transferring compute into presentation images
        (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::MEMORY_READ,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::MEMORY_READ,
        ),

        _ => panic!(
            "Unsupported layout transition: {:?} → {:?}",
            old_layout, new_layout
        ),
    };

    let barrier = vk::ImageMemoryBarrier {
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range,
        src_access_mask: src_access,
        dst_access_mask: dst_access,
        ..Default::default()
    };

    unsafe {
        context.device().cmd_pipeline_barrier(
            *cmd_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
