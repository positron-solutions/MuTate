// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Image
//!
//! ⚠️ This module was just a stub to get going.
//!
//! What we want is support for an [`ImageSpec`] that can be used to populate the arguments of
//! dependents.  My shader uses an image?  A spec tells the runtime how to initialize that image and
//! the handle is provided so that recording can push constants / write control headers or however
//! else the shader reads the image.  Does the image normally get fed by an upstream workload?  Spin
//! up that workload and only give us pointers to images that will be ready at render time.  Is the
//! image just a temporary output used somewhere downstream?  Are we the upstream workload?  That's
//! how the upstream and downstream tie together.  They found each other via runtime resolved spec.
//!
//! In the meantime, we need enough manual functionality to write provisioning and teardown methods
//! and to get usage handles into command recorders.  Provide borrowing views that can be shared
//! where necessary.  Those will morph into runtime-supplied handles that the user doesn't care
//! about.
//!
//! Not everything in the Vulkan spec is modern.  Add support as needed.  Add ergonomics where
//! certain decisions have collapsed to subsets.  Provide escape hatches so that the subsets are not
//! restrictive.
//!
//! Image granularity and dedicated allocations are some validity concerns / optimizations that will
//! tangle with sub-allocation behavior.  Be sure, a lot of very smart people have battled
//! extensively to get image packing and usages optimal in the extreme cases.

// DEBT see decision on general layout.  Others have reported that we might not really be able to do
// ourselves many favors (or not ones worth the tradeoffs) using explicit layouts on modern desktop
// cards.
// NOTE It took a lot of self control not to go down the rabbit hole of figuring out what this
// module needs to look like in 2026.  Good luck!
// MAYBE the vk::ImageCreateInfo already supports lots of knobs and is a builder.  Maybe some trait
// extensions for reduced API surface would be appropriate.
// XXX The barrier stuff is likely only useful to get a vague idea of the problems being solved.
// Full re-write for sure.
// DEBT sub-allocation support

use crate::internal::*;

/// The memory and dimensions for an allocated Vulkan Image.
pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Image {
    pub fn new(
        device: &Device,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, VulkanError> {
        // NEXT image_ci needs the high-level AsRef + direct treatment.
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

        // XXX basically do the same thing that we did for buffers.  However, the valid memory types
        // are likely more constrained.
        let image = unsafe { device.create_image(&image_ci, None)? };

        let mem_req = unsafe { device.get_image_memory_requirements(image) };

        // Ah, bind up into a memory object
        // XXX use device.memory for this allocation as well.
        let memory_type_index = 0;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index,
            ..Default::default()
        };

        // XXX Can we
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_image_memory(image, memory, 0)? };

        Ok(Self {
            image,
            memory,
            format,
            extent,
        })
    }

    pub fn destroy(self, device: &Device) -> Result<(), VulkanError> {
        unsafe {
            device.as_raw().destroy_image(self.image, None);
            device.as_raw().free_memory(self.memory, None);
        }
        Ok(())
    }

    /// Obtain a view of the image.
    pub fn view(
        &self,
        device: &Device,
        subresource_range: vk::ImageSubresourceRange,
    ) -> Result<ImageView, VulkanError> {
        let view_ci = vk::ImageViewCreateInfo {
            image: self.image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: self.format,
            subresource_range: subresource_range,
            ..Default::default()
        };

        let view = unsafe { device.as_raw().create_image_view(&view_ci, None)? };

        Ok(ImageView {
            view,
            image: self.image,
            format: self.format,
            subresource_range,
        })
    }

    pub fn default_view(&self, device: &Device) -> Result<ImageView, VulkanError> {
        let subresource_range = range();
        self.view(device, subresource_range)
    }

    /// Forwards to `transition_layout` function for regular `vk::image`.
    pub fn transition_layout(
        &self,
        cmd_buffer: vk::CommandBuffer,
        subresource_range: vk::ImageSubresourceRange,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        device: &Device,
    ) {
        transition_layout(
            self.image,
            &cmd_buffer,
            subresource_range,
            old_layout,
            new_layout,
            device,
        );
    }

    /// Transition from UNDEFINED → TRANSFER_DST_OPTIMAL for uploading data.
    pub fn transition_to_transfer_dst(&self, cmd_buffer: vk::CommandBuffer, device: &Device) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            device,
        );
    }

    /// Transition from TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL for sampling in shaders.
    pub fn transition_to_shader_read(&self, cmd_buffer: vk::CommandBuffer, device: &Device) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            device,
        );
    }

    /// Transition from UNDEFINED → DEPTH_STENCIL_ATTACHMENT_OPTIMAL for depth/stencil attachments.
    // XXX This one is known to be pretty incomplete but not in use yet, so fix it when you need it.
    pub fn transition_to_depth_attachment(&self, cmd_buffer: vk::CommandBuffer, device: &Device) {
        self.transition_layout(
            cmd_buffer,
            range_stencil(), // full depth/stencil range
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            device,
        );
    }

    /// Transition from COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR for presenting swapchain images.
    pub fn transition_to_present(&self, cmd_buffer: vk::CommandBuffer, device: &Device) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            device,
        );
    }

    /// Transition from PRESENT_SRC_KHR → COLOR_ATTACHMENT_OPTIMAL for rendering to swapchain images.
    pub fn transition_from_present(&self, cmd_buffer: vk::CommandBuffer, device: &Device) {
        self.transition_layout(
            cmd_buffer,
            range(), // full color range
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            device,
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
    pub fn destroy(self, device: &Device) -> Result<(), VulkanError> {
        unsafe {
            device.as_raw().destroy_image_view(self.view, None);
        }
        Ok(())
    }

    // XXX extremely raw.  Bludgeon if found.
    pub fn sampled(&self, device: &mut Device, layout: vk::ImageLayout) -> SampledImageIdx {
        // MAYBE not so sure about the layout choice
        let f = device.as_raw().clone();
        let descriptors = &mut device.descriptors;
        descriptors.bind_sampled_image(&f, self.view, layout)
        // device.bind_sampled_image(self.view, layout)
    }
}

/// Full color range, the most common.
pub fn range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

/// Full depth/stencil range.
pub fn range_stencil() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        .level_count(1)
        .layer_count(1)
}

/// Transition image layout with an appropriate barrier.  Automatically infers src/dst masks and
/// pipeline stages for common usage patterns.  `Image` uses this implementation.
// MAYBE some recommended using `vk::ImageLayout::GENERAL` layouts everywhere if supported, but
// these transitions are extremely straightforward to automate compared to other hazards.
pub fn transition_layout(
    image: vk::Image,
    cmd_buffer: &vk::CommandBuffer,
    subresource_range: vk::ImageSubresourceRange,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    device: &Device,
) {
    // XXX This is likely bullshit.  Barrier work is garbage so far.  Needs ransacking and a design
    // pass.
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

        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL) => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),

        _ => panic!(
            "Unsupported layout transition: {:?} -> {:?}",
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
        device.as_raw().cmd_pipeline_barrier(
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
