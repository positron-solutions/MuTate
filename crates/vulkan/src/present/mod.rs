// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Present
//!
//! Presentation tends to live slightly outside of command recording and has a bit specific
//! synchronization requirements.  This module encapsulates the pre and post-render integration with
//! surrounding `Surface` and `Swapchain`.

pub mod surface;
pub mod swapchain;

use std::marker::PhantomData;
use std::slice;

use ash::vk::Handle;

use crate::dispatch::pw;
use crate::internal::*;

// NEXT update command buffer wrapper to support beginning and ending rendering, then use that to
// implement GraphicsPresent
// NEXT create kinds of targets that may only use specific layouts that are valid for the upstream
// render target.
// XXX Target is not yet used
pub struct Target<Layout> {
    /// The image view is necessary to call begin rendering.
    pub image_view: vk::ImageView,
    /// Extent describes the dimensions of the output image.
    pub extent: vk::Extent2D,
    /// Format is determined at runtime for swapchain presentation but can be any format for
    /// non-presentation cases.
    pub format: vk::Format,
    _layout: PhantomData<Layout>,
}

pub struct GraphicsPresent {
    pool_ring: PoolRing<Graphics>,
    /// Present wait measurement
    present: pw::PresentConsumer,
    queue: Queue<Graphics>,
    swapchain: SwapchainContext,
}

impl GraphicsPresent {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Result<Self, VulkanError> {
        let swapchain = SwapchainContext::new(device_context, vk_context, surface, extent)?;
        let queue = device_context
            .queues
            .graphics(vk_context, surface, QueuePriority::High)
            .ok_or(VulkanError::QueueNotFound)?;
        let pool_ring = PoolRing::new(device_context, &queue)?;
        let present = pw::PresentConsumer::new(vk_context, device_context, *swapchain.as_raw())?;
        Ok(Self {
            present,
            pool_ring,
            queue,
            swapchain,
        })
    }

    pub fn draw<F, G>(
        &mut self,
        device_ctx: &DeviceContext,
        draw_fn: F,
        post_draw_fn: G,
    ) -> Result<(), VulkanError>
    where
        F: FnOnce(&RecordingBuffer<Graphics, OneTime>, &AcquiredImage),
        G: FnOnce(),
    {
        let acquired_image = self.swapchain.acquire()?;

        // DEBT Full 1s timeout, but realistically, self-pacing should make this rarely crash until
        // error handling gets cleaned up.  Slow frames are warnings.
        let (pool, intent) = self.pool_ring.acquire(device_ctx, 1_000_000_000).unwrap();
        let cb = pool.primary(device_ctx).unwrap();

        // DEBT this code uses the old style struct pattern.  It is some of the last remaining code
        // using that style.
        let barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image: acquired_image.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let dep_info = vk::DependencyInfo {
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier,
            ..Default::default()
        };

        unsafe { device_ctx.device().cmd_pipeline_barrier2(*cb, &dep_info) };

        let color_attachment = vk::RenderingAttachmentInfo {
            image_view: todo!(), // XXX swapchain AcquiredImage missing ImageView
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            ..Default::default()
        };

        let render_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: todo!(), // XXX AcquiredImage missing extent
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        unsafe { device_ctx.device().cmd_begin_rendering(*cb, &render_info) };

        draw_fn(&cb, &acquired_image);

        let device = device_ctx.device();
        unsafe { device.cmd_end_rendering(*cb) };

        let barrier2 = vk::ImageMemoryBarrier2 {
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
            dst_access_mask: vk::AccessFlags2::empty(),

            // NOTE source layout differs in graphics vs compute case
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,

            image: acquired_image.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let dep2 = vk::DependencyInfo {
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier2,
            ..Default::default()
        };

        unsafe { device.cmd_pipeline_barrier2(*cb, &dep2) };

        let recorded = cb.end(device_ctx).unwrap();
        self.queue
            .submission()
            .wait_binary(
                acquired_image.image_available,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .execute(recorded)
            .signal_binary(
                acquired_image.present_ready,
                vk::PipelineStageFlags2::ALL_GRAPHICS,
            )
            .signal(intent, vk::PipelineStageFlags2::ALL_COMMANDS)
            .submit(device_ctx, vk::Fence::null())
            .unwrap();
        post_draw_fn();

        match self.present.read_last_present() {
            Some(last) => {
                // if last.last_window != std::time::Duration::MAX {
                //     println!("present duration: {:8.0}", last.last_window.as_micros())
                // }
            }
            None => {}
        }
        let next_id = self.present.next_present_id();
        let mut present_id = vk::PresentIdKHR::default().present_ids(slice::from_ref(&next_id));
        let present_ready = acquired_image.present_ready.as_raw();
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&present_ready))
            .swapchains(slice::from_ref(&self.swapchain.as_raw()))
            .image_indices(slice::from_ref(&acquired_image.swapchain_image_index))
            .push_next(&mut present_id);

        self.swapchain
            .present(unsafe { self.queue.raw() }, &present_info);
        self.present.notify_waiter();
        Ok(())
    }

    pub fn destroy(self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            let Self {
                swapchain,
                pool_ring,
                ..
            } = self;

            swapchain.destroy(context);
            pool_ring.destroy(context);
        }
    }

    pub fn recreate_images(
        &mut self,
        device_ctx: &DeviceContext,
        surface: &VkSurface,
        size: vk::Extent2D,
    ) {
        let device = &device_ctx.device;
        // XXX replace with semaphore wait on last frame in flight
        unsafe {
            device.device_wait_idle().unwrap();
        }

        self.swapchain.recreate(device_ctx, surface, size);
        self.present
            .notify_swapchain_recreation(*self.swapchain.as_raw());
    }
}

/// Present a swapchain image written by compute pipeline.  This struct encapsulates the swapchain
/// acquisition, queue submission, and presentation.
pub struct ComputePresent {
    pool_ring: PoolRing<Graphics>,
    /// Present wait measurement
    present: pw::PresentConsumer,
    queue: Queue<Graphics>,
    swapchain: SwapchainContext,
}

impl ComputePresent {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Result<Self, VulkanError> {
        let swapchain = SwapchainContext::new(device_context, vk_context, surface, extent)?;
        let queue = device_context
            .queues
            .graphics(vk_context, surface, QueuePriority::High)
            .ok_or(VulkanError::QueueNotFound)?;
        let pool_ring = PoolRing::new(device_context, &queue)?;
        let present = pw::PresentConsumer::new(vk_context, device_context, *swapchain.as_raw())?;
        Ok(Self {
            present,
            pool_ring,
            queue,
            swapchain,
        })
    }

    pub fn draw<F, G>(
        &mut self,
        device_ctx: &DeviceContext,
        draw_fn: F,
        post_draw_fn: G,
    ) -> Result<(), VulkanError>
    where
        F: FnOnce(&RecordingBuffer<Graphics, OneTime>, &AcquiredImage),
        G: FnOnce(),
    {
        let acquired_image = self.swapchain.acquire()?;

        // DEBT Full 1s timeout, but realistically, self-pacing should make this rarely crash until
        // error handling gets cleaned up.  Slow frames are warnings.
        let (pool, intent) = self.pool_ring.acquire(device_ctx, 1_000_000_000).unwrap();
        let cb = pool.primary(device_ctx).unwrap();

        draw_fn(&cb, &acquired_image);

        let recorded = cb.end(device_ctx).unwrap();
        self.queue
            .submission()
            .wait_binary(
                acquired_image.image_available,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .execute(recorded)
            .signal_binary(
                acquired_image.present_ready,
                vk::PipelineStageFlags2::ALL_GRAPHICS,
            )
            .signal(intent, vk::PipelineStageFlags2::ALL_COMMANDS)
            .submit(device_ctx, vk::Fence::null())
            .unwrap();
        post_draw_fn();

        match self.present.read_last_present() {
            Some(last) => {
                // if last.last_window != std::time::Duration::MAX {
                //     println!("present duration: {:8.0}", last.last_window.as_micros())
                // }
            }
            None => {}
        }
        let next_id = self.present.next_present_id();
        let mut present_id = vk::PresentIdKHR::default().present_ids(slice::from_ref(&next_id));
        // NOTE had to stack this as_raw while internal use of raw is already stable.
        let present_ready = acquired_image.present_ready.as_raw();
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&present_ready))
            .swapchains(slice::from_ref(self.swapchain.as_raw()))
            .image_indices(slice::from_ref(&acquired_image.swapchain_image_index))
            .push_next(&mut present_id);

        self.swapchain
            .present(unsafe { self.queue.raw() }, &present_info);
        self.present.notify_waiter();
        Ok(())
    }

    pub fn destroy(self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            let Self {
                swapchain,
                pool_ring,
                ..
            } = self;

            swapchain.destroy(context);
            pool_ring.destroy(context);
        }
    }

    // NOTE unless the surface retains knowledge of the window, it cannot know the size.  We need to
    // bind window and surface but also to provide a non-window surface for those odd cases.
    pub fn recreate_images(
        &mut self,
        device_ctx: &DeviceContext,
        surface: &VkSurface,
        size: vk::Extent2D,
    ) {
        let device = device_ctx.device();
        // XXX replace with semaphore wait on last frame in flight
        unsafe {
            device.device_wait_idle().unwrap();
        }
        self.swapchain.recreate(device_ctx, surface, size);
        self.present
            .notify_swapchain_recreation(*self.swapchain.as_raw());
    }
}
