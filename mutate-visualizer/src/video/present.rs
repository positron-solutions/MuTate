// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// What remains is to break this apart into the graphics vs compute-present paths and move it into
// Vulkan.  The extent handling is gross between here and main.  Synchronization might need a little
// better first class treatment.  Existing sync is just inherited first-pass code.
//
// Current vibes are that presentation is a runtime thing.  User will ask for a fully equipped
// window or an off-screen render loop. The contents we plug into those boxes don't need to care
// what they are rendering to.
//
// Both graphics-present and compute-present ultimately provide recording slots and their command
// buffers, so all graphics and compute users downstream can just consume the command buffers as-is.
// They don't need to know or care if presentation will happen or rendering is even on screen.

use std::slice;

use ash::khr::present_wait;
use ash::vk;

use mutate_lib::vulkan::{
    dispatch::pw,
    prelude::*,
    present::swapchain::{AcquiredImage, SwapchainContext},
};
use mutate_lib::{self as utate, prelude::*};

pub struct SurfacePresent {
    swapchain: SwapchainContext,

    /// Present wait measurement
    present: pw::PresentConsumer,

    queue: Queue<Graphics>,
    pool_ring: PoolRing<Graphics>,
    /// A ring index, always advances by 1.  Longer pool rings could be used if there's a reason for
    /// longer serial work to pipeline in parallel, likely using only a few warps (because otherwise
    /// it's asking the scheduler to make it parallel).
    recording_index: u32,
}

impl SurfacePresent {
    // MAYBE give a swapchain during construction?
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let swapchain = SwapchainContext::new(device_context, vk_context, surface, extent);
        let queue = device_context
            .queues
            .graphics(vk_context, surface, QueuePriority::High)
            .unwrap();
        let pool_ring = PoolRing::new(device_context, &queue).unwrap();
        let present = pw::PresentConsumer::new(vk_context, device_context, swapchain.swapchain)
            .expect("Could not start up the present wait gizmo");
        Self {
            present,

            recording_index: 0,
            pool_ring,
            queue,

            swapchain,
        }
    }

    pub fn destroy(self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            let SurfacePresent {
                swapchain,
                pool_ring,
                // in_flight,
                ..
            } = self;

            swapchain.destroy(context);
            pool_ring.destroy(context);
            // device.destroy_semaphore(in_flight, None);
        }
    }

    pub fn recreate_images(
        &mut self,
        surface: &VkSurface,
        size: vk::Extent2D,
        context: &DeviceContext,
    ) {
        let device = &context.device;
        let physical_device = context.physical_device;
        // XXX replace with semaphore wait on last frame in flight
        unsafe {
            device.device_wait_idle().unwrap();
        }

        self.swapchain.recreate(context, surface, size);
        self.present
            .notify_swapchain_recreation(self.swapchain.swapchain);
    }

    /// Return a hot command buffer, image, and associated information used for drawing.
    pub fn render_target(
        &mut self,
        context: &DeviceContext,
        clear: vk::ClearValue, // XXX extract this because it's basically pretty annoying.
    ) -> (
        SignalIntent,
        RecordingBuffer<Graphics, OneTime>,
        AcquiredImage,
    ) {
        let device = &context.device();
        // DEBT Full 1s timeout, but realistically, self-pacing should make this rarely crash until
        // error handling gets cleaned up.  Slow frames are warnings.
        let (pool, intent) = self.pool_ring.acquire(&context, 1_000_000_000).unwrap();
        let cb = pool.primary(&context).unwrap();
        let acquired_image = self.swapchain.acquire();

        // let barrier = vk::ImageMemoryBarrier2 {
        //     src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
        //     dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        //     old_layout: vk::ImageLayout::UNDEFINED,
        //     new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        //     src_access_mask: vk::AccessFlags2::empty(),
        //     dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        //     image,
        //     subresource_range: vk::ImageSubresourceRange {
        //         aspect_mask: vk::ImageAspectFlags::COLOR,
        //         level_count: 1,
        //         layer_count: 1,
        //         ..Default::default()
        //     },
        //     ..Default::default()
        // };

        // let dep_info = vk::DependencyInfo {
        //     image_memory_barrier_count: 1,
        //     p_image_memory_barriers: &barrier,
        //     ..Default::default()
        // };

        // unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        // let color_attachment = vk::RenderingAttachmentInfo {
        //     image_view: image_view,
        //     image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        //     load_op: vk::AttachmentLoadOp::CLEAR,
        //     store_op: vk::AttachmentStoreOp::STORE,
        //     clear_value: clear,
        //     ..Default::default()
        // };

        // let render_info = vk::RenderingInfo {
        //     render_area: vk::Rect2D {
        //         offset: vk::Offset2D { x: 0, y: 0 },
        //         extent: size,
        //     },
        //     layer_count: 1,
        //     color_attachment_count: 1,
        //     p_color_attachments: &color_attachment,
        //     ..Default::default()
        // };

        // unsafe { device.cmd_begin_rendering(command_buffer, &render_info) };

        (intent, cb, acquired_image)
    }

    /// Sync presentation, image transform, and present.
    // NEXT close out the command buffer and do submission separately.
    pub fn post_draw(
        &mut self,
        context: &DeviceContext,
        render_done: SignalIntent,
        cb: RecordingBuffer<Graphics, OneTime>,
        acquired_image: &AcquiredImage,
    ) {
        let device = context.device();

        // XXX Unique for graphics
        // unsafe { device.cmd_end_rendering(target.command_buffer) };

        // let barrier2 = vk::ImageMemoryBarrier2 {
        //     src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        //     src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        //     dst_stage_mask: vk::PipelineStageFlags2::NONE,
        //     dst_access_mask: vk::AccessFlags2::empty(),

        //     // XXX pre-present barrier was incorrect
        //     // old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        //     old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        //     new_layout: vk::ImageLayout::PRESENT_SRC_KHR,

        //     image: target.image,
        //     subresource_range: vk::ImageSubresourceRange {
        //         aspect_mask: vk::ImageAspectFlags::COLOR,
        //         level_count: 1,
        //         layer_count: 1,
        //         ..Default::default()
        //     },
        //     ..Default::default()
        // };

        // let dep2 = vk::DependencyInfo {
        //     image_memory_barrier_count: 1,
        //     p_image_memory_barriers: &barrier2,
        //     ..Default::default()
        // };

        // unsafe {
        //     vk_context
        //         .device
        //         .cmd_pipeline_barrier2(target.command_buffer, &dep2)
        // };

        let recorded = cb.end(context).unwrap();
        let raw_cb = recorded.kill(context).unwrap(); // XXX consuming method still needed

        self.queue
            .submission()
            .wait_binary(
                acquired_image.image_available,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .execute(raw_cb)
            .signal_binary(
                acquired_image.present_ready,
                vk::PipelineStageFlags2::ALL_GRAPHICS,
            )
            .signal(render_done, vk::PipelineStageFlags2::ALL_COMMANDS)
            .submit(device, vk::Fence::null())
            .unwrap();
    }

    // XXX this function demonstrates that command buffer presentation kind of re-couples at
    // presentation for swapchain stuff.  The present wait tangling demonstrates that we will need
    // some builders and intermediate structures to fan in any kind of composed behaviors.
    pub fn present(&mut self, device_context: &DeviceContext, acquired_image: AcquiredImage) {
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
            .swapchains(slice::from_ref(&self.swapchain.swapchain))
            .image_indices(slice::from_ref(&acquired_image.swapchain_image_index))
            .push_next(&mut present_id);

        // NEXT holding queue on the swapchain probably not problematic
        self.swapchain
            .present(unsafe { self.queue.raw() }, &present_info);
        self.present.notify_waiter();
    }
}
