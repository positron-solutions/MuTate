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
use smallvec::SmallVec;

// XXX go fix up prelude for most of these.
use mutate_lib::vulkan::{
    dispatch::{
        command::{CommandPool, RecordingSlot},
        sync,
    },
    prelude::*,
    present::swapchain::{AcquiredImage, SwapchainContext},
};
use mutate_lib::{self as utate, prelude::*};

pub struct SurfacePresent {
    swapchain: SwapchainContext,

    frame_counter: u64,
    in_flight: vk::Semaphore,

    /// Present wait measurement
    present: sync::PresentConsumer,

    // NEXT Pool rings can likely be abstracted.  It's a very solid piece of infrastructure.
    // Just enough for front & back frame.  Didn't we do this with recording slots and rings?
    queue: Queue<Graphics>,
    pools: [CommandPool; 2],
    /// A ring index, always advances by 1.  Longer pool rings could be used if there's a reason for
    /// longer serial work to pipeline in parallel, likely using only a few warps (because otherwise
    /// it's asking the scheduler to make it parallel).
    recording_index: u32,
}

impl SurfacePresent {
    // TODO give swapchain during construction?
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext {
            entry, instance, ..
        } = &vk_context;

        let swapchain = SwapchainContext::new(device_context, vk_context, surface, extent);

        // NEXT bon builder for semaphores.  This is nonsense.
        let mut type_ci = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };
        let semaphore_ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_ci);
        let in_flight = unsafe {
            device_context
                .device()
                .create_semaphore(&semaphore_ci, None)
                .unwrap()
        };

        // NEXT make the command pools / rings / recording slots do this without dropping to raw types.
        let queue = device_context
            .queues
            .graphics(vk_context, surface, QueuePriority::High)
            .unwrap();
        let pools: [CommandPool; 2] =
            std::array::from_fn(|_| CommandPool::new(device_context, queue.family()));

        let present = sync::PresentConsumer::new(vk_context, device_context, swapchain.swapchain)
            .expect("Could not start up the present wait gizmo");

        Self {
            frame_counter: 0,
            in_flight,
            present,

            recording_index: 0,
            pools,
            queue,

            swapchain,
        }
    }

    pub fn destroy(&self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            self.swapchain.destroy(context);
            for pool in &self.pools {
                pool.destroy(context);
            }
            device.destroy_semaphore(self.in_flight, None);
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

    /// Wait for the previous submission to complete
    pub fn draw_wait(&mut self, device_context: &DeviceContext) {
        let wait_value = self.frame_counter;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&self.in_flight))
            .values(slice::from_ref(&wait_value));
        unsafe {
            device_context
                .device()
                .wait_semaphores(&wait_info, 100_000_000) // 100ms
                .unwrap();
        }
    }

    /// Return a hot command buffer, image, and associated information used for drawing.
    pub fn render_target(
        &mut self,
        context: &DeviceContext,
        clear: vk::ClearValue, // XXX extract this because it's basically pretty annoying.
    ) -> (RecordingSlot, AcquiredImage) {
        let device = &context.device();
        let acquired_image = self.swapchain.acquire();

        // NEXT recording slots does indeed seem to be some kind of useful abstraction.  Its pools
        // and submission semaphores will
        let idx = self.recording_index as usize;
        let pool = &mut self.pools[idx];
        pool.reset(&context);
        let command_buffer = pool.buffer(&context);
        // XXX maybe do this on creation?
        // this is the compute variant.  the graphics variant below has only a little more ceremony
        // for roughly the same abstraction.
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(command_buffer, &begin).unwrap();
        }
        self.recording_index ^= 1;
        let slot = RecordingSlot { command_buffer };

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

        (slot, acquired_image)
    }

    /// Sync presentation image transform and present.
    /// NEXT close out the command buffer and do submission separately.
    pub fn post_draw(
        &mut self,
        context: &DeviceContext,
        slot: &RecordingSlot,
        acquired_image: &AcquiredImage,
    ) {
        let device = context.device();
        unsafe {
            context
                .device
                .end_command_buffer(slot.command_buffer)
                .unwrap()
        };

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

        let wait_info = vk::SemaphoreSubmitInfo {
            semaphore: acquired_image.image_available,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        };

        let next_value = self.frame_counter + 1;
        self.frame_counter = next_value;
        // XXX Pacing on in_flight, but this kind of duplicates present_ready?  present_ready is
        // a also useful for swapchain reconstruction.
        let in_flight = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.in_flight)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .value(next_value);

        let signal_infos = [
            vk::SemaphoreSubmitInfo {
                semaphore: acquired_image.present_ready,
                stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
                ..Default::default()
            },
            in_flight,
        ];
        let cb_info = vk::CommandBufferSubmitInfo {
            command_buffer: slot.command_buffer,
            ..Default::default()
        };

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(slice::from_ref(&wait_info))
            .signal_semaphore_infos(&signal_infos)
            .command_buffer_infos(slice::from_ref(&cb_info));

        unsafe {
            context
                .device()
                .queue_submit2(self.queue.raw(), &[submit], vk::Fence::null())
                .unwrap();
        }
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
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&acquired_image.present_ready))
            .swapchains(slice::from_ref(&self.swapchain.swapchain))
            .image_indices(slice::from_ref(&acquired_image.swapchain_image_index))
            .push_next(&mut present_id);

        self.swapchain
            .present(unsafe { self.queue.raw() }, &present_info);
        self.present.notify_waiter();
    }
}
