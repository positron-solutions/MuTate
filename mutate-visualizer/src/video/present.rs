// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// A swapchain exists when we are presenting to a Surface.  We can use it as a render target.  Not
// all render targets need presentation, but a swapchain does.  Aligning the fields and structs with
// this abstraction is underway.

// NEXT enable options for variable and fixed frame rate rendering, likely using MAILBOX for
// variable and FIFO for fixed rate.  This requires looking at the actual swapchain image count
// before allocating per-image resources like command buffers.

// XXX move into Vulkan!
use std::slice;

use ash::khr::present_wait;
use ash::vk;
use smallvec::SmallVec;

use mutate_lib::vulkan::dispatch::{
    command::{CommandPool, RecordingSlot},
    sync,
};
use mutate_lib::{self as utate, prelude::*};

// Acquired images come from the swapchain, which might have to do something obtuse and hand us back
// a bizarre index, such as 2, due to the compositor being late at giving us the image back.
pub struct AcquiredImage {
    /// Used during acquisition.
    pub image_available: vk::Semaphore,
    /// The index provided by the swapchain (which may cycle swapchain images out of order) during
    /// acquisition.
    pub swapchain_image_index: u32,
    pub image: vk::Image,
    pub present_ready: vk::Semaphore,
}

/// A package deal!
///
/// ### Rule of Four
///
/// Let's just say no sane swapchain is coming back with four images.  With four slots, we always
/// have enough space for any size of swapchain.  We only take up a few 64bit pointers that won't
/// pack or load much better for three elements anyway.  In return, we save a lot of complication.
pub struct SwapchainContext {
    swapchain: vk::SwapchainKHR,
    loader: ash::khr::swapchain::Device,
    image_views: SmallVec<vk::ImageView, 4>,
    images: SmallVec<vk::Image, 4>,
    frames: usize,
    frame_index: usize,
    image_available_semaphores: [vk::Semaphore; 4],
    present_ready_semaphores: [vk::Semaphore; 4],
}

impl SwapchainContext {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext { entry, instance } = &vk_context;
        // XXX device_context method?
        let loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &device_context.device());

        let swapchain_ci = vk::SwapchainCreateInfoKHR {
            surface: surface.as_raw(),
            min_image_count: 3,
            image_format: surface.format.format,
            image_color_space: surface.format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface.pre_transform,
            composite_alpha: surface.composite_alpha,
            present_mode: surface.present_mode,
            clipped: vk::TRUE,
            flags: vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
            ..Default::default()
        };

        let swapchain = unsafe { loader.create_swapchain(&swapchain_ci, None).unwrap() };
        let images = unsafe { loader.get_swapchain_images(swapchain).unwrap() };
        let frames = images.len();

        let image_views: SmallVec<_, 4> = images
            .iter()
            .map(|&image| {
                let view_ci = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface.format.format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                unsafe {
                    device_context
                        .device
                        .create_image_view(&view_ci, None)
                        .unwrap()
                }
            })
            .collect();

        let image_available_semaphores = std::array::from_fn(|_| device_context.make_semaphore());
        let present_ready_semaphores = std::array::from_fn(|_| device_context.make_semaphore());
        Self {
            present_ready_semaphores,
            image_available_semaphores,
            loader,
            images: images.into(),
            image_views: image_views,
            swapchain,
            frames,
            frame_index: 0,
        }
    }

    pub fn recreate(
        &mut self,
        device_context: &DeviceContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) {
        let device = device_context.device();

        // Destroy old image views — images themselves are owned by the swapchain
        unsafe {
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
        }

        let swapchain_ci = vk::SwapchainCreateInfoKHR {
            surface: surface.as_raw(),
            min_image_count: self.frames as u32,
            image_format: surface.format.format,
            image_color_space: surface.format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface.pre_transform,
            composite_alpha: surface.composite_alpha,
            present_mode: surface.present_mode,
            clipped: vk::TRUE,
            flags: vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
            old_swapchain: self.swapchain,
            ..Default::default()
        };

        let new_swapchain = unsafe { self.loader.create_swapchain(&swapchain_ci, None).unwrap() };

        // Old swapchain is retired — safe to destroy now that new one is created
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }

        self.swapchain = new_swapchain;

        self.images = unsafe {
            self.loader
                .get_swapchain_images(self.swapchain)
                .unwrap()
                .into()
        };

        self.image_views = self
            .images
            .iter()
            .map(|&image| {
                let view_ci = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface.format.format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&view_ci, None).unwrap() }
            })
            .collect();

        let new_frames = self.images.len();
        if new_frames != self.frames {
            self.frames = new_frames;
            // LIES while this avoids a degenerate index that would crash, it might attempt to use
            // an image in flight unless acquire is properly called.  The semaphore wait before
            // swapchain recreation is what *actually* protects us.
            self.frame_index = self.frame_index.min(new_frames - 1);
        }
    }

    // XXX a horrible function?
    pub fn present(&self, queue: vk::Queue, present_info: &vk::PresentInfoKHR) {
        unsafe {
            match self.loader.queue_present(queue, present_info) {
                Ok(_) => {
                    // MAYBE How to interpret false?
                }
                Err(result) => eprintln!("presentation error: {:?}", result),
            };
        }
    }

    pub fn destroy(&self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            for view in &self.image_views {
                device.destroy_image_view(*view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
            self.image_available_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
            self.present_ready_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
        }
    }

    /// XXX mark chain dirty if this fails
    pub fn acquire(&mut self) -> AcquiredImage {
        let idx = self.frame_index as usize;
        let image_available = self.image_available_semaphores[idx];
        let present_ready = self.present_ready_semaphores[idx];
        self.frame_index = (idx + 1) % self.frames;

        let (swapchain_image_index, _) = unsafe {
            self.loader
                .acquire_next_image(
                    self.swapchain,
                    100_000_000, // 100ms
                    image_available,
                    vk::Fence::null(),
                )
                .unwrap() // XXX will error in ways we should catch
        };

        let image = self.images[swapchain_image_index as usize];

        AcquiredImage {
            image_available,
            swapchain_image_index,
            image,
            present_ready,
        }
    }
}

pub struct SurfacePresent {
    swapchain: SwapchainContext,

    frame_counter: u64,
    in_flight: vk::Semaphore,

    /// Present wait measurement
    present: sync::PresentConsumer,

    // NEXT Pool rings can likely be abstracted.  It's a very solid piece of infrastructure.
    // Just enough for front & back frame.
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
        let VkContext { entry, instance } = &vk_context;

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

        let queue_family_index = device_context.queues.graphics_family_index;
        let pools: [CommandPool; 2] =
            std::array::from_fn(|_| CommandPool::new(device_context, queue_family_index));

        let present = sync::PresentConsumer::new(vk_context, device_context, swapchain.swapchain)
            .expect("Could not start up the present wait gizmo");

        Self {
            frame_counter: 0,
            in_flight,
            present,

            recording_index: 0,
            pools,

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

        let queue = &context.queues.graphics_queue();
        unsafe {
            context
                .device()
                .queue_submit2(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }
    }

    // XXX this function demonstrates that command buffer presentation kind of re-couples at
    // presentation for swapchain stuff.  The present wait tangling demonstrates that we will need
    // some builders and intermediate structures to fan in any kind of composed behaviors.
    pub fn present(&mut self, device_context: &DeviceContext, acquired_image: AcquiredImage) {
        let queue = &device_context.queues.graphics_queue();
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

        self.swapchain.present(*queue, &present_info);
        self.present.notify_waiter();
    }
}
