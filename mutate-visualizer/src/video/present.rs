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
use winit::{event_loop::ActiveEventLoop, window::Window};

use mutate_lib::vulkan::dispatch::sync;
use mutate_lib::{self as utate, prelude::*};

use crate::window::WindowExt;

pub struct DrawSync {
    pub render_finished: vk::Semaphore,
    pub image_available: vk::Semaphore,
    pub image_index: u32,
}

// XXX command buffers things mixed with images.. this is not right at all.  Perhaps by the time we
// hit render graph, dependencies and inputs / outputs are
pub struct DrawTarget {
    pub image: vk::Image,
    pub command_buffer: vk::CommandBuffer,
}

pub struct SurfacePresent {
    pub frames: usize,
    pub frame_index: usize,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub frame_counter: u64,
    pub in_flight: vk::Semaphore,

    pub render_finished_semaphores: Vec<vk::Semaphore>,
    /// Present wait measurement
    pub present: sync::PresentConsumer,

    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_images: Vec<vk::Image>,

    command_buffers: Vec<vk::CommandBuffer>,
}

impl SurfacePresent {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext { entry, instance } = &vk_context;
        // XXX device_context method?
        let swapchain_loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &device_context.device());
        // XXX the images are not actually being counted
        let swapchain_info = vk::SwapchainCreateInfoKHR {
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

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .unwrap()
        };
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        // Create image views
        let image_views: Vec<_> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo {
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
                        .create_image_view(&view_info, None)
                        .unwrap()
                }
            })
            .collect();

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

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // FIXME propagate image counts
        let image_available_semaphores = (0..3)
            .map(|_| unsafe {
                device_context
                    .device()
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let render_finished_semaphores = (0..3)
            .map(|_| unsafe {
                device_context
                    .device()
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let command_pool = device_context.queues.graphics_pool();

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: command_pool,
            command_buffer_count: 3,
            ..Default::default()
        };

        let command_buffers = unsafe {
            device_context
                .device
                .allocate_command_buffers(&alloc_info)
                .unwrap()
        };

        let present = sync::PresentConsumer::new(vk_context, device_context, swapchain)
            .expect("Could not start up the present wait gizmo");

        Self {
            swapchain,
            swapchain_image_views: image_views,
            swapchain_images: images,
            swapchain_loader,

            frames: 3,
            frame_index: 0,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            frame_counter: 0,
            in_flight,

            present,
        }
    }

    pub fn destroy(&self, context: &DeviceContext) {
        let device = &context.device;
        unsafe {
            for view in &self.swapchain_image_views {
                device.destroy_image_view(*view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.image_available_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
            self.render_finished_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
            device.destroy_semaphore(self.in_flight, None);
        }
    }

    // Basically carbon copy bullshit of the swapchain stuff.
    pub fn recreate_images(
        &mut self,
        surface: &VkSurface,
        size: vk::Extent2D,
        context: &DeviceContext,
    ) {
        let device = &context.device;
        let physical_device = context.physical_device;

        unsafe {
            device.device_wait_idle().unwrap();
        }

        // partial destruction
        unsafe {
            for view in &self.swapchain_image_views {
                device.destroy_image_view(*view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        // XXX Kill this
        // Recreation
        unsafe {
            let swapchain_info = vk::SwapchainCreateInfoKHR {
                surface: surface.as_raw(),
                // NOTE this minimum is minimum.  At least under MAILBOX, extra images may result.
                min_image_count: self.frames as u32,
                image_format: surface.format.format,
                image_color_space: surface.format.color_space,
                image_extent: size,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST,
                // NOTE the choice here needs to change if we ever support cross-queue present.
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: surface.pre_transform,
                composite_alpha: surface.composite_alpha,
                present_mode: surface.present_mode,
                clipped: vk::TRUE,
                flags: vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
                ..Default::default()
            };

            let swapchain = self
                .swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .unwrap();
            let images = self
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap();

            let image_views: Vec<_> = images
                .iter()
                .map(|&image| {
                    let view_info = vk::ImageViewCreateInfo {
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
                    device.create_image_view(&view_info, None).unwrap()
                })
                .collect();

            self.present.notify_swapchain_recreation(swapchain);

            self.swapchain = swapchain;
            self.swapchain_images = images;
            self.swapchain_image_views = image_views;
        }
    }

    /// XXX self pacing still not yet integrated
    pub fn draw_wait(&mut self, device_context: &DeviceContext) {
        let wait_value = self.frame_counter;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&self.in_flight))
            .values(slice::from_ref(&wait_value));
        unsafe {
            device_context
                .device()
                .wait_semaphores(&wait_info, u64::MAX)
                .unwrap();
        }
    }

    /// Return a hot command buffer, image, and associated information used for drawing.
    pub fn render_target(
        &mut self,
        context: &DeviceContext,
        clear: vk::ClearValue,
    ) -> (DrawSync, DrawTarget) {
        let device = &context.device;
        let idx = self.frame_index as usize;
        let image_available = self.image_available_semaphores[idx];
        let render_finished = self.render_finished_semaphores[idx];
        self.frame_index = (idx + 1) % self.frames;

        let (image_index, _) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    std::u64::MAX,
                    image_available,
                    vk::Fence::null(),
                )
                .unwrap()
        };

        let sync = DrawSync {
            image_available,
            render_finished,
            image_index,
        };

        let command_buffer = self.command_buffers[image_index as usize];
        let image = self.swapchain_images[image_index as usize];
        let image_view = self.swapchain_image_views[image_index as usize];

        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(command_buffer, &begin).unwrap();
        }

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

        let target = DrawTarget {
            image,
            command_buffer,
        };

        (sync, target)
    }

    /// Sync presentation image transform and present
    pub fn post_draw(
        &mut self,
        context: &DeviceContext,
        sync: DrawSync,
        target: DrawTarget,
        window: &Window,
    ) {
        let device = context.device();

        // XXX Needed for graphics
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

        unsafe {
            context
                .device
                .end_command_buffer(target.command_buffer)
                .unwrap()
        };

        let wait_info = vk::SemaphoreSubmitInfo {
            semaphore: sync.image_available,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        };

        let next_value = self.frame_counter + 1;
        self.frame_counter = next_value;
        let in_flight = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.in_flight)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .value(next_value);

        let signal_infos = [
            vk::SemaphoreSubmitInfo {
                semaphore: sync.render_finished,
                stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
                ..Default::default()
            },
            in_flight,
        ];

        let cb_info = vk::CommandBufferSubmitInfo {
            command_buffer: target.command_buffer,
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

        match self.present.read_last_present() {
            Some(last) => {
                if last.last_window != std::time::Duration::MAX {
                    println!("present duration: {:8.0}", last.last_window.as_micros())
                }
            }
            None => {}
        }
        let next_id = self.present.next_present_id();
        let mut present_id = vk::PresentIdKHR::default().present_ids(slice::from_ref(&next_id));

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&sync.render_finished))
            .swapchains(slice::from_ref(&self.swapchain))
            .image_indices(slice::from_ref(&sync.image_index))
            .push_next(&mut present_id);

        // XXX REMOVE to get rid of almost all window tangling
        // XXX break up this method into two, right here.
        // Winit says this helps align the window system latching with REDRAW_REQUESTED events.
        // However, it is supported only on Wayland at this time.
        window.pre_present_notify();

        unsafe {
            match self.swapchain_loader.queue_present(*queue, &present_info) {
                Ok(_) => {
                    // MAYBE How to interpret false?
                }
                Err(result) => eprintln!("presentation error: {:?}", result),
            };
        }
        self.present.notify_waiter();
    }
}
