// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// A swapchain exists when we are presenting to a Surface.  We can use it as a render target.  Not
// all render targets need presentation, but a swapchain does.  Aligning the fields and structs with
// this abstraction is underway.

// NEXT It is said that we can replace most of the fences with timeline semaphores.  Not sure what
// extensions are involved, but if so, it would be preferable to synchronize on the more flexible
// timeline semaphores.

// NEXT enable options for variable and fixed frame rate rendering, likely using MAILBOX for
// variable and FIFO for fixed rate.  This requires looking at the actual swapchain image count
// before allocating per-image resources like command buffers.

// XXX move into Vulkan!

use ash::{khr::present_wait::Device as PwDevice, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{event_loop::ActiveEventLoop, window::Window};

use mutate_lib::{self as utate, prelude::*};

use crate::window::WindowExt;

// XXX less bad.  consider if we can use timeline semaphores
pub struct DrawSync {
    pub in_flight: vk::Fence,
    pub render_finished: vk::Semaphore,
    pub image_available: vk::Semaphore,
    pub image_index: usize,
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
    pub in_flight_fences: Vec<vk::Fence>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub pw_device: PwDevice,
    // Even values are usable ids.  Odd successors indicate waiting on that id has not been
    // completed yet.
    present_id: u64,

    pub swapchain_loader: ash::khr::swapchain::Device,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_images: Vec<vk::Image>,

    command_buffers: Vec<vk::CommandBuffer>,
}

impl SurfacePresent {
    pub fn new(
        context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext { entry, instance } = &vk_context;
        // XXX device_context method?
        let swapchain_loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &context.device());
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
                unsafe { context.device.create_image_view(&view_info, None).unwrap() }
            })
            .collect();

        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let in_flight_fences: Vec<vk::Fence> = (0..3)
            .map(|_| unsafe { context.device().create_fence(&fence_info, None).unwrap() })
            .collect();

        let semaphore_info = vk::SemaphoreCreateInfo::default();

        // FIXME propagate image counts
        let image_available_semaphores = (0..3)
            .map(|_| unsafe {
                context
                    .device()
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let render_finished_semaphores = (0..3)
            .map(|_| unsafe {
                context
                    .device()
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let command_pool = context.queues.graphics_pool();

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: command_pool,
            command_buffer_count: 3,
            ..Default::default()
        };

        let command_buffers = unsafe {
            context
                .device
                .allocate_command_buffers(&alloc_info)
                .unwrap()
        };

        Self {
            swapchain,
            swapchain_image_views: image_views,
            swapchain_images: images,
            swapchain_loader,

            frames: 3,
            frame_index: 0,
            command_buffers,
            image_available_semaphores,
            in_flight_fences,
            render_finished_semaphores,
            pw_device: PwDevice::new(&vk_context.instance, &context.device()),
            present_id: 0,
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
            self.in_flight_fences.iter().for_each(|f| {
                device.destroy_fence(*f, None);
            });
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

            self.swapchain = swapchain;
            self.swapchain_images = images;
            self.swapchain_image_views = image_views;
        }
    }

    // XXX Needs to use self-pacing deadline timing stuff.
    /// Wait for the last queue submission to clear.
    pub fn draw_wait(&mut self, context: &DeviceContext) {
        let device = &context.device;
        let idx = self.frame_index as usize;
        let in_flight = self.in_flight_fences[idx];
        unsafe {
            device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .unwrap();

            device.reset_fences(&[in_flight]).unwrap();
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
        let in_flight = self.in_flight_fences[idx];
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

        let image_index = image_index as usize;
        let sync = DrawSync {
            image_available,
            in_flight,
            render_finished,
            image_index: image_index,
        };

        let command_buffer = self.command_buffers[image_index];
        let image = self.swapchain_images[image_index];
        let image_view = self.swapchain_image_views[image_index];

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
            value: 0,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            device_index: 0,
            ..Default::default()
        };

        let signal_info = vk::SemaphoreSubmitInfo {
            semaphore: sync.render_finished,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            device_index: 0,
            ..Default::default()
        };

        let cb_info = vk::CommandBufferSubmitInfo {
            command_buffer: target.command_buffer,
            device_mask: 0,
            ..Default::default()
        };

        let submit = vk::SubmitInfo2 {
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cb_info,
            ..Default::default()
        };

        let queue = &context.queues.graphics_queue();
        unsafe {
            context
                .device()
                .queue_submit2(*queue, &[submit], sync.in_flight)
                .unwrap();
        }

        let present_wait = [sync.render_finished];
        let swapchains = [self.swapchain];
        let indices = [sync.image_index as u32];

        let present_id = vk::PresentIdKHR {
            swapchain_count: 1,
            p_present_ids: [self.present_id].as_ptr(),
            ..Default::default()
        };
        self.present_id = self.present_id + 1;

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: present_wait.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: indices.as_ptr(),
            p_next: &present_id as *const _ as *const std::ffi::c_void,
            ..Default::default()
        };

        // XXX REMOVE to get rid of almost all window tangling
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

            if self.present_id % 2 == 1 {
                // let pre_wait = std::time::Instant::now();
                match self.pw_device.wait_for_present(
                    self.swapchain,
                    self.present_id - 1,
                    32_000_000, // 32 milliseconds
                ) {
                    Ok(_code) => {
                        // idk
                    }
                    Err(e) => match e {
                        vk::Result::TIMEOUT => {
                            // nothing
                        }
                        e => {
                            println!("present wait return code: {:?}", e);
                        }
                    },
                };
                self.present_id = self.present_id + 1;

                // let post_wait = std::time::Instant::now();
                // println!(
                //     "present wait: {:10.4}",
                //     (post_wait - pre_wait).as_micros() as f64 / 1000.0
                // );
            }
        }
    }
}
