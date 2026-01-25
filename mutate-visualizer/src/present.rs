// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// A swapchain exists when we are presenting to a Surface.  We can use it as a render target.  Not
// all render targets need presentation, but a swapchain does.  Aligning the fields and structs with
// this abstraction is underway.

// NEXT enable options for variable and fixed frame rate rendering, likely using MAILBOX for
// variable and FIFO for fixed rate.  This requires looking at the actual swapchain image count
// before allocating per-image resources like command buffers.

use ash::{khr::present_wait::Device as PwDevice, khr::xlib_surface, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::vk_context::VkContext;
use crate::Args;

pub struct DrawSync {
    pub in_flight: vk::Fence,
    pub render_finished: vk::Semaphore,
    pub image_available: vk::Semaphore,
    pub image_index: usize,
}

pub struct DrawTarget {
    pub image: vk::Image,
    pub extent: vk::Extent2D,
    pub command_buffer: vk::CommandBuffer,
}

pub struct WindowPresent {
    pub frames: usize,
    pub frame_index: usize,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_loader: ash::khr::swapchain::Device,

    command_buffers: Vec<vk::CommandBuffer>,

    surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub window: Window,

    pub pw_device: PwDevice,
    /// Even values are usable ids.  Odd successors indicate waiting on that id has not been
    /// completed yet.
    pub present_id: u64,
}

impl WindowPresent {
    pub fn new(vk_context: &VkContext, event_loop: &ActiveEventLoop, args: &Args) -> Self {
        let mut attrs = Window::default_attributes().with_title("µTate");
        if args.fullscreen {
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        if args.fullscreen {
            window.set_cursor_visible(false);
        }
        let surface = window_surface(&window, &vk_context);

        let formats = unsafe {
            vk_context
                .surface_loader
                .get_physical_device_surface_formats(vk_context.physical_device, surface)
                .unwrap()
        };
        let surface_format = formats[0];

        let supported = unsafe {
            vk_context
                .surface_loader
                .get_physical_device_surface_support(
                    vk_context.physical_device,
                    vk_context.queue_family_index,
                    surface,
                )
                .unwrap()
        };
        assert!(supported, "Physical device must support this surface!");

        let surface_caps = unsafe {
            vk_context
                .surface_loader
                .get_physical_device_surface_capabilities(vk_context.physical_device, surface)
                .unwrap()
        };
        // TODO Watch for degenerate extents
        let extent = surface_caps.current_extent;

        let composite_alpha = pick_alpha(&surface_caps);

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &vk_context.device);
        let swapchain_info = vk::SwapchainCreateInfoKHR {
            surface: surface,
            min_image_count: 3,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_caps.current_transform,
            composite_alpha: composite_alpha,
            present_mode: vk::PresentModeKHR::FIFO_RELAXED,
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
                    format: surface_format.format,
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
                    vk_context
                        .device
                        .create_image_view(&view_info, None)
                        .unwrap()
                }
            })
            .collect();

        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let in_flight_fences: Vec<vk::Fence> = (0..3)
            .map(|_| unsafe { vk_context.device.create_fence(&fence_info, None).unwrap() })
            .collect();

        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        // FIXME propagate image counts
        let image_available_semaphores = (0..3)
            .map(|_| unsafe {
                vk_context
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let render_finished_semaphores = (0..3)
            .map(|_| unsafe {
                vk_context
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let command_pool = *vk_context.graphics_pool();

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 3,
            ..Default::default()
        };

        let command_buffers = unsafe {
            vk_context
                .device
                .allocate_command_buffers(&alloc_info)
                .unwrap()
        };

        Self {
            command_buffers,
            frames: 3,
            frame_index: 0,
            image_available_semaphores,
            in_flight_fences,
            render_finished_semaphores,
            swapchain_extent: extent,
            swapchain,
            swapchain_image_views: image_views,
            swapchain_images: images,
            swapchain_loader,

            surface,
            surface_format,
            window,

            pw_device: PwDevice::new(&vk_context.instance, &vk_context.device),
            present_id: 0,
        }
    }

    pub fn destroy(&self, context: &VkContext) {
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
            context.surface_loader.destroy_surface(self.surface, None);
        }
    }

    pub fn recreate_images(&mut self, vk_context: &VkContext) {
        let device = &vk_context.device;
        let physical_device = vk_context.physical_device;

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
            let surface_caps = vk_context
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
                .unwrap();
            let current_extent = surface_caps.current_extent;
            let extent = if current_extent.width != u32::MAX {
                self.swapchain_extent = current_extent;
                self.swapchain_extent
            } else {
                // FIXME a number of cases this can be wrong.
                window_size(&self.window)
            };
            let surface = &self.surface;
            let surface_format = &self.surface_format;

            let swapchain_info = vk::SwapchainCreateInfoKHR {
                surface: *surface,
                // NOTE this minimum is minimum.  At least under MAILBOX, extra images may result.
                min_image_count: self.frames as u32,
                image_format: surface_format.format,
                image_color_space: surface_format.color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: surface_caps.current_transform,
                composite_alpha: pick_alpha(&surface_caps),
                present_mode: vk::PresentModeKHR::FIFO_RELAXED,
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
                        format: surface_format.format,
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

    /// Wait for the last queue submission to clear.
    pub fn present_wait(&mut self, context: &VkContext) {
        let device = &context.device;
        let idx = self.frame_index as usize;
        let in_flight = self.in_flight_fences[idx];
        unsafe {
            device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .unwrap();

            device.reset_fences(&[in_flight]).unwrap();

            if self.present_id % 2 == 1 {
                match self.pw_device.wait_for_present(
                    self.swapchain,
                    self.present_id - 1,
                    12_000_000, // 12 milliseconds
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

            }
        }
    }

    /// Return a hot command buffer, image, and associated information used for drawing.
    pub fn render_target(
        &mut self,
        context: &VkContext,
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
        let extent = self.swapchain_extent;

        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();

            let begin = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(command_buffer, &begin).unwrap();
        }

        let barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier,
            ..Default::default()
        };

        unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        let color_attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: clear,
            ..Default::default()
        };

        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: extent,
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        unsafe { device.cmd_begin_rendering(command_buffer, &render_info) };

        let target = DrawTarget {
            image,
            command_buffer,
            extent,
        };

        (sync, target)
    }

    /// Sync presentation image transform and present
    pub fn post_draw(&mut self, vk_context: &VkContext, sync: DrawSync, target: DrawTarget) {
        let device = vk_context.device();
        unsafe { device.cmd_end_rendering(target.command_buffer) };
        let barrier2 = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
            dst_access_mask: vk::AccessFlags2::empty(),

            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,

            image: target.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let dep2 = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier2,
            ..Default::default()
        };

        unsafe {
            vk_context
                .device
                .cmd_pipeline_barrier2(target.command_buffer, &dep2)
        };

        unsafe {
            vk_context
                .device
                .end_command_buffer(target.command_buffer)
                .unwrap()
        };

        let wait_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: sync.image_available,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            device_index: 0,
            ..Default::default()
        };

        let signal_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: sync.render_finished,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            device_index: 0,
            ..Default::default()
        };

        let cb_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: target.command_buffer,
            device_mask: 0,
            ..Default::default()
        };

        let submit = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cb_info,
            ..Default::default()
        };

        let queue = vk_context.graphics_queue();
        unsafe {
            vk_context
                .device
                .queue_submit2(*queue, &[submit], sync.in_flight)
                .unwrap();
        }

        let present_wait = [sync.render_finished];
        let swapchains = [self.swapchain];
        let indices = [sync.image_index as u32];

        let present_id = vk::PresentIdKHR {
            s_type: vk::StructureType::PRESENT_ID_KHR,
            swapchain_count: 1,
            p_present_ids: [self.present_id].as_ptr(),
            ..Default::default()
        };
        self.present_id = self.present_id + 1;

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: present_wait.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: indices.as_ptr(),
            p_next: &present_id as *const _ as *const std::ffi::c_void,
            ..Default::default()
        };

        // Winit says this helps align the window system latching with REDRAW_REQUESTED events.
        // However, it is supported only on Wayland at this time.
        self.window.pre_present_notify();

        unsafe {
            match self.swapchain_loader.queue_present(*queue, &present_info) {
                Ok(_) => {
                    // MAYBE How to interpret false?
                }
                Err(result) => eprintln!("presentation error: {:?}", result),
            };
        }
    }

    pub fn toggle_fullscreen(&self) {
        let win = &self.window;
        match win.fullscreen() {
            Some(winit::window::Fullscreen::Borderless(None)) => {
                win.set_fullscreen(None);
                win.set_cursor_visible(true);
            }
            _ => {
                win.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                win.set_cursor_visible(false);
            }
        }
    }
}

fn pick_alpha(&surface_caps: &vk::SurfaceCapabilitiesKHR) -> vk::CompositeAlphaFlagsKHR {
    if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
    {
        vk::CompositeAlphaFlagsKHR::OPAQUE
    } else if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
    {
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
    } else if surface_caps
        .supported_composite_alpha
        .contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
    {
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
    } else {
        vk::CompositeAlphaFlagsKHR::INHERIT
    }
}

fn window_size(window: &Window) -> vk::Extent2D {
    let size = window.inner_size();
    vk::Extent2D {
        width: size.width,
        height: size.height,
    }
}

fn window_surface(window: &Window, vk_context: &VkContext) -> vk::SurfaceKHR {
    let win_handle = window.window_handle().unwrap().as_raw();
    let display_handle = window.display_handle().unwrap().as_raw();

    match (win_handle, display_handle) {
        (RawWindowHandle::Xlib(win_handle), RawDisplayHandle::Xlib(display_handle)) => {
            let win_thing = win_handle.window;
            let xlib_create_info = vk::XlibSurfaceCreateInfoKHR {
                s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
                window: win_thing.into(),
                dpy: display_handle.display.unwrap().as_ptr(),
                ..Default::default()
            };
            let xlib_surface_loader =
                xlib_surface::Instance::new(&vk_context.entry, &vk_context.instance);

            unsafe { xlib_surface_loader.create_xlib_surface(&xlib_create_info, None) }
                .expect("Failed to create surface")
        }
        (RawWindowHandle::Wayland(win_handle), RawDisplayHandle::Wayland(display_handle)) => {
            let wayland_create_info = vk::WaylandSurfaceCreateInfoKHR {
                s_type: vk::StructureType::WAYLAND_SURFACE_CREATE_INFO_KHR,
                display: display_handle.display.as_ptr(),
                surface: win_handle.surface.as_ptr(),
                ..Default::default()
            };

            let wayland_surface_loader =
                ash::khr::wayland_surface::Instance::new(&vk_context.entry, &vk_context.instance);

            unsafe {
                wayland_surface_loader
                    .create_wayland_surface(&wayland_create_info, None)
                    .expect("Failed to create Wayland surface")
            }
        }
        (_, _) => {
            panic!("Unsupported surface type!");
        }
    }
}
