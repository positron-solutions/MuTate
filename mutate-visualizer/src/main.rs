// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
mod assets;

use std::ffi::{c_void, CStr, CString};

use ash::khr::xlib_surface;
use ash::{vk, Entry};
use palette::convert::FromColorUnclamped;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use ringbuf::traits::*;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use mutate_lib as utate;

// This is a temporary extraction to support fullscreen.  A better solution will roll off old
// swapchain resources to phase in resizing image by image, creating these images asynchronously.
struct SwapChain {
    frames: usize,
    frame_index: usize,
    image_available_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    render_finished_semaphores: Vec<vk::Semaphore>,

    swapchain: vk::SwapchainKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_images: Vec<vk::Image>,
    swapchain_loader: ash::khr::swapchain::Device,
}

impl SwapChain {
    fn new(rb: &RenderBase, rt: &RenderTarget) -> Self {
        // &surface, &surface_caps, surface_format, swapchain_size
        let surface = &rt.surface;
        let surface_caps = &rt.surface_caps;
        let surface_format = &rt.surface_format;
        let extent = window_size(&rt.window);

        let composite_alpha = pick_alpha(&surface_caps);

        let swapchain_loader = ash::khr::swapchain::Device::new(&rb.instance, &rb.device);
        let swapchain_info = vk::SwapchainCreateInfoKHR {
            surface: *surface,
            min_image_count: 3, // XXX frame counts
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_caps.current_transform,
            composite_alpha: composite_alpha,
            present_mode: vk::PresentModeKHR::FIFO,
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
                unsafe { rb.device.create_image_view(&view_info, None).unwrap() }
            })
            .collect();

        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let in_flight_fences: Vec<vk::Fence> = (0..3)
            .map(|_| unsafe { rb.device.create_fence(&fence_info, None).unwrap() })
            .collect();

        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        // FIXME propagate image counts
        let image_available_semaphores = (0..3)
            .map(|_| unsafe { rb.device.create_semaphore(&semaphore_info, None).unwrap() })
            .collect();

        let render_finished_semaphores = (0..3)
            .map(|_| unsafe { rb.device.create_semaphore(&semaphore_info, None).unwrap() })
            .collect();

        Self {
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
        }
    }

    fn destroy(&self, device: &ash::Device) {
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

    fn recreate_images(&mut self, rb: &RenderBase, rt: &RenderTarget) {
        let device = &rb.device;
        let physical_device = rb.physical_device;

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
            let surface_caps = rt
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, rt.surface)
                .unwrap();
            let current_extent = surface_caps.current_extent;
            let extent = if current_extent.width != u32::MAX {
                self.swapchain_extent = current_extent;
                self.swapchain_extent
            } else {
                // FIXME a number of cases this can be wrong.
                window_size(&rt.window)
            };
            let surface = &rt.surface;
            let surface_format = &rt.surface_format;

            let swapchain_info = vk::SwapchainCreateInfoKHR {
                surface: *surface,
                min_image_count: self.frames as u32,
                image_format: surface_format.format,
                image_color_space: surface_format.color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: surface_caps.current_transform,
                composite_alpha: pick_alpha(&surface_caps),
                present_mode: vk::PresentModeKHR::FIFO,
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

    fn render_target(&self, index: usize) -> (vk::Image, vk::ImageView) {
        let image = self.swapchain_images[index];
        let view = self.swapchain_image_views[index];
        (image, view)
    }
}

struct RenderTarget {
    surface: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
    surface_format: vk::SurfaceFormatKHR,
    surface_caps: vk::SurfaceCapabilitiesKHR,

    window: Window,
}

impl RenderTarget {
    fn new(rb: &RenderBase, event_loop: &ActiveEventLoop) -> Self {
        let attrs = Window::default_attributes()
            .with_title("µTate")
            .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));

        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        window.set_cursor_visible(false);

        let win_handle = window.window_handle().unwrap().as_raw();
        let xlib_window_handle = match win_handle {
            RawWindowHandle::Xlib(handle) => handle,
            _ => panic!("Only Xlib supported!"),
        };
        let xlib_window = xlib_window_handle.window;

        let display_handle = window.display_handle().unwrap().as_raw();
        let xlib_display = match display_handle {
            RawDisplayHandle::Xlib(handle) => handle,
            _ => panic!("Only Xlib supported!"),
        };

        let xlib_create_info = vk::XlibSurfaceCreateInfoKHR {
            s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
            window: xlib_window.into(),
            dpy: xlib_display.display.unwrap().as_ptr(),
            ..Default::default()
        };

        let xlib_surface_loader = xlib_surface::Instance::new(&rb.entry, &rb.instance);

        let surface = unsafe { xlib_surface_loader.create_xlib_surface(&xlib_create_info, None) }
            .expect("Failed to create surface");

        let surface_loader = ash::khr::surface::Instance::new(&rb.entry, &rb.instance);

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(rb.physical_device, surface)
                .unwrap()
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(rb.physical_device, surface)
                .unwrap()
        };
        let surface_format = formats[0];

        let supported = unsafe {
            surface_loader
                .get_physical_device_surface_support(
                    rb.physical_device,
                    rb.queue_family_index,
                    surface,
                )
                .unwrap()
        };
        assert!(supported, "Physical device must support this surface!");

        Self {
            surface,
            surface_loader,
            surface_format: surface_format,
            surface_caps: surface_caps,

            window,
        }
    }

    fn destroy(&self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

struct RenderBase {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,

    graphics_queue: vk::Queue,
    #[allow(dead_code)]
    compute_queue: vk::Queue,
    #[allow(dead_code)]
    transfer_queue: vk::Queue,

    queue_family_index: u32,
}

impl RenderBase {
    fn new() -> Self {
        let entry = unsafe { Entry::load().expect("failed to load Vulkan library") };
        let available_exts = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")
        };

        // FIXME insufficiently accurate platform check
        assert!(
            available_exts.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == ash::vk::KHR_WAYLAND_SURFACE_NAME
            }),
            "Only xlib is currently supported"
        );

        let required_exts = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),
            ash::vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
            // NEXT CLI switch gate
            ash::vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        let validation_layers = [VALIDATION_LAYER.as_ptr()];

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: required_exts.len() as u32,
            pp_enabled_extension_names: required_exts.as_ptr(),
            enabled_layer_count: validation_layers.len() as u32,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("No Vulkan devices")
        };
        let physical_device = physical_devices[0];

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find_map(|(index, q)| {
                    if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        Some(index as u32)
                    } else {
                        None
                    }
                })
                .expect("No graphics queue family found")
        };

        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let device_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            ash::vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
            ash::vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),
            ash::vk::KHR_DYNAMIC_RENDERING_NAME.as_ptr(),
            ash::vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_BUFFER_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_INDEXING_NAME.as_ptr(),
            ash::vk::KHR_PIPELINE_LIBRARY_NAME.as_ptr(),
            ash::vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr(),
            // ROLL holding off on this until other hardware vendors have supporting drivers
            // ash::vk::EXT_SHADER_OBJECT_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE1_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE2_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE3_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE4_NAME.as_ptr(),
        ];

        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default();
        sync2_features.synchronization2 = vk::TRUE;

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        dynamic_rendering_features.dynamic_rendering = vk::TRUE;

        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        features2.p_next = &mut sync2_features as *mut _ as *mut c_void;

        sync2_features.p_next = &mut dynamic_rendering_features as *mut _ as *mut c_void;

        let mut device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: queue_info.as_ptr(),
            pp_enabled_extension_names: device_extensions.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            ..Default::default()
        };
        device_info.p_next = &mut features2 as *mut _ as *mut c_void;

        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Self {
            entry,
            instance,
            physical_device,
            device,

            graphics_queue: queue.clone(),
            compute_queue: queue.clone(),
            transfer_queue: queue,

            queue_family_index,
        }
    }

    fn graphics_queue(&self) -> &vk::Queue {
        &self.graphics_queue
    }

    fn device(&self) -> &ash::Device {
        &self.device
    }

    // XXX this consumes the base in reality...
    fn destroy(&self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None)
        };
    }
}

struct App {
    running: bool,

    render_base: Option<RenderBase>,
    render_target: Option<RenderTarget>,
    swapchain: Option<SwapChain>,

    command_buffers: Vec<vk::CommandBuffer>,
    command_pool: Option<vk::CommandPool>,
    pipeline_layout: Option<vk::PipelineLayout>,
    pipelines: Option<Vec<vk::Pipeline>>,

    audio_events: ringbuf::wrap::caching::Caching<
        std::sync::Arc<ringbuf::SharedRb<ringbuf::storage::Heap<(f32, f32, f32, f32)>>>,
        false,
        true,
    >,
    hue: f32,
    value: f32,
}

impl App {
    fn draw_frame(&mut self) {
        let render_base = self.render_base.as_ref().unwrap();
        let device = &render_base.device;
        let queue = render_base.graphics_queue();

        let rt = self.render_target.as_ref().unwrap();
        let sc = self.swapchain.as_ref().unwrap();

        if self.audio_events.is_full() {
            eprintln!("audio event backpressure drop");
            self.audio_events.skip(1);
        }
        let (_slow, fast) = match self.audio_events.try_pop() {
            Some(got) => ((got.0 + got.1), (got.2 + got.3)),
            None => {
                eprintln!("No audio event was ready");
                (0.1, 0.1)
            }
        };

        self.value = fast;

        self.hue += 0.002 * fast;
        if self.hue > 1.0 || self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        } else if self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        }

        let idx = sc.frame_index;
        let image_available = sc.image_available_semaphores[idx];
        let render_finished = sc.render_finished_semaphores[idx];
        let in_flight = sc.in_flight_fences[idx];

        // While there is a lot of silly code, this is much more silly.
        let sc = self.swapchain.as_mut().unwrap();
        sc.frame_index = (sc.frame_index + 1) % sc.frames;
        let sc = self.swapchain.as_ref().unwrap();

        unsafe {
            device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("wait_for_fences failed");

            device
                .reset_fences(&[in_flight])
                .expect("reset_fences failed");
        }

        let (image_index, _) = unsafe {
            sc.swapchain_loader
                .acquire_next_image(
                    sc.swapchain,
                    std::u64::MAX,
                    image_available,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image")
        };

        let render_target = sc.render_target(image_index as usize);

        // Wait for the image-available semaphore before executing commands.
        let wait_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: image_available,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            device_index: 0,
            ..Default::default()
        };

        // Signal when rendering is done.
        let signal_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: render_finished,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            device_index: 0,
            ..Default::default()
        };

        let rt = self.render_target.as_ref().unwrap();

        let command_buffer = self.command_buffers[image_index as usize];

        self.record_command_buffer(render_target, &sc.swapchain_extent, command_buffer);

        // Which command buffer to submit.
        let cmd_buffer = self.command_buffers[image_index as usize];

        let cmd_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: cmd_buffer,
            device_mask: 0,
            ..Default::default()
        };

        // Submit struct (synchronization2)
        let submit = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cmd_info,
            ..Default::default()
        };

        unsafe {
            device
                .queue_submit2(*queue, &[submit], in_flight)
                .expect("queue_submit2 failed");
        }

        let present_wait = [render_finished];
        let swapchains = [sc.swapchain];
        let indices = [image_index];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: present_wait.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: indices.as_ptr(),
            ..Default::default()
        };

        unsafe {
            match sc.swapchain_loader.queue_present(*queue, &present_info) {
                Ok(_) => {
                    // MAYBE How to interpret false?
                }
                Err(result) => eprintln!("presentation error: {:?}", result),
            };
        }
        rt.window.request_redraw();
    }

    fn record_command_buffer(
        &self,
        render_target: (vk::Image, vk::ImageView),
        extent: &vk::Extent2D,
        cb: vk::CommandBuffer,
    ) {
        let render_base = self.render_base.as_ref().unwrap();
        let device = &render_base.device;

        unsafe {
            device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .expect("reset_cb failed");

            let begin = vk::CommandBufferBeginInfo::default();
            device
                .begin_command_buffer(cb, &begin)
                .expect("begin failed");
        }

        let barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image: render_target.0,
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

        unsafe { device.cmd_pipeline_barrier2(cb, &dep_info) };
        // XXX Command barrier reset leaking into draw

        let tweaked = self.value * 0.02 + 0.3;
        let value = tweaked.clamp(0.0, 1.0);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(self.hue * 360.0, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [rgb.red, rgb.green, rgb.blue, 1.0],
            },
        };

        let color_attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: render_target.1,
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
                extent: *extent,
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        unsafe { device.cmd_begin_rendering(cb, &render_info) };

        let pipeline = self.pipelines.as_ref().unwrap()[0];
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        // Update the color used in the fragment shader
        let mut trie_hue = self.hue * 360.0 + 180.0;
        if trie_hue > 360.0 {
            trie_hue -= 360.0;
        }
        let scale = 0.8 + (0.2 * self.value);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(trie_hue, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);
        let combined_push: [f32; 5] = [rgb.red, rgb.green, rgb.blue, 1.0, scale];
        let pipeline_layout = self.pipeline_layout.as_ref().unwrap();
        unsafe {
            device.cmd_push_constants(
                cb,
                *pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    combined_push.as_ptr() as *const u8,
                    std::mem::size_of::<[f32; 5]>(),
                ),
            );
        }

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: *extent,
        };

        unsafe {
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[scissor]);
        }

        unsafe { device.cmd_draw(cb, 3, 1, 0, 0) };

        unsafe { device.cmd_end_rendering(cb) };

        // XXX presentation details leaking into draw
        let barrier2 = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags2::empty(),
            image: render_target.0,
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

        unsafe { device.cmd_pipeline_barrier2(cb, &dep2) };

        unsafe { device.end_command_buffer(cb).expect("end_cb failed") };
    }
}

static VALIDATION_LAYER: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let rb = RenderBase::new();
        let rt = RenderTarget::new(&rb, event_loop);
        let sc = SwapChain::new(&rb, &rt);

        let queue_family_index = rb.queue_family_index;
        let device = &rb.device;

        // MAYBE getting command pools requires the queue index
        let command_pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        };

        // allocate one command buffer per swapchain image
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: sc.swapchain_images.len() as u32,
            ..Default::default()
        };

        let buffers = unsafe { device.allocate_command_buffers(&alloc_info).unwrap() };

        let assets = assets::AssetDirs::new();
        let vert_spv = assets
            .find_bytes("vertex", assets::AssetKind::Shader)
            .unwrap();
        let frag_spv = assets
            .find_bytes("fragment", assets::AssetKind::Shader)
            .unwrap();

        let vert_module_ci = vk::ShaderModuleCreateInfo {
            code_size: vert_spv.len(),
            p_code: vert_spv.as_ptr() as *const u32,
            ..Default::default()
        };

        let frag_module_ci = vk::ShaderModuleCreateInfo {
            code_size: frag_spv.len(),
            p_code: frag_spv.as_ptr() as *const u32,
            ..Default::default()
        };

        let vert_shader_module =
            unsafe { device.create_shader_module(&vert_module_ci, None).unwrap() };
        let frag_shader_module =
            unsafe { device.create_shader_module(&frag_module_ci, None).unwrap() };

        // Static
        let entry_vert = CString::new("main").unwrap();
        let entry_frag = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_shader_module,
                p_name: entry_vert.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_shader_module,
                p_name: entry_frag.as_ptr(),
                ..Default::default()
            },
        ];

        // I messed around with two ranges but the validation of the shader code made me believe
        // that using two separate push constants was at best hacky.  Therefore, I merged the ranges
        // and just used the relevant data in each shader.
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<[f32; 5]>() as u32,
        };

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_attribute_description_count: 0,
            vertex_binding_description_count: 0,
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        };

        let color_blend = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let color_formats = [rt.surface_format.format];
        let pipeline_rendering_info = vk::PipelineRenderingCreateInfo {
            s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachment_formats: color_formats.as_ptr(),
            ..Default::default()
        };

        let pipeline_ci = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: &pipeline_rendering_info as *const _ as *const std::ffi::c_void,
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass: vk::RenderPass::null(), // dynamic rendering
            subpass: 0,
            ..Default::default()
        };

        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
        }
        .unwrap();

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        self.command_pool = Some(command_pool);
        self.command_buffers = buffers;
        self.pipelines = Some(pipelines);
        self.pipeline_layout = Some(pipeline_layout);

        self.render_target = Some(rt);
        self.render_base = Some(rb);
        self.swapchain = Some(sc);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    println!("window resize reported degenerate size");
                } else {
                    let rb = self.render_base.as_ref().unwrap();
                    let rt = self.render_target.as_ref().unwrap();
                    self.swapchain.as_mut().unwrap().recreate_images(&rb, &rt);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.running {
                    self.draw_frame();
                }
            }
            WindowEvent::CloseRequested => unsafe {
                self.running = false;
                let render_base = self.render_base.as_ref().unwrap();
                let device = &render_base.device();

                device.device_wait_idle().unwrap();

                self.pipelines.as_ref().unwrap().iter().for_each(|p| {
                    device.destroy_pipeline(*p, None);
                });

                if let Some(layout) = self.pipeline_layout {
                    device.destroy_pipeline_layout(layout, None);
                }
                self.command_pool.map(|p| {
                    device.destroy_command_pool(p, None);
                });

                self.swapchain.as_ref().unwrap().destroy(&device);
                self.render_target.as_ref().unwrap().destroy();
                render_base.destroy();

                event_loop.exit();
            },
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}

fn main() -> Result<(), utate::MutateError> {
    let context = utate::AudioContext::new()?;
    println!("Choose the audio source:");

    let mut first_choices = Vec::new();
    let check = |choices: &[utate::AudioChoice]| {
        first_choices.extend_from_slice(choices);
    };

    context.with_choices_blocking(check).unwrap();
    first_choices.iter().enumerate().for_each(|(i, c)| {
        println!("[{}] {} AudioChoice: {:?}", i, c.id(), c.name());
    });

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let choice_idx = input.trim().parse().unwrap();
    let choice = first_choices.remove(choice_idx);

    let rx = context.connect(&choice, "mutate".to_owned()).unwrap();

    // audio events, processed results of the buffer, using an independent ring to provide some
    // buffering, synchronized communication, and back pressure support.
    let ae_ring = ringbuf::HeapRb::new(3);
    let (mut ae_tx, ae_rx) = ae_ring.split();

    let audio_thread = std::thread::spawn(move || {
        // This thread continuously emits events.  The scheme is a sliding window with a 120Hz width
        // and sliding in 240Hz increments.  The production of events is faster than the frame rate,
        // and balanced back pressure is accomplished by looking at the ring buffer size.

        // To subtract the noise floor, we track the moving average with a 240 sample exponential
        // moving average.
        let mut window_buffer = [0u8; 3200];
        let window_size = 3200; // one 240FPS frame at 48kHz and 8 bytes per frame
        let read_behind = 3200; // one frame of read-behind
        let mut left_max = 0f32;
        let mut right_max = 0f32;
        let mut left_noise = 0f32;
        let mut right_noise = 0f32;

        let alpha = 2.0 / (240.0 + 1.0);
        let alpha_resid = 1.0 - alpha;

        let mut left_fast_accum = 0f32;
        let mut right_fast_accum = 0f32;
        let mut left_fast = 0f32;
        let mut right_fast = 0f32;
        let alpha_f = 2.0 / (8.0 + 1.0);
        let alpha_f_resid = 1.0 - alpha_f;

        // FIXME Ah yes, the user friendly API for real Gs
        let mut conn = std::mem::ManuallyDrop::new(unsafe { Box::from_raw(rx.conn) });

        while ae_tx.read_is_held() {
            let avail = conn.buffer.occupied_len();
            if avail >= window_size {
                let read = conn.buffer.peek_slice(&mut window_buffer);
                assert!(read == window_size);

                // Estimate the energy by absolute delta.  IIRC not only is this physically wrong
                // but also doesn't map to perceptual very well.
                let (mut last_l, mut last_r) = (0.0, 0.0);
                let (left_sum, right_sum) = window_buffer
                    .chunks_exact(8) // 2 samples per frame × 4 bytes = 8 bytes per frame
                    .map(|frame| {
                        let left = f32::from_le_bytes(frame[0..4].try_into().unwrap());
                        let right = f32::from_le_bytes(frame[4..8].try_into().unwrap());
                        (left, right)
                    })
                    .fold((0f32, 0f32), |(acc_l, acc_r), (l, r)| {
                        // absolute delta + absolute amplitude
                        let accum = (
                            acc_l + (l - last_l).abs() + l.abs(),
                            acc_r + (r - last_r).abs() + r.abs(),
                        );
                        last_l = l;
                        last_r = r;
                        accum
                    });

                left_noise = (alpha * left_sum) + (alpha_resid * left_noise);
                right_noise = (alpha * right_sum) + (alpha_resid * right_noise);

                // Cut noise and normalize remaining to noise
                let left_excess = (left_sum - (left_noise * 1.3)) / left_noise.max(0.000001);
                let right_excess = (right_sum - (right_noise * 1.3)) / right_noise.max(0.000001);

                // Fast EMA of the cleaned signal for beats
                left_fast = (alpha_f * left_excess) + (alpha_f_resid * left_fast);
                right_fast = (alpha_f * right_excess) + (alpha_f_resid * right_fast);

                // Instantaneous response on climb
                if left_fast < left_excess {
                    left_fast = left_excess;
                }
                if right_fast < right_excess {
                    right_fast = right_excess;
                }

                left_fast_accum = left_fast + left_fast_accum;
                right_fast_accum = right_fast + right_fast_accum;

                left_max = left_max.max(left_excess);
                right_max = right_max.max(right_excess);

                // Backoff using queue size
                if ae_tx.vacant_len() > 1 {
                    match ae_tx.try_push((left_max, right_max, left_fast_accum, right_fast_accum)) {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("sending audio event failed: {:?}", e);
                            if ae_tx.is_full() {
                                eprintln!("audio event consumer is falling behind");
                            }
                        }
                    }
                    left_max = 0.0;
                    right_max = 0.0;
                    left_fast_accum = 0.0;
                    right_fast_accum = 0.0;
                }

                if avail >= (window_size * 2) + read_behind {
                    conn.buffer.skip(window_size / 2 + 200); // LIES +200 🤔
                }

                std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 240.0));
            } else {
                // Underfed, either we can pad with "empty" data or wait for new data.  Let's wait.
                match rx.wait() {
                    Ok(_) => {
                        eprintln!("audio buffered ⏰");
                    }
                    Err(e) => {
                        eprintln!("listening aborted: {}", e);
                        break;
                    }
                }
            }
        }
    });

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        running: true, // XXX

        render_base: None,
        render_target: None,
        swapchain: None,

        command_buffers: Vec::new(),
        command_pool: None,
        pipeline_layout: None,
        pipelines: None,

        audio_events: ae_rx,
        hue: rand::random::<f32>(),
        value: 0.0,
    };
    event_loop.run_app(&mut app).unwrap();
    drop(app);
    audio_thread.join().unwrap();
    Ok(())
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
