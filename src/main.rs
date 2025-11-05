// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::CStr;

use ash::khr::xlib_surface;
use ash::{vk, Entry};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

struct App {
    window: Option<Window>,

    entry: Option<ash::Entry>,
    instance: Option<ash::Instance>,
    surface_loader: Option<ash::khr::surface::Instance>,
    surface: Option<vk::SurfaceKHR>,
    physical_device: Option<vk::PhysicalDevice>,
    device: Option<ash::Device>,
    queue: Option<vk::Queue>,
    queue_family_index: u32,

    swapchain_loader: Option<ash::khr::swapchain::Device>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create a window using default attributes
        let attrs = Window::default_attributes().with_title("µTate"); // customizing attribute
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");
        let entry = unsafe { Entry::load().expect("failed to load Vulkan library") };
        let available_exts = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")
        };

        assert!(
            available_exts.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == ash::vk::KHR_XLIB_SURFACE_NAME
            }),
            "Only xlib is currently supported"
        );

        let required_exts = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),
            ash::vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
        ];

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: required_exts.len() as u32,
            pp_enabled_extension_names: required_exts.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
        let xlib_surface_loader = xlib_surface::Instance::new(&entry, &instance);
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

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

        let surface = unsafe { xlib_surface_loader.create_xlib_surface(&xlib_create_info, None) }
            .expect("Failed to create surface");

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

        let device_extensions = [ash::vk::KHR_SWAPCHAIN_NAME.as_ptr()];

        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: queue_info.as_ptr(),
            pp_enabled_extension_names: device_extensions.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            ..Default::default()
        };

        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let surface_format = formats[0];

        let composite_alpha = if surface_caps
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
        };

        let supported = unsafe {
            surface_loader
                .get_physical_device_surface_support(physical_device, queue_family_index, surface)
                .unwrap()
        };
        assert!(supported, "Physical device must support this surface!");

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: 2, // double buffered
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: surface_caps.current_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::CONCURRENT,
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
                unsafe { device.create_image_view(&view_info, None).unwrap() }
            })
            .collect();

        // Store all
        self.entry = Some(entry);
        self.instance = Some(instance);
        self.surface_loader = Some(surface_loader);
        self.surface = Some(surface);
        self.physical_device = Some(physical_device);
        self.device = Some(device);
        self.queue = Some(queue);
        self.queue_family_index = queue_family_index;
        self.window = Some(window);

        self.swapchain_loader = Some(swapchain_loader);
        self.swapchain = Some(swapchain);
        self.swapchain_images = images;
        self.swapchain_image_views = image_views;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let WindowEvent::CloseRequested = event {
            unsafe {
                if let Some(device) = &self.device {
                    for view in &self.swapchain_image_views {
                        device.destroy_image_view(*view, None);
                    }
                }
                if let Some(loader) = &self.swapchain_loader {
                    if let Some(swapchain) = self.swapchain {
                        loader.destroy_swapchain(swapchain, None);
                    }
                }
                if let Some(surface_loader) = &self.surface_loader {
                    if let Some(surface) = self.surface {
                        surface_loader.destroy_surface(surface, None);
                    }
                }
                if let Some(device) = &self.device {
                    device.destroy_device(None);
                }
                if let Some(instance) = &self.instance {
                    instance.destroy_instance(None);
                }
            }
            event_loop.exit();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        entry: None,
        instance: None,
        surface_loader: None,
        surface: None,
        physical_device: None,
        device: None,
        queue: None,
        queue_family_index: 0,
        swapchain_loader: None,
        swapchain: None,
        swapchain_images: Vec::new(),
        swapchain_image_views: Vec::new(),
    };
    event_loop.run_app(&mut app).unwrap();
}
