// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use ash::khr::xlib_surface;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{event_loop::ActiveEventLoop, window::Window};

use crate::vk_context::VkContext;
use crate::Args;

pub struct RenderTarget {
    pub surface: vk::SurfaceKHR,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface_format: vk::SurfaceFormatKHR,
    pub surface_caps: vk::SurfaceCapabilitiesKHR,

    pub window: Window,
}

impl RenderTarget {
    pub fn new(vk_context: &VkContext, event_loop: &ActiveEventLoop, args: &Args) -> Self {
        let mut attrs = Window::default_attributes().with_title("µTate");

        if args.fullscreen {
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        }

        // XXX extract this and just give a window
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

        let xlib_surface_loader =
            xlib_surface::Instance::new(&vk_context.entry, &vk_context.instance);

        let surface = unsafe { xlib_surface_loader.create_xlib_surface(&xlib_create_info, None) }
            .expect("Failed to create surface");

        let surface_loader =
            ash::khr::surface::Instance::new(&vk_context.entry, &vk_context.instance);

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(vk_context.physical_device, surface)
                .unwrap()
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(vk_context.physical_device, surface)
                .unwrap()
        };
        let surface_format = formats[0];

        let supported = unsafe {
            surface_loader
                .get_physical_device_surface_support(
                    vk_context.physical_device,
                    vk_context.queue_family_index,
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

    pub fn destroy(&self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
