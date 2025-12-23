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
    pub surface_format: vk::SurfaceFormatKHR,
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

        Self {
            surface,
            surface_format: surface_format,
            window,
        }
    }

    pub fn destroy(&self, vk_context: &VkContext) {
        unsafe {
            vk_context
                .surface_loader
                .destroy_surface(self.surface, None);
        }
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
