// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Window
//!
//! We use the [`winit::window::Window`] without much indirection.  It is integrated with some
//! inputs and fullscreen behaviors that are specific to our application but not general enough to
//! warrant belonging in lib.  Using windows is a frontend behavior.  If multiple frontends use
//! windows, consider lifting this code into a shared module.

use ash::{khr::xlib_surface, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{event_loop::ActiveEventLoop, window::Window};

use mutate_lib::vulkan::context::VkContext;

use crate::Args;

pub trait WindowExt {
    fn from_args(args: &Args, event_loop: &ActiveEventLoop) -> Window;
    fn toggle_fullscreen(&self);
    fn surface(&self, vk_context: &VkContext) -> vk::SurfaceKHR;
}

impl WindowExt for Window {
    /// Create the window according to our sauce. 😋
    fn from_args(args: &Args, event_loop: &ActiveEventLoop) -> Window {
        let mut attrs = Window::default_attributes().with_title("µTate");
        if args.fullscreen {
            // LIES None is not correct here.  We should pick a window.  Maybe all windows.
            attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        }
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");

        if args.fullscreen {
            window.set_cursor_visible(false);
        }
        window
    }

    fn toggle_fullscreen(&self) {
        // LIES None is wrong if we know the monitor.  For multi-monitor fullscreen, we will need to
        // use that argument.
        match self.fullscreen() {
            Some(winit::window::Fullscreen::Borderless(None)) => {
                self.set_fullscreen(None);
                self.set_cursor_visible(true);
            }
            _ => {
                self.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                self.set_cursor_visible(false);
            }
        }
    }

    /// Use the Vulkan surface loader for this window's platform to return a surface ready for
    /// rendering.
    // DEBT platform specific, incomplete
    fn surface(&self, vk_context: &VkContext) -> vk::SurfaceKHR {
        let win_handle = self.window_handle().unwrap().as_raw();
        let display_handle = self.display_handle().unwrap().as_raw();

        match (win_handle, display_handle) {
            (RawWindowHandle::Xlib(win_handle), RawDisplayHandle::Xlib(display_handle)) => {
                let win_thing = win_handle.window;
                let xlib_create_info = vk::XlibSurfaceCreateInfoKHR {
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
                    display: display_handle.display.as_ptr(),
                    surface: win_handle.surface.as_ptr(),
                    ..Default::default()
                };

                let wayland_surface_loader = ash::khr::wayland_surface::Instance::new(
                    &vk_context.entry,
                    &vk_context.instance,
                );

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
}
