// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Window
//!
//! We use the [`winit::window::Window`] without much indirection.  It is integrated with some
//! inputs and fullscreen behaviors that are specific to our application but not general enough to
//! warrant belonging in lib.  Using windows is a frontend behavior.  If multiple frontends use
//! windows, consider lifting this code into a shared module.

use ash::{khr::xlib_surface, vk};
use winit::{event_loop::ActiveEventLoop, window::Window};

use mutate_lib::vulkan::context::VkContext;

use crate::Args;

pub trait WindowExt {
    fn from_args(args: &Args, event_loop: &ActiveEventLoop) -> Window;
    fn toggle_fullscreen(&self);
    fn render_size(&self) -> vk::Extent2D;
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

    fn render_size(&self) -> vk::Extent2D {
        let size = self.inner_size();
        vk::Extent2D {
            width: size.width,
            height: size.height,
        }
    }
}
