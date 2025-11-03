// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use ash::{vk, Entry};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create a window using default attributes
        let attrs = Window::default_attributes().with_title("µTate"); // customizing attribute
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");
        self.window = Some(window);

        // Loading and getting an instance is just verifying that we have accessed the libs provided
        // in the shell.  The lifecycle for Android etc is all down the road.
        let entry = unsafe { Entry::load().unwrap() };
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        unsafe {
            let instance = entry.create_instance(&create_info, None).unwrap();
            instance.destroy_instance(None);
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let WindowEvent::CloseRequested = event {
            // We would clean up here if we had held onto the Instance

            event_loop.exit();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { window: None };
    event_loop.run_app(&mut app).unwrap();
}
