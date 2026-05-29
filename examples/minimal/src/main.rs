// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # MuTate Minimal
//!
//! See the README.  See the MuTate Visualizer's main module for details on the lifecycle.

mod draw;

use std::collections::HashMap;

use ash::vk;
use mutate_lib::{self as utate, prelude::*, vulkan::present::ComputePresent};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard as kb,
    window::{Window, WindowId},
};

use draw::HelloDraw;

/// Lives alongside each window.
struct WindowContext {
    window: Window,
    surface: VkSurface,
    compute_present: ComputePresent,
    renderer: HelloDraw,
}

impl WindowContext {
    fn new(
        vk_context: &VkContext,
        device_context: &mut DeviceContext,
        window: Window,
        raw_surface: vk::SurfaceKHR,
    ) -> Self {
        let surface = VkSurface::new(raw_surface, vk_context, device_context);
        let extent = surface.resolve_size(device_context, &window).unwrap();

        let compute_present = ComputePresent::new(device_context, vk_context, &surface, extent);
        let mut renderer = HelloDraw::new(
            device_context,
            // XXX Go support Float4 in slang module
            [0.0, 0.8, 0.1, 1.0], // BGRA
        );
        renderer.provision(device_context, extent).unwrap();
        Self {
            window,
            surface,
            compute_present,
            renderer,
        }
    }

    /// Draw using the
    fn draw_frame(&mut self, device_context: &DeviceContext) {
        let extent = {
            let size = self.window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };
        self.compute_present.draw(
            device_context,
            |cb, acquired_image| {
                self.renderer
                    .draw(cb, acquired_image, device_context, extent);
            },
            || self.window.pre_present_notify(),
        );
    }

    fn handle_resize(&mut self, device_context: &DeviceContext) {
        if let Ok(size) = { self.surface.resolve_size(&device_context, &self.window) } {
            self.compute_present
                .recreate_images(&self.surface, size, device_context);
        }
    }

    fn destroy(self, device_context: &mut DeviceContext) {
        self.renderer.destroy(device_context);
        self.compute_present.destroy(device_context);
        self.surface.destroy();
    }
}

/// The resumed state of the application, represented as the `Active` `AppState` variant.
struct ActiveApp {
    device_context: DeviceContext,
    // NOTE the hash map treatment is not minimal, instead going in the direction of multiple window
    // support.  For a single demo application a simpler field would suffice.
    windows: HashMap<WindowId, WindowContext>,
}

impl ActiveApp {
    fn new(vk_context: &VkContext, event_loop: &ActiveEventLoop) -> Self {
        let window = make_window(event_loop);
        let raw_surface = vk_context.surface(&window, event_loop);

        // Surfaces might only be supported on some devices.  This check ensures that we will be
        // able to use the chosen device for this window.
        let supported: Vec<SupportedDevice> = vk_context
            .supported_devices(&[])
            .into_iter()
            .filter(|sd| sd.supports_surface(raw_surface, vk_context))
            .collect();
        assert!(
            !supported.is_empty(),
            "ActiveApp::new: no device supports the surface"
        );

        let selected = supported[0].clone();
        println!("device: {}", selected.name);
        let mut device_context = selected.into_logical(vk_context);

        // Once we have chosen the physical device, then we can create the logical device and
        // swapchain for the window & surface.
        let wc = WindowContext::new(vk_context, &mut device_context, window, raw_surface);
        let window_id = wc.window.id();
        let mut windows = HashMap::new();
        windows.insert(window_id, wc);

        Self {
            device_context,
            windows,
        }
    }

    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                if let Some(wc) = self.windows.get_mut(&window_id) {
                    wc.draw_frame(&self.device_context);
                    wc.window.request_redraw();
                }
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(wc) = self.windows.get_mut(&window_id) {
                    wc.handle_resize(&self.device_context);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !event.repeat && event.state == winit::event::ElementState::Pressed {
                    match event.physical_key {
                        kb::PhysicalKey::Code(kb::KeyCode::KeyQ)
                        | kb::PhysicalKey::Code(kb::KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::CloseRequested => {
                if let Some(wc) = self.windows.remove(&window_id) {
                    unsafe { self.device_context.device.device_wait_idle().unwrap() };
                    wc.destroy(&mut self.device_context);
                }
                if self.windows.is_empty() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }
}

// NOTE see the visualizers WindowExt for how to enable full screen support.
fn make_window(event_loop: &ActiveEventLoop) -> Window {
    let mut attrs = Window::default_attributes().with_title("µTate Minimal!");
    event_loop
        .create_window(attrs)
        .expect("Failed to create window")
}

/// An Enum is used to represent resume and suspend with a single type on the `MinimalApp`.
enum AppState {
    Dormant,
    Active(ActiveApp),
}

/// Longest lived structure, holds the instance (`VkContext`, some renaming is planned).
struct MinimalApp {
    vk_context: VkContext,
    state: AppState,
}

impl ApplicationHandler for MinimalApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.state = AppState::Active(ActiveApp::new(&self.vk_context, event_loop));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let AppState::Active(active) = &mut self.state else {
            return;
        };
        active.handle_window_event(event_loop, window_id, event);
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let AppState::Active(active) = &mut self.state else {
            return;
        };
        active.device_context.wait_idle().unwrap();
        for (_, wc) in active.windows.drain() {
            wc.destroy(&mut active.device_context);
        }
        active.device_context.destroy();
    }
}

fn main() -> Result<(), utate::MutateError> {
    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = MinimalApp {
        vk_context: VkContext::with_display(&event_loop, &[]),
        state: AppState::Dormant,
    };

    event_loop.run_app(&mut app).unwrap();
    app.vk_context.destroy();
    Ok(())
}
