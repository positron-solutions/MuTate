// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # MuTate Minimal
//!
//! See the README.  See the MuTate Visualizer's main module for details on the lifecycle.

mod draw;

use std::collections::HashMap;

use ash::vk;
use mutate_lib::{self as utate, prelude::*};
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
    surface: Surface,
    present_ring: PresentRing,
    renderer: HelloDraw,
}

impl WindowContext {
    fn new(
        instance: &Instance,
        device: &mut Device,
        window: Window,
        raw_surface: vk::SurfaceKHR,
    ) -> Self {
        let surface = Surface::new(instance, device, raw_surface, &window).unwrap();
        let compute_present =
            PresentRing::new(device, instance, &surface, surface.extent()).unwrap();
        let mut renderer = HelloDraw::new(device);
        renderer.provision(device, surface.extent()).unwrap();
        Self {
            window,
            surface,
            present_ring: compute_present,
            renderer,
        }
    }

    /// Use the compute present ring
    fn draw_frame(&mut self, device: &mut Device) {
        // XXX the window presentation notification is a tension we would like to take away.
        self.present_ring
            .record(
                device,
                compute_present(device, |device, cb, acquired_image| {
                    self.renderer.draw(device, cb, acquired_image);
                }),
                || self.window.pre_present_notify(),
            )
            .map_err(|e| match e {
                utate::vulkan::VulkanError::SwapchainOutOfDate
                | utate::vulkan::VulkanError::SwapchainSuboptimal => {
                    self.handle_resize(device);
                }
                _ => {
                    eprintln!("application: draw failed {:?}", e);
                }
            });
    }

    // XXX make this into a try_resize method that can propagate recreation back up to the
    // application for device re-creation.
    fn handle_resize(&mut self, device: &mut Device) -> Result<(), MutateError> {
        let new_size = self
            .present_ring
            .maybe_update_swapchain(device, &mut self.surface, &self.window)
            .unwrap();
        self.renderer.provision(device, new_size)?;
        self.window.request_redraw();
        Ok(())
    }

    fn destroy(self, device: &mut Device) {
        self.renderer.destroy(device);
        self.present_ring.destroy(device);
        self.surface.destroy();
    }
}

/// The resumed state of the application, represented as the `Active` `AppState` variant.
struct ActiveApp {
    device: Box<Device>,
    // NOTE the hash map treatment is not minimal, instead going in the direction of multiple window
    // support.  For a single demo application a simpler field would suffice.
    windows: HashMap<WindowId, WindowContext>,
}

impl ActiveApp {
    fn new(instance: &Instance, event_loop: &ActiveEventLoop) -> Self {
        let window = make_window(event_loop);
        let raw_surface = instance.surface(event_loop, &window);

        // Surfaces might only be supported on some devices.  This check ensures that we will be
        // able to use the chosen device for this window.
        let supported: Vec<SupportedDevice> = instance
            .supported_devices(&[])
            .into_iter()
            .filter(|sd| sd.supports_surface(raw_surface, instance))
            .collect();
        assert!(
            !supported.is_empty(),
            "ActiveApp::new: no device supports the surface"
        );

        let selected = supported[0].clone();
        println!("device: {}", selected.name);
        let mut device = Box::new(selected.into_logical(instance));

        // Once we have chosen the physical device, then we can create the logical device and
        // swapchain for the window & surface.
        let wc = WindowContext::new(instance, &mut device, window, raw_surface);
        let window_id = wc.window.id();
        let mut windows = HashMap::new();
        windows.insert(window_id, wc);

        Self { device, windows }
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
                    wc.draw_frame(&mut self.device);
                    wc.window.request_redraw();
                }
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(wc) = self.windows.get_mut(&window_id) {
                    wc.handle_resize(&mut self.device).unwrap();
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
                    self.device.wait_idle().unwrap();
                    wc.destroy(&mut self.device);
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

/// Longest lived structure, holds the `Instance`.
struct MinimalApp {
    instance: Instance,
    state: AppState,
}

impl ApplicationHandler for MinimalApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.state = AppState::Active(ActiveApp::new(&self.instance, event_loop));
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
        active.device.wait_idle().unwrap();
        for (_, wc) in active.windows.drain() {
            wc.destroy(&mut active.device);
        }
        active.device.destroy();
    }
}

fn main() -> Result<(), utate::MutateError> {
    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = MinimalApp {
        instance: Instance::with_display(&event_loop, &[]),
        state: AppState::Dormant,
    };

    event_loop.run_app(&mut app).unwrap();
    app.instance.destroy();
    Ok(())
}
