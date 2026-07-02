// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # MuTate Minimal
//!
//! See the README.  See the MuTate Visualizer's main module for details on the lifecycle.

mod draw;

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

/// Event loop messages are sent into the render thread, which self-paces rather than waiting on
/// [`WindowEvent::RedrawRequested`] events.
enum RenderMsg {
    Resized(winit::dpi::PhysicalSize<u32>),
    Shutdown,
}

struct WindowContext {
    handle: Option<std::thread::JoinHandle<()>>,
    tx: std::sync::mpsc::Sender<RenderMsg>,
}

impl WindowContext {
    /// The thread body initializes the renderer in scope.  The rest of the body self-paces the
    /// render loop, accepting messages from the event loop and executing draw or updating resources
    /// as necessary.
    fn spawn(
        instance: &'static Instance,
        device: &'static Device,
        window: &Window,
        raw_surface: vk::SurfaceKHR,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<RenderMsg>();
        let mut surface = Surface::new(instance, device, raw_surface, window).unwrap();
        let handle = std::thread::spawn(move || {
            let mut present_ring = PresentRing::new(device, instance, &surface).unwrap();
            let extent = surface.extent();
            let mut renderer = HelloDraw::new(device);
            renderer.provision(device, extent).unwrap();

            let frame_budget = std::time::Duration::from_secs_f64(1.0 / 60.0);
            let now = std::time::Instant::now();
            let mut next_tick = std::time::Instant::now();

            'render: loop {
                let timeout = next_tick.saturating_duration_since(std::time::Instant::now());
                match rx.recv_timeout(timeout) {
                    Ok(RenderMsg::Resized(mut size)) => {
                        // Collapse a burst of resize events; only the final extent matters.
                        loop {
                            match rx.try_recv() {
                                Ok(RenderMsg::Resized(newer)) => size = newer,
                                Ok(RenderMsg::Shutdown) => break 'render,
                                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                                Err(std::sync::mpsc::TryRecvError::Disconnected) => break 'render,
                            }
                        }
                        let extent = vk::Extent2D::default()
                            .height(size.height)
                            .width(size.width);
                        let new_size = present_ring
                            .maybe_update_swapchain(device, &mut surface, extent)
                            .unwrap();
                        renderer.provision(device, new_size).unwrap();
                        continue 'render;
                    }
                    Ok(RenderMsg::Shutdown) => break 'render,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        present_ring
                            .record(
                                device,
                                compute_present(device, |device, cb, acquired_image| {
                                    renderer.draw(device, cb, acquired_image);
                                }),
                                || {},
                            )
                            .map_err(|e| match e {
                                utate::vulkan::VulkanError::SwapchainOutOfDate
                                | utate::vulkan::VulkanError::SwapchainSuboptimal
                                | utate::vulkan::VulkanError::SwapchainRecreationRequired => {
                                    let new_size = present_ring
                                        .maybe_update_swapchain(device, &mut surface, extent)
                                        .unwrap();
                                    renderer.provision(device, new_size).unwrap();
                                }
                                _ => eprintln!("application: draw failed {:?}", e),
                            })
                            .inspect(|_| {
                                next_tick += frame_budget;
                                // println!("next tick: {:4.2}ms", (next_tick - now).as_micros() / 1000);
                            });
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break 'render,
                }
            }
            // Drain anything in flight on the device, such as final submissions that occurred
            // before the thread join.
            device.wait_idle().unwrap();
            renderer.destroy(device);
            present_ring.destroy(device);
            surface.destroy();
        });
        Self {
            handle: Some(handle),
            tx,
        }
    }
}

/// The resumed state of the application, represented as the `Active` `AppState` variant.
struct ActiveApp {
    window: Window,
    device: &'static Device,
    render_thread: WindowContext,
}

impl ActiveApp {
    fn new(instance: &'static Instance, event_loop: &ActiveEventLoop) -> Self {
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
        // Once we have chosen the physical device, then we can create the logical device.
        let device: &'static Device = Box::leak(Box::new(selected.into_logical(instance)));
        // Rendering gear is created inside the self-contained render thread scope.  We will talk to
        // it via a channel.
        let window_context = WindowContext::spawn(instance, device, &window, raw_surface);
        Self {
            window,
            device,
            render_thread: window_context,
        }
    }

    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            // ROLL winit latest docs seem to show a new `SurfaceResized` event.  Will pick up on
            // next winit update.
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                let _ = self.render_thread.tx.send(RenderMsg::Resized(size));
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
                event_loop.exit();
            }
            _ => {}
        }
    }

    fn shutdown(&mut self) {
        let _ = self.render_thread.tx.send(RenderMsg::Shutdown);
        if let Some(handle) = self.render_thread.handle.take() {
            handle.join().unwrap();
        }
        self.device.destroy();
    }
}

// NOTE see the visualizers WindowExt for how to enable full screen support.
fn make_window(event_loop: &ActiveEventLoop) -> Window {
    let attrs = Window::default_attributes().with_title("µTate Minimal!");
    event_loop
        .create_window(attrs)
        .expect("Failed to create window")
}

/// An Enum is used to represent resume and suspend with a single type on the `MinimalApp`.
enum AppState {
    Dormant,
    Active(ActiveApp),
}

/// Longest lived structure in the application lifecycle.  Holds various states of initialization in
/// `state` and forwards window events through `ActiveApp` when resumed.
struct MinimalApp {
    instance: &'static Instance,
    state: AppState,
}

impl ApplicationHandler for MinimalApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.state = AppState::Active(ActiveApp::new(self.instance, event_loop));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let AppState::Active(active) = &mut self.state {
            active.handle_window_event(event_loop, window_id, event);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let AppState::Active(active) = &mut self.state {
            active.shutdown();
        }
    }
}

fn main() -> Result<(), utate::MutateError> {
    let event_loop = EventLoop::builder().build().unwrap();
    let instance: &'static Instance = Box::leak(Box::new(Instance::with_display(&event_loop, &[])));
    let mut app = MinimalApp {
        instance,
        state: AppState::Dormant,
    };
    event_loop.run_app(&mut app).unwrap();
    // After close, all users of the instance have dropped and destroyed everything.
    unsafe {
        let instance = Box::from_raw(instance as *const Instance as *mut Instance);
        instance.destroy();
    }
    Ok(())
}
