// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Visualizer
//!
//! - [`WindowContext`] owns per-window resources and lifecycle
//! - [`ActiveApp`] is a variant of [`AppState`] that encapsulates resources that live within the
//!   resumed lifecycle segments of the application.
//! - [`MutateApp`] owns the longest lived resources such as the Vulkan context and audio input
//!   stream.

mod audio;
mod graph;
mod video;
mod window;

use std::collections::HashMap;

use ash::vk;
use clap::Parser;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard as kb,
    platform::wayland::{EventLoopBuilderExtWayland, EventLoopExtWayland},
    window::{Window, WindowId},
};

use mutate_lib::{self as utate, prelude::*};

use graph::node;
use window::WindowExt;

// NEXT on-device audio processing is immenent.
// NEXT break up the "graph" module or at least make it less misleading.
// FIXME extent threading through swapchain is a *mess*.  It's a simple piece of data.  The renderer
// can basically trust swapchain image sizes (that's the physical memory we are writing to) and
// these other cues should be treated as hints about the current swapchain still being valid.

#[derive(Parser, Debug)]
struct Args {
    /// Start in fullscreen mode
    #[arg(short = 'f', long = "fullscreen")]
    fullscreen: bool,
}

/// Each time we construct a window, we need a surface and swapchain to run the render loop for that
/// window.
struct WindowContext {
    window: winit::window::Window,
    surface: Surface,
    present_ring: PresentRing,

    // NEXT As the render architecture gets more sophisticated, a lot of the spectrum work on the
    // device can be shared per window that the device supports.  Rare device-per-window cases, if
    // they still exist, would require only one audio downstream per device and then that data can
    // be reused for all windows.
    render_node: video::spectrum::SpectrumNode,
}

impl WindowContext {
    fn new(
        instance: &Instance,
        device: &mut Device,
        window: winit::window::Window,
        raw_surface: vk::SurfaceKHR,
    ) -> Self {
        let surface = Surface::new(instance, device, raw_surface, &window).unwrap();
        let present_ring = PresentRing::new(device, instance, &surface).unwrap();
        let mut render_node = video::spectrum::SpectrumNode::new(device);
        render_node.provision(device, surface.extent()).unwrap();
        Self {
            window,
            surface,
            present_ring,
            render_node,
        }
    }

    fn draw_frame(&mut self, audio: &mut Audio, device: &mut Device) {
        // NEXT: audio is consumed once per draw_frame call; in a multi-window
        // setup this would double-consume.  Hoist the audio pump above the
        // per-window loop in ActiveApp when that becomes necessary.
        let raw_state = audio.raw.consume().unwrap();
        let raw_out = audio.raw.produce().unwrap();
        audio.cqt.consume(&raw_out);
        if raw_state == node::SeekState::UnderProduced {
            let raw_out = audio.raw.produce().unwrap();
            audio.cqt.consume(&raw_out);
        }
        let cqt = audio.cqt.produce();
        self.present_ring
            .record(
                device,
                compute_present(device, |device, cb, acquired_image| {
                    self.render_node.draw(device, cb, acquired_image, cqt);
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

    fn handle_resize(&mut self, device: &mut Device) -> Result<(), MutateError> {
        let new_size = self
            .present_ring
            .maybe_update_swapchain(device, &mut self.surface, &self.window)
            .unwrap();
        self.render_node.provision(device, new_size)?;
        self.window.request_redraw();
        Ok(())
    }

    /// Consumes self; call only after the device queue is idle for this window.
    fn destroy(self, device: &mut Device) {
        self.render_node.destroy(device);
        self.present_ring.destroy(device);
        self.surface.destroy();
    }
}

/// Lives from first-window creation to last-window destruction.
// NEXT allow devices to vary at runtime and use different devices per window or different devices
// for different roles.
struct ActiveApp {
    device: Device,
    windows: HashMap<WindowId, WindowContext>,
}

impl ActiveApp {
    fn new(instance: &Instance, args: &Args, event_loop: &ActiveEventLoop) -> Self {
        let window = Window::from_args(args, event_loop);
        let raw_surface = instance.surface(event_loop, &window);

        let supported_devices: Vec<SupportedDevice> = instance
            .supported_devices(&[])
            .into_iter()
            .filter(|sd| sd.supports_surface(raw_surface, instance))
            .collect();
        if supported_devices.is_empty() {
            panic!("ActiveApp::new: no Vulkan device supports the created surface.");
        }
        // NEXT: read a preferred device from config instead of always picking index 0.
        let selected = supported_devices[0].clone();
        println!("device selected: {}", selected.name);
        let mut device = selected.into_logical(instance);

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
        instance: &Instance,
        audio: &mut Audio,
    ) {
        match event {
            // MAYBE do they get before matching the variant?
            WindowEvent::RedrawRequested => {
                if let Some(wc) = self.windows.get_mut(&window_id) {
                    wc.draw_frame(audio, &mut self.device);
                    wc.window.request_redraw();
                }
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(wc) = self.windows.get_mut(&window_id) {
                    wc.handle_resize(&mut self.device);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(wc) = self.windows.get(&window_id) {
                    handle_keyboard(&event, wc, event_loop);
                }
            }
            WindowEvent::CloseRequested => {
                if let Some(wc) = self.windows.remove(&window_id) {
                    // NEXT expose pool epoch for more precise waiting.
                    self.device.wait_idle();
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

fn handle_keyboard(
    event: &winit::event::KeyEvent,
    wc: &WindowContext,
    event_loop: &ActiveEventLoop,
) {
    if !event.repeat && event.state == winit::event::ElementState::Pressed {
        match event.physical_key {
            kb::PhysicalKey::Code(kb::KeyCode::KeyF) => {
                wc.window.toggle_fullscreen();
            }
            kb::PhysicalKey::Code(kb::KeyCode::KeyQ)
            | kb::PhysicalKey::Code(kb::KeyCode::Escape) => {
                event_loop.exit();
            }
            _ => {}
        }
    }
}

/// Represents the possible states of construction as variants of a single type so that the
/// MutateApp can be a single type updating a field to go through transitions.
enum AppState {
    Dormant,
    Active(ActiveApp),
}

/// Longest lived resources, truly create-once context.  Implements `ApplicationHandler` and
/// delegates to the state appropriately.
struct MutateApp {
    args: Args,
    instance: Instance,
    audio: Audio,
    state: AppState,
}

impl ApplicationHandler for MutateApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Transition Dormant -> Active by creating the first window.
        // Device selection happens here once; subsequent windows reuse it.
        debug_assert!(matches!(self.state, AppState::Dormant));
        let active = ActiveApp::new(&self.instance, &self.args, event_loop);
        self.state = AppState::Active(active);
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
        active.handle_window_event(
            event_loop,
            window_id,
            event,
            &self.instance,
            &mut self.audio,
        );
    }

    // handles all exit paths
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let AppState::Active(active) = &mut self.state else {
            return;
        };
        unsafe { active.device.wait_idle().unwrap() };
        for (_, wc) in active.windows.drain() {
            wc.destroy(&mut active.device);
        }
        active.device.destroy();
    }
}

/// Process-scoped audio pipeline.  Only one audio upstream is necessary for the whole application.
/// Downstream consumers can use the data without additional support.
struct Audio {
    raw: audio::raw::RawAudioNode,
    cqt: audio::cqt::CqtNode, // NEXT on-GPU audio and ditch this at last
}

impl Audio {
    fn new() -> Result<Self, MutateError> {
        Ok(Self {
            raw: audio::raw::RawAudioNode::new()?,
            cqt: audio::cqt::CqtNode::new(600, 48000, 60.0),
        })
    }

    fn destroy(self) {
        self.raw.destroy();
    }
}

fn main() -> Result<(), MutateError> {
    let args = Args::parse();
    let event_loop = EventLoop::builder().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = MutateApp {
        instance: Instance::with_display(&event_loop, &[]),
        audio: Audio::new()?,
        args,
        state: AppState::Dormant,
    };
    event_loop.run_app(&mut app).unwrap();

    app.audio.destroy();
    app.instance.destroy();
    Ok(())
}
