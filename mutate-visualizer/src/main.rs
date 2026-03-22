// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod audio;
mod graph;
mod video;
mod window;

use ash::vk;
use clap::Parser;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard as kb,
    platform::wayland::{EventLoopBuilderExtWayland, EventLoopExtWayland},
    window::Window,
};

use mutate_lib::{self as utate, prelude::*, vulkan::prelude::*};

use graph::node;
use window::WindowExt;

#[derive(Parser, Debug)]
struct Args {
    /// Run in fullscreen mode
    #[arg(short = 'f', long = "fullscreen")]
    fullscreen: bool,
}

struct App {
    args: Args,
    running: bool,
    vk_context: VkContext,

    // Initialized on resume.
    // NEXT surface, window, and the enclosing context, SurfacePresent all have a pretty closely
    // tied lifecycle and likey can be abstracted.
    surface: Option<VkSurface>,
    window: Option<winit::window::Window>,
    device_context: Option<DeviceContext>,
    surface_present: Option<video::present::SurfacePresent>,

    // These fields will turn into a graph when graphs are ready
    render_node: Option<video::spectrum::SpectrumNode>,
    raw_audio: audio::raw::RawAudioNode,
    // rms: audio::rms::RmsNode,
    // k_weights: audio::kweight::KWeightsNode,
    // colors: audio::colors::AudioColorsNode,
    cqt: audio::cqt::CqtNode,
}

impl App {
    fn draw_frame(&mut self) {
        let device_context = &self.device_context.as_ref().unwrap();

        // NEXT dynamically waiting down to the approximate latch timing to late bind the last
        // possible audio.
        // std::thread::sleep(std::time::Duration::from_millis(6));

        // NOTE A manually driven, unrolled render graph.  These are the associations that must
        // be described in the eventual graph connectivity APIs.
        let raw_state = self.raw_audio.consume().unwrap();

        let raw_out = self.raw_audio.produce().unwrap();
        self.cqt.consume(&raw_out);

        // The control loop, unrolled
        if raw_state == crate::graph::node::SeekState::UnderProduced {
            let raw_out = self.raw_audio.produce().unwrap();
            self.cqt.consume(&raw_out);
        };

        let cqt = self.cqt.produce();

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let sp = self.surface_present.as_mut().unwrap();
        sp.draw_wait(device_context);

        // Obtain swapchain image and hot command buffer
        let (recording_slot, acquired_image) = sp.render_target(device_context, clear);

        // XXX hold size on swapchain updates?  A parameter system would 😉
        let size = self.window.as_ref().unwrap().render_size();
        // Node draws to command buffer.  The idea we've isolated is that drawing to a target has
        // little to do with the source or fate of that target.
        self.render_node.as_mut().unwrap().draw(
            &recording_slot,
            &acquired_image,
            cqt,
            device_context,
            size,
        );

        // Presentation closes the command buffer, submits to queue, transforms image, and presents.
        // Also waits on presentation.
        let window = self.window.as_ref().unwrap();
        sp.post_draw(device_context, &recording_slot, &acquired_image);
        // Winit says this helps align the window system latching with REDRAW_REQUESTED events.
        // However, it is supported only on Wayland at this time.
        window.pre_present_notify();
        sp.present(device_context, acquired_image);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // MAYBE This sequence of initialization is almost a series of contracts.  It may be
        // appropriate to bind lifetimes and encode a series of choices.  There may be more than one
        // window, and one device may serve multiple windows.  Headless rendering needs to remain
        // supported.
        let vk_context = &self.vk_context;
        let window = Window::from_args(&self.args, event_loop);
        let surface = {
            let display_handle = event_loop
                .display_handle()
                .expect("Event loop has no display handle")
                .as_raw();
            let window_handle = window
                .window_handle()
                .expect("raw_window_handle: platform unsupported")
                .as_raw();
            let VkContext { entry, instance } = vk_context;
            unsafe {
                ash_window::create_surface(entry, instance, display_handle, window_handle, None)
                    .expect("ash_window: could not create surface")
            }
        };
        // NOTE while we can create the surface before we have a logical device,
        // Get the surface support
        let supported_devices: Vec<SupportedDevice> = vk_context
            .supported_devices(&[])
            .into_iter()
            .filter(|sd| sd.supports_surface(surface, &vk_context))
            .collect();
        if supported_devices.is_empty() {
            panic!("main: no devices supporting surface found.");
        }
        // If we were going to ask the user, the time is now.
        let selected = supported_devices[0].clone();
        println!("device selected: {}", selected.name);
        // XXX During swapchain rebuild, remember to re-poll the surface since a lot of the answers
        // might be dynamic over surface lifetime(?)
        let surface = VkSurface::new(surface, vk_context, selected.device());
        // Inspect devices for present queue support and other support
        let device_context = selected.into_logical(&vk_context);
        let fallback = window.render_size();
        let extent = surface
            .resolve_size(&device_context, Some(fallback))
            .unwrap_or(vk::Extent2D {
                width: 800,
                height: 600,
            });

        // XXX why so much needed downstream?
        let sp = video::present::SurfacePresent::new(&device_context, vk_context, &surface, extent);

        let mut render_node =
            video::spectrum::SpectrumNode::new(&device_context, surface.format.format);
        // DEBT memory management, resources, render graph
        render_node
            .provision(
                &device_context,
                // XXX made up size :-)
                vk::Extent2D {
                    height: 800,
                    width: 800,
                },
            )
            .unwrap();
        self.render_node = Some(render_node);

        self.window = Some(window);
        self.surface = Some(surface);
        self.device_context = Some(device_context);
        self.surface_present = Some(sp);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                if !event.repeat && event.state == winit::event::ElementState::Pressed {
                    match event.physical_key {
                        kb::PhysicalKey::Code(kb::KeyCode::KeyF) => {
                            self.window.as_ref().unwrap().toggle_fullscreen();
                        }
                        kb::PhysicalKey::Code(kb::KeyCode::KeyQ)
                        | kb::PhysicalKey::Code(kb::KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    println!("window resize reported degenerate size");
                } else {
                    let device_context = self.device_context.as_ref().unwrap();
                    let fallback = self.window.as_ref().unwrap().render_size();
                    let surface = self.surface.as_ref().unwrap();
                    if let Some(size) = surface.resolve_size(&device_context, Some(fallback)) {
                        self.surface_present.as_mut().unwrap().recreate_images(
                            surface,
                            size,
                            device_context,
                        );
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if self.running {
                    self.draw_frame();
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::CloseRequested => unsafe {
                self.running = false;
                let vk_context = &self.vk_context;
                let device_context = self.device_context.as_ref().unwrap();
                let device = &device_context.device;

                device.device_wait_idle().unwrap();
                self.render_node
                    .as_ref()
                    .unwrap()
                    .destroy(device_context)
                    .unwrap();
                self.surface_present
                    .as_ref()
                    .unwrap()
                    .destroy(device_context);
                self.surface.as_ref().unwrap().destroy();
                device_context.destroy();
                vk_context.destroy();
                event_loop.exit();
            },
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}

fn main() -> Result<(), utate::MutateError> {
    // NEXT Merge over toml config values obtained as resources
    let args = Args::parse();
    let mut builder = EventLoop::builder();
    #[cfg(target_os = "linux")]
    {
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            builder.with_wayland();
        }
    }
    let event_loop = builder.build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let display_handle = event_loop
        .display_handle()
        .expect("winit: event loop has no raw display handle")
        .as_raw();
    let required_exts = ash_window::enumerate_required_extensions(display_handle)
        .expect("ash_window: unknown platform");
    let vk_context = VkContext::with_extensions(required_exts);

    let mut app = App {
        args,
        running: true,
        vk_context,

        window: None,
        surface: None,
        device_context: None,
        surface_present: None, // XXX separate
        render_node: None,
        raw_audio: audio::raw::RawAudioNode::new().unwrap(),
        cqt: audio::cqt::CqtNode::new(600, 48000, 60.0),
    };
    event_loop.run_app(&mut app).unwrap();
    app.raw_audio.destroy();
    Ok(())
}
