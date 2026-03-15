// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod audio;
mod graph;
mod video;
mod window;

use ash::vk;
use clap::Parser;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard as kb,
    platform::wayland::{EventLoopBuilderExtWayland, EventLoopExtWayland},
    window::Window,
};

use mutate_lib::{self as utate, prelude::*, vulkan::context::VkContext};

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
    context: DeviceContext,
    surface_present: Option<video::present::SurfacePresent>,
    window: Option<winit::window::Window>,

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
        let context = &self.context;

        let sp = self.surface_present.as_mut().unwrap();
        // LIES the previous frame fence is implicitly always signaled already
        sp.draw_wait(context);

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

        // Obtain swapchain image and hot command buffer
        let (sync, target) = sp.render_target(context, clear);

        // Node draws to command buffer.  The idea we've isolated is that drawing to a target has
        // little to do with the source or fate of that target.
        self.render_node
            .as_mut()
            .unwrap()
            .draw(&target, cqt, &self.context, &target.extent);

        // Presentation closes the command buffer, submits to queue, transforms image, and presents.
        // Also waits on presentation.
        let window = self.window.as_ref().unwrap();
        sp.post_draw(context, sync, target, window);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let vk_context = &self.vk_context;
        let context = &self.context;
        let window = Window::from_args(&self.args, event_loop);
        let surface = window.surface(&vk_context);
        let sp =
            video::present::SurfacePresent::new(context, vk_context, event_loop, &window, surface);

        let mut render_node =
            video::spectrum::SpectrumNode::new(context, sp.surface_format.format.clone());
        render_node
            .provision(
                context,
                // XXX made up size :-)
                vk::Extent2D {
                    height: 800,
                    width: 800,
                },
            )
            .unwrap();
        self.render_node = Some(render_node);
        self.window = Some(window);
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
                    let context = &self.context;
                    let window = self.window.as_ref().unwrap();
                    // FIXME a number of cases this can be wrong.
                    let size = window.inner_size();
                    let extent = vk::Extent2D {
                        width: size.width,
                        height: size.height,
                    };
                    self.surface_present
                        .as_mut()
                        .unwrap()
                        .recreate_images(extent, context);
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
                let context = &self.context;
                let device = &context.device();

                device.device_wait_idle().unwrap();
                self.render_node
                    .as_ref()
                    .unwrap()
                    .destroy(&context)
                    .unwrap();
                self.surface_present.as_ref().unwrap().destroy(context);
                self.context.destroy();
                self.vk_context.destroy();
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

    let vk_context = VkContext::new();
    let context = DeviceContext::new(&vk_context);
    let mut app = App {
        args,
        running: true,
        context,
        vk_context,

        window: None,
        surface_present: None,
        render_node: None,
        raw_audio: audio::raw::RawAudioNode::new().unwrap(),
        cqt: audio::cqt::CqtNode::new(600, 48000, 60.0),
    };
    event_loop.run_app(&mut app).unwrap();
    app.raw_audio.destroy();
    Ok(())
}
