// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod assets;
mod graph;
mod node;
mod present;
mod vk_context;

use clap::Parser;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    event_loop::{ControlFlow, EventLoop},
    keyboard as kb,
};

use vk_context::VkContext;

use mutate_lib as utate;

use crate::node::audio;

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
    window_present: Option<present::WindowPresent>,

    // These fields will turn into a graph when graphs are ready
    render_node: Option<node::video::RenderNode>,
    raw_audio: audio::raw::RawAudioNode,
    rms: audio::rms::RmsNode,
    k_weights: audio::kweight::KWeightsNode,
    colors: audio::colors::AudioColorsNode,
}

impl App {
    fn draw_frame(&mut self) {
        let vk_context = &self.vk_context;

        let wp = self.window_present.as_mut().unwrap();
        // LIES the previous frame fence is implicitly always signaled already
        wp.draw_wait(vk_context);

        // NEXT dynamically waiting down to the approximate latch timing to late bind the last
        // possible audio.
        std::thread::sleep(std::time::Duration::from_millis(12));

        // NOTE A manually driven, unrolled render graph.  These are the associations that must
        // be described in the eventual graph connectivity APIs.
        let raw_state = self.raw_audio.consume().unwrap();
        let raw_out = self.raw_audio.produce().unwrap();
        self.k_weights.consume(&raw_out);

        // The control loop, unrolled
        if raw_state == crate::node::SeekState::UnderProduced {
            let raw_out = self.raw_audio.produce().unwrap();
            self.k_weights.consume(&raw_out);
        };

        let kweights_out = self.k_weights.produce();
        self.rms.consume(&kweights_out);
        let rms_out = self.rms.produce();
        self.colors.consume(&rms_out);
        let colors = self.colors.produce();

        // Obtain swapchain image and hot command buffer
        let (sync, target) = wp.render_target(vk_context, colors.clear);

        // Node draws to command buffer.  The idea we've isolated is that drawing to a target has
        // little to do with the source or fate of that target.
        self.render_node.as_ref().unwrap().draw(
            target.command_buffer,
            vk_context,
            colors.color,
            colors.scale,
            &target.extent,
        );

        // Presentation closes the command buffer, submits to queue, transforms image, and presents.
        // Also waits on presentation.
        wp.post_draw(vk_context, sync, target);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let vk_context = &self.vk_context;
        let device = &vk_context.device;
        let wp = present::WindowPresent::new(vk_context, event_loop, &self.args);

        // Render nodes need a device in order to allocate things.  They will need an entire vk_context to
        // properly interact with memory management.
        self.render_node = Some(node::video::RenderNode::new(
            device,
            wp.surface_format.format.clone(),
        ));
        self.window_present = Some(wp);
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
                            self.window_present.as_ref().unwrap().toggle_fullscreen();
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
                    let vk_context = &self.vk_context;
                    self.window_present
                        .as_mut()
                        .unwrap()
                        .recreate_images(vk_context);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.running {
                    self.draw_frame();
                    self.window_present
                        .as_ref()
                        .unwrap()
                        .window
                        .request_redraw();
                }
            }
            WindowEvent::CloseRequested => unsafe {
                self.running = false;
                let vk_context = &self.vk_context;
                let device = &vk_context.device();

                device.device_wait_idle().unwrap();
                self.render_node.as_ref().unwrap().destroy(&device);
                self.window_present.as_ref().unwrap().destroy(vk_context);
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        args,
        running: true,

        vk_context: VkContext::new(),
        window_present: None,

        render_node: None,

        raw_audio: audio::raw::RawAudioNode::new().unwrap(),
        k_weights: audio::kweight::KWeightsNode::new(),
        rms: audio::rms::RmsNode::new(),
        colors: audio::colors::AudioColorsNode::new(),
    };
    event_loop.run_app(&mut app).unwrap();
    app.raw_audio.destroy();
    Ok(())
}
