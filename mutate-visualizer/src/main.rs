// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod assets;
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

#[derive(Parser, Debug)]
struct Args {
    /// Run in fullscreen mode
    #[arg(short = 'f', long = "fullscreen")]
    fullscreen: bool,
}

struct App {
    args: Args,
    running: bool,

    vk_context: Option<VkContext>,
    window_present: Option<present::WindowPresent>,

    // This field will turn into a graph when graphs are ready
    render_node: Option<node::RenderNode>,
    audio_node: node::AudioNode,
}

impl App {
    fn draw_frame(&mut self) {
        let audio_colors = self.audio_node.process();

        // Obtain image and hot command buffer
        let vk_context = self.vk_context.as_ref().unwrap();
        let wp = self.window_present.as_mut().unwrap();

        let (sync, target) = wp.render_target(vk_context, audio_colors.clear);

        // Node draws to command buffer.  The idea we've isolated is that drawing to a target has
        // little to do with the source or fate of that target.
        self.render_node.as_ref().unwrap().draw(
            target.command_buffer,
            vk_context,
            audio_colors.color,
            audio_colors.scale,
            &target.extent,
        );

        let wp = self.window_present.as_ref().unwrap();
        // Presentation closes the command buffer, submits to queue, transforms image, and presents
        wp.post_draw(vk_context, sync, target);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let vk_context = VkContext::new();
        let wp = present::WindowPresent::new(&vk_context, event_loop, &self.args);

        // Render nodes need a device in order to allocate things.  They will need an entire vk_context to
        // properly interact with memory management.
        let device = &vk_context.device;
        self.render_node = Some(node::RenderNode::new(
            device,
            wp.surface_format.format.clone(),
        ));
        self.vk_context = Some(vk_context);
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
                    let vk_context = self.vk_context.as_ref().unwrap();
                    self.window_present
                        .as_mut()
                        .unwrap()
                        .recreate_images(&vk_context);
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
                let vk_context = self.vk_context.as_ref().unwrap();
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
    let audio_node = node::AudioNode::new()?;
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        args,
        running: true,

        vk_context: None,
        window_present: None,

        render_node: None,
        audio_node,
    };
    event_loop.run_app(&mut app).unwrap();
    app.audio_node.destroy();
    Ok(())
}
