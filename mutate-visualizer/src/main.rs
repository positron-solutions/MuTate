// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod assets;
mod node;
mod render_target;
mod swapchain;
mod vk_context;

use ash::vk;
use clap::Parser;
use palette::convert::FromColorUnclamped;
use ringbuf::traits::*;
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
    render_target: Option<render_target::RenderTarget>,
    swapchain: Option<swapchain::SwapChain>,

    render_node: Option<node::RenderNode>,

    // move into a render graph node, an audio to color stream
    audio_events: ringbuf::wrap::caching::Caching<
        std::sync::Arc<ringbuf::SharedRb<ringbuf::storage::Heap<(f32, f32, f32, f32)>>>,
        false,
        true,
    >,
    hue: f32,
    value: f32,
}

impl App {
    fn toggle_fullscreen(&self) {
        let win = &self.render_target.as_ref().unwrap().window;
        match win.fullscreen() {
            Some(winit::window::Fullscreen::Borderless(None)) => {
                win.set_fullscreen(None);
                win.set_cursor_visible(true);
            }
            _ => {
                win.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                win.set_cursor_visible(false);
            }
        }
    }

    fn draw_frame(&mut self) {
        // Receive available audio inputs
        if self.audio_events.is_full() {
            eprintln!("audio event backpressure drop");
            self.audio_events.skip(1);
        }
        let (_slow, fast) = match self.audio_events.try_pop() {
            Some(got) => ((got.0 + got.1), (got.2 + got.3)),
            None => {
                eprintln!("No audio event was ready");
                (0.1, 0.1)
            }
        };

        // Push color information into audio node
        self.value = fast;
        self.hue += 0.002 * fast;
        if self.hue > 1.0 || self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        } else if self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        }

        let vk_context = self.vk_context.as_ref().unwrap();
        let sc = self.swapchain.as_mut().unwrap();
        let output = sc.render_target(vk_context);
        let sync = &output.4;

        self.record_command_buffer(&output);

        let wait_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: sync.image_available,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            device_index: 0,
            ..Default::default()
        };

        let signal_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: sync.render_finished,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            device_index: 0,
            ..Default::default()
        };

        // Submission is mainly a device/queue behavior
        let cb_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: output.3,
            device_mask: 0,
            ..Default::default()
        };

        let submit = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cb_info,
            ..Default::default()
        };

        let queue = vk_context.graphics_queue();
        unsafe {
            vk_context
                .device
                .queue_submit2(*queue, &[submit], sync.in_flight)
                .expect("queue_submit2 failed");
        }

        // Presenters present
        let sc = self.swapchain.as_ref().unwrap();
        let present_wait = [sync.render_finished];
        let swapchains = [sc.swapchain];
        let indices = [sync.image_index as u32];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: present_wait.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: indices.as_ptr(),
            ..Default::default()
        };

        unsafe {
            match sc.swapchain_loader.queue_present(*queue, &present_info) {
                Ok(_) => {
                    // MAYBE How to interpret false?
                }
                Err(result) => eprintln!("presentation error: {:?}", result),
            };
        }

        let rt = self.render_target.as_ref().unwrap();
        rt.window.request_redraw();
    }

    fn record_command_buffer(
        &self,
        render_target: &(
            vk::Image,
            vk::ImageView,
            vk::Extent2D,
            vk::CommandBuffer,
            crate::swapchain::DrawSync,
        ),
    ) {
        // Extract audio -> color stream
        let tweaked = self.value * 0.02 + 0.3;
        let value = tweaked.clamp(0.0, 1.0);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(self.hue * 360.0, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [rgb.red, rgb.green, rgb.blue, 1.0],
            },
        };

        let mut trie_hue = self.hue * 360.0 + 180.0;
        if trie_hue > 360.0 {
            trie_hue -= 360.0;
        }
        let scale = 0.8 + (0.2 * self.value);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(trie_hue, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        let vk_context = self.vk_context.as_ref().unwrap();
        let device = &vk_context.device;

        let cb = render_target.3;

        // More synchronization, creation of inputs, and culmination at our one draw command
        unsafe {
            device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .expect("reset_cb failed");

            let begin = vk::CommandBufferBeginInfo::default();
            device
                .begin_command_buffer(cb, &begin)
                .expect("begin failed");
        }

        let barrier = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            image: render_target.0,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let dep_info = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier,
            ..Default::default()
        };

        unsafe { device.cmd_pipeline_barrier2(cb, &dep_info) };

        let color_attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: render_target.1,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: clear,
            ..Default::default()
        };

        let render_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: render_target.2,
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        unsafe { device.cmd_begin_rendering(cb, &render_info) };

        let context = self.vk_context.as_ref().unwrap();
        self.render_node
            .as_ref()
            .unwrap()
            .draw(cb, context, rgb, scale, &render_target.2);

        unsafe { device.cmd_end_rendering(cb) };

        // XXX presentation details leaking into draw
        let barrier2 = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags2::empty(),
            image: render_target.0,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let dep2 = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &barrier2,
            ..Default::default()
        };

        unsafe { device.cmd_pipeline_barrier2(cb, &dep2) };

        unsafe { device.end_command_buffer(cb).expect("end_cb failed") };
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let vk_context = VkContext::new();
        let rt = render_target::RenderTarget::new(&vk_context, event_loop, &self.args);

        let device = &vk_context.device;

        let sc = swapchain::SwapChain::new(&vk_context, &rt);

        // Render nodes need a device in order to allocate things.  They will need an entire vk_context to
        // properly interact with memory management.
        self.render_node = Some(node::RenderNode::new(
            device,
            rt.surface_format.format.clone(),
        ));
        self.render_target = Some(rt);
        self.vk_context = Some(vk_context);
        self.swapchain = Some(sc);
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
                            self.toggle_fullscreen();
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
                    let rt = self.render_target.as_ref().unwrap();
                    self.swapchain
                        .as_mut()
                        .unwrap()
                        .recreate_images(&vk_context, &rt);
                }
            }
            WindowEvent::RedrawRequested => {
                if self.running {
                    self.draw_frame();
                }
            }
            WindowEvent::CloseRequested => unsafe {
                self.running = false;
                let vk_context = self.vk_context.as_ref().unwrap();
                let device = &vk_context.device();

                device.device_wait_idle().unwrap();

                self.render_node.as_ref().unwrap().destroy(&device);

                self.swapchain.as_ref().unwrap().destroy(&device);
                self.render_target.as_ref().unwrap().destroy(vk_context);
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

    let context = utate::AudioContext::new()?;
    println!("Choose the audio source:");

    let mut first_choices = Vec::new();
    let check = |choices: &[utate::AudioChoice]| {
        first_choices.extend_from_slice(choices);
    };

    context.with_choices_blocking(check).unwrap();
    first_choices.iter().enumerate().for_each(|(i, c)| {
        println!("[{}] {} AudioChoice: {:?}", i, c.id(), c.name());
    });

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    let choice_idx = input.trim().parse().unwrap();
    let choice = first_choices.remove(choice_idx);

    let rx = context.connect(&choice, "mutate".to_owned()).unwrap();

    // audio events, processed results of the buffer, using an independent ring to provide some
    // buffering, synchronized communication, and back pressure support.
    let ae_ring = ringbuf::HeapRb::new(3);
    let (mut ae_tx, ae_rx) = ae_ring.split();

    let audio_thread = std::thread::spawn(move || {
        // This thread continuously emits events.  The scheme is a sliding window with a 120Hz width
        // and sliding in 240Hz increments.  The production of events is faster than the frame rate,
        // and balanced back pressure is accomplished by looking at the ring buffer size.

        // To subtract the noise floor, we track the moving average with a 240 sample exponential
        // moving average.
        let mut window_buffer = [0u8; 3200];
        let window_size = 3200; // one 240FPS frame at 48kHz and 8 bytes per frame
        let read_behind = 3200; // one frame of read-behind
        let mut left_max = 0f32;
        let mut right_max = 0f32;
        let mut left_noise = 0f32;
        let mut right_noise = 0f32;

        let alpha = 2.0 / (240.0 + 1.0);
        let alpha_resid = 1.0 - alpha;

        let mut left_fast_accum = 0f32;
        let mut right_fast_accum = 0f32;
        let mut left_fast = 0f32;
        let mut right_fast = 0f32;
        let alpha_f = 2.0 / (8.0 + 1.0);
        let alpha_f_resid = 1.0 - alpha_f;

        // FIXME Ah yes, the user friendly API for real Gs
        let mut conn = std::mem::ManuallyDrop::new(unsafe { Box::from_raw(rx.conn) });

        while ae_tx.read_is_held() {
            let avail = conn.buffer.occupied_len();
            if avail >= window_size {
                let read = conn.buffer.peek_slice(&mut window_buffer);
                assert!(read == window_size);

                // Estimate the energy by absolute delta.  IIRC not only is this physically wrong
                // but also doesn't map to perceptual very well.
                let (mut last_l, mut last_r) = (0.0, 0.0);
                let (left_sum, right_sum) = window_buffer
                    .chunks_exact(8) // 2 samples per frame × 4 bytes = 8 bytes per frame
                    .map(|frame| {
                        let left = f32::from_le_bytes(frame[0..4].try_into().unwrap());
                        let right = f32::from_le_bytes(frame[4..8].try_into().unwrap());
                        (left, right)
                    })
                    .fold((0f32, 0f32), |(acc_l, acc_r), (l, r)| {
                        // absolute delta + absolute amplitude
                        let accum = (
                            acc_l + (l - last_l).abs() + l.abs(),
                            acc_r + (r - last_r).abs() + r.abs(),
                        );
                        last_l = l;
                        last_r = r;
                        accum
                    });

                left_noise = (alpha * left_sum) + (alpha_resid * left_noise);
                right_noise = (alpha * right_sum) + (alpha_resid * right_noise);

                // Cut noise and normalize remaining to noise
                let left_excess = (left_sum - (left_noise * 1.3)) / left_noise.max(0.000001);
                let right_excess = (right_sum - (right_noise * 1.3)) / right_noise.max(0.000001);

                // Fast EMA of the cleaned signal for beats
                left_fast = (alpha_f * left_excess) + (alpha_f_resid * left_fast);
                right_fast = (alpha_f * right_excess) + (alpha_f_resid * right_fast);

                // Instantaneous response on climb
                if left_fast < left_excess {
                    left_fast = left_excess;
                }
                if right_fast < right_excess {
                    right_fast = right_excess;
                }

                left_fast_accum = left_fast + left_fast_accum;
                right_fast_accum = right_fast + right_fast_accum;

                left_max = left_max.max(left_excess);
                right_max = right_max.max(right_excess);

                // Backoff using queue size
                if ae_tx.vacant_len() > 1 {
                    match ae_tx.try_push((left_max, right_max, left_fast_accum, right_fast_accum)) {
                        Ok(_) => {}
                        Err(e) => {
                            eprintln!("sending audio event failed: {:?}", e);
                            if ae_tx.is_full() {
                                eprintln!("audio event consumer is falling behind");
                            }
                        }
                    }
                    left_max = 0.0;
                    right_max = 0.0;
                    left_fast_accum = 0.0;
                    right_fast_accum = 0.0;
                }

                if avail >= (window_size * 2) + read_behind {
                    conn.buffer.skip(window_size / 2 + 200); // LIES +200 🤔
                }

                std::thread::sleep(std::time::Duration::from_secs_f64(1.0 / 240.0));
            } else {
                // Underfed, either we can pad with "empty" data or wait for new data.  Let's wait.
                match rx.wait() {
                    Ok(_) => {
                        eprintln!("audio buffered ⏰");
                    }
                    Err(e) => {
                        eprintln!("listening aborted: {}", e);
                        break;
                    }
                }
            }
        }
    });

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        args,

        running: true,

        vk_context: None,
        render_target: None,
        swapchain: None,
        render_node: None,

        audio_events: ae_rx,
        hue: rand::random::<f32>(),
        value: 0.0,
    };
    event_loop.run_app(&mut app).unwrap();
    drop(app);
    audio_thread.join().unwrap();
    Ok(())
}
