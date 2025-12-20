// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

mod assets;
mod render_target;
mod swapchain;
mod vk_context;

use std::ffi::CString;

use ash::vk;
use clap::Parser;
use palette::convert::FromColorUnclamped;
use ringbuf::traits::*;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    event_loop::{ControlFlow, EventLoop},
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

    render_base: Option<VkContext>,
    render_target: Option<render_target::RenderTarget>,
    swapchain: Option<swapchain::SwapChain>,

    command_buffers: Vec<vk::CommandBuffer>,
    command_pool: Option<vk::CommandPool>,
    pipeline_layout: Option<vk::PipelineLayout>,
    pipelines: Option<Vec<vk::Pipeline>>,

    audio_events: ringbuf::wrap::caching::Caching<
        std::sync::Arc<ringbuf::SharedRb<ringbuf::storage::Heap<(f32, f32, f32, f32)>>>,
        false,
        true,
    >,
    hue: f32,
    value: f32,
}

impl App {
    fn draw_frame(&mut self) {
        let render_base = self.render_base.as_ref().unwrap();
        let device = &render_base.device;
        let queue = render_base.graphics_queue();

        let sc = self.swapchain.as_ref().unwrap();

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

        self.value = fast;

        self.hue += 0.002 * fast;
        if self.hue > 1.0 || self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        } else if self.hue < 0.0 {
            self.hue = self.hue - self.hue.floor();
        }

        let idx = sc.frame_index;
        let image_available = sc.image_available_semaphores[idx];
        let render_finished = sc.render_finished_semaphores[idx];
        let in_flight = sc.in_flight_fences[idx];

        // While there is a lot of silly code, this is much more silly.
        let sc = self.swapchain.as_mut().unwrap();
        sc.frame_index = (sc.frame_index + 1) % sc.frames;
        let sc = self.swapchain.as_ref().unwrap();

        unsafe {
            device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("wait_for_fences failed");

            device
                .reset_fences(&[in_flight])
                .expect("reset_fences failed");
        }

        let (image_index, _) = unsafe {
            sc.swapchain_loader
                .acquire_next_image(
                    sc.swapchain,
                    std::u64::MAX,
                    image_available,
                    vk::Fence::null(),
                )
                .expect("Failed to acquire next image")
        };

        let render_target = sc.render_target(image_index as usize);

        // Wait for the image-available semaphore before executing commands.
        let wait_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: image_available,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            device_index: 0,
            ..Default::default()
        };

        // Signal when rendering is done.
        let signal_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: render_finished,
            value: 0,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            device_index: 0,
            ..Default::default()
        };

        let rt = self.render_target.as_ref().unwrap();

        let command_buffer = self.command_buffers[image_index as usize];

        self.record_command_buffer(render_target, &sc.swapchain_extent, command_buffer);

        // Which command buffer to submit.
        let cmd_buffer = self.command_buffers[image_index as usize];

        let cmd_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: cmd_buffer,
            device_mask: 0,
            ..Default::default()
        };

        // Submit struct (synchronization2)
        let submit = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cmd_info,
            ..Default::default()
        };

        unsafe {
            device
                .queue_submit2(*queue, &[submit], in_flight)
                .expect("queue_submit2 failed");
        }

        let present_wait = [render_finished];
        let swapchains = [sc.swapchain];
        let indices = [image_index];

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
        rt.window.request_redraw();
    }

    fn record_command_buffer(
        &self,
        render_target: (vk::Image, vk::ImageView),
        extent: &vk::Extent2D,
        cb: vk::CommandBuffer,
    ) {
        let render_base = self.render_base.as_ref().unwrap();
        let device = &render_base.device;

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
        // XXX Command barrier reset leaking into draw

        let tweaked = self.value * 0.02 + 0.3;
        let value = tweaked.clamp(0.0, 1.0);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(self.hue * 360.0, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [rgb.red, rgb.green, rgb.blue, 1.0],
            },
        };

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
                extent: *extent,
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        unsafe { device.cmd_begin_rendering(cb, &render_info) };

        let pipeline = self.pipelines.as_ref().unwrap()[0];
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        // Update the color used in the fragment shader
        let mut trie_hue = self.hue * 360.0 + 180.0;
        if trie_hue > 360.0 {
            trie_hue -= 360.0;
        }
        let scale = 0.8 + (0.2 * self.value);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(trie_hue, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);
        let combined_push: [f32; 5] = [rgb.red, rgb.green, rgb.blue, 1.0, scale];
        let pipeline_layout = self.pipeline_layout.as_ref().unwrap();
        unsafe {
            device.cmd_push_constants(
                cb,
                *pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
                0,
                std::slice::from_raw_parts(
                    combined_push.as_ptr() as *const u8,
                    std::mem::size_of::<[f32; 5]>(),
                ),
            );
        }

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: *extent,
        };

        unsafe {
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[scissor]);
        }

        unsafe { device.cmd_draw(cb, 3, 1, 0, 0) };

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
        let sc = swapchain::SwapChain::new(&vk_context, &rt);

        let queue_family_index = vk_context.queue_family_index;
        let device = &vk_context.device;

        // MAYBE getting command pools requires the queue index
        let command_pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        };

        // allocate one command buffer per swapchain image
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: sc.swapchain_images.len() as u32,
            ..Default::default()
        };

        let buffers = unsafe { device.allocate_command_buffers(&alloc_info).unwrap() };

        let assets = assets::AssetDirs::new();
        let vert_spv = assets
            .find_bytes("vertex", assets::AssetKind::Shader)
            .unwrap();
        let frag_spv = assets
            .find_bytes("fragment", assets::AssetKind::Shader)
            .unwrap();

        let vert_module_ci = vk::ShaderModuleCreateInfo {
            code_size: vert_spv.len(),
            p_code: vert_spv.as_ptr() as *const u32,
            ..Default::default()
        };

        let frag_module_ci = vk::ShaderModuleCreateInfo {
            code_size: frag_spv.len(),
            p_code: frag_spv.as_ptr() as *const u32,
            ..Default::default()
        };

        let vert_shader_module =
            unsafe { device.create_shader_module(&vert_module_ci, None).unwrap() };
        let frag_shader_module =
            unsafe { device.create_shader_module(&frag_module_ci, None).unwrap() };

        // Static
        let entry_vert = CString::new("main").unwrap();
        let entry_frag = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_shader_module,
                p_name: entry_vert.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag_shader_module,
                p_name: entry_frag.as_ptr(),
                ..Default::default()
            },
        ];

        // I messed around with two ranges but the validation of the shader code made me believe
        // that using two separate push constants was at best hacky.  Therefore, I merged the ranges
        // and just used the relevant data in each shader.
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<[f32; 5]>() as u32,
        };

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_attribute_description_count: 0,
            vertex_binding_description_count: 0,
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            color_write_mask: vk::ColorComponentFlags::RGBA,
        };

        let color_blend = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constant_range,
            ..Default::default()
        };
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let color_formats = [rt.surface_format.format];
        let pipeline_rendering_info = vk::PipelineRenderingCreateInfo {
            s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachment_formats: color_formats.as_ptr(),
            ..Default::default()
        };

        let pipeline_ci = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: &pipeline_rendering_info as *const _ as *const std::ffi::c_void,
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass: vk::RenderPass::null(), // dynamic rendering
            subpass: 0,
            ..Default::default()
        };

        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
        }
        .unwrap();

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        self.command_pool = Some(command_pool);
        self.command_buffers = buffers;
        self.pipelines = Some(pipelines);
        self.pipeline_layout = Some(pipeline_layout);

        self.render_target = Some(rt);
        self.render_base = Some(vk_context);
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
                device_id,
                event,
                is_synthetic,
            } => {
                if !event.repeat
                    && event.state == winit::event::ElementState::Pressed
                    && event.physical_key
                        == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyF)
                {
                    let win = &self.render_target.as_ref().unwrap().window;
                    match win.fullscreen() {
                        Some(winit::window::Fullscreen::Borderless(None)) => {
                            win.set_fullscreen(None);
                        }
                        _ => {
                            win.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                        }
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    println!("window resize reported degenerate size");
                } else {
                    let vk_context = self.render_base.as_ref().unwrap();
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
                let render_base = self.render_base.as_ref().unwrap();
                let device = &render_base.device();

                device.device_wait_idle().unwrap();

                self.pipelines.as_ref().unwrap().iter().for_each(|p| {
                    device.destroy_pipeline(*p, None);
                });

                if let Some(layout) = self.pipeline_layout {
                    device.destroy_pipeline_layout(layout, None);
                }
                self.command_pool.map(|p| {
                    device.destroy_command_pool(p, None);
                });

                self.swapchain.as_ref().unwrap().destroy(&device);
                self.render_target.as_ref().unwrap().destroy();
                render_base.destroy();

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

        render_base: None,
        render_target: None,
        swapchain: None,

        command_buffers: Vec::new(),
        command_pool: None,
        pipeline_layout: None,
        pipelines: None,

        audio_events: ae_rx,
        hue: rand::random::<f32>(),
        value: 0.0,
    };
    event_loop.run_app(&mut app).unwrap();
    drop(app);
    audio_thread.join().unwrap();
    Ok(())
}
