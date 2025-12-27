// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Nodes convert inputs to outputs, creating a directed graph.
// The intent is to develop configurable composition, similar to sockets of nodes.  It won't happen
// tomorrow.
use std::ffi::CString;

use ash::vk;
use palette::convert::FromColorUnclamped;
use ringbuf::traits::*;

use mutate_lib as utate;

use crate::assets;

// This will be an interface after more nodes exist
pub struct RenderNode {
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
}

impl RenderNode {
    pub fn new(device: &ash::Device, format: vk::Format) -> Self {
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

        let color_formats = [format];
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

        Self {
            pipeline_layout,
            pipelines,
        }
    }

    pub fn draw(
        &self,
        cb: vk::CommandBuffer,
        context: &crate::VkContext,
        rgb: palette::Srgb<f32>,
        scale: f32,
        extent: &vk::Extent2D,
    ) {
        let device = context.device();
        let pipeline = self.pipelines[0];
        unsafe {
            context
                .device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        }

        let combined_push: [f32; 5] = [rgb.red, rgb.green, rgb.blue, 1.0, scale];
        unsafe {
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
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
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            self.pipelines.iter().for_each(|p| {
                device.destroy_pipeline(*p, None);
            });

            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

// Output type for our rudimentary audio -> color node
pub struct AudioColors {
    pub clear: vk::ClearValue,
    pub color: palette::Srgb<f32>,
    pub scale: f32,
}

// This is a first pass extraction of the original node.  It will be refined into a more
// render-graph like construction, an audio input node with separate processing before feeding into
// the graphics node.
pub struct AudioNode {
    audio_events: ringbuf::wrap::caching::Caching<
        std::sync::Arc<ringbuf::SharedRb<ringbuf::storage::Heap<(f32, f32, f32, f32)>>>,
        false,
        true,
    >,
    hue: f32,
    value: f32,
    handle: std::thread::JoinHandle<()>,
    context: utate::AudioContext,
}

impl AudioNode {
    pub fn new() -> Result<Self, utate::MutateError> {
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

        // Of course this needs retry and a default.  Also, the stream in use does not seem to be
        // respecting our choice of stream anyway.  That should be fixed for cases where multiple output
        // streams are valid.
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let choice_idx = input.trim().parse().unwrap();
        let choice = first_choices.remove(choice_idx);

        let rx = context.connect(&choice, "mutate".to_owned()).unwrap();

        // audio events, processed results of the buffer, using an independent ring to provide some
        // buffering, synchronized communication, and back pressure support.
        let ae_ring = ringbuf::HeapRb::new(3);
        let (mut ae_tx, ae_rx) = ae_ring.split();

        // NEXT Package this into a node.  The node will store a thread handle.  The node is used to
        // yield inputs to the visual node in draw_frame.  Treat each output as some independent
        // transformation so that we may begin creating the kinds of tension that our later render graph
        // architecture will have to solve.
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
                    let right_excess =
                        (right_sum - (right_noise * 1.3)) / right_noise.max(0.000001);

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
                        match ae_tx.try_push((
                            left_max,
                            right_max,
                            left_fast_accum,
                            right_fast_accum,
                        )) {
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

        Ok(Self {
            hue: rand::random::<f32>(),
            value: 0.0,
            handle: audio_thread,
            context,
            audio_events: ae_rx,
        })
    }

    pub fn process(&mut self) -> AudioColors {
        // NEXT extract this audio event -> color stream as nodes
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

        // Extract audio -> color stream
        let tweaked = self.value * 0.02 + 0.3;
        let value = tweaked.clamp(0.0, 1.0);
        let hsv: palette::Hsv = palette::Hsv::new_srgb(self.hue * 360.0, 1.0, value);
        let rgb: palette::Srgb<f32> = palette::Srgb::from_color_unclamped(hsv);

        // XXX Transitioning the image to get ready for drawing performs the clear, but the clear
        // color selection on the output is a node behavior.  This creates some coupling between the
        // node and target that either requires the node to provide the clear color early or to
        // perform the entire image layout transition, which is not bad, but adds a function call to
        // each node.  We can enforce the correct behavior by passing the untransitioned target and
        // then transitioning it with a clear color as an argument.
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

        AudioColors {
            clear,
            color: rgb,
            scale,
        }
    }

    pub fn destroy(self) {
        // Note, dropping the rx will tell the tx thread to break
        drop(self.audio_events);
        self.handle.join().unwrap()
    }
}
