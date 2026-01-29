// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::CString;

use ash::vk;

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
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert_shader_module,
                p_name: entry_vert.as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
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

        // UPSTREAM shouldn't the transitive Default take care of this?
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            ..Default::default()
        };

        let color_blend = vk::PipelineColorBlendStateCreateInfo {
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
            view_mask: 0,
            color_attachment_count: 1,
            p_color_attachment_formats: color_formats.as_ptr(),
            ..Default::default()
        };

        let pipeline_ci = vk::GraphicsPipelineCreateInfo {
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
