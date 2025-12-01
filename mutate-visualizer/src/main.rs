// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
mod assets;

use std::ffi::{c_void, CStr, CString};

use ash::khr::xlib_surface;
use ash::{vk, Entry};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

struct App {
    queue: Option<vk::Queue>,
    queue_family_index: u32,
    command_buffers: Vec<vk::CommandBuffer>,
    command_pool: Option<vk::CommandPool>,

    framebuffers: Vec<vk::Framebuffer>,
    pipelines: Option<Vec<vk::Pipeline>>, // XXX Empty Vec instead of optional Vec
    pipeline_layout: Option<vk::PipelineLayout>,

    image_available_semaphore: Option<vk::Semaphore>,
    in_flight_fence: Option<vk::Fence>,
    render_finished_semaphore: Option<vk::Semaphore>,

    device: Option<ash::Device>,
    entry: Option<ash::Entry>,
    instance: Option<ash::Instance>,
    physical_device: Option<vk::PhysicalDevice>,

    surface: Option<vk::SurfaceKHR>,
    surface_loader: Option<ash::khr::surface::Instance>,
    swapchain: Option<vk::SwapchainKHR>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_images: Vec<vk::Image>,
    swapchain_loader: Option<ash::khr::swapchain::Device>,
    window: Option<Window>,
}

impl App {
    fn draw_frame(&mut self) {
        let device = self.device.as_ref().unwrap();
        let queue = *self.queue.as_ref().unwrap();
        let swapchain = self.swapchain.unwrap();
        let swapchain_loader = self.swapchain_loader.as_ref().unwrap();

        let image_available = self.image_available_semaphore.unwrap();
        let render_finished = self.render_finished_semaphore.unwrap();
        let in_flight = self.in_flight_fence.unwrap();

        unsafe {
            device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("wait_for_fences failed");

            device
                .reset_fences(&[in_flight])
                .expect("reset_fences failed");
        }

        let (image_index, _) = unsafe {
            swapchain_loader
                .acquire_next_image(swapchain, std::u64::MAX, image_available, vk::Fence::null())
                .expect("Failed to acquire next image")
        };

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

        self.record_command_buffer(image_index);

        // Which command buffer to submit.
        let cmd = self.command_buffers[image_index as usize];

        let cmd_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: cmd,
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
                .queue_submit2(queue, &[submit], in_flight)
                .expect("queue_submit2 failed");
        }

        let present_wait = [render_finished];
        let swapchains = [swapchain];
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
            swapchain_loader
                .queue_present(queue, &present_info)
                .expect("queue_present failed");
        }
    }

    fn record_command_buffer(&self, image_index: u32) {
        let device = self.device.as_ref().unwrap();
        let cmd = self.command_buffers[image_index as usize];
        let image = self.swapchain_images[image_index as usize];
        let view = self.swapchain_image_views[image_index as usize];
        let extent = unsafe {
            let caps = self
                .surface_loader
                .as_ref()
                .unwrap()
                .get_physical_device_surface_capabilities(
                    self.physical_device.unwrap(),
                    self.surface.unwrap(),
                )
                .unwrap();
            caps.current_extent
        };

        // reset CB
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("reset_command_buffer failed");

            // begin CB
            let begin = vk::CommandBufferBeginInfo::default();
            device
                .begin_command_buffer(cmd, &begin)
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
            image,
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

        unsafe { device.cmd_pipeline_barrier2(cmd, &dep_info) };

        let clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 1.0, 0.1, 1.0],
            },
        };

        let color_attachment = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: view,
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
                extent,
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment,
            ..Default::default()
        };

        unsafe { device.cmd_begin_rendering(cmd, &render_info) };

        let pipeline = self.pipelines.as_ref().unwrap()[0];
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
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
            extent,
        };

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
        }

        // Naive triangle draw
        unsafe { device.cmd_draw(cmd, 3, 1, 0, 0) };

        unsafe { device.cmd_end_rendering(cmd) };

        let barrier2 = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
            old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            dst_access_mask: vk::AccessFlags2::empty(),
            image,
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

        unsafe { device.cmd_pipeline_barrier2(cmd, &dep2) };

        unsafe {
            device
                .end_command_buffer(cmd)
                .expect("end_command_buffer failed")
        };
    }
}

static VALIDATION_LAYER: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create a window using default attributes
        let attrs = Window::default_attributes().with_title("µTate"); // customizing attribute
        let window = event_loop
            .create_window(attrs)
            .expect("Failed to create window");
        let entry = unsafe { Entry::load().expect("failed to load Vulkan library") };
        let available_exts = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")
        };

        assert!(
            available_exts.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == ash::vk::KHR_XLIB_SURFACE_NAME
            }),
            "Only xlib is currently supported"
        );

        let required_exts = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),
            ash::vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
            // XXX CLI switch gate
            ash::vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        let validation_layers = [VALIDATION_LAYER.as_ptr()];

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: required_exts.len() as u32,
            pp_enabled_extension_names: required_exts.as_ptr(),
            enabled_layer_count: validation_layers.len() as u32,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };
        let xlib_surface_loader = xlib_surface::Instance::new(&entry, &instance);
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        let win_handle = window.window_handle().unwrap().as_raw();
        let xlib_window_handle = match win_handle {
            RawWindowHandle::Xlib(handle) => handle,
            _ => panic!("Only Xlib supported!"),
        };
        let xlib_window = xlib_window_handle.window;

        let display_handle = window.display_handle().unwrap().as_raw();
        let xlib_display = match display_handle {
            RawDisplayHandle::Xlib(handle) => handle,
            _ => panic!("Only Xlib supported!"),
        };

        let xlib_create_info = vk::XlibSurfaceCreateInfoKHR {
            s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
            window: xlib_window.into(),
            dpy: xlib_display.display.unwrap().as_ptr(),
            ..Default::default()
        };

        let surface = unsafe { xlib_surface_loader.create_xlib_surface(&xlib_create_info, None) }
            .expect("Failed to create surface");

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("No Vulkan devices")
        };
        let physical_device = physical_devices[0];

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find_map(|(index, q)| {
                    if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        Some(index as u32)
                    } else {
                        None
                    }
                })
                .expect("No graphics queue family found")
        };

        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let device_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            ash::vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
            ash::vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),
            ash::vk::KHR_DYNAMIC_RENDERING_NAME.as_ptr(),
            ash::vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_BUFFER_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_INDEXING_NAME.as_ptr(),
            ash::vk::KHR_PIPELINE_LIBRARY_NAME.as_ptr(),
            ash::vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr(),
            // ROLL holding off on this until other hardware vendors have supporting drivers
            // ash::vk::EXT_SHADER_OBJECT_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE1_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE2_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE3_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE4_NAME.as_ptr(),
        ];

        // Enable synchronization2 and dynamic rendering features:
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default();
        sync2_features.synchronization2 = vk::TRUE;

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        dynamic_rendering_features.dynamic_rendering = vk::TRUE;

        // Query base features if you want existing supported features - optional but good practice:
        let mut features2 = vk::PhysicalDeviceFeatures2::default();
        // If you need to set specific core features, populate features2.features here.
        // Chain the extension feature structs via p_next:
        features2.p_next = &mut sync2_features as *mut _ as *mut c_void;
        sync2_features.p_next = &mut dynamic_rendering_features as *mut _ as *mut c_void;

        let mut device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: queue_info.as_ptr(),
            pp_enabled_extension_names: device_extensions.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            ..Default::default()
        };
        device_info.p_next = &mut features2 as *mut _ as *mut c_void;

        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };
        let surface_format = formats[0];

        let composite_alpha = if surface_caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else if surface_caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
        {
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        } else if surface_caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
        {
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };

        let supported = unsafe {
            surface_loader
                .get_physical_device_surface_support(physical_device, queue_family_index, surface)
                .unwrap()
        };
        assert!(supported, "Physical device must support this surface!");

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let swapchain_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: 3, // double buffered
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: surface_caps.current_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface_caps.current_transform,
            composite_alpha: composite_alpha,
            present_mode: vk::PresentModeKHR::FIFO,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .unwrap()
        };
        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        // Create image views
        let image_views: Vec<_> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface_format.format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&view_info, None).unwrap() }
            })
            .collect();

        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let fence = unsafe { device.create_fence(&fence_info, None).unwrap() };

        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        let image_available_semaphore =
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };
        let render_finished_semaphore =
            unsafe { device.create_semaphore(&semaphore_info, None).unwrap() };

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

        // allocate one command buffer per swapchain image (use `images.len()` — not self.swapchain_images)
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: images.len() as u32,
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

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };
        let swapchain_format = surface_format.format;

        let color_formats = [swapchain_format];
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

        // Store all
        self.entry = Some(entry);
        self.instance = Some(instance);
        self.surface_loader = Some(surface_loader);
        self.surface = Some(surface);
        self.physical_device = Some(physical_device);
        self.device = Some(device);
        self.queue = Some(queue);
        self.queue_family_index = queue_family_index;
        self.window = Some(window);

        self.swapchain_loader = Some(swapchain_loader);
        self.swapchain = Some(swapchain);
        self.swapchain_images = images;
        self.swapchain_image_views = image_views;

        self.in_flight_fence = Some(fence);
        self.image_available_semaphore = Some(image_available_semaphore);
        self.render_finished_semaphore = Some(render_finished_semaphore);

        self.command_pool = Some(command_pool);
        self.command_buffers = buffers;

        self.pipelines = Some(pipelines);
        self.pipeline_layout = Some(pipeline_layout);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                self.draw_frame();
            }
            WindowEvent::CloseRequested => unsafe {
                if let Some(device) = &self.device {
                    device.device_wait_idle().unwrap();

                    for fb in &self.framebuffers {
                        device.destroy_framebuffer(*fb, None);
                    }

                    for view in &self.swapchain_image_views {
                        device.destroy_image_view(*view, None);
                    }

                    self.pipelines.as_ref().unwrap().iter().for_each(|p| {
                        device.destroy_pipeline(*p, None);
                    });

                    if let Some(layout) = self.pipeline_layout {
                        device.destroy_pipeline_layout(layout, None);
                    }

                    if let Some(loader) = &self.swapchain_loader {
                        if let Some(swapchain) = self.swapchain {
                            loader.destroy_swapchain(swapchain, None);
                        }
                    }
                    if let Some(surface_loader) = &self.surface_loader {
                        if let Some(surface) = self.surface {
                            surface_loader.destroy_surface(surface, None);
                        }
                    }

                    self.image_available_semaphore.map(|s| {
                        device.destroy_semaphore(s, None);
                    });
                    self.render_finished_semaphore.map(|s| {
                        device.destroy_semaphore(s, None);
                    });
                    self.in_flight_fence.map(|f| {
                        device.destroy_fence(f, None);
                    });
                    self.command_pool.map(|p| {
                        device.destroy_command_pool(p, None);
                    });

                    device.destroy_device(None);
                }
                if let Some(instance) = &self.instance {
                    instance.destroy_instance(None);
                }
                event_loop.exit();
            },
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        command_buffers: Vec::new(),
        command_pool: None,
        device: None,
        entry: None,
        framebuffers: Vec::new(),
        pipelines: None,
        pipeline_layout: None,
        image_available_semaphore: None,
        in_flight_fence: None,
        instance: None,
        physical_device: None,
        queue: None,
        queue_family_index: 0,
        render_finished_semaphore: None,
        surface: None,
        surface_loader: None,
        swapchain: None,
        swapchain_image_views: Vec::new(),
        swapchain_images: Vec::new(),
        swapchain_loader: None,
        window: None,
    };
    event_loop.run_app(&mut app).unwrap();
}
