// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// A swapchain exists when we are presenting to a Surface.  We can use it as a render target.  Not
// all render targets need presentation, but a swapchain does.  Aligning the fields and structs with
// this abstraction is underway.

// NEXT enable options for variable and fixed frame rate rendering, likely using MAILBOX for
// variable and FIFO for fixed rate.  This requires looking at the actual swapchain image count
// before allocating per-image resources like command buffers.

// XXX move into Vulkan!
use std::slice;

use ash::khr::present_wait;
use ash::vk;
use smallvec::SmallVec;

use mutate_lib::vulkan::dispatch::sync;
use mutate_lib::{self as utate, prelude::*};

// The delightfully unsound thing about this abstraction for now is that CommandBuffers are freed
// whenever the pool is freed, so we don't track them, and everyone just needs to be adults for a
// little while. ☔  Maybe forever.  If you're an adult too long, they call you cowboy.  🤠
pub struct CommandPool {
    pool: vk::CommandPool,
    // XXX Store the more useful queue object when its ready
    queue: u32,
    outstanding: SmallVec<vk::CommandBuffer, 8>,
    recycled: SmallVec<vk::CommandBuffer, 8>,
}

impl CommandPool {
    // MAYBE temporary lifetimes for making things... for better chaining and less context
    // proliferation..  This is a good example.
    // XXX new queue
    pub fn new(device_context: &DeviceContext, queue_family_index: u32) -> Self {
        let command_pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(queue_family_index);

        let pool = unsafe {
            device_context
                .device()
                .create_command_pool(&command_pool_ci, None)
                .unwrap()
        };

        Self {
            pool,
            queue: queue_family_index,
            outstanding: SmallVec::new(),
            recycled: SmallVec::new(),
        }
    }

    // XXX typed command buffer
    pub fn buffer(&mut self, device_context: &DeviceContext) -> vk::CommandBuffer {
        let buf = if let Some(buf) = self.recycled.pop() {
            buf
        } else {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.pool,
                command_buffer_count: 1,
                ..Default::default()
            };
            unsafe {
                device_context
                    .device()
                    .allocate_command_buffers(&alloc_info)
                    .unwrap()[0]
            }
        };

        self.outstanding.push(buf);
        buf
    }

    pub fn buffers(
        &mut self,
        device_context: &DeviceContext,
        count: u32,
    ) -> Vec<vk::CommandBuffer> {
        let count = count as usize;
        let from_recycled = self.recycled.len().min(count);
        let need_alloc = count - from_recycled;

        let mut result: Vec<vk::CommandBuffer> = self
            .recycled
            .drain(self.recycled.len() - from_recycled..)
            .collect();

        if need_alloc > 0 {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.pool,
                command_buffer_count: need_alloc as u32,
                ..Default::default()
            };
            let fresh = unsafe {
                device_context
                    .device()
                    .allocate_command_buffers(&alloc_info)
                    .unwrap()
            };
            result.extend_from_slice(&fresh);
        }

        self.outstanding.extend_from_slice(&result);
        result
    }

    // Safety (lol): The contract is that all buffers are dead when you call reset.  Nice chat.
    pub fn reset(&mut self, device_context: &DeviceContext) {
        unsafe {
            device_context
                .device()
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        self.recycled.extend(self.outstanding.drain(..));
    }

    pub fn destroy(&self, device_context: &DeviceContext) {
        unsafe {
            // When a pool is destroyed, all command buffers allocated from the pool are freed.
            device_context
                .device()
                .destroy_command_pool(self.pool, None);
        }
    }
}

pub struct DrawSync {
    pub render_finished: vk::Semaphore,
    pub image_available: vk::Semaphore,
    pub image_index: u32,
}

// XXX command buffers things mixed with images.. this is not right at all.  Perhaps by the time we
// hit render graph, dependencies and inputs / outputs are
pub struct DrawTarget {
    pub image: vk::Image,
    pub command_buffer: vk::CommandBuffer,
}

/// A package deal!
pub struct SwapchainContext {
    swapchain: vk::SwapchainKHR,
    loader: ash::khr::swapchain::Device,
    image_views: SmallVec<vk::ImageView, 4>,
    images: SmallVec<vk::Image, 4>,
    frames: usize,
    frame_index: usize,

    render_finished_semaphores: [vk::Semaphore; 4],
    image_available_semaphores: [vk::Semaphore; 4],
}

impl SwapchainContext {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext { entry, instance } = &vk_context;
        // XXX device_context method?
        let loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &device_context.device());

        // XXX the images are not actually being counted
        let swapchain_ci = vk::SwapchainCreateInfoKHR {
            surface: surface.as_raw(),
            min_image_count: 3,
            image_format: surface.format.format,
            image_color_space: surface.format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface.pre_transform,
            composite_alpha: surface.composite_alpha,
            present_mode: surface.present_mode,
            clipped: vk::TRUE,
            flags: vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
            ..Default::default()
        };

        let swapchain = unsafe { loader.create_swapchain(&swapchain_ci, None).unwrap() };
        let images = unsafe { loader.get_swapchain_images(swapchain).unwrap() };
        let frames = images.len();

        let image_views: SmallVec<_, 4> = images
            .iter()
            .map(|&image| {
                let view_ci = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface.format.format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                unsafe {
                    device_context
                        .device
                        .create_image_view(&view_ci, None)
                        .unwrap()
                }
            })
            .collect();

        // Aim for max images. With 3, we don't have to reallocate.
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let make_semaphore = |_| unsafe {
            device_context
                .device()
                .create_semaphore(&semaphore_ci, None)
                .unwrap()
        };
        let image_available_semaphores = std::array::from_fn(make_semaphore);
        let render_finished_semaphores = std::array::from_fn(make_semaphore);

        Self {
            image_available_semaphores,
            render_finished_semaphores,
            loader,
            images: images.into(),
            image_views: image_views,
            swapchain,
            frames,
            frame_index: 0,
        }
    }

    pub fn recreate(
        &mut self,
        device_context: &DeviceContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) {
        let device = device_context.device();

        // Destroy old image views — images themselves are owned by the swapchain
        unsafe {
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
        }

        let swapchain_ci = vk::SwapchainCreateInfoKHR {
            surface: surface.as_raw(),
            min_image_count: self.frames as u32,
            image_format: surface.format.format,
            image_color_space: surface.format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: surface.pre_transform,
            composite_alpha: surface.composite_alpha,
            present_mode: surface.present_mode,
            clipped: vk::TRUE,
            flags: vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT,
            old_swapchain: self.swapchain,
            ..Default::default()
        };

        let new_swapchain = unsafe { self.loader.create_swapchain(&swapchain_ci, None).unwrap() };

        // Old swapchain is retired — safe to destroy now that new one is created
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }

        self.swapchain = new_swapchain;

        self.images = unsafe {
            self.loader
                .get_swapchain_images(self.swapchain)
                .unwrap()
                .into()
        };

        self.image_views = self
            .images
            .iter()
            .map(|&image| {
                let view_ci = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: surface.format.format,
                    components: vk::ComponentMapping::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&view_ci, None).unwrap() }
            })
            .collect();

        // XXX review this
        // Reconcile semaphore count if the driver gave us a different image count
        let new_frames = self.images.len();
        if new_frames != self.frames {
            // let semaphore_ci = vk::SemaphoreCreateInfo::default();
            // let make = |_| unsafe { device.create_semaphore(&semaphore_ci, None).unwrap() };

            // self.image_available_semaphores
            //     .resize_with(new_frames, make);
            // self.render_finished_semaphores
            //     .resize_with(new_frames, make);
            self.frames = new_frames;
            self.frame_index = self.frame_index.min(new_frames - 1);
        }
    }

    // XXX a horrible function?
    pub fn present(&self, queue: vk::Queue, present_info: &vk::PresentInfoKHR) {
        unsafe {
            match self.loader.queue_present(queue, present_info) {
                Ok(_) => {
                    // MAYBE How to interpret false?
                }
                Err(result) => eprintln!("presentation error: {:?}", result),
            };
        }
    }

    pub fn destroy(&self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            for view in &self.image_views {
                device.destroy_image_view(*view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
            self.image_available_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
            self.render_finished_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
        }
    }

    /// XXX mark chain dirty if this fails
    pub fn acquire(&mut self) -> (vk::Image, DrawSync) {
        let idx = self.frame_index as usize;
        let image_available = self.image_available_semaphores[idx];
        let render_finished = self.render_finished_semaphores[idx];
        self.frame_index = (idx + 1) % self.frames;

        let (image_index, _) = unsafe {
            self.loader
                .acquire_next_image(
                    self.swapchain,
                    1_000_000_000, // 1000ms
                    image_available,
                    vk::Fence::null(),
                )
                .unwrap()
        };

        let image = self.images[image_index as usize];
        let image_view = self.image_views[image_index as usize];

        let sync = DrawSync {
            image_available,
            render_finished,
            image_index,
        };

        (image, sync)
    }
}

pub struct SurfacePresent {
    swapchain: SwapchainContext,

    frame_counter: u64,
    in_flight: vk::Semaphore,

    /// Present wait measurement
    present: sync::PresentConsumer,

    // Just enough for front & back frame.
    pools: [CommandPool; 2],
    recording_index: u32,
}

impl SurfacePresent {
    // TODO give swapchain during construction?
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext { entry, instance } = &vk_context;

        let swapchain = SwapchainContext::new(device_context, vk_context, surface, extent);

        let mut type_ci = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };
        let semaphore_ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_ci);
        let in_flight = unsafe {
            device_context
                .device()
                .create_semaphore(&semaphore_ci, None)
                .unwrap()
        };

        let queue_family_index = device_context.queues.graphics_family_index;
        let pools: [CommandPool; 2] =
            std::array::from_fn(|_| CommandPool::new(device_context, queue_family_index));

        let present = sync::PresentConsumer::new(vk_context, device_context, swapchain.swapchain)
            .expect("Could not start up the present wait gizmo");

        Self {
            frame_counter: 0,
            in_flight,
            present,

            recording_index: 0,
            pools,

            swapchain,
        }
    }

    pub fn destroy(&self, context: &DeviceContext) {
        let device = &context.device();
        unsafe {
            self.swapchain.destroy(context);
            for pool in &self.pools {
                pool.destroy(context);
            }
            device.destroy_semaphore(self.in_flight, None);
        }
    }

    pub fn recreate_images(
        &mut self,
        surface: &VkSurface,
        size: vk::Extent2D,
        context: &DeviceContext,
    ) {
        let device = &context.device;
        let physical_device = context.physical_device;
        // XXX replace with semapahore wait on last frame in flight
        unsafe {
            device.device_wait_idle().unwrap();
        }

        self.swapchain.recreate(context, surface, size);
        self.present
            .notify_swapchain_recreation(self.swapchain.swapchain);
    }

    pub fn draw_wait(&mut self, device_context: &DeviceContext) {
        let wait_value = self.frame_counter;
        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(slice::from_ref(&self.in_flight))
            .values(slice::from_ref(&wait_value));
        unsafe {
            device_context
                .device()
                .wait_semaphores(&wait_info, u64::MAX)
                .unwrap();
        }
    }

    /// Return a hot command buffer, image, and associated information used for drawing.
    pub fn render_target(
        &mut self,
        context: &DeviceContext,
        clear: vk::ClearValue,
    ) -> (DrawSync, DrawTarget) {
        let device = &context.device();

        let (image, sync) = self.swapchain.acquire();

        let pool = &mut self.pools[self.recording_index as usize];
        pool.reset(&context);
        let buffer = pool.buffer(&context);
        self.recording_index ^= 1;

        // XXX maybe do this on creation?
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(buffer, &begin).unwrap();
        }

        // let barrier = vk::ImageMemoryBarrier2 {
        //     src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
        //     dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        //     old_layout: vk::ImageLayout::UNDEFINED,
        //     new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        //     src_access_mask: vk::AccessFlags2::empty(),
        //     dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        //     image,
        //     subresource_range: vk::ImageSubresourceRange {
        //         aspect_mask: vk::ImageAspectFlags::COLOR,
        //         level_count: 1,
        //         layer_count: 1,
        //         ..Default::default()
        //     },
        //     ..Default::default()
        // };

        // let dep_info = vk::DependencyInfo {
        //     image_memory_barrier_count: 1,
        //     p_image_memory_barriers: &barrier,
        //     ..Default::default()
        // };

        // unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        // let color_attachment = vk::RenderingAttachmentInfo {
        //     image_view: image_view,
        //     image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        //     load_op: vk::AttachmentLoadOp::CLEAR,
        //     store_op: vk::AttachmentStoreOp::STORE,
        //     clear_value: clear,
        //     ..Default::default()
        // };

        // let render_info = vk::RenderingInfo {
        //     render_area: vk::Rect2D {
        //         offset: vk::Offset2D { x: 0, y: 0 },
        //         extent: size,
        //     },
        //     layer_count: 1,
        //     color_attachment_count: 1,
        //     p_color_attachments: &color_attachment,
        //     ..Default::default()
        // };

        // unsafe { device.cmd_begin_rendering(command_buffer, &render_info) };

        let target = DrawTarget {
            image,
            command_buffer: buffer,
        };

        (sync, target)
    }

    /// Sync presentation image transform and present
    pub fn post_draw(&mut self, context: &DeviceContext, sync: &DrawSync, target: &DrawTarget) {
        let device = context.device();

        // XXX Unique for graphics
        // unsafe { device.cmd_end_rendering(target.command_buffer) };

        // let barrier2 = vk::ImageMemoryBarrier2 {
        //     src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        //     src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        //     dst_stage_mask: vk::PipelineStageFlags2::NONE,
        //     dst_access_mask: vk::AccessFlags2::empty(),

        //     // XXX pre-present barrier was incorrect
        //     // old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        //     old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        //     new_layout: vk::ImageLayout::PRESENT_SRC_KHR,

        //     image: target.image,
        //     subresource_range: vk::ImageSubresourceRange {
        //         aspect_mask: vk::ImageAspectFlags::COLOR,
        //         level_count: 1,
        //         layer_count: 1,
        //         ..Default::default()
        //     },
        //     ..Default::default()
        // };

        // let dep2 = vk::DependencyInfo {
        //     image_memory_barrier_count: 1,
        //     p_image_memory_barriers: &barrier2,
        //     ..Default::default()
        // };

        // unsafe {
        //     vk_context
        //         .device
        //         .cmd_pipeline_barrier2(target.command_buffer, &dep2)
        // };

        unsafe {
            context
                .device
                .end_command_buffer(target.command_buffer)
                .unwrap()
        };

        let wait_info = vk::SemaphoreSubmitInfo {
            semaphore: sync.image_available,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        };

        let next_value = self.frame_counter + 1;
        self.frame_counter = next_value;
        let in_flight = vk::SemaphoreSubmitInfo::default()
            .semaphore(self.in_flight)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .value(next_value);

        let signal_infos = [
            vk::SemaphoreSubmitInfo {
                semaphore: sync.render_finished,
                stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
                ..Default::default()
            },
            in_flight,
        ];

        let cb_info = vk::CommandBufferSubmitInfo {
            command_buffer: target.command_buffer,
            ..Default::default()
        };

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(slice::from_ref(&wait_info))
            .signal_semaphore_infos(&signal_infos)
            .command_buffer_infos(slice::from_ref(&cb_info));

        let queue = &context.queues.graphics_queue();
        unsafe {
            context
                .device()
                .queue_submit2(*queue, &[submit], vk::Fence::null())
                .unwrap();
        }
    }

    // XXX this function demonstrates that command buffer presentation kind of re-couples at
    // presentation for swapchain stuff.
    pub fn present(&mut self, device_context: &DeviceContext, sync: &DrawSync) {
        let queue = &device_context.queues.graphics_queue();
        match self.present.read_last_present() {
            Some(last) => {
                // if last.last_window != std::time::Duration::MAX {
                //     println!("present duration: {:8.0}", last.last_window.as_micros())
                // }
            }
            None => {}
        }
        let next_id = self.present.next_present_id();
        let mut present_id = vk::PresentIdKHR::default().present_ids(slice::from_ref(&next_id));
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&sync.render_finished))
            .swapchains(slice::from_ref(&self.swapchain.swapchain))
            .image_indices(slice::from_ref(&sync.image_index))
            .push_next(&mut present_id);

        self.swapchain.present(*queue, &present_info);
        self.present.notify_waiter();
    }
}
