// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// A swapchain exists when we are presenting to a Surface.  We can use it as a render target.  Not
// all render targets need presentation, but a swapchain does.  Aligning the fields and structs with
// this abstraction is underway.

use ash::vk;
use winit::window::Window;

use crate::render_target::RenderTarget;
use crate::vk_context::VkContext;

pub struct SwapChain {
    pub frames: usize,
    pub frame_index: usize,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_loader: ash::khr::swapchain::Device,
}

impl SwapChain {
    pub fn new(vk_context: &VkContext, rt: &RenderTarget) -> Self {
        // &surface, &surface_caps, surface_format, swapchain_size
        let surface = &rt.surface;
        let surface_caps = &rt.surface_caps;
        let surface_format = &rt.surface_format;
        let extent = window_size(&rt.window);

        let composite_alpha = pick_alpha(&surface_caps);

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &vk_context.device);
        let swapchain_info = vk::SwapchainCreateInfoKHR {
            surface: *surface,
            min_image_count: 3, // XXX frame counts
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
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
                unsafe {
                    vk_context
                        .device
                        .create_image_view(&view_info, None)
                        .unwrap()
                }
            })
            .collect();

        let fence_info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let in_flight_fences: Vec<vk::Fence> = (0..3)
            .map(|_| unsafe { vk_context.device.create_fence(&fence_info, None).unwrap() })
            .collect();

        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        // FIXME propagate image counts
        let image_available_semaphores = (0..3)
            .map(|_| unsafe {
                vk_context
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        let render_finished_semaphores = (0..3)
            .map(|_| unsafe {
                vk_context
                    .device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            })
            .collect();

        Self {
            frames: 3,
            frame_index: 0,
            image_available_semaphores,
            in_flight_fences,
            render_finished_semaphores,
            swapchain_extent: extent,
            swapchain,
            swapchain_image_views: image_views,
            swapchain_images: images,
            swapchain_loader,
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            for view in &self.swapchain_image_views {
                device.destroy_image_view(*view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            self.image_available_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
            self.render_finished_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
            self.in_flight_fences.iter().for_each(|f| {
                device.destroy_fence(*f, None);
            });
        }
    }

    pub fn recreate_images(&mut self, vk_context: &VkContext, rt: &RenderTarget) {
        let device = &vk_context.device;
        let physical_device = vk_context.physical_device;

        unsafe {
            device.device_wait_idle().unwrap();
        }

        // partial destruction
        unsafe {
            for view in &self.swapchain_image_views {
                device.destroy_image_view(*view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        // Recreation
        unsafe {
            let surface_caps = rt
                .surface_loader
                .get_physical_device_surface_capabilities(physical_device, rt.surface)
                .unwrap();
            let current_extent = surface_caps.current_extent;
            let extent = if current_extent.width != u32::MAX {
                self.swapchain_extent = current_extent;
                self.swapchain_extent
            } else {
                // FIXME a number of cases this can be wrong.
                window_size(&rt.window)
            };
            let surface = &rt.surface;
            let surface_format = &rt.surface_format;

            let swapchain_info = vk::SwapchainCreateInfoKHR {
                surface: *surface,
                min_image_count: self.frames as u32,
                image_format: surface_format.format,
                image_color_space: surface_format.color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                pre_transform: surface_caps.current_transform,
                composite_alpha: pick_alpha(&surface_caps),
                present_mode: vk::PresentModeKHR::FIFO,
                clipped: vk::TRUE,
                ..Default::default()
            };

            let swapchain = self
                .swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .unwrap();
            let images = self
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .unwrap();

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
                    device.create_image_view(&view_info, None).unwrap()
                })
                .collect();

            self.swapchain = swapchain;
            self.swapchain_images = images;
            self.swapchain_image_views = image_views;
        }
    }

    pub fn render_target(&self, index: usize) -> (vk::Image, vk::ImageView) {
        let image = self.swapchain_images[index];
        let view = self.swapchain_image_views[index];
        (image, view)
    }
}

fn pick_alpha(&surface_caps: &vk::SurfaceCapabilitiesKHR) -> vk::CompositeAlphaFlagsKHR {
    if surface_caps
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
    }
}

fn window_size(window: &Window) -> vk::Extent2D {
    let size = window.inner_size();
    vk::Extent2D {
        width: size.width,
        height: size.height,
    }
}
