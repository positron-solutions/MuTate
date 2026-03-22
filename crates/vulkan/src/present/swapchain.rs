// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Swapchain
//!
//! Drawing on a screen, at least "normal" ones.
//!
//! A swapchain is a lazy allocation of some images that can be scanned out to a surface, usually
//! for a physical display.  We obtain images, some command buffers draw things on them, and we hand
//! them to a compositor or DRM for physical scan-out to some screen.
//!
//! ## Recreation
//!
//! Whenever windows are resized, the swapchain needs to be recreated.  We re-use the old swapchain
//! to make it easy on the driver to handle the memory.  Images are lazily allocated to reduce the
//! pressure during recreation.
//!
//! ## Index Behavior
//!
//! The number of images created is not directly under our control.  During acquisition, the image
//! index we get back is actually not under our control.  Even though we might be strictly double
//! buffering, the compositor may be late.  The semaphore control is not given back in time.
//!
//! The only way to keep it straight is to keep one of each thing per *potential* image in flight
//! (where in flight duration is up to a compositor usually).  The memory for actual images might
//! not be allocated and the memory for semaphores etc is kind of cheap.  To make it all super
//! simple, we just set up four slots for Vulkan handles.  We make semaphores and ImageViews.  Most
//! of the time, only three will be used.  At least nobody ever has to count.

// XXX no error handling for *highly expected* swapchain results like SUBOPTIMAL
// XXX present function still seems tangled with old present module.
// XXX extent handling and surface agreement likely silly

use ash::vk;
use smallvec::SmallVec;

use super::surface::VkSurface;
use crate::context::{DeviceContext, VkContext};

// Acquired images come from the swapchain, which might have to do something obtuse and hand us back
// a bizarre index, such as 2, due to the compositor being late at giving us the image back.
pub struct AcquiredImage {
    /// Used during acquisition.
    pub image_available: vk::Semaphore,
    /// The index provided by the swapchain (which may cycle swapchain images out of order) during
    /// acquisition.
    pub swapchain_image_index: u32,
    pub image: vk::Image,
    pub present_ready: vk::Semaphore,
}

/// A package deal!
///
/// ### Rule of Four
///
/// Let's just say no sane swapchain is coming back with four images.  With four slots, we always
/// have enough space for any size of swapchain.  We only take up a few 64bit pointers that won't
/// pack or load much better for three elements anyway.  In return, we save a lot of complication.
pub struct SwapchainContext {
    /// XXX encapsulate
    pub swapchain: vk::SwapchainKHR,
    loader: ash::khr::swapchain::Device,
    image_views: SmallVec<vk::ImageView, 4>,
    images: SmallVec<vk::Image, 4>,
    frames: usize,
    frame_index: usize,
    image_available_semaphores: [vk::Semaphore; 4],
    present_ready_semaphores: [vk::Semaphore; 4],
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

        let image_available_semaphores = std::array::from_fn(|_| device_context.make_semaphore());
        let present_ready_semaphores = std::array::from_fn(|_| device_context.make_semaphore());
        Self {
            present_ready_semaphores,
            image_available_semaphores,
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

        let new_frames = self.images.len();
        if new_frames != self.frames {
            self.frames = new_frames;
            // LIES while this avoids a degenerate index that would crash, it might attempt to use
            // an image in flight unless acquire is properly called.  The semaphore wait before
            // swapchain recreation is what *actually* protects us.
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
            self.present_ready_semaphores.iter().for_each(|s| {
                device.destroy_semaphore(*s, None);
            });
        }
    }

    /// XXX mark chain dirty if this fails
    pub fn acquire(&mut self) -> AcquiredImage {
        let idx = self.frame_index as usize;
        let image_available = self.image_available_semaphores[idx];
        let present_ready = self.present_ready_semaphores[idx];
        self.frame_index = (idx + 1) % self.frames;

        let (swapchain_image_index, _) = unsafe {
            self.loader
                .acquire_next_image(
                    self.swapchain,
                    100_000_000, // 100ms
                    image_available,
                    vk::Fence::null(),
                )
                .unwrap() // XXX will error in ways we should catch
        };

        let image = self.images[swapchain_image_index as usize];

        AcquiredImage {
            image_available,
            swapchain_image_index,
            image,
            present_ready,
        }
    }
}
