// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Swapchain
//!
//! This module abstracts over swapchain management and behavior.  Window resizing requires image
//! recreation.  We don't control exactly how many swapchain images exist or which one is acquired.
//! Each image we use need external synchronization primitives.  All of that juggling has been
//! brought under one structure, the [`Swapchain`].
//!
//! ## Usage
//!
//! While you can manually acquire images and manage a command pool, see the `crate::present` module
//! for the `Render` and its `GraphicsPresent` and `ComputePresent`
//!
//! ## Recreation
//!
//! Whenever windows are resized, the swapchain needs to be recreated.  We re-use the old swapchain
//! to make it easy on the driver to handle the memory.  Images are lazily allocated to reduce the
//! load during recreation.
//!
//! The signal for recreation can be a window resize event or an ash result of SUBOPTIMAL or
//! OUT_OF_DATE.  More serious errors indicate a need to recreate the logical device instead.
//!
//! ## Surrounding Context
//!
//! A swapchain is a lazy allocation of some images that can be scanned out to a surface, usually
//! for a physical display.  We obtain images, some command buffers draw things on them, and then we hand
//! control of the images back to a compositor or DRM for physical scan-out to some screen.
//!
//! The [`VkSurface`] knows the format we select.  This is important for the renderer to ensure that
//! the output of drawing is either compatible with or can be copied to the swapchain image for
//! presentation.  The [`AcquiredImage`] carries along format, extent, and other information for
//! renderers to write correctly to the output.
//!
//! ## Index Behavior
//!
//! Due to the index rotation and array length being out of our control, the only way to indexes it
//! straight is to keep one of each thing per *potential* image in flight (where in flight duration
//! is up to a compositor usually).
//!
//! To simplify having enough space for handles, we just set up four slots and then only fill the
//! slots after the swapchain creation tells us how many images there actually are, most of the
//! time, only three.
//!
//! In actual practice, our acquire-render-present loop primitives are designed around just-in-time
//! render support and simultaneous machine learning workloads, all of which favor only one image
//! being drawn to per-frame.  This behavior can *nearly* be satisfied by a front and back buffer
//! only.  Compare with render-as-fast-as-you-can, which obviously uses all images.  Even so, the
//! common use of three images prevents tiny bubbles where the compositor is slow to give us control
//! of an image following its present phase before we can begin drawing to it (which usually is not
//! *right* after the present phase anyway).

use ash::vk::{self, Handle};
use smallvec::SmallVec;

use super::surface::VkSurface;

use crate::internal::*;

/// An image from the swapchain and associated format, extent, and synchronization data necessary
/// for presentation.
// NEXT see presentation render target, which may be a trait for several types to implement,
// enabling renderers to draw via common interface.
pub struct AcquiredImage {
    /// Swapchain signals when image is ready for use (start rendering).
    pub image_available: BinaryWait,
    /// User signals when compositor may present image (after rendering).
    pub present_ready: BinarySignal,

    /// Necessary to begin any rendering
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    pub extent: vk::Extent2D,

    /// The index provided by the swapchain during acquisition.  Used in all presentation to tell
    /// the other side of the swapchain which image we are requesting to present.
    pub swapchain_image_index: u32,
}

/// An initialized swapchain with necessary synchronization and accounting data included.
// Let's just say no sane swapchain is coming back with four images.  With four slots, we always
// have enough space for any size of swapchain.  We only take up a few 64bit pointers that won't
// pack or load much better for three elements anyway.  In return, we save a lot of complication.
// DEBT rename to just Swapchain and replace any ash types with vk prefix.
pub struct SwapchainContext {
    raw: vk::SwapchainKHR,
    loader: ash::khr::swapchain::Device,
    image_views: SmallVec<vk::ImageView, 4>,
    images: SmallVec<vk::Image, 4>,
    frames: usize,
    /// During acquisition, we must provide a binary semaphore for the compositor to signal when the image
    /// is ready.  Since we don't yet know the index of the image yet, we use our own index to cycle
    /// through semaphores, ensuring no re-use on different images.
    frame_index: usize,
    // note about choice of 4 in module docs
    image_available_semaphores: [BinarySemaphore; 4],
    present_ready_semaphores: [BinarySemaphore; 4],
    extent: vk::Extent2D,
    recreation_required: bool,
}

impl SwapchainContext {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Self {
        let VkContext {
            entry, instance, ..
        } = &vk_context;
        let loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &device_context.device());

        // somewhat duplicate with recreation.
        let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.as_raw())
            .min_image_count(surface.swapchain_image_count)
            .image_format(surface.format.format)
            .image_color_space(surface.format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface.pre_transform)
            .composite_alpha(surface.composite_alpha)
            .present_mode(surface.present_mode)
            .clipped(true)
            .flags(vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT);
        };

        let swapchain = unsafe { loader.create_swapchain(&swapchain_ci, None).unwrap() };
        let images = unsafe { loader.get_swapchain_images(swapchain).unwrap() };
        let frames = images.len();
        let image_views =
            create_image_views(&device_context.device(), &images, surface.format.format);

        let image_available_semaphores =
            std::array::from_fn(|_| device_context.make_binary_semaphore().unwrap());
        let present_ready_semaphores =
            std::array::from_fn(|_| device_context.make_binary_semaphore().unwrap());
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
        unsafe {
            device_context.device();
        }

        // XXX The current window-swap-surface thing (still in visualizer?) doesn't know how to tell
        // use when we can safely recreate images.  The present-wait must be poll-looped to
        // determine if we have no images in flight.  The plumbing through the external present-wait
        // structure means that only the window thing (which also gets resize events) can tell us
        // when it's clear to recreate.  Until then....... device_wait_idle 🥉
        unsafe { device.device_wait_idle() };

        // Destroy old image views — images themselves are owned by the swapchain
        unsafe {
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
        }

        // XXX recreation data is out of date and should be refreshed up in the Window-Swap-Surface
        // stuff.
        let swapchain_ci = vk::SwapchainCreateInfoKHR {
            surface: surface.as_raw(),
            min_image_count: surface.swapchain_image_count,
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
        self.image_views = create_image_views(
            &device_context.device(),
            &self.images,
            surface.format.format,
        );

        let new_frames = self.images.len();
        if new_frames != self.frames {
            self.frames = new_frames;
            // LIES while this avoids a degenerate index that would crash, it might attempt to use
            // an image in flight unless acquire is properly called.  The semaphore wait before
            // swapchain recreation is what *actually* protects us.
            self.frame_index = self.frame_index.min(new_frames - 1);
        }
    }

    // The caller is attaching extra present info and we don't have a good way to let everyone add
    // their data to the chain, so it's being remixed externally before passed in as `present_info`.
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
                s.destroy(context);
            });
            self.present_ready_semaphores.iter().for_each(|s| {
                s.destroy(context);
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
                    image_available.as_raw(),
                    vk::Fence::null(),
                )
                .unwrap() // XXX will error in ways we should catch
                          // XXX where all will these occur?
                          // VK_ERROR_OUT_OF_DATE_KHR
                          // VK_SUBOPTIMAL_KHR
        };

        let image = self.images[swapchain_image_index as usize];

        AcquiredImage {
            image_available: image_available.wait(),
            swapchain_image_index,
            image,
            present_ready: present_ready.signal(),
        }
    }
}

// Creation and re-creation both make image views.
fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> SmallVec<vk::ImageView, 4> {
    images
        .iter()
        .map(|&image| {
            let view_ci = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
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
        .collect()
}
