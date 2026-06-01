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
    /// Signaled by presentation when image is completely finished with.  Used for draining the
    /// swapchain, but can also be used by other waiters.
    pub present_finished: Fence,

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
    slots: usize,
    /// During acquisition, we must provide a binary semaphore for the compositor to signal when the image
    /// is ready.  Since we don't yet know the index of the image yet, we use our own index to cycle
    /// through semaphores, ensuring no re-use on different images.
    slot_index: usize,
    // note about choice of 4 in module docs
    image_available_semaphores: [BinarySemaphore; 4],
    present_ready_semaphores: [BinarySemaphore; 4],
    present_finished_fences: [Fence; 4],
    extent: vk::Extent2D,
    recreation_required: bool,
}

impl SwapchainContext {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &VkSurface,
        extent: vk::Extent2D,
    ) -> Result<Self, VulkanError> {
        let VkContext {
            entry, instance, ..
        } = &vk_context;
        let loader =
            ash::khr::swapchain::Device::new(&vk_context.instance, &device_context.device());

        let swapchain_ci = surface.swapchain_ci();

        // XXX failed creation case might not be able to re-create and this has not been resolved yet.
        let swapchain = unsafe {
            loader
                .create_swapchain(&swapchain_ci, None)
                .unwrap_or(vk::SwapchainKHR::null())
        };

        let images = unsafe { loader.get_swapchain_images(swapchain)? };
        // Even if we only have 2 images, we need at least three semaphores to avoid lapping those
        // in-flight.
        let slots = images.len().max(3);
        let image_views = create_image_views(&device_context.device(), &images, surface.format());

        // XXX only make semaphores for slots?  Also.. this panics.
        let image_available_semaphores =
            std::array::from_fn(|_| device_context.make_binary_semaphore().unwrap());
        let present_ready_semaphores =
            std::array::from_fn(|_| device_context.make_binary_semaphore().unwrap());
        let present_finished_fences =
            std::array::from_fn(|_| device_context.make_fence(true).unwrap());

        Ok(Self {
            raw: swapchain,
            loader,

            image_available_semaphores,
            present_ready_semaphores,
            present_finished_fences,

            images: images.into(),
            image_views: image_views,

            slots,
            slot_index: 0,
            extent,
            recreation_required: swapchain.is_null(),
        })
    }

    /// Recreates the swapchain and images.  If this procedure fails, swapchain is likely in an
    /// inconsistent state and more thorough teardown is advised.
    pub fn recreate(
        &mut self,
        device_context: &DeviceContext,
        surface: &VkSurface,
    ) -> Result<(), VulkanError> {
        // XXX not reaching through the surface caused a bug earlier because of persisting a value
        // from the initial extent.  This is duplication.  Let's begin tying the lifetimes where
        // natural.
        self.extent = surface.extent();
        // We made the image views.  We have to destroy them.
        let device = device_context.device();
        unsafe {
            for view in self.image_views.drain(..) {
                device.destroy_image_view(view, None);
            }
        }
        let mut swapchain_ci = surface.swapchain_ci();
        let swapchain_ci = swapchain_ci.old_swapchain(self.raw);

        // XXX try to handle the broken recreation case... We have already used "old swapchain" in
        // the call.  Would we still need to destroy it if this call fails or is it in limbo?
        // In addition to failing, would we set our pointer to null?
        let new_swapchain = unsafe { self.loader.create_swapchain(&swapchain_ci, None)? };

        // Old swapchain retired.  Safe to destroy.
        unsafe {
            self.loader.destroy_swapchain(self.raw, None);
        }
        self.raw = new_swapchain;

        // Recreate images
        self.images = unsafe { self.loader.get_swapchain_images(self.raw)?.into() };
        self.image_views =
            create_image_views(&device_context.device(), &self.images, surface.format());

        let new_slots = self.images.len().max(3);
        if new_slots != self.slots {
            self.slots = new_slots;
            self.slot_index = 0; // XXX drain within recreation and start back at the first slot.
        }

        Ok(())
    }

    // XXX this method is extremely thin.  Probably better to just expose the loader for raw usage.
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
            self.loader.destroy_swapchain(self.raw, None);
            self.image_available_semaphores.iter().for_each(|s| {
                s.destroy(context);
            });
            self.present_ready_semaphores.iter().for_each(|s| {
                s.destroy(context);
            });
            self.drain_present(context).unwrap();
            for fence in &self.present_finished_fences {
                device.destroy_fence(**fence, None);
            }
        }
    }

    /// Get the next swapchain image.  Errors may indicate need to request recreation.
    pub fn acquire(&mut self) -> Result<AcquiredImage, VulkanError> {
        let idx = self.slot_index as usize;
        let image_available = self.image_available_semaphores[idx];
        let present_ready = self.present_ready_semaphores[idx];
        let present_finished = self.present_finished_fences[idx];
        self.slot_index = (idx + 1) % self.slots;

        let (swapchain_image_index, _) = unsafe {
            self.loader
                .acquire_next_image(
                    self.raw,
                    100_000_000, // 100ms
                    image_available.as_raw(),
                    vk::Fence::null(),
                )
                .inspect_err(|e| {
                    #[cfg(debug_assertions)]
                    eprintln!("warning: swapchain acquisition failed: {:?}", e);
                    self.recreation_required = true;
                })?
        };

        let image = self.images[swapchain_image_index as usize];
        let image_view = self.image_views[swapchain_image_index as usize];

        Ok(AcquiredImage {
            image_available: image_available.wait(),
            present_ready: present_ready.signal(),
            present_finished,
            image,
            image_view,
            extent: self.extent,
            swapchain_image_index,
        })
    }

    /// This may be called by high-level support, but if you're wrapping a swapchain on your own,
    /// this function enables waiting on all present-in-flight fences to signal, meaning it's safe
    /// to recreate the swapchain.
    pub fn drain_present(&self, device_ctx: &DeviceContext) -> Result<(), VulkanError> {
        // XXX Grrrrr that newtype is in our way.  Let's drop a colony on the Earth to fix it.
        let raw: [vk::Fence; 4] = self.present_finished_fences.map(|f| f.0);
        unsafe { Ok(device_ctx.device().wait_for_fences(&raw, true, u64::MAX)?) }
    }

    pub fn as_raw(&self) -> &vk::SwapchainKHR {
        &self.raw
    }

    pub fn into_raw(self) -> vk::SwapchainKHR {
        self.raw
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
