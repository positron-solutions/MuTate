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
//! A mature application should acquire command buffers and manage recording and presentation on its
//! own.  The dependencies of queue submissions in flight depends too much on the needs of the
//! specific application.  See the [`present`](crate::present) module for some basic components that
//! can get off the ground a bit faster.
//!
//! ### Recreation
//!
//! Whenever windows are resized, the swapchain needs to be recreated.  We reuse and retire the old
//! swapchain during recreation make it easy on the driver with respect to memory.  Images are
//! lazily allocated to further reduce the load.
//!
//! The signal for recreation can be a window resize event or an ash result of `SUBOPTIMAL` or
//! `OUT_OF_DATE_KHR`.  More serious errors likely indicate a need to recreate the logical device
//! instead and should be propagated to the application.
//!
//! The [`Option(vk::Extent2D)`](vk::Extent2D) from recreation indicates if the size of the
//! swapchain changed.  This is critical for swapchain size dependents to be re-provisioned.
//!
//! ## Surrounding Context
//!
//! A swapchain is a lazy allocation of some images that can be scanned out to a surface, usually
//! for a physical display.  We obtain images, some command buffers draw things on them, and then we hand
//! control of the images back to a compositor or DRM for physical scan-out to some screen.
//!
//! The [`Surface`] knows the format we select.  The format is important for the renderer to ensure
//! that the output of drawing is either compatible with or can be copied to the swapchain image for
//! presentation.  The [`AcquiredImage`] carries along format, extent, and other information for
//! renderers to write correctly to the output.
//!
//! `Surface` also controls the present mode.  When doing just-in-time rendering, all modes are used
//! in a way that achieves the same effect as simple `FIFO` with a minimal queue depth, but there
//! may be some nuances for frame probes and VRR vs FRR while determining the downstream display
//! refresh rates (ahead of broader `EXT_present_timing` support shipping).
//!
//! ## Index Behavior
//!
//! We have only slight control over the number of swapchain images.  We have no control over the
//! image index from acquisition to acquisition.  To simplify our rotation through sync primitives,
//! we use an independent index and always allocate four of each sync primitive, which may be more
//! than the number of images in the swapchain.  This should be enough for most presentation
//! pipelines:
//!
//! - One image is acquired for recording
//! - One image is being used by an in-flight submission
//! - One image being presented by the compositor
//! - One spare slot to avoid bubbles at the edges, such as acquisition `N+3` stalling on
//!   presentation `N` leaving the compositor late.
//!
//! No amount of images will overcome a lazy enough downstream that neglects to give us back our
//! images.  We strive to have only one present prior to each latch so that `FIFO` does exactly what
//! we want.  While deeper pipelining may see frame `N+1` work overlapping with frame `N`, the
//! overlapped work does not require acquisition and should be broken into separate submissions
//! rather than acquiring the output image that won't be written to until after almost a whole
//! presentation frame of delay.

// MAYBE interfaces for a render target, recording, and submission separate from the destination
// frontend seem obvious.  `AcquiredImage` is specific to the swapchain as a destination.  These
// interfaces were not forthcoming in a first pass.
// NEXT presentation likely wants to control the actual present call but return the result to this
// swapchain since presentation errors are commonly going to be triggers for recreation.  Make a new
// method for the presenter to notify the swapchain on sub-optimal or out-of-date.
// NOTE overall the current swapchain encapsulation is becoming somewhat pleasant and this code may
// be ready for town planning to make it clean and document the interface.  It's likely reaching
// stability.
// NEXT there seems to be no good reason for the swapchain to not just own the surface.  Even
// headless surfaces still uses surfaces.  Let's go ahead and try to bind them.
// NOTE requiring the window as an extent source will pull the window into a render thread.  If the
// Window needs to route calls back through the main thread anyway (Android, MacOS?), deadlock
// surface could be exposed.  If we do not have access to the window in the render thread, we have
// to wait for events in order to resize even if we already know the swapchain is garbage.  The
// consequence is we're missing any up-to-date extent and must have the render thread wait for the
// window event to show up.  Will it show up?
// NOTE If two `AcquiredImages` were in flight, the presentation failure path for one can destroy the
// swapchain out from under the other `AcquiredImage`.  This would be tricky for accounting.
//
// Consider why an image would ever be acquired before the previous presents.  If recording is
// extremely slow and serial but can be parallelized to increase a frame rate, it may be tempting to
// acquire to on one thread before presentation is done on the other.  This also may occur if the
// dispatches take a long time but may be pipelined on the device.  In either such case, the user
// may instead break up the dispatches so that acquires are delayed to match presentation pacing
// rather than acquiring before the previous acquisition has even queued present.

// FIXME Four images seems to be the correct number.  Update surface to request four images.
// MAYBE If acquisition should fail when the swapchain is dirty, the swapchain may as well attempt
// recreation by polling the `Surface`.  If presentation fails on the user side, the user *may*
// attempt recreation right away or could defer to next acquisition at the cost of some latency on
// that frame.  We need acquisition and post-presentation to support these usage models.
// NEXT Reaping the retired swapchains can be deferred until post-present methods and thus hidden
// from the presenter that doesn't know if any retired swapchains remain.
// XXX We may be able to simplify PresentSync reclaim by storing any dead fence on that PresentSync.  If
// we can guarantee that there is at most one dead fence per PresentSync (we don't try to re-use the
// PresentSync after even one `OUT_OF_DATE`), a simple dead_fence field would suffice.  It can be null
// by default or non-null if one of the fences is dead (and we null that slot in its fences!).
// Pre-vs-post increment of the index must be carefully paid attention to if we are to infer the
// dead fence without scanning the fences for the one used in the submission.

use ash::vk::{self, Handle};
use smallvec::SmallVec;

use super::surface::Surface;

use crate::internal::*;

/// An image from the swapchain and associated format, extent, and synchronization data necessary
/// for presentation.
pub struct AcquiredImage {
    /// Swapchain signals when image is ready for use (start rendering).
    pub image_available: BinaryWait,
    /// User signals when compositor may present image (after rendering).
    pub present_ready: BinarySignal,
    /// Signaled by presentation when image is completely finished with.  Used for draining the
    /// swapchain, but can also be used by other waiters.
    // NOTE crate visibility since failed presentation results in a fence that will never signal.
    pub(crate) present_finished: Fence,

    /// Necessary to begin any rendering
    pub image_view: vk::ImageView,
    pub image: vk::Image,
    /// The size of the output image.  Useful for updating push constants etc for techniques that
    /// use the screen size at the last moment rather than store it at provision time.
    pub extent: vk::Extent2D,

    /// The index provided by the swapchain during acquisition.  Used in all presentation to tell
    /// the other side of the swapchain which image we are requesting to present.
    pub(crate) swapchain_image_index: u32,
    /// The `FrameSync` index used for this acquisition.  If presentation fails, we need this in
    /// order to not wait on the stranded fence.
    pub(crate) sync_index: usize,
}

/// When recreating swapchains, we may need to hold onto old in-flight resources to drain and
/// destroy them together.  The swapchain controls its own deferred destruction for now.  A later
/// deletion queue solution may be able to reclaim this without ever looking at the fences in the
/// render loop.
struct PresentSync {
    // note about choice of 4 in module docs
    image_available: [BinarySemaphore; 4],
    present_ready: [BinarySemaphore; 4],
    present_finished: [Fence; 4],
    slots: usize,
    /// During acquisition, we must provide a binary semaphore for the compositor to signal when the
    /// image is ready.  Since we don't know the index of the image until we acquire it, we use our
    /// own independent index to cycle through semaphores, ensuring no re-use on different images.
    sync_index: usize,
    /// If presentation fails with `OUT_OF_DATE_KHR`, the associated fence will be unsignalled but
    /// will never be signaled.  We park the stranded fence inside this field to make destruction /
    /// culling tidy.  The fence is just a null fence when not actually occupied.
    stranded_fence: Fence,
}

impl PresentSync {
    fn new(device: &Device, slots: usize) -> Result<Self, VulkanError> {
        // Four of each so the slot arrays always have room, whatever the image count.
        let binary = || device.make_binary_semaphore();
        let fence = || device.make_fence(true);
        // XXX Not my favorite expression.
        Ok(Self {
            image_available: [binary()?, binary()?, binary()?, binary()?],
            present_ready: [binary()?, binary()?, binary()?, binary()?],
            present_finished: [fence()?, fence()?, fence()?, fence()?],
            slots,
            sync_index: 0,
            stranded_fence: Fence(vk::Fence::null()),
        })
    }

    /// A fence cleared for presentation failed to signal and must not be waited on.  Replace its
    /// entry with the null fence.
    fn strand_fence(&mut self, sync_index: usize) {
        // Stranding is at most once per PresentSync!  It may be cleaner to scan for the actual
        // fence since it must be within the current PresentSync.
        debug_assert!(self.stranded_fence.0.is_null());
        std::mem::swap(
            &mut self.stranded_fence,
            &mut self.present_finished[sync_index],
        );
    }

    /// Advance the acquisition cursor, returning the slot to use *this* acquire.
    fn next_slot(&mut self) -> usize {
        let slot = self.sync_index;
        self.sync_index = (slot + 1) % self.slots;
        slot
    }

    /// Wait for all present-finished fences to signal.  See swapchain's drain for meaning of return
    /// values.
    pub fn drain(&self, device: &Device, timeout: u64) -> Result<bool, VulkanError> {
        // We can't wait on the null fence fence.  There are cleaner ways to do this, but we have
        // things to do.
        let mut live = [vk::Fence::null(); 4];
        let mut n = 0;
        for f in &self.present_finished {
            if !f.0.is_null() {
                live[n] = f.0;
                n += 1;
            }
        }

        match unsafe { device.as_raw().wait_for_fences(&live[..n], true, timeout) } {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }

    /// Callers must drain before destruction.
    pub fn destroy(&self, device: &Device) {
        let raw = device.as_raw();
        unsafe {
            for s in &self.image_available {
                raw.destroy_semaphore(s.as_raw(), None);
            }
            for s in &self.present_ready {
                raw.destroy_semaphore(s.as_raw(), None);
            }
            for f in &self.present_finished {
                if !f.0.is_null() {
                    raw.destroy_fence(f.0, None);
                }
            }
            if !self.stranded_fence.0.is_null() {
                raw.destroy_fence(self.stranded_fence.0, None);
            }
        }
    }
}

/// After swapchain recreation, we need to store the old handles until the sync primitives can be
/// reclaimed.  The Retired structure packages all of the old Vulkan resources together.
struct Retired {
    swapchain: vk::SwapchainKHR,
    image_views: SmallVec<vk::ImageView, 4>,
    sync: PresentSync,
}

/// An initialized swapchain with necessary synchronization and accounting data included.  Acquire
/// images and go.
pub struct Swapchain {
    raw: vk::SwapchainKHR,
    loader: ash::khr::swapchain::Device,
    image_views: SmallVec<vk::ImageView, 4>,
    images: SmallVec<vk::Image, 4>,

    /// Synchronization primitives are gathered into their own structure for simple epoch management.
    sync: PresentSync,
    /// Sync primitives for previous swapchains are stored here so that their deletion is deferred.
    retired: SmallVec<Retired, 4>,

    /// Extent is held to understand if size updates actually require swapchain recreation.
    // XXX Not actually inspected for resize decisions.  Go return an option!
    extent: vk::Extent2D,
    /// Flag set by `OUT_OF_DATE_KHR` and `SUBPOTIMAL` errors.
    recreation_required: bool,
}

impl Swapchain {
    pub fn new(
        device: &Device,
        instance: &Instance,
        surface: &Surface,
    ) -> Result<Self, VulkanError> {
        let Instance {
            entry,
            raw: instance,
            ..
        } = &instance;
        let extent = surface.extent();
        let loader = ash::khr::swapchain::Device::new(instance, device.as_raw());
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
        let image_views = create_image_views(&device, &images, surface.format());

        Ok(Self {
            raw: swapchain,
            loader,

            sync: PresentSync::new(device, slots)?,
            retired: SmallVec::new(),

            images: images.into(),
            image_views: image_views,

            extent,
            recreation_required: swapchain.is_null(),
        })
    }

    /// Recreates the swapchain and images.  If this procedure fails, swapchain is likely in an
    /// inconsistent state and more thorough rebuild at the application level is advised.
    pub fn recreate(&mut self, device: &Device, surface: &Surface) -> Result<(), VulkanError> {
        self.extent = surface.extent();

        let mut swapchain_ci = surface.swapchain_ci();
        let swapchain_ci = swapchain_ci.old_swapchain(self.raw);

        // DEBT partial failure cleanup.
        let new_swapchain = unsafe { self.loader.create_swapchain(&swapchain_ci, None)? };
        let new_images: SmallVec<vk::Image, 4> =
            unsafe { self.loader.get_swapchain_images(new_swapchain)?.into() };
        let new_slots = new_images.len().clamp(3, 4);
        let new_image_views = create_image_views(device, &new_images, surface.format());
        let new_sync = PresentSync::new(device, new_slots)?;

        // Old in-flight Vulkan resources will be moved to retirement for later culling.
        // Post-present and pre-acquisition can perform such culling lazily.  Destruction must
        // properly drain in-flight resources.
        let old_swapchain = std::mem::replace(&mut self.raw, new_swapchain);
        let old_image_views = std::mem::replace(&mut self.image_views, new_image_views);
        let old_sync = std::mem::replace(&mut self.sync, new_sync);

        self.retired.push(Retired {
            swapchain: old_swapchain,
            image_views: old_image_views,
            sync: old_sync,
        });

        self.images = new_images;
        // A fresh FrameSync means the cursor is already back at slot 0.
        self.recreation_required = false;

        Ok(())
    }

    // MAYBE this method is where AcquiredImage comes back to the swapchain, and there is something
    // oddly smelling about handing this back to the swapchain for present.  While the swapchain
    // does need to be informed if presentation fails, whatever is driving presentation is more
    // competent to actually do the presentation.
    pub fn present(
        &mut self,
        queue: vk::Queue,
        sync_index: usize,
        present_info: &vk::PresentInfoKHR,
    ) -> Result<(), VulkanError> {
        unsafe {
            return match self.loader.queue_present(queue, present_info) {
                Ok(false) => {
                    // happy path!
                    Ok(())
                }
                // Ok(true) is vk::Result::SUBOPTIMAL_KHR.  Fence will signal.  Recreation not
                // required but might as well.
                Ok(true) => {
                    self.recreation_required = true;
                    eprintln!("presentation: swapchain suboptimal");
                    Err(VulkanError::SwapchainSuboptimal)
                }
                // No present occurred; the fence we reset in `record` will NOT be signaled.
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreation_required = true;
                    self.sync.strand_fence(sync_index);
                    eprintln!("presentation: ERROR_OUT_OF_DATE_KHR");
                    Err(VulkanError::SwapchainOutOfDate)
                }
                Err(result) => {
                    self.recreation_required = true;
                    eprintln!("presentation: {:?}", result);
                    Err(result.into())
                }
            };
        }
    }

    /// Swapchain destruction may need to drain in-flight resources.  Wait is controllable with
    /// `timeout`.  Destruction may not proceed until all current and retired swapchains have fully
    /// drained.
    ///
    /// `Ok(true)`  - all fences signaled
    /// `Ok(false)` - not yet drained
    /// `Err(_)`    - device loss or other real failure
    ///
    /// Pass `timeout == 0` to return immediately.  `timeout` is nanoseconds, matching `wait_for_fences`.
    pub(crate) fn drain(&self, device: &Device, timeout: u64) -> Result<bool, VulkanError> {
        let mut drained = self.sync.drain(device, timeout)?;
        for retired in &self.retired {
            drained = drained && retired.sync.drain(device, timeout)?;
        }
        Ok(drained)
    }

    /// Destruction requires first calling `drain` whenever resources may still be in flight.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            for view in &self.image_views {
                device.as_raw().destroy_image_view(*view, None);
            }
            self.loader.destroy_swapchain(self.raw, None);
        }
        self.sync.destroy(device);
        for retired in &self.retired {
            unsafe {
                for view in &retired.image_views {
                    device.as_raw().destroy_image_view(*view, None);
                }
                self.loader.destroy_swapchain(retired.swapchain, None);
            }
            retired.sync.destroy(device);
        }
    }

    pub fn recreation_required(&self) -> bool {
        self.recreation_required
    }

    /// Get the next swapchain image.  Errors may indicate need to request recreation.
    pub fn acquire(&mut self) -> Result<AcquiredImage, VulkanError> {
        // XXX add pre-check methods for the client side
        if self.recreation_required {
            return Err(VulkanError::SwapchainRecreationRequired);
        }

        let sync_index = self.sync.next_slot();
        let image_available = self.sync.image_available[sync_index];
        let present_ready = self.sync.present_ready[sync_index];
        let present_finished = self.sync.present_finished[sync_index];

        let (swapchain_image_index, _) = unsafe {
            self.loader
                .acquire_next_image(
                    self.raw,
                    32_000_000, // 32ms
                    image_available.as_raw(),
                    vk::Fence::null(),
                )
                .inspect_err(|e| {
                    #[cfg(debug_assertions)]
                    eprintln!("warning: swapchain acquisition failed: {:?}", e);

                    // XXX this recreation required has not been studied closely and may not be
                    // coherent with other recreation_required flagging.
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
            sync_index,
        })
    }

    pub fn as_raw(&self) -> &vk::SwapchainKHR {
        &self.raw
    }

    pub fn into_raw(self) -> vk::SwapchainKHR {
        self.raw
    }
}

fn create_image_views(
    device: &Device,
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
            unsafe { device.as_raw().create_image_view(&view_ci, None).unwrap() }
        })
        .collect()
}
