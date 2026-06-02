// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Present
//!
//! With a `CommandPool`, `Queue`, and `Swapchain`, it's not much more work to write the render
//! phase that can be called in an application's redraw loop.  This module provides reference
//! implementations, which provide users with a single recording command buffer and an output image.
//! The [`PresentMode`] encapsulates some barrier and presentation boilerplate.

// MAYBE Queue presentation itself is really only dependent on at least one signaled semaphore and
// the output image. This module could grow some abstraction to avoid duplicating repetitive raw
// ash.
// XXX The Target traite seems likely to not materialize.  It was intended that testing abstractions
// over a single Pool and output Image would be able to express different kinds of targets other
// than `AcquiredImage`, but it's questionable whethere an abstraction will be helpful or just
// ceremony with no meat in the abstraction.
// NEXT update command buffer wrapper to support beginning and ending rendering, then use that to
// implement GraphicsPresent
// NEXT create kinds of targets that may only use specific layouts that are valid for the upstream
// render target.

pub mod surface;
pub mod swapchain;

use std::marker::PhantomData;
use std::slice;

use ash::vk::Handle;

use crate::dispatch::pw;
use crate::internal::*;
use crate::present::surface::ExtentSource;
use crate::resource::image;

pub mod prelude {
    pub use super::compute_present;
    pub use super::graphics_present;
    pub use super::surface::Surface;
    pub use super::swapchain::{AcquiredImage, Swapchain};
    pub use super::PresentRing;
}

pub struct PresentRing {
    pool_ring: PoolRing<Graphics>,
    present: pw::PresentConsumer,
    queue: Queue<Graphics>,
    swapchain: Swapchain,
}

impl PresentRing {
    pub fn new(
        device_context: &DeviceContext,
        vk_context: &VkContext,
        surface: &Surface,
        extent: vk::Extent2D,
    ) -> Result<Self, VulkanError> {
        let swapchain = Swapchain::new(device_context, vk_context, surface, extent)?;
        let queue = device_context
            .queues
            .graphics(vk_context, surface, QueuePriority::High)
            .ok_or(VulkanError::QueueNotFound)?;
        let pool_ring = PoolRing::new(device_context, &queue)?;
        let present = pw::PresentConsumer::new(vk_context, device_context, *swapchain.as_raw())?;
        Ok(Self {
            present,
            pool_ring,
            queue,
            swapchain,
        })
    }

    /// Draw with a user-supplied recording function.
    ///
    /// **Contract**: `record_fn` receives a started command buffer and the acquired image.
    /// The image begins in `UNDEFINED` layout. `record_fn` **must** leave the image in
    /// `PRESENT_SRC_KHR` before returning.  Use [`graphics_bracketed`] or [`compute_bracketed`]
    /// to satisfy this contract without writing barriers by hand.
    pub fn record<F, G>(
        &mut self,
        device_ctx: &DeviceContext,
        record_fn: F,
        post_draw_fn: G,
    ) -> Result<(), VulkanError>
    where
        F: FnOnce(&DeviceContext, &RecordingBuffer<Graphics, OneTime>, &AcquiredImage),
        G: FnOnce(),
    {
        let acquired_image = self.swapchain.acquire()?;
        let (pool, intent) = self.pool_ring.acquire(device_ctx, 1_000_000_000).unwrap();
        let cb = pool.primary(device_ctx).unwrap();

        record_fn(device_ctx, &cb, &acquired_image);

        let recorded = cb.end(device_ctx).unwrap();
        self.queue
            .submission()
            .wait_binary(
                acquired_image.image_available,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            )
            .execute(recorded)
            .signal_binary(
                acquired_image.present_ready,
                vk::PipelineStageFlags2::ALL_GRAPHICS,
            )
            .signal(intent, vk::PipelineStageFlags2::ALL_COMMANDS)
            .submit(device_ctx, vk::Fence::null())
            .unwrap();
        post_draw_fn();

        if let Some(_last) = self.present.read_last_present() {}
        let next_id = self.present.next_present_id();
        let mut present_id = vk::PresentIdKHR::default().present_ids(slice::from_ref(&next_id));
        let present_ready = acquired_image.present_ready.as_raw();
        unsafe {
            device_ctx
                .device()
                .reset_fences(slice::from_ref(&acquired_image.present_finished))?;
        }
        let mut present_finished = vk::SwapchainPresentFenceInfoEXT::default()
            .fences(std::slice::from_ref(&acquired_image.present_finished));
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(slice::from_ref(&present_ready))
            .swapchains(slice::from_ref(self.swapchain.as_raw()))
            .image_indices(slice::from_ref(&acquired_image.swapchain_image_index))
            .push_next(&mut present_id)
            .push_next(&mut present_finished);

        self.swapchain
            .present(unsafe { self.queue.raw() }, &present_info);
        self.present.notify_waiter();
        Ok(())
    }

    pub fn maybe_update_swapchain<'a>(
        &mut self,
        device_ctx: &DeviceContext,
        surface: &mut Surface,
        extent_source: impl Into<ExtentSource<'a>>,
    ) -> Result<vk::Extent2D, VulkanError> {
        // XXX Check if surface actually needs recreation!
        let new_size = surface.update(device_ctx, extent_source)?;
        self.swapchain.drain_present(device_ctx)?;
        self.swapchain.recreate(device_ctx, surface);
        self.present
            .notify_swapchain_recreation(*self.swapchain.as_raw());
        Ok(new_size)
    }

    pub fn destroy(self, context: &DeviceContext) {
        let Self {
            swapchain,
            pool_ring,
            ..
        } = self;
        swapchain.destroy(context);
        pool_ring.destroy(context);
    }
}

/// Call around your own drawing function. Transitions the acquired image to
/// [`vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL`] for drawing commands. Finishes in
/// [`vk::ImageLayout::PRESENT_SRC_KHR`] for presentation.  Inserts necessary barriers.  The user
/// just begins rendering and starts issuing drawing commands.
///
/// Function **does not** issue the `cmd_begin_rendering` or `cmd_end_rendering` commands.  This
/// enables the user to execute commands before beginning rendering and then to select their own
/// render information setup.
pub fn graphics_present<'d, F>(
    device_ctx: &'d DeviceContext,
    extent: vk::Extent2D,
    user_fn: F,
) -> impl FnOnce(&DeviceContext, &RecordingBuffer<Graphics, OneTime>, &AcquiredImage) + 'd
where
    F: FnOnce(&DeviceContext, &RecordingBuffer<Graphics, OneTime>, &AcquiredImage) + 'd,
{
    move |device_ctx, cb, acquired_image| {
        let device = device_ctx.device();
        image::transition_layout(
            acquired_image.image,
            &**cb,
            image::range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            device_ctx,
        );
        // The money
        user_fn(device_ctx, cb, acquired_image);
        image::transition_layout(
            acquired_image.image,
            &**cb,
            image::range(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            device_ctx,
        );
    }
}

/// Call around your own command recording function. Transitions the acquired image to
/// [`vk::ImageLayout::TRANSFER_DST_OPTIMAL`] for commands. Finishes in
/// [`vk::ImageLayout::PRESENT_SRC_KHR`] for presentation.  Inserts necessary barriers.  The user
/// only needs to issue drawing commands and copy output to the destination buffer.
pub fn compute_present<'d, F>(
    device_ctx: &'d DeviceContext,
    user_fn: F,
) -> impl FnOnce(&DeviceContext, &RecordingBuffer<Graphics, OneTime>, &AcquiredImage) + 'd
where
    F: FnOnce(&DeviceContext, &RecordingBuffer<Graphics, OneTime>, &AcquiredImage) + 'd,
{
    move |device_ctx, cb, acquired_image| {
        image::transition_layout(
            acquired_image.image,
            &**cb,
            image::range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            device_ctx,
        );
        user_fn(device_ctx, &cb, &acquired_image);
        image::transition_layout(
            acquired_image.image,
            &**cb,
            image::range(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            device_ctx,
        );
    }
}
