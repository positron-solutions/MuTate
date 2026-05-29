// Copyright 202o6 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Surface
//!
//! Encapsulate decisions about the surface that are used to inform the swapchain creation (and
//! re-creation).  Image format, size, and supported swapchain parameters are decided here.

// NEXT should be okay to allow window and surface lifetime to travel together.  You know what,
// letting the user re-supply a window is just extra bug surface.
// NEXT split once-per-surface, once-per-user-settings update decisions from dynamic ones like
// extent.

use ash::{
    khr::{surface::Instance as SurfaceInstance, xlib_surface},
    vk,
};
#[cfg(feature = "winit")]
use winit::window::Window;

use crate::internal::*;

/// Embodied decisions about how we will use a [`ash::vk::SurfaceKHR`] and the raw handles for
/// bookkeeping.
pub struct VkSurface {
    /// Raw Vulkan surface handle.
    pub(crate) inner: vk::SurfaceKHR,
    /// Color format and color space.  Use sRGB nonlinear + BGRA8 when available.  Can affect the
    /// swapchain image format.
    pub format: vk::SurfaceFormatKHR,
    /// MAILBOX if supported.  FIFO_RELAXED is pretty close.
    pub present_mode: vk::PresentModeKHR,
    /// How the surface output will be blended by any compositor.
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    pub swapchain_image_count: u32,
    surface_loader: ash::khr::surface::Instance,
}

impl VkSurface {
    pub fn new(
        surface: vk::SurfaceKHR,
        vk_context: &VkContext,
        device_context: &DeviceContext,
    ) -> Self {
        let physical_device = device_context.physical_device;
        let surface_loader = vk_context.surface_loader();
        let VkContext {
            entry, instance, ..
        } = vk_context;

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };

        // NEXT support HDR whenever we get some hardware that can display it.  Regular unorm
        // pseudo-float outputs fill up a larger dynamic range on HDR, so it's generally not that
        // hard.  See VK_EXT_swapchain_colorspace.
        let format = [
            // HDR formats
            // requires VK_EXT_swapchain_colorspace / VK_AMD_display_native_hdr
            // (vk::Format::R16G16B16A16_SFLOAT,  vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT),
            // (vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::HDR10_ST2084_EXT),
            // (vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::HDR10_HLG_EXT),
            // (vk::Format::B10G11R11_UFLOAT_PACK32, vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD),

            // SDR formats
            (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            // NEXT add VkComponentMapping to support a different channel order.
            // (vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            // NEXT unless we are controlling gamma encode, no use for this
            // (
            //     vk::Format::B8G8R8A8_UNORM,
            //     vk::ColorSpaceKHR::SRGB_NONLINEAR,
            // ),
            // (
            //     vk::Format::R8G8B8A8_UNORM,
            //     vk::ColorSpaceKHR::SRGB_NONLINEAR,
            // ),
        ]
        .iter()
        .find_map(|&(format, color_space)| {
            formats
                .iter()
                .copied()
                .find(|f| f.format == format && f.color_space == color_space)
        })
        .unwrap_or_else(|| {
            let fallback = formats[0];
            eprintln!("warning: fallback surface format selected: {:?}", fallback);
            fallback
        });

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
                .unwrap()
        };
        // Present mode choices are related to VRR vs FRR.  Since we intentionally delay rendering
        // to be nearer to the latch (to reduce power draw and to be closer to reduce audio lag), we
        // prefer modes where only the most recent frame can be presented.
        let present_mode = [
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO_RELAXED,
            vk::PresentModeKHR::FIFO,
        ]
        .iter()
        .copied()
        .find(|mode| present_modes.contains(mode))
        .unwrap_or(vk::PresentModeKHR::FIFO);

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };
        // Generally we're expecting any surface we write to might have a compositor behind it.  If
        // so, we might get a funky compositor alpha blend back.  It might mean something to
        // upstream writers, but we can adapt it.
        // NEXT Adapt funky alpha?  Maybe if you find one.
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::INHERIT,
        ]
        .iter()
        .copied()
        .find(|mode| surface_caps.supported_composite_alpha.contains(*mode))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::INHERIT);

        // XXX Make an actual decision
        let pre_transform = surface_caps.current_transform;

        let swapchain_image_count = {
            let mode_minimum = match present_mode {
                vk::PresentModeKHR::MAILBOX => 3,
                _ => 2,
            };
            let desired = mode_minimum.max(surface_caps.min_image_count);
            if surface_caps.max_image_count == 0 {
                desired
            } else {
                desired.min(surface_caps.max_image_count)
            }
        };

        Self {
            inner: surface,
            format,
            present_mode,
            composite_alpha,
            pre_transform,
            surface_loader,
            swapchain_image_count,
        }
    }

    /// Return the raw surface.
    // MAYBE deref as vk::SurfaceKHR instead?
    pub fn as_raw(&self) -> vk::SurfaceKHR {
        self.inner.clone()
    }

    /// Surface size is a bit subjective.  We use this function to decide how big to make swapchains
    /// etc.  The behavior is platform specific and depends on headless vs windowed rendering.  If
    /// the result seems degenerate.  We return an error, and Applications should do nothing and
    /// instead wait on another event to try again.
    ///
    /// - `device_context` - the physical device for this surface.
    /// - `window` - the window under inspection.
    #[cfg(feature = "winit")]
    pub fn resolve_size(
        &self,
        device_context: &DeviceContext,
        window: &Window,
    ) -> Result<vk::Extent2D, VulkanError> {
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(
                    device_context.physical_device,
                    self.inner,
                )
        }
        .map_err(|e| {
            let err = VulkanError::from(e);
            eprintln!("warning: surface queries could not be performed: {}", err);
            err
        })?;

        // A valid surface must advertise at least a 1x1 max extent.  Zeros here indicate either a
        // driver bug or a surface that is transitionally invalid (lost but not yet reported).
        // Treat it as SurfaceLost so the caller tears down and recreates rather than idling.
        if caps.max_image_extent.width == 0 || caps.max_image_extent.height == 0 {
            eprintln!(
                "warning: surface caps report zero max_image_extent; \
                 treating as surface lost (driver bug or transitional state)"
            );
            return Err(VulkanError::SurfaceLost);
        }

        // Resolve the raw extent before any clamping.  On X11/Win32 the compositor fills this in
        // directly.  On Wayland it is always u32::MAX and we must use the window size instead.
        let raw_extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            let window_size = window.inner_size();
            if window_size.width == 0 || window_size.height == 0 {
                // Wayland + zero window: degenerate, cannot proceed.
                return Err(VulkanError::DegenerateExtent {
                    width: window_size.width,
                    height: window_size.height,
                });
            }
            vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            }
        };

        // Guard degenerate extents (e.g. X11 during minimize) before clamping.
        if raw_extent.width == 0 || raw_extent.height == 0 {
            return Err(VulkanError::DegenerateExtent {
                width: raw_extent.width,
                height: raw_extent.height,
            });
        }

        Ok(vk::Extent2D {
            width: raw_extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: raw_extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        })
    }

    pub fn destroy(&self) {
        unsafe {
            self.surface_loader.destroy_surface(self.inner, None);
        }
    }
}
