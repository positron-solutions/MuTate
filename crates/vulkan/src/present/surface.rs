// Copyright 202o6 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Surface
//!
//! We're not doing a lot of work on top of the [`vk::SurfaceKHR`], but most of the decisions only
//! happen once and are then repeated for the lifetime of the surface, so we will just encapsulate
//! them as state in the [`VkSurface`] struct.

use ash::{
    khr::{surface::Instance as SurfaceInstance, xlib_surface},
    vk,
};

use crate::context::{DeviceContext, VkContext};

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
    surface_loader: ash::khr::surface::Instance,
}

impl VkSurface {
    pub fn new(
        surface: vk::SurfaceKHR,
        vk_context: &VkContext,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let surface_loader = vk_context.surface_loader();
        let VkContext { entry, instance } = vk_context;

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap()
        };

        // NEXT support HDR whenever we get some hardware that can display it.  Regular unorm
        // pseudo-float outputs fill up a larger dynamic range on HDR, so it's generally not that
        // hard.
        let format = [
            // HDR formats
            // requires VK_EXT_swapchain_colorspace / VK_AMD_display_native_hdr
            // (vk::Format::R16G16B16A16_SFLOAT,  vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT),
            // (vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::HDR10_ST2084_EXT),
            // (vk::Format::A2B10G10R10_UNORM_PACK32, vk::ColorSpaceKHR::HDR10_HLG_EXT),
            // (vk::Format::B10G11R11_UFLOAT_PACK32, vk::ColorSpaceKHR::DISPLAY_NATIVE_AMD),

            // SDR
            (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            (vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            (
                vk::Format::B8G8R8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
            (
                vk::Format::R8G8B8A8_UNORM,
                vk::ColorSpaceKHR::SRGB_NONLINEAR,
            ),
        ]
        .iter()
        .find_map(|&(format, color_space)| {
            formats
                .iter()
                .copied()
                .find(|f| f.format == format && f.color_space == color_space)
        })
        .unwrap_or(formats[0]);

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
        Self {
            inner: surface,
            format,
            present_mode,
            composite_alpha,
            pre_transform,
            surface_loader,
        }
    }

    /// Return the raw surface.
    // MAYBE deref as vk::SurfaceKHR instead?
    pub fn as_raw(&self) -> vk::SurfaceKHR {
        self.inner.clone()
    }

    /// Surface size is a bit subjective.  We use this function to decide how big to make swapchains
    /// etc.  The behavior is platform specific and depends on headless vs windowed rendering.
    /// There are cases where we throw up our hands and the result seems degenerate.  We return a
    /// `None` in those cases, and Applications should do nothing and wait on another event.
    ///
    /// - `device_context` - the physical device for this surface.
    /// - `fallback` - when there is a window, its `inner_size` attribute may be helpful,  but there
    ///    are cases where we infer that the window is degenerate (zero size) etc.
    pub fn resolve_size(
        &self,
        device_context: &DeviceContext,
        fallback: Option<vk::Extent2D>,
    ) -> Option<vk::Extent2D> {
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(
                    device_context.physical_device,
                    self.inner,
                )
        }
        .ok()?;

        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            fallback?
        };

        if extent.width == 0 || extent.height == 0 {
            return None;
        }

        Some(vk::Extent2D {
            width: extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: extent
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
