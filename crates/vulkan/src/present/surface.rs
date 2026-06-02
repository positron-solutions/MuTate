// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Surface
//!
//! Whenever a window configuration changes, we may need to refresh our view of the surface.  This
//! module pulls together the decisions and data into a single [`Surface`] type.
//!
//! Polling surface capabilities enables us to build a contract with the downstream program that
//! uses our output.  The contract covers format, transform, compositing, output data layout, output
//! size, and the synchronization scheme.  Vulkan uses the surface to abstract over platform
//! differences and different downstream users.
//!
//! ## Usage
//!
//! Obtain a raw surface handle.  With winit, this can be done by calling `surface` on the Vulkan
//! instance.  The raw surface can be used to initialize the `Surface`.  Use this to obtain a
//! swapchain and request queues with presentation support.  The `Surface` implements `Deref` to
//! [`ash::vk::Surface`] for use of raw ash.
//!
//! When you get window resize events or swapchain errors such as
//! [`SwapchainSuboptimal`](crate::VulkanError::SwapchainSuboptimal), it's time to update the view
//! of the capabilities.  If rendering to a window, just call [`update`] with the window.  If not
//! using a window, you can pass a literal fallback extent that will only be used if the platform
//! doesn't attach surface dimensions the surface capabilities directly.  The updated `Surface` can
//! then be used to re-create the swapchain.
//!
//! ## Surface & Platform Dependency Story
//!
//! The surface contract begins affecting decisions very early, affecting all initialization phases
//! of a typical Vulkan application.
//!
//! - Surfaces are usually created from raw window handles obtained from winit.  To make surface
//!   from handle, different platforms require the Vulkan instance to have different instance
//!   extensions enabled.  As a consequence the Vulkan instance module uses a winit event loop to
//!   make decisions about later surface support before we even create the Vulkan instance.
//!
//! - Presenting on a queue then requires a queue that supports the surface, so the instance will
//!   look at the surface when filtering physical devices, before even deciding which queues to
//!   enable during logical device creation.
//!
//! - Creating a [`Window`](winit::window::Window) will begin to talk to the display platform.
//!   Aside from naming title bars and making fullscreen, we also use the Window to obtain the raw
//!   handles that our properly equipped instance can convert to a [`ash::vk::SurfaceKHR`].
//!
//! - The memory we write out to uses a ring of several images, the swapchain.  The swapchain
//!   creation must know about the surface details (the subject of this module) and any events that
//!   change these details usually require swapchain recreation.
//!
//! This module focuses on the surface interrogation necessary to inform swapchain creation and
//! re-creation.  Nonetheless, the surface and dependencies for the surface can be found from the
//! earliest stages of the application.

// XXX We really need to propagate the surface update back to the application so that provisioned
// things can be re-provisioned.  A runtime will eventually handle the threading, but for now...

#[cfg(feature = "winit")]
use winit::window::Window;

use crate::internal::*;

/// The set of decisions resulting from surface updates, obtained from querying the Vulkan surface
/// capabilities and selecting in our preference order the values we support.
#[derive(Clone, Debug)]
pub struct SurfaceCaps {
    /// MAILBOX if supported.  FIFO_RELAXED is pretty close.
    present_mode: vk::PresentModeKHR,
    swapchain_image_count: u32,

    /// Color data format and color space mapping for display.
    pub format: vk::SurfaceFormatKHR,
    /// How the surface output will be blended by any compositor.
    pub composite_alpha: vk::CompositeAlphaFlagsKHR,
    /// Such as screen rotation and scaling on mobile.
    pub pre_transform: vk::SurfaceTransformFlagsKHR,
    /// Resolved swapchain image size.
    pub extent: vk::Extent2D,
}

// We want to unify the signatures over the source of fallback extent to support windowed and
// non-windowed cases.
/// On windowed platforms, we use the window as the extent source on platforms that do not set the
/// extent directly on the Vulkan surface capabilities query result.  For non-window platform use,
/// you can supply a fallback for that will be used if the surface capabilities cannot.
pub enum ExtentSource<'a> {
    #[cfg(feature = "winit")]
    Window(&'a Window),
    Fallback(vk::Extent2D),
}

// Enable callers to skip naming the Enum variants.
#[cfg(feature = "winit")]
impl<'a> From<&'a Window> for ExtentSource<'a> {
    fn from(w: &'a Window) -> Self {
        ExtentSource::Window(w)
    }
}

impl From<vk::Extent2D> for ExtentSource<'_> {
    fn from(e: vk::Extent2D) -> Self {
        ExtentSource::Fallback(e)
    }
}

/// A Wrapper for [`ash::vk::SurfaceKHR`] that also stores results from a
/// [`VkSurfaceCapabilitiesKHR`](https://docs.vulkan.org/refpages/latest/refpages/source/VkSurfaceCapabilitiesKHR.html)
/// query and a surface loader to conveniently use it.
pub struct Surface {
    /// Raw Vulkan surface handle.
    raw: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
    /// Last updated view of our choices to support for this surface.
    pub caps: SurfaceCaps,
}

impl Surface {
    /// Create a [`Surface`].  You can pass either a [`Window`](winit::window::Window) or a raw
    /// [`Extent2D`](vk::Extent2D) as the [`ExtentSource`] to be used whenever the platform surface
    /// provider does not set the `current_extent` field.
    pub fn new<'a>(
        instance: &Instance,
        device_context: &DeviceContext,
        surface: vk::SurfaceKHR,
        extent_source: impl Into<ExtentSource<'a>>,
    ) -> Result<Self, VulkanError> {
        let extent_source = extent_source.into();
        let surface_loader = instance.surface_loader();
        let raw_caps =
            Self::fetch_raw_caps(&surface_loader, device_context.physical_device, surface)?;
        let extent = Self::resolve_extent(&raw_caps, extent_source)?;
        let caps = Self::resolve_caps(
            &surface_loader,
            device_context.physical_device,
            surface,
            &raw_caps,
            extent,
        )?;
        Ok(Self {
            raw: surface,
            surface_loader,
            caps,
        })
    }

    /// Re-query the surface capabilities
    ///
    /// Call this on resize events or after a
    /// [`SwapchainSuboptimal`](crate::VulkanError::SwapchainSuboptimal) error.  After more serious
    /// errors such as [`SurfaceLost`](crate::VulkanError::SurfaceLost)
    pub fn update<'a>(
        &mut self,
        device_context: &DeviceContext,
        extent_source: impl Into<ExtentSource<'a>>,
    ) -> Result<vk::Extent2D, VulkanError> {
        let extent_source = extent_source.into();
        let raw_caps = Self::fetch_raw_caps(
            &self.surface_loader,
            device_context.physical_device,
            self.raw,
        )?;
        let extent = Self::resolve_extent(&raw_caps, extent_source)?;
        self.caps = Self::resolve_caps(
            &self.surface_loader,
            device_context.physical_device,
            self.raw,
            &raw_caps,
            extent,
        )?;
        Ok(self.caps.extent)
    }

    /// Query [`vk::SurfaceCapabilitiesKHR`] from the driver.  If queried caps advertise zero max
    /// extent, fail with [`VulkanError::SurfaceLost`] to notify caller to rebuild farther upstream
    /// rather than retrying on next events.
    fn fetch_raw_caps(
        surface_loader: &ash::khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceCapabilitiesKHR, VulkanError> {
        let caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(|e| {
                    let err = VulkanError::from(e);
                    eprintln!("warning: surface capabilities query failed: {}", err);
                    err
                })?
        };

        if caps.max_image_extent.width == 0 || caps.max_image_extent.height == 0 {
            eprintln!("warning: surface caps report zero max_image_extent");
            return Err(VulkanError::DegenerateExtent {
                width: caps.max_image_extent.width,
                height: caps.max_image_extent.height,
            });
        }
        Ok(caps)
    }

    /// Find the extent either via the capabilities or the supplied [`ExtentSource`] on platforms
    /// that do not set it on the capabilities directly.
    fn resolve_extent(
        caps: &vk::SurfaceCapabilitiesKHR,
        source: ExtentSource<'_>,
    ) -> Result<vk::Extent2D, VulkanError> {
        // Platforms that set current_extent directly (X11, Windows) use u32::MAX as the sentinel
        // meaning "not set"; Wayland is the common case that leaves it unset and expects the
        // application to supply the extent from the window.

        // FIXME we have not forwarded any concrete indication that Wayland is the current platform
        // extension, and we definitely know at this point if the instance has Wayland surface
        // support as a required extension, so while this check is fine given that `u32::MAX` is
        // obviously not the extent, we can't warn if some other platform does something weird and
        // we don't know what platform we're on to decide if the behavior is expected.
        let caps_extent = (caps.current_extent.width != u32::MAX).then_some(caps.current_extent);
        let raw = match source {
            #[cfg(feature = "winit")]
            ExtentSource::Window(window) => caps_extent.unwrap_or_else(|| {
                let s = window.inner_size();
                vk::Extent2D {
                    width: s.width,
                    height: s.height,
                }
            }),
            ExtentSource::Fallback(fallback) => caps_extent.unwrap_or(fallback),
        };

        if raw.width == 0 || raw.height == 0 {
            // MAYBE the window minimized claim is an inference?  The caller must react since we're
            // not configuring a zero-size swapchain under any circumstances, but the error
            // condition may be... optimistically certain.
            return Err(match source {
                #[cfg(feature = "winit")]
                ExtentSource::Window(_) => VulkanError::WindowMinimized,
                ExtentSource::Fallback(_) => VulkanError::DegenerateExtent {
                    width: raw.width,
                    height: raw.height,
                },
            });
        }
        Ok(raw)
    }

    /// Query surface caps and build a [`SurfaceCaps`] using our preference order of the values we
    /// can coherently support.
    fn resolve_caps(
        surface_loader: &ash::khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        raw_caps: &vk::SurfaceCapabilitiesKHR,
        extent: vk::Extent2D,
    ) -> Result<SurfaceCaps, VulkanError> {
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
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
            // XXX change this to a failure
            let fallback = formats[0];
            eprintln!("warning: fallback surface format selected: {:?}", fallback);
            fallback
        });

        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
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
        .find(|mode| raw_caps.supported_composite_alpha.contains(*mode))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::INHERIT);

        // XXX Make an actual decision
        let pre_transform = raw_caps.current_transform;

        let swapchain_image_count = {
            let mode_minimum = match present_mode {
                vk::PresentModeKHR::MAILBOX => 3,
                _ => 2,
            };
            // Even if the compositor is happy to use two images, we need at least three to avoid
            // lapping the semaaphores that synchronize them.
            let desired = mode_minimum.max(raw_caps.min_image_count);
            if raw_caps.max_image_count == 0 {
                desired
            } else {
                // We only use up to four semaphores, so more than four images can't be
                // synchronized.
                desired.min(raw_caps.max_image_count.min(4))
            }
        };

        // Extent degeneracy was checked upstream.  This clamp just complies with the spec.
        let extent = vk::Extent2D {
            width: extent.width.clamp(
                raw_caps.min_image_extent.width,
                raw_caps.max_image_extent.width,
            ),
            height: extent.height.clamp(
                raw_caps.min_image_extent.height,
                raw_caps.max_image_extent.height,
            ),
        };

        Ok(SurfaceCaps {
            format,
            present_mode,
            composite_alpha,
            pre_transform,
            swapchain_image_count,
            extent,
        })
    }

    /// Convert the [`SurfaceCaps`] capability snapshot into swapchain creation info.
    pub fn swapchain_ci(&self) -> vk::SwapchainCreateInfoKHR {
        let caps = &self.caps;
        vk::SwapchainCreateInfoKHR::default()
            .surface(self.raw)
            .composite_alpha(caps.composite_alpha)
            .image_color_space(caps.format.color_space)
            .image_extent(caps.extent)
            .image_format(caps.format.format)
            .min_image_count(caps.swapchain_image_count)
            .pre_transform(caps.pre_transform)
            .present_mode(caps.present_mode)
            // MAYBE these extra settings are digging into the swapchain responsibilities a bit.  It
            // might be appropriate to only coerce this far, enabling the decisions to be converted
            // to CI while also keeping fields private while letting the swapchain dictate remaining
            // configuration.
            .clipped(true)
            .flags(vk::SwapchainCreateFlagsKHR::DEFERRED_MEMORY_ALLOCATION_EXT)
            .image_array_layers(1)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            // XXX Can we add more flags here?
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
    }

    /// Get the chosen supported format.
    pub fn format(&self) -> vk::Format {
        self.caps.format.format
    }

    /// Get the resolved surface size.
    pub fn extent(&self) -> vk::Extent2D {
        self.caps.extent
    }

    /// Return the raw surface.
    pub fn as_raw(&self) -> &vk::SurfaceKHR {
        &self.raw
    }

    /// Return the raw surface.
    pub fn into_raw(self) -> vk::SurfaceKHR {
        self.raw
    }

    pub fn destroy(&self) {
        unsafe {
            self.surface_loader.destroy_surface(self.raw, None);
        }
    }
}

impl std::ops::Deref for Surface {
    type Target = vk::SurfaceKHR;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}
