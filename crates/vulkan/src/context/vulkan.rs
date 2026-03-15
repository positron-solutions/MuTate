// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # VkContext
//!
//! The entry and instance.  This module verifies their capabilities.  I'm sleepy.


use std::ffi::{c_void, CStr};

use ash::vk;

use mutate_assets as assets;

pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
}

impl VkContext {
    pub fn new () -> Self {
        let entry = unsafe { ash::Entry::load().expect("failed to load Vulkan library") };
        assert_loader_version(&entry);

        let available_entry_extensions = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")
        };

        let platform_ext = if cfg!(target_os = "linux") {
            if std::env::var("WAYLAND_DISPLAY").is_ok() {
                ash::vk::KHR_WAYLAND_SURFACE_NAME.as_ptr()
            } else {
                ash::vk::KHR_XLIB_SURFACE_NAME.as_ptr()
            }
        } else if cfg!(target_os = "macos") {
            ash::vk::MVK_MACOS_SURFACE_NAME.as_ptr()
        } else if cfg!(target_os = "ios") {
            ash::vk::MVK_IOS_SURFACE_NAME.as_ptr()
        } else if cfg!(target_os = "android") {
            ash::vk::KHR_ANDROID_SURFACE_NAME.as_ptr()
        } else if cfg!(target_os = "windows") {
            ash::vk::KHR_WIN32_SURFACE_NAME.as_ptr()
        } else {
            ash::vk::EXT_HEADLESS_SURFACE_NAME.as_ptr()
        };

        let required_exts = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),

            // Evidently this is a cool new way to tune validation layer settings.  Also see another
            // instance extension, VK_EXT_validation_features.
            // ash::vk::EXT_LAYER_SETTINGS_NAME.as_ptr(),

            platform_ext
        ];

        for &req in &required_exts {
            let found = available_entry_extensions.iter().any(|ext| unsafe {
                let ext_cstr = CStr::from_ptr(ext.extension_name.as_ptr());
                let req_cstr = CStr::from_ptr(req);
                ext_cstr == req_cstr
            });
            assert!(
                found,
                "Required Vulkan extension {} not found",
                unsafe { CStr::from_ptr(req).to_str().unwrap() }
            );
        }

        let app_info =
            vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));

        let validation_layers = [
            // #[cfg(debug_assertions)]
            // NOTE Leaving this on all the time because there are still issues in `--release`
            // builds and we need to default to leaving it on via the dev shells or something.
            c"VK_LAYER_KHRONOS_validation".as_ptr()
        ];

        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_exts)
            .enabled_layer_names(&validation_layers);

        let instance = unsafe { entry.create_instance(&instance_ci, None).unwrap() };


        Self {
            entry,
            instance,
        }
    }

    pub fn destroy(&self) {
        unsafe {self.instance.destroy_instance(None)}
    }
}

/// Runs a block with an initialized Vulkan context, handling creation and destruction automatically.
///
/// Two forms are available depending on whether you need access to the underlying [`VkContext`]:
///
/// # Usage
///
/// ```rust
/// // With context only:
/// with_context!(|context| {
///     // `context: DeviceContext` is available here
/// });
///
/// // With both context and vk_context:
/// with_context!(|context, vk_context| {
///     // `context: DeviceContext` is available here
///     // `vk_context: VkContext` is available here
/// });
/// ```
#[macro_export]
macro_rules! with_context {
    (|$context:ident| $($body:tt)*) => {{
        let mut vk_context = $crate::context::VkContext::new();
        let mut $context = $crate::context::DeviceContext::new(&vk_context);
        let __result = (|| { $($body)* })();
        $context.destroy();
        vk_context.destroy();
        __result
    }};
    (|$context:ident, mut $vk_context:ident| $($body:tt)*) => {{
        let mut $vk_context = $crate::context::VkContext::new();
        let mut $context = $crate::context::DeviceContext::new(&$vk_context);
        let __result = (|| { $($body)* })();
        $context.destroy();
        $vk_context.destroy();
        __result
    }};
}

fn assert_loader_version (entry: &ash::Entry) {
    let loader_version = unsafe {
        entry.try_enumerate_instance_version()
            .unwrap_or(Some(vk::make_api_version(0, 1, 0, 0)))
            .unwrap()
    };

    if loader_version < vk::make_api_version(0, 1, 3, 0) {
        panic!("Loader does not support Vulkan 1.3");
    }
}
