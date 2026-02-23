// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Vulkan Context
//!
//! Fundamentally required resources, including the entry, instance, hardware devices
//! are encapsulated by `VkContext`.
//!
//! *NEXT* The Devices and memory management are much more tightly bound together than the
//! `ash::Entry` and `ash::Instance`, so these will be separated when convenient.
//!
//! ## Enabled Features
//!
//! We aim to support a minimum set of modern tactics to offer a complete, high performance
//! experience:
//!
//! - Buffer device address.
//! - One big descriptor set with one descriptor array per type (bindless).
//! - Flexible push constants & UBOs with scalar layout and 8/16-bit support.
//! - Vulkan 1.3+ minimum support, 1.4 when reasonable.
//! - Dynamic rendering

pub mod descriptors;
pub mod queue;

use std::ffi::{c_void, CStr};

use ash::{ext::subgroup_size_control, vk};

use mutate_assets as assets;

pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    /// Used to access surface creation functions
    pub surface_loader: ash::khr::surface::Instance,

    pub physical_device: vk::PhysicalDevice,
    /// Vulkan logical device
    pub device: ash::Device,
    /// Queues and command buffers for device in use.
    pub queues: queue::Queues,
    #[deprecated(note = "Moving to bindless.  Don't write new descriptors.")]
    pub descriptor_pool: vk::DescriptorPool,
    descriptors: descriptors::Descriptors,

    /// Initialized assets
    assets: assets::AssetDirs,
}

impl VkContext {
    /// Obtain an entry, instance, and initialized device.
    ///
    /// NEXT Device initialization should be moved into a separate method to support UIs that
    /// enumerate and may even switch devices.
    pub fn new() -> Self {
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

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("No Vulkan devices")
        };
        // NEXT support choices, via config, environment, and heuristics (discrete vs on-CPU)!)
        let physical_device = physical_devices[0];

        assert_physical_device_version(&instance, physical_device);
        assert_physical_device_features(&instance, physical_device);

        let device_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME.as_ptr(),


            ash::vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),

            ash::vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),

            // NEXT better debug gating (see validation layer activation above).
            // Enables some debug functionality in shaders.
            ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr(),
            ash::vk::EXT_TOOLING_INFO_NAME.as_ptr(),
            // MAYBE I might just need to install something, but this fails at runtime on my machine.
            // ash::vk::EXT_DEBUG_UTILS_NAME.as_ptr(),

            // MAYBE If we start running into lots of pipeline creation costs for slight variants,
            // we are advised to look at this extension.
            // ash::vk::EXT_GRAPHICS_PIPELINE_LIBRARY_NAME.as_ptr(),
            // ROLL holding off on this until other hardware vendors have supporting drivers.  This
            // is another path to reducing the cost of pipeline combinatorics.
            // ash::vk::EXT_SHADER_OBJECT_NAME,

            // "gives an implementation the opportunity to reduce the number of indirections an
            // implementation takes to access uniform values, when only a few values are used"
            ash::vk::EXT_INLINE_UNIFORM_BLOCK_NAME.as_ptr(),

            // MAYBE So we can proactively change our memory behavior, downsampling etc.
            // ash::vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            // ash::vk::EXT_MEMORY_PRIORITY_NAME.as_ptr(),

            ash::vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
            // ROLL VK_EXT_present_timing is still too new.  Support must be dynamic and... someone
            // needs a card / driver that supports it to develop the support.
            // ash::vk::EXT_PRESENT_TIMING_NAME,
            ash::vk::KHR_PRESENT_WAIT_NAME.as_ptr(),
            ash::vk::KHR_PRESENT_ID_NAME.as_ptr(),
        ];

        let available_device_extensions = unsafe {
            instance.enumerate_device_extension_properties(physical_device)
                .expect("Failed to enumerate device extensions")
        };

        for &req in &device_extensions {
            let found = available_device_extensions.iter().any(|ext| unsafe {
                let ext_cstr = CStr::from_ptr(ext.extension_name.as_ptr());
                let req_cstr = CStr::from_ptr(req);
                ext_cstr == req_cstr
            });

            assert!(
                found,
                "Required Vulkan device extension {} not found",
                unsafe { CStr::from_ptr(req).to_str().unwrap() }
            );
        }

        let mut pwid_features = vk::PhysicalDevicePresentIdFeaturesKHR::default()
            .present_id(true);

        let mut pw_features = vk::PhysicalDevicePresentWaitFeaturesKHR::default()
            .present_wait(true);

        let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .storage_push_constant16(true)
            .storage_input_output16(true)
            .uniform_and_storage_buffer16_bit_access(true);

        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .runtime_descriptor_array(true)
            .scalar_block_layout(true)
            .shader_float16(true)
            .shader_int8(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .storage_buffer8_bit_access(true)
            .storage_push_constant8(true)
            .uniform_and_storage_buffer8_bit_access(true);

        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .maintenance4(true);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().shader_int16(true))
            .push_next(&mut pw_features)
            .push_next(&mut pwid_features)
            .push_next(&mut features_1_3)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_1);

        let queue_families = queue::QueueFamilies::new(&instance, &physical_device);
        let queue_priorities = [1.0];
        let queue_cis = queue_families.queue_cis(&queue_priorities);
        let mut device_info = vk::DeviceCreateInfo {
            ..Default::default()
        }
            .push_next(&mut features2)
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&device_extensions);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queues = queue::Queues::new(&device, queue_families);

        // DEBT Max descriptor size calculation / management.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 256
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 256
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 256
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 256
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        let descriptors = descriptors::Descriptors::new(&device, &descriptor_pool).unwrap();

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        Self {
            entry,
            instance,
            physical_device,
            device,
            surface_loader,

            queues,

            descriptor_pool,
            descriptors,

            assets: assets::AssetDirs::new(),
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    // XXX in reality, this consumes the context, but ownership friction needs worked out.
    pub fn destroy(&self) {
        unsafe {
            self.descriptors.destroy(&self.device);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.queues.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None)
        };
    }
}

fn assert_physical_device_version (instance: &ash::Instance, physical_device: vk::PhysicalDevice) {
    let props = unsafe {
        instance.get_physical_device_properties(physical_device)
    };

    let api_version = props.api_version;

    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    let patch = vk::api_version_patch(api_version);

    if major == 1 && minor < 3 {
        panic!("Vulkan 1.3 required, found {}.{}.{}", major, minor, patch);
    }
}

fn assert_physical_device_features(instance: &ash::Instance, physical_device: vk::PhysicalDevice) {
    let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default();

    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(vk::PhysicalDeviceFeatures::default())
        .push_next(&mut features_1_3)
        .push_next(&mut features_1_2)
        .push_next(&mut features_1_1);

    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features2);
    }

    assert_eq!(features2.features.shader_int16, vk::TRUE);

    assert_eq!(features_1_1.storage_buffer16_bit_access, vk::TRUE);
    assert_eq!(features_1_1.storage_input_output16, vk::TRUE);
    assert_eq!(features_1_1.storage_push_constant16, vk::TRUE);
    assert_eq!(features_1_1.uniform_and_storage_buffer16_bit_access, vk::TRUE);

    assert_eq!(features_1_2.buffer_device_address, vk::TRUE);
    assert_eq!(features_1_2.descriptor_binding_partially_bound, vk::TRUE);
    assert_eq!(features_1_2.descriptor_binding_sampled_image_update_after_bind, vk::TRUE);
    assert_eq!(features_1_2.descriptor_binding_storage_buffer_update_after_bind, vk::TRUE);
    assert_eq!(features_1_2.descriptor_binding_storage_image_update_after_bind, vk::TRUE);
    assert_eq!(features_1_2.descriptor_binding_variable_descriptor_count, vk::TRUE);
    assert_eq!(features_1_2.runtime_descriptor_array, vk::TRUE);
    assert_eq!(features_1_2.scalar_block_layout , vk::TRUE);
    assert_eq!(features_1_2.shader_float16, vk::TRUE);
    assert_eq!(features_1_2.shader_int8, vk::TRUE);
    assert_eq!(features_1_2.shader_sampled_image_array_non_uniform_indexing, vk::TRUE);
    assert_eq!(features_1_2.shader_storage_buffer_array_non_uniform_indexing, vk::TRUE);
    assert_eq!(features_1_2.shader_storage_image_array_non_uniform_indexing, vk::TRUE);
    assert_eq!(features_1_2.storage_buffer8_bit_access, vk::TRUE);
    assert_eq!(features_1_2.storage_push_constant8, vk::TRUE);
    assert_eq!(features_1_2.uniform_and_storage_buffer8_bit_access, vk::TRUE);

    assert_eq!(features_1_3.maintenance4, vk::TRUE);
    assert_eq!(features_1_3.synchronization2, vk::TRUE);
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_context_lifecycle() {
        let vk_context = VkContext::new();
        vk_context.destroy();
    }
}
