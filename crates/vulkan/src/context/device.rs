// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # DeviceContext
//!
//! Fully initialized logical devices have a lot of associated things.  We know their queues, memory
//! capability, and we have set up their descriptor tables.  The [`DeviceContext`] pulls all of
//! these behaviors together into one unified lifecycle.

use std::ffi::{c_void, CStr};

use ash::vk;

use mutate_assets as assets;

use super::descriptors;
use super::queue;
use super::vulkan;

pub struct DeviceContext {
    // XXX Wrap it up dawg
    pub physical_device: vk::PhysicalDevice,
    /// Vulkan logical device
    pub device: ash::Device,
    /// Queues and command buffers for device in use.
    pub queues: queue::Queues,
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Descriptor table for
    pub descriptors: descriptors::Descriptors,

    // XXX some other higher context?
    /// Initialized assets
    pub assets: assets::AssetDirs,
}

impl DeviceContext {
    /// Obtain an entry, instance, and initialized device.
    ///
    /// NEXT Device initialization should be moved into a separate method to support UIs that
    /// enumerate and may even switch devices.
    pub fn new(crap_o_context: &vulkan::VkContext) -> Self {
        let vulkan::VkContext { entry, instance } = &crap_o_context;

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
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),
            vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
            // NEXT better debug gating (see validation layer activation above).
            // Enables some debug functionality in shaders.
            vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr(),
            vk::EXT_TOOLING_INFO_NAME.as_ptr(),
            // MAYBE I might just need to install something, but this fails at runtime on my machine.
            // vk::EXT_DEBUG_UTILS_NAME.as_ptr(),

            // MAYBE If we start running into lots of pipeline creation costs for slight variants,
            // we are advised to look at this extension.
            // vk::EXT_GRAPHICS_PIPELINE_LIBRARY_NAME.as_ptr(),
            // ROLL holding off on this until other hardware vendors have supporting drivers.  This
            // is another path to reducing the cost of pipeline combinatorics.
            // vk::EXT_SHADER_OBJECT_NAME,

            // "gives an implementation the opportunity to reduce the number of indirections an
            // implementation takes to access uniform values, when only a few values are used"
            vk::EXT_INLINE_UNIFORM_BLOCK_NAME.as_ptr(),
            // MAYBE So we can proactively change our memory behavior, downsampling etc.
            // vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            // vk::EXT_MEMORY_PRIORITY_NAME.as_ptr(),
            vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
            // ROLL VK_EXT_present_timing is still too new.  Support must be dynamic and... someone
            // needs a card / driver that supports it to develop the support.
            // vk::EXT_PRESENT_TIMING_NAME,
            vk::KHR_PRESENT_WAIT_NAME.as_ptr(),
            vk::KHR_PRESENT_ID_NAME.as_ptr(),
        ];

        let available_device_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
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

        let mut pwid_features = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);

        let mut pw_features =
            vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);

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
        let descriptors = descriptors::Descriptors::new(&device).unwrap();

        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Self {
            physical_device,
            device,
            memory_props,
            queues,
            descriptors,

            // XXX there is another context
            assets: assets::AssetDirs::new(),
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn bind_sampled_image(
        &mut self,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> descriptors::SampledImageIndex {
        let device = &self.device;
        let descriptors = &mut self.descriptors;
        descriptors.bind_sampled_image(device, view, layout)
    }

    // XXX in reality, this consumes the context, but ownership friction needs worked out.
    pub fn destroy(&self) {
        unsafe {
            self.descriptors.destroy(&self.device);
            self.queues.destroy(&self.device);
            self.device.destroy_device(None);
        };
    }
}

fn assert_physical_device_version(instance: &ash::Instance, physical_device: vk::PhysicalDevice) {
    let props = unsafe { instance.get_physical_device_properties(physical_device) };

    let api_version = props.api_version;

    let major = vk::api_version_major(api_version);
    let minor = vk::api_version_minor(api_version);
    let patch = vk::api_version_patch(api_version);

    if major == 1 && minor < 3 {
        panic!("Vulkan 1.3 required, found {}.{}.{}", major, minor, patch);
    }
}

// NEXT return an error lol
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
    assert_eq!(
        features_1_1.uniform_and_storage_buffer16_bit_access,
        vk::TRUE
    );

    assert_eq!(features_1_2.buffer_device_address, vk::TRUE);
    assert_eq!(features_1_2.descriptor_binding_partially_bound, vk::TRUE);
    assert_eq!(
        features_1_2.descriptor_binding_sampled_image_update_after_bind,
        vk::TRUE
    );
    assert_eq!(
        features_1_2.descriptor_binding_storage_buffer_update_after_bind,
        vk::TRUE
    );
    assert_eq!(
        features_1_2.descriptor_binding_storage_image_update_after_bind,
        vk::TRUE
    );
    assert_eq!(
        features_1_2.descriptor_binding_variable_descriptor_count,
        vk::TRUE
    );
    assert_eq!(features_1_2.runtime_descriptor_array, vk::TRUE);
    assert_eq!(features_1_2.scalar_block_layout, vk::TRUE);
    assert_eq!(features_1_2.shader_float16, vk::TRUE);
    assert_eq!(features_1_2.shader_int8, vk::TRUE);
    assert_eq!(
        features_1_2.shader_sampled_image_array_non_uniform_indexing,
        vk::TRUE
    );
    assert_eq!(
        features_1_2.shader_storage_buffer_array_non_uniform_indexing,
        vk::TRUE
    );
    assert_eq!(
        features_1_2.shader_storage_image_array_non_uniform_indexing,
        vk::TRUE
    );
    assert_eq!(features_1_2.storage_buffer8_bit_access, vk::TRUE);
    assert_eq!(features_1_2.storage_push_constant8, vk::TRUE);
    assert_eq!(
        features_1_2.uniform_and_storage_buffer8_bit_access,
        vk::TRUE
    );

    assert_eq!(features_1_3.maintenance4, vk::TRUE);
    assert_eq!(features_1_3.synchronization2, vk::TRUE);
}
