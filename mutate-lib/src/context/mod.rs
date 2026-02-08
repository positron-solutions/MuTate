// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Context
//!
//! Fundamentally required resources, including the entry, instance, hardware devices
//! are encapsulated by `VkContext`.
//!
//! Initializing a physical device for use results in a logical `ash::Device`, which is used in most
//! calls to Vulkan.
//!
//! *NEXT* The Devices and memory management are much more tightly bound together than the
//! `ash::Entry` and `ash::Instance`, so these will be separated when convenient.
pub mod queue;

use std::ffi::{c_void, CStr};

use ash::vk;

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
    pub descriptor_pool: vk::DescriptorPool,
}

static VALIDATION_LAYER: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

impl VkContext {
    /// Obtain an entry, instance, and initialized device.
    ///
    /// LIES *debugging:* In debug builds, validation layers are enabled.
    ///
    /// NEXT Device initialization should be moved into a separate method to support UIs that
    /// enumerate and may even switch devices.
    pub fn new() -> Self {
        let entry = unsafe { ash::Entry::load().expect("failed to load Vulkan library") };
        let available_exts = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to enumerate instance extensions")
        };

        // FIXME insufficiently accurate platform check
        assert!(
            available_exts.iter().any(|ext| unsafe {
                CStr::from_ptr(ext.extension_name.as_ptr()) == ash::vk::KHR_WAYLAND_SURFACE_NAME
            }),
            "Only xlib is currently supported"
        );

        let required_exts = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),
            ash::vk::KHR_XLIB_SURFACE_NAME.as_ptr(),
            ash::vk::KHR_WAYLAND_SURFACE_NAME.as_ptr(),
            // NEXT CLI switch gate
            ash::vk::EXT_DEBUG_UTILS_NAME.as_ptr(),
        ];

        // LIES needs env variable and config switch, MUTATE_VALIDATION, any non-empty value.
        let validation_layers = [VALIDATION_LAYER.as_ptr()];

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: required_exts.len() as u32,
            pp_enabled_extension_names: required_exts.as_ptr(),
            enabled_layer_count: validation_layers.len() as u32,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe { entry.create_instance(&create_info, None).unwrap() };

        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("No Vulkan devices")
        };
        // NEXT support choices, via config, environment, and heuristics (discrete vs on-CPU)!)
        let physical_device = physical_devices[0];

        let device_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            ash::vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
            ash::vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),
            ash::vk::KHR_DYNAMIC_RENDERING_NAME.as_ptr(),
            ash::vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_BUFFER_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_INDEXING_NAME.as_ptr(),
            ash::vk::KHR_PIPELINE_LIBRARY_NAME.as_ptr(),
            ash::vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr(),
            // ROLL holding off on this until other hardware vendors have supporting drivers
            // ash::vk::EXT_SHADER_OBJECT_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE1_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE2_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE3_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE4_NAME.as_ptr(),
            // ROLL VK_EXT_present_timing is still too new.  Support must be dynamic and... someone
            // needs a card / driver that supports it to develop the support.
            // ash::vk::EXT_PRESENT_TIMING_NAME.as_ptr(),
            ash::vk::KHR_PRESENT_WAIT_NAME.as_ptr(),
            ash::vk::KHR_PRESENT_ID_NAME.as_ptr(),
        ];

        let mut present_wait_id = vk::PhysicalDevicePresentIdFeaturesKHR {
            present_id: vk::TRUE,
            ..Default::default()
        };

        let mut present_wait = vk::PhysicalDevicePresentWaitFeaturesKHR {
            p_next: &mut present_wait_id as *mut _ as *mut c_void,
            present_wait: vk::TRUE,
            ..Default::default()
        };

        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features {
            p_next: &mut present_wait as *mut _ as *mut c_void,
            synchronization2: vk::TRUE,
            ..Default::default()
        };

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures {
            p_next: &mut sync2_features as *mut _ as *mut c_void,
            dynamic_rendering: vk::TRUE,
            ..Default::default()
        };

        let mut features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut dynamic_rendering_features as *mut _ as *mut c_void,
            ..Default::default()
        };

        let queue_families = queue::QueueFamilies::new(&instance, &physical_device);
        let queue_priorities = [1.0];
        let queue_infos = queue_families.queue_infos(&queue_priorities);
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            pp_enabled_extension_names: device_extensions.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            p_next: &mut features2 as *mut _ as *mut c_void,
            ..Default::default()
        };
        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queues = queue::Queues::new(&device, queue_families);

        // DEBT memory management.  We obviously don't want to recreate descriptor pools for every visual.
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 32, // we're promoting bindless, so this is plenty?
        }];

        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            flags: vk::DescriptorPoolCreateFlags::empty(),
            ..Default::default()
        };

        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        Self {
            entry,
            instance,
            physical_device,
            device,
            surface_loader,

            queues,

            descriptor_pool,
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    // XXX in reality, this consumes the context, but ownership friction needs worked out.
    pub fn destroy(&self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.queues.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None)
        };
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
