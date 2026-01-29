// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// VkContext encapsulates the global resources, independent of presentation mode.  This includes
/// hardware and create-once abstractions of hardware.
use std::ffi::{c_void, CStr};

use ash::vk;

// Hardware, drivers, and the lowest level abstractions of hardware should be encapsulated within
// this module.  Devices may need to be split out later for use of multiple devices.  Memory
// likewise may be globally managed but will stay with associated devices.
pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub surface_loader: ash::khr::surface::Instance,

    graphics_queue: vk::Queue,
    #[allow(dead_code)]
    compute_queue: vk::Queue,
    #[allow(dead_code)]
    transfer_queue: vk::Queue,

    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
}

static VALIDATION_LAYER: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

impl VkContext {
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
        let physical_device = physical_devices[0];

        let queue_family_index = unsafe {
            instance
                .get_physical_device_queue_family_properties(physical_device)
                .iter()
                .enumerate()
                .find_map(|(index, q)| {
                    if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        Some(index as u32)
                    } else {
                        None
                    }
                })
                .expect("No graphics queue family found")
        };

        let queue_priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

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

        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: queue_info.as_ptr(),
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
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let command_pool_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        };

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        Self {
            entry,
            instance,
            physical_device,
            device,
            surface_loader,

            graphics_queue: queue.clone(),
            compute_queue: queue.clone(),
            transfer_queue: queue,

            command_pool,
            queue_family_index,
        }
    }

    pub fn graphics_queue(&self) -> &vk::Queue {
        &self.graphics_queue
    }

    pub fn graphics_pool(&self) -> &vk::CommandPool {
        &self.command_pool
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    // XXX in reality, this consumes the context, but ownership friction needs worked out.
    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
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
