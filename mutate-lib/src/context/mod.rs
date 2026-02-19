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
pub mod descriptors;

use std::ffi::{c_void, CStr};

use ash::{ext::subgroup_size_control, vk};

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
}

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

impl VkContext {
    /// Obtain an entry, instance, and initialized device.
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

        // FIXME these are not the only ones we require.
        let required_exts = [
            ash::vk::KHR_SURFACE_NAME.as_ptr(),
            platform_ext
        ];

        for &req in &required_exts {
            let found = available_exts.iter().any(|ext| unsafe {
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
            VALIDATION_LAYER.as_ptr()
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

        let device_extensions = [
            ash::vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            ash::vk::KHR_SYNCHRONIZATION2_NAME.as_ptr(),
            ash::vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
            ash::vk::KHR_DYNAMIC_RENDERING_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            ash::vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),
            ash::vk::KHR_PIPELINE_LIBRARY_NAME.as_ptr(),
            // NOTE Descriptor sets are kind of a "choose your fighter" moment in using Vulkan.  We
            // are going for a bindless scheme where we use 1-2 descriptor sets per scene, one
            // descriptor slot per type, and in each slot, we hold a descriptor array with the
            // maximum number of things we might actually use simultaneously.
            //
            // Long story short, if we have a type, we need to be able to store it in a descriptor
            // array, not a single slot of a descriptor set, which would force us to overwrite those
            // descriptors a lot.  Push descriptors can still be used.  Push constants can also be
            // used.  For the most part, get them into an array and then we only have to dynamically
            // give the indexes (and types!) to the shader programs.
            ash::vk::EXT_INLINE_UNIFORM_BLOCK_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_BUFFER_NAME.as_ptr(),
            ash::vk::EXT_DESCRIPTOR_INDEXING_NAME.as_ptr(),
            ash::vk::KHR_DESCRIPTOR_UPDATE_TEMPLATE_NAME.as_ptr(),
            ash::vk::KHR_PUSH_DESCRIPTOR_NAME.as_ptr(),
            ash::vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
            // Not strictly in use, but anticipated
            ash::vk::KHR_SHADER_NON_SEMANTIC_INFO_NAME.as_ptr(),
            // ash::vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr(),
            ash::vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            // ROLL holding off on this until other hardware vendors have supporting drivers
            // ash::vk::EXT_SHADER_OBJECT_NAME,
            ash::vk::KHR_MAINTENANCE1_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE2_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE3_NAME.as_ptr(),
            ash::vk::KHR_MAINTENANCE4_NAME.as_ptr(),
            // ROLL VK_EXT_present_timing is still too new.  Support must be dynamic and... someone
            // needs a card / driver that supports it to develop the support.
            // ash::vk::EXT_PRESENT_TIMING_NAME,
            ash::vk::KHR_PRESENT_WAIT_NAME.as_ptr(),
            ash::vk::KHR_PRESENT_ID_NAME.as_ptr(),
        ];

        let mut pwid_features = vk::PhysicalDevicePresentIdFeaturesKHR::default();
        pwid_features.present_id = vk::TRUE;

        let mut pw_features = vk::PhysicalDevicePresentWaitFeaturesKHR::default();
        pw_features.present_wait = vk::TRUE;

        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default();
        sync2_features.synchronization2 = vk::TRUE;

        let mut dr_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        dr_features.dynamic_rendering = vk::TRUE;

        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        bda_features.buffer_device_address = vk::TRUE;

        let mut di_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        di_features.shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
        di_features.shader_storage_buffer_array_non_uniform_indexing = vk::TRUE;
        di_features.shader_storage_image_array_non_uniform_indexing = vk::TRUE;
        di_features.descriptor_binding_partially_bound = vk::TRUE;
        di_features.runtime_descriptor_array = vk::TRUE;
        di_features.descriptor_binding_variable_descriptor_count = vk::TRUE;

        let mut db_features = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
        db_features.descriptor_buffer = vk::TRUE;
        db_features.descriptor_buffer_capture_replay = vk::TRUE;
        db_features.descriptor_buffer_image_layout_ignored = vk::TRUE;

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut bda_features)
            .push_next(&mut dr_features)
            .push_next(&mut sync2_features)
            .push_next(&mut pw_features)
            .push_next(&mut pwid_features)
            .push_next(&mut di_features)
            .push_next(&mut db_features);

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_context_lifecycle() {
        let vk_context = VkContext::new();
        vk_context.destroy();
    }
}
