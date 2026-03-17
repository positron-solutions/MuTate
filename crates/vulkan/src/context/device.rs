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
use super::vulkan::{self, HasPresentation, NoPresentation, SupportedDevice};

pub struct DeviceContext {
    // XXX Wrap it up dawg
    pub physical_device: vk::PhysicalDevice,
    /// Vulkan logical device
    pub device: ash::Device,
    /// Queues family info.
    pub queues: queue::Queues,
    // NEXT work on abstracting this to memory decisions.
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
    pub fn new(
        vk_context: &vulkan::VkContext,
        supported_device: SupportedDevice<HasPresentation>,
    ) -> Self {
        let vulkan::VkContext { entry, instance } = &vk_context;
        let physical_device = supported_device.device().clone();

        // DEBT currently the requirements and support checks are all hardcoded.  There is duplicate
        // information in side of VkContext that needs to be runtime decided and then passed into
        // this function to avoid the hardcode.
        let device_extensions = [
            vk::KHR_SWAPCHAIN_NAME.as_ptr(),
            // MAYBE this is Windows only?  Evidently only old windows?
            // vk::EXT_FULL_SCREEN_EXCLUSIVE_NAME.as_ptr(),
            vk::EXT_EXTENDED_DYNAMIC_STATE_NAME.as_ptr(),
            vk::EXT_EXTENDED_DYNAMIC_STATE2_NAME.as_ptr(),
            vk::EXT_EXTENDED_DYNAMIC_STATE3_NAME.as_ptr(),
            // XXX Remove / redundant
            // vk::KHR_BUFFER_DEVICE_ADDRESS_NAME.as_ptr(),
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
            // XXX redundant
            // vk::EXT_INLINE_UNIFORM_BLOCK_NAME.as_ptr(),
            // MAYBE So we can proactively change our memory behavior, downsampling etc.
            // vk::EXT_MEMORY_BUDGET_NAME.as_ptr(),
            // vk::EXT_MEMORY_PRIORITY_NAME.as_ptr(),
            // XXX redundant
            // vk::KHR_TIMELINE_SEMAPHORE_NAME.as_ptr(),
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
            .shader_draw_parameters(true)
            .storage_buffer16_bit_access(true)
            .storage_push_constant16(true)
            // XXX remove this
            // .storage_input_output16(true)
            .uniform_and_storage_buffer16_bit_access(true);

        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_indexing(true)
            .draw_indirect_count(true)
            .runtime_descriptor_array(true)
            .scalar_block_layout(true)
            .shader_float16(true)
            .shader_int8(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .shader_uniform_buffer_array_non_uniform_indexing(true)
            .storage_buffer8_bit_access(true)
            .storage_push_constant8(true)
            .uniform_and_storage_buffer8_bit_access(true);

        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .compute_full_subgroups(true)
            .dynamic_rendering(true)
            .shader_demote_to_helper_invocation(true)
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

            // XXX there is another context where this will likely belong better.
            assets: assets::AssetDirs::new(),
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    // XXX pass-through... uuuuuuuuugly though.  Not sure we want to do it like this at all....
    // maybe hold on via another field.
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
