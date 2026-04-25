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
use super::vulkan::{self, SupportedDevice};

pub struct DeviceContext {
    pub physical_device: vk::PhysicalDevice,
    /// Vulkan logical device
    pub device: ash::Device,
    /// Queues family info.
    pub queues: queue::Queues,
    // NEXT work on abstracting this to memory decisions.
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Descriptor table for
    pub descriptors: descriptors::Descriptors,

    // XXX move to some other higher context?
    /// Initialized assets
    pub assets: assets::AssetDirs,
}

impl DeviceContext {
    pub(crate) fn new(vk_context: &vulkan::VkContext, supported_device: SupportedDevice) -> Self {
        let vulkan::VkContext { entry, instance } = &vk_context;
        let physical_device = supported_device.device();
        let extensions = &supported_device.extensions;

        let mut pwid_features = vk::PhysicalDevicePresentIdFeaturesKHR::default().present_id(true);
        let mut pw_features =
            vk::PhysicalDevicePresentWaitFeaturesKHR::default().present_wait(true);

        let mut swapchain_maintenance1 =
            vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                .swapchain_maintenance1(true);

        let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true)
            .storage_buffer16_bit_access(true)
            .storage_push_constant16(true)
            // XXX remove this
            // .storage_input_output16(true)
            .uniform_and_storage_buffer16_bit_access(true);

        // MAYBE wonder if these attributes could be de-duped with a macro 🤔
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
            .timeline_semaphore(true)
            .uniform_and_storage_buffer8_bit_access(true);

        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .compute_full_subgroups(true)
            .dynamic_rendering(true)
            .inline_uniform_block(true)
            .shader_demote_to_helper_invocation(true)
            .synchronization2(true)
            .maintenance4(true);

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(vk::PhysicalDeviceFeatures::default().shader_int16(true))
            .push_next(&mut pw_features)
            .push_next(&mut pwid_features)
            .push_next(&mut features_1_3)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_1)
            .push_next(&mut swapchain_maintenance1);

        // XXX present families!
        let queue_plan = queue::QueuePlan::new(&instance, physical_device, &[]).unwrap();
        let queue_cis = queue_plan.queue_cis(); // borrows queue_plan, no allocation
        let extensions: Vec<*const i8> = extensions.iter().map(|ext| ext.as_ptr()).collect();
        let mut device_info = vk::DeviceCreateInfo {
            ..Default::default()
        }
        .push_next(&mut features2)
        .queue_create_infos(&queue_cis)
        .enabled_extension_names(&extensions);

        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queues = queue::Queues::new(&device, queue_plan);
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
    ) -> descriptors::SampledImageIdx {
        let device = &self.device;
        let descriptors = &mut self.descriptors;
        descriptors.bind_sampled_image(device, view, layout)
    }

    // XXX in reality, this consumes the context, but ownership friction needs worked out.
    pub fn destroy(&self) {
        unsafe {
            self.descriptors.destroy(&self.device);
            self.device.destroy_device(None);
        };
    }

    /// Returns a vanilla binary semaphore.
    // NEXT bon this for making timeline and other semaphores?
    pub fn make_semaphore(&self) -> vk::Semaphore {
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        unsafe { self.device().create_semaphore(&semaphore_ci, None).unwrap() }
    }
}
