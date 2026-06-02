// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Device
//!
//! Fully initialized logical devices have a lot of associated things.  We know their queues, memory
//! capability, and we have set up their descriptor tables.  The [`Device`] pulls all of these
//! behaviors together.

use std::ffi::{c_void, CStr};

use mutate_assets as assets;

use crate::internal::*;

use super::descriptors;
use super::queue;

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub raw: ash::Device,
    pub queues: queue::Queues,
    // NEXT work on abstracting this to memory decisions.
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Descriptor table and runtime management of its entries.
    pub descriptors: descriptors::Descriptors,

    // XXX move to some other higher context?  PSOs are device-dependent, and we're using this to
    // get PSOs, so probably the other context is runtime support for shader loading.
    /// Initialized assets
    pub assets: assets::AssetDirs,
}

impl Device {
    pub(crate) fn new(instance: &Instance, supported_device: SupportedDevice) -> Self {
        let Instance {
            entry,
            raw: instance,
            ..
        } = &instance;
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
            .push_next(&mut features_1_3)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_1);

        if supported_device.profile.surface {
            features2 = features2
                .push_next(&mut pw_features)
                .push_next(&mut pwid_features)
                .push_next(&mut swapchain_maintenance1);
        }

        let queue_plan = queue::QueuePlan::new(&instance, physical_device).unwrap();
        let queue_cis = queue_plan.queue_cis(); // borrows queue_plan, no allocation
        let extensions: Vec<*const i8> = extensions.iter().map(|ext| ext.as_ptr()).collect();
        let mut device_info = vk::DeviceCreateInfo {
            ..Default::default()
        }
        .push_next(&mut features2)
        .queue_create_infos(&queue_cis)
        .enabled_extension_names(&extensions);

        let raw = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .unwrap()
        };
        let queues = queue::Queues::new(&raw, queue_plan);
        let descriptors = descriptors::Descriptors::new(&raw).unwrap();

        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        Self {
            physical_device,
            raw,
            memory_props,
            queues,
            descriptors,

            // XXX there is another context where this will likely belong better.
            assets: assets::AssetDirs::new(),
        }
    }

    /// Device wait idle.  Waits until there is no in-flight work.  ⚠️ May not return if another
    /// thread is continuously feeding device, resulting in **unbounded waiting**.  Caller should
    /// ensure by application design that device will definitely idle before calling.
    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        unsafe { self.raw.device_wait_idle()? }
        Ok(())
    }

    // XXX pass-through... uuuuuuuuugly though.  Not sure we want to do it like this at all....
    // maybe hold on via another field.
    pub fn bind_sampled_image(
        &mut self,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> descriptors::SampledImageIdx {
        let device = &self.raw;
        let descriptors = &mut self.descriptors;
        descriptors.bind_sampled_image(device, view, layout)
    }

    // XXX in reality, this consumes the context, but ownership friction needs worked out.
    pub fn destroy(&self) {
        unsafe {
            self.descriptors.destroy(&self.raw);
            self.raw.destroy_device(None);
        };
    }

    /// Create a binary semaphore.  These should only be used in Vulkan APIs that require them, such
    /// as swapchain image acquisition.  Use timeline semaphores elsewhere.
    pub(crate) fn make_binary_semaphore(&self) -> Result<BinarySemaphore, VulkanError> {
        let semaphore_ci = vk::SemaphoreCreateInfo::default();
        let raw = unsafe { self.raw.create_semaphore(&semaphore_ci, None)? };
        Ok(BinarySemaphore::new(raw))
    }

    /// Creates a Vulkan fence, initialized to the `signaled` state.  Prefer timeline semaphores
    /// instead where possible.
    pub(crate) fn make_fence(&self, signaled: bool) -> Result<Fence, VulkanError> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let ci = vk::FenceCreateInfo::default().flags(flags);
        let fence = unsafe { self.raw.create_fence(&ci, None)? };
        Ok(Fence(fence))
    }

    /// Creates a new wrapped [`TimelineSemaphore`](crate::dispatch::sync::TimelineSemaphore).
    pub fn make_timeline_semaphore(&self) -> Result<TimelineSemaphore, VulkanError> {
        let mut type_ci = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_ci);
        let raw = unsafe { self.raw.create_semaphore(&ci, None)? };

        Ok(TimelineSemaphore::new(raw))
    }

    pub fn as_raw(&self) -> &ash::Device {
        &self.raw
    }

    pub fn into_raw(self) -> ash::Device {
        self.raw
    }
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

// DEBT RAII.  Perhaps when these types grow state and methods, we will also understand their
// lifetimes and how the handles need to travel across threads and finally be destroyed.
#[derive(Copy, Clone, Debug)]
/// A signal-once fence that was traditionally used for GPU-to-CPU signaling.
pub struct Fence(pub vk::Fence);

impl Fence {
    pub fn into_raw(self) -> vk::Fence {
        self.0
    }

    pub fn as_raw(&self) -> vk::Fence {
        self.0
    }

    pub fn destroy(self, device: &Device) {
        unsafe { device.as_raw().destroy_fence(self.0, None) }
    }
}

impl std::ops::Deref for Fence {
    type Target = vk::Fence;
    fn deref(&self) -> &vk::Fence {
        &self.0
    }
}
