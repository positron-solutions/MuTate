// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Queue
//!
//! Vulkan device queues and command buffer pools are tightly coupled.  All Vulkan devices must
//! expose at least a graphics capable queue and may expose dedicated transfer and compute queues.
//!

use ash::vk;

use crate::prelude::*;

/// Queues
pub struct Queues {
    graphics: Queue,
    compute: Option<Queue>,
    transfer: Option<Queue>,

    /// XXX Used at surface creation, but there is likely much incorrect about the structure and
    /// usage.
    pub graphics_family_index: u32,
}

impl Queues {
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics.queue
    }

    pub fn compute_queue(&self) -> vk::Queue {
        self.compute
            .as_ref()
            .map(|q| q.queue.clone())
            .unwrap_or_else(|| self.graphics_queue())
    }

    pub fn transfer_queue(&self) -> vk::Queue {
        self.transfer
            .as_ref()
            .map(|q| q.queue.clone())
            .unwrap_or_else(|| self.compute_queue())
    }

    pub fn graphics_pool(&self) -> vk::CommandPool {
        self.graphics.command_pool.clone()
    }

    pub fn compute_pool(&self) -> vk::CommandPool {
        self.compute
            .as_ref()
            .map(|p| p.command_pool.clone())
            .unwrap_or_else(|| self.graphics_pool())
    }

    pub fn transfer_pool(&self) -> vk::CommandPool {
        self.transfer
            .as_ref()
            .map(|p| p.command_pool.clone())
            .unwrap_or_else(|| self.compute_pool())
    }

    pub fn new(device: &ash::Device, queue_families: QueueFamilies) -> Self {
        Queues {
            graphics_family_index: queue_families.graphics.clone(),

            graphics: Queue::new(device, queue_families.graphics),
            compute: queue_families.compute.map(|i| Queue::new(device, i)),
            transfer: queue_families.transfer.map(|i| Queue::new(device, i)),
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        self.graphics.destroy(device);
        self.compute.as_ref().map(|q| q.destroy(device));
        self.transfer.as_ref().map(|q| q.destroy(device));
    }
}

struct Queue {
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
}

impl Queue {
    fn new(device: &ash::Device, queue_family_index: u32) -> Self {
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let command_pool_ci = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index,
            ..Default::default()
        };

        let command_pool = unsafe {
            device.create_command_pool(&command_pool_ci, None).unwrap() // DEBT error handling
        };
        Self {
            queue,
            command_pool,
        }
    }

    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
        }
        // NOTE device owns queues.  Just drop handles.
    }
}

/// The goal of QueueFamilies is to collect and forward preferred queue choices, leaving them as
/// None when there is no dedicated queue.
pub struct QueueFamilies {
    graphics: u32,
    compute: Option<u32>,
    transfer: Option<u32>,
}

impl QueueFamilies {
    /// Find queue indices for queues that support the minimum capabilities to prefer exclusive
    /// queues.
    pub fn new(instance: &ash::Instance, physical_device: &vk::PhysicalDevice) -> Self {
        let qfps =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
        // NOTE Spec says at least one queue with graphics bit must exist
        let graphics_index = min_caps_family(&qfps, vk::QueueFlags::GRAPHICS).unwrap();
        let compute_index = min_caps_family(&qfps, vk::QueueFlags::COMPUTE);
        let mut transfer_index = min_caps_family(&qfps, vk::QueueFlags::TRANSFER);

        // No dedicated transfer queue
        if transfer_index == compute_index {
            transfer_index = None
        }
        QueueFamilies {
            graphics: graphics_index,
            compute: compute_index.filter(|i| *i != graphics_index),
            transfer: transfer_index.filter(|i| *i != graphics_index),
        }
    }

    pub fn queue_cis(&self, priorities: &[f32]) -> Vec<vk::DeviceQueueCreateInfo> {
        [Some(self.graphics), self.compute, self.transfer]
            .into_iter()
            .filter_map(|opt| opt)
            .map(|index| vk::DeviceQueueCreateInfo {
                queue_family_index: index,
                queue_count: 1,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            })
            .collect()
    }
}

/// Return the queue index of the family with the minimum support for the requested `flags`.
fn min_caps_family(qfps: &Vec<vk::QueueFamilyProperties>, flags: vk::QueueFlags) -> Option<u32> {
    let mut found_flags = vk::QueueFlags::empty();
    let mut found_index: Option<u32> = None;

    for (i, qf) in qfps.iter().enumerate() {
        if qf.queue_flags.contains(flags) {
            if let Some(found) = found_index {
                if found_flags.as_raw().count_ones() > qf.queue_flags.as_raw().count_ones() {
                    found_flags = qf.queue_flags.clone();
                    found_index = Some(i as u32);
                }
            } else {
                found_flags = qf.queue_flags.clone();
                found_index = Some(i as u32);
            }
        }
    }
    found_index
}
