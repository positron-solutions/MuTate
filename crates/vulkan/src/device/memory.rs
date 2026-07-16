// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Device Memory
//!
//! ⚠️ This is a stub.  Likely boundaries seeming clear enough to start pulling early stuff out of
//! Device and other areas it had started scattering to.
//!
//! Query memory.  Create primary allocations.  Resolve physical device to our semantic usage model.
//!
//! ## Design Constraints
//!
//! The things that fix this module into place.
//!
//! - Guaranteed allocation cap is low (4096) and costs driver overhead.  Sub-allocation is favored.
//! - One mapping per allocation.  Mapped sub-allocations should come from a separate allocation.
//! - Distinct memory domains are better for different roles:
//!   + Fast host to device writes (typically ReBAR or UMA mapped `DEVICE_LOCAL`)
//!   + Hot read-back on host for cheap polling of state that shaders publish (probably
//!     `HOST_CACHED` and not `DEVICE_LOCAL` unless UMA)
//!   + Device-only memory, a reduced API even if the backing physical memory supports more, typical
//!     on UMA
//!   + Staging for uploading images before first layout transition or niche writes that DMA does
//!     better with lower CPU usage.
//!
//! ### Early Conclusions
//!
//! - Memory the host will read should reside on the host.
//! - Memory should reside on the device if only the device will ever touch it.
//! - Device mapped memory is best for writes.  Without ReBAR or for certain writes, both determined
//!   at runtime, staging may still be optimal or required, but this is decreasing in frequency and
//!   lower priority.
//! - UMA will often back buffers with one kind of physical memory.  The same semantics and
//!   structures designed for ReBAR will all just work on UMA.
//!
//! ## What We Want From This Module
//!
//! - `WriteGuard` for convenient scoped flushing
//! - Repeatable invalidation for periodic readback
//! - Types for memory roles
//! - Device memory budget awareness
//! - Allocated pools acquired from that budget
//! - Sub-allocation types that support the APIs of the primary allocation
//!
//! ### Scope Limitation
//!
//! A resource runtime should be responsible for maintaining shared aliases, hydration / deletion
//! queues etc.  This module should focus on getting from useless physical memory to API-wrapped
//! sub-allocations.

// NEXT hand out allocations
// NEXT reclaim allocations

use crate::internal::*;

pub(crate) mod core {
    // NOTE please encourage use of higher level interfaces so we can free the implementations up
    // and start cleaving off the unused parts of the Vulkan API
    pub use super::MemoryUse;
}

// NEXT a more mature allocator would configure this as a runtime exclusion mask.  This hard code
// does not support even turning that knob.  After we get an allocator, make this exclusion mask
// configurable so that someone interested in this flag will be able to do something.
const FORBIDDEN: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD.as_raw()
        | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD.as_raw(),
);

pub struct Memory {
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    pub non_coherent_atom_size: vk::DeviceSize,
    // Owned handle just to prevent needing the handle from the owning Device wrapper.  Not super
    // expensive.  Safe to just drop.
    device: ash::Device,
}

impl Memory {
    pub(crate) fn new(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        logical_device: &ash::Device,
    ) -> Self {
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };
        // Likely we need more info later, but keep the memory query results here.
        let non_coherent_atom_size = unsafe {
            instance
                .get_physical_device_properties(*physical_device)
                .limits
                .non_coherent_atom_size
        };

        Self {
            memory_props,
            non_coherent_atom_size,
            device: logical_device.clone(),
        }
    }

    /// Ranked candidates for `request`, skipping any type in `exclude` mask that were already
    /// tried.  Initialize your exclude mask with those that should never be used.
    fn memory_types(
        &self,
        mem_req: &vk::MemoryRequirements,
        request: &MemoryTypeRequest,
        exclude: u32,
    ) -> impl Iterator<Item = u32> {
        let legal = mem_req.memory_type_bits & !exclude;
        let mut scored: Vec<((u32, u32), u32)> = Vec::new();
        for i in 0..self.memory_props.memory_type_count {
            if legal & (1 << i) == 0 {
                continue;
            }
            let props = self.memory_props.memory_types[i as usize].property_flags;
            if !props.contains(request.required) || props.intersects(FORBIDDEN) {
                continue;
            }
            let missing = (request.preferred & !props).as_raw().count_ones();
            let unwanted = (request.avoided & props).as_raw().count_ones();
            scored.push(((missing, unwanted), i));
        }
        scored.sort_by_key(|&(cost, _)| cost);
        scored.into_iter().map(|(_, i)| i)
    }

    /// Interim allocation method for a buffer.  Sub-allocation not ready yet.
    /// `memory_type_requests` can be built manually or supplied from [`MemoryUse`].
    pub(crate) fn allocate_buffer(
        &self,
        buffer_ci: &vk::BufferCreateInfo,
        memory_type_requests: impl AsRef<[MemoryTypeRequest]>,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, vk::DeviceSize), VulkanError> {
        let buffer = unsafe { self.device.create_buffer(&buffer_ci, None)? };
        let requests = memory_type_requests.as_ref();
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mut alloc_info = vk::MemoryAllocateInfo::default().allocation_size(requirements.size);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        if buffer_ci
            .usage
            .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        {
            alloc_info = alloc_info.push_next(&mut flags);
        }
        let mut tried: u32 = 0;
        for request in requests {
            for index in self.memory_types(&requirements, request, tried) {
                let alloc_info = alloc_info.memory_type_index(index);
                match unsafe { self.device.allocate_memory(&alloc_info, None) } {
                    Ok(memory) => {
                        unsafe {
                            self.device.bind_buffer_memory(buffer, memory, 0)?;
                        }
                        return Ok((buffer, memory, requirements.size));
                    }
                    Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                    | Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => continue,
                    Err(e) => {
                        // DEBT fallible resource creation
                        unsafe { self.device.destroy_buffer(buffer, None) };
                        return Err(VulkanError::Ash(e));
                    }
                }
            }
        }
        // DEBT fallible resource creation
        unsafe { self.device.destroy_buffer(buffer, None) };
        Err(VulkanError::AllocationFailed {
            request: requests[0],
        })
    }

    /// Interim allocation method for an image.  Sub-allocation not ready yet.
    /// `memory_type_requests` can be built manually or supplied from [`MemoryUse`].
    // NEXT images are far more constrained than buffers in practice: most want
    // `MemoryUse::DeviceLocal`.  `MemoryUse` is currently shared with buffers; if image roles
    // diverge (dedicated allocations, granularity), split the presets then.
    pub(crate) fn allocate_image(
        &self,
        image_ci: &vk::ImageCreateInfo,
        memory_type_requests: impl AsRef<[MemoryTypeRequest]>,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::DeviceSize), VulkanError> {
        let image = unsafe { self.device.create_image(&image_ci, None)? };
        let requests = memory_type_requests.as_ref();
        let requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alloc_info = vk::MemoryAllocateInfo::default().allocation_size(requirements.size);
        let mut tried: u32 = 0;
        for request in requests {
            for index in self.memory_types(&requirements, request, tried) {
                tried |= 1 << index;
                let alloc_info = alloc_info.memory_type_index(index);
                match unsafe { self.device.allocate_memory(&alloc_info, None) } {
                    Ok(memory) => {
                        unsafe {
                            self.device.bind_image_memory(image, memory, 0)?;
                        }
                        return Ok((image, memory, requirements.size));
                    }
                    Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                    | Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY) => continue,
                    Err(e) => {
                        // DEBT fallible resource creation
                        unsafe { self.device.destroy_image(image, None) };
                        return Err(VulkanError::Ash(e));
                    }
                }
            }
        }
        // DEBT fallible resource creation
        unsafe { self.device.destroy_image(image, None) };
        Err(VulkanError::AllocationFailed {
            request: requests[0],
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryTypeRequest {
    /// Hard constraint. A type missing any of these bits is rejected.
    pub required: vk::MemoryPropertyFlags,
    /// Soft pull. Types are ranked by how many of these they're missing.
    pub preferred: vk::MemoryPropertyFlags,
    /// Soft push. Types are ranked by how many of these they carry.
    pub avoided: vk::MemoryPropertyFlags,
}

impl AsRef<[MemoryTypeRequest]> for MemoryTypeRequest {
    fn as_ref(&self) -> &[MemoryTypeRequest] {
        std::slice::from_ref(self)
    }
}

/// Presets for [`MemoryTypeRequest`].  Call [`requests`] to obtain the requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryUse {
    /// Only written or read by the device or a transfer from staging.
    DeviceLocal,
    /// Optimal for writes, usually ReBAR or UMA.
    HostMapped,
    /// Optimal for host reads of shader published state updates.
    Readback,
    /// Optimal for uploading images that need a layout transition or larger transfers using DMA /
    /// UMA shortcuts.
    Staging,
}

impl MemoryUse {
    /// Preference-ordered fallback chain. Walk on `NoTypeSatisfies` from selection
    /// or `OUT_OF_DEVICE_MEMORY` from allocation. The last entry of each chain
    /// requires at most `HOST_VISIBLE`, which the spec guarantees satisfiable, so
    /// exhausting a chain means the device is genuinely out of memory.
    // XXX NoTypeSatisfies is made up right now.  Allocation is not yet attempted for each request
    // and this is just vapor ware.
    pub const fn requests(self) -> &'static [MemoryTypeRequest] {
        use vk::MemoryPropertyFlags as F;
        const fn req(required: F, preferred: F, avoided: F) -> MemoryTypeRequest {
            MemoryTypeRequest {
                required,
                preferred,
                avoided,
            }
        }
        const fn or(a: F, b: F) -> F {
            F::from_raw(a.as_raw() | b.as_raw())
        }

        const DEVICE_LOCAL: &[MemoryTypeRequest] = &[
            // Pure VRAM; keep off the (possibly 256 MiB) BAR window.
            req(F::DEVICE_LOCAL, F::empty(), F::HOST_VISIBLE),
            // VRAM exhausted or masked out: take anything in the mask,
            // steering off the heap that just failed.
            req(F::empty(), F::empty(), F::DEVICE_LOCAL),
        ];
        const HOST_MAPPED: &[MemoryTypeRequest] = &[
            // ReBAR / UMA: write straight into device memory.
            req(
                or(F::DEVICE_LOCAL, F::HOST_VISIBLE),
                F::HOST_COHERENT,
                F::HOST_CACHED,
            ),
            // No BAR type, or BAR heap exhausted: write-combined sysmem.
            req(
                F::HOST_VISIBLE,
                F::HOST_COHERENT,
                or(F::DEVICE_LOCAL, F::HOST_CACHED),
            ),
        ];
        const READBACK: &[MemoryTypeRequest] = &[
            // Cached sysmem; coherent saves the vkInvalidate per poll.
            req(
                or(F::HOST_VISIBLE, F::HOST_CACHED),
                F::HOST_COHERENT,
                F::DEVICE_LOCAL,
            ),
            // No cached type (rare): reads will crawl, but they'll complete.
            req(
                F::HOST_VISIBLE,
                or(F::HOST_CACHED, F::HOST_COHERENT),
                F::DEVICE_LOCAL,
            ),
        ];
        const STAGING: &[MemoryTypeRequest] = &[
            // One entry: the ideal staging type *is* the guaranteed type.
            req(
                F::HOST_VISIBLE,
                F::HOST_COHERENT,
                or(F::DEVICE_LOCAL, F::HOST_CACHED),
            ),
        ];

        match self {
            MemoryUse::DeviceLocal => DEVICE_LOCAL,
            MemoryUse::HostMapped => HOST_MAPPED,
            MemoryUse::Readback => READBACK,
            MemoryUse::Staging => STAGING,
        }
    }
}

impl AsRef<[MemoryTypeRequest]> for MemoryUse {
    fn as_ref(&self) -> &[MemoryTypeRequest] {
        self.requests()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    pub fn resolve_all_domains() {
        with_context!(|device, instance| {
            // XXX we need to go ahead and test the memory resolution for each `MemoryUse` gives us valid indexes.
        })
    }
}
