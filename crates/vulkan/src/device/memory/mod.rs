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
// NOTE staging + transfer is an operation that can be abstracted over by a runtime as part of an
// optionally two-step provision.  Explicit BAR is reported, but mainly so that applications
// expecting it don't fall over.

pub mod bar;

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
    device_type: vk::PhysicalDeviceType,
    pub memory_props: vk::PhysicalDeviceMemoryProperties,
    pub(crate) bar_window: u32,
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
        let mut props2 = vk::PhysicalDeviceProperties2::default();
        unsafe { instance.get_physical_device_properties2(*physical_device, &mut props2) };
        let device_type = props2.properties.device_type;
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };
        let bar_window = bar::detect_bar(&memory_props);

        // Likely we need more info later, but keep the memory query results here.
        let non_coherent_atom_size = unsafe {
            instance
                .get_physical_device_properties(*physical_device)
                .limits
                .non_coherent_atom_size
        };

        Self {
            device_type,
            memory_props,
            non_coherent_atom_size,
            bar_window,
            device: logical_device.clone(),
        }
    }

    /// Ranked candidates for `request`, skipping types in `tried` and any the request's
    /// [`bar::BarPolicy`] fences off on this device.  Pass only types already attempted; the
    /// device-specific exclusion is folded in here so callers can't omit it.
    ///
    /// The returned iterator snapshots its inputs, so a caller may set bits in its own `tried`
    /// mask while draining it; the change takes effect on the next call, not this one.
    fn memory_types(
        &self,
        mem_req: &vk::MemoryRequirements,
        request: &MemoryTypeRequest,
        tried: u32,
    ) -> impl Iterator<Item = (u32, vk::MemoryPropertyFlags)> {
        let legal = mem_req.memory_type_bits & !tried & !request.excluded(self.bar_window);
        let mut scored: Vec<((u32, u32), u32, vk::MemoryPropertyFlags)> = Vec::new();
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
            scored.push(((missing, unwanted), i, props));
        }
        scored.sort_by_key(|&(cost, _, _)| cost);
        scored.into_iter().map(|(_, i, props)| (i, props))
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
            for (index, _props) in self.memory_types(&requirements, request, tried) {
                let alloc_info = alloc_info.memory_type_index(index);
                tried |= 1 << index;
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
            for (index, _props) in self.memory_types(&requirements, request, tried) {
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

    #[cfg(test)]
    pub fn describe_type(&self, index: u32) -> String {
        let ty = self.memory_props.memory_types[index as usize];
        let heap = self.memory_props.memory_heaps[ty.heap_index as usize];
        let mib = heap.size / (1024 * 1024);
        let device_heap = heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        format!(
            "type {index:2} -> heap {} ({mib} MiB) {:?}",
            ty.heap_index, ty.property_flags,
        )
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
    /// Hard constraint, device-dependent.  Resolved against this device's carve-out.
    pub bar_policy: bar::BarPolicy,
}

impl MemoryTypeRequest {
    /// Types this request must never touch, given `bar_window` from [`bar::detect_bar`].
    /// Zero whenever the device has no scarce carve-out.
    pub(crate) fn excluded(&self, bar_window: u32) -> u32 {
        match self.bar_policy {
            bar::BarPolicy::Claim => 0,
            bar::BarPolicy::Yield => bar_window,
        }
    }
}

impl AsRef<[MemoryTypeRequest]> for MemoryTypeRequest {
    fn as_ref(&self) -> &[MemoryTypeRequest] {
        std::slice::from_ref(self)
    }
}

/// Presets for [`MemoryTypeRequest`].  Call [`requests`] to obtain the requests.
// NOTE Later typed allocations must reflect these semantics, exposing only the necessary APIs and
// not allowing pathologically slow or what could be impossible usages of wrong kinds of
// allocations.  The memory may collapse to fewer or different heaps on some devices, but the same
// semantics work without drawbacks while making use of distinct heaps when available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryUse {
    /// Only ever written or read by the device or as a destination for transfer from staging.
    /// Treat as device read-write.
    DeviceLocal,
    /// usually ReBAR on discrete.  May be BAR or another heap on UMA, avoiding cached.  Optimal for
    /// small writes, pushed by host to device memory.  Treat as host write-only / device read-only.
    Upload,
    /// Cached for host read.  On both discrete and UMA, residency is sysmem.  Discrete devices will
    /// write via PCIe.  Optimal for shaders publishing updates to the host.  Treat as device
    /// write-only / host read-only.
    Readback,
    /// Device-visible sysmem.  Primarily used as a transfer source.  Optimal for uploading images
    /// that need a layout transition or scratch space for larger transfers.  Device reads via PCIe
    /// (usually on DMA queue) or UMA.  Host reads are full speed, but treat as host write-only,
    /// device read-only.
    Transfer,
    /// Only ever transiently used by the device.  An explicit heap on tilers.  Usages on non-tiling
    /// devices enable opportunistic optimizations.
    Lazy,
    // NEXT protected.  No DRM yet for AE AYE matey! 🏴‍☠️
}

impl MemoryUse {
    /// Preference-ordered fallback chain. Walk on `NoTypeSatisfies` from selection
    /// or `OUT_OF_DEVICE_MEMORY` from allocation. The last entry of each chain
    /// requires at most `HOST_VISIBLE`, which the spec guarantees satisfiable, so
    /// exhausting a chain means the device is genuinely out of memory.
    // XXX NoTypeSatisfies is made up right now.  Allocation is not yet attempted for each request
    // and this is just vapor ware.
    pub const fn requests(self) -> &'static [MemoryTypeRequest] {
        use bar::BarPolicy::{Claim, Yield};
        use vk::MemoryPropertyFlags as F;
        const fn req(
            required: F,
            preferred: F,
            avoided: F,
            bar_policy: bar::BarPolicy,
        ) -> MemoryTypeRequest {
            MemoryTypeRequest {
                required,
                preferred,
                avoided,
                bar_policy,
            }
        }
        const fn or(a: F, b: F) -> F {
            F::from_raw(a.as_raw() | b.as_raw())
        }

        const DEVICE_LOCAL: &[MemoryTypeRequest] = &[
            // Pure VRAM only.  No fallback: if this fails, the device is out of VRAM and
            // the caller needs to know, not to be silently demoted into the BAR window or
            // across PCIe into sysmem.
            req(F::DEVICE_LOCAL, F::empty(), F::HOST_VISIBLE, Yield),
        ];
        // Residency-first: land in device memory and map it.  Rung 1 is BAR/ReBAR/UMA.
        // Rung 2 is sysmem, which is *not* the same deal -- the device now reads over
        // PCIe and the caller probably wanted to stage.  See NOTE below.
        const UPLOAD: &[MemoryTypeRequest] = &[
            req(
                or(F::DEVICE_LOCAL, F::HOST_VISIBLE),
                F::HOST_COHERENT,
                F::HOST_CACHED,
                Claim,
            ),
            req(
                F::HOST_VISIBLE,
                F::HOST_COHERENT,
                or(F::DEVICE_LOCAL, F::HOST_CACHED),
                Yield,
            ),
        ];
        const READBACK: &[MemoryTypeRequest] = &[
            // Cached sysmem; coherent saves the vkInvalidate per poll.
            req(
                or(F::HOST_VISIBLE, F::HOST_CACHED),
                F::HOST_COHERENT,
                F::DEVICE_LOCAL,
                Yield,
            ),
            // No cached type (rare): reads will crawl, but they'll complete.
            req(F::HOST_VISIBLE, F::HOST_COHERENT, F::DEVICE_LOCAL, Yield),
        ];
        const TRANSFER: &[MemoryTypeRequest] = &[
            // One entry: the ideal staging type *is* the guaranteed type.
            req(
                F::HOST_VISIBLE,
                F::HOST_COHERENT,
                or(F::DEVICE_LOCAL, F::HOST_CACHED),
                Yield,
            ),
        ];
        const LAZY: &[MemoryTypeRequest] = &[
            // Tilers expose a real transient heap.  Everyone else falls to rung 2 and
            // gets ordinary device memory, which is what LAZILY_ALLOCATED degrades to
            // by spec.
            req(
                or(F::DEVICE_LOCAL, F::LAZILY_ALLOCATED),
                F::empty(),
                F::HOST_VISIBLE,
                Yield,
            ),
            req(F::DEVICE_LOCAL, F::empty(), F::HOST_VISIBLE, Yield),
        ];

        match self {
            MemoryUse::DeviceLocal => DEVICE_LOCAL,
            MemoryUse::Upload => UPLOAD,
            MemoryUse::Readback => READBACK,
            MemoryUse::Transfer => TRANSFER,
            MemoryUse::Lazy => LAZY,
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

    const ALL_USES: [MemoryUse; 5] = [
        MemoryUse::DeviceLocal,
        MemoryUse::Upload,
        MemoryUse::Transfer,
        MemoryUse::Readback,
        MemoryUse::Lazy,
    ];

    /// Unconstrained requirements: exercises selection, not a resource's type mask.
    const ANY: vk::MemoryRequirements = vk::MemoryRequirements {
        size: 1,
        alignment: 1,
        memory_type_bits: !0,
    };

    #[test]
    pub fn resolve_all_domains() {
        with_context!(|device, instance| {
            let memory = &device.memory;

            println!("\n=== memory types present ===");
            for i in 0..memory.memory_props.memory_type_count {
                println!("  {}", memory.describe_type(i));
            }

            println!("\n=== preset resolution ===");
            for use_ in ALL_USES {
                println!("{use_:?}");

                let mut tried: u32 = 0;
                for (n, request) in use_.requests().iter().enumerate() {
                    println!(
                        "  request {n}: required: {:?} | preferred: {:?} | avoided: {:?} | bar: {:?}",
                        request.required, request.preferred, request.avoided, request.bar_policy,
                    );
                    let mut empty = true;
                    for (rank, (index, _)) in memory.memory_types(&ANY, request, tried).enumerate()
                    {
                        empty = false;
                        tried |= 1 << index;
                        let marker = if rank == 0 { "->" } else { "  " };
                        println!("    {marker} {}", memory.describe_type(index));
                    }
                    if empty {
                        println!("    (no candidates)");
                    }
                }
            }
        })
    }
}
