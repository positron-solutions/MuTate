// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Buffer
//!
//! The `MappedAllocation` is just a first pass at wrapping up a persistently mapped Vulkan buffer
//! that we will flush to the device periodically.
//!
//! Early on, it's going to look like the APIs are intended for manual use.  Long-term, we want
//! pipelines to specify `BufferSpec` dependencies and let the run time provide views of memory that
//! satisfies those specs.  There's... just no runtime yet to speak of 🤡.

// NEXT With or without sub-allocation, some buffers need to not support certain interfaces.  Why
// would I attempt to write a non-host visible from the host?  Why would I attempt to read back from
// a buffer that I didn't try to bind to a cached allocation?
// NEXT the T in a buffer is going to take some work.  We can either attach or detach headers for
// control data, and that changes T.  First class treatment of headers is likely unnecessary /
// overly rigid.  The trade-off is now the user needs two buffers or the type-agreement has to check
// (Header, [T]).  Checking tuples is not against the law, but then the pointer de-reference needs
// on slang needs to understand this.  De-reference is where some investigative development is
// needed.  Or...here me out.  The header will usually contain a pointer to &T because the alignment
// of T will likely not match the flush atoms or some other requirement in some cases.
// DEBT sub-allocation / resource runtime. 💸
// NEXT hang buffer & image creation methods off of the device.memory.  Limit buffer & image module
// scope to handling post-allocation.

use std::ptr::NonNull;

use crate::internal::*;

pub(crate) mod core {
    pub use super::DeviceBuffer;
    pub use super::MappedAllocation;
    pub use super::MappedWriteView;
}

/// Very interim.  We will separate buffer from allocation and future `MappedBuffer` will build on
/// top of `MappedSubAllocation`, itself on top of a **real** `MappedAllocation` or some
/// runtime-only info that never gets let out of the basement for users to see.
pub struct MappedAllocation<T> {
    /// A buffer.
    pub buffer: vk::Buffer,
    /// The allocation (which totally shouldn't be part of this type)
    pub memory: vk::DeviceMemory,
    /// Mapped buffers support host writes.
    pub ptr: NonNull<T>,
    /// Number of `T` elements.
    pub len: usize,
    /// Actual size of the buffer depends on size, alignment, usage.
    pub size: vk::DeviceSize,
}

impl<T> MappedAllocation<T> {
    /// `len` is the number of `T` this buffer will hold.
    pub fn new(device: &Device, len: usize) -> Result<Self, VulkanError> {
        // NEXT these are pretty temporito.  See the abstract choices in the memory module.  That's
        // basically how we want to do this.
        let buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<T>() * len) as u64,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (buffer, memory, size) = device
            .memory
            .allocate_buffer(&buffer_info, MemoryUse::Upload)?;
        let raw_ptr = unsafe {
            device
                .as_raw()
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
        };
        let ptr = NonNull::new(raw_ptr as *mut T).unwrap();
        Ok(Self {
            buffer,
            ptr,
            len,
            memory,
            size,
        })
    }

    pub fn destroy(&self, device: &Device) -> Result<(), VulkanError> {
        unsafe {
            device.unmap_memory(self.memory);
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
        Ok(())
    }

    /// Don't forget to flush 🚽
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Move writes to device memory.
    pub fn flush(&mut self, device: &Device) -> Result<(), VulkanError> {
        let flush_range = vk::MappedMemoryRange {
            memory: self.memory,
            offset: 0,
            size: self.size,
            ..Default::default()
        };
        unsafe {
            device.flush_mapped_memory_ranges(&[flush_range])?;
        }
        Ok(())
    }

    /// Refresh host view with device writes.
    pub fn invalidate(&mut self, device: &Device) -> Result<(), VulkanError> {
        let invalidate_range = vk::MappedMemoryRange {
            memory: self.memory,
            offset: 0,
            size: self.size,
            ..Default::default()
        };
        unsafe {
            device.invalidate_mapped_memory_ranges(&[invalidate_range])?;
        }
        Ok(())
    }

    pub fn bound(&self, device: &Device) -> SsboIdx {
        let descriptors = &device.descriptors;
        // XXX WTF is this?  No srsly what happened here?  Kill something with fire.
        let byte_size = (std::mem::size_of::<T>() * self.len) as u64;
        descriptors.bind_ssbo(&device.raw, self.buffer, 0, byte_size)
    }

    // XXX Slang type for buffer device address
    pub fn device_address(&self, device: &Device) -> Result<vk::DeviceAddress, VulkanError> {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        Ok(unsafe { device.get_buffer_device_address(&info) })
    }

    /// After-compute shader barrier.  Use after some compute shader writes to a buffer.
    pub fn barrier_compute_post(&self, cb: &vk::CommandBuffer, device: &Device) {
        let buffer_barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::TRANSFER_READ,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: self.buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };

        unsafe {
            device.cmd_pipeline_barrier(
                *cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );
        }
    }

    /// Pre-compute shader barrier.
    /// Use after host writes + flush, before a compute shader reads/writes the buffer.
    // XXX these barrier "simplifications" are utter garbage
    pub fn barrier_compute_pre(&self, cb: &vk::CommandBuffer, device: &Device) {
        let buffer_barrier = vk::BufferMemoryBarrier {
            src_access_mask: vk::AccessFlags::HOST_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: self.buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };

        unsafe {
            device.cmd_pipeline_barrier(
                *cb,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_barrier],
                &[],
            );
        }
    }

    /// Lend a flush-capable write view for moving into a writer thread.
    ///
    /// Clones a raw device handle into the view. Caller must join that thread
    /// before calling `destroy`.
    pub fn write_view(&self, device: &Device) -> MappedWriteView<T> {
        MappedWriteView {
            device: device.as_raw().clone(),
            memory: self.memory,
            ptr: self.ptr,
            len: self.len,
            size: self.size,
        }
    }
}

pub fn buffer_image_copy_full(extent: vk::Extent2D) -> vk::BufferImageCopy {
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0, // tightly packed
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
    }
}

/// Share write support for a buffer into a thread without moving responsibility for freeing.
pub struct MappedWriteView<T> {
    device: ash::Device,
    memory: vk::DeviceMemory,
    ptr: NonNull<T>,
    len: usize,
    size: vk::DeviceSize,
}

unsafe impl<T: Send> Send for MappedWriteView<T> {}

impl<T> MappedWriteView<T> {
    /// # Safety
    /// Caller ensures this view (and its thread) is dropped before the owning
    /// `MappedAllocation` is destroyed, and that owner-side code does not write
    /// the same region concurrently.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Make host writes device-visible. Flushes the whole mapped range.
    pub fn flush(&self) -> Result<(), VulkanError> {
        let range = vk::MappedMemoryRange {
            memory: self.memory,
            offset: 0,
            size: self.size,
            ..Default::default()
        };
        unsafe {
            self.device.flush_mapped_memory_ranges(&[range])?;
        }
        Ok(())
    }

    /// Flush a precomputed byte range. The caller (typically a layout that
    /// aligned its offsets to `nonCoherentAtomSize` at init) guarantees that
    /// `offset` and `size` satisfy Vulkan's atom-alignment rules — either both
    /// are atom multiples, or `offset + size == size_bytes`.
    ///
    /// The debug alignment check lives here rather than in the buffer so the
    /// invariant is asserted at the point of use.
    pub fn flush_range(
        &self,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<(), VulkanError> {
        debug_assert!(
            offset + size <= self.size,
            "flush_range out of bounds: {offset}+{size} > {}",
            self.size
        );
        let range = vk::MappedMemoryRange {
            memory: self.memory,
            offset,
            size,
            ..Default::default()
        };
        unsafe {
            self.device.flush_mapped_memory_ranges(&[range])?;
        }
        Ok(())
    }
}

/// A companion to the still half-baked `MappedAllocation`.  Intended for device memory.  Does not
/// support any flushes etc because the backing allocation likely cannot do that.  `T` might be back
/// but just winging to discover where the grass meets the sky.
pub struct DeviceBuffer {
    /// A buffer.
    pub buffer: vk::Buffer,
    /// The allocation (which totally shouldn't be part of this type)
    pub memory: vk::DeviceMemory,
    /// Actual size of the buffer depends on size, alignment, usage.
    pub size: vk::DeviceSize,
}

impl DeviceBuffer {
    pub fn new(device: &Device, size: u64) -> Result<Self, VulkanError> {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let (buffer, memory, size) = device
            .memory
            .allocate_buffer(&buffer_info, MemoryUse::DeviceLocal)?;
        Ok(Self {
            buffer,
            memory,
            size,
        })
    }

    pub fn destroy(&self, device: &Device) -> Result<(), VulkanError> {
        unsafe {
            device.free_memory(self.memory, None);
            device.destroy_buffer(self.buffer, None);
        }
        Ok(())
    }

    // NEXT generic over buffer type!
    pub fn device_address(&self, device: &Device) -> Result<vk::DeviceAddress, VulkanError> {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        Ok(unsafe { device.get_buffer_device_address(&info) })
    }
}
