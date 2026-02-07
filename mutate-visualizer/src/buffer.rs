// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Buffer
//!
//! The `MappedAllocation` is just a first pass at wrapping up a persistently mapped Vulkan buffer
//! that we will flush to the GPU on every frame.  This module should grow to provide a decent
//! baseline of SSBO techniques.
//!
//! This treatment does not use any kind of RAII.  You have validation layers and other Vulkan
//! debugging tools to spot lifecycle issues.

use std::ptr::NonNull;

use ash::vk;

use mutate_lib::{self as utate, context::VkContext};

use crate::util;

// DEBT memory management
pub struct MappedAllocation<T> {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: NonNull<T>,
    pub len: usize,

    // XXX Why?
    pub size_bytes: vk::DeviceSize,
    pub memory_type_index: u32,
}

// DEBT memory allocation.  Consolidate device concerns up into VkContext.devices
impl<T> MappedAllocation<T> {
    pub fn new(size: usize, context: &VkContext) -> Result<Self, utate::MutateError> {
        let device = context.device();
        let buffer_info = vk::BufferCreateInfo {
            size: (std::mem::size_of::<T>() * size) as u64,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { device.create_buffer(&buffer_info, None).unwrap() };
        let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_props = unsafe {
            context
                .instance
                .get_physical_device_memory_properties(context.physical_device)
        };

        let memory_type_index = util::find_memory_type_index(
            &mem_req,
            &mem_props,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .ok_or(utate::MutateError::Vulkan(
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ))?;

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index,
            ..Default::default()
        };
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)?;
        }

        let raw_ptr =
            unsafe { device.map_memory(memory, 0, mem_req.size, vk::MemoryMapFlags::empty())? };

        let ptr = NonNull::new(raw_ptr as *mut T).unwrap();

        Ok(Self {
            buffer,
            ptr,
            len: size,
            memory,
            size_bytes: mem_req.size,
            memory_type_index: 0,
        })
    }

    // DEBT memory management.  We just need to devolve the allocation into a memento that can be
    // recycled or destroyed asynchronously.
    pub fn destroy(&self, context: &VkContext) -> Result<(), utate::MutateError> {
        let device = context.device();
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
    pub fn flush(&mut self, context: &VkContext) -> Result<(), utate::MutateError> {
        let flush_range = vk::MappedMemoryRange {
            memory: self.memory,
            offset: 0,
            size: self.size_bytes,
            ..Default::default()
        };
        unsafe {
            context
                .device()
                .flush_mapped_memory_ranges(&[flush_range])?;
        }
        Ok(())
    }

    /// After-compute shader barrier.  Use after some compute shader writes to a buffer.
    pub fn barrier_compute_post(&self, cb: &vk::CommandBuffer, context: &VkContext) {
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
            context.device().cmd_pipeline_barrier(
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
    pub fn barrier_compute_pre(&self, cb: &vk::CommandBuffer, context: &VkContext) {
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
            context.device().cmd_pipeline_barrier(
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
}

// NEXT we will need support for slicing up the screen for the multiple choice UI.
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
