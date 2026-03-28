// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Descriptors
//!
//! (◕‿◕)︵‿︵‿︵‿︵┻━┻
//!
//! (◕‿◕)ノ彡☆
//!
//! > You make a single descriptor set (or even better: descriptor heap!) with 2^20 texture
//! > descriptors and  2^12 sampler descriptors. You use buffer device address instead of buffer
//! > descriptors.  You pass around pointer and indices into the set via push constants. And then
//! > you never think about descriptor sets ever again. **- Afiery1**
//!
//! The implementation requires thinking about descriptors just one more time.  If it's not clear,
//! the resulting geometry is:
//!
//!   `[DescriptorSet[DescriptorArray[Index]]]`
//!
//! A descriptor array is a descriptor for an array, not an array of descriptors... well, maybe it
//! is an array of descriptors, but the point is we want to only care about indexes.  The size of
//! this array is:
//!
//! `[No. of Vulkan Applications (one)][No. of types (a few)][No. of items (a zillion)]`
//!
//! **So we have one array for each type and we index into it.**  Simple!
//!
//! The rest of the complexity pretty much boils down to re-claiming recycled descriptors.  We will
//! make one structure to track all of our descriptors.  It initializes with a big descriptor set.
//! It has a static fixed size because any kind of dynamic growth messes up the descriptor slots and
//! forces us to think about descriptors.  Okay, glad we are experts at Vulkan now!

// NOTE Vulkan does *not* required us to unbind descriptors.  We don't have to null slots.  Debug
// assert cheap invariants, but memory leaks will dominate the signal for any leaks of descriptor
// bound resources that occur.

use std::{collections::VecDeque, slice};

use ash::vk;

use crate::descriptor_newtype;
use crate::prelude::*;
use crate::resource::image::ImageView;

pub use crate::slang::{
    DescriptorIndex, SampledImageIdx, SamplerIdx, SsboIdx, StorageImageIdx, UInt, UboIdx,
};

// Plural to make it kind of obvious that these are array slot indexes.
pub const SLOT_SAMPLED_IMAGES: u32 = 0;
pub const SLOT_SAMPLERS: u32 = 1;
pub const SLOT_STORAGE_IMAGES: u32 = 2;
pub const SLOT_UNIFORM_BUFFERS: u32 = 3;
pub const SLOT_STORAGE_BUFFERS: u32 = 4;

// NEXT Methods to hand out and recycle indexes, likely guarded through the context interface.
// Planning on coordinating Image and Buffer creation because not having descriptors would make them
// kind of useless.
pub struct Descriptors {
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,

    // Track the next never-used index.  This is implicitly a high-water mark for properly sizing
    // arrays.
    next_sampled_image: SampledImageIdx,
    next_sampler: SamplerIdx,
    next_storage_image: StorageImageIdx,
    next_ubo: UboIdx,
    next_ssbo: SsboIdx,

    // Keep any re-usable indexes.
    freelist_sampled_images: VecDeque<SampledImageIdx>,
    freelist_samplers: VecDeque<SamplerIdx>,
    freelist_storage_images: VecDeque<StorageImageIdx>,
    freelist_ubos: VecDeque<UboIdx>,
    freelist_ssbos: VecDeque<SsboIdx>,

    default_samplers: [vk::Sampler; samplers::N_DEFAULTS],
}

impl Descriptors {
    pub fn new(device: &ash::Device) -> Result<Self, VulkanError> {
        // DEBT Max descriptor size calculation / management.
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 256,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 256,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 256,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 256,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 256,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        // NEXT obviously users might want to specify different limits.  Bon them into the logical
        // device creation as 🪄
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(SLOT_SAMPLED_IMAGES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(SLOT_SAMPLERS)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(SLOT_STORAGE_IMAGES)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(SLOT_UNIFORM_BUFFERS)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(SLOT_STORAGE_BUFFERS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

        // NOTE Stack storage like this common for Vulkan conversations
        let mut default_samplers = [vk::Sampler::null(); samplers::N_DEFAULTS];
        let mut image_infos = [vk::DescriptorImageInfo::default(); samplers::N_DEFAULTS];
        let mut writes = [vk::WriteDescriptorSet::default(); samplers::N_DEFAULTS];

        for (i, ((sampler, image_info), write)) in default_samplers
            .iter_mut()
            .zip(image_infos.iter_mut())
            .zip(writes.iter_mut())
            .enumerate()
        {
            let ci = &samplers::default_samplers()[i];
            *sampler = unsafe { device.create_sampler(ci, None)? };
            *image_info = vk::DescriptorImageInfo {
                sampler: *sampler,
                image_view: vk::ImageView::null(),
                image_layout: vk::ImageLayout::UNDEFINED,
            };
            *write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(SLOT_SAMPLERS)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .dst_array_element(i as u32)
                .image_info(std::slice::from_ref(image_info));
        }
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(Self {
            pool,
            set,
            layout,

            next_sampled_image: SampledImageIdx::new(0),
            next_sampler: SamplerIdx::new(samplers::N_DEFAULTS as u32),
            next_storage_image: StorageImageIdx::new(0),
            next_ubo: UboIdx::new(0),
            next_ssbo: SsboIdx::new(0),

            // FIXME duplicates pool sizes
            freelist_sampled_images: VecDeque::with_capacity(256),
            freelist_samplers: VecDeque::with_capacity(256),
            freelist_storage_images: VecDeque::with_capacity(256),
            freelist_ubos: VecDeque::with_capacity(256),
            freelist_ssbos: VecDeque::with_capacity(256),

            default_samplers,
        })
    }

    /// Access the descriptor set, basically for binding pipelines.
    // This might become a lot more implicit
    pub fn set(&self) -> vk::DescriptorSet {
        self.set.clone()
    }

    /// Return the default layout.  Useful for creating pipelines etc.
    pub fn layout(&self) -> &[vk::DescriptorSetLayout] {
        std::slice::from_ref(&self.layout)
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            for &s in &self.default_samplers {
                device.destroy_sampler(s, None);
            }
            device.destroy_descriptor_set_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
        }
    }

    /// `layout` must be the layout that is intended for use, not the image's current layout.  If
    /// you need multiple layouts, you need multiple descriptors.  The returned index may be used in
    /// shaders and will later type-check against DescriptorHandles during introspection.
    pub fn bind_sampled_image(
        &mut self,
        device: &ash::Device,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> SampledImageIdx {
        let descriptor_info = vk::DescriptorImageInfo::default()
            .image_layout(layout)
            .image_view(view);

        let index = self.freelist_sampled_images.pop_back().unwrap_or_else(|| {
            let next = self.next_sampled_image;
            self.next_sampled_image = SampledImageIdx::new(next.raw() + 1);
            next
        });

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(SLOT_SAMPLED_IMAGES)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_array_element(index.raw())
            .image_info(slice::from_ref(&descriptor_info));

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write), &[]);
        }
        index
    }

    pub fn bind_ssbo(
        &mut self,
        device: &ash::Device,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> SsboIdx {
        let descriptor_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(size);

        let index = self.freelist_ssbos.pop_back().unwrap_or_else(|| {
            let next = self.next_ssbo;
            self.next_ssbo = SsboIdx::new(next.raw() + 1);
            next
        });

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(SLOT_STORAGE_BUFFERS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_array_element(index.raw())
            .buffer_info(slice::from_ref(&descriptor_info));

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write), &[]);
        }
        index
    }

    /// Return the index for the descriptor slot to the free list.  You should not use this index
    /// again because it may belong to a new resource.
    pub fn unbind_image(&mut self, index: SampledImageIdx) {
        debug_assert!(
            index.raw() < self.next_sampled_image.raw(),
            "unbind_ssbo: index {:?} was never bound",
            index
        );
        self.freelist_sampled_images.push_back(index);
    }

    /// Return the index for the descriptor slot to the free list.  You should not use this index
    /// again because it may belong to a new resource.
    pub fn unbind_ssbo(&mut self, index: SsboIdx) {
        debug_assert!(
            index.raw() < self.next_ssbo.raw(),
            "unbind_ssbo: index {:?} was never bound",
            index
        );
        self.freelist_ssbos.push_back(index);
    }
}

pub(crate) mod samplers {
    use super::*;

    // NOTE didn't want to explicitly double wrap, but without into, this is the way?
    pub const NEAREST_CLAMP: SamplerIdx = SamplerIdx(UInt(0));
    /// Bi-linear, clamp-to-edge.  Smooth, no tiling.
    pub const LINEAR_CLAMP: SamplerIdx = SamplerIdx(UInt(1));
    /// Bi-linear, repeat.  Standard tiling textures.
    pub const LINEAR_REPEAT: SamplerIdx = SamplerIdx(UInt(2));
    /// Bi-linear + tri-linear mip interpolation, clamp.  World geometry etc.
    pub const LINEAR_MIP: SamplerIdx = SamplerIdx(UInt(3));
    /// Linear, clamp, border=1.  For `sampler2DShadow` PCF.
    pub const SHADOW: SamplerIdx = SamplerIdx(UInt(4));

    pub const N_DEFAULTS: usize = 5;

    pub(crate) fn default_samplers() -> [vk::SamplerCreateInfo<'static>; N_DEFAULTS] {
        [
            // 0 — NEAREST_CLAMP
            vk::SamplerCreateInfo {
                mag_filter: vk::Filter::NEAREST,
                min_filter: vk::Filter::NEAREST,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..vk::SamplerCreateInfo::default()
            },
            // 1 — LINEAR_CLAMP
            vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..vk::SamplerCreateInfo::default()
            },
            // 2 — LINEAR_REPEAT
            vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::REPEAT,
                address_mode_v: vk::SamplerAddressMode::REPEAT,
                address_mode_w: vk::SamplerAddressMode::REPEAT,
                ..vk::SamplerCreateInfo::default()
            },
            // 3 — LINEAR_MIP
            vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                min_lod: 0.0,
                max_lod: vk::LOD_CLAMP_NONE,
                ..vk::SamplerCreateInfo::default()
            },
            // 4 — SHADOW
            vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_BORDER,
                border_color: vk::BorderColor::FLOAT_OPAQUE_WHITE,
                compare_enable: vk::TRUE,
                compare_op: vk::CompareOp::LESS_OR_EQUAL,
                ..vk::SamplerCreateInfo::default()
            },
        ]
    }
}
