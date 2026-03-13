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

use std::{collections::VecDeque, slice};

use ash::vk;

use crate::prelude::*;
use crate::resource::image::ImageView;
use crate::slang::{prelude::*, Int32};
use crate::slang_newtype;

// NEXT these SlangType strings will need some correlation with newtype wrappers in Slang.
slang_newtype!(SampledImageIndex, u32, "SampledImageIndex");
slang_newtype!(SamplerIndex, u32, "SamplerIndex");
slang_newtype!(StorageImageIndex, u32, "StorageImageIndex");
slang_newtype!(UboIndex, u32, "UboIndex");
slang_newtype!(SsboIndex, u32, "SsboIndex");

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
    set: vk::DescriptorSet,
    layout: vk::DescriptorSetLayout,

    // Track the next never-used index.  This is implicitly a high-water mark for properly sizing
    // arrays.
    next_sampled_image: u32,
    next_sampler: u32,
    next_storage_image: u32,
    next_ubo: u32,
    next_ssbo: u32,

    // Keep any re-usable indexes.
    freelist_sampled_images: VecDeque<u32>,
    freelist_samplers: VecDeque<u32>,
    freelist_storage_images: VecDeque<u32>,
    freelist_ubos: VecDeque<u32>,
    freelist_ssbos: VecDeque<u32>,

    default_samplers: [vk::Sampler; samplers::N_DEFAULTS],
}

impl Descriptors {
    pub fn new(device: &ash::Device, pool: &ash::vk::DescriptorPool) -> Result<Self, VulkanError> {
        // NEXT obviously users might want to specify different limits.
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
            .descriptor_pool(*pool)
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
            set,
            layout,

            next_sampled_image: 0,
            next_sampler: samplers::N_DEFAULTS as u32,
            next_storage_image: 0,
            next_ubo: 0,
            next_ssbo: 0,

            freelist_sampled_images: VecDeque::with_capacity(256),
            freelist_samplers: VecDeque::with_capacity(256),
            freelist_storage_images: VecDeque::with_capacity(256),
            freelist_ubos: VecDeque::with_capacity(256),
            freelist_ssbos: VecDeque::with_capacity(256),

            default_samplers,
        })
    }

    pub fn destroy(&self, device: &ash::Device) {
        for &s in &self.default_samplers {
            unsafe { device.destroy_sampler(s, None) };
        }
        unsafe {
            device.destroy_descriptor_set_layout(self.layout, None);
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
    ) -> SampledImageIndex {
        let descriptor_info = vk::DescriptorImageInfo::default()
            .image_layout(layout)
            .image_view(view);

        let index = self.freelist_sampled_images.pop_back().unwrap_or_else(|| {
            let next = self.next_sampled_image;
            self.next_sampled_image += 1;
            next
        });

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(SLOT_SAMPLED_IMAGES)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_array_element(index)
            .image_info(slice::from_ref(&descriptor_info));

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write), &[]);
        }

        SampledImageIndex(index)
    }

    pub fn bind_ssbo(
        &mut self,
        device: &ash::Device,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> SsboIndex {
        let descriptor_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(size);

        let index = self.freelist_ssbos.pop_back().unwrap_or_else(|| {
            let next = self.next_ssbo;
            self.next_ssbo += 1;
            next
        });

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(SLOT_STORAGE_BUFFERS)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_array_element(index)
            .buffer_info(slice::from_ref(&descriptor_info));

        unsafe {
            device.update_descriptor_sets(slice::from_ref(&write), &[]);
        }

        SsboIndex(index)
    }

    /// Return the default layout.  Useful for creating pipelines etc.
    pub fn layout(&self) -> &[vk::DescriptorSetLayout] {
        std::slice::from_ref(&self.layout)
    }
}

pub(crate) mod samplers {
    use super::*;

    pub const NEAREST_CLAMP: SamplerIndex = SamplerIndex(0);
    /// Bilinear, clamp-to-edge.  Smooth, no tiling.
    pub const LINEAR_CLAMP: SamplerIndex = SamplerIndex(1);
    /// Bilinear, repeat.  Standard tiling textures.
    pub const LINEAR_REPEAT: SamplerIndex = SamplerIndex(2);
    /// Bilinear + trilinear mip interpolation, clamp.  World geometry etc.
    pub const LINEAR_MIP: SamplerIndex = SamplerIndex(3);
    /// Linear, clamp, border=1.  For `sampler2DShadow` PCF.
    pub const SHADOW: SamplerIndex = SamplerIndex(4);

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
