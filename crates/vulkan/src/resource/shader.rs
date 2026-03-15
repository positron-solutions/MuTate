// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Shader
//!
//! Load shaders from assets.  We may do some caching for mass PSO creation someday.  Right now,
//! these modules tend to get used, abused, and dropped.

use ash::vk;

use mutate_assets as assets;

use crate::{context::VkContext, VulkanError};

// Let me see that sla--
//                       -aaa-
//                             --ang

pub struct ShaderModule<'ctx> {
    device: &'ctx ash::Device,
    pub module: vk::ShaderModule,
}

impl<'ctx> ShaderModule<'ctx> {
    pub fn load(path: &'static str, context: &'ctx VkContext) -> Result<Self, VulkanError> {
        // NEXT We could further type shader names to verify that hardcoded names exist.  Dynamic names for
        // source files doesn't really make sense unless the GPU has gone AGI and is emitting fresh slang
        // code to hot swap with itself.  Static shader file names would do some justice.
        let spv = context.assets.find_shader(path).unwrap();
        let module_ci = vk::ShaderModuleCreateInfo::default().code(spv.as_slice());
        let module = unsafe { context.device.create_shader_module(&module_ci, None)? };

        Ok(Self {
            device: &context.device,
            module,
        })
    }
}

impl std::ops::Deref for ShaderModule<'_> {
    type Target = vk::ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl Drop for ShaderModule<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.module, None) }
    }
}
