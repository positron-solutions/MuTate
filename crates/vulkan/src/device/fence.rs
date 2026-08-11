// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Fence
//!
//! Newtype wrapper around `vk::Fence`.  Has not yet grown signaled vs unsignaled split API.  Fences
//! are only being used where absolutely requireed, so not a high priority.

use crate::internal::*;

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
        unsafe { device.destroy_fence(self.0, None) }
    }
}

impl std::ops::Deref for Fence {
    type Target = vk::Fence;
    fn deref(&self) -> &vk::Fence {
        &self.0
    }
}
