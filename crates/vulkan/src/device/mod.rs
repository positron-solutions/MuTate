// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Logical Device

pub mod descriptors;
pub mod device;
pub mod queue;

pub use device::Device;

pub mod prelude {
    pub use super::device::Device;
    pub use super::device::Fence;
    pub use super::queue::prelude::*;
}
