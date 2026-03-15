// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Vulkan Context
//!
//! Fundamentally required resources, including the entry, instance, hardware devices
//! are encapsulated by `VkContext`.
//!
//! *NEXT* The Devices and memory management are much more tightly bound together than the
//! `ash::Entry` and `ash::Instance`, so these will be separated when convenient.
//!
//! ## Enabled Features
//!
//! We aim to support a minimum set of modern tactics while still offering a complete, high
//! performance experience:
//!
//! - Buffer device address.
//! - One big descriptor set with one descriptor array per type (bindless).
//! - Flexible push constants & UBOs with scalar layout and 8/16-bit support.
//! - Vulkan 1.3+ minimum support, (switch to 1.4 when reasonable).
//! - Dynamic rendering

pub mod descriptors;
pub mod device;
pub mod queue;
pub mod vulkan;

pub use device::DeviceContext;
pub use vulkan::VkContext;
