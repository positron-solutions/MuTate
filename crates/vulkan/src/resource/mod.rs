// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Resource
//!
//! Gather up things that need asynchronous creation and might be discovered by name or reactively
//! recreated.  UBOs, SSBOs, and Images are definitely resources.  Shader modules might be resources
//! (such as dealing with Molten translation layer delay).

pub mod buffer;
pub mod image;
pub mod shader;
pub mod ubo;
