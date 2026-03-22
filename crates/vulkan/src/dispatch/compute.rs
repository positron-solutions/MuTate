// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Compute
//!
//! Explicitly supported commands specific to `CommandBuffer`s with the `ComputeCap` capability.

use super::{CommandBuffer, Compute, Recording};
use crate::context::DeviceContext;

impl CommandBuffer<Compute, Recording> {
    /// Record a compute dispatch.
    ///
    /// Corresponds to `vkCmdDispatch`.
    pub fn dispatch(
        &self,
        context: &DeviceContext,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) {
        unsafe {
            context
                .device()
                .cmd_dispatch(self.raw, group_count_x, group_count_y, group_count_z);
        }
    }
}
