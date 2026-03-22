// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Graphics
//!
//! Explicitly supported commands specific to `CommandBuffer`s with the `GraphicsCap` capability.

impl CommandBuffer<Graphics, Recording> {
    /// Record a non-indexed draw call.
    ///
    /// Corresponds to `vkCmdDraw`.
    pub fn draw(
        &self,
        context: &DeviceContext,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            context.device().cmd_draw(
                self.raw,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }
}
