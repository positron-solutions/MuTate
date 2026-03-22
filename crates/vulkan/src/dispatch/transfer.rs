// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Transfer
//!
//! Explicitly supported commands specific to `CommandBuffer`s with the `TransferCap` capability.

use super::{CommandBuffer, CommandBufferView, Recording, Transfer, TransferCap};

pub trait TransferCommands {
    fn copy_buffer(&mut self, src: u64, dst: u64, size: u64);
}

impl<Cap: TransferCap> TransferCommands for CommandBuffer<Cap, Recording> {
    fn copy_buffer(&mut self, src: u64, dst: u64, size: u64) {
        // vkCmdCopyBuffer
    }
}

impl<'a> TransferCommands for CommandBufferView<'a, Transfer, Recording> {
    fn copy_buffer(&mut self, src: u64, dst: u64, size: u64) {
        // vkCmdCopyBuffer
    }
}
