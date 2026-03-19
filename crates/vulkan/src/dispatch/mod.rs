// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Dispatch
//!
//! Command buffers and you.  This module encapsulates the valid lifecycle of a command buffer.  The
//! overall strategy can be described as **optionally typed** for callees.  Give any command buffer
//! to ash bindings.  However, at internal API boundaries that are prone to abuse, typed buffers can
//! be specified to prevent actual plausible mix-ups.
//!
//! Guarantees We Support:
//!
//! - A command buffer obtained as if from a transfer queue cannot be given to a graphics-only call
//!   (unless you opt out by directly calling ash bindings).  The same guarantee is made for
//!   a buffer requested as compute-only becoming used as graphics.
//! - Command buffer recording lifecycles are enforced.
//! - Just call `into_inner` if some contract is in the way for now.
//!
//! ## Lifecycle
//!
//! ```text
//! Initial ──begin()──► Recording ──end()──► Executable ──submit()──► Pending
//!                          │                                              │
//!                     (commands)                                        (reset)
//! ```
//!
//! State transitions are consuming — you cannot hold a `CommandBuffer<_, Recording>` and
//! accidentally call `submit()` on it.  The raw handle is recoverable at any state via `Deref`.
// pub mod compute;
// pub mod graphics;
pub mod sync;
// pub mod transfer;

use std::marker::PhantomData;

use ash::vk;

use crate::{context::DeviceContext, VulkanError};

// Marker traits for capabilities
trait TransferCap {}
trait ComputeCap: TransferCap {}
trait GraphicsCap: ComputeCap {}

// Command buffer capabilities
pub struct Graphics;
pub struct Compute;
pub struct Transfer;

impl TransferCap for Transfer {}
impl TransferCap for Compute {}
impl TransferCap for Graphics {}
impl ComputeCap for Compute {}
impl ComputeCap for Graphics {}
impl GraphicsCap for Graphics {}

/// Allocated but `vkBeginCommandBuffer` has not been called.
pub struct Initial;
/// `vkBeginCommandBuffer` has been called; commands may be recorded.
pub struct Recording;
/// `vkEndCommandBuffer` has been called; ready for submission.
pub struct Executable;
/// Submitted to a queue; GPU may be executing this buffer.
/// The buffer must not be re-recorded or re-submitted until the
/// associated fence has signalled and the pool has reset it.
pub struct Pending;

/// A command buffer parameterised by both capability (`Cap`) and lifecycle state (`State`).
/// Type-state transitions are consuming.
#[derive(Copy, Clone)]
pub struct CommandBuffer<Cap, State> {
    raw: vk::CommandBuffer,
    _cap: PhantomData<Cap>,
    _state: PhantomData<State>,
}

impl<Cap, State> CommandBuffer<Cap, State> {
    /// Return a clone of the raw handle for any function calls not yet supported through contract.
    #[inline]
    pub fn raw(self) -> vk::CommandBuffer {
        self.raw
    }
}

impl<Cap> CommandBuffer<Cap, Initial> {
    /// Begin recording.  Consumes the `Initial` buffer and returns it in the `Recording` state.
    ///
    /// Corresponds to `vkBeginCommandBuffer`.
    pub fn begin(
        self,
        context: &DeviceContext,
    ) -> Result<CommandBuffer<Cap, Recording>, VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            context
                .device()
                .begin_command_buffer(self.raw, &begin_info)?
        };
        Ok(CommandBuffer {
            raw: self.raw,
            _cap: PhantomData,
            _state: PhantomData,
        })
    }
}

impl<Cap> CommandBuffer<Cap, Recording> {
    /// Finish recording.  Consumes the `Recording` buffer and returns it in the `Executable` state.
    ///
    /// Corresponds to `vkEndCommandBuffer`.
    pub fn end(
        self,
        context: &DeviceContext,
    ) -> Result<CommandBuffer<Cap, Executable>, VulkanError> {
        unsafe { context.device().end_command_buffer(self.raw)? };
        Ok(CommandBuffer {
            raw: self.raw,
            _cap: PhantomData,
            _state: PhantomData,
        })
    }
}

/// A temporary borrow that presents a `CommandBuffer` at a lower capability level.
/// Useful for passing a `Graphics` buffer into a function that only needs `Transfer`.
struct CommandBufferView<'a, Cap, State> {
    raw: vk::CommandBuffer,
    _borrow: PhantomData<&'a mut ()>,
    _cap: PhantomData<Cap>,
    _state: PhantomData<State>,
}

impl<Cap: GraphicsCap, State> CommandBuffer<Cap, State> {
    pub fn as_compute(&mut self) -> CommandBufferView<'_, Compute, State> {
        CommandBufferView {
            raw: self.raw,
            _borrow: PhantomData,
            _cap: PhantomData,
            _state: PhantomData,
        }
    }
}

impl<Cap: ComputeCap, State> CommandBuffer<Cap, State> {
    pub fn as_transfer(&mut self) -> CommandBufferView<'_, Transfer, State> {
        CommandBufferView {
            raw: self.raw,
            _borrow: PhantomData,
            _cap: PhantomData,
            _state: PhantomData,
        }
    }
}

// XXX Command Context?
/// omg,
pub struct CommandPool {
    pool: vk::CommandPool,
    // buffers: [CommandBuffer<Cap, Initial>; N],
    // _cap: PhantomData<Cap>,
}

impl CommandPool {
    pub fn new(context: &DeviceContext, frames: usize) -> Self {
        // pub queue: vk::Queue,
        // pub command_pool: vk::CommandPool,
        todo!()
    }

    pub fn destroy(&self, context: &DeviceContext) {
        todo!()
    }

    // /// Reset and return the command buffer for the given frame index,
    // /// ready to begin recording.
    // pub fn reset(&mut self, context: &DeviceContext, frame: usize) -> &CommandBuffer<Cap, Initial> {
    //     todo!()
    // }
}

#[cfg(test)]
mod test {
    use super::*;

    // fn needs_transfer(cmd: &mut impl TransferCommands) {
    //     cmd.copy_buffer(0xDEAD, 0xBEEF, 1024);
    // }

    trait Null: Sized {
        fn null() -> Self;
    }

    impl<Cap, State> Null for CommandBuffer<Cap, State> {
        fn null() -> Self {
            Self {
                raw: vk::CommandBuffer::null(),
                _cap: PhantomData,
                _state: PhantomData,
            }
        }
    }

    // #[test]
    // fn recording_borrow_downcast() {
    //     let mut graphics_cmd: CommandBuffer<Graphics, Recording> = CommandBuffer::null();
    //     let mut transfer_view = graphics_cmd.as_transfer();
    //     needs_transfer(&mut transfer_view);
    // }
}
