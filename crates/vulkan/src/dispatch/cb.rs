// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Command Buffer
//!
//! Raw Vulkan command buffers, wrapped in compile-time guarantees to enable building reliable APIs
//! for consumers.
//!
//! ## Typed Begin & End
//!
//! - no commands before beginning recording
//! - no rendering-specific commands outside rendering
//! - no submission while still recording
//!
//! ## Enforced Usage
//!
//! Buffers and secondaries must be consumed via [`ExecutableBuffer`] and [`ExecutableSecondary`],
//! preventing accidental implicit drops of buffers that would invalidate upstream command pool
//! accounting.  It is considered a **bug** to implicitly drop any command buffer.  Always use a
//! consuming method.
//!
//! ## Typed by Origin & Usage Scheme
//!
//! The originating command pool and its queue family control a lot of functionality and are part of
//! each type!
//!
//! - Queue capabilities (see **Capabilities** section of [queue](crate::context::queue) module.)
//! - Reset, simultaenous use, and one-time-use semantics
//! - Support device recorded commands (see NEXT)
//!
//! ## Flexible for Consumers
//!
//! When delegating draw and compute to code that plugs into a runtime, the code being delegated too
//! shouldn't care about the buffer re-use strategy.  Delegates accept buffers by trait, all types
//! of which are the same size.
//!
//! - [`RecordingBuffer`] or [`RecordingSecondary`]
//! - [`RenderingBuffer`] or [`RenderingSecondary`]
//! - [`ExecutableBuffer`] or [`ExecutableSecondary`]
//!
//! All Ash methods that accept a raw command buffer can also use the `Deref` implementation for
//! buffers.  If you make higher level interfaces that consume typed buffers, great!  If you need to
//! drop to `ash` bindings within typed interfaces, also great!
//!
//! The `as_raw()` and `into_raw()` methods returns the [`ash::vk::CommandBuffer`] handle as a
//! temporary escape hatch for missing support.
//!
//! ## Submission Ergonomics
//!
//! - group together submissions
//! - append signal or wait semaphores and fences to each group
//! - obtain signal semaphores for use in other groups

// MAYBE barrier chaining API to add a barrier between A and B.  Can use extra command buffers to
// decouple recording ordering, enabling B to be recorded and then A->B barriers to be inserted
// before B in the queue submission.  Other solutions involve stack discipline and accounting
// structures to create the ordering and dependency before recording.  Seems like extra sync buffers
// would be preferred amirite?
// NEXT support device recorded commands.  Unsure workflow.  Go figure it out.
use std::marker::PhantomData;

use drop_bomb::DropBomb;

use crate::internal::*;

use super::SubmissionModel;

pub struct CommandBuffer<C: Capability, M: SubmissionModel> {
    pub(crate) raw: vk::CommandBuffer,
    pub(crate) bomb: DropBomb,
    _cap: PhantomData<C>,
    _model: PhantomData<M>,
}

impl<C: Capability, M: SubmissionModel> CommandBuffer<C, M> {
    /// Borrow the raw `ash::vk::CommandBuffer` handle.
    pub unsafe fn as_raw(&self) -> &vk::CommandBuffer {
        &self.raw
    }

    /// Consume the raw `ash::vk::CommandBuffer` handle.  Drop protection is disarmed, so the caller
    /// is responsible for tracking further usage.
    pub unsafe fn into_raw(self) -> vk::CommandBuffer {
        let Self { mut bomb, raw, .. } = self;
        bomb.defuse();
        raw
    }
}

// Define the state.  All states have Deref to raw for use with ash APIs.  Most ash APIs that use
// raw command buffers are already unsafe, so we will not add any extra ceremony onto the deref.
// The borrow is consistent with drop accounting and type-state.
macro_rules! cb_state {
    (
        $(#[$meta:meta])*
        $name:ident < $( $param:ident : $bound:path ),+ >
    ) => {
        $(#[$meta])*
        pub struct $name< $( $param: $bound ),+ > {
            pub(crate) raw: vk::CommandBuffer,
            pub(crate) bomb: DropBomb,
            _phantom: PhantomData<( $( *const $param ),+ )>,
        }

        impl< $( $param: $bound ),+ > $name< $( $param ),+ > {
            // XXX re-private to crate
            pub fn from_raw(raw: vk::CommandBuffer) -> Self {
                Self {
                    raw,
                    bomb: DropBomb::new(concat!(
                        stringify!($name),
                        " must be consumed via a typed transition method, not implicitly dropped"
                    )),
                    _phantom: PhantomData,
                }
            }

            pub(crate) fn into_parts(self) -> vk::CommandBuffer {
                let Self {mut bomb, raw, ..} = self;
                bomb.defuse();
                raw
            }
        }

        impl< $( $param: $bound ),+ > std::ops::Deref for $name< $( $param ),+ > {
            type Target = vk::CommandBuffer;
            fn deref(&self) -> &Self::Target {
                &self.raw
            }
        }
    };
}

cb_state!(ExecutableBuffer<C: Capability, M: SubmissionModel>);
cb_state!(ExecutableSecondary<C: Capability, M: SubmissionModel>);
cb_state!(RecordingBuffer<C: Capability, M: SubmissionModel>);
cb_state!(RecordingSecondary<C: Capability, M: SubmissionModel>);
// NOTE Rendering buffers and adjacent type states implicitly bear Graphics Capability
cb_state!(RenderingBuffer<M: SubmissionModel>);
cb_state!(RenderingSecondary<M: SubmissionModel>);

impl<C: Capability, M: SubmissionModel> RecordingBuffer<C, M> {
    pub fn end(
        self,
        device_context: &DeviceContext,
    ) -> Result<ExecutableBuffer<C, M>, VulkanError> {
        let raw = self.into_parts();
        unsafe {
            device_context.device().end_command_buffer(raw)?;
        }
        Ok(ExecutableBuffer::from_raw(raw))
    }
}

// XXX get rid of this
impl<C: Capability, M: SubmissionModel> ExecutableBuffer<C, M> {
    pub fn kill(self, device_context: &DeviceContext) -> Result<vk::CommandBuffer, VulkanError> {
        Ok(self.into_parts())
    }
}

// DEBT Rendering info is quite unsafe.
impl<M: SubmissionModel> RecordingBuffer<Graphics, M> {
    /// Begin a dynamic rendering scope.  end rendering via [`RenderingBuffer::end_rendering`] or
    /// [`RenderinBuffer::end`] to directly close the buffer as well.
    pub fn begin_rendering(
        self,
        device_context: &DeviceContext,
        rendering_info: &vk::RenderingInfo,
    ) -> RenderingBuffer<M> {
        let raw = self.into_parts();
        unsafe {
            device_context
                .device()
                .cmd_begin_rendering(raw, rendering_info);
        }
        RenderingBuffer::from_raw(raw)
    }
}

impl<M: SubmissionModel> RenderingBuffer<M> {
    /// End the rendering scope.
    pub fn end_rendering(self, device_context: &DeviceContext) -> RecordingBuffer<Graphics, M> {
        let raw = self.into_parts();
        unsafe {
            device_context.device().cmd_end_rendering(raw);
        }
        RecordingBuffer::from_raw(raw)
    }

    // MAYBE do finished buffers even need Capability besides for secondaries?
    pub fn end(
        self,
        device_context: &DeviceContext,
    ) -> Result<ExecutableBuffer<Graphics, M>, VulkanError> {
        let raw = self.into_parts();
        unsafe {
            device_context.device().cmd_end_rendering(raw);
            device_context.device().end_command_buffer(raw)?;
        }
        Ok(ExecutableBuffer::from_raw(raw))
    }
}

#[cfg(test)]
mod test {
    use super::*;
}
