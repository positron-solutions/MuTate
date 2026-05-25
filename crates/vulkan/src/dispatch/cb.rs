// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Command Buffer
//!
//! Raw Vulkan command buffers, wrapped in compile-time guarantees to enable building reliable APIs
//! for consumers.
//!
//! ## Type-State Lifecycle
//!
//! - no commands before beginning recording
//! - no rendering-specific commands outside rendering
//! - no submission while still recording
//!
//! See the [vulkan spec](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html) section on
//! command buffers for an explanation of the full state machine we're wrapping.
//!
//! ### Reuse Ergonomics
//!
//! Valid re-use of sequential buffers requires on a timeline semaphore to exclude concurrent
//! execution of the same buffer. `SimultaneousUse` buffers can be used several times even within
//! the same submission although data dependencies would require barriers anyway.
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

// NEXT support device recorded commands.  Unsure workflow.  Go figure it out.
// NEXT it would be useful to enforce single-use-in-flight for Sequential buffers, but that will
// likely involve a wrapper type for each buffer so that necessary sync state can travel along with
// the buffer.
// NOTE Simultaneous model buffers can be re-used, but reset of an in-flight buffer is invalid, so
// likely this remains unsafe since enforcing return of all handles across threads is not fun.
use std::marker::PhantomData;

use drop_bomb::DropBomb;

use crate::internal::*;

use super::{Resettable, SubmissionModel};

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
                        " must be consumed via a typed transition method but was dropped"
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

cb_state!(InitialBuffer<C: Capability, M: SubmissionModel>);
cb_state!(InitialSecondary<C: Capability, M: SubmissionModel>);
cb_state!(ExecutableBuffer<C: Capability, M: SubmissionModel>);
cb_state!(ExecutableSecondary<C: Capability, M: SubmissionModel>);
cb_state!(RecordingBuffer<C: Capability, M: SubmissionModel>);
cb_state!(RecordingSecondary<C: Capability, M: SubmissionModel>);
// NOTE Rendering buffers and adjacent type states implicitly bear Graphics Capability
cb_state!(RenderingBuffer<M: SubmissionModel>);
cb_state!(RenderingSecondary<M: SubmissionModel>);

// Executable buffers are not used in any calls that would violate thread safety of the Pool.  They
// are basically read only at this point and may be shared across threads.
unsafe impl<C: Capability + Send, M: SubmissionModel + Send> Send for ExecutableBuffer<C, M> {}
unsafe impl<C: Capability + Send, M: SubmissionModel + Send> Send for ExecutableSecondary<C, M> {}

impl<C: Capability, M: SubmissionModel> InitialBuffer<C, M> {
    /// Begin recording.  Consumes the initial-state handle and returns a recording-state handle.
    pub fn begin(
        self,
        device_context: &DeviceContext,
    ) -> Result<RecordingBuffer<C, M>, VulkanError> {
        let raw = self.into_parts();
        let begin_info = vk::CommandBufferBeginInfo::default().flags(M::BUFFER_FLAGS);
        unsafe {
            device_context
                .device()
                .begin_command_buffer(raw, &begin_info)?;
        }
        Ok(RecordingBuffer::from_raw(raw))
    }
}

impl<C: Capability, M: SubmissionModel> InitialSecondary<C, M> {
    /// Begin recording a secondary buffer.  For compute secondaries the inheritance info is
    /// all-defaults; for graphics, the caller must supply the appropriate inheritance.
    pub fn begin(
        self,
        device_context: &DeviceContext,
        inheritance: &vk::CommandBufferInheritanceInfo,
    ) -> Result<RecordingSecondary<C, M>, VulkanError> {
        let raw = self.into_parts();
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(M::BUFFER_FLAGS)
            .inheritance_info(inheritance);
        unsafe {
            device_context
                .device()
                .begin_command_buffer(raw, &begin_info)?;
        }
        Ok(RecordingSecondary::from_raw(raw))
    }
}

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

impl<C: Capability, M: SubmissionModel + Resettable> ExecutableBuffer<C, M> {
    /// Reinterpret this buffer as being in the initial state.  This is implicit if you can
    /// guarantee that the source pool has been reset.
    pub unsafe fn assume_initial(self) -> InitialBuffer<C, M> {
        InitialBuffer::from_raw(self.into_parts())
    }

    /// When using recording models that support individual buffer resets, such as `Sequential` or
    /// `Simultaneous`, it is valid to reset individual buffers.  This must not be done while the
    /// buffers are in flight anywhere.
    pub unsafe fn reset(
        self,
        device_context: &DeviceContext,
        flags: vk::CommandBufferResetFlags,
    ) -> Result<InitialBuffer<C, M>, VulkanError> {
        let raw = self.into_parts();
        device_context.device().reset_command_buffer(raw, flags)?;
        Ok(InitialBuffer::from_raw(raw))
    }
}

impl<C: Capability, M: SubmissionModel + Resettable> ExecutableSecondary<C, M> {
    /// Reinterpret this buffer as being in the initial state.  This is implicit if you can
    /// guarantee that the source pool has been reset.
    pub unsafe fn assume_initial(self) -> InitialSecondary<C, M> {
        InitialSecondary::from_raw(self.into_parts())
    }

    /// When using recording models that support individual buffer resets, such as `Sequential` or
    /// `Simultaneous`, it is valid to reset individual buffers.  This must not be done while the
    /// buffers are in flight anywhere.
    pub unsafe fn reset(
        self,
        device_context: &DeviceContext,
        flags: vk::CommandBufferResetFlags,
    ) -> Result<InitialSecondary<C, M>, VulkanError> {
        let raw = self.into_parts();
        device_context.device().reset_command_buffer(raw, flags)?;
        Ok(InitialSecondary::from_raw(raw))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[cfg(test)]
    pub fn test() {}
}
