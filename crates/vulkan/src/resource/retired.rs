// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Retired
//!
//! > Those who would hold it, lose it.
//! >
//! > - *Some virgin probably*
//!
//! This module shall be the location of all deferred destruction.
//!
//! ## Motivation
//!
//! The very typical case is a workload such as audio import wants to drop but has no idea if the
//! device might still be reading processed audio outputs.  The audio importer could package up the
//! resources and hand them back for deferred deletion after in-flight frames have retired, but now
//! the caller has the hot potato and it doesn't know what to do with them either.  If only there
//! was a central place to hand back resources for later destruction after some worst case pipeline
//! depth of ticks has elapsed...
//!
//! Welcome to Vulkan, aka scheduled adult programming.  Pre-marital deletions happen, and a mature
//! approach requires discussing protection, consent, and realistic solutions for the ideals our
//! souls are comprised of.  Most notably we are *transitively* handing out a lot of "lightweight
//! view" types that are only to be used ephemerally, passed into the dispatch and not retained any
//! longer.  We neglect their fine-grained accounting and instead trust them to evaporate into the
//! epoch of their usage.  Nonetheless, there remains a slight delay between the actual owner
//! dropping and the resource being safe to destroy.  A deferred destruction queue handles this
//! problem.
//!
//! Naive interpretations of ownership and shared mutability may not be ready to approach these
//! realities, but nor should their lack of readiness justify censorship of these discussions and
//! practices.  Those who are not ready should step away from the heat of the fire.
//!
//! ## Sound Usage
//!
//! - Only the object's **owner** may push a handle to the deletion queue.
//! - The deletion epoch must outlive the slowest in-flight items.
//! - The queue must be driven by some tick and drained when the driving clock has ceased.
//! - Dependent deletions must be queued in the correct order.
//!
//! ### Basically Correct Usage
//!
//! You can build your own queue and use it to drain some resources *later*.  Configure enough
//! epochs for the pipeline depth.  Call [`tick`] at some point when the pipeline is guaranteed to
//! have retired one item (ie a frame).  For all items that *could* be in flight, provide one epoch.
//! Queue things in the order that they would be deleted for manual deletion.  When you are done
//! with the whole bag, call [`drain`] to delete all remaining epochs.
//!
//! ## Design Tradeoffs
//!
//! - Deleting faster reclaims memory faster, but for small resources is not really saving memory
//!   and instead just makes it more likely to accidentally use-after-free.
//! - Delaying deletion more than the pipeline depth can only protect shared owners who are doing
//!   something forbidden, such as holding handles instead of only using them in dispatches.
//!   Extending the epoch time would only protect these users in an indeterminate way and so they
//!   must instead be discovered as soon as possible by always using a minimum safe epoch time.
//! - Only "owner" wrapper types may be blessed with the [`Retire`] trait.  Raw Vulkan handles
//!   frequently will be used by lightweight view types that do **not** own the resources, so we
//!   will require unsafe for raw handle types to be pushed to a queue.
//! - The chunked bags are FIFO.  This semantically maps correctly to the call sites, where the user
//!   can queue several dependents in order and achieve the effect of destroying them in order.
//! - Reference counting is for deciding when something *should* be deleted, not deciding when the
//!   last in-flight usage is *safe* to be deleted.  Reference counting is a job for abstract
//!   resources.  Epoch based destruction is what makes destruction of the abstract resource safe.
//!
//! ## Implementation
//!
//! A ring of FIFO access primitives as the epochs.  Handles and their types are queued separately
//! so that we know how to delete them.  The deletion queue owns a ring of epoch queues and drains
//! them individually on calls [`tick`] or drains all of them when [`drain`] is called.

// MAYBE Abstract resources live on concrete devices.  If two devices need distinct Vulkan
// resources for the same abstract resource, when the abstraction is being destroyed, it would be
// appropriate to give the handles back to a device-specific deletion queue.  What we will do for
// now is use the device directly, but as a runtime emerges, the abstraction of concrete Vulkan
// objects will be given to the runtime so the user doesn't need to know if multiple concrete
// resources needed to exist at some point.
// MAYBE Some types require extra function pointers or other handles for deletion.  These dependent
// resources are likely better served by an ownership model.  See [`PoolRing`] etc.
// NEXT Mass deletion will prefer privatization, owning a local FIFO for unsynchronized additions.
// The ergonomics and synchronization overhead would be better.
// NEXT The common case will be bursts of deletion on technique transitions, so a common pool of
// slack and several normally small epochs that can grow into the shares slack will behave near
// optimal for most workloads.

use std::sync::Mutex;

use crate::internal::*;

pub mod core {
    pub use super::DeletionQueue;
}

// Create the type mapping from ash handles to our `HandleType` + `DeadHandle` decomposition.
pub unsafe trait RawHandle: Copy {
    const HANDLE_TYPE: HandleType;
    fn as_raw(self) -> u64;
}

macro_rules! raw_handle {
    ($($vk:ty => $tag:ident),* $(,)?) => {$(
        unsafe impl RawHandle for $vk {
            const HANDLE_TYPE: HandleType = HandleType::$tag;
            fn as_raw(self) -> u64 { vk::Handle::as_raw(self) }
        }
    )*};
}

raw_handle! {
    vk::Buffer         => Buffer,
    vk::BufferView     => BufferView,
    vk::DeviceMemory   => DeviceMemory,
    vk::Fence          => Fence,
    vk::Image          => Image,
    vk::ImageView      => ImageView,
    vk::Pipeline       => Pipeline,
    vk::PipelineCache  => PipelineCache,
    vk::PipelineLayout => PipelineLayout,
    vk::Semaphore      => Semaphore,
    vk::ShaderModule   => ShaderModule,
}

/// A type-erased raw ash handle.
struct DeadHandle(u64);

/// A type tag for a `DeadHandle`.  Types are compacted and do not correspond to the Vulkan spec
/// stucture types.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum HandleType {
    Buffer,
    BufferView,
    DeviceMemory,
    Fence,
    Image,
    ImageView,
    Pipeline,
    PipelineCache,
    PipelineLayout,
    Semaphore,
    ShaderModule,
}

impl HandleType {
    fn destroy(self, device: &ash::Device, handle: DeadHandle) {
        let raw = handle.0;
        unsafe {
            match self {
                Self::Buffer => device.destroy_buffer(vk::Buffer::from_raw(raw), None),
                Self::BufferView => device.destroy_buffer_view(vk::BufferView::from_raw(raw), None),
                Self::DeviceMemory => device.free_memory(vk::DeviceMemory::from_raw(raw), None),
                Self::Fence => device.destroy_fence(vk::Fence::from_raw(raw), None),
                Self::Image => device.destroy_image(vk::Image::from_raw(raw), None),
                Self::ImageView => device.destroy_image_view(vk::ImageView::from_raw(raw), None),
                Self::Pipeline => device.destroy_pipeline(vk::Pipeline::from_raw(raw), None),
                Self::PipelineCache => {
                    device.destroy_pipeline_cache(vk::PipelineCache::from_raw(raw), None)
                }
                Self::PipelineLayout => {
                    device.destroy_pipeline_layout(vk::PipelineLayout::from_raw(raw), None)
                }
                Self::Semaphore => device.destroy_semaphore(vk::Semaphore::from_raw(raw), None),
                Self::ShaderModule => {
                    device.destroy_shader_module(vk::ShaderModule::from_raw(raw), None)
                }
            }
        }
    }
}

#[derive(Default)]
struct Bag {
    handles: Vec<DeadHandle>,
    types: Vec<HandleType>,
}

impl Bag {
    fn push(&mut self, ty: HandleType, handle: DeadHandle) {
        self.handles.push(handle);
        self.types.push(ty);
    }

    fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    fn drain(&mut self, device: &ash::Device) {
        for (ty, handle) in self.types.drain(..).zip(self.handles.drain(..)) {
            ty.destroy(device, handle);
        }
    }
}

pub struct DeletionQueue<const EPOCHS: usize = 4> {
    device: ash::Device,
    inner: Mutex<Ring<EPOCHS>>,
}

struct Ring<const EPOCHS: usize> {
    epochs: [Bag; EPOCHS],
    cursor: usize,
}

impl<const EPOCHS: usize> DeletionQueue<EPOCHS> {
    pub fn new(device: &ash::Device) -> Self {
        const { assert!(EPOCHS >= 2, "need at least one epoch of deferral") }
        Self {
            device: device.clone(),
            inner: Mutex::new(Ring {
                epochs: std::array::from_fn(|_| Bag::default()),
                cursor: 0,
            }),
        }
    }

    /// Add a handle to the deletion queue.
    // MAYBE make the raw handle methods unsafe and give the safe interface to owned, wrapped
    // handles.
    pub fn push<H: RawHandle>(&self, handle: H) {
        let mut ring = self.inner.lock().unwrap();
        let cursor = ring.cursor;
        ring.epochs[cursor].push(H::HANDLE_TYPE, DeadHandle(handle.as_raw()));
    }

    /// Advance the epoch, destroying everything in the bag we rotate onto.
    pub fn tick(&self) {
        let mut bag = {
            let mut ring = self.inner.lock().unwrap();
            ring.cursor = (ring.cursor + 1) % EPOCHS;
            let cursor = ring.cursor;
            std::mem::take(&mut ring.epochs[cursor])
        };
        bag.drain(&self.device);
    }

    /// Delete everything in all epochs.  Call this during final destruction, after in-flights are
    /// guaranteed retired by semaphore or [`Device::wait_idle`].
    pub unsafe fn drain(&self) {
        let mut ring = self.inner.lock().unwrap();
        let start = ring.cursor;

        for i in 1..=EPOCHS {
            let mut bag = std::mem::take(&mut ring.epochs[(start + i) % EPOCHS]);
            bag.drain(&self.device);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    pub fn delete_some_semaphores() {
        with_context!(|device, instance| {
            // XXX find some ways to test... Need API calls that don't explode so we can test
            // ticking and rotation
            let semaphore = device.make_timeline_semaphore().unwrap();
            device.deletion_queue.push(semaphore.into_raw());
            for i in 0..4 {
                device.deletion_queue.tick();
            }
            unsafe { device.deletion_queue.drain() };
        })
    }
}
