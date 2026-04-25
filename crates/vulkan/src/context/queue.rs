// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Queue
//!
//! Surface users need a queue that can present.  ML workloads want low-priority compute queues.
//! DMA transfers use a dedicated Transfer queue.  A single queue is not thread-safe for submission
//! and has implicit ordering guarantees that could affect pipelining.  Different queue families are
//! much easier for the driver and hardware to schedule independently.  These are some reasons users
//! may want different queues and queue families.
//!
//! However, not all devices support all queue configurations.  To be compliant, the spec only
//! requires a single graphics queue.  The abstraction in this module maps the semantics users might
//! intend to best-effort choices of queues handles to serve a reasonable set of use cases:
//!
//! - High priority graphics with presentation capability for a surface
//! - Lower priority compute & off-screen rendering for synthetic training
//! - Dedicated transfer (see DEBT, plans to abstract with UMA patterns)
//!
//! ## Priorities
//!
//! Behavior supporting queue priorities is mostly up to drivers, but to the extent that it is
//! supported, we can give the driver hints.  Incidentally, we are exposing queues appropriate for
//! various implementation objectives and providing the driver opportunities to pipeline better.
//!
//! Priority deliberately does **not** touch the type system.  Queues with any priority are safe to
//! use with workloads of any other priority.  This is a semantic indication from the user, not a
//! contractual obligation for downstream consuming functions.
//!
//! ## Capabilities
//!
//! Queue capabilities are total ordered: `Graphics` > `Compute` > `Transfer`.  Queues with
//! insufficient capabilities cannot be used for some operations, so queues **are** typed on their
//! capabilities, enforcing the restrictions for downstream callers.
//!
//! Because of the clean ordering, anything that supports more capability can be used in calls that
//! need less capabilities via [`Deref`](`std::ops::Deref`).  This makes overloading less painful in
//! cases we don't actually get the queues we wanted.
//!
//! ## Synchronization
//!
//! Raw queue handles are not thread safe.  On devices that don't provide enough queues, we may have
//! to overload queues, and so the user can't sure if a queue has an exclusive owner or not.  To
//! compensate, we synchronize submissions when using the [`dispatch`] interface.  **If using raw
//! handles, you assume that obligation.**
//!
//! ## Queue Resolution
//!
//! All queues must be created at logical device creation.  This means we must decide which queues
//! we *might* need up front.  While inspecting the logical device, we first attempt to satisfy all
//! use cases within our semantics.  If the queues present can't cover all semantics, we begin
//! substituting and possibly overloading some choices while attempting to preserve user semantics.
//!
//! ### Presentation Support
//!
//! Presentation support for a window surface is checked prior to logical device creation.  Windows
//! and their surfaces come and go.  To ensure that we initialize queues in all families that we
//! might ever need for presentation on the logical device, we ensure at least one queue is enabled
//! in all graphics families.
//!
//! When creating new surfaces, such as if windows are moving around, if a new queue family is
//! required, the same logical device can be used.  If no family supports the new surface, the
//! correct workflow is to back up to scanning physical device support for a surface using the
//! [`VkContext`].

// NEXT optional support for other queue families, expand options to include rare queue types like
// video decode and sparse binding.
// MAYBE move towards a user-intent API where intents are keys for pulling the concrete queues back
// out after logical device creation?  Physical device inspection also will likely need to grow and
// integrate with queue intents.
// XXX Users will likely be confused as to whether queues have exclusive ownership.  This can only
// be cleared up with with an intent based API or reservation system.
// NEXT see VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR for Vulkan 1.4.
// NEXT see VK_NV_compute_occupancy_priority, Nvidia specific priority info.
// XXX If we really need to use two different graphics queue families, most assets will already be
// exclusive to one family.  We can either pessimistically mark them for use in all graphics queues
// (using pQueueFamilyIndices, not CONCURRENT) or resign ourselves to re-creating them and swapping
// them in.  The mutable assets will mostly be exclusive to a surface and render loop per window.
// Audio input can either be duplicated or shared.  A lot of hard problems await that will not get
// simpler until the resource runtime infrastructure exists.

use std::collections::HashMap;
use std::marker::PhantomData;

use ash::vk;
use smallvec::SmallVec;

use crate::internal::*;

pub mod prelude {
    pub use super::Queue;
    pub use super::QueuePriority;

    pub use super::Compute;
    pub use super::Graphics;
    pub use super::Transfer;
}

mod sealed {
    pub trait Capability {
        const CAPABILITY: super::QueueCapability;
    }
}

/// Can do everything, but using others for more focused work may do a better job at utilization
/// and stable operation.
#[derive(Clone, Copy)]
pub enum Graphics {}
/// Simple pipelines, but if you need presentation, you probably need `Graphics` instead.
#[derive(Clone, Copy)]
pub enum Compute {}
/// Always low priority, backed by dedicated hardware on DMA devices.  See DEBT regarding UMA
/// and DMA.
#[derive(Clone, Copy)]
pub enum Transfer {}

impl sealed::Capability for Graphics {
    const CAPABILITY: QueueCapability = QueueCapability::Graphics;
}
impl sealed::Capability for Compute {
    const CAPABILITY: QueueCapability = QueueCapability::Compute;
}
impl sealed::Capability for Transfer {
    const CAPABILITY: QueueCapability = QueueCapability::Transfer;
}

/// Public re-export of the sealed trait.  Downstream can name and bound on it,
/// but cannot implement it.
pub use sealed::Capability;

/// Runtime mirror of the [`Capability`] type parameter.  Use this where you need
/// to inspect or match on a queue's capability level at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueueCapability {
    Transfer,
    Compute,
    Graphics,
}

/// Hints to the driver about which submissions should receive priority scheduling.  Not widely or
/// deeply supported.  Manually pacing your work with exclusive phases for deadline-sensitive work
/// should be preferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuePriority {
    /// Use for opportunistic work that is free to be preempted or deferred.  Transfer, off-screen
    /// machine learning, or rendering for synthetic training are examples of this kind of work.
    Low,
    /// Submissions with deadlines.
    High,
}

impl QueuePriority {
    pub fn as_f32(self) -> f32 {
        // NEXT User configurations & a test case to demonstrate a difference.  (Expect no
        // difference.  Sources are fairly consistent that drivers are only starting to implement
        // some priority, likely coincident to work on supporting global priority).
        match self {
            QueuePriority::High => 1.0,
            QueuePriority::Low => 0.1,
        }
    }
}

/// Whether a slot was satisfied exactly or overloaded onto a higher-capability queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueMatch {
    /// The queue family and capability exactly matched the semantic request.
    Exact,
    /// A queue with **sufficient** capabilities was made available, but does not exactly meet the
    /// semantics and **may undermine the goal of using the queue** (eg using a spare graphics queue
    /// as the transfer queue when no DMA hardware exists and a transfer queue serves less real
    /// purpose).
    ///
    /// The boolean for aliased means the queue is both a substitute and overloaded.  A queue
    /// already in use for a different semantic role had to be used.
    ///
    /// For all substitutes, either the priority or capabilities are higher than
    /// requested.  Only re-used queues are aliased.
    Substitute { aliased: bool },
}

/// The queue provided will always have the capabilities requested.  However, returned queues can
/// also be inspected to determine whether the returned queue exactly matches the semantics or is an
/// overloaded queue of another variety.  This can enable implementation paths to degrade if using a
/// single queue would undermine the implementation concept, such as a dedicated DMA transfer queue
/// vs a compute queue that can implicitly do transfers.

/// Queues
pub struct Queues {
    physical_device: vk::PhysicalDevice,

    high_graphics: SmallVec<Queue<Graphics>, 8>,
    low_graphics: SmallVec<Queue<Graphics>, 8>,

    high_compute: Queue<Compute>,
    low_compute: Queue<Compute>,
    transfer: Queue<Transfer>,
}

impl Queues {
    pub fn new(device: &ash::Device, plan: QueuePlan) -> Self {
        Queues {
            physical_device: plan.physical_device,

            high_graphics: plan
                .high_graphics
                .iter()
                .map(|&slot| Queue::new(device, slot))
                .collect(),
            low_graphics: plan
                .low_graphics
                .iter()
                .map(|&slot| Queue::new(device, slot))
                .collect(),

            high_compute: Queue::new(device, plan.high_compute),
            low_compute: Queue::new(device, plan.low_compute),
            transfer: Queue::new(device, plan.transfer),
        }
    }

    /// ## Compute Queues
    ///
    /// High-priority compute uses a dedicated compute family if available.  Uses low-priority
    /// graphics otherwise.  Low-priority compute will use the low-priority graphics if there is no
    /// alternative.
    pub fn compute(&self, priority: QueuePriority) -> Queue<Compute> {
        match priority {
            QueuePriority::High => self.high_compute,
            QueuePriority::Low => self.low_compute,
        }
    }

    /// ## Transfer
    ///
    /// Transfer queues have no high-priority semantics, although they will use an overloaded
    /// high-priority queue in the worst case.  If at all possible, it will use a low-priority
    /// queue.
    pub fn transfer(&self) -> Queue<Transfer> {
        self.transfer
    }

    /// ## Graphics With Presentation Support
    ///
    /// Finds the graphics queue of the requested priority that can present to `surface` on this
    /// physical device.
    ///
    /// Usually this will be identical to the regular high-priority graphics queue, but when cards
    /// use different queue for physical routes to displays, surfaces on different physical displays
    /// may require use of queues from different queue families.  This function resolves those
    /// queues for `surface`.
    ///
    /// If `None` is returned, the physical device for these queues cannot present to your surface.
    /// You need to scan for present-capable devices again using the [`VkContext`] and create a new
    /// logical device [`DeviceContext`], starting from scratch basically.
    ///
    /// Returns `None` only if no graphics family on this physical device can present to the surface
    /// at all, which means you need a different physical device or logical device.
    pub fn graphics(
        &self,
        vk_context: &VkContext,
        surface: &VkSurface,
        priority: QueuePriority,
    ) -> Option<Queue<Graphics>> {
        let surface_loader = vk_context.surface_loader();
        let surface = surface.as_raw();
        let candidates = match priority {
            QueuePriority::High => &self.high_graphics,
            QueuePriority::Low => &self.low_graphics,
        };
        candidates
            .iter()
            .find(|q| unsafe {
                surface_loader
                    .get_physical_device_surface_support(self.physical_device, q.family(), surface)
                    .unwrap_or(false)
            })
            .copied()
    }

    /// ## Off-screen Graphics
    ///
    /// Use **only for off-screen rendering**.  For presentation support, use [`graphics`] instead.
    /// The low priority queue returned will be in the same queue family as the high priority queue
    /// if at all possible (it's not guaranteed to always be possible).
    pub fn graphics_offscreen(&self, priority: QueuePriority) -> Queue<Graphics> {
        match priority {
            // A graphics queue is guaranteed to exist, and we make a high priority graphics for all
            // possible "present" queue families.  If not choosing for a surface, just use the first
            // one, and the unwrap will never fail unless the device doesn't match spec.
            QueuePriority::High => self.high_graphics[0],
            // Our overloading strategy guarantees that there is at least one low-priority graphics,
            // but it may be an overload of another `Queue`
            QueuePriority::Low => self.low_graphics[0],
        }
    }

    // NOTE queues are owned by the device.  Just drop when done.
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Queue<C: Capability> {
    raw: vk::Queue,
    family: u32,
    priority: QueuePriority,
    /// When overloaded, queues might have more capabilities than required for a given request from
    /// the user.
    actual_flags: vk::QueueFlags,
    queue_match: QueueMatch,
    _marker: PhantomData<C>,
}

impl<C: Capability> Queue<C> {
    fn new(device: &ash::Device, slot: SlotAssignment) -> Self {
        let raw = unsafe { device.get_device_queue(slot.family, slot.index) };
        Self {
            raw,
            family: slot.family,
            priority: slot.priority,
            actual_flags: slot.actual_flags,
            queue_match: slot.queue_match,
            _marker: PhantomData,
        }
    }

    /// Useful for handling new surfaces by checking if the as-is logical device already has enough
    /// queue support.
    pub fn family(&self) -> u32 {
        self.family
    }

    /// The priority actually submitted to the driver for this queue.  This may be higher than the
    /// semantic role requested if the underlying slot was shared with a higher-priority claim.
    pub fn priority(&self) -> QueuePriority {
        self.priority
    }

    /// Actual capabilities for this queue, which may exceed the requested capabilities if queue was overloaded.
    pub fn capabilities(&self) -> vk::QueueFlags {
        self.actual_flags
    }

    /// Returns `true` if this slot shares its underlying `VkQueue` with another semantic role.
    /// An aliased queue is not overloaded (it has the right capability), but it is not exclusive.
    pub fn is_aliased(&self) -> bool {
        self.queue_match == QueueMatch::Substitute { aliased: true }
    }

    /// Returns `true` if this slot has its own exclusive `VkQueue` of exactly the right
    /// capability.
    pub fn is_exact(&self) -> bool {
        self.queue_match == QueueMatch::Exact
    }

    /// Returns the raw [`ash::vk::Queue`] handle for doing command buffer work manually.  You know
    /// what you're doing or at least you are responsible.
    pub unsafe fn raw(&self) -> vk::Queue {
        self.raw
    }

    // NOTE destroy not implemented since logical devices own their queues and we can just drop handles.
}

impl std::ops::Deref for Queue<Graphics> {
    type Target = Queue<Compute>;
    fn deref(&self) -> &Queue<Compute> {
        // SAFETY: all Queue<C> differ only on phantom data
        unsafe { &*(self as *const Queue<Graphics> as *const Queue<Compute>) }
    }
}

impl std::ops::Deref for Queue<Compute> {
    type Target = Queue<Transfer>;
    fn deref(&self) -> &Queue<Transfer> {
        // SAFETY: all Queue<C> differ only on phantom data
        unsafe { &*(self as *const Queue<Compute> as *const Queue<Transfer>) }
    }
}

/// Essentially a queue spec.  After the logical device exists, this information describes the queue
/// that was actually provided from the driver.
///
/// A resolved assignment for one semantic queue slot.  Produced during physical device inspection
/// and consumed both by [`QueuePlan::queue_cis`] and [`Queues::new`].
///
/// The `priority` field reflects the priority actually submitted to the driver, which may be higher
/// than the semantic role requested if the underlying slot was shared with a higher-priority
/// assignment.
#[derive(Clone, Copy, Debug)]
pub struct SlotAssignment {
    pub family: u32,
    pub index: u32,
    pub actual_flags: vk::QueueFlags,
    pub priority: QueuePriority,
    pub queue_match: QueueMatch,
}

impl SlotAssignment {
    fn from_raw(
        (family, index, actual_flags, queue_match): RawSlot,
        priority: QueuePriority,
    ) -> Self {
        Self {
            family,
            index,
            actual_flags,
            queue_match,
            priority,
        }
    }

    fn alias(source: &Self) -> Self {
        Self {
            queue_match: QueueMatch::Substitute { aliased: true },
            ..*source
        }
    }
}

type RawSlot = (u32, u32, vk::QueueFlags, QueueMatch);

fn raw_exact(family: u32, index: u32, flags: vk::QueueFlags) -> RawSlot {
    (family, index, flags, QueueMatch::Exact)
}

fn raw_spare(family: u32, index: u32, flags: vk::QueueFlags) -> RawSlot {
    (
        family,
        index,
        flags,
        QueueMatch::Substitute { aliased: false },
    )
}

/// Everything we need to know about one queue family during plan construction.
#[derive(Debug)]
struct FamilyInfo {
    index: u32,
    flags: vk::QueueFlags,
    count: u32, // number of queues the family exposes
}

/// Collected view of the physical device's queue families, bucketed by minimum capability.
/// A family appears in exactly one bucket — the most specific one it qualifies for.
#[derive(Debug, Default)]
struct FamilyCandidates {
    /// Families whose flags include Graphics, sorted in order of the most queues.
    graphics: Vec<FamilyInfo>,
    /// Families that support Compute but not Graphics (dedicated compute).
    compute: Vec<FamilyInfo>,
    /// Families that support Transfer but not Compute or Graphics (dedicated DMA).
    transfer: Vec<FamilyInfo>,
}

impl FamilyCandidates {
    fn collect(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        let props =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut out = FamilyCandidates::default();

        for (i, p) in props.iter().enumerate() {
            let i = i as u32;
            let info = FamilyInfo {
                index: i,
                flags: p.queue_flags,
                count: p.queue_count,
            };

            if p.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                out.graphics.push(info);
            } else if p.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                out.compute.push(info);
            } else if p.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                out.transfer.push(info);
            }
        }
        // Sort graphics families so that family with the most queues comes first.
        out.graphics.sort_by_key(|f| std::cmp::Reverse(f.count));
        out
    }
}

/// During logical device creation, we first collect our queue decisions into a QueuePlan
/// structure which can produce the queue creation info.  After the device is created, we forward
/// our choices to instantiate [`Queues`], which needss to know what creation info we used in order
/// to collect the resulting queues off of the logical device.
///
/// Call [`QueuePlan::queue_cis`] to get the `DeviceQueueCreateInfo` slice for logical device
/// creation, then forward the `QueuePlan` to [`Queues::new`] after the device is live.
///
/// ## Queue Claiming Preference Order
///
/// The order of preference for discovering queues prior to overloading:
///
/// 1. The high-priority graphics queue for all graphics-supporting families
/// 2. Low-priority graphics queues for all graphics-supporting families
/// 3. High-priority compute queue
/// 4. Low-priority compute queue
/// 5. Dedicated transfer queue if present
///
/// To respect priority semantics, we need at least two graphics queues in one family.  We overload
/// from lower capability to higher capability first, so all queues may alias to graphics.  Second,
/// we overload from low priority to high, so all low priority queue requests might wind up as low
/// priority graphics, but priority will be respected if at all possible.  In the degenerate case,
/// all queues are just one high priority graphics queue.
pub struct QueuePlan {
    physical_device: vk::PhysicalDevice,

    // One high-priority and one low-priority slot per graphics-capable family,
    // in the same order as FamilyCandidates::graphics (present-capable families first).
    // Every graphics family gets both a high and a low entry; if the family only
    // exposes a single queue the low entry aliases the high one.
    high_graphics: SmallVec<SlotAssignment, 8>,
    low_graphics: SmallVec<SlotAssignment, 8>,

    high_compute: SlotAssignment,
    low_compute: SlotAssignment,
    transfer: SlotAssignment,

    /// Per-family priority indexed by queue index within that family.  Built incrementally during
    /// construction.
    priorities: HashMap<u32, Vec<f32>>,
}

impl QueuePlan {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, VulkanError> {
        let candidates = FamilyCandidates::collect(instance, physical_device);
        if candidates.graphics.is_empty() {
            return Err(VulkanError::DriverError(
                "physical device exposes no graphics-capable queue family".into(),
            ));
        }

        // Use queue 0 for graphics_high for each family, then queue 1 if available or else overload
        // onto 0 for graphics_low for that family.
        let mut high_graphics: SmallVec<SlotAssignment, 8> = SmallVec::new();
        let mut low_graphics: SmallVec<SlotAssignment, 8> = SmallVec::new();
        let mut spare_gfx: SmallVec<RawSlot, 32> = SmallVec::new();

        for fi in &candidates.graphics {
            let high =
                SlotAssignment::from_raw(raw_exact(fi.index, 0, fi.flags), QueuePriority::High);
            let low = if fi.count >= 2 {
                SlotAssignment::from_raw(raw_exact(fi.index, 1, fi.flags), QueuePriority::Low)
            } else {
                SlotAssignment::alias(&high)
            };

            high_graphics.push(high);
            low_graphics.push(low);
            for idx in 2..fi.count {
                spare_gfx.push(raw_spare(fi.index, idx, fi.flags));
            }
        }

        let mut spare_gfx = spare_gfx.into_iter();

        // use dedicated compute or fall back to graphics spares or overload
        let mut spare_compute: SmallVec<RawSlot, 32> = SmallVec::new();
        let high_compute = candidates
            .compute
            .first()
            .map(|fi| {
                for cfi in &candidates.compute {
                    let start = if cfi.index == fi.index { 1 } else { 0 };
                    for idx in start..cfi.count {
                        spare_compute.push(raw_spare(cfi.index, idx, cfi.flags));
                    }
                }
                raw_exact(fi.index, 0, fi.flags)
            })
            .or_else(|| spare_gfx.next())
            .map(|raw| SlotAssignment::from_raw(raw, QueuePriority::High))
            .unwrap_or_else(|| SlotAssignment::alias(&high_graphics[0]));

        // use spare compute or fall back to spare graphics or overload to graphics low (leaving
        // compute high free to overload onto graphics high without pulling along compute low)
        let mut spare_compute = spare_compute.into_iter();
        let low_compute = spare_compute
            .next()
            .or_else(|| spare_gfx.next())
            .map(|raw| SlotAssignment::from_raw(raw, QueuePriority::Low))
            .unwrap_or_else(|| SlotAssignment::alias(&low_graphics[0]));

        // use dedicated transfer, spare compute, spare graphics, or overloaded compute low.
        let transfer = candidates
            .transfer
            .first()
            .map(|fi| raw_exact(fi.index, 0, fi.flags))
            .or_else(|| spare_compute.next())
            .or_else(|| spare_gfx.next())
            .map(|raw| SlotAssignment::from_raw(raw, QueuePriority::Low))
            .unwrap_or_else(|| SlotAssignment::alias(&low_compute));

        // Build priority float table for queue_cis
        let priorities =
            build_priority_floats(high_graphics.iter().chain(low_graphics.iter()).chain([
                &high_compute,
                &low_compute,
                &transfer,
            ]));

        Ok(QueuePlan {
            physical_device,
            high_graphics,
            low_graphics,
            high_compute,
            low_compute,
            transfer,
            priorities,
        })
    }

    /// Provide queue creation info for logical device creation, borrowing data computed during
    /// planning.
    pub fn queue_cis(&self) -> Vec<vk::DeviceQueueCreateInfo<'_>> {
        self.priorities
            .iter()
            .map(|(&family, floats)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(floats.as_slice())
            })
            .collect()
    }
}

fn build_priority_floats<'a>(
    assignments: impl Iterator<Item = &'a SlotAssignment>,
) -> HashMap<u32, Vec<f32>> {
    let mut by_family: HashMap<u32, Vec<QueuePriority>> = HashMap::new();
    for s in assignments {
        let slots = by_family.entry(s.family).or_default();
        let needed = s.index as usize + 1;
        if slots.len() < needed {
            slots.resize(needed, QueuePriority::Low);
        }
        let slot = &mut slots[s.index as usize];
        if s.priority > *slot {
            *slot = s.priority;
        }
    }
    by_family
        .into_iter()
        .map(|(family, prios)| (family, prios.iter().map(|p| p.as_f32()).collect()))
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::with_context;

    #[test]
    fn graphics_high_always_exact() {
        with_context!(|device_context, vk_context| {
            // obtaining device context implicitly created queues
            let gfx = device_context
                .queues
                .graphics_offscreen(QueuePriority::High);
            assert_eq!(gfx.priority, QueuePriority::High);
            assert!(gfx.actual_flags.contains(vk::QueueFlags::GRAPHICS));
            assert!(gfx.capabilities().contains(vk::QueueFlags::GRAPHICS));
            assert_eq!(gfx.is_exact(), true);
        });
    }

    // NEXT create an off-screen surface for surface integration testing.

    #[test]
    fn transfer_is_always_low_priority() {
        with_context!(|device_context, _vk_context| {
            let compute_low = device_context.queues.compute(QueuePriority::Low);
            let transfer = device_context.queues.transfer();
            // If compute had to be overloaded onto graphics high (degenerate case) then this test
            // has no meaning on the test runner.
            if compute_low.priority == QueuePriority::Low {
                assert_eq!(
                    transfer.priority(),
                    QueuePriority::Low,
                    "transfer queue must always carry Low priority (got {:?})",
                    transfer.priority()
                );
            }
        });
    }

    #[test]
    fn compute_low_is_always_low_priority() {
        with_context!(|device_context, _vk_context| {
            let gfx_low = device_context.queues.graphics_offscreen(QueuePriority::Low);
            let compute_low = device_context.queues.compute(QueuePriority::Low);
            // If graphics had to be overloaded onto graphics high (degenerate case) then this test
            // has no meaning on the test runner.
            if gfx_low.priority == QueuePriority::Low {
                assert_eq!(
                    compute_low.priority(),
                    QueuePriority::Low,
                    "low compute queue must always carry Low priority (got {:?})",
                    compute_low.priority()
                );
            }
        });
    }

    #[test]
    fn dedicated_transfer_when_present() {
        with_context!(|device_context, vk_context| {
            let physical = device_context.physical_device;
            let instance = &vk_context.instance;

            let family_props =
                unsafe { instance.get_physical_device_queue_family_properties(physical) };

            // A dedicated transfer family is one that can transfer but cannot
            // do graphics or compute.  The spec guarantees any graphics-capable
            // family implicitly supports transfer, so a family with only the
            // TRANSFER bit is unambiguously dedicated DMA hardware.
            let dedicated: Option<u32> = family_props
                .iter()
                .enumerate()
                .find(|(_, p)| {
                    let f = p.queue_flags;
                    f.contains(vk::QueueFlags::TRANSFER)
                        && !f.contains(vk::QueueFlags::GRAPHICS)
                        && !f.contains(vk::QueueFlags::COMPUTE)
                })
                .map(|(i, _)| i as u32);

            println!("dedicated transfer queue: {:?}", dedicated);

            // Now ask the high-level API — no plan inspection, no internal fields.
            let reported_family = device_context.queues.transfer().family();

            match dedicated {
                Some(xfer_family) => {
                    assert_eq!(
                        reported_family, xfer_family,
                        "physical device has a dedicated transfer family ({xfer_family}) \
                     but Queues::transfer() reported family {reported_family}"
                    );
                }
                None => {
                    // No dedicated family exists.  The queue must have landed on
                    // something graphics- or compute-capable — just confirm it is
                    // at least a valid family index rather than garbage.
                    assert!(
                        (reported_family as usize) < family_props.len(),
                        "Queues::transfer() returned out-of-range family {reported_family}"
                    );
                }
            }
        })
    }
}
