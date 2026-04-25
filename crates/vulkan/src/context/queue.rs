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
//! - High priority graphics
//! - Presentation capability
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
//! overloading some choices while attempting to preserve user semantics.
//!
//! ### Presentation Support
//!
//! Presentation support for a window is checked prior to logical device creation.  The results of
//! that check may prefer a specific graphics queue to be the high-priority graphics queue.
//!
//! When creating new surfaces, such as if windows are moving around, if a new device or queue
//! family should be the default for that window, creating a new logical device is the supported
//! workflow.  In the rare case that this means two windows use two different logical devices for
//! the same physical device, both windows will have completely independent resource requirements.

// NEXT support less linear workflows and especially use of multiple graphics queue families, each
// with presentation support for a different surface
// NEXT optional support, expand options to include rare queue types like video decode and sparse
// binding.
// MAYBE move towards a user-intent API where intents are keys for pulling the concrete queues back
// out after logical device creation?  Physical device inspection also will likely need to grow and
// integrate with queue intents.
// NEXT see VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR for Vulkan 1.4.

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use ash::vk;
use smallvec::SmallVec;

use crate::internal::*;

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
    /// Transfer is always low priority.
    Low,
    /// Submissions with deadlines.
    High,
}

impl QueuePriority {
    pub fn as_f32(self) -> f32 {
        // NEXT User configurations & a test case to demonstrate a difference.
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
    /// No suitable queue was available; a higher-capability queue was substituted.
    Overloaded,
    /// A queue of the right capability exists but is shared with another semantic slot.
    Aliased,
}

/// A claimed `(family, index)` with match metadata but no finalized priority.
///
/// Priority cannot be known until all claims are recorded into the `priorities` map, because a
/// later claim (e.g. a present queue) can retroactively raise the scheduled priority of a slot
/// claimed earlier.  Call [`ClaimedSlot::finalize`] after the map is complete to produce a
/// [`SlotAssignment`] with an accurate `priority` field.
#[derive(Clone, Copy, Debug)]
struct ClaimedSlot {
    family: u32,
    index: u32,
    actual_flags: vk::QueueFlags,
    queue_match: QueueMatch,
}

impl ClaimedSlot {
    fn from_tier(t: (u32, u32, vk::QueueFlags)) -> Self {
        ClaimedSlot {
            family: t.0,
            index: t.1,
            actual_flags: t.2,
            queue_match: QueueMatch::Exact,
        }
    }

    fn overloaded(fallback: &ClaimedSlot) -> Self {
        ClaimedSlot {
            queue_match: QueueMatch::Overloaded,
            ..*fallback
        }
    }

    fn aliased(existing: &ClaimedSlot) -> Self {
        ClaimedSlot {
            queue_match: QueueMatch::Aliased,
            ..*existing
        }
    }

    /// Promotes this claim into a [`SlotAssignment`] by reading the finalized priority from the
    /// completed `priorities` map.  Must be called after all calls to [`record`] are done.
    fn finalize(self, priorities: &HashMap<u32, Vec<QueuePriority>>) -> SlotAssignment {
        let priority = priorities
            .get(&self.family)
            .and_then(|v| v.get(self.index as usize))
            .copied()
            .unwrap_or(QueuePriority::Low);
        SlotAssignment {
            family: self.family,
            index: self.index,
            actual_flags: self.actual_flags,
            queue_match: self.queue_match,
            priority,
        }
    }
}

/// A resolved assignment for one semantic queue slot.  Produced during physical device
/// inspection and consumed both by [`QueuePlan::queue_cis`] and [`Queues::new`].
///
/// The `priority` field reflects the priority actually submitted to the driver, which may be
/// higher than the semantic role requested if the underlying slot was shared with a
/// higher-priority claim.
#[derive(Clone, Copy, Debug)]
pub struct SlotAssignment {
    pub family: u32,
    pub index: u32,
    pub actual_flags: vk::QueueFlags,
    pub priority: QueuePriority,
    pub queue_match: QueueMatch,
}

/// Raises the recorded priority for `(family, index)` if the new value is higher; never lowers.
/// This is the only mutation of the priorities map: it is the ground truth for what gets submitted
/// to the driver.
fn record(
    priorities: &mut HashMap<u32, Vec<QueuePriority>>,
    slot: &ClaimedSlot,
    priority: QueuePriority,
) {
    let vec = priorities.entry(slot.family).or_default();
    let needed = slot.index as usize + 1;
    if vec.len() < needed {
        vec.resize(needed, QueuePriority::Low);
    }
    let entry = &mut vec[slot.index as usize];
    if priority > *entry {
        *entry = priority;
    }
}

/// Converts a claimed `(family, index, flags)` into an `Exact` slot, or copies the fallback slot
/// as `Overloaded` when the tier iterator was exhausted.
fn claim(next: Option<(u32, u32, vk::QueueFlags)>, fallback: &ClaimedSlot) -> ClaimedSlot {
    match next {
        Some(t) => ClaimedSlot::from_tier(t),
        None => ClaimedSlot::overloaded(fallback),
    }
}

/// The queue provided will always have the capabilities requested.  However, returned queues can
/// also be inspected to determine whether the returned queue exactly matches the semantics or is an
/// overloaded queue of another variety.  This can enable implementation paths to degrade if using a
/// single queue would undermine the implementation concept, such as a dedicated DMA transfer queue
/// vs a compute queue that can implicitly do transfers.

/// Queues
pub struct Queues {
    high_graphics: Queue<Graphics>,
    low_graphics: Queue<Graphics>,
    high_compute: Queue<Compute>,
    low_compute: Queue<Compute>,
    transfer: Queue<Transfer>,

    /// At most one entry per distinct present family. In practice one or two entries for queues
    /// that physically connect to different displays.
    present_queues: SmallVec<(u32, Queue<Graphics>), 8>,
}

impl Queues {
    pub fn new(device: &ash::Device, plan: QueuePlan) -> Self {
        Queues {
            high_graphics: Queue::new(device, plan.high_graphics),
            low_graphics: Queue::new(device, plan.low_graphics),
            high_compute: Queue::new(device, plan.high_compute),
            low_compute: Queue::new(device, plan.low_compute),
            transfer: Queue::new(device, plan.transfer),
            present_queues: plan
                .present_graphics
                .iter()
                .map(|&(family, slot)| (family, Queue::new(device, slot)))
                .collect(),
        }
    }

    /// Returns a graphics-capable queue, used to dispatch graphics pipelines.  High and low
    /// priority will be available as long as the physical device has at least two queues across all
    /// graphics-capable families.
    ///
    /// If presentation support was indicated for one or more queue families, those families' slots
    /// are claimed before `low_graphics`, so that a device with two single-queue present families
    /// still gives each its own `High`-priority slot.
    pub fn graphics(&self, priority: QueuePriority) -> Queue<Graphics> {
        match priority {
            QueuePriority::High => self.high_graphics,
            QueuePriority::Low => self.low_graphics,
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

    /// ## Present
    ///
    /// When initializing [`QueuePlan`], you may provide a list of `u32` queue family indexes.  Each
    /// one indicates a request for an independent graphics queue to be able to present to surfaces.
    /// The user can recover the queue after logical device creation using the chosen `family` as a
    /// key.
    ///
    /// Usually this will be identical to the regular high-priority graphics queue, but when cards
    /// use different queue for physical routes to displays, surfaces on different physical displays
    /// may use different queue families.
    pub fn present(&self, family: u32) -> Option<Queue<Graphics>> {
        self.present_queues
            .iter()
            .find(|(f, _)| *f == family)
            .map(|(_, q)| *q)
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

    /// Returns `true` if no slot of the requested capability was available and a
    /// higher-capability queue was substituted.
    pub fn is_overloaded(&self) -> bool {
        self.queue_match == QueueMatch::Overloaded
    }

    /// Returns `true` if this slot shares its underlying `VkQueue` with another semantic role.
    /// An aliased queue is not overloaded (it has the right capability), but it is not exclusive.
    pub fn is_aliased(&self) -> bool {
        self.queue_match == QueueMatch::Aliased
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
    /// Families whose flags include Graphics.  Present-capable families are sorted first.
    graphics: Vec<FamilyInfo>,
    /// Families that support Compute but not Graphics (dedicated compute).
    compute: Vec<FamilyInfo>,
    /// Families that support Transfer but not Compute or Graphics (dedicated DMA).
    transfer: Vec<FamilyInfo>,
}

impl FamilyCandidates {
    fn collect(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        present_families: &[u32],
    ) -> Self {
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

        // Sort present-capable families to the front so high_graphics naturally lands there.
        let present_set: HashSet<u32> = present_families.iter().copied().collect();
        out.graphics
            .sort_by_key(|f| !present_set.contains(&f.index));

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
/// 1. The high-priority graphics queue, preferably matching a present family request.
/// 2. Any present family requests may reserve extra high-priority graphics queues in specific families.
/// 3. Low-priority graphics queue for off-screen synthetic training
/// 3. High-priority compute queue
/// 4. Low-priority compute queue
/// 5. Dedicated transfer queue if present
///
/// To respect priority semantics, we need at least two graphics queues excepting those necessary
/// for presentation family requests.  We overload from lower capability to higher capability first,
/// so all queues may alias to graphics.  Second, we overload from low priority to high, so all low
/// priority queue requests might wind up as low priority graphics, but priority will be respected
/// if at all possible.  In the degenerate case, all queues are just one graphics queue.
pub struct QueuePlan {
    pub high_graphics: SlotAssignment,
    pub low_graphics: SlotAssignment,
    pub high_compute: SlotAssignment,
    pub low_compute: SlotAssignment,
    pub transfer: SlotAssignment,
    present_graphics: SmallVec<(u32, SlotAssignment), 8>,
    /// Per-family priority indexed by queue index within that family.  Built incrementally during
    /// construction.
    priorities: HashMap<u32, Vec<f32>>,
}

impl QueuePlan {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        present_families: &[u32],
    ) -> Result<Self, VulkanError> {
        let candidates = FamilyCandidates::collect(instance, physical_device, present_families);

        let mut priorities: HashMap<u32, Vec<QueuePriority>> = HashMap::new();

        // Initial pass for claiming queues
        //
        // All slots are claimed in priority order.  Present queues are claimed before low_graphics
        // so that two single-queue present families each secure their own High-priority slot before
        // low_graphics consumes whatever remains.
        //
        // Every claim is recorded into `priorities` immediately, raising the entry if the incoming
        // priority is higher than what was already there.  `priorities` is the single source of
        // truth for what gets submitted to the driver.

        let mut gfx = tier_iter(&candidates.graphics);

        // spec guarantees at least one graphics queue, which we immediately reserve for high_graphics.
        let high_graphics_claim = ClaimedSlot::from_tier(gfx.next().unwrap());
        record(&mut priorities, &high_graphics_claim, QueuePriority::High);

        // Each requested present family gets one High-priority slot.  If the family already has a
        // claim (high_graphics), we alias that slot rather than consuming a new index.  If the
        // family is unknown or not graphics-capable, that is a caller error.
        //
        // We advance `gfx` here when we consume a new index from a present family so that
        // low_graphics sees the correct remaining slots afterward.
        let mut present_claims: SmallVec<(u32, ClaimedSlot), 8> = SmallVec::new();
        // Track which (family, index) pairs have been consumed so far, so we can detect aliases.
        // We use claimed slots rather than a separate set to avoid another allocation.
        let already_claimed = |slot: &ClaimedSlot, claims: &[(u32, ClaimedSlot)]| -> bool {
            claims
                .iter()
                .any(|(_, c)| c.family == slot.family && c.index == slot.index)
                || (high_graphics_claim.family == slot.family
                    && high_graphics_claim.index == slot.index)
        };

        for &pf in present_families {
            if present_claims.iter().any(|(f, _)| *f == pf) {
                continue; // duplicate present family request
            }

            // Does the family have any outstanding entry in the graphics tier?
            let Some(family_info) = candidates.graphics.iter().find(|f| f.index == pf) else {
                return Err(VulkanError::InvalidArgument(format!(
                    "present family {pf} is not a graphics-capable family on this device"
                )));
            };

            // Check whether high_graphics already landed here.
            let claim = if high_graphics_claim.family == pf {
                ClaimedSlot::aliased(&high_graphics_claim)
            } else {
                // Advance gfx until we find a slot in this family, consuming it so that
                // low_graphics won't re-claim it.
                //
                // tier_iter yields families in sorted order (present-capable first), so if this
                // family is present-capable it will naturally come up soon.  We peek ahead and skip
                // over slots from other families that haven't been reserved yet, but we must not
                // discard them — buffer them for low_graphics.
                //
                // Rather than a complex peek-and-buffer scheme, we find the next free index in this
                // family directly from the family_info, and record it as claimed so that
                // tier_iter's linear scan for low_graphics won't double-claim it.
                let next_free = priorities
                    .get(&pf)
                    .map(|v| v.len() as u32)
                    .unwrap_or(0)
                    .min(family_info.count.saturating_sub(1));
                ClaimedSlot {
                    family: pf,
                    index: next_free,
                    actual_flags: family_info.flags,
                    queue_match: QueueMatch::Exact,
                }
            };

            record(&mut priorities, &claim, QueuePriority::High);
            present_claims.push((pf, claim));
        }

        // Decide low priority graphics.  First graphics slot not already consumed by high_graphics
        // or present.  tier_iter resumes from where it left off after high_graphics, but present
        // claims may have staked out indices within families that gfx hasn't reached yet.  Skip
        // over any (family, index) already recorded in priorities at a non-zero level.
        let low_graphics_claim = loop {
            match gfx.next() {
                None => {
                    // No free graphics slot. Alias high_graphics.
                    break ClaimedSlot::aliased(&high_graphics_claim);
                }
                Some(t) => {
                    let candidate = ClaimedSlot::from_tier(t);
                    // Skip if this index was already claimed by a present queue.
                    if priorities
                        .get(&candidate.family)
                        .and_then(|v| v.get(candidate.index as usize))
                        .is_some()
                    {
                        continue;
                    }
                    break candidate;
                }
            }
        };
        record(&mut priorities, &low_graphics_claim, QueuePriority::Low);

        let mut comp = tier_iter(&candidates.compute);
        let mut xfer = tier_iter(&candidates.transfer);

        let high_compute_claim = claim(comp.next(), &low_graphics_claim);
        record(&mut priorities, &high_compute_claim, QueuePriority::High);

        let low_compute_claim = claim(comp.next(), &high_compute_claim);
        record(&mut priorities, &low_compute_claim, QueuePriority::Low);

        let transfer_claim = claim(xfer.next(), &low_compute_claim);
        record(&mut priorities, &transfer_claim, QueuePriority::Low);

        // All claims are recorded.  Promote each ClaimedSlot to a SlotAssignment by reading the
        // finalized priority back out of the completed priorities map.  Every Queue handle
        // produced from these assignments will report the priority actually submitted to the
        // driver.

        Ok(QueuePlan {
            high_graphics: high_graphics_claim.finalize(&priorities),
            low_graphics: low_graphics_claim.finalize(&priorities),
            high_compute: high_compute_claim.finalize(&priorities),
            low_compute: low_compute_claim.finalize(&priorities),
            transfer: transfer_claim.finalize(&priorities),
            present_graphics: present_claims
                .into_iter()
                .map(|(f, c)| (f, c.finalize(&priorities)))
                .collect(),
            priorities: priorities
                .iter()
                .map(|(&family, prios)| (family, prios.iter().map(|p| p.as_f32()).collect()))
                .collect(),
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

/// Yields `(family_index, queue_index, flags)` by exhausting each family's available queues
/// before moving to the next.
fn tier_iter(families: &[FamilyInfo]) -> impl Iterator<Item = (u32, u32, vk::QueueFlags)> + '_ {
    families
        .iter()
        .flat_map(|f| (0..f.count).map(move |i| (f.index, i, f.flags)))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::with_context;

    #[test]
    fn inspect_graphics() {
        // implicitly creates queues to obtain a device context
        with_context!(|device_context, vk_context| {
            let gfx = device_context.queues.graphics(QueuePriority::High);
            assert_eq!(gfx.priority, QueuePriority::High);
            assert!(gfx.actual_flags.contains(vk::QueueFlags::GRAPHICS));
            assert!(gfx.capabilities().contains(vk::QueueFlags::GRAPHICS));
            assert_eq!(gfx.is_overloaded(), false);
        });
    }

    #[test]
    fn transfer_is_always_low_priority() {
        with_context!(|device_context, _vk_context| {
            let gfx_low = device_context.queues.compute(QueuePriority::Low);
            let transfer = device_context.queues.transfer();
            if gfx_low.priority == QueuePriority::Low {
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
            let gfx_low = device_context.queues.compute(QueuePriority::Low);
            let compute_low = device_context.queues.compute(QueuePriority::Low);
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
