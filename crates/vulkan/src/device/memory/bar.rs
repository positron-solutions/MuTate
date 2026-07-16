// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # BAR
//!
//! Detect device Base Address Register (BAR) capability, directly host-addressable memory over PCIe
//! or a specific carve-out on some UMA devices.  Used by [`Memory`](super::Memory) to influence
//! heap selections for different [`MemoryUse`](super::MemoryUse).

use crate::internal::*;

/// A request's stake in a scarce BAR carve-out.
///
/// Only meaningful when a carve-out exists ([`detect_bar`] returns non-zero).  On
/// ReBAR, UMA, and no-BAR parts the mask is zero and both variants behave alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarPolicy {
    /// This request is the reason the window exists.  Permits, never requires: on a
    /// device with no carve-out the request's own `required` bits decide.
    Claim,
    /// This request must not land in the window.  Everything that isn't pushing
    /// small writes at device memory.
    Yield,
}

pub(crate) fn detect_bar(props: &vk::PhysicalDeviceMemoryProperties) -> u32 {
    use vk::MemoryPropertyFlags as F;
    const MAPPED: F = F::from_raw(F::DEVICE_LOCAL.as_raw() | F::HOST_VISIBLE.as_raw());
    let types = &props.memory_types[..props.memory_type_count as usize];
    let heaps = &props.memory_heaps[..props.memory_heap_count as usize];

    let mapped_heaps: Vec<u32> = {
        let mut v: Vec<u32> = types
            .iter()
            .filter(|t| t.property_flags.contains(MAPPED))
            .map(|t| t.heap_index)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if mapped_heaps.is_empty() {
        return 0;
    }

    let vram_heap = heaps
        .iter()
        .enumerate()
        .filter(|(_, h)| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .max_by_key(|&(_, h)| h.size)
        .map(|(i, _)| i as u32)
        .expect("a mapped DEVICE_LOCAL type implies a DEVICE_LOCAL heap");

    // VRAM itself mapped => ReBAR or UMA => window is not scarce.
    if mapped_heaps.contains(&vram_heap) {
        return 0;
    }

    // Otherwise the smallest mapped heap is the carve-out.
    let window_heap = *mapped_heaps
        .iter()
        .min_by_key(|&&h| heaps[h as usize].size)
        .unwrap();

    types
        .iter()
        .enumerate()
        .filter(|(_, t)| t.property_flags.contains(MAPPED) && t.heap_index == window_heap)
        .fold(0u32, |m, (i, _)| m | 1 << i)
}
