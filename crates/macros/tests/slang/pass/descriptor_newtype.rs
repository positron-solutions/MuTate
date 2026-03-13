// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_vulkan::slang::prelude::*;

descriptor_newtype!(ShadowMapIdx, SampledImageIdx, "ShadowMapIdx");

fn require_gpu_pod<L: LayoutRule, T: GpuPod<L>>() {}

fn main() {
    require_gpu_pod::<ScalarLayout, ShadowMapIdx>();

    // PRIMITIVE and size/align forward from inner (SampledImageIdx → UInt32)
    assert_eq!(
        <ShadowMapIdx as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt32
    );
    assert_eq!(<ShadowMapIdx as GpuType<ScalarLayout>>::SIZE, 4);
    assert_eq!(<ShadowMapIdx as GpuType<ScalarLayout>>::ALIGN, 4);
    // SLANG_NAME is the newtype's own name — not "SampledImageIdx"
    assert_eq!(
        <ShadowMapIdx as GpuType<ScalarLayout>>::SLANG_NAME,
        "ShadowMapIdx"
    );

    let idx = ShadowMapIdx::new(7);
    assert_eq!(idx.raw(), 7u32);

    assert!(idx.is_valid());
    assert!(!ShadowMapIdx::INVALID.is_valid());

    assert_eq!(ShadowMapIdx::INVALID.raw(), u32::MAX);

    assert!(ShadowMapIdx::new(0).is_valid());

    let base = SampledImageIdx::new(3);
    let wrapped = ShadowMapIdx::from(base);
    assert_eq!(wrapped.raw(), 3u32);

    let inner: SampledImageIdx = wrapped.into_inner();
    assert_eq!(inner.raw(), 3u32);

    let via_into: ShadowMapIdx = SampledImageIdx::new(5).into();
    assert_eq!(via_into.raw(), 5u32);

    let bytes: &[u8] = bytemuck::bytes_of(&idx);
    assert_eq!(bytes.len(), 4);
    let recovered: ShadowMapIdx = *bytemuck::from_bytes(bytes);
    assert_eq!(recovered.raw(), 7u32);

    let z: ShadowMapIdx = bytemuck::Zeroable::zeroed();
    assert_eq!(z.raw(), 0u32);
    assert!(z.is_valid());
}
