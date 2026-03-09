// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must compile and assert: descriptor newtypes carry correct layout, INVALID sentinel,
// is_valid(), and satisfy GpuPod. Tests all four base descriptor kinds.

use mutate_vulkan::slang::prelude::*;

descriptor_newtype!(ShadowMapIdx, SampledImageIdx, "ShadowMapIdx");
descriptor_newtype!(GBufferAlbedoIdx, SampledImageIdx, "GBufferAlbedoIdx");
descriptor_newtype!(
    SkinnedMeshBufferIdx,
    StorageBufferIdx,
    "SkinnedMeshBufferIdx"
);
descriptor_newtype!(TileStorageIdx, StorageImageIdx, "TileStorageIdx");

fn require_gpu_pod<L: LayoutRule, T: GpuPod<L>>() {}

fn main() {
    require_gpu_pod::<ScalarLayout, ShadowMapIdx>();
    require_gpu_pod::<ScalarLayout, GBufferAlbedoIdx>();
    require_gpu_pod::<ScalarLayout, SkinnedMeshBufferIdx>();
    require_gpu_pod::<ScalarLayout, TileStorageIdx>();

    // All descriptor newtypes are uint32_t on the wire
    assert_eq!(
        <ShadowMapIdx as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt32
    );
    assert_eq!(
        <SkinnedMeshBufferIdx as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt32
    );
    assert_eq!(<ShadowMapIdx as GpuType<ScalarLayout>>::SIZE, 4);
    assert_eq!(<ShadowMapIdx as GpuType<ScalarLayout>>::ALIGN, 4);

    // SLANG_NAME is the newtype's name
    assert_eq!(
        <ShadowMapIdx as GpuType<ScalarLayout>>::SLANG_NAME,
        "ShadowMapIdx"
    );
    assert_eq!(
        <SkinnedMeshBufferIdx as GpuType<ScalarLayout>>::SLANG_NAME,
        "SkinnedMeshBufferIdx"
    );

    // INVALID sentinel is u32::MAX
    assert_eq!(ShadowMapIdx::INVALID.raw(), u32::MAX);
    assert!(!ShadowMapIdx::INVALID.is_valid());

    // Valid index round-trips
    let idx = ShadowMapIdx::new(7);
    assert_eq!(idx.raw(), 7);
    assert!(idx.is_valid());

    // Into<> via From<inner>
    let base = SampledImageIdx::new(3);
    let wrapped: ShadowMapIdx = base.into();
    assert_eq!(wrapped.raw(), 3);
    assert_eq!(wrapped.into_inner().raw(), 3);
}
