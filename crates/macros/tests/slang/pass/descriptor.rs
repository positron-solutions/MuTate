// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_vulkan::slang::prelude::*;

fn require_gpu_pod<L: LayoutRule, T: GpuPod<L>>() {}

fn main() {
    require_gpu_pod::<ScalarLayout, SampledImageIdx>();

    assert_eq!(
        <SampledImageIdx as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::UInt32
    );
    assert_eq!(
        <SampledImageIdx as GpuType<ScalarLayout>>::SLANG_NAME,
        "SampledImageIdx"
    );
    assert_eq!(<SampledImageIdx as GpuType<ScalarLayout>>::SIZE, 4);
    assert_eq!(<SampledImageIdx as GpuType<ScalarLayout>>::ALIGN, 4);

    let idx = SampledImageIdx::new(42);
    assert_eq!(idx.raw(), 42u32);

    assert!(idx.is_valid());
    assert!(!SampledImageIdx::INVALID.is_valid());

    assert_eq!(SampledImageIdx::INVALID.raw(), u32::MAX);

    assert!(SampledImageIdx::new(0).is_valid());

    let bytes: &[u8] = bytemuck::bytes_of(&idx);
    assert_eq!(bytes.len(), 4);
    let recovered: SampledImageIdx = *bytemuck::from_bytes(bytes);
    assert_eq!(recovered.raw(), 42u32);

    let z: SampledImageIdx = bytemuck::Zeroable::zeroed();
    assert_eq!(z.raw(), 0u32);
    assert!(z.is_valid());
}
