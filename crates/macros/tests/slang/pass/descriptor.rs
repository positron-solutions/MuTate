// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_vulkan::slang::prelude::*;

fn require_gpu_pod<D: DataLayout, T: GpuPod<D>>() {}

fn main() {
    require_gpu_pod::<Scalar, SampledImageIdx>();

    assert_eq!(<SampledImageIdx as GpuScalar>::PRIMITIVE, SlangType::UInt);
    assert_eq!(
        <SampledImageIdx as GpuScalar>::SLANG_NAME,
        "SampledImageIdx"
    );
    assert_eq!(<SampledImageIdx as GpuScalar>::SIZE, 4);
    assert_eq!(<SampledImageIdx as GpuPrimitive<Scalar>>::ALIGN, 4);

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
