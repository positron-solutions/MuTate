// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_vulkan::slang::prelude::*;

fn require_gpu_pod<D: DataLayout, T: GpuPod<D>>() {}
fn require_gpu_pod_scalar<T: GpuPod>() {
    require_gpu_pod::<Scalar, T>()
}

fn main() {
    require_gpu_pod::<Scalar, Float>();
    require_gpu_pod_scalar::<Float>();

    assert_eq!(<Float as GpuScalar>::PRIMITIVE, SlangType::Float);
    assert_eq!(<Float as GpuScalar>::SLANG_NAME, "float");
    assert_eq!(<Float as GpuType>::SLANG_NAME, "float");
    assert_eq!(<Float as GpuScalar>::SIZE, 4);
    assert_eq!(<Float as GpuPrimitive>::ALIGN, 4);

    let a = Float::from(1.0f32);
    assert_eq!(a.into_inner(), 1.0f32);

    let b: Float = 2.0f32.into();
    assert_eq!(b.into_inner(), 2.0f32);

    let original = Float::from(std::f32::consts::PI);
    let bytes: &[u8] = bytemuck::bytes_of(&original);
    assert_eq!(bytes.len(), 4);
    let recovered: Float = *bytemuck::from_bytes(bytes);
    assert_eq!(recovered.into_inner(), std::f32::consts::PI);

    let z: Float = bytemuck::Zeroable::zeroed();
    assert_eq!(z.into_inner(), 0.0f32);
}
