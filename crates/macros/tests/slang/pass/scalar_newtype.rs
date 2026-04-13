// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_vulkan::slang::prelude::*;

// XXX think about this.. do we slang type as the base slang type or the base Rust type?  My bet is
// on base Slang type, but it has a fundamental Rust type too, so we might want both.
slang_newtype!(JointWeight, Float, "JointWeight");

fn require_gpu_pod<D: DataLayout, T: GpuPod<D>>() {}

fn main() {
    require_gpu_pod::<Scalar, JointWeight>();

    // PRIMITIVE and size/align forward from inner
    assert_eq!(<JointWeight as GpuScalar>::PRIMITIVE, SlangType::Float);
    assert_eq!(<JointWeight as GpuScalar>::SIZE, 4);
    assert_eq!(<JointWeight as GpuPrimitive<Scalar>>::ALIGN, 4);
    // SLANG_NAME is the wrapper's own name — not the inner's "float"
    assert_eq!(<JointWeight as GpuScalar>::SLANG_NAME, "JointWeight");

    let a = JointWeight(Float::from(0.75f32));
    assert_eq!(a.into_inner().into_inner(), 0.75f32);

    let b: JointWeight = JointWeight(Float::from(0.25f32));
    assert_eq!(b.into_inner().into_inner(), 0.25f32);

    let inner: Float = a.into_inner();
    assert_eq!(inner.into_inner(), 0.75f32);

    let original = JointWeight(Float::from(std::f32::consts::E));
    let bytes: &[u8] = bytemuck::bytes_of(&original);
    assert_eq!(bytes.len(), 4);
    let recovered: JointWeight = *bytemuck::from_bytes(bytes);
    assert_eq!(recovered.into_inner().into_inner(), std::f32::consts::E);

    let z: JointWeight = bytemuck::Zeroable::zeroed();
    assert_eq!(z.into_inner().into_inner(), 0.0f32);
}
