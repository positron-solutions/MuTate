// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_vulkan::slang::prelude::*;

fn require_gpu_pod<L: LayoutRule, T: GpuPod<L>>() {}

fn main() {
    require_gpu_pod::<ScalarLayout, Float32>();

    assert_eq!(
        <Float32 as GpuType<ScalarLayout>>::PRIMITIVE,
        SlangType::Float32
    );
    assert_eq!(<Float32 as GpuType<ScalarLayout>>::SLANG_NAME, "float");
    assert_eq!(<Float32 as GpuType<ScalarLayout>>::SIZE, 4);
    assert_eq!(<Float32 as GpuType<ScalarLayout>>::ALIGN, 4);

    let a = Float32::from(1.0f32);
    assert_eq!(a.into_inner(), 1.0f32);

    let b: Float32 = 2.0f32.into();
    assert_eq!(b.into_inner(), 2.0f32);

    let original = Float32::from(std::f32::consts::PI);
    let bytes: &[u8] = bytemuck::bytes_of(&original);
    assert_eq!(bytes.len(), 4);
    let recovered: Float32 = *bytemuck::from_bytes(bytes);
    assert_eq!(recovered.into_inner(), std::f32::consts::PI);

    let z: Float32 = bytemuck::Zeroable::zeroed();
    assert_eq!(z.into_inner(), 0.0f32);
}
