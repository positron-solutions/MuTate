// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// Must compile: slang_newtype! forwards PRIMITIVE, SIZE, ALIGN from its inner type.

use mutate_vulkan::slang::prelude::*;

slang_newtype!(Hotness, Float, "Hotness");
slang_newtype!(JointIndex, UInt16, "JointIndex");

fn require_gpu_type<D: DataLayout, T: GpuType<D>>() {}

fn main() {
    require_gpu_type::<Scalar, Hotness>();
    require_gpu_type::<Scalar, JointIndex>();

    // PRIMITIVE forwards from inner
    assert_eq!(<Hotness as GpuScalar>::PRIMITIVE, SlangType::Float,);
    assert_eq!(<JointIndex as GpuScalar>::PRIMITIVE, SlangType::UInt16,);

    // SIZE / ALIGN match the inner scalar
    assert_eq!(<Hotness as GpuScalar>::SIZE, 4);
    assert_eq!(<Hotness as GpuPrimitive<Scalar>>::ALIGN, 4);
    assert_eq!(<JointIndex as GpuScalar>::SIZE, 2);
    assert_eq!(<JointIndex as GpuPrimitive<Scalar>>::ALIGN, 2);

    // SLANG_NAME is the *wrapper* name, not the inner one
    assert_eq!(<Hotness as GpuScalar>::SLANG_NAME, "Hotness");
    assert_eq!(<JointIndex as GpuScalar>::SLANG_NAME, "JointIndex");
}
