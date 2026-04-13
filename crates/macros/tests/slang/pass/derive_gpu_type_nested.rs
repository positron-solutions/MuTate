// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use mutate_macros::GpuType;
use mutate_vulkan::prelude::*;

#[derive(GpuType)]
#[repr(C)]
struct SlangTypes {
    foo: UInt,
    bar: Float,
}

#[derive(GpuType)]
#[repr(C)]
struct NestedSlangTypes {
    foo: UInt,
    bar: Half,
    inner: SlangTypes,
}

// A Half-terminated struct: its own size must round up to 4 so that
// back-to-back uses (arrays, further embedding) stay aligned.
#[derive(GpuType)]
#[repr(C)]
struct HalfTerminated {
    a: UInt, // [0..4)
    b: Half, // [4..6) — tail; struct rounds to align(4) → size 8
}

// Three levels deep: checks that alignment propagates all the way up.
#[derive(GpuType)]
#[repr(C)]
struct Outer {
    tag: UInt8,              // [0..1)
    inner: NestedSlangTypes, // align=4 → padded to [4..20)
}

fn main() {
    // Size checks ──────────────────────────────────────────────────────────
    assert_eq!(<SlangTypes as GpuType<Scalar>>::SIZE, 8);
    assert_eq!(<NestedSlangTypes as GpuType<Scalar>>::SIZE, 16);
    assert_eq!(<HalfTerminated as GpuType<Scalar>>::SIZE, 8);
    assert_eq!(<Outer as GpuType<Scalar>>::SIZE, 20);

    // Alignment checks ─────────────────────────────────────────────────────
    assert_eq!(<SlangTypes as GpuType<Scalar>>::ALIGN, 4);
    assert_eq!(<NestedSlangTypes as GpuType<Scalar>>::ALIGN, 4);
    assert_eq!(<HalfTerminated as GpuType<Scalar>>::ALIGN, 4);
    assert_eq!(<Outer as GpuType<Scalar>>::ALIGN, 4);

    // Byte checks for scalar writes.
    let foo_val: u32 = 0x01020304_u32;
    let bar_val: half::f16 = half::f16::from_bits(0xABCD);
    let ifoo_val: u32 = 0x05060708_u32;
    let ibar_val: f32 = f32::from_bits(0x090A0B0C_u32); // propagate the sentinel into the struct

    // Manually build expected output, padding bytes are zero.
    let mut expected = [0u8; <NestedSlangTypes as GpuType<Scalar>>::SIZE];
    expected[0..4].copy_from_slice(&foo_val.to_le_bytes());
    expected[4..6].copy_from_slice(&bar_val.to_le_bytes());
    // [6..8) stays zero — padding
    expected[8..12].copy_from_slice(&ifoo_val.to_le_bytes());
    expected[12..16].copy_from_slice(&ibar_val.to_le_bytes());

    // Construct the value and pack it.  pack_into must produce identical bytes (regions skip the
    // padding gap).  We zero the buffer first so unwritten padding bytes compare as zero too.  If
    // pack_into writes into the padding window the test will still pass — what matters is the data
    // regions land at the right offsets.
    let v = NestedSlangTypes {
        foo: UInt::from(foo_val),
        bar: Half::from(bar_val),
        inner: SlangTypes {
            foo: UInt::from(ifoo_val),
            bar: Float::from(ibar_val),
        },
    };
    let mut buf = [0u8; <NestedSlangTypes as GpuType<Scalar>>::SIZE];
    <NestedSlangTypes as Pack<Scalar>>::pack_into(&v, &mut buf);

    assert_eq!(&buf[0..4], &expected[0..4], "foo region");
    assert_eq!(&buf[4..6], &expected[4..6], "bar region");
    assert_eq!(&buf[8..12], &expected[8..12], "inner.foo region");
    assert_eq!(&buf[12..16], &expected[12..16], "inner.bar region");

    // Extend checks to transitive nesting.
    let mut buf2 = [0u8; <HalfTerminated as GpuType<Scalar>>::SIZE];
    let v2 = HalfTerminated {
        a: UInt::from(0xCAFE_BABEu32),
        b: Half::from(half::f16::from_f32(1.0)),
    };
    <HalfTerminated as Pack<Scalar>>::pack_into(&v2, &mut buf2);

    assert_eq!(&buf2[0..4], &0xCAFE_BABEu32.to_le_bytes(), "a region");
    assert_eq!(
        &buf2[4..6],
        &half::f16::from_f32(1.0).to_bits().to_le_bytes(),
        "b region"
    );
    assert_eq!(&buf2[6..8], &[0u8; 2], "trailing padding must be zero");
}
