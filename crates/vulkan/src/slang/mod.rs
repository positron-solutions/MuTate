// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Slang
//!
//! Make Rust types that correspond to Slang types and can be used in Pipeline-Shader agreement
//! checking.  Built-in Slang types are provided.  Macros are provided for creating new-type wrappers
//! around existing Slang types, adding further semantic type specification.

use half::prelude::*;

/// Types that can be used in layout / type agreement checking implement the Slang type first.
/// For a type to "agree" it's strictly necessary to have the same layout size and type for each
/// field of any struct.  The names of fields and types can be more flexible when explicitly allowed
/// to limit friction during refactoring, for example.
pub struct Slang;

/// A type that has a known layout under layout rule `L`.
/// The witness — what layout checking reads from the leaf.
pub trait Layout<L> {
    const SLANG_TYPE: SlangType;
}

macro_rules! slang_scalar {
    ($name:ident, $base:ty) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $base);

        // We have a way to opt into lowering the type.
        impl $name {
            #[inline(always)]
            pub fn into_inner(self) -> $base {
                self.0
            }
        }

        // We can always make base into the new type.
        impl From<$base> for $name {
            #[inline(always)]
            fn from(v: $base) -> Self {
                Self(v)
            }
        }

        // Delegate the Layout to the underling type's Layout implementation.
        impl<L> Layout<L> for $name {
            const SLANG_TYPE: SlangType = SlangType::$name;
        }
    };
}

// Shorthand to implement Layout for basic Slang scalars.
macro_rules! impl_layout {
    ($rust_type:ty => $variant:ident) => {
        impl<L> Layout<L> for $rust_type {
            const SLANG_TYPE: SlangType = SlangType::$variant;
        }
    };
}

slang_scalar!(Bool, bool);

slang_scalar!(Int8, i8);
slang_scalar!(Int16, i16);
slang_scalar!(Int32, i32);
slang_scalar!(Int64, i64);

slang_scalar!(UInt8, u8);
slang_scalar!(UInt16, u16);
slang_scalar!(UInt32, u32);
slang_scalar!(UInt64, u64);

slang_scalar!(Half, f16);
slang_scalar!(Float, f32);
slang_scalar!(Double, f64);

slang_scalar!(Bool2, [bool; 2]);
slang_scalar!(Bool3, [bool; 3]);
slang_scalar!(Bool4, [bool; 4]);

slang_scalar!(Int8_2, [i8; 2]);
slang_scalar!(Int8_3, [i8; 3]);
slang_scalar!(Int8_4, [i8; 4]);

slang_scalar!(Int16_2, [i16; 2]);
slang_scalar!(Int16_3, [i16; 3]);
slang_scalar!(Int16_4, [i16; 4]);

slang_scalar!(Int2, [i32; 2]);
slang_scalar!(Int3, [i32; 3]);
slang_scalar!(Int4, [i32; 4]);

slang_scalar!(Int64_2, [i64; 2]);
slang_scalar!(Int64_3, [i64; 3]);
slang_scalar!(Int64_4, [i64; 4]);

slang_scalar!(UInt8_2, [u8; 2]);
slang_scalar!(UInt8_3, [u8; 3]);
slang_scalar!(UInt8_4, [u8; 4]);

slang_scalar!(UInt16_2, [u16; 2]);
slang_scalar!(UInt16_3, [u16; 3]);
slang_scalar!(UInt16_4, [u16; 4]);

slang_scalar!(UInt2, [u32; 2]);
slang_scalar!(UInt3, [u32; 3]);
slang_scalar!(UInt4, [u32; 4]);

slang_scalar!(UInt64_2, [u64; 2]);
slang_scalar!(UInt64_3, [u64; 3]);
slang_scalar!(UInt64_4, [u64; 4]);

slang_scalar!(Half2, [f16; 2]);
slang_scalar!(Half3, [f16; 3]);
slang_scalar!(Half4, [f16; 4]);

slang_scalar!(Float2, [f32; 2]);
slang_scalar!(Float3, [f32; 3]);
slang_scalar!(Float4, [f32; 4]);

slang_scalar!(Double2, [f64; 2]);
slang_scalar!(Double3, [f64; 3]);
slang_scalar!(Double4, [f64; 4]);

slang_scalar!(Float2x2, [[f32; 2]; 2]);
slang_scalar!(Float3x3, [[f32; 3]; 3]);
slang_scalar!(Float4x4, [[f32; 4]; 4]);

slang_scalar!(Half2x2, [[f16; 2]; 2]);
slang_scalar!(Half3x3, [[f16; 3]; 3]);
slang_scalar!(Half4x4, [[f16; 4]; 4]);

slang_scalar!(Double2x2, [[f64; 2]; 2]);
slang_scalar!(Double3x3, [[f64; 3]; 3]);
slang_scalar!(Double4x4, [[f64; 4]; 4]);
