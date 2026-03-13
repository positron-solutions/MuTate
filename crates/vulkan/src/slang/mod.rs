// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Slang
//!
//! Make Rust types that correspond to Slang types and can be used in Pipeline-Shader agreement
//! checking.  Macros are provided for creating new-type wrappers
//! around existing Slang types, adding further semantic type specification.
//!
//! ## Semantic Type Wrappers
//!
//! Built-in Slang types are provided by using `slang_scalar!` to wrap trivial Rust types with
//! wrappers that use the Slang type naming.
//!
//! In cases where a Slang type has a type-safe wrapper:
//!
//! ```slang
//! struct Temperature { int value; }
//! ```
//!
//! The field name is **not** checked for tuples to single-field structs.
//!
//! ## Layout
//!
//! We prefer scalar block layout but have left the door open to supporting other layouts.  Layouts
//! are enforced at consumption time through trait bounds.
//!
//! ```ignore
//! pub fn write_buffer<L: LayoutRule, T>(data: &[T])
//! where
//!     T: GpuPod<L>,  //! this bound does all the work
//! {
//!     ...
//! }
//! ```
//!
//! To obtain the bounds, a type must implement GpuPod<L>.  For many types in block scalar layout,
//! this bound is trivially satisfied with no extra work.
//!
//! A type that requires padding for std430 will not satisfy `GpuPod<Std430>`:
//!
//! ```ignore
//! #[derive(GpuType)]
//! #[repr(C)]
//! struct Skinning {
//!     joints: Float3, // under std430: occupies 16 bytes, not 12
//!     weights: Float4,
//! }
//! ```
//!
//! The user can either fix the base (add appropriate padding) or add a re-packing trait:
//!
//! ```ignore
//! #[derive(GpuType)]
//! #[repr(C)]
//! struct Skinning {
//!     joints: Float3, // under std430: occupies 16 bytes, not 12
//!     weights: Float4,
//! }
//! ```
//!
//! Now the type can be used to write to arrays, (but consider AoS solutions for more flexible
//! buffer usage, such as adding and removing columns, and faster performance).
//!
//! # Buffer Device Address
//!
//! A `DeviceAddress` type has been provided for handling pointers without cutting our hands on bare
//! u64. Use `device_address_newtype!` to create type-safe derivatives.
//!
//! # Descriptor Indexes
//!
//! We use a single descriptor table with known slots.  We use a Slang library to handle raw uint
//! indexes into that table.  On the rust side, we provide both the base wrapped indexes and a
//! `descriptor_newtype!` macro for creating type-safe derivatives.

use bytemuck::{Pod, Zeroable};
use half::prelude::*;

pub mod prelude {
    // Layout rules — users name these at call sites and in trait bounds
    pub use super::{LayoutRule, ScalarLayout};

    // Core traits — needed for generic bounds and introspection
    pub use super::{GpuPod, GpuType, HasBlock};

    // Type discrimination — users match/compare PRIMITIVE constants
    pub use super::SlangType;

    // Scalars — the everyday currency of GPU data
    pub use super::{
        Bool, Float16, Float32, Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8,
    };

    // Pointer type and its trait
    pub use super::{DeviceAddress, IsDeviceAddress};

    // Descriptor index base types and their trait
    pub use super::{
        DescriptorIndex, SampledImageIdx, StorageBufferIdx, StorageImageIdx, UniformBufferIdx,
    };

    pub use crate::descriptor_newtype;
    pub use crate::device_address_newtype;
    pub use crate::slang_newtype;

    // Re-exports for convenience
    pub use bytemuck;
    pub use bytemuck::{Pod, Zeroable};
    // XXX may need the whole f16 prelude
    pub use half;
}

mod sealed {
    pub trait Sealed {}
}

pub trait LayoutRule: sealed::Sealed {}

pub struct ScalarLayout;
pub struct Std430; // reserved, not yet implemented

impl sealed::Sealed for ScalarLayout {}
impl sealed::Sealed for Std430 {}
impl LayoutRule for ScalarLayout {}
impl LayoutRule for Std430 {}

/// A type that requires a separate packed representation under layout rule L.
/// The associated Block type IS its GpuPod.
/// Implemented only where the layout does NOT collapse.
pub trait HasBlock<L: LayoutRule>: GpuType<L> {
    type Block: GpuPod<L>;
    fn into_block(self) -> Self::Block;
}

/// Closed enumeration of Slang primitive types.  Every GPU type bottoms out at one of these.  User
/// newtypes and structs carry their own SLANG_NAME but delegate PRIMITIVE downward to the leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Struct, // composites — SLANG_NAME carries the identity
}

pub trait GpuType<L: LayoutRule> {
    /// The irreducible Slang primitive this type reduces to.
    /// Drives wire layout and type-safety checks (u16 vs f16, etc.)
    const PRIMITIVE: SlangType;

    /// The Slang-side name for introspection matching.
    /// For primitives: matches the Slang builtin name ("float16_t" etc.)
    /// For newtypes:   the Slang struct name ("Temperature" etc.)
    /// For structs:    the Slang struct name
    const SLANG_NAME: &'static str;

    const SIZE: usize;
    const ALIGN: usize;
}

/// Marker trait for types whose layout under L is correct as-is —
/// no repacking needed. Requires `bytemuck::Pod` for safe transmute.
///
/// SAFETY: implementor must ensure SIZE/ALIGN match the actual
/// in-memory representation under L.
pub unsafe trait GpuPod<L: LayoutRule>: GpuType<L> + Pod {}

/// Registers a Rust type as a Slang scalar primitive.
/// Emits a repr(transparent) newtype that:
///   - implements GpuType<L> for all L (primitives are layout-agnostic at scalar level)
///   - implements GpuPod<L> for all L
///   - allows From<$base> but NOT From<NewType> for base (one direction only)
macro_rules! slang_scalar {
    ($name:ident, $base:ty, $variant:expr, $slang_name:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $base);

        impl $name {
            #[inline(always)]
            pub fn into_inner(self) -> $base {
                self.0
            }
        }

        // Base → wrapper only. No reverse.
        impl From<$base> for $name {
            #[inline(always)]
            fn from(v: $base) -> Self {
                Self(v)
            }
        }

        impl<L: LayoutRule> GpuType<L> for $name {
            const PRIMITIVE: SlangType = $variant;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = std::mem::size_of::<$base>();
            const ALIGN: usize = std::mem::align_of::<$base>();
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<L: LayoutRule> GpuPod<L> for $name {}
    };
}

/// Wraps an existing slang_scalar type with a semantic name.
/// The inner type must already implement GpuType<L> — that bound
/// IS the witness. If it doesn't, the impl fails at the definition site.
///
/// SLANG_NAME is the Slang struct name for introspection.
/// PRIMITIVE and layout constants are forwarded from the inner type.
#[macro_export]
macro_rules! slang_newtype {
    ($name:ident, $inner:ty, $slang_name:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $inner);

        impl $name {
            #[inline(always)]
            pub fn into_inner(self) -> $inner {
                self.0
            }
        }

        impl From<$name> for $inner {
            #[inline(always)]
            fn from(v: $name) -> Self {
                v.0
            }
        }

        impl<L: LayoutRule> GpuType<L> for $name
        where
            $inner: GpuType<L>, // witness — if inner isn't registered, fails here
        {
            const PRIMITIVE: SlangType = <$inner as GpuType<L>>::PRIMITIVE;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = <$inner as GpuType<L>>::SIZE;
            const ALIGN: usize = <$inner as GpuType<L>>::ALIGN;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<L: LayoutRule> GpuPod<L> for $name where $inner: GpuType<L> {}
    };
}

// Bool is special: Slang bool is 4 bytes on GPU, not 1.
// We do NOT wrap Rust's bool directly — use a u32 newtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Zeroable, Pod)]
#[repr(transparent)]
pub struct Bool(u32);

impl Bool {
    pub const TRUE: Bool = Bool(1);
    pub const FALSE: Bool = Bool(0);
}

impl From<bool> for Bool {
    fn from(b: bool) -> Self {
        Bool(b as u32)
    }
}

impl<L: LayoutRule> GpuType<L> for Bool {
    const PRIMITIVE: SlangType = SlangType::Bool;
    const SLANG_NAME: &'static str = "bool";
    const SIZE: usize = 4;
    const ALIGN: usize = 4;
}

unsafe impl<L: LayoutRule> GpuPod<L> for Bool {}

slang_scalar!(Int8, i8, SlangType::Int8, "int8_t");
slang_scalar!(Int16, i16, SlangType::Int16, "int16_t");
slang_scalar!(Int32, i32, SlangType::Int32, "int32_t");
slang_scalar!(Int64, i64, SlangType::Int64, "int64_t");

slang_scalar!(UInt8, u8, SlangType::UInt8, "uint8_t");
slang_scalar!(UInt16, u16, SlangType::UInt16, "uint16_t");
slang_scalar!(UInt32, u32, SlangType::UInt32, "uint32_t");
slang_scalar!(UInt64, u64, SlangType::UInt64, "uint64_t");

// Integer types have total equality; floats do not (NaN != NaN).
impl Eq for Int8 {}
impl Eq for Int16 {}
impl Eq for Int32 {}
impl Eq for Int64 {}
impl Eq for UInt8 {}
impl Eq for UInt16 {}
impl Eq for UInt32 {}
impl Eq for UInt64 {}

slang_scalar!(Float16, f16, SlangType::Float16, "float16_t");
slang_scalar!(Float32, f32, SlangType::Float32, "float");
slang_scalar!(Float64, f64, SlangType::Float64, "double");

pub trait IsDeviceAddress {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Zeroable, Pod)]
#[repr(transparent)]
pub struct DeviceAddress(pub(crate) UInt64);

impl DeviceAddress {
    #[inline(always)]
    pub fn into_inner(self) -> UInt64 {
        self.0
    }
    #[inline(always)]
    pub fn raw(self) -> u64 {
        self.0.into_inner()
    }
    pub const NULL: DeviceAddress = DeviceAddress(UInt64(0));
}

impl From<UInt64> for DeviceAddress {
    fn from(v: UInt64) -> Self {
        Self(v)
    }
}
impl From<u64> for DeviceAddress {
    fn from(v: u64) -> Self {
        Self(UInt64::from(v))
    }
}

impl IsDeviceAddress for DeviceAddress {}

impl<L: LayoutRule> GpuType<L> for DeviceAddress {
    const PRIMITIVE: SlangType = SlangType::UInt64;
    const SLANG_NAME: &'static str = "uint64_t";
    const SIZE: usize = 8;
    const ALIGN: usize = 8;
}

unsafe impl<L: LayoutRule> GpuPod<L> for DeviceAddress {}

#[macro_export]
macro_rules! device_address_newtype {
    ($name:ident, $slang_name:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name(pub(crate) DeviceAddress);

        impl $name {
            #[inline(always)]
            pub fn into_inner(self) -> DeviceAddress {
                self.0
            }
            #[inline(always)]
            pub fn raw(self) -> u64 {
                self.0.raw()
            }
            pub const NULL: $name = $name(DeviceAddress::NULL);
        }

        impl From<DeviceAddress> for $name {
            fn from(v: DeviceAddress) -> Self {
                Self(v)
            }
        }

        impl IsDeviceAddress for $name {}

        impl<L: LayoutRule> GpuType<L> for $name {
            const PRIMITIVE: SlangType = SlangType::UInt64;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = 8;
            const ALIGN: usize = 8;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<L: LayoutRule> GpuPod<L> for $name {}
    };
}

pub trait DescriptorIndex {
    /// u32::MAX is reserved as the invalid/unpopulated sentinel, consistent with common Vulkan
    /// bindless conventions.
    const INVALID_RAW: u32 = u32::MAX;
}

/// Descriptor index base types
///
/// All four concrete types share identical layout: uint32_t, 4 bytes, 4-byte aligned. Type identity
/// is carried solely by the struct name, which matches the corresponding Slang struct for
/// introspection agreement.
macro_rules! descriptor_base {
    ($name:ident, $slang_name:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name(pub(crate) UInt32);

        impl $name {
            pub const INVALID: $name = $name(UInt32(u32::MAX));

            #[inline(always)]
            pub fn new(index: u32) -> Self {
                Self(UInt32(index))
            }

            #[inline(always)]
            pub fn raw(self) -> u32 {
                self.0.into_inner()
            }

            #[inline(always)]
            pub fn is_valid(self) -> bool {
                self.0.into_inner() != u32::MAX
            }
        }

        impl DescriptorIndex for $name {}

        impl<L: LayoutRule> GpuType<L> for $name {
            const PRIMITIVE: SlangType = SlangType::UInt32;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = 4;
            const ALIGN: usize = 4;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<L: LayoutRule> GpuPod<L> for $name {}
    };
}

descriptor_base!(SampledImageIdx, "SampledImageIdx");
descriptor_base!(StorageImageIdx, "StorageImageIdx");
descriptor_base!(UniformBufferIdx, "UniformBufferIdx");
descriptor_base!(StorageBufferIdx, "StorageBufferIdx");

/// Wraps one of the four concrete descriptor index types with a project-
/// specific name. The inner type must be one of the four base descriptor
/// types — that bound is the witness enforced at the impl site.
/// ```skip
/// descriptor_newtype!(ShadowMapIdx, SampledImageIdx, "ShadowMapIdx");
/// descriptor_newtype!(GBufferAlbedoIdx, SampledImageIdx, "GBufferAlbedoIdx");
/// descriptor_newtype!(SkinnedMeshBufferIdx, StorageBufferIdx, "SkinnedMeshBufferIdx");
/// ```
#[macro_export]
macro_rules! descriptor_newtype {
    ($name:ident, $inner:ty, $slang_name:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $inner);

        impl $name {
            pub const INVALID: $name = $name(<$inner>::INVALID);

            #[inline(always)]
            pub fn new(index: u32) -> Self {
                Self(<$inner>::new(index))
            }

            #[inline(always)]
            pub fn raw(self) -> u32 {
                self.0.raw()
            }

            #[inline(always)]
            pub fn is_valid(self) -> bool {
                self.0.is_valid()
            }

            #[inline(always)]
            pub fn into_inner(self) -> $inner {
                self.0
            }
        }

        impl From<$inner> for $name {
            #[inline(always)]
            fn from(v: $inner) -> Self {
                Self(v)
            }
        }

        impl DescriptorIndex for $name {}

        impl<L: LayoutRule> GpuType<L> for $name
        where
            $inner: GpuType<L>, // witness — inner must be a registered descriptor type
        {
            const PRIMITIVE: SlangType = <$inner as GpuType<L>>::PRIMITIVE;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = 4;
            const ALIGN: usize = 4;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<L: LayoutRule> GpuPod<L> for $name where $inner: GpuType<L> {}
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    // Does a primitive satisfy GpuPod at all?
    fn assert_gpu_pod<L: LayoutRule, T: GpuPod<L>>() {}
    fn assert_gpu_type<L: LayoutRule, T: GpuType<L>>() {}

    #[test]
    fn float32_is_gpu_pod_scalar() {
        assert_gpu_pod::<ScalarLayout, Float32>();
    }

    #[test]
    fn newtype_satisfies_gpu_type() {
        slang_newtype!(Hotness, Float32, "Hotness");
        slang_newtype!(JointIndex, UInt16, "JointIndex");

        assert_gpu_type::<ScalarLayout, Hotness>();
        assert_gpu_type::<ScalarLayout, JointIndex>();

        // Constants forward correctly from inner type
        assert_eq!(
            <Hotness as GpuType<ScalarLayout>>::PRIMITIVE,
            SlangType::Float32
        );
        assert_eq!(<Hotness as GpuType<ScalarLayout>>::SIZE, 4);
        assert_eq!(<Hotness as GpuType<ScalarLayout>>::ALIGN, 4);

        assert_eq!(
            <JointIndex as GpuType<ScalarLayout>>::PRIMITIVE,
            SlangType::UInt16
        );
        assert_eq!(<JointIndex as GpuType<ScalarLayout>>::SIZE, 2);
        assert_eq!(<JointIndex as GpuType<ScalarLayout>>::ALIGN, 2);
    }

    #[test]
    fn from_base_into_wrapper() {
        let _f: Float32 = Float32::from(1.0f32);
        let _u: UInt16 = UInt16::from(42u16);
        let _b: Bool = Bool::from(true);
        let _h: Float16 = Float16::from(half::f16::from_f32(1.0));

        slang_newtype!(Hotness, Float32, "Hotness");
        let _t: Hotness = Hotness::from(Float32::from(98.6f32));
    }

    #[test]
    fn type_identity_constants() {
        assert_eq!(<Float32 as GpuType<ScalarLayout>>::SLANG_NAME, "float");
        assert_eq!(<Float16 as GpuType<ScalarLayout>>::SLANG_NAME, "float16_t");
        assert_eq!(<UInt32 as GpuType<ScalarLayout>>::SLANG_NAME, "uint32_t");
        assert_eq!(<Bool as GpuType<ScalarLayout>>::SLANG_NAME, "bool");
        assert_eq!(<Bool as GpuType<ScalarLayout>>::SIZE, 4);
        assert_eq!(<Bool as GpuType<ScalarLayout>>::ALIGN, 4);

        slang_newtype!(Hotness, Float32, "Hotness");
        assert_eq!(<Hotness as GpuType<ScalarLayout>>::SLANG_NAME, "Hotness");
        assert_eq!(
            <Hotness as GpuType<ScalarLayout>>::PRIMITIVE,
            SlangType::Float32
        );
    }

    #[test]
    fn integer_eq_agrees_with_inner() {
        assert_eq!(Int32::from(42), Int32::from(42));
        assert_ne!(Int32::from(1), Int32::from(2));
        assert_eq!(UInt64::from(0u64), UInt64::from(0u64));
    }
}
