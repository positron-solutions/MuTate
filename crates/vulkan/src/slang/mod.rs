// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Slang
//!
//! Make Rust types that correspond to Slang types and can be used in compile-time pipeline-shader
//! agreement checking.  Macros are provided for creating new-type wrappers around existing Slang
//! types, adding further semantic type safety.
//!
//! ## Scalar & Vector Types
//!
//! Built-in Slang types are provided already:
//!
//! ```rust
//! use mutate_vulkan::prelude::*;
//!
//! let _i: Int = Int::from(0_i32);
//! let _u: UInt = UInt::from(0_u32);
//! let _f: Float = Float::from(0.0_f32);
//! let _b: Bool = Bool::from(false);
//!
//! let _i: Int8 = Int8::from(0_i8);
//! let _u: UInt16 = UInt16::from(0_u16);
//! let _d: Double = Double::from(0_f64);
//! ```
//!
//! ### Ergonomic, Safe Conversions
//!
//! ```
//! use mutate_vulkan::prelude::*;
//!
//! let n = UInt::from(42_u32);
//!
//! let n: UInt = 42_u32.into();
//!
//! // Recover the raw value when you need to cross back into plain Rust.
//! let raw: u32 = n.into_inner();
//!
//! assert_eq!(raw, 42_u32);
//!
//! // Deref as the base type
//! fn foo(i: &u32) {}
//!
//! foo(&n);
//! ```
//!
//! ### Semantic Type Wrappers
//!
//! In cases where a Slang struct wraps a primitive with a domain name:
//!
//! ```slang
//! struct Temperature { int32_t value; }
//! ```
//!
//! Use `slang_newtype!` to mirror that wrapper in Rust.  The two wrapper types are distinct from
//! each other and from the base type — you cannot accidentally pass a `Pressure` where a
//! `Temperature` is expected — but you can always recover the inner value when you need to cross
//! back into plain Rust.
//!
//! ```rust
//! use mutate_vulkan::prelude::*;
//!
//! slang_newtype!(Temperature, Int, "Temperature");
//! slang_newtype!(Pressure,    Int, "Pressure");
//!
//! fn apply_temperature(_t: Temperature) {}
//!
//! let t = Temperature(Int::from(300));
//! let p = Pressure(Int::from(101));
//!
//! apply_temperature(t);
//! // apply_temperature(p);  // ← compile error: expected Temperature, found Pressure
//!
//! // Recover the base Int, then the raw i32, when you need to cross the boundary.
//! let raw: i32 = t.into_inner().into_inner();
//! assert_eq!(raw, 300);
//! ```
//!
//! Semantic safety in Slang meets semantic safety in Rust.  No accidentally mishandling of values
//! that share base types.
//!
//! ## Layout
//!
//! Using Slang reflection, we obtain the layout and type of each field of a structure such as:
//!
//! ```slang
//! struct Particle {
//!     float3 position;   // 12 bytes under scalar layout
//!     float  mass;       // 4 bytes
//! }
//! ```
//!
//! We add a compile-time check to ensure that Rust types being written for Slang agree in the name
//! (for newtypes) and base type (`u32` matches `uint32` etc), byte by byte, padding included.
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
//! **Currently only scalar block layout is supported!** but the code was written generically for
//! std430 and std140 in case you find a use for them.
//!
//! ## Buffer Device Address
//!
//! A `DeviceAddress` type has been provided for handling pointers without cutting our hands on bare
//! u64. Use `device_address_newtype!` to create type-safe derivatives.
//!
//! ```rust
//! use mutate_vulkan::prelude::*;
//!
//! device_address_newtype!(VertexBufferPtr, "VertexBufferPtr");
//!
//! let null = VertexBufferPtr::NULL;  // NULL sentinel is address 0
//! assert_eq!(null.raw(), 0);
//!
//! let addr = VertexBufferPtr::from(DeviceAddress::from(0xDEAD_0000_u64));
//! assert_eq!(addr.raw(), 0xDEAD_0000_u64);
//! ```
//!
//! ## Descriptor Indexes
//!
//! We use a single descriptor table with known slots.  We use a Slang library to define type-safe
//! indexes into that table.  On the Rust side, we provide both the base wrapped indexes and a
//! `descriptor_newtype!` macro for creating type-safe semantic wrappers, meaning valid indexes that
//! must point to a resource with the correct inner type.
//!
//! ```rust
//! use mutate_vulkan::prelude::*;
//!
//! descriptor_newtype!(ShadowMapIdx,      SampledImageIdx, "ShadowMapIdx");
//! descriptor_newtype!(GBufferAlbedoIdx,  SampledImageIdx, "GBufferAlbedoIdx");
//! descriptor_newtype!(SkinnedMeshBufIdx, SsboIdx,         "SkinnedMeshBufIdx");
//!
//! fn bind_shadow_map(_idx: ShadowMapIdx) {}
//!
//! let shadow = ShadowMapIdx::new(7);
//! assert!(shadow.is_valid());
//! assert_eq!(shadow.raw(), 7);
//!
//! bind_shadow_map(shadow);
//! // bind_shadow_map(GBufferAlbedoIdx::new(7));  // ← compile error: wrong semantic type
//!
//! // The INVALID sentinel round-trips correctly.
//! assert!(!ShadowMapIdx::INVALID.is_valid());
//! assert_eq!(ShadowMapIdx::INVALID.raw(), u32::MAX);
//! ```

// XXX Buffer Device Address destination payload types not implemented
// XXX Writing a descriptor index via push constants etc not yet implemented
// XXX Descriptor indexes and types of the contents at the handle are not yet implemented
// XXX There may be some aliases in Slang
// XXX Remove need for double (or more into) calls
// XXX Newtype for newtypes tests (and define how to interpret this)
// XXX Finish up some Deref implementations and tests
// NEXT we are definitely implementing vectors (ie float3) as leaf types (`GpuPrimitive`) to avoid
// vector complexity infecting calculations for struct fields and enums.

use bytemuck::{Pod, Zeroable};
use half::prelude::*;

pub mod prelude {
    // Layout rules — users name these at call sites and in trait bounds
    pub use super::{DataLayout, Scalar};

    // Core traits — needed for generic bounds and introspection
    pub use super::{GpuPod, GpuPrimitive, GpuScalar, GpuType};

    // Type discrimination — users match/compare PRIMITIVE constants
    pub use super::SlangType;

    // Scalars — the everyday currency of GPU data
    pub use super::{
        Bool, Double, Float, Half, Int, Int16, Int64, Int8, UInt, UInt16, UInt64, UInt8,
    };

    // Pointer type and its trait
    pub use super::{DeviceAddress, IsDeviceAddress};

    // Descriptor index base types and their trait
    pub use super::{DescriptorIndex, SampledImageIdx, SsboIdx, StorageImageIdx, UboIdx};

    pub use crate::descriptor_newtype;
    pub use crate::device_address_newtype;
    pub use crate::slang_newtype;

    // Re-exports for convenience
    pub use bytemuck;
    pub use bytemuck::{Pod, Zeroable};
    // MAYBE may need the whole f16 prelude
    pub use half;
}

mod sealed {
    pub trait Sealed {}
}

/// Only used by `DataLayout` to overcome some const trait features that are not stable yet.
#[derive(Clone, Copy)]
pub enum DataLayoutToken {
    Scalar,
    Std430,
}

pub trait DataLayout: sealed::Sealed {
    /// A token to forward trait implementor type into free functions so they can match on that type
    /// as a form of dynamic dispatch.
    ///
    /// `const` traits are not stabilized, so we can't dispatch `const` methods by type.  Until
    /// someone needs std430 support, this is all hypothetical anyway.
    ///
    /// See the [tracking issue] for more details.
    ///
    /// [tracking issue]: https://github.com/rust-lang/rust/issues/143874
    // ROLL waiting on stabilization above to remove this hack.
    const DATA_LAYOUT: DataLayoutToken;
}

pub struct Scalar;
pub struct Std430; // reserved, not yet implemented

impl sealed::Sealed for Scalar {}
impl sealed::Sealed for Std430 {}
impl DataLayout for Scalar {
    const DATA_LAYOUT: DataLayoutToken = DataLayoutToken::Scalar;
}
impl DataLayout for Std430 {
    const DATA_LAYOUT: DataLayoutToken = DataLayoutToken::Std430;
}

/// Closed enumeration of Slang primitive types.  Every GPU type bottoms out at one of these.  User
/// newtypes and structs carry their own SLANG_NAME but delegate PRIMITIVE downward to the leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlangType {
    Bool,
    Int8,
    Int16,
    Int,
    Int64,
    UInt8,
    UInt16,
    UInt,
    UInt64,
    Half,
    Float,
    Double,
    // XXX Is this actually in use / necessary or vestigal?
    /// Composites — `SLANG_NAME` carries the identity.
    Struct,
}

/// Truly atomic types do not depend on the layout rules and may be encoded more simply.
pub trait GpuScalar {
    /// The irreducible Slang primitive this type reduces to.
    /// Drives wire layout and type-safety checks (u16 vs f16, etc.)
    const PRIMITIVE: SlangType;
    /// The Slang-side name for introspection matching.
    const SLANG_NAME: &'static str;
    /// On scalar types, alignment = size.  As you may expect, the Rust size and Slang sizes are the
    /// same.
    // MAYBE farther downstream, it may be apparent that we can use std::mem::size_of::<T>() without
    // encoding this as an associated const.
    const SIZE: usize;
}

// XXX I don't think this doc comment is correct
/// Marker trait for types whose layout under D is correct as-is —
/// no repacking needed. Requires `bytemuck::Pod` for safe transmute.
///
/// SAFETY: implementor must ensure SIZE/ALIGN match the actual
/// in-memory representation under D.
pub unsafe trait GpuPod<D: DataLayout>: GpuScalar + Pod {}

/// Registers a Rust type as a Slang scalar primitive.
/// Emits a repr(transparent) newtype that:
///   - implements GpuType<D> for all D (primitives are layout-agnostic at scalar level)
///   - implements GpuPod<D> for all D
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

        impl From<$base> for $name {
            #[inline(always)]
            fn from(v: $base) -> Self {
                Self(v)
            }
        }

        impl std::ops::Deref for $name {
            type Target = $base;
            #[inline(always)]
            fn deref(&self) -> &$base {
                &self.0
            }
        }

        impl GpuScalar for $name {
            const PRIMITIVE: SlangType = $variant;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = std::mem::size_of::<$base>();
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<D: DataLayout> GpuPod<D> for $name {}
    };
}

/// Wraps an existing slang_scalar type with a semantic name.
// XXX comment still fresh?
/// The inner type must already implement GpuType<D> — that bound
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

        impl std::ops::Deref for $name {
            type Target = $inner;
            #[inline(always)]
            fn deref(&self) -> &$inner {
                &self.0
            }
        }

        impl GpuScalar for $name
        where
            $inner: GpuScalar, // witness — if inner isn't registered, fails here
        {
            const PRIMITIVE: SlangType = <$inner as GpuScalar>::PRIMITIVE;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = <$inner as GpuScalar>::SIZE;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<D: DataLayout> GpuPod<D> for $name where $inner: GpuScalar {}
    };
}

//
// ```json
// {
//     "name": "cleared_for_takeoff",
//     "type": {
//         "kind": "scalar",
//         "scalarType": "bool"
//     },
//     "binding": {"kind": "uniform", "offset": 8, "size": 4, "elementStride": 0}
// }
// ```
// Size is 4.  SPIR-V probably supports other types, especially since we have byte support enabled
// in features.  By default, we use a 32bit until someone finds a way to default differently.
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

impl std::ops::Deref for Bool {
    type Target = u32;
    #[inline(always)]
    fn deref(&self) -> &u32 {
        &self.0
    }
}

impl GpuScalar for Bool {
    const PRIMITIVE: SlangType = SlangType::Bool;
    const SLANG_NAME: &'static str = "bool";
    const SIZE: usize = 4; // Slang bool is 4 bytes on GPU
}

unsafe impl<D: DataLayout> GpuPod<D> for Bool {}

// https://shader-slang.org/slang/user-guide/conventional-features.html#types
slang_scalar!(Int8, i8, SlangType::Int8, "int8_t");
slang_scalar!(Int16, i16, SlangType::Int16, "int16_t");
slang_scalar!(Int, i32, SlangType::Int, "int");
slang_scalar!(Int64, i64, SlangType::Int64, "int64_t");

slang_scalar!(UInt8, u8, SlangType::UInt8, "uint8_t");
slang_scalar!(UInt16, u16, SlangType::UInt16, "uint16_t");
slang_scalar!(UInt, u32, SlangType::UInt, "uint");
slang_scalar!(UInt64, u64, SlangType::UInt64, "uint64_t");

// Integer types have total equality; floats do not (NaN != NaN).
impl Eq for Int8 {}
impl Eq for Int16 {}
impl Eq for Int {}
impl Eq for Int64 {}
impl Eq for UInt8 {}
impl Eq for UInt16 {}
impl Eq for UInt {}
impl Eq for UInt64 {}

slang_scalar!(Half, f16, SlangType::Half, "half");
slang_scalar!(Float, f32, SlangType::Float, "float");
slang_scalar!(Double, f64, SlangType::Double, "double");

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

impl GpuScalar for DeviceAddress {
    const PRIMITIVE: SlangType = SlangType::UInt64;
    const SLANG_NAME: &'static str = "uint64_t";
    const SIZE: usize = 8;
}

unsafe impl<D: DataLayout> GpuPod<D> for DeviceAddress {}

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

        impl GpuScalar for $name {
            const PRIMITIVE: SlangType = SlangType::UInt64;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = 8;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<D: DataLayout> GpuPod<D> for $name {}
    };
}

pub trait DescriptorIndex {
    /// u32::MAX is reserved as the invalid/unpopulated sentinel, consistent with common Vulkan
    /// bindless conventions.
    // XXX Check if INVALID is a good constant name.  NULL wouldn't make sense because this isn't
    // zero, but do people use "INVALID" in real semantics?  Unlike pointers, we use the zero
    // descriptor index.
    const INVALID: u32 = u32::MAX;
}

/// Descriptor index base types
///
/// All four concrete types share identical layout: uint32, 4 bytes, 4-byte aligned. Type identity
/// is carried solely by the struct name, which matches the corresponding Slang struct for
/// introspection agreement.
macro_rules! descriptor_base {
    ($name:ident, $slang_name:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $name(pub(crate) UInt);

        impl $name {
            pub const INVALID: $name = $name(UInt(u32::MAX));

            #[inline(always)]
            pub fn new(index: u32) -> Self {
                Self(UInt(index))
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

        impl GpuScalar for $name {
            const PRIMITIVE: SlangType = SlangType::UInt;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = 4;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<D: DataLayout> GpuPod<D> for $name {}
    };
}

descriptor_base!(SampledImageIdx, "SampledImageIdx");
descriptor_base!(StorageImageIdx, "StorageImageIdx");
descriptor_base!(SamplerIdx, "SamplerIdx");
descriptor_base!(UboIdx, "UboIdx");
descriptor_base!(SsboIdx, "SsboIdx");

/// Wraps one of the four concrete descriptor index types with a project-
/// specific name. The inner type must be one of the four base descriptor
/// types — that bound is the witness enforced at the impl site.
/// ```skip
/// descriptor_newtype!(ShadowMapIdx, SampledImageIdx, "ShadowMapIdx");
/// descriptor_newtype!(GBufferAlbedoIdx, SampledImageIdx, "GBufferAlbedoIdx");
/// descriptor_newtype!(SkinnedMeshBufferIdx, SsboIdx, "SkinnedMeshBufferIdx");
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

        impl GpuScalar for $name {
            const PRIMITIVE: SlangType = SlangType::UInt;
            const SLANG_NAME: &'static str = $slang_name;
            const SIZE: usize = 4;
        }

        // DEBT bytemuck
        unsafe impl $crate::__bytemuck::Zeroable for $name {}
        unsafe impl $crate::__bytemuck::Pod for $name {}

        unsafe impl<D: DataLayout> GpuPod<D> for $name where $inner: GpuScalar {}
    };
}

/// Types that are built into Slang and have unique alignment requirements but are composed of
/// scalars and considered "leaf" types for packing / marshaling purposes.
pub trait GpuPrimitive<D: DataLayout> {
    /// The irreducible Slang primitives this type reduces to.
    const PRIMITIVE: SlangType;
    /// The Slang-side name for introspection matching.
    const SLANG_NAME: &'static str;
    /// The size is usually equal to the comprising scalars.
    const SIZE: usize;
    /// Vector types have layout-dependent alignment.  See float3 on std140 etc.
    const ALIGN: usize;
}

// All scalars are primitives
impl<T: GpuScalar, D: DataLayout> GpuPrimitive<D> for T {
    const PRIMITIVE: SlangType = T::PRIMITIVE;
    const SLANG_NAME: &'static str = T::SLANG_NAME;
    const SIZE: usize = T::SIZE;
    const ALIGN: usize = T::SIZE; // scalars: align == size
}

// XXX We might want to go ahead and implement at least float2-float4 in order to exercise the
// GpuPrimitive code.

/// A type that can describe its own GPU layout as a FieldNode.
///
/// Leaf types (scalars, newtypes, device addresses, descriptor indices) implement this by
/// projecting their `GpuScalar` and `GpuPrimitive` consts into a `FieldNode::Leaf`.
///
/// Composite types (structs, enums — via proc macro) implement this by returning
/// a FieldNode::Tree whose children are their fields' FieldNodes.
///
pub trait GpuType<D: DataLayout> {
    const FIELD_NODE: FieldNode;
}

impl<T: GpuPrimitive<D>, D: DataLayout> GpuType<D> for T {
    const FIELD_NODE: FieldNode = FieldNode::Leaf(FieldDesc {
        primitive: T::PRIMITIVE,
        size: T::SIZE,
        align: T::ALIGN,
        slang_name: T::SLANG_NAME,
    });
}

/// Concrete, type-erased `GpuType` data.  Checks using `FieldDesc` are generic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldDesc {
    pub primitive: SlangType,
    pub size: usize,
    pub align: usize,
    /// The name of the Slang primitive type.
    /// - **primitives**:   matches the Slang builtin name ("float16_t" etc.)
    /// - **newtypes**:     the Slang struct name ("Temperature" etc.)
    /// - **structs**:      the Slang struct name
    pub slang_name: &'static str,
}

pub enum FieldNode {
    /// Terminal — a primitive or transparent newtype over one.
    Leaf(FieldDesc),
    /// Composite — recurse into its own field list.
    Tree {
        slang_name: &'static str,
        fields: &'static [FieldNode],
    },
}

// XXX region_count must descend into the tree and return a distinct region whenever there
// are padding bytes.  If there's no padding, we can just keep growing the region in
// depth-first traversal, resulting in fewer regions for the compiler to reason about.
const fn flatten_node(
    node: &FieldNode,
    rule: DataLayoutToken,
    mut ctx: FlattenCtx,
    out: &mut [PackRegion],
) -> FlattenCtx {
    match node {
        FieldNode::Leaf(d) => {
            out[ctx.idx] = PackRegion {
                src_offset: ctx.src,
                dst_offset: ctx.dst,
                size: d.size,
            };
            ctx.src += d.size;
            ctx.dst += d.size;
            ctx.idx += 1;
            ctx
        }
        FieldNode::Tree { fields, .. } => {
            let mut i = 0;
            while i < fields.len() {
                let align = node_align(&fields[i], rule);
                ctx.dst = align_up(ctx.dst, align);

                // The recursion
                ctx = flatten_node(&fields[i], rule, ctx, out);

                i += 1;
            }
            ctx
        }
    }
}

/// A contiguous byte range that can be copied verbatim from host memory to GPU memory.
/// Where layout padding would appear, the region is split so no padding bytes are copied.
#[derive(Debug, Clone, Copy)]
pub struct PackRegion {
    pub src_offset: usize,
    pub dst_offset: usize,
    pub size: usize,
}

/// Count the number of contiguous copy regions needed to pack `node` under `rule`.
/// Regions split at every padding boundary introduced by the layout rule.
pub const fn region_count(node: &FieldNode) -> usize {
    match node {
        FieldNode::Leaf(_) => 1,
        FieldNode::Tree { fields, .. } => {
            let mut count = 0;
            let mut i = 0;
            while i < fields.len() {
                count += region_count(&fields[i]);
                i += 1;
            }
            count
        }
    }
}

/// Packed size of `node` under `rule`.
pub const fn packed_size(node: &FieldNode, rule: DataLayoutToken) -> usize {
    match node {
        FieldNode::Leaf(d) => d.size,
        FieldNode::Tree { fields, .. } => match rule {
            DataLayoutToken::Scalar => {
                // Scalar layout: no inter-field padding.
                let mut size = 0;
                let mut i = 0;
                while i < fields.len() {
                    size += packed_size(&fields[i], rule);
                    i += 1;
                }
                size
            }
            DataLayoutToken::Std430 => {
                // Std430: each field aligned to its own alignment requirement.
                let mut offset = 0;
                let mut i = 0;
                while i < fields.len() {
                    let align = node_align(&fields[i], rule);
                    offset = align_up(offset, align);
                    offset += packed_size(&fields[i], rule);
                    i += 1;
                }
                // Struct size rounds up to struct alignment.
                let align = tree_align(fields, rule);
                align_up(offset, align)
            }
        },
    }
}

/// Alignment of `node` under `rule`.
pub const fn node_align(node: &FieldNode, rule: DataLayoutToken) -> usize {
    match node {
        FieldNode::Leaf(d) => d.align,
        FieldNode::Tree { fields, .. } => tree_align(fields, rule),
    }
}

/// Alignment of a struct (max of field alignments) under `rule`.
pub const fn tree_align(fields: &[FieldNode], rule: DataLayoutToken) -> usize {
    let mut max_align = 1;
    let mut i = 0;
    while i < fields.len() {
        let a = node_align(&fields[i], rule);
        if a > max_align {
            max_align = a;
        }
        i += 1;
    }
    max_align
}

struct FlattenCtx {
    src: usize,
    dst: usize,
    idx: usize,
}

/// Flatten the `FieldNode` tree into a fixed-length `[PackRegion; N]` array that can be
/// evaluated entirely at compile time and iterated at runtime for the actual byte copies.
///
/// `N` must equal `region_count(node)` — the caller is responsible for this invariant
/// (enforced by the `Pack<D>` blanket impl via associated const).
pub const fn flatten_pack_regions<const N: usize>(
    node: &FieldNode,
    rule: DataLayoutToken,
) -> [PackRegion; N] {
    // Start at zero and then flatten recursively
    let mut out = [PackRegion {
        src_offset: 0,
        dst_offset: 0,
        size: 0,
    }; N];
    let mut ctx = FlattenCtx {
        src: 0,
        dst: 0,
        idx: 0,
    };
    ctx = flatten_node(node, rule, ctx, &mut out);
    let _ = ctx;
    out
}

const fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

pub trait Pack<D: DataLayout> {
    /// The packed size of this type under D, in bytes.
    const PACKED_SIZE: usize;
    const PLAN: PackPlan;

    /// Packs `self` into the destination byte slice according to the data layout `D`.
    ///
    /// `T` is copied region-by-region from its native (host) memory layout into the layout required
    /// by `D` (e.g. std140, std430, or a custom GPU layout).  The region map is computed entirely
    /// at compile time via `const` evaluation and iterated at runtime for the actual byte copies.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is shorter than [`Pack::packed_size()`]. Use `packed_size` (a const )
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut buf = [0u8; <MyType as Pack<Scalar>>::PACKED_SIZE];
    /// my_value.pack_into(&mut buf);
    /// ```
    fn pack_into(&self, dst: &mut [u8]);
}

/// ROLL again, a little better const generic support can remove some extra hoops we shouldn't need
/// to jump through:
/// https://github.com/rust-lang/rust/issues/132980
/// https://github.com/rust-lang/rust/issues/143874
// Can't even fit eight buffer device addresses into a 128-byte push constant with padding, so 16
// contiguous regions covers quite a bit of types.  Doing SoA means we just shouldn't have ragged
// structures with 16 regions of padding.  This small value is realistically kind of gigantic.  The
// workaround is clear, use indirection, SoA, and types that pack better.
pub const MAX_PACK_REGIONS: usize = 16;

#[derive(Clone, Copy)]
pub struct PackPlan {
    pub regions: [PackRegion; MAX_PACK_REGIONS],
    pub count: usize,
}

/// Build a PackPlan entirely at compile time.
pub const fn make_pack_plan(node: &FieldNode, rule: DataLayoutToken) -> PackPlan {
    let count = region_count(node);
    // Validate eagerly — the const evaluator surfaces this as a compile error.
    assert!(count <= MAX_PACK_REGIONS, "struct exceeds MAX_PACK_REGIONS");
    let regions = flatten_pack_regions::<MAX_PACK_REGIONS>(node, rule);
    PackPlan { regions, count }
}

impl<D: DataLayout, T: GpuType<D> + Pod> Pack<D> for T {
    const PACKED_SIZE: usize = packed_size(&T::FIELD_NODE, D::DATA_LAYOUT);
    const PLAN: PackPlan = make_pack_plan(&T::FIELD_NODE, D::DATA_LAYOUT);

    fn pack_into(&self, dst: &mut [u8]) {
        assert!(dst.len() >= Self::PACKED_SIZE);
        let src = bytemuck::bytes_of(self);
        let mut i = 0;
        while i < Self::PLAN.count {
            let r = &Self::PLAN.regions[i];
            dst[r.dst_offset..][..r.size].copy_from_slice(&src[r.src_offset..][..r.size]);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Field test vectors from actual compiler output.

    #[test]
    fn bool_is_four_bytes() {
        // From Scalar hardcode
        assert_eq!(<Bool as GpuScalar>::SIZE, 4);
        // Using the Pack trait
        assert_eq!(<Bool as Pack<Scalar>>::PACKED_SIZE, 4,);
    }

    #[test]
    fn device_address_layout() {
        assert_eq!(<DeviceAddress as GpuScalar>::SIZE, 8);
    }

    #[test]
    fn descriptor_index_invalid_sentinel() {
        assert!(!SampledImageIdx::new(0).is_valid() == false);
        assert!(!SampledImageIdx::INVALID.is_valid());
        assert_eq!(SampledImageIdx::INVALID.raw(), u32::MAX);
    }

    #[test]
    fn pack_scalar_round_trip() {
        let v = UInt::from(0xDEAD_BEEF_u32);
        let mut buf = [0u8; 4];
        <UInt as Pack<Scalar>>::pack_into(&v, &mut buf);
        assert_eq!(buf, 0xDEAD_BEEF_u32.to_ne_bytes());

        let mut buf = [0u8; <Bool as Pack<Scalar>>::PACKED_SIZE];
        let v = Bool::TRUE;
        <Bool as Pack<Scalar>>::pack_into(&v, &mut buf);
        assert_eq!(buf, [0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn pack_int64_round_trip() {
        let v = Int64::from(-1_i64);
        let mut buf = [0u8; 8];
        <Int64 as Pack<Scalar>>::pack_into(&v, &mut buf);
        assert_eq!(buf, (-1_i64).to_ne_bytes());
    }
}
