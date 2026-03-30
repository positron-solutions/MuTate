// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

// # Mutate-Lib
//
// Provides semantic newtype wrappers around raw `uint` descriptor indices,
// one family per descriptor slot in the bindless table.
//
// ## Slot map (mirrors descriptors.rs)
//
//   Slot 0  SAMPLER               g_samplers[], g_comparison_samplers[]
//   Slot 1  (reserved — combined image+sampler, unused)
//   Slot 2  SAMPLED_IMAGE         (see note — no generic global array)
//   Slot 3  STORAGE_IMAGE         (see note — no generic global array)
//   Slot 4  UNIFORM_BUFFER        g_uniform_buffers[]
//   Slot 5  STORAGE_BUFFER        g_storage_buffers[]
//   Slot 6  UNIFORM_TEXEL_BUFFER  g_uniform_texel_buffers[]
//   Slot 7  STORAGE_TEXEL_BUFFER  g_storage_texel_buffers[]
//   Slot 8  (unknown/reserved)
//   Slot 9  ACCEL_STRUCTURE       (disabled — awaiting extension support in Rust)
//
// One of the challenges is that each slot is a family and the indexes are
// heterogeneous.  A descriptor family maps to a set of compatible Slang
// resource types, and many resource types are themselves generic over its
// element type T. For example, vk::DescriptorType::STORAGE_BUFFER includes
// several distinct Slang storage buffer types, each parameterized by a
// different T.
//
// This means there are three dimensions of type to consider:
//
// - The index to get the correct array, which decides the broad type
// - The non-generic type component of the Slang resource, such as Buffer or
//   Texture2D
// - The generic parameter T for resources that are generic, such as
//   StructuredBuffer<T>.  Not all resources are generic!
//
// Another challenge is that the global arrays we are indexing into via
// descriptor bindings like [[vk::binding(6, 0)]] must be given a concrete type
// for access.  The way we handle this is to use the least specific concrete
// types like ByteAddressBuffer and then we cast the Load to the specific T we
// want.  This T must agree with the Buffer type provided in Rust, ensuring
// host-GPU agreement.
//
// Because there are three dimensions of type, we have different levels of
// guarantees we can provide.
//
// - correct slot (descriptor family)
// - correct Slang resource type
// - correct generic parameter T (for resources that are generic)
//
// In order to write a new Slang type, you can use a distinct newtype or a
// semantic alias, such as to specify T or just clear up semantic usage beyond
// the field name.  If you need to make the indexing concrete, you will also
// need to define the concrete global index, re-using the correct slot.
//
// In the case of many types like image samplers, a generic index does not
// exist.  In these cases, to create newtypes, you will also need to declare the
// global array binding with a concrete type that the newtype can index into.

// XXX Yet unknown implications of newtype vs alias for reflection and host-GPU
// agreement checks.
// XXX Slang module macros for creating newtypes and aliases do not yet know
// enough to guarantee that this module is correct.  The ends are still growing
// towards each other.
// XXX Probably better to just provide a thin alias for indexes that offer no
// help at getting resources.
// XXX Not clear yet how array bounds will push and pull with the host-GPU
// checks and Rust marshaling newtypes.

// ---------------------------------------------------------------------------
// Slot 0 — SAMPLER
// Overloaded on the same binding: plain and comparison samplers share slot 0.
// ---------------------------------------------------------------------------

[[vk::binding(0, 0)]] SamplerState           g_samplers[];
[[vk::binding(0, 0)]] SamplerComparisonState g_comparison_samplers[];

struct SamplerIdx {
    uint inner;
    SamplerState get() { return g_samplers[inner]; }
};

struct CompSamplerIdx {
    uint inner;
    SamplerComparisonState get() { return g_comparison_samplers[inner]; }
};

// ---------------------------------------------------------------------------
// Slots 2–3 — SAMPLED_IMAGE / STORAGE_IMAGE
//
// No generic global arrays are declared here; see module header for rationale.
// These aliases carry the index only.  Shaders that need to read/write images
// declare their own typed binding and index into it using either this generic
// handle or by creating their own newtypes.
// ---------------------------------------------------------------------------

typealias SampledImageIdx = uint;
typealias StorageImageIdx = uint;

// ---------------------------------------------------------------------------
// Slot 4 — UNIFORM_BUFFER
// Read-only byte-address buffer.  T resolved at call site.
// ---------------------------------------------------------------------------

// On slot 4: ConstantBuffer<T> cannot be declared without a T, making it
// incompatible with a heterogeneous bindless pool. ByteAddressBuffer is used
// instead. If you need true UBO caching behavior for a specific type, declare a
// ConstantBuffer<T> and new global array committed to a specific type.
[[vk::binding(4, 0)]] ByteAddressBuffer g_uniform_buffers[];

/// Un-typed index into the uniform-buffer slot.
/// Prefer TypedUboIdx<T> when T is known at the declaration site.
struct UboIdx {
    uint inner;

    T get<T>() {
        return g_uniform_buffers[inner].Load<T>(0);
    }

    T get_element<T>(uint element_index) {
        return g_uniform_buffers[inner].Load<T>(element_index * sizeof(T));
    }
};

/// Typed index — T is baked at declaration; get() needs no annotation. For most
/// convenient ergonomics, use a type alias to fix T.
///
/// ```slang
///   typealias ParamUbo = TypedUboIdx<MyParams>;
/// ```
struct TypedUboIdx<T> {
    uint inner;

    T get() {
        return g_uniform_buffers[inner].Load<T>(0);
    }

    T get_element(uint element_index) {
        return g_uniform_buffers[inner].Load<T>(element_index * sizeof(T));
    }
};

// ---------------------------------------------------------------------------
// Slot 5 — STORAGE_BUFFER
// Read-write byte-address buffer.  T resolved at call site.
// ---------------------------------------------------------------------------

[[vk::binding(5, 0)]] RWByteAddressBuffer g_storage_buffers[];

/// Untyped index into the storage-buffer slot.
/// Prefer TypedSsboIdx<T> when T is known at the declaration site.
struct SsboIdx {
    uint inner;

    T get<T>() {
        return g_storage_buffers[inner].Load<T>(0);
    }

    T get_element<T>(uint element_index) {
        return g_storage_buffers[inner].Load<T>(element_index * sizeof(T));
    }

    void set<T>(T value) {
        g_storage_buffers[inner].Store<T>(0, value);
    }

    void set_element<T>( uint element_index, T value) {
        g_storage_buffers[inner].Store<T>(element_index * sizeof(T), value);
    }
};

/// Typed index — T is baked at declaration; get/set needs no T. For most
/// convenient ergonomics, use a type alias to fix T.
///
/// ```slang
///   typealias SpectrumSsbo = TypedSsboIdx<SpectrumSample>;
/// ```
struct TypedSsboIdx<T> {
    uint inner;

    T get() {
        return g_storage_buffers[inner].Load<T>(0);
    }

    T get_element(uint element_index) {
        return g_storage_buffers[inner].Load<T>(element_index * sizeof(T));
    }

    void set(T value) {
        g_storage_buffers[inner].Store<T>(0, value);
    }

    void set_element(uint element_index, T value) {
        g_storage_buffers[inner].Store<T>(element_index * sizeof(T), value);
    }
};

// ---------------------------------------------------------------------------
// Slot 6 — UNIFORM_TEXEL_BUFFER
//
// In Vulkan, uniform texel buffers are VkBufferViews bound as
// UNIFORM_TEXEL_BUFFER. Slang exposes these as Buffer<T> (typed, read-only).
// We cannot declare a single untyped global array for the same reason as
// images: Buffer<T> requires committing to T at the declaration site.
//
// If your shaders only ever use one element type (e.g. float4), declare:
//
//   [[vk::binding(6, 0)]] Buffer<float4> g_uniform_texel_buffers[];
//
// and add a TypedUtboIdx<float4> alias. Until a concrete T is settled, this
// slot's global array and index types are left here as a scaffold.
// ---------------------------------------------------------------------------

// [[vk::binding(6, 0)]] Buffer<??> g_uniform_texel_buffers[];  // T unknown

typealias UniformTexelBufferIdx = uint;

// ---------------------------------------------------------------------------
// Slot 7 — STORAGE_TEXEL_BUFFER
//
// Same situation as slot 6: RWBuffer<T> requires a concrete T.
// Scaffold only until T is decided.
// ---------------------------------------------------------------------------

// [[vk::binding(7, 0)]] RWBuffer<??> g_storage_texel_buffers[];  // T unknown

typealias StorageTexelBufferIdx = uint;

// ---------------------------------------------------------------------------
// Slot 9 — ACCELERATION_STRUCTURE
// Disabled: awaiting runtime extension dependency resolution on the Rust side.
// ---------------------------------------------------------------------------

// [[vk::binding(9, 0)]] RaytracingAccelerationStructure g_accel_structures[];

// struct AccelStructureIdx {
//     uint inner;
//     RaytracingAccelerationStructure get() { return g_accel_structures[inner]; }
// };
