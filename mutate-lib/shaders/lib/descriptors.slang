/// descriptors.slang
///
/// Typed descriptor index wrappers for a single bindless descriptor set.
///
/// Import this module from any shader that needs bindless resource access:
///
///   import descriptors;
///
/// Slot assignments and the descriptor set index are defined here as the
/// canonical source of truth. Keep them in sync with your Vulkan descriptor
/// set layout creation on the Rust side.
///
/// Wire layout of each index type: 4 bytes, no padding, identical to a bare
/// `uint`. Rust mirror: `DescriptorIndex<SampledImage>` etc.
module descriptors;

// ---------------------------------------------------------------------------
// Slot assignments — edit these to match your VkDescriptorSetLayoutBinding
// order and your DescriptorSetLayout in Rust.
// ---------------------------------------------------------------------------
static const uint kDescriptorSet      = 0;
static const uint kSlotSampledImages  = 0;
static const uint kSlotSamplers       = 1;
static const uint kSlotStorageImages  = 2;
static const uint kSlotUniformBuffers = 3;
static const uint kSlotStorageBuffers = 4;

// ---------------------------------------------------------------------------
// Global descriptor arrays — declared once here, used everywhere.
// ---------------------------------------------------------------------------
[vk::binding(kSlotSampledImages,  kDescriptorSet)] Texture2D                g_sampledImages[];
[vk::binding(kSlotSamplers,       kDescriptorSet)] SamplerState             g_samplers[];
[vk::binding(kSlotStorageImages,  kDescriptorSet)] RWTexture2D              g_storageImages[];
[vk::binding(kSlotUniformBuffers, kDescriptorSet)] ConstantBuffer<uint>     g_uniformBuffers[];
[vk::binding(kSlotStorageBuffers, kDescriptorSet)] RWStructuredBuffer<uint> g_storageBuffers[];

// ---------------------------------------------------------------------------
// Typed index wrappers.
//
// Each is a newtype over `uint`. The only way to resolve an index to a
// resource is through the typed `.get()` accessor — cross-slot usage is a
// compile error.
// ---------------------------------------------------------------------------
public struct SampledImageIdx
{
    public uint index;
    public Texture2D get() { return g_sampledImages[index]; }
};

public struct SamplerIdx
{
    public uint index;
    public SamplerState get() { return g_samplers[index]; }
};

public struct StorageImageIdx
{
    public uint index;
    public RWTexture2D get() { return g_storageImages[index]; }
};

public struct UniformBufferIdx
{
    public uint index;
    public ConstantBuffer<uint> get() { return g_uniformBuffers[index]; }
};

public struct StorageBufferIdx
{
    public uint index;
    public RWStructuredBuffer<uint> get() { return g_storageBuffers[index]; }
};

// ---------------------------------------------------------------------------
// Convenience pair for the common sample-a-texture case.
// Packs as two uints; matches a [SampledImageIdx, SamplerIdx] field pair
// in push constants.
//
// NOTE: We do samplers differently in practice — this is here for
// completeness but may not see much use.
// ---------------------------------------------------------------------------
public struct TextureAndSamplerIdx
{
    public SampledImageIdx texture;
    public SamplerIdx      sampler;
};
