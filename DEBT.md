# Debt

This is a record of "crimes" and the plans to later un-crime them.  Debt specifically covers crimes that cost us more later the longer we keep doing them and the rationale to keep doing them for now.

# Currently Paying Down

Crimes where the solution has been chosen and all new work should burn down existing problems.  Separate any distinct crimes that emerge into new debt.

## Lifetime Alignment

Prototype code is not yet attempting to build structs to gather related lifetimes or enable multiple window setups.  It should.  Boundaries emerging:

- Device, instance, other detected & configured choices
- The window dependent objects such as swapchain images
- Compositing buffers
- Pipelines and their exclusive resources

User setting updates, dynamic scripting, and generation will all as usual require a lot of re-creation and re-allocation that can share duty with teardown, destruction being the first step of re-creation.

## Memory Management

Dynamic usage of allocated memory is likely unavoidable.  Several tools are like VMA bindings or the gpu-allocator crate are being looked at.  Expectations are that memory usage will be relatively low but less predictable due to generation and scripting.  Bindless rendering will certainly be coupled to the memory use strategy.  The tradeoffs of existing approaches are not clear yet, but the need to manage a pool and dependent addresses does suggest more rather than less work will pay off.

# Charging Interest

Each element includes two parts:

- A description of the problem being managed and how it may be solved better later.
- "For now" instructions to minimize the cost of interest that will be paid when cleaning up the debt.

## Graph Scheduling & Plumbing

Starting with a solution to the general problem would be appealing.  We know there are CPU and GPU dependencies.  We can do superscalar tricks to parallelize different stages and parallel tricks either CPU or GPU side.  Calculating what needs to be done and what opportunities can be taken is a coupled set of problems.

### For Now

Just do whatever works and attempt to read the tea leaves until it's clear which hard things need precise treatment and what data model they impose.

## Error Handling

Probably using thiserror to facilitate shipping as a library.

### For Now

- Return Result types from fallible operations to ensure proper combinator usages
- Unwrap and panic liberally (but do **not** clone haphazardly!)

## Vulkan Versions & Device Compatibility

Anticipate monolithic platform builds that switch at runtime for more specific support.

### For Now

- Use 1.3+ and any extensions from 1.4 that enhance productivity significantly
- Use `cfg` gates only for platforms, not for Vulkan versions.  To switch on Vulkan support, use runtime conditions.
