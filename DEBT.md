# Debt

This is a record of "crimes" and the plans to later un-crime them.  Debt specifically covers crimes that cost us more later the longer we keep doing them and the rationale to keep doing them for now.

# Currently Paying Down

Crimes where the solution has been chosen and all new work should burn down existing problems.  Separate any distinct crimes that emerge into new debt.

## Lifetime Alignment

Build structs to gather firmly bound lifetimes.  Consider multiple window setups.  Nodes encapsulate resources exclusive to their lifetimes.  Boundaries emerging:

- Instance, entry
- Device, physical device, memory
- Presentation targets
- Drawing nodes
- Other processing nodes
- Transfer & deletion mementos
- Timing graph

User setting updates, dynamic scripting, and generation will all as usual require a lot of re-creation and re-allocation that can share duty with teardown, destruction being the first step of re-creation.

## Reactive Node Updates

Graph dependents should be notified reactively when their dependencies change configuration.  The first instance problem coming up is to resize a screen-dimension-sensitive CTQ when the screen size changes.  We have to re-allocate the buffers and re-size the internal data structures.  The resize information comes from the presentation target, and the graph must transmit this change to the dependent nodes.

Downstream reactions will usually affect the lifetime of fields and allocations such as buffers, not the lifetimes of structs such as nodes themselves.

# Charging Interest

Each element includes two parts:

- A description of the problem being managed and how it may be solved better later.
- "For now" instructions to minimize the cost of interest that will be paid when cleaning up the debt.

## Memory Management

Dynamic usage of allocated memory is likely unavoidable.  Several tools are like VMA bindings or the gpu-allocator crate are being looked at.  Expectations are that memory usage will be relatively low but less predictable due to generation and scripting.  Bindless rendering will certainly be coupled to the memory use strategy.  The tradeoffs of existing approaches are not clear yet, but the need to manage a pool and dependent addresses does suggest more rather than less work will pay off.

The CQT (Constant Q Transform) Window-resize problem is really informative.  There are several valid strategies to replace a missized CQT:

- Create a parallel CQT and switch the downstream binding when the replacement is ready, then mark the previous for garbage collection and stop scheduling it.
- Use the existing CQT output, up-sampled for 1-2 frames while creating the new CQT.  Once the new CQT is producing outputs, the downstream bindings will see new updates.
- Immediately stop drawing downstreams until the new upstream is ready.

The first technique will lead to the best fidelity, but requires extra memory.  The last option is the easiest to implement (and will usually be on time and should be used first).  Under memory / compute pressure, high-cost asset rotations can fall back to low-cost ones.

### For Now

Nodes are just given a device context (also WIP) and create and destroy their own assets.  The node interface needs to emerge along with the render graph behaviors to interrogate the nodes for what operations need to be done.

## Graph Scheduling & Plumbing

Starting with a solution to the general problem would be appealing.  We know there are CPU and GPU dependencies.  We know DAGs can model exclusive dependencies etc.

Calculating what needs to be done and what opportunities can be taken is a coupled set of problems.  Some things will benefit more from explicit control.  Some from automation.  There is usually overlap to fill gaps and automation tends to subsume manual interventions.

The current strategy being selected uses the following building blocks:

- high precision timing thread.
- worker threads for workloads that don't obviously need their own thread.
- dedicated threads where parallelism is guaranteed anyway.
- rendering phases will be explicitly modeled and timed
- late binding wherever possible, re-using frame n data when n+1 is not available

### For Now

Just do whatever works and attempt to read the tea leaves until it's clear which hard things need precise treatment and what data model they impose.

## Error Handling

The lib side is using `thiserror` and will present a single error `MutateError` type to consumers.  Currently the hierarchy has little semantic or diagnostic value.  Providing views into the underlying causes depends on what error handlers want to get out of the downstream error source.  Without that forcing pressure, we don't really know what types to separate or what information to expose yet.

Error handling has traditionally been an area of ergonomic innovation in Rust.  It's likely not beyond the innovation phase.

### For Now

- Use any MuTate error that seams appropriate or make a new one, and be honest about it use when documenting.
- Return Result types from fallible operations to ensure proper combinator usages.
- Unwrap and panic liberally (but do **not** clone haphazardly!)

## Vulkan Versions & Device Compatibility

Anticipate monolithic platform builds that switch at runtime for more specific support.

### For Now

- Use 1.3+ and any extensions from 1.4 that enhance productivity significantly
- Use `cfg` gates only for platforms, not for Vulkan versions.  To switch on Vulkan support, use runtime conditions.
