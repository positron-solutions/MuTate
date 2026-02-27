# Debt

This is a record of "crimes" and the plans to later un-crime them.  Debt specifically covers crimes that cost us more later the longer we keep doing them and the rationale to keep doing them for now.

# Currently Paying Down

Crimes where the solution has been chosen and all new work should burn down existing problems.  Separate any distinct crimes that emerge into new debt.

## Logs & Tracing

Along with error handling in type signatures, we're starting to need some real infra for errors.  Having a library crate in addition to binaries opens lots of questions that just need decisions.  In the end, debugging becomes one of the biggest differentiators for professionals, so work here is highly appreciated!

## Ash & Raw Pointers

As we go, replace C pointer casting and `as_ptr()` calls with `push_next` and structure methods.  These accept more Rusty types and are safer (pointer castings is pretty unsafe).  See this commit in blame.

## Shader Boilerplate

Shaders must declare their inputs.  Push constant ranges and types must align.  Indexes must be typed for the right kinds of descriptors etc.  It's 1:1 and should be automated.

- Emit slang introspection data during build
  + Compile to spirv or MSL etc
- Read introspection data in macros to check agreement or generate agreeing structs
- Declaration macros and types they will express are in heavy development.

It's really only once we have a collection of pipelines for a coherent technique that we can see all dependencies for a single Visual.

## Devices vs VkContext

The way we find and initialize devices is still pretty noobish.  The VkContext needs to abstract over **multiple** devices, both for multi-GPU setups and perhaps to offload learning & inference.

## Task Graphs

We have multiple loops (audio, render, allocation, destruction, transfer, training, inference, compute) all driven independently and externally.  When these synchronize, they form *kind of a graph?*  It's a bit more like several event loops with systems that run on those loops.

In any case, there will be schedulers for these loops and things that want to run stuff will be able to inject function calls.  Dynamic dispatch so that we don't lock ourselves into an OOP style fill-in-the-blank set of choices about when to run things.

Buffers etc things will be allocated and then the scheduled tasks will be allowed to run on the next tick.  Do everything in reverse for teardown.

The somewhat well-trodden idea of the render graph will likely be obliterated by our needs for **crossfading different render styles**.  This is really different than a simple render graphs inserting barriers (an automation so cheap we can lazily decide it at runtime while recording buffers).

What we really care about are ergonomics and enabling style crossfades.  That requires deeper awareness of the semantic meaning of buffer contents, not just their types.  We have to be able to interleave different pipelines.  Just throw the idea of statically decided render graphs out the window.  They will not be flexible enough.

This is about to be under heavy development and will completely replace how all current visuals and task orderings are currently hand coded.

## Reactive Updates

Dependents should be notified reactively when their dependencies change configuration.  The first instance problem coming up is to resize a screen-dimension-sensitive visual when the screen size changes.  We have to re-allocate the buffers and re-size the internal data structures.  The resize information comes from the presentation target, and the reactive system must transmit this change to the dependent nodes.

Downstream reactions will usually affect the lifetime of fields and allocations such as buffers, not the lifetimes of structs such as nodes themselves.

The way this seems likely to be implemented is similar to other reactive systems but not nearly as granular.  Register inputs.  Re-instantiate on changes.  Swap resource pointers when ready.  Re-scale old resources when not yet ready.  This is coupled with the spec/hydration system being developed.

## Moving Spectrum Analyzer to GPU

The first-pass at the CQT has a number of problems that have excellent solutions available.

- Rather than constant Q (quality factor), some high frequency bins end up with 800 samples making them too precise making us miss energy between bins
- Low frequency iso226 correction is to extreme or the bins we are applying it to are missing energy due to accuracy issues and then the correction drops them out entirely
- No roll-on / roll-off behavior to speed up summing
- Decimation does not low-pass off the high pitches, so we fold noise from higher pitches until it dominates lower bins

There is more.  Filter banks require *engineering*.  See the [longer discussion](https://github.com/positron-solutions/MuTate/discussions/1)

The problem that is almost in the way of development is that even with `--release` the frame time at 1440p will be around 12ms of *just* audio processing time. That is too much.  We need to move this onto the GPU and try to kill some of the other issues as soon as possible.

Rather than making the CQT faster on the CPU, which will mainly involve doing things that worsen quality or fight very hard to avoid making it terrible, we should focus on moving to the GPU where we can suddenly do "expensive" things like adding a lot more filter bins and then make it cheaper because ... it's the right thing to do even though we have 512 cores or so 😉.

# Charging Interest

Each element includes two parts:

- A description of the problem being managed and how it may be solved better later.
- "For now" instructions to minimize the cost of interest that will be paid when cleaning up the debt.

## Audio Formats

The type of the input buffers is **not** bytes.  We should either coerce all input streams to one format or handle multiple formats if we cannot coerce all target platforms to a common denominator (and convert ourselves under the hood).  GPUs (and CPUs) prefer SoA and we should aim to make this easy by doing it all the time with a good set of tools.

### For Now

Hardcode and mark with `// DEBT`

## Memory Management

Expectations are that memory usage will be relatively low but less predictable due to generation, transitions, and scripting.

Where we're going, Visuals will provide their resource requirements as specs (which Images, SSBOs, Uniforms need to exist) and these will be instantiated by some allocator when they don't exist (discovered by Id etc).  These specs should give us some good, predictable high water marks for more intentional allocation strategies.

### For Now

We don't really have any infra for one-big-allocation or deletion & compaction.  Specs will just hydrate kind of dumbly at first while we nail the ergonomics.

Don't go crazy avoiding copies just yet, especially where sizes are in low kilobytes.  We can suffer reallocating buffers of these sizes per frame.

There is a better ring buffer crate for the task graph use case.  The existing `GraphBuffer` will / should die soon.

## Error Handling

The lib side is using `thiserror` and will present a single error `MutateError` type to consumers.  Currently the hierarchy has little semantic or diagnostic value.  Providing views into the underlying causes depends on what error handlers want to get out of the downstream error source.  Without that forcing pressure, we don't really know what types to separate or what information to expose yet.

Error handling has traditionally been an area of ergonomic innovation in Rust.  It's likely not beyond the innovation phase.

### For Now

- Use any MuTate error that seams appropriate or make a new one, and be honest about its use when documenting.
- Return Result types from fallible operations to ensure proper combinator usages.
- Unwrap and panic liberally

## Vulkan Versions, Device & Platform Compatibility

Anticipate monolithic platform builds.

### For Now

- Use 1.3+ and any extensions from 1.4 that enhance productivity significantly
- Use `cfg` gates only for platforms, not for Vulkan versions.  To switch on Vulkan support, use runtime conditions.
- Plan on using Molten on Apple.  The slang compiler can target (Metal Shader Language) but likley first pass, just rely on Molten to translate.  You need an Apple tool for MSL ⇒ metallibs.   If we switch to MSL though, the type agreement must use Apple-specific introspection data and modified macro logic!
