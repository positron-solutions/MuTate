# Debt

This is a record of "crimes" and the plans to later un-crime them.  Debt specifically covers crimes that cost us more later the longer we keep doing them and the rationale to keep doing them for now.

For more forward-looking feature
[discussion](https://github.com/positron-solutions/MuTate/discussions), see
Github.

## Yee-Haw Index: 7 of 10 🤠

This is a foreword.  Pick your favorite three-archetypes model, such as:

- pioneers
- settlers
- town planners

**This is absolutely not the time for town planners.**  If you can't ignore dirty code, move along or learn!  Code will change out from under things, and all your premature polishing will be for naught.  Brutal refactorings are welcome.  Last-write-wins.

Put Clippy away.  Add `#[allow(unused)]` to your dirty tree and don't tell
anyone.  Slop in the blanks.  Just be sure to encode some useful facts and
preserve truth faster than you destroy it.  Write code for a 5 or 6 out of 10 so that we can get there.

This phase will last until approximately the render graph API is being used and
render crossfades are supported.

# Currently Paying Down

Crimes where the solution has been chosen and all new work should burn down existing problems.  Separate any distinct crimes that emerge into new debt.

## Logs & Tracing

Along with error handling in type signatures, we're starting to need some real infra for errors.  Having a library crate in addition to binaries opens lots of questions that just need decisions.  In the end, debugging becomes one of the biggest differentiators for professionals, so work here is highly appreciated!

Tracing selected.  We can do log fallback later for people who don't want tracing.  Silent release build option.

## Buffer Item Layouts

We're going with scalar block layout.  While it's pretty flexible, it's not `repr(C)`.  We don't yet have full scalar block checking everywhere (anywhere).  Manually align while we implement the contracts around the ergonomics.

## Ash & Raw Pointers

As we go, replace C pointer casting and `as_ptr()` calls with `push_next` and structure methods.  These accept more Rusty types and are safer (pointer castings is pretty unsafe).  See this commit in blame.

## Shader Boilerplate

Shaders must declare their inputs.  Push constant ranges and types must align.  Indexes must be typed for the right kinds of descriptors etc.  It's 1:1 and should be automated.

- Emit slang introspection data during build
  + Compile to spirv or MSL etc
- Read introspection data in macros to check agreement or generate agreeing structs
- Declaration macros and types they will express are in heavy development.

It's really only once we have a collection of pipelines for a coherent technique that we can see all dependencies for a single Visual.

## Reactive Updates

Dependents should be notified reactively when their dependencies change configuration.  The first instance problem coming up is to resize a screen-dimension-sensitive visual when the screen size changes.  We have to re-allocate the buffers and re-size the internal data structures.  The resize information comes from the presentation target, and the reactive system must transmit this change to the dependent nodes.

Downstream reactions will usually affect the lifetime of fields and allocations such as buffers, not the lifetimes of structs such as nodes themselves.

The way this seems likely to be implemented is similar to other reactive systems but not nearly as granular.  Register inputs.  Re-instantiate on changes.  Swap resource pointers when ready.  Re-scale old resources when not yet ready.  This is coupled with the spec/hydration system being developed.

## Moving Spectrum Analyzer to GPU

The first-pass at the CQT has a number of problems:

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

## Bytemuck Traits in Slang Module

Current code is a rough draft.  We need `Pod` and `Zeroable` but getting the derive macro paths right is very fiddly  *inside* the crate.  Proper fix might be to split the crate and do integration tests downstream.

### For Now

The `Pod` and `Zeroable` markers were just thrown in by hand so we don't even need derive.  This might not catch situations where the traits don't actually work 😬

## Timeline & Render Graph

The big behaviors we're after:

- Aliasing memory both for re-use and crossfade rendering
- Independent timelines with exclusive phase support to do some machine learning between frames and handle audio graph self-pacing vs VRR frame latch deadlines
- Automatic hazard detection (runtime) and barrier insertion (render graph)
- Intent language style primitives that can evaluate to a graph efficiently and tell us most of these things relatively quickly

### For Now

We just have to do these things manually until some pain builds up.  General scheduling is hard, mainly because it's under-constrained for our use case, and should be avoided.  The same with general memory allocation.  **Focus on concrete needs** rather than perfect designs that make us over-commit to a particular pattern.

The one thing that seems super clear is that without a single layer indirection for pointers, many cool things are not possible later.  Think in terms of late binding hot-swapping pointers on the GPU.  The pointer is only guaranteed not to be deleted while in flight.  Easy for the user.  Aliasing, reallocation, garbage collection etc all just boil down to swapping the pointers that readers are holding / looking up.

## Memory Management

Expectations are that memory usage will be relatively low but less predictable due to generation, transitions, and scripting.

- https://lib.rs/crates/gpu-allocator for obtaining GPU memory
- https://lib.rs/crates/offset-allocator for slicing it up and handing it out
- https://lib.rs/crates/slotmap for tracking what we handed out

Where we're going, workloads will provide their resource requirements as specs (which Images, SSBOs, Uniforms need to exist) and these will be instantiated by some allocator when they don't exist (discovered by Id etc).  These specs should give us some good, predictable high water marks for more intentional allocation strategies.

### For Now

We don't really have any infra for one-big-allocation or deletion & compaction.  Specs will just hydrate kind of dumbly at first while we nail the ergonomics.

Don't go crazy avoiding copies just yet, especially where sizes are in low kilobytes.  We can suffer reallocating buffers of these sizes per frame.  Until we have a solution that will do better than the driver, just allocate for each image / buffer.

There is a better ring buffer crate for the task graph use case.  The existing `GraphBuffer` will / should die soon.  See the `mutate-slide` crate with its `SlidingWindow` as a foundation.  Probably we have to loan out slices and manually protect those borrows from torn read with render loop pacing sync instead of using the slices or window as sync primitives themselves.

## General Image Layouts

This is a pretty boring area of automation in terms of design.  Tracking or computing layouts is not hard.  We are supposed to do it.. for mobile?  Someday.  There are much more interesting things to automate that don't really depend on layouts.

### For Now

The performance of `vk::ImageLayout::GENERAL` is just not bad.  It is sometimes guaranteed to be negligible and the driver is supposed to figure things out.  To keep ergonomics simple, let's lean on general where possible and then consider adding other layouts back in to be about device support & performance.

## Error Handling

The lib side is using `thiserror` and will present a single error `MutateError` type to consumers.  Upstream crates are forwarded through `MutateError` variants.  Currently the hierarchy may still have little semantic or diagnostic value.  Providing views into the underlying causes depends on what error handlers want to get out of the upstream error source.  Without the forcing pressure from error consumers, we don't really know what types to separate or what information to expose yet.

Error handling has traditionally been an area of ergonomic innovation in Rust.  It's likely not beyond the innovation phase.

### For Now

- Unwrap and panic liberally 🤠
- Return `Result` types from fallible operations to ensure proper combinator usages are happening.
- Use any MuTate error that seams appropriate or make a new one, and be honest about its use when documenting.
- If you are a saint, go implement proper tracing, tracing formatting, options for consumers that want to ignore tracing, spans and the like.
- If you are less of a saint, find panics where continuing has some meaningful use case and conver them to `Result` and do something useful after returning it.

## Vulkan Versions, Device & Platform Compatibility

We don't have any infrastructure for falling back when devices don't support requested features.

### For Now

The go-to pattern is use whatever is most ergonomic for development and then back-port features if there is still some target worth supporting.  It's most ergonomic to assume everyone is Vulkan 1.4 and supports everything. 😬

- Use 1.3+ and any extensions from 1.4 that enhance productivity significantly
- Use `cfg` gates only for platforms, not for Vulkan versions.  To switch on Vulkan support, use runtime conditions.
- Plan on using Molten on Apple.  The slang compiler can target (Metal Shader Language) but likely first pass, just rely on Molten to translate SPIR-V.  You need an Apple tool for MSL ⇒ metallibs.   If we switch to MSL though, the type agreement must use Apple-specific introspection data and modified macro logic!

## Dynamic Command Buffer State Shadow

We don't really want to unset dynamic states that were set somewhere else.  The dependency might be real?  It will be.  There are cases.

In the end, we want push/pop style states for different rendering techniques so that the techniques are not conscious of what they interleave with.

### For Now

Manual mode!  Grep for states and set / unset the relevant ones.

## Audio Formats

The actual type of the input buffers is **not** bytes.  We should either coerce
all input streams to one format or handle multiple formats if we cannot coerce
all target platform audio survers to give us a common denominator (and convert
ourselves under the hood).  GPUs (and CPUs) prefer SoA and we should aim to make
this easy by doing it all the time with a good set of tools.

### For Now

Hardcode and mark with `// DEBT`

## Presentation Capable Queue Families

Detecting the need to do a queue transfer before present is unavoidably tedious.  These cases are said to be rare.  To properly support, we have to check if the command buffer and its queue family can do presentation and, if not, find a transfer capable queue and do presentation over there, meaning another command buffer pool too!

```
//! (◕‿◕)︵‿︵‿︵‿︵┻━┻
```

### For Now

You know what?  We assume the first queue to present (usually graphics, usually the zeroth index) is the right one.  Any support that looks complete is an accident.  If you need weird things, try commercial support or go do some coding `(◕‿◕)ノ彡☆`
