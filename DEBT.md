# Debt

This is a record of "crimes" and the plans to later un-crime them.  Debt specifically covers crimes that cost us more later the longer we keep doing them and the rationale to keep doing them for now.

For more forward-looking feature
[discussion](https://github.com/positron-solutions/MuTate/discussions), see
Github.

# Currently Paying Down

Crimes where the solution has been chosen and all new work should burn down existing problems.  Separate any distinct crimes that emerge into new debt.

## Moving Spectrum Analyzer to GPU

The first-pass at the CQT has a number of problems:

- Rather than constant Q (quality factor), some high frequency bins end up with 800 samples making them too precise making us miss energy between bins
- Low frequency iso226 correction is to extreme or the bins we are applying it to are missing energy due to accuracy issues and then the correction drops them out entirely
- No roll-on / roll-off behavior to speed up summing
- Decimation does not low-pass off the high pitches, so we fold noise from higher pitches until it dominates lower bins

There is more.  Filter banks require *engineering*.  See the [longer discussion](https://github.com/positron-solutions/MuTate/discussions/1)

The problem that is almost in the way of development is that even with `--release` the frame time at 1440p will be around 12ms of *just* audio processing time. That is too much.  We need to move this onto the GPU and try to kill some of the other issues as soon as possible.

Rather than making the CQT faster on the CPU, which will mainly involve doing things that worsen quality or fight very hard to avoid making it terrible, we should focus on moving to the GPU where we can suddenly do "expensive" things like adding a lot more filter bins and then make it cheaper because ... it's the right thing to do even though we have 512 cores or so 😉.

The migration to Slang was going to itself require better engineering around pipeline-shader development.

## From Manual Destruction to Drop

In the beginning, all was done to expedite change.  An example of the longest lived things is the `Instance` and `Device`.  We need them nearly everywhere, on every thread, threaded directly or indirectly through nearly every function call.  They would be used for a lot of RAII, but then we're figuring out a root-to-leaf network on the first pass.

The strategy being used to develop actual lifetime contracts is to start at leaf types, things with the **shortest lifetimes first**.  The `QueueSubmit` is one of the first types that started using lifetime contracts.  It is much easier to work out lifetime contracts from shortest lived things, which tend to live and die in a single scope.

Until we reach the root, prefer manual destruction.  Manual destruction avoids a class of bugs where an RAII wrapper holds a cloned handle to enable cleanup but then outlives the wrapper for the handle.  **Not destroying things produces very clear validation errors.**  Unclear drop orders of temporaries can lurk in barely-working code, so manual destruction **is** more conservative.

There are also must-consume types, for which drops are almost assuredly program bugs, and so `DropBomb` is being used until some other kind of linear type solution (ever?) exists.  Why would a user drop a command buffer that is in the middle of recording, especially if the allocation would leak even after pool reset?

# Charging Interest

Each element includes two parts:

- A description of the problem being managed and how it may be solved better later.
- "For now" instructions to minimize the cost of interest that will be paid when cleaning up the debt.

## Logs & Tracing

We're need some real infra for emitting error feedback.  Tracing selected.  We can do log fallback later for people who don't want tracing.  Option to make release builds silent should be supported.  In the end, debugging becomes one of the biggest differentiators for professionals, so work here is highly appreciated!

### For Now

`println!` and `eprintln!`.  Easy enough to text-replace later.

## Reactive Updates

Dependents should be notified reactively (or enabling them to poll) when their dependencies change configuration.  They may then choose to kick off asynchronous resource updates.

The first instance problem coming up is to resize a screen-dimension-sensitive visual when the screen size changes.  We may have to re-allocate the buffers and re-size the internal data structures.  The resize information comes from the presentation target, and these changes must reach dependents.

Long term, reacting is coupled with the spec/hydration system being developed.

### For Now

We just need things like new extents to reach dependents.  We can likely afford to just recreate resources on the fly.  Try to prepare for pointer swaps.  In general, code that can tolerate pointer swaps makes it easier to just swap in updated resources from a new ring while draining an old ring.

## Timeline & Render Graph

The big behaviors we're after:

- Aliasing memory both for re-use and crossfade rendering
- Independent timelines with exclusive phase support to do some machine learning between frames and handle audio graph self-pacing vs VRR frame latch deadlines
- Automatic hazard detection (runtime) and barrier insertion (render graph)
- Intent language style primitives that can evaluate to a graph efficiently and tell us most of these things relatively quickly

### For Now

We just have to do these things manually until some pain builds up.  General scheduling is hard, mainly because it's under-constrained for our use case, and should be avoided.  The same with general memory allocation.  **Focus on concrete needs** rather than perfect designs that make us over-commit to a particular pattern.

The one thing that seems super clear is that without a single layer indirection for pointers, many cool things are not possible later.  Think in terms of late binding hot-swapping pointers on the GPU.  The pointer is only guaranteed not to be deleted while in flight.  Easy for the user.  Aliasing, reallocation, garbage collection etc all just boil down to swapping the pointers that readers are holding / looking up.  Pointers are atomic.  It makes life a lot simpler.

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

The lib side is using `thiserror` and will present a single error `MutateError` type to consumers.  Farther upstream crates like `vulkan` have their own type (`VulkanError`) that is forwarded through `MutateError` variants.

The hierarchies may still have little semantic or diagnostic value at first. We need to know what error handlers want to get out of the upstream error source
before providing views into the underlying causes depends on.  Without the
forcing pressure from error consumers, we don't really know what types to
separate or what information to expose yet.

Error handling has traditionally been an area of ergonomic innovation in Rust.  It's likely not beyond the innovation phase.

### For Now

- Unwrap and panic liberally 🤠
- Return `Result` types from fallible operations to ensure proper combinator usages are happening.
- Use any MuTate error that seams appropriate or make a new one, and be honest about its use when documenting.
- If you are a saint, go implement proper tracing, tracing formatting, options for consumers that want to ignore tracing, spans and the like.
- If you are less of a saint, find panics where continuing has some meaningful use case and convert them to `Result` and do something useful after returning it.

## Fallible Resource Acquisition

Creating multiple Vulkan objects within a higher level constructor can be viewed as a resource transaction.  If one thing fails, the partial construction must be unwound.  Because the partial values are typically created on the stack, consuming RAII patterns can be useful to enforce destruction on temporaries while lifting the RAII restrictions on the final output value.

### For Now

Manual destruction is tolerable for simple cases.  See [When do Drop](#from-manual-destruction-to-drop).

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

## Transfer / Staging vs UMA

UMA architectures, becoming increasingly important, don't really benefit from transfer queues.  This suggests we would want an abstraction to hide the implementation so users don't need to change what they express on DMA vs UMA.

We'd probably like to make thread-safe writes over sub-ranges and that infra will be extremely similar to 1) preparing and streaming data in a worker thread 2) using upload slots with a timeline semaphore and notifications for the render thread.  The streaming data write case will make DMA and UMA wind up at the same semantics and API, which is the right surface to abstract away.

### For Now

Transfer?  Use the UMA path until something is actually big.  Maybe put it behind a dummy interface with some kind of sensible semantics that will work for the above.

## Bytemuck Traits in Slang Module

We need `Pod` and `Zeroable` on emitted types.

### For Now

The `Pod` and `Zeroable` markers were just thrown in by hand so we don't even need derive.  This might not catch situations where the traits don't actually work 😬

## Untorn

- Needs to implement a triple buffer variant (seqlock can be interrupted) or some other truly wait-free implementation.  Seqlock is a bit better for distributed systems where the blocked writer is *below* the thread switch granularity and all runtimes are async anyway.
- Integrate local stack copies if possible

### For Now

Focus on the semantics.  We want synchronous, local stack, then finally trick out the implementation.

## CI & Nix Caching

We should be using Crane to build test artifacts and caching the dependency buids in Nix or using a Rust cache of some kind (impure, but better than nothing).

### For Now

Using cold target dirs.  Fortunately most of our builds are cheap enough to not cause 45min CI times.  The PMR tool hast the heaviest dependencies right now.

There's more work coming to build and distribute binaries.  That's a good time to take a look at Nix caching.

## Host-Slang Agreement

The slang module and proc macros (`ComputePipeline`, `PushConstants`, etc) should be reading the reflection data and emitting const checks.

We're going with scalar block layout.  While it's pretty flexible, it's not `repr(C)`.  We don't yet have full scalar block checking everywhere (anywhere).

### For Now

Ergonomics over contracts.  The APIs are *sufficient* to add the reflection checks.  Get `GraphicsPipeline` working first.

## Audio Formats

The actual type of the input buffers is **not** bytes.  We should either coerce
all input streams to one format or handle multiple formats if we cannot coerce
all target platform audio survers to give us a common denominator (and convert
ourselves under the hood).  GPUs (and CPUs) prefer SoA and we should aim to make
this easy by doing it all the time with a good set of tools.

### For Now

Hardcode and mark with `// DEBT`
