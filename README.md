<p align="center">
  <img src="./.github/logo/mutate-logo-light.svg#gh-light-mode-only" alt="µTate - The Mutating Music Visualizer" width="50%">
  <img src="./.github/logo/mutate-logo-dark.svg#gh-dark-mode-only" alt="µTate - The Mutating Music Visualizer" width="50%">
</p>
<p align="center">
  <a href="https://github.com/positron-solutions/MuTate/actions/workflows/ci.yml"><img src="https://github.com/positron-solutions/MuTate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://discord.gg/gxptWX6aCR"><img alt="µTate Community Discord invite.  Need help getting ramped up?  Come talk to us!" src="https://img.shields.io/discord/1498766098764533891?logo=discord&logoSize=auto&label=%C2%B5Tate%20Community&color=%232fbf50&labelColor=%23333a40"></a>
</p>
<p align="center">
  <i>...real-time programs that display impressive graphics, music, and mind-blowing effects sometimes <b>considered impossible given the limits of the hardware</b></i> - <a title="Making Art With Code" href=https://youtu.be/vIQ74_DRWEM?si=vK3zc6X-kbbZC9aM&t=770">The Incredible Demoscene</a>
</p>
<p>
  The need to be smaller drives more sophistication, not less.  It also drives faster iteration.  µTate (/ˈmjuːteɪt/) gives researchers developing post-attention heuristics a path to quick viability for their work without stacks of H200's and petabytes of data.  Music visualization that learns online demands more efficiency.  The techniques required will apply far beyond music, reaching into hard problems that will enable the next round of breakthroughs.
</p>

### Contents

- [Building the Alliance](#building-the-alliance)
- [The Vulkan Engine](#the-vulkan-engine)
  - [Staying Low-Level](#staying-low-level)
- [The DSP](#the-dsp)
- [The Machine Learning](#the-machine-learning)
- [Build & Run](#build--run)
  - [Visualizer](#visualizer)
  - [Minimal Vulkan Example](#minimal-vulkan-example)
  - [DSP Workbench](#dsp-workbench)
- [Contributing](#contributing)
  - [Platform Support](#platform-support)
  - [License](#license)
    - [Your contributions](#your-contributions)

## Building the Alliance

Bringing together a wide group of overlapping interests creates opportunities to solve problems that no one beneficiary could easily approach alone.  Technology that is both upstream of a huge consumer market and has even bigger downstream applications makes new things possible.

µTate's direct usages provide synchronized visuals for live music performances or casual home entertainment visualization.  µTate uses gaming-adjacent technologies like [slang](https://shader-slang.org/) because these enable us to tap into demand for things like better artificial intelligence and more natural procedural content, tuned with feedback from a live renderer.

Incidentally, the common parts of the problems will emit open source implementations of critical components for others to advance in their own fields.  Today's neural rendering can feed into tomorrow's heuristics toolkits that enable fast iteration on multi-disciplinary problems with prohibitive search spaces, problems like using synthetic biology to more cheaply make sustainable aviation fuel so that supersonic international travel becomes trivial.

## The Vulkan Engine

µTate's online learning goals require some specific runtime capabilities that will incidentally create a user-friendly declarative graphics programming interface.  The design phase complete, and this is what we are building:

- **Host-shader Type Agreement** - Use slang reflection data and Rust proc-macro fan-in to verify type and layout agreement of declared sets of pipeline stages.
- **Declarative Resources** - Enable pipelines to specify the geometry and semantic type of upstream inputs, intermediate buffers, and outputs.
- **Resource Runtime** - Maintain a single state of what should be loaded and to operate reconciliation processes to continually maintain the intended state.
- **Workload Management** - Requesting resources fed by upstream dispatches and external processes like pipewire should spin up those processes to feed those input buffers.
- **Memory management** - Maintain sub-allocations, epoch recycling, and borrow counts for shared resources in use by multiple independent downstreams.  Support aliasing.
- **Reactive Argument Injection** - Enable command recording for pipelines to pick up pointer swaps and other dynamic values from a single source, decoupling the recorder from knowledge of the specific upstream sources.  Enable publishes to affect the correct epoch without knowledge of the downstream.

The argument injection and declarative resource spin-up mean composing pipelines does not require any knowledge of how to provision them.  This enables machine learning to focus on declaring how to draw, modulating arguments, and which data to use.  Meanwhile the runtime will put those declarations into effect.

### Staying Low Level

Only Vulkan (and Molten on MacOS) will be supported, meaning no WGPU entanglement or requisite backend abstraction.  Only a modern subset of Vulkan is supported first-class, eliminating a lot of its API surface.

Types are built on top of bare [ash](https://docs.rs/ash/latest/ash/) bindings and all wrapper types provide `as_raw` methods to take off the guard rails if necessary.  Some types `Deref` to their `ash::vk` handles for direct use in the ash APIs.  Use the raw device and your own allocations to work around any limitations of the ergonomics scaffolding.

With regards to safety, the strategy being taken is to start with guard rails off, tolerating unsound Rust (🤡) and invalid Vulkan (🤠) in order to build up contracts only where their introduction does not restrict capability.  This leans on validation layers and enables starting at a C++ style of user responsibility and walking towards Rust without creating unhelpful friction about guarantees that would only apply on the CPU side anyway.

## The DSP

µTate is bringing real-time DSP onto the GPU so that prohibitively expensive on-CPU techniques can be done at conspicuously high resolutions in real time, directly adjacent to the video pipelines that need that output.  4k filter banks with multiple high-quality Goertzel filters and wide-band IIRs are intended to provide sources of neat textures, modulation signals to move things with the music, and inputs for real-time beat detection & inference.

Past visualizers users simple beat detection based on simple heuristics like volume thresholds.  These are easily faked out.  They don't understand patterns or layering.  Only a handful of dynamic values and waveform textures were available to preset authors, so the results were inevitably highly abstract programmer art that cannot reflect the mood or spirit of the music being played.  Compared to traditional heuristics, these high quality procedural inputs will enable:

- Distinguish layered melodies and instrument changes.
- Interpret timbre and texture to create visuals unique to how an instrument is played on that particular day.
- Beat **prediction** and visual cues when something *is expected*.
- Automating visual "hard cuts" when prediction fails, following the change in musical tone with changes in visual tone.

## The Machine Learning

Rather than integrating slow, network-dependent generative AI based on heavy transformers and integrating via MCP, this project is a home for development of extremely lightweight alternatives, focusing on architecture sophistication above all.  This is because constrained consumer hardware (and not lighting it on fire all day) demands it.

Positron will be contributing some alternative training / online learning techniques that are not dependent on differentiable behavior in the feed forward.  In our opinion, music visualization is the perfect [forgiving problem](https://positron.solutions/articles/finding-alignment-by-visualizing-music "Note, this site and PrizeForge are likely still offline if you are reading this") for open source machine learning to to do some cooking.

The plan is it develop the first implementations as libraries loaded at runtime (a plugin).  This means other implementations can tap into the visualizer and its user base without everyone having to ship the entire infrastructure and having no way to combine sets of pipelines and resources.

## Build & Run

You need `slangc` available for the build scripts.  You need a competent Vulkan implementation, so for example, install `vulkan-loader` and `vulkan-validation-layers`.

This repository (optionally) provides non-Rust dependencies via a Nix shell with direnv integration available.  `direnv allow` or `nix develop .#x11` etc will load an environment.  Use `nix flake show` to list other shells for other platforms and runtime situations.

### Visualizer

The default binary selected by `cargo run` is the standard frontend, found in the mutate-visualizer crate.

### Minimal Vulkan Example

```
cargo run -p mutate-minimal
```

The internal `mutate-vulkan` crate is intended to become a competent Vulkan runtime engine.  The minimal example crate demonstrates getting started with an application.  To develop small toy shaders, see the dispatch integration tests in `mutate-lib`.

### DSP Workbench

`cargo workbench --help` will list the CLI interface for the workbench, a CLI program being developed to assist in engineering filter bank configurations for use on the GPU.  A separate `pmr` bin (requires pm_remez) can generate static FIR filter settings. Try `cargo pmr lowpass --taps 23`.

## Contributing

Start with the [CONTRIBUTING.md](./CONTRIBUTING.md) guide.  See [DEBT.md](./DEBT.md) for an idea of what compromises are in place.  See [discussions](https://github.com/positron-solutions/MuTate/discussions) for design and feature planning.  Chat on [our Discord](https://discord.gg/KzSpewYU) if you want to work on this library or the visualizer.  There's both very technically challenging and relatively simple work.

At this time, we **really** need involvement of users of various platforms and hardware to help smoke test different desktop environments and test cross-platform integrations like CPAL.  There are lots of places to contribute to the project at a basic level, such as working on logging or even cleaning up warnings in modules that seem to be stable.  If your work is at least good enough to save us time, we will give you time.

### Platform Support

Platform-specific audio servers may be used for audio monitors:

- [ ] CPAL (cross-platform, uses Pulse Audio on Linux)
- [x] Pipewire
- [ ] Other platforms optionally if CPAL monitoring doesn't work

Surface support on each platform should be already working, but MacOS may need some tuning for the Molten support to function.  See [the recruiting thread](https://github.com/positron-solutions/MuTate/discussions/2) for more details on the state of hardware and platform support.

### License

This project is licensed under either of

 * Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

#### Your contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
