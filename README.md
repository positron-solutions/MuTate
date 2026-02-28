<p align="center">
  <img src="./.github/logo/mutate-logo-light.svg#gh-light-mode-only" alt="µTate - The Mutating Music Visualizer" width="50%">
  <img src="./.github/logo/mutate-logo-dark.svg#gh-dark-mode-only" alt="µTate - The Mutating Music Visualizer" width="50%">
</p>

<p align="center">
µTate (MuTate) is a project to build a modern, adaptive music visualizer and music-driven neural rendering library.
</p>

## Feature Goals

- **high-resolution on-GPU DSP** with multi-window sliding DFTs for simultaneous high dynamic range, pitch, amplitude, and time precision.
- **visual crossfades** between multiple rendering techniques.
- **neural rendering** including real-time adaptive shading.

µTate uses Vulkan, building on top of [ash](https://github.com/ash-rs/ash) low-level bindings, with the [slang](https://shader-slang.org/) shader language, enabling introspection-based compile-time checks, automatic differentiation, and unification of graphics & ML shader programming.

### Build & Run

This repository (optionally) provides non-Rust dependencies via a Nix shell with direnv integration available.  `direnv allow` or `nix develop .#x11` etc will load an environment.  Use `nix flake show` to list other shells for other platforms and runtime situations.

#### Visualizer

The default binary selected by `cargo run` is the Vulkan frontend, found in the mutate-visualizer crate.

#### DSP Workbench

`cargo run --bin workbench --help` will list the CLI interface for the workbench, a CLI program being developed to assist in engineering and maintaining filter bank configurations to be used on the GPU.

## Status

Work to implement an ergonomic architecture while implementing the headline features is underway:

- [ ] Move initial constant-Q prototype spectrogram onto the GPU.
- [ ] Declarative pipeline composition with introspection-based type and layout agreement over all shader inputs.
- [ ] Multiple frontend support, such as using off-screen rendering converted to terminal output similar to [alemidev/scope-tui](https://github.com/alemidev/scope-tui).
- [ ] Online real-time machine learning to modulate and feed procedural rendering techniques.

See [DEBT.md](./DEBT.md) and [discussions](https://github.com/positron-solutions/MuTate/discussions) for design and feature planning.

### Platform Support

**These are good places to contribute**.  We need to draw on several kinds of Surfaces:

- [x] Xlib
- [ ] MacOS
- [x] Wayland (untested)
- [ ] Windows
- [ ] Android (apk Packaging is the larger share of work)

Platform-specific audio servers may be used for audio monitors:

- [ ] CPAL (cross-platform, uses Pulse Audio on Linux)
- [x] Pipewire
- [ ] Other platforms optionally if CPAL monitoring doesn't work

## What Can Done Better With A New Visualizer?

- Beat anticipation from musical patterns
- Distinguish layered melodies and instrument changes
- Recognize lyrics
- Mung together traditional procedural techniques with emerging generative techniques
- Lightweight customizations via text prompts
- Steel scheme scripting language for more in-depth preset programming and embedded extensibility
- Live performance-oriented features such as alternative frontend integration (LEDs, stage lights etc) and sequence pre-baking
  
Past visualizers users simple beat detection based on simple heuristics like volume thresholds.  These are easily faked out.  They don't understand patterns or layering.  Only a handful of dynamic values and waveform textures were available to preset authors, so the results were inevitably highly abstract programmer art that cannot reflect the mood or spirit of the music being played.

### Fast Local Machine Learning

Rather than integrating slow, network-dependent generative AI based on heavy transformers and integrating via MCP, this project seeks to attract development of extremely lightweight alternatives, focusing on architecture sophistication above all.

Positron will be contributing some alternative training / online learning techniques that are not dependent on differentiable behavior in the feed forward.  In our opinion, music visualization is the perfect [forgiving problem](https://positron.solutions/articles/finding-alignment-by-visualizing-music) for open source machine learning to to do some cooking.

### Bringing Open Development to the Non-Programming Consumer

µTate is being developed by Positron to motivate use of [PrizeForge](https://prizeforge.com), a new set of social finance tools.  The [stream for mutate](https://prizeforge.com/streams/details/163AQ5rQj92) is continuously raising funds.  Want to write code for µTate?  The PrizeForge stream exists to reward people like you.

µTate enthusiasts can view PrizeForge like a better Patreon or Kickstarter specific to this repo for now.  Funds are matched bit-by-bit with other users' funds before being disbursed.  Funds remain controlled by the backers and can be disbursed to anyone who contributes on µTate, not just the repo owner.  *That's one way Positron aims to change the model.*

### License

This project is licensed under either of

 * Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

#### Your contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
