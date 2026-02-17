<p align="center">
  <img src="./.github/logo/mutate-logo.svg" alt="µTate - The Mutating Music Visualizer" width="50%">
</p>

<p align="center">
µTate (MuTate) is a project to build a modern music visualization program and
library, using Vulkan and local AI models in Rust.
</p>

### Build & Run

This repository (optionally) provides non-Rust dependencies via a Nix shell with direnv integration available.

#### Visualizer

The default binary selected by `cargo run` is the Vulkan frontend, found in the mutate-visualizer crate.

#### DSP Workbench

`cargo run --bin workbench --help` will list the CLI interface for the workbench, a CLI program being developed to assist in engineering and maintaining filter bank configurations to be used on the GPU.

## Status

The music moving a triangle phase is complete.  A first-pass of the spectrogram was completed.  The next milestone is to move the improved DSP work onto the GPU.

Work to create a real architecture is underway:

- [x] Fullscreen support (includes resizing and swapchain re-creation)
- [ ] Reactive updates for render graph dependents, such as images that depend on the window
      extent
- [x] Separate drawing and presentation
- [ ] Indirect / off-screen rendering for presentation by CLI frontend, similar
      to [alemidev/scope-tui](https://github.com/alemidev/scope-tui).
- [ ] Memory, devices, queue handling
- [ ] Audio processing as an upstream input for an *optional* downstream render graphs

Before going deep on capabilities such as multi-GPU, we will just develop interfaces that have the semantics while implementing them in the most simple way first.

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

### Distribution

Only the TUI binary will likely be appropriate for `cargo install`.  The other binaries will need to be built via CI for releases, which will document the process for distributors.  Dependencies will be maintained as a Nix flake and associated shells.

### Incidental goals of this phase:

- Gather up tools and workflows
- Figure out lifecycle dependencies
- Decide where to write macros
- Get working PoCs for all headline features

Tool selections are aiming to be as modern as possible without becoming pioneering.  Development efficiency **and ultimate capability** will win most decisions.  While bleeding edge uses of Vulkan are compatible with the scope, the intent is to shift focus into the application of modern and frontier ML techniques toward simpler, forgiving audio visual problems.

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

### Open Product, Directly Sponsored by Positron

µTate is being developed by Positron to motivate use of [PrizeForge](https://prizeforge.com), a new set of social finance tools.  The [stream for mutate](https://prizeforge.com/streams/details/163AQ5rQj92) is continuously raising funds.  Want to write code for µTate?  The PrizeForge stream exists to reward people like you.

µTate enthusiasts can view PrizeForge like a better Patreon or Kickstarter specific to this repo for now.  Funds are matched bit-by-bit with other users before being disbursed.  Funds remain controlled by the backers and can be disbursed to anyone who contributes on µTate, not just the repo owner.  *That's one way Positron aims to change the model.*

### License

This project is licensed under either of

 * Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

#### Your contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
