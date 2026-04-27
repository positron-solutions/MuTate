# Onboarding

- `cargo run` will run the visualizer.
- `cargo test --features vulkan` in `/mutate-lib` will run the integration tests for key crates like `vulkan` and `macros`.
- `cargo workbench --help` uses a cargo alias to run the workbench program (a binary CLI tool using `mutate-lib` with the `dsp` feature for testing filter behaviors and generating pre-baked filter bank setups.
- `cargo pmr` runs the Parks-McClellen-Remez solver for FIR weight generation.

## Discussions

Design considerations have been landing in the Github [discussions](https://github.com/positron-solutions/MuTate/discussions).  In particular, the [recruiting contributors](https://github.com/positron-solutions/MuTate/discussions/2) discussion lists some good places to get started.

## Debt

See the [DEBT.md](./DEBT.md) for a maintained list of places we are trading a little temporary convenience & certainty for a bit more pain in the future.  It also records recommendations to reduce that pain until we ~~declare technical bankruptcy~~ pay off the debt.

## Layout

- `./crates/assets/` Build-time support for build scripts.  Runs `slangc` to emit SPIR-V and reflection data.  Runtime support for loading assets.
- `./crates/vulkan/` Vulkan is a buffet.  Our Vulkan crate is a plate from the buffet, a coherent set of features given a much reduced API that abstracts over a Vulkan subset to present a fully functional but much more ergonomic interface.
- `./crates/macros/` Proc macros for fanning in data form multiple types for agreement checking of ensembles like pipelines and their stage layouts.

### For Publishing

- `./mutate-lib/` An integration crate.  Includes the DSP crate right now.  Re-exports vulkan.  Intended as the public library for applications that want real-time DSP with vulkan integration.

- `./mutate-visualizer/` Uses mutate-lib to deliver a functioning music visualizer with window integration.

### Future Spin-Out Crates

- `./crates/slide/` - We use sliding windows.  Ring buffers typically have partially filled semantics, which means downstream has to deal with incomplete windows and potentially buffer the window themselves. This crate will likely mature and be spun out.

- `./crates/untorn/` - A triple buffering (seqlock first pass implementation) solution to effectively give us atomic structs with completely synchronous semantics over shared mutable memory.  There are times where we just don't want the ceremony of locking.  This crate will likely also mature and be spun out.

## AI Use Policy

**Quality of the submission is the only durable metric.**  We will not be witch-hunting em dashes or counting your fingers etc.  Submissions are likely to be given a first pass by various AI tools.  Verbose submissions etc disrespect people's time and will lead to being told off, so try to communicate like it's the 2000's internet and everyone is a dog, just professional dogs that speak engineer and have limited time and a lot of code to write.

## A Word on Rigor

Yee-Haw Index: 7 of 10 🤠.  Pick your favorite three-archetypes of engineers model, such as:

- pioneers
- settlers
- town planners

**This is absolutely not the time for town planners.**  If you can't ignore dirty code, move along or learn!  Code will change out from under things, and all your premature polishing will be for naught.  Brutal refactorings are welcome.  Last-write-wins.

Put Clippy away.  Add `#[allow(warnings)]` to your dirty tree and don't tell
anyone.  Slop in the blanks.  Just be sure to encode some useful facts and
**preserve truth faster than you destroy it.**  Write code for a yee-haw level of 5 or 6 out of 10 so that we can get there via [strangler fig](https://en.wikipedia.org/wiki/Strangler_fig_pattern) effects.

This chaotic phase will last until approximately the render graph API is being
used and render crossfades are supported.

## Pull Request Recommendations

These are not project specific, but maintainer tendencies on mature projects (this is not a mature project).

- Always attempt to separate structural from behavioral code.  If you rearrange hunks, try to commit those changes separately so that behavior is very easy to see.
- Small commits are preferred, especially those so tiny that each change is self-evident.

## PrizeForge, User-Lead Funding

All contributors may be selected for paying out PrizeForge awards.  The aim is for users of PrizeForge to decide who gets paid and what features are important.  PrizeForge keeps decision power with the backers, not the project maintainers, so **don't ask us for backer money.  We don't control any.**  We are building discussion tools for PrizeForge to enable much more effective support and communication between backers and contributors, so eventually you can talk there.
