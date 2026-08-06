# Contributing

- [Onboarding](#onboarding)
  - [Discussions](#discussions)
  - [Debt](#debt)
  - [Workspace Layout](#workspace-layout)
    - [For Publishing](#for-publishing)
    - [Future Spin-Out Crates](#future-spin-out-crates)
- [Ways We Work](#ways-we-work)
  - [Typed Comments](#typed-comments)
  - [AI Use Policy](#ai-use-policy)
  - [A Word on Rigor](#a-word-on-rigor)
  - [Pull Request Recommendations](#pull-request-recommendations)
  - [Code Style & Conventions](#code-style--conventions)
  - [Timing Model](#timing-model)
  - [PrizeForge, User-Lead Funding](#prizeforge-user-lead-funding)
  - [The Name](#the-name)

# Onboarding

- `cargo run` will run the visualizer.
- `cargo run -p mutate-minimal` will run at least a small application using our Vulkan library.
- `cargo test --features vulkan` in `/mutate-lib` will run the integration tests for key crates like `vulkan` and `macros`.
- `cargo workbench --help` uses a cargo alias to run the workbench program (a binary CLI tool using `mutate-lib` with the `dsp` feature for testing filter behaviors and generating pre-baked filter bank setups.
- `cargo pmr` runs the Parks-McClellen-Remez solver for FIR weight generation.

## Discussions

Design considerations have been landing in the Github [discussions](https://github.com/positron-solutions/MuTate/discussions).  In particular, the [recruiting contributors](https://github.com/positron-solutions/MuTate/discussions/2) discussion lists some good places to get started.

## Debt

See the [DEBT.md](./DEBT.md) for a maintained list of places we are trading a little temporary convenience & certainty for a bit more pain in the future.  It also records recommendations to reduce that pain until we ~~declare technical bankruptcy~~ pay off the debt.

## Workspace Layout

- `./crates/assets/` Build-time support for build scripts.  Runs `slangc` to emit SPIR-V and reflection data.  Runtime support for loading assets.
- `./crates/vulkan/` Vulkan is a buffet.  Our Vulkan crate is a plate from the buffet, a coherent set of features given a much reduced API that abstracts over a Vulkan subset to present a fully functional but much more ergonomic interface.
- `./crates/macros/` Proc macros for fanning in data form multiple types for agreement checking of ensembles like pipelines and their stage layouts.

### For Publishing

- `./mutate-lib/` An integration crate.  Includes the DSP crate right now.  Re-exports vulkan.  Intended as the public library for applications that want real-time DSP with vulkan integration.

- `./mutate-visualizer/` Uses mutate-lib to deliver a functioning music visualizer with window integration.

### Future Spin-Out Crates

- `./crates/slide/` - We use sliding windows.  Ring buffers typically have partially filled semantics, which means downstream has to deal with incomplete windows and potentially buffer the window themselves. This crate will likely mature and be spun out.

- `./crates/untorn/` - A triple buffering (seqlock first pass implementation) solution to effectively give us atomic structs with completely synchronous semantics over shared mutable memory.  There are times where we just don't want the ceremony of locking.  This crate will likely also mature and be spun out.

## Typed Comments

As an ad-hoc local management tool and a way to communicate at a high level about modules with each other, module level regular comments often follow the doc comments.  The all-caps token enables judging the nature of a comment without reading it.

- `XXX` - This probably should not have shipped, but if it did, it means the code is actually only working on the happy paths.  Something is very wrong.  Mostly synonymous with `FIXME`.  Most often found inline, near the problem.
- `NEXT` - The next thing(s) that may be worked on.  Writing this relieves the author from implementing features and behaviors that seem **obvious from the point where they left off**.
- `LIES` - The code semantically appears to be doing something but in fact is not doing that thing or is doing something else entirely.  The semantics may need fixing or we may be hacking around something or achieving some side effect.
- `DEBT` - Very specifically there is something that is intentionally being done consistent with a trade-off documented in [DEBT.md](./DEBT.md).
- `ROLL` - We're waiting on something that is at least partially out of our control, an when this is unblocked, we will **roll off** the old ways and into a new era.
- `NOTE` - Just an observation, something to help get oriented with the mental model or the long term goals.  Not relevant to users, only contributors.
- `MAYBE` - A genuine ponderance.  We should be on the lookout for a problem that is brewing or an opportunity that isn't quite clear.  The uncertainty is too great for commitment, but the impact and likelihood are enough to warrant a reminder.

Most modules begin with doc comments and then have several typed comments.  Typed comments, especially `NEXT` tend to go out of date and become scattered.  Before working on a module, do attempt to retire or refine comments and place the changes into a "line noise" commit along with other superficial changes.

**Searching these comments is a good way to find things to work on.** 👽

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
- Use commit titles like `crate::module;` or just `crate;` if the title makes the module obvious.  Detours into fixing comments can be rolled into `line noise` commits.

## Code Style & Conventions

- All raw `ash` handles **must** be used behind either the `ash::` or `vk::` (`ash::vk::`) prefix.  Only µTate types should be used without prefix.  This makes raw types very easy to see in implementation code.
- Imports are recommended to use a single prefix for out-of-crate dependencies.  Example: `vk::DeviceAddress` instead of just `DeviceAddress`.
- Imports are sorted and divided as:

  + `std`
  + External dependencies
  + Workspace dependencies
  + Crate dependencies

## Timing Model

Tracking and slewing in data-time is one of the technically trickier aspects of µTate.  Audio server and display refresh tick on **different clocks**.  Therefore, we are not only frequently handling data rate adaptations, but also the dissimilarly scaled ticks on unaligned grids.

**key design choices**:

- Track a **virtual** write head that is the continuous interpretation of the discretely chunked stream.
- Maintain local offsets from the virtual write head, which is located nearby in time, instead of global index relations.
- Use integral ticks greatest-common-multiple rate expressions on the host math (`u64`) but use smaller integers and physical indexes with wrap on the device.  On-device rings **must** use PoT sizes unless expressed otherwise.

With local offset tracking, we discard knowledge of the absolute index drift and instead focus on local data re-sampling scale accuracy.  Globally we are re-sampling inaccurately.  Locally, the ratio of input to output is quite accurate and the error self-corrects rather than accumulates.

Maintaining a local time anchor erases underruns, jitter, and drift from history.  Tracking and slewing is about maintaining the correct distance behind the input, and relative time does exactly that and does not require consistency with absolute time epochs, only deltas on ticks and relative read-write head positions.

- External clocks ticks are filtered to uncover the hidden phase.  The estimated phase grid of input sources are then published for all downstream consumers.
- Discrete ticks are interpreted as a continuous data flows which may be safely tracked one tick length behind the continuous approximation to account for phase-related underruns.
- The next prediction step is used to re-anchor the time grid on each tick.  Local states apply this shift-of-reference and all calculations are relative to the read head.
- Read goals are computed from phase duration and jitter as a buffer length measured in time, configured to avoid underrun with a desired success rate.
- Different data rates mean that output grid zeroes do not align, and a sub-datum phase component is stored to track the relative grid offset.
- The buffer length has the grid phase delay subtracted because any reader is already `grid delay` behind in continuously interpreted input-time.
- Only whole datums are exposed to consumers.  Partial datum support would require overwriting stale partial outputs, and when cascaded through arbitrary downstream application, the provenance is lost unless tracked (provenance tracking schemes may succeed).
- To appropriately feed high-speed displays and stretched output, buffers intended for downstream consumption should emit data at 240Hz or above.
- Interpolation occurs by transitioning from the old output datum to the new output datum *over one datum of time*, fairly weighing each datum while applying FIR filtering at the point of consumption.

## Power Consumption

Keeping potatoes cool and living under a strict budget.

### Nvidia Devices

Dig supported values out of `nvidia-smi -q -d SUPPORTED_CLOCKS`

```
nvidia-smi -lmc 810,810      # pin memory
nvidia-smi -lgc 300,300      # pin core
nvidia-smi -rmc && nvidia-smi -rgc   # reset both
```

## Performance Debugging

Request help if you need features enabled on the device in order to use external performance debugging tools.

### Nsight

```
# Nsight systems is available in nixpkgs as cudaPackages.nsight_systems
# ie nix shell nixpkgs#cudaPackages.nsight_systems

# Get device
nsys profile --gpu-metrics-devices=help

# Get metrics support
nsys profile --gpu-metrics-set=help

# Profile
nsys profile --trace=vulkan,nvtx,osrt  \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=tu10x-gfxt \
  --gpu-metrics-frequency=10000 \
  -o trace
  ./target/debug/mutate-visualizer
  
# Inspect
nsys-ui ./trace.nsys-rep
```

## PrizeForge, User-Lead Funding

All contributors may be selected for paying out PrizeForge awards.  The aim is for users of PrizeForge to decide who gets paid and what features are important.  PrizeForge keeps decision power with the backers, not the project maintainers, so **don't ask us for backer money.  We don't control any.**  We are building discussion tools for PrizeForge to enable much more effective support and communication between backers and contributors, so eventually you can talk there.

## The Name

µ is the [micro sign](https://www.compart.com/en/unicode/U+00B5).  Unicode point is `U+00B5` or `181`.  It may also be typed with `Alt + 0181` on Windows or `Option + M` on MacOS.  When in a hurry, "MuTate" or "uTate" are acceptable.  Packaging had been leaning towards `mutate` prefix, but a new [organization](https://github.com/utate-community) was created with `utate` in anticipation of migrating to `utate` or `µTate` everywhere.

When spoken, "µTate" must be pronounced /ˈmjuːteɪt/ or "MYOO-tayt" to indicate vigorous mogging of all other visualization programs.   \ˌmīk-ˈrō-ˌtāt\ or "Mike Rotate" must not be used because it sounds too similar with "Developers! developers! developers!" or a very small cuttlefish named Tate.  When greeting other µTate contributors, they must be addressed as /ˈmjuːteɪtərz/ or "Myoo Taters" but not too hickishly, such as \ˈmyü-ˌtā-dərz\.
