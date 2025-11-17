<p align="center">
  <img src="./.github/logo/mutate-logo.svg" alt="µTate - The Mutating Music Visualizer" width="50%">
</p>

<p align="center">
µTate (MuTate) is a project to build a modern music visualization program and
library, using Vulkan and local AI models in Rust.
</p>

## Status

Just started.  Writing the throwaway prototype first:

- [x] Window
- [x] Vulkan swap chain
- [ ] PipeWire (audio crate selection pending) input ring
- [ ] Draw inputs as outputs

### Incidental goals of this phase:

- Gather up tools and workflows
- Figure out lifecycle dependencies
- Decide where to write macros
- Get working PoCs for all headline features

Tool selections are aiming to be as modern as possible without becoming pioneering.  Development efficiency **and ultimate capability** will win most decisions.  While bleeding edge uses of Vulkan are compatible with the scope, the intent is to shift focus into the application of modern and frontier ML techniques toward simpler, forgiving audio visual problems.

## What Can Done Better With A New Visualizer?

- Beat anticipation from patterns
- Distinguish layered melodies and instrument changes
- Recognize lyrics and musical features
- Mung together traditional procedural techniques with generative techniques
- Lightweight customizations via text prompts
- Steel scheme scripting language for more in-depth preset programming and embedded extensibility
- Live performance-oriented features such as alternative frontend integration (LEDs, stage lights etc) and sequence pre-baking
  
Past visualizers users simple beat detection based on simple heuristics like volume thresholds.  These are easily faked out.  They don't understand patterns or layering.  Only a handful of dynamic values and waveform textures were available to preset authors, so the results were inevitably highly abstract programmer art that cannot reflect the mood or spirit of the music being played.

Got a better idea?  We would love to hear it.
[contact@prizeforge.com](mailto:contact@prizeforge.com).  If you want other
people to hear it, check out our [subreddit](https://reddit.com/r/prizeforge)
until we have our on-platform social reasoning MVP built.

### Open Product, Directly Sponsored by Positron

µTate is being developed by Positron to demonstrate the features of [PrizeForge](https://prizeforge.com).  The [stream for mutate](https://prizeforge.com/streams/details/163AQ5rQj92) is continuously raising funds.  Want to write code?  The PrizeForge stream exists to reward people like you.  Anyone can contribute funds, code or ideas.  In any case, Positron is committed to standing the project up.

### License

This project is licensed under either of

 * Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

#### Your contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
