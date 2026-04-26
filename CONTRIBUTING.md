# Onboarding

Coming soon!

- [ ] list binaries and test commands
- [ ] CI
- [ ] crate layout

## Discussions

Design considerations have been landing in the Github [discussions](https://github.com/positron-solutions/MuTate/discussions).  In particular, the [recruiting contributors](https://github.com/positron-solutions/MuTate/discussions/2) discussion lists some good places to get started.

## DEBT

As mentioned in that link, see the [DEBT.md](./DEBT.md) for a listing of things that are trading a little temporary speed and certainty for a bit more pain in the future and attempting to reduce that pain until we ~~default~~ pay off the debt.

## AI Use Policy

**Quality of the submission is the only durable metric.**  We will not be witch-hunting em dashes or counting your fingers etc.  Submissions are likely to be given a first pass by various AI tools.  Verbose submissions and those 

## A Word on Rigor

Yee-Haw Index: 7 of 10 🤠.  Pick your favorite three-archetypes of engineers model, such as:

- pioneers
- settlers
- town planners

**This is absolutely not the time for town planners.**  If you can't ignore dirty code, move along or learn!  Code will change out from under things, and all your premature polishing will be for naught.  Brutal refactorings are welcome.  Last-write-wins.

Put Clippy away.  Add `#[allow(warnings)]` to your dirty tree and don't tell
anyone.  Slop in the blanks.  Just be sure to encode some useful facts and
preserve truth faster than you destroy it.  Write code for a yee-haw level of 5 or 6 out of 10 so that we can get there via [strangler fig](https://en.wikipedia.org/wiki/Strangler_fig_pattern) effects.

This chaotic phase will last until approximately the render graph API is being
used and render crossfades are supported.

## Pull Request Recommendations

These are not project specific, but maintainer tendencies on mature projects (this is not a mature project).

- Always attempt to separate structural from behavioral code.  If you rearrange hunks, try to commit those changes separately so that behavior is very easy to see.
- Small commits are preferred, especially those so tiny that each change is self-evident.

## PrizeForge, User-Lead Funding

All contributors may be selected for paying out PrizeForge awards.  The aim is for users of PrizeForge to decide who gets paid and what features are important.  PrizeForge keeps decision power with the backers, not the project maintainers, so **don't ask us for backer money.  We don't control any.**  We are building discussion tools for PrizeForge to enable much more effective support and communication between backers and contributors, so eventually you can talk there.
