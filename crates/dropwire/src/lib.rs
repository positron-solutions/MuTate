// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # DropWire
//!
//! **Compile-time** drop contracts enabling enforcement of "must consume" semantics for your
//! types.
//!
//!```
//! use mutate_dropwire::DropWire;
//!
//! // Add to any type for which dropping without consuming is always invalid program behavior.
//! struct RecordedBuffer {
//!     raw: u64, // raw Vulkan handle
//!     // DropWire defaults to armed state.  Type parameter, which can be customized, propagates
//!     // into compiler output, enabling a bit more specific error feedback.
//!     wire: DropWire<RecordedBuffer>,
//! }
//!
//! // A re-structured final state with the tripwire removed, the output type for a consuming
//! // method.
//! struct SubmittedBuffer { raw: u64 }
//!
//! impl RecordedBuffer {
//!     // In your constructor, use the handy `ARMED` constant to remind yourself that this type
//!     // requires disarming (and consumption).
//!     pub fn new(raw: u64) -> Self {
//!         Self { raw, wire: DropWire::armed() }
//!     }
//!
//!     // Valid consuming methods should destructure self, disarm the wire, and consume the payload
//!     // fields.
//!     pub fn submit(self) -> SubmittedBuffer {
//!         let Self { raw, wire } = self;
//!         wire.disarm();
//!
//!         // Use the payload fields
//!         // queue.submit(raw) etc...
//!
//!         // and return the new valid type-state wrapper or just drop if no longer needed
//!         SubmittedBuffer { raw }
//!     }
//! }
//! ```
//!
//! ## Motivation
//!
//! If you put a letter in an envelop and lick the glue, it's probably a bug if you don't seal and
//! send the letter.  We can enforce the transition from filled to licked to sealed to sent with
//! type states, making the API shape only allow valid transitions, but what guarantees no letters
//! will be dropped on the floor?  We need "must consume" semantics in our API and ways to enforce
//! those semantics.
//!
//! Implementing `Drop` on all values is both heavy and still doesn't give us good tools for
//! encouraging semantically correct programs.  The timing of the drop might be sensitive.  If I
//! load the missile into the tube and the program crashes but the `Drop` implementation says
//! "fire!" before I've opened the launch doors, violence happens.  The inversion of control,
//! allowing the world to be called by dropped values, adds both weight to runtime types and
//! opportunities to violate phase alignment contracts that keep things from happening at the wrong
//! times.
//!
//! ## How DropWire Works
//!
//! We want to enforce "must consume" semantics.  DropWire implements the necessary guarantees by
//! simply adding a ZST field to structures that must not be dropped without first destructuring and
//! disarming.  Destructuring decouples the ownership so that `disarm` can simply forget the
//! `DropWire`.  If we re-package the remaining data as a new type, removing the ZST essentially
//! becomes a variant of the familiar type-state pattern.
//!
//! The ZST tripwire uses a const generic as the const context for the compile-time check.
//! Post-monomorphization, the `Drop` implementation will evaluate a const expression to a compile
//! time error unless the ZST has been transmuted to a disarmed type.
//!
//! Calling `disarm` is an attestation that one of the proper consuming methods was called.  By
//! keeping this part of your type's private interface, only your blessed methods can consume,
//! destructure, and attest that the contract was upheld.  If the user drops the type without going
//! through this blessed interface, the tripwire is tripped.
//!
//! One field.  One method.  Zero runtime overhead.
//!
//! ## Locating Illegal Drops
//!
//! The compile-time check cannot lead you to the call site.  It can only tell you that your program
//! contains a code path that drops an armed wire.  The **type** in the compiler error output may
//! narrow down the possibilities enough.
//!
//! If you can't find the drop on inspection, you can try to trigger the drop at runtime.  Enable
//! the `runtime` feature of the crate and the compile-time check will be replaced with a runtime
//! panic that indicates both the wire's construction and drop location.  Even if you cannot trigger
//! the drop by using your program, you can rule out paths that are executed and isolate the
//! inspection surface to code that did not execute.
//!
//! If the drop site is actually triggered by running your program,
//!
//! ## Limitations
//!
//! - **Lack of Scope Knowledge** - The offending scope in which a drop occurs cannot be indicated.
//!   The failing const expression will be evaluated if a `DropWire` is ever dropped, but this is
//!   per-monomorphization of the const expression, not unique to the dropping scope.  Appending a
//!   type parameter as a token, the `O` generic, enables the compiler to at least tell us what type
//!   has a problem.
//!
//! - **Requires Pattern Destructuring** - Disarming the wire requires changing the owning type.
//!   De-structuring the type to remove the wire is one such way.  Using a type-state proxy from the
//!   owner is another that is not yet supported.  If you do not have sufficient access to the
//!   owning type to re-structure it, you may be unable to write a consuming method that disarms the
//!   wire.
//!
//! ## This Crate is Stolen
//!
//! The technique may have earlier examples in the wild, but the `PhantomDrop`
//! [example](https://internals.rust-lang.org/t/an-approach-to-linear-ish-types/21111/2?u=psionic-must-drop)
//! by Pitaj was the inspiration for this crate.  That implementation uses a post-monomorphization
//! trick, but it was determined that the armed const state essentially just makes a single type
//! drop-illegal and that the transmutation in order to drop can just as well be done by simply
//! forgetting the armed value.
//!
//! ## Well Ackshually
//!
//! **Ahem** since `Drop` may still be implemented for the composing type in order to handle panics,
//! the types are not strictly guaranteed to be linear.  According to Bothan spies, they are known
//! as "relevant" types, but only people without boats understand this kind of stuff.

// NEXT We can possibly write a new DropWire that piggybacks on the owning type's drop-illegal
// typestate parameters to enable disarming by simply modifying the owner's typestate in a normal
// typestate transition.

#[cfg(not(feature = "runtime"))]
mod ct;
#[cfg(feature = "runtime")]
mod rt;

#[cfg(not(feature = "runtime"))]
pub use ct::DropWire;
#[cfg(feature = "runtime")]
pub use rt::DropWire;
