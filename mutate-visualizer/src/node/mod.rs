// Copyright 2025 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Node
//!
//! ⚠️ Much of this is intended, not yet realized.  Regard this module's documentation as planning
//! and guidance for node authors to begin going in the right directions.
//!
//! `Node`s consume input and yield output.  Nodes are connected in a directed graph.  Edges
//! represent data connections between nodes. The graph can be used to calculate memory
//! requirements, calculate memory hazards, parallelize independent work, and coordinate between
//! upstream and downstream nodes in order to remove specific couplings between nodes and enable
//! greater independence and flexibility of composition.
//!
//! ## Reactivity
//!
//! Nodes are cooperative in memory, compute, configuration, throttling, and scheduling.  Updates to
//! nodes may force updates to dependent nodes.  These updates will be transmitted reactively, by
//! registering dependencies and transitively informing dependents if their dependencies were
//! updated.
//!
//! ### Configuration
//!
//! Nodes may depend on resolution, audio input type, choice of audio stream, parts of a viewport
//! etc.  These dependencies may change due to user actions.  Nodes will be informed that values
//! they depend on have been updated, enabling them to re-allocate and clear old states.
//!
//! ## Backpressure
//!
//! Nodes downstream of a stalled node will receive calls informing them of their stalled upstream.
//! Nodes that are themselves stalled will be instructed to reduce their compute requirements.
//!
//! Nodes may have a consumer / producer mismatch.  The decision to duplicate or drop upstream or
//! downstream depends on whatever behavior can be more correct for the given nodes.
//!
//! ## Resource Pressure
//!
//! A new node may not see enough memory available.  In such cases, all nodes, beginning with the
//! heaviest nodes, will be asked to downscale resolutions of assets and buffers or perform less
//! precise calculation.  This will continue until the new node can be created.

pub mod audio;
pub mod video;

use mutate_lib as utate;

use crate::graph;

/// Push nodes can cooperate with the graph to have `produce` called multiple times, enabling them
/// to seek forward in the upstream buffer.  The other states are implicitly reacted to with the
/// contents of the `GraphEvent` they return.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum SeekState {
    /// Insufficient upstream data was available, so this node is slowing down consumption to allow
    /// upstream to catch up.
    OverProduced,
    /// Yielding full events within acceptable seek ranges in all input streams.
    OnTime,
    /// This node is speeding up consumption and wants to yield extra events in order to align the
    /// seek range with the presentation ranges downstream.
    UnderProduced,
}

/// The [Graph]: crate::graph::Graph builds, destroys, and drives `Node`s.  `Node`s implement
/// `consume` to receive new input and `produce` to yield their most recent output.  The decoupling
/// of consumption and production is so designed because some streaming updates are cheap compared
/// to producing an up-to-date output.

// This is just a rough draft.  Nodes in development are working to define constrains on this
// interface.  There will likely be several kinds of nodes or they will be subtyped via generics.
#[allow(unused)]
pub trait Node {
    type Input;
    type Output;
    type NodeDeps;

    // XXX Get rid of
    type Produced;
    type Consume;

    /// Configure the node
    fn new(state: &mut graph::GraphContext) -> Result<Box<Self>, utate::MutateError>;

    /// Create buffers, including any buffers that will be provided as output.
    fn provision(state: &mut graph::GraphContext) -> Result<(), utate::MutateError>;

    /// Update internal state by consuming upstream inputs.
    fn consume(&mut self, input: Self::Input) -> Result<SeekState, utate::MutateError>;

    /// Give us the most up-to-date output ready for downstream.
    // NEXT decide how to handle double buffering for late-binding and wait-free!
    fn produce(&mut self, output: &mut Self::Output) -> Result<Self::Produced, utate::MutateError>;

    fn destroy(self, device: &mut graph::GraphContext) -> Result<(), utate::MutateError>;

    /// Respond to upstream configuration changes.
    fn update(self, device: &mut graph::GraphContext) -> Result<usize, utate::MutateError>;
}
