// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Video
//!
//! Things that draw things.
//!
//! To any following along, the intended development path ahead is complex and will require
//! continuous application of strangler fig pattern.  Aggressive, haphazard development is not only
//! welcome, but a requirement.
//!
//! The intended path is to support loading libraries (plugins) that describe how a runtime can call
//! into them to set up drawing methods or load resources.  Visualizations that land in the core
//! will be curated to catalog techniques and establish the APIs under maintenance.
//!
//! The natural path forward:
//!
//! - visualizations can be toggled at
//! - toggle at runtime
//! - composing steps
//! - transitions between different graphs
//! - runtime modulation of steps

pub mod pulse;
pub mod ring;
pub mod triangle;
pub mod verticlysm;
