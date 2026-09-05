// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Downsampling Restriction Stencil
//!
//! > We have created lab-grown beef.  Next, we are working on lab-grown *ground* beef.
//! >
//! > - Professor of Mathematics Arima Nayar
//!
//! Our high quality mother wavelet must be squeezed into N taps.  Naively, we may sample the
//! motherlet at the point nearest to `k`, and tell our boss that we have done the job.
//!
//! ## That is Not Good Enough
//!
//! At high omega, the motherlet is beginning to alias with N.  Her smooth curves will become
//! reduced to formless points, like pushing Michelangelo's David through a cheese grater.
//!
//! Instead, we want the result of the cheese greater to *look* the same to audio as the full-rate
//! wavelet.  Since the audio has gone through roughly the same cheese grater, our job is made a
//! little bit simpler.  In the stencil step, we are mainly concerned with ensuring that the shape
//! of the wavelet we are refining is maximally well preserved ever time we shove it through the
//! grater.
//!
//! The solution is pretty simple.  We pad the wavelet at the truncation and reflect it at the
//! origin.  We then use a polynomial solver to nudge taps in the directions that will fix up the
//! key properties, the first, second, and 3rd moments that the next stage, the linear solver, can
//! make corrections to.  The solution only requires the moments of the source motherlet integrated
//! over the tap and then spits out the each tap based on a solution that preserves each points
//! relation to its neighbors.
