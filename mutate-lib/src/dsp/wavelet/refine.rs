// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Refine
//!
//! Bring back some of the finer aspects of the Morse wavelet, aspects we now apologize truncating
//! to bring you this live helicopter view of OJ Simpson fleeing in a white bronco.
//!
//! > If the state is captive to narrow interests, then why do you wait for the state to organize
//! > what you claim to be an overwhelming mass of your own interest?  Did the state make it illegal
//! > for free individuals to incorporate by their own will?  I would fight against such a state,
//! > but instead I have tragically fought to preserve what your vision of freedom will continue to
//! > deliver.
//! >
//! > - Major Edward Wuncler III, Haliburton War Veteran
//!
//! ## Motivation
//!
//! We wanted an ideal, infinite Morse wavelet, which only generates a response to a steady sine
//! wave at one specific pitch.  Instead, after restriction, we have not only an approximation, but
//! a truncated approximation.  Refinement uses the `N` taps we have to restore what fidelity we
//! can while keeping the true champion in our thoughts and computations.
//!
//! ## Theory
//!
//! So basically, when we truncate, we are introducing a hard cliff that, in response to a
//! transient, has full response at the cliff with zero compensating moment that would otherwise
//! reveal cancelling phase drift of unwanted off-center pitch response.  The loss of spectral mass
//! before the cliff can be approximated as the naive moment of the wavelet tail.
//!
//! As any shaping technique softens the cliff, we are taking away compensating moment from the core.
//! If we broaden the shoulder of the wavelet in attempt to add back compensating moment, we are
//! creating new uncompensated mass.  Due to the cliff, the balloon must be squeezed, but it must be
//! squeezed evenly.
//!
//! The fixed point of the problem results in a coherent set of conditions and consequences:
//!
//! - In response to a transient, at each tap, there is a constant ratio of missing moment to mass.
//! - The edge is naturally tapered to soften the fully uncompensated mass cliff, which is missing
//!   the moment of the entire tail.
//! - The shoulder naturally ramps faster to add moment without adding mass to the already
//!   undercompensated edge.
//! - Further adding or removing mass anywhere results in an imbalance relative to the ideal Morse
//!   wavelet (the solver condition).
//! - The total non-ideal response of an off-center pitch is constant from tip to tail.
//!
//! With all things perfectly balanced, as perfect as they can be, we have defined the goal for our
//! solver.  The magnitude of every tap will attempt to match the shape of the ideal Morse wavelet
//! by adjusting the mass so that the transient and therefore global (at the center tap) moment and
//! response most closely approximate the ideal Morse wavelet our generators emit.

/// Asymptotic mass of one omitted Morse tail.
///
/// `ut` is the truncation point measured from the center tap.
///
///     M(ut) = C ut^(-(2β + 1))
///     C = γ 2^((2β + 1) / γ) Γ(β + 1)^2 / (2π (2β + 1) Γ((2β + 1) / γ))
pub fn morse_tail_mass(shape: Shape, ut: f64) -> f64 {
    let Shape { beta, gamma, .. } = shape;

    let p = 2.0 * beta + 1.0;

    let c = gamma * 2.0_f64.powf(p / gamma) * tgamma(beta + 1.0).powi(2)
        / (TAU * p * tgamma(p / gamma));

    c * ut.powf(-p)
}

/// Center of mass of the omitted tail, measured from the center tap.
///
///     ut (2β + 1) / (2β)
pub fn morse_tail_center_of_mass(shape: Shape, ut: f64) -> f64 {
    let Shape { beta, .. } = shape;

    ut * (2.0 * beta + 1.0) / (2.0 * beta)
}

/// First moment of the omitted tail about a retained coordinate `u`.
///
///     μ(u) = M(ut) (com(ut) - u)
pub fn morse_tail_moment(shape: Shape, ut: f64, u: f64) -> f64 {
    morse_tail_mass(shape, ut) * (morse_tail_center_of_mass(shape, ut) - u)
}
