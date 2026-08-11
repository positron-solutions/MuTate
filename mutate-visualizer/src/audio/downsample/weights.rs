// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Weights
//!
//! > She heavy.
//! >
//! > - Your dad
//!
//! Weights generated from the packaged PMR tool.  See README to re-generate other lengths.  Weights
//! are designed for the shader execution structure.  Details on the filter design in downsample.rs.

// NEXT Design for main lobe widths!  There's a lot of subjective tuning done so far.
// NEXT Proc macro!  The CLI tool is kind of a dumpster fire just good enough to waddle across the
// finish line!  At a minimum, the CLI tool needs to be configured for our use case instead of
// leaving the generation commands in the comments.

// $ cargo pmr lowpass --taps 25 --pass 0.125 --stop 0.375 --pass-guard 0.1 --stop-guard 0.2
// --decimate 2`
//
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 25
// Band frequencies are normalized to sample rate of 1.0
//
// passband:   0.0-0.1250 (guarded to 0.1500)
// stopband:   0.3750-0.5 (guarded from 0.3250)
// transition: 0.2500 wide, 0.1750 after guards
//
// Design Results
// =========================================================
//   weighted error: 0.01381347
//   flatness: 0.00000611
//   iterations: 8
//
// Gain Testing
// =========================================================
//   deep pass band:                            0.99978405
//   pass band:                                 0.99984044
//   new Nyquist:                               0.32821527
//   upper passband fold mirror:                0.00007019
//   deep passband fold mirror:                 0.00004710
//   stop band:                                 0.00007019
//   deep stop band:                            0.00004710
pub(crate) const DOWN_TWO: [f32; 25] = [
    f32::from_bits(0xba3d6162), // -7.224289002e-4
    f32::from_bits(0xba57a042), // -8.225479396e-4
    f32::from_bits(0x3b3e0b0c), // +2.899828367e-3
    f32::from_bits(0x3b8aa08c), // +4.230564460e-3
    f32::from_bits(0xbbec4190), // -7.209964097e-3
    f32::from_bits(0xbc5ecb44), // -1.359826699e-2
    f32::from_bits(0x3c5f764a), // +1.363904215e-2
    f32::from_bits(0x3d109501), // +3.529835120e-2
    f32::from_bits(0xbca9a000), // -2.070617676e-2
    f32::from_bits(0xbdb087ce), // -8.619652689e-2
    f32::from_bits(0x3cd89648), // +2.643884718e-2
    f32::from_bits(0x3e9f5183), // +3.111687601e-1
    f32::from_bits(0x3ef1619c), // +4.714478254e-1
    f32::from_bits(0x3e9f5183), // +3.111687601e-1
    f32::from_bits(0x3cd89648), // +2.643884718e-2
    f32::from_bits(0xbdb087ce), // -8.619652689e-2
    f32::from_bits(0xbca9a000), // -2.070617676e-2
    f32::from_bits(0x3d109501), // +3.529835120e-2
    f32::from_bits(0x3c5f764a), // +1.363904215e-2
    f32::from_bits(0xbc5ecb44), // -1.359826699e-2
    f32::from_bits(0xbbec4190), // -7.209964097e-3
    f32::from_bits(0x3b8aa08c), // +4.230564460e-3
    f32::from_bits(0x3b3e0b0c), // +2.899828367e-3
    f32::from_bits(0xba57a042), // -8.225479396e-4
    f32::from_bits(0xba3d6162), // -7.224289002e-4
];

// $ cargo pmr lowpass --taps 49 --pass 0.0625 --stop 0.1875 --pass-guard 0.1 --stop-guard 0.2
// --decimate 4
//
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 49
// Band frequencies are normalized to sample rate of 1.0
//
// passband:   0.0-0.0625 (guarded to 0.0750)
// stopband:   0.1875-0.5 (guarded from 0.1625)
// transition: 0.1250 wide, 0.0875 after guards
//
// Design Results
// =========================================================
//   weighted error: 0.02173785
//   flatness: 0.00000000
//   iterations: 7
//
// Gain Testing
// =========================================================
//   deep pass band:                            0.99957645
//   pass band:                                 0.99975353
//   new Nyquist:                               0.34572175
//   upper passband fold mirror:                0.00012299
//   deep passband fold mirror:                 0.00004243
//   stop band:                                 0.00012299
//   deep stop band:                            0.00008010
pub(crate) const DOWN_FOUR: [f32; 49] = [
    f32::from_bits(0xb9353eda), // -1.728491916e-4
    f32::from_bits(0xb9bfb2f4), // -3.656368935e-4
    f32::from_bits(0xb9b24159), // -3.399949346e-4
    f32::from_bits(0x39248b21), // +1.569208835e-4
    f32::from_bits(0x3a939d4f), // +1.126209158e-3
    f32::from_bits(0x3b06a0cd), // +2.054262208e-3
    f32::from_bits(0x3b047459), // +2.021095017e-3
    f32::from_bits(0x398b70cd), // +2.659618913e-4
    f32::from_bits(0xbb448500), // -2.998650074e-3
    f32::from_bits(0xbbc8fdde), // -6.133778952e-3
    f32::from_bits(0xbbd72768), // -6.565976888e-3
    f32::from_bits(0xbb18e2ae), // -2.332847100e-3
    f32::from_bits(0x3bc6065e), // +6.043239497e-3
    f32::from_bits(0x3c701115), // +1.465251017e-2
    f32::from_bits(0x3c8e857b), // +1.739763282e-2
    f32::from_bits(0x3c18b7fa), // +9.321207181e-3
    f32::from_bits(0xbc1a0f04), // -9.402994066e-3
    f32::from_bits(0xbcfff290), // -3.124359250e-2
    f32::from_bits(0xbd2f8f0e), // -4.286103696e-2
    f32::from_bits(0xbcfac31c), // -3.061061352e-2
    f32::from_bits(0x3c482f84), // +1.221835986e-2
    f32::from_bits(0x3da424b8), // +8.014816046e-2
    f32::from_bits(0x3e1f3cde), // +1.555056274e-1
    f32::from_bits(0x3e5b9daa), // +2.144686282e-1
    f32::from_bits(0x3e7272f8), // +2.367666960e-1
    f32::from_bits(0x3e5b9daa), // +2.144686282e-1
    f32::from_bits(0x3e1f3cde), // +1.555056274e-1
    f32::from_bits(0x3da424b8), // +8.014816046e-2
    f32::from_bits(0x3c482f84), // +1.221835986e-2
    f32::from_bits(0xbcfac31c), // -3.061061352e-2
    f32::from_bits(0xbd2f8f0e), // -4.286103696e-2
    f32::from_bits(0xbcfff290), // -3.124359250e-2
    f32::from_bits(0xbc1a0f04), // -9.402994066e-3
    f32::from_bits(0x3c18b7fa), // +9.321207181e-3
    f32::from_bits(0x3c8e857b), // +1.739763282e-2
    f32::from_bits(0x3c701115), // +1.465251017e-2
    f32::from_bits(0x3bc6065e), // +6.043239497e-3
    f32::from_bits(0xbb18e2ae), // -2.332847100e-3
    f32::from_bits(0xbbd72768), // -6.565976888e-3
    f32::from_bits(0xbbc8fdde), // -6.133778952e-3
    f32::from_bits(0xbb448500), // -2.998650074e-3
    f32::from_bits(0x398b70cd), // +2.659618913e-4
    f32::from_bits(0x3b047459), // +2.021095017e-3
    f32::from_bits(0x3b06a0cd), // +2.054262208e-3
    f32::from_bits(0x3a939d4f), // +1.126209158e-3
    f32::from_bits(0x39248b21), // +1.569208835e-4
    f32::from_bits(0xb9b24159), // -3.399949346e-4
    f32::from_bits(0xb9bfb2f4), // -3.656368935e-4
    f32::from_bits(0xb9353eda), // -1.728491916e-4
];

// $ cargo pmr lowpass --taps 97 --pass 0.03125 --stop 0.09375 --pass-guard 0.1 --stop-guard 0.2
// --decimate 8
//
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 97
// Band frequencies are normalized to sample rate of 1.0
//
// passband:   0.0-0.0312 (guarded to 0.0375)
// stopband:   0.0938-0.5 (guarded from 0.0813)
// transition: 0.0625 wide, 0.0438 after guards
//
// Design Results
// =========================================================
//   weighted error: 0.02150317
//   flatness: 0.00000692
//   iterations: 8
//
// Gain Testing
// =========================================================
//   deep pass band:                            0.99958384
//   pass band:                                 0.99975669
//   new Nyquist:                               0.34913528
//   upper passband fold mirror:                0.00014459
//   deep passband fold mirror:                 0.00005380
//   stop band:                                 0.00014459
//   deep stop band:                            0.00006923
pub(crate) const DOWN_EIGHT: [f32; 97] = [
    f32::from_bits(0xb8b975e0), // -8.843443356e-5
    f32::from_bits(0xb9145ef5), // -1.414975413e-4
    f32::from_bits(0xb93e3b79), // -1.814196730e-4
    f32::from_bits(0xb953d40f), // -2.020152606e-4
    f32::from_bits(0xb93a561a), // -1.777041762e-4
    f32::from_bits(0xb8c38469), // -9.322987898e-5
    f32::from_bits(0x3881b9a4), // +6.185777602e-5
    f32::from_bits(0x3993bdb4), // +2.817936474e-4
    f32::from_bits(0x3a0edf38), // +5.450132303e-4
    f32::from_bits(0x3a5460cb), // +8.101581479e-4
    f32::from_bits(0x3a85e08b), // +1.021401375e-3
    f32::from_bits(0x3a9202a4), // +1.113970298e-3
    f32::from_bits(0x3a869d08), // +1.027018763e-3
    f32::from_bits(0x3a3bc7f6), // +7.163280388e-4
    f32::from_bits(0x3931e25f), // +1.696436520e-4
    f32::from_bits(0xba18b7e5), // -5.825742264e-4
    f32::from_bits(0xbabf68b8), // -1.460335217e-3
    f32::from_bits(0xbb192993), // -2.337072743e-3
    f32::from_bits(0xbb47f40d), // -3.051045584e-3
    f32::from_bits(0xbb60a254), // -3.427644260e-3
    f32::from_bits(0xbb58d0b3), // -3.308337880e-3
    f32::from_bits(0xbb295f9b), // -2.584433882e-3
    f32::from_bits(0xbaa101b5), // -1.228383393e-3
    f32::from_bits(0x3a32bcc0), // +6.818287075e-4
    f32::from_bits(0x3b41827f), // +2.952724462e-3
    f32::from_bits(0x3bad1c5a), // +5.282920785e-3
    f32::from_bits(0x3beeeff9), // +7.291790564e-3
    f32::from_bits(0x3c0c5c34), // +8.566904813e-3
    f32::from_bits(0x3c0efb5b), // +8.726920001e-3
    f32::from_bits(0x3bf57720), // +7.491007447e-3
    f32::from_bits(0x3b9b8212), // +4.745730199e-3
    f32::from_bits(0x3a1c8bcb), // +5.971758510e-4
    f32::from_bits(0xbb96c542), // -4.601151682e-3
    f32::from_bits(0xbc27fe7f), // -1.025354769e-2
    f32::from_bits(0xbc7efd93), // -1.556338649e-2
    f32::from_bits(0xbca0a522), // -1.960999146e-2
    f32::from_bits(0xbcafbf0e), // -2.145340666e-2
    f32::from_bits(0xbca5efef), // -2.025601082e-2
    f32::from_bits(0xbc7c6408), // -1.540470868e-2
    f32::from_bits(0xbbd8d691), // -6.617375184e-3
    f32::from_bits(0x3bc41099), // +5.983423907e-3
    f32::from_bits(0x3cb2ed08), // +2.184154093e-2
    f32::from_bits(0x3d23cde2), // +3.999126703e-2
    f32::from_bits(0x3d72367d), // +5.913399532e-2
    f32::from_bits(0x3d9f4153), // +7.776131481e-2
    f32::from_bits(0x3dc125b4), // +9.431019425e-2
    f32::from_bits(0x3ddbd10c), // +1.073323190e-1
    f32::from_bits(0x3decdce0), // +1.156556606e-1
    f32::from_bits(0x3df2b9ac), // +1.185182035e-1
    f32::from_bits(0x3decdce0), // +1.156556606e-1
    f32::from_bits(0x3ddbd10c), // +1.073323190e-1
    f32::from_bits(0x3dc125b4), // +9.431019425e-2
    f32::from_bits(0x3d9f4153), // +7.776131481e-2
    f32::from_bits(0x3d72367d), // +5.913399532e-2
    f32::from_bits(0x3d23cde2), // +3.999126703e-2
    f32::from_bits(0x3cb2ed08), // +2.184154093e-2
    f32::from_bits(0x3bc41099), // +5.983423907e-3
    f32::from_bits(0xbbd8d691), // -6.617375184e-3
    f32::from_bits(0xbc7c6408), // -1.540470868e-2
    f32::from_bits(0xbca5efef), // -2.025601082e-2
    f32::from_bits(0xbcafbf0e), // -2.145340666e-2
    f32::from_bits(0xbca0a522), // -1.960999146e-2
    f32::from_bits(0xbc7efd93), // -1.556338649e-2
    f32::from_bits(0xbc27fe7f), // -1.025354769e-2
    f32::from_bits(0xbb96c542), // -4.601151682e-3
    f32::from_bits(0x3a1c8bcb), // +5.971758510e-4
    f32::from_bits(0x3b9b8212), // +4.745730199e-3
    f32::from_bits(0x3bf57720), // +7.491007447e-3
    f32::from_bits(0x3c0efb5b), // +8.726920001e-3
    f32::from_bits(0x3c0c5c34), // +8.566904813e-3
    f32::from_bits(0x3beeeff9), // +7.291790564e-3
    f32::from_bits(0x3bad1c5a), // +5.282920785e-3
    f32::from_bits(0x3b41827f), // +2.952724462e-3
    f32::from_bits(0x3a32bcc0), // +6.818287075e-4
    f32::from_bits(0xbaa101b5), // -1.228383393e-3
    f32::from_bits(0xbb295f9b), // -2.584433882e-3
    f32::from_bits(0xbb58d0b3), // -3.308337880e-3
    f32::from_bits(0xbb60a254), // -3.427644260e-3
    f32::from_bits(0xbb47f40d), // -3.051045584e-3
    f32::from_bits(0xbb192993), // -2.337072743e-3
    f32::from_bits(0xbabf68b8), // -1.460335217e-3
    f32::from_bits(0xba18b7e5), // -5.825742264e-4
    f32::from_bits(0x3931e25f), // +1.696436520e-4
    f32::from_bits(0x3a3bc7f6), // +7.163280388e-4
    f32::from_bits(0x3a869d08), // +1.027018763e-3
    f32::from_bits(0x3a9202a4), // +1.113970298e-3
    f32::from_bits(0x3a85e08b), // +1.021401375e-3
    f32::from_bits(0x3a5460cb), // +8.101581479e-4
    f32::from_bits(0x3a0edf38), // +5.450132303e-4
    f32::from_bits(0x3993bdb4), // +2.817936474e-4
    f32::from_bits(0x3881b9a4), // +6.185777602e-5
    f32::from_bits(0xb8c38469), // -9.322987898e-5
    f32::from_bits(0xb93a561a), // -1.777041762e-4
    f32::from_bits(0xb953d40f), // -2.020152606e-4
    f32::from_bits(0xb93e3b79), // -1.814196730e-4
    f32::from_bits(0xb9145ef5), // -1.414975413e-4
    f32::from_bits(0xb8b975e0), // -8.843443356e-5
];
