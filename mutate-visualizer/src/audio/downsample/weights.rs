// Copyright 2026 The MuTate Contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Weights
//!
//! > She heavy.
//! >
//! > - Your dad
//!
//! Weights generated from the packaged PMR tool.  See README and included command examples to
//! re-generate other lengths.  Weights are designed for the shader execution structure.  Details on
//! the filter design in downsample.rs.

// NEXT Design for main lobe widths!  There's a lot of subjective tuning done so far.
// NEXT Proc macro!  The CLI tool is kind of a dumpster fire just good enough to waddle across the
// finish line!  At a minimum, the CLI tool needs to be configured for our use case instead of
// leaving the generation commands in the comments.
// NOTE The final down sampler should always use a flat pass band down to DC.  The --terminal option
// for the PMR tool sets this low-ripple configuration.
// NOTE 16x group delay is about 1/8th of a 60FPS frame.  If we buy more downsampling, we pay for it
// in time that faster attacking wavelets cannot buy back.  Not that any transient can exist that
// outpaces the wavelength of the carrier when perceived by the listener without percussive onsets
// lighting up the screen.
// NEXT Document frequencies at usual input rate!  I can't read 0.03123445 in a hurry!!

/// Low pass filter.  Odd.  Symmetric.
pub(crate) struct Lowpass<const N: usize> {
    read_band: f64,
    guard_band: f64,
    /// Input samples consumed per output sample.
    pub decimation: u32,
    /// Raw weights.
    pub taps: [f32; N],
}

impl<const N: usize> Lowpass<N> {
    /// Use as an iteration count from the outside in.
    pub const fn radius(&self) -> u32 {
        (N as u32 - 1) / 2
    }

    /// Outward-in half of weights, center inclusive.  Works as a zero-to-center or backwards
    /// iterated end-to-center array of weights.
    pub fn folded(&self) -> &[f32] {
        &self.taps[..=self.radius() as usize]
    }

    /// Group delay in *input* samples.  Linear phase, so it's just the center tap index.
    pub const fn delay_in(&self) -> u32 {
        self.radius()
    }

    /// Group delay in *output* samples, post-decimation.
    pub const fn delay_out(&self) -> u32 {
        self.radius() / self.decimation
    }

    /// Cutoff frequency for analysis.  `rate_in` is the pre-decimation sample rate.  Upper passband
    /// edge in cycles/sample of the *input* rate.  Main lobes of filters must not be centered above
    /// this cutoff.
    pub fn cutoff(&self, rate_in: f64) -> f64 {
        self.read_band * rate_in
    }

    /// Cutoff frequency beyond which folded noise begins to appear.  `rate_in` is the
    /// pre-decimation sample rate.  Upper edge of the guarded (solver configuration) band in
    /// cycles/sample of the *input* rate.  Filter lobes that exceed this band will begin to see
    /// folded noise.
    pub fn guarded_cutoff(&self, rate_in: f64) -> f64 {
        self.guard_band * rate_in
    }
}

// NOTE The first filter is intended to allow reads of the bottom half of the new Nyquist.  Old
// Nyquist is half.  New Nyquist is a quarter.  Half of a quarter is an eighth.  All subsequent
// filters provide read bands below half that down to the next filter.

pub(crate) const DOWN_TWO: Lowpass<21> = Lowpass {
    read_band: 0.125,
    guard_band: 0.1375,
    decimation: 2,
    taps: DOWN_TWO_TAPS,
};

pub(crate) const DOWN_FOUR: Lowpass<41> = Lowpass {
    read_band: 0.0625,
    guard_band: 0.06875,
    decimation: 4,
    taps: DOWN_FOUR_TAPS,
};

pub(crate) const DOWN_EIGHT: Lowpass<81> = Lowpass {
    read_band: 0.03125,
    guard_band: 0.034375,
    decimation: 8,
    taps: DOWN_EIGHT_TAPS,
};

pub(crate) const DOWN_SIXTEEN: Lowpass<161> = Lowpass {
    read_band: 0.015625,
    guard_band: 0.0171875,
    decimation: 16,
    taps: DOWN_SIXTEEN_TAPS,
};

// cargo pmr lowpass --taps 21 --decimate 2
//
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 21
// decimation: 2x
// output rate:    0.5000  (target Nyquist 0.2500)
//
// All fractions in cycles/sample of the input rate (Nyquist = 0.5).
// passband:   0.0-0.1250 (guarded through 0.1375)
// shoulder:   0.1375-0.2500 (lands in unread band)
// fold:       0.2500-0.3625 (folds into unread band)
// stopband:   0.3625-0.5 (folds into passband)
// transition: 0.2250 wide
// weighting:  non-terminal (DC ripple relaxed)
//
// Design Results
// =========================================================
//   weighted error: 0.16595566
//   flatness: 0.00000000
//   iterations: 11
//
// Gain Testing
// =========================================================
//   deep pass band:                            1.00052061     +0.00 dB
//   pass band:                                 1.00157115     +0.01 dB
//   new Nyquist:                               0.16595566    -15.60 dB
//   upper passband fold mirror:                0.00001381    -97.19 dB
//   deep passband fold mirror:                 0.00000825   -101.67 dB
//   stop band:                                 0.00001660    -95.60 dB
//   deep stop band:                            0.00001462    -96.70 dB
//
const DOWN_TWO_TAPS: [f32; 21] = [
    f32::from_bits(0xba0fd4b3), // -5.486711743e-4
    f32::from_bits(0xbb7cc388), // -3.856869414e-3
    f32::from_bits(0xbc034f81), // -8.014560677e-3
    f32::from_bits(0xba0fc98c), // -5.485049915e-4
    f32::from_bits(0x3c9cc336), // +1.913605258e-2
    f32::from_bits(0x3c4a691d), // +1.235416252e-2
    f32::from_bits(0xbd37ed85), // -4.490425065e-2
    f32::from_bits(0xbd8ffdbe), // -7.030819356e-2
    f32::from_bits(0x3d71cd5b), // +5.903373286e-2
    f32::from_bits(0x3e99e477), // +3.005711734e-1
    f32::from_bits(0x3edaa466), // +4.270355105e-1
    f32::from_bits(0x3e99e477), // +3.005711734e-1
    f32::from_bits(0x3d71cd5b), // +5.903373286e-2
    f32::from_bits(0xbd8ffdbe), // -7.030819356e-2
    f32::from_bits(0xbd37ed85), // -4.490425065e-2
    f32::from_bits(0x3c4a691d), // +1.235416252e-2
    f32::from_bits(0x3c9cc336), // +1.913605258e-2
    f32::from_bits(0xba0fc98c), // -5.485049915e-4
    f32::from_bits(0xbc034f81), // -8.014560677e-3
    f32::from_bits(0xbb7cc388), // -3.856869414e-3
    f32::from_bits(0xba0fd4b3), // -5.486711743e-4
];

// $ cargo pmr lowpass --taps 41 --decimate 4
//
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 41
// decimation: 4x
// output rate:    0.2500  (target Nyquist 0.1250)
//
// All fractions in cycles/sample of the input rate (Nyquist = 0.5).
// passband:   0.0-0.0625 (guarded through 0.0688)
// shoulder:   0.0688-0.1250 (lands in unread band)
// fold:       0.1250-0.1812 (folds into unread band)
// stopband:   0.1812-0.5 (folds into passband)
// transition: 0.1125 wide
// weighting:  non-terminal (DC ripple relaxed)
//
// Design Results
// =========================================================
//   weighted error: 0.18469634
//   flatness: 0.00000000
//   iterations: 13
//
// Gain Testing
// =========================================================
//   deep pass band:                            1.00190087     +0.02 dB
//   pass band:                                 1.00177958     +0.02 dB
//   new Nyquist:                               0.18469634    -14.67 dB
//   upper passband fold mirror:                0.00001640    -95.70 dB
//   deep passband fold mirror:                 0.00001187    -98.51 dB
//   stop band:                                 0.00001848    -94.67 dB
//   deep stop band:                            0.00001780    -94.99 dB
//
const DOWN_FOUR_TAPS: [f32; 41] = [
    f32::from_bits(0xb90a37e9), // -1.318153372e-4
    f32::from_bits(0xba136d02), // -5.623848410e-4
    f32::from_bits(0xbabb8a7e), // -1.430824166e-3
    f32::from_bits(0xbb29cf4f), // -2.591091907e-3
    f32::from_bits(0xbb608f86), // -3.426523414e-3
    f32::from_bits(0xbb409d97), // -2.939080587e-3
    f32::from_bits(0xb9919558), // -2.776782494e-4
    f32::from_bits(0x3b911e8d), // +4.428690765e-3
    f32::from_bits(0x3c1a5fcf), // +9.422256611e-3
    f32::from_bits(0x3c3c4087), // +1.148999389e-2
    f32::from_bits(0x3befb4d7), // +7.315258961e-3
    f32::from_bits(0xbb8d9335), // -4.320526961e-3
    f32::from_bits(0xbca85d3d), // -2.055227198e-2
    f32::from_bits(0xbd0b51dc), // -3.401361406e-2
    f32::from_bits(0xbd0eac8e), // -3.483252972e-2
    f32::from_bits(0xbc70c20e), // -1.469470374e-2
    f32::from_bits(0x3ceac9b0), // +2.866062522e-2
    f32::from_bits(0x3db5cb9b), // +8.876725286e-2
    f32::from_bits(0x3e1aecf8), // +1.512945890e-1
    f32::from_bits(0x3e4b5b7b), // +1.985911578e-1
    f32::from_bits(0x3e5d6659), // +2.162107378e-1
    f32::from_bits(0x3e4b5b7b), // +1.985911578e-1
    f32::from_bits(0x3e1aecf8), // +1.512945890e-1
    f32::from_bits(0x3db5cb9b), // +8.876725286e-2
    f32::from_bits(0x3ceac9b0), // +2.866062522e-2
    f32::from_bits(0xbc70c20e), // -1.469470374e-2
    f32::from_bits(0xbd0eac8e), // -3.483252972e-2
    f32::from_bits(0xbd0b51dc), // -3.401361406e-2
    f32::from_bits(0xbca85d3d), // -2.055227198e-2
    f32::from_bits(0xbb8d9335), // -4.320526961e-3
    f32::from_bits(0x3befb4d7), // +7.315258961e-3
    f32::from_bits(0x3c3c4087), // +1.148999389e-2
    f32::from_bits(0x3c1a5fcf), // +9.422256611e-3
    f32::from_bits(0x3b911e8d), // +4.428690765e-3
    f32::from_bits(0xb9919558), // -2.776782494e-4
    f32::from_bits(0xbb409d97), // -2.939080587e-3
    f32::from_bits(0xbb608f86), // -3.426523414e-3
    f32::from_bits(0xbb29cf4f), // -2.591091907e-3
    f32::from_bits(0xbabb8a7e), // -1.430824166e-3
    f32::from_bits(0xba136d02), // -5.623848410e-4
    f32::from_bits(0xb90a37e9), // -1.318153372e-4
];

// $ cargo pmr lowpass --taps 81 --decimate 8
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 81
// decimation: 8x
// output rate:    0.1250  (target Nyquist 0.0625)
//
// All fractions in cycles/sample of the input rate (Nyquist = 0.5).
// passband:   0.0-0.0312 (guarded through 0.0344)
// shoulder:   0.0344-0.0625 (lands in unread band)
// fold:       0.0625-0.0906 (folds into unread band)
// stopband:   0.0906-0.5 (folds into passband)
// transition: 0.0562 wide
// weighting:  non-terminal (DC ripple relaxed)
//
// Design Results
// =========================================================
//   weighted error: 0.20429698
//   flatness: 0.00000000
//   iterations: 18
//
// Gain Testing
// =========================================================
//   deep pass band:                            1.00164373     +0.01 dB
//   pass band:                                 1.00198054     +0.02 dB
//   new Nyquist:                               0.20429697    -13.79 dB
//   upper passband fold mirror:                0.00001973    -94.10 dB
//   deep passband fold mirror:                 0.00001806    -94.86 dB
//   stop band:                                 0.00002044    -93.79 dB
//   deep stop band:                            0.00001192    -98.47 dB
//
const DOWN_EIGHT_TAPS: [f32; 81] = [
    f32::from_bits(0xb81d07f1), // -3.743911293e-5
    f32::from_bits(0xb8c14437), // -9.215663158e-5
    f32::from_bits(0xb9485f9a), // -1.910910069e-4
    f32::from_bits(0xb9b26e11), // -3.403281153e-4
    f32::from_bits(0xba0e0eae), // -5.419057561e-4
    f32::from_bits(0xba4e41a8), // -7.868059911e-4
    f32::from_bits(0xba89d0ee), // -1.051453641e-3
    f32::from_bits(0xbaa9e09a), // -1.296061324e-3
    f32::from_bits(0xbac02423), // -1.465920708e-3
    f32::from_bits(0xbac4246a), // -1.496446552e-3
    f32::from_bits(0xbaad4e5e), // -1.322220778e-3
    f32::from_bits(0xba692dab), // -8.895049687e-4
    f32::from_bits(0xb9331a9e), // -1.708068594e-4
    f32::from_bits(0x3a57272d), // +8.207436767e-4
    f32::from_bits(0x3b0462bb), // +2.020044951e-3
    f32::from_bits(0x3b5893e6), // +3.304713871e-3
    f32::from_bits(0x3b9367f1), // +4.498474766e-3
    f32::from_bits(0x3bb07453), // +5.384960677e-3
    f32::from_bits(0x3bbbd504), // +5.732180551e-3
    f32::from_bits(0x3bae8695), // +5.326102022e-3
    f32::from_bits(0x3b83656a), // +4.009892233e-3
    f32::from_bits(0x3ae1ee62), // +1.723718131e-3
    f32::from_bits(0xbabf818f), // -1.461075502e-3
    f32::from_bits(0xbbae51e2), // -5.319819786e-3
    f32::from_bits(0xbc1b282f), // -9.470029734e-3
    f32::from_bits(0xbc5b5ae2), // -1.338836737e-2
    f32::from_bits(0xbc86c28a), // -1.645018533e-2
    f32::from_bits(0xbc935eca), // -1.798953488e-2
    f32::from_bits(0xbc8e5479), // -1.737426408e-2
    f32::from_bits(0xbc66d0f9), // -1.408790890e-2
    f32::from_bits(0xbbffdbd1), // -7.808186579e-3
    f32::from_bits(0x3ac86038), // +1.528746448e-3
    f32::from_bits(0x3c604461), // +1.368817780e-2
    f32::from_bits(0x3ce66d7d), // +2.812837996e-2
    f32::from_bits(0x3d345648), // +4.402759671e-2
    f32::from_bits(0x3d772d31), // +6.034583226e-2
    f32::from_bits(0x3d9b79e1), // +7.591605932e-2
    f32::from_bits(0x3db768d1), // +8.955539018e-2
    f32::from_bits(0x3dcd2d0f), // +1.001835987e-1
    f32::from_bits(0x3ddb00ba), // +1.069349796e-1
    f32::from_bits(0x3ddfbe6e), // +1.092499346e-1
    f32::from_bits(0x3ddb00ba), // +1.069349796e-1
    f32::from_bits(0x3dcd2d0f), // +1.001835987e-1
    f32::from_bits(0x3db768d1), // +8.955539018e-2
    f32::from_bits(0x3d9b79e1), // +7.591605932e-2
    f32::from_bits(0x3d772d31), // +6.034583226e-2
    f32::from_bits(0x3d345648), // +4.402759671e-2
    f32::from_bits(0x3ce66d7d), // +2.812837996e-2
    f32::from_bits(0x3c604461), // +1.368817780e-2
    f32::from_bits(0x3ac86038), // +1.528746448e-3
    f32::from_bits(0xbbffdbd1), // -7.808186579e-3
    f32::from_bits(0xbc66d0f9), // -1.408790890e-2
    f32::from_bits(0xbc8e5479), // -1.737426408e-2
    f32::from_bits(0xbc935eca), // -1.798953488e-2
    f32::from_bits(0xbc86c28a), // -1.645018533e-2
    f32::from_bits(0xbc5b5ae2), // -1.338836737e-2
    f32::from_bits(0xbc1b282f), // -9.470029734e-3
    f32::from_bits(0xbbae51e2), // -5.319819786e-3
    f32::from_bits(0xbabf818f), // -1.461075502e-3
    f32::from_bits(0x3ae1ee62), // +1.723718131e-3
    f32::from_bits(0x3b83656a), // +4.009892233e-3
    f32::from_bits(0x3bae8695), // +5.326102022e-3
    f32::from_bits(0x3bbbd504), // +5.732180551e-3
    f32::from_bits(0x3bb07453), // +5.384960677e-3
    f32::from_bits(0x3b9367f1), // +4.498474766e-3
    f32::from_bits(0x3b5893e6), // +3.304713871e-3
    f32::from_bits(0x3b0462bb), // +2.020044951e-3
    f32::from_bits(0x3a57272d), // +8.207436767e-4
    f32::from_bits(0xb9331a9e), // -1.708068594e-4
    f32::from_bits(0xba692dab), // -8.895049687e-4
    f32::from_bits(0xbaad4e5e), // -1.322220778e-3
    f32::from_bits(0xbac4246a), // -1.496446552e-3
    f32::from_bits(0xbac02423), // -1.465920708e-3
    f32::from_bits(0xbaa9e09a), // -1.296061324e-3
    f32::from_bits(0xba89d0ee), // -1.051453641e-3
    f32::from_bits(0xba4e41a8), // -7.868059911e-4
    f32::from_bits(0xba0e0eae), // -5.419057561e-4
    f32::from_bits(0xb9b26e11), // -3.403281153e-4
    f32::from_bits(0xb9485f9a), // -1.910910069e-4
    f32::from_bits(0xb8c14437), // -9.215663158e-5
    f32::from_bits(0xb81d07f1), // -3.743911293e-5
];

// $ cargo pmr lowpass --taps 161 --decimate 16 --terminal
//
// Optimal Lowpass FIR Weights:
// =========================================================
// Filter length: 161
// decimation: 16x
// output rate:    0.0625  (target Nyquist 0.0312)
//
// All fractions in cycles/sample of the input rate (Nyquist = 0.5).
// passband:   0.0-0.0156 (guarded through 0.0172)
// shoulder:   0.0172-0.0312 (lands in unread band)
// fold:       0.0312-0.0453 (folds into unread band)
// stopband:   0.0453-0.5 (folds into passband)
// transition: 0.0281 wide
// weighting:  terminal (flat down to DC)
//
// Design Results
// =========================================================
//   weighted error: 0.22646222
//   flatness: 0.00000000
//   iterations: 14
//
// Gain Testing
// =========================================================
//   deep pass band:                            1.00080161     +0.01 dB
//   pass band:                                 1.00250371     +0.02 dB
//   new Nyquist:                               0.22646221    -12.90 dB
//   upper passband fold mirror:                0.00002263    -92.91 dB
//   deep passband fold mirror:                 0.00002265    -92.90 dB
//   stop band:                                 0.00002264    -92.90 dB
//   deep stop band:                            0.00001941    -94.24 dB
//
const DOWN_SIXTEEN_TAPS: [f32; 161] = [
    f32::from_bits(0x362dc5e3), // +2.589419410e-6
    f32::from_bits(0xb7956429), // -1.780882485e-5
    f32::from_bits(0xb7c77ab3), // -2.377978490e-5
    f32::from_bits(0xb81e039a), // -3.767348971e-5
    f32::from_bits(0xb86e42c1), // -5.680579125e-5
    f32::from_bits(0xb8aaab15), // -8.138098201e-5
    f32::from_bits(0xb8ea813e), // -1.118206274e-4
    f32::from_bits(0xb91b96b5), // -1.483809465e-4
    f32::from_bits(0xb9485376), // -1.910457795e-4
    f32::from_bits(0xb97b1612), // -2.394544717e-4
    f32::from_bits(0xb9998837), // -2.928392205e-4
    f32::from_bits(0xb9b77c95), // -3.499730083e-4
    f32::from_bits(0xb9d680d4), // -4.091324518e-4
    f32::from_bits(0xb9f568b0), // -4.680803977e-4
    f32::from_bits(0xba0961ef), // -5.240728497e-4
    f32::from_bits(0xba167160), // -5.738940090e-4
    f32::from_bits(0xba20efa1), // -6.139223115e-4
    f32::from_bits(0xba27d520), // -6.402302533e-4
    f32::from_bits(0xba2a0ea2), // -6.487165811e-4
    f32::from_bits(0xba26885c), // -6.352716591e-4
    f32::from_bits(0xba1c3af4), // -5.959712435e-4
    f32::from_bits(0xba0a3a22), // -5.272944691e-4
    f32::from_bits(0xb9df88dc), // -4.263584269e-4
    f32::from_bits(0xb998a707), // -2.911614429e-4
    f32::from_bits(0xb8fd62a4), // -1.208235335e-4
    f32::from_bits(0x38b08d88), // +8.418696234e-5
    f32::from_bits(0x39a8be70), // +3.218534403e-4
    f32::from_bits(0x3a1a4243), // +5.884507555e-4
    f32::from_bits(0x3a6646f3), // +8.784375968e-4
    f32::from_bits(0x3a9b3e4d), // +1.184412860e-3
    f32::from_bits(0x3ac43bf4), // +1.497148070e-3
    f32::from_bits(0x3aecad4c), // +1.805701759e-3
    f32::from_bits(0x3b09784d), // +2.097624587e-3
    f32::from_bits(0x3b1a9db9), // +2.359254519e-3
    f32::from_bits(0x3b28d3d8), // +2.576103434e-3
    f32::from_bits(0x3b3321a0), // +2.733327448e-3
    f32::from_bits(0x3b389138), // +2.816272900e-3
    f32::from_bits(0x3b383a2c), // +2.811084501e-3
    f32::from_bits(0x3b314c54), // +2.705355175e-3
    f32::from_bits(0x3b231b21), // +2.488799626e-3
    f32::from_bits(0x3b0d28e2), // +2.153926063e-3
    f32::from_bits(0x3ade630a), // +1.696677180e-3
    f32::from_bits(0x3a9268db), // +1.117016538e-3
    f32::from_bits(0x39dbe6ba), // +4.194283974e-4
    f32::from_bits(0xb9cabd2e), // -3.866939223e-4
    f32::from_bits(0xbaa8a9c1), // -1.286797342e-3
    f32::from_bits(0xbb142e97), // -2.261077752e-3
    f32::from_bits(0xbb5741e0), // -3.284566104e-3
    f32::from_bits(0xbb8dcce2), // -4.327402450e-3
    f32::from_bits(0xbbaf7b91), // -5.355306435e-3
    f32::from_bits(0xbbcf6de7), // -6.330240052e-3
    f32::from_bits(0xbbec4c65), // -7.211255375e-3
    f32::from_bits(0xbc0257d7), // -7.955512963e-3
    f32::from_bits(0xbc0b9523), // -8.519443683e-3
    f32::from_bits(0xbc1129a7), // -8.860028349e-3
    f32::from_bits(0xbc1268f0), // -8.936151862e-3
    f32::from_bits(0xbc0eb463), // -8.709999733e-3
    f32::from_bits(0xbc05810d), // -8.148443885e-3
    f32::from_bits(0xbbecba66), // -7.224368863e-3
    f32::from_bits(0xbbc1eafb), // -5.917904433e-3
    f32::from_bits(0xbb8a32ed), // -4.217496607e-3
    f32::from_bits(0xbb0afcf7), // -2.120790770e-3
    f32::from_bits(0x39bf37fe), // +3.647207632e-4
    f32::from_bits(0x3b531cb9), // +3.221316496e-3
    f32::from_bits(0x3bd266e2), // +6.420955993e-3
    f32::from_bits(0x3c229e8b), // +9.925494902e-3
    f32::from_bits(0x3c604056), // +1.368721388e-2
    f32::from_bits(0x3c909601), // +1.764965244e-2
    f32::from_bits(0x3cb22a67), // +2.174873464e-2
    f32::from_bits(0x3cd449f1), // +2.591416426e-2
    f32::from_bits(0x3cf6578d), // +3.007104434e-2
    f32::from_bits(0x3d0bd827), // +3.414168581e-2
    f32::from_bits(0x3d1bd7bf), // +3.804754838e-2
    f32::from_bits(0x3d2ad969), // +4.171124473e-2
    f32::from_bits(0x3d388f5b), // +4.505858943e-2
    f32::from_bits(0x3d44b136), // +4.802056402e-2
    f32::from_bits(0x3d4efdfd), // +5.053519085e-2
    f32::from_bits(0x3d573dda), // +5.254922062e-2
    f32::from_bits(0x3d5d43af), // +5.401962623e-2
    f32::from_bits(0x3d60ee51), // +5.491477624e-2
    f32::from_bits(0x3d622977), // +5.521532521e-2
    f32::from_bits(0x3d60ee51), // +5.491477624e-2
    f32::from_bits(0x3d5d43af), // +5.401962623e-2
    f32::from_bits(0x3d573dda), // +5.254922062e-2
    f32::from_bits(0x3d4efdfd), // +5.053519085e-2
    f32::from_bits(0x3d44b136), // +4.802056402e-2
    f32::from_bits(0x3d388f5b), // +4.505858943e-2
    f32::from_bits(0x3d2ad969), // +4.171124473e-2
    f32::from_bits(0x3d1bd7bf), // +3.804754838e-2
    f32::from_bits(0x3d0bd827), // +3.414168581e-2
    f32::from_bits(0x3cf6578d), // +3.007104434e-2
    f32::from_bits(0x3cd449f1), // +2.591416426e-2
    f32::from_bits(0x3cb22a67), // +2.174873464e-2
    f32::from_bits(0x3c909601), // +1.764965244e-2
    f32::from_bits(0x3c604056), // +1.368721388e-2
    f32::from_bits(0x3c229e8b), // +9.925494902e-3
    f32::from_bits(0x3bd266e2), // +6.420955993e-3
    f32::from_bits(0x3b531cb9), // +3.221316496e-3
    f32::from_bits(0x39bf37fe), // +3.647207632e-4
    f32::from_bits(0xbb0afcf7), // -2.120790770e-3
    f32::from_bits(0xbb8a32ed), // -4.217496607e-3
    f32::from_bits(0xbbc1eafb), // -5.917904433e-3
    f32::from_bits(0xbbecba66), // -7.224368863e-3
    f32::from_bits(0xbc05810d), // -8.148443885e-3
    f32::from_bits(0xbc0eb463), // -8.709999733e-3
    f32::from_bits(0xbc1268f0), // -8.936151862e-3
    f32::from_bits(0xbc1129a7), // -8.860028349e-3
    f32::from_bits(0xbc0b9523), // -8.519443683e-3
    f32::from_bits(0xbc0257d7), // -7.955512963e-3
    f32::from_bits(0xbbec4c65), // -7.211255375e-3
    f32::from_bits(0xbbcf6de7), // -6.330240052e-3
    f32::from_bits(0xbbaf7b91), // -5.355306435e-3
    f32::from_bits(0xbb8dcce2), // -4.327402450e-3
    f32::from_bits(0xbb5741e0), // -3.284566104e-3
    f32::from_bits(0xbb142e97), // -2.261077752e-3
    f32::from_bits(0xbaa8a9c1), // -1.286797342e-3
    f32::from_bits(0xb9cabd2e), // -3.866939223e-4
    f32::from_bits(0x39dbe6ba), // +4.194283974e-4
    f32::from_bits(0x3a9268db), // +1.117016538e-3
    f32::from_bits(0x3ade630a), // +1.696677180e-3
    f32::from_bits(0x3b0d28e2), // +2.153926063e-3
    f32::from_bits(0x3b231b21), // +2.488799626e-3
    f32::from_bits(0x3b314c54), // +2.705355175e-3
    f32::from_bits(0x3b383a2c), // +2.811084501e-3
    f32::from_bits(0x3b389138), // +2.816272900e-3
    f32::from_bits(0x3b3321a0), // +2.733327448e-3
    f32::from_bits(0x3b28d3d8), // +2.576103434e-3
    f32::from_bits(0x3b1a9db9), // +2.359254519e-3
    f32::from_bits(0x3b09784d), // +2.097624587e-3
    f32::from_bits(0x3aecad4c), // +1.805701759e-3
    f32::from_bits(0x3ac43bf4), // +1.497148070e-3
    f32::from_bits(0x3a9b3e4d), // +1.184412860e-3
    f32::from_bits(0x3a6646f3), // +8.784375968e-4
    f32::from_bits(0x3a1a4243), // +5.884507555e-4
    f32::from_bits(0x39a8be70), // +3.218534403e-4
    f32::from_bits(0x38b08d88), // +8.418696234e-5
    f32::from_bits(0xb8fd62a4), // -1.208235335e-4
    f32::from_bits(0xb998a707), // -2.911614429e-4
    f32::from_bits(0xb9df88dc), // -4.263584269e-4
    f32::from_bits(0xba0a3a22), // -5.272944691e-4
    f32::from_bits(0xba1c3af4), // -5.959712435e-4
    f32::from_bits(0xba26885c), // -6.352716591e-4
    f32::from_bits(0xba2a0ea2), // -6.487165811e-4
    f32::from_bits(0xba27d520), // -6.402302533e-4
    f32::from_bits(0xba20efa1), // -6.139223115e-4
    f32::from_bits(0xba167160), // -5.738940090e-4
    f32::from_bits(0xba0961ef), // -5.240728497e-4
    f32::from_bits(0xb9f568b0), // -4.680803977e-4
    f32::from_bits(0xb9d680d4), // -4.091324518e-4
    f32::from_bits(0xb9b77c95), // -3.499730083e-4
    f32::from_bits(0xb9998837), // -2.928392205e-4
    f32::from_bits(0xb97b1612), // -2.394544717e-4
    f32::from_bits(0xb9485376), // -1.910457795e-4
    f32::from_bits(0xb91b96b5), // -1.483809465e-4
    f32::from_bits(0xb8ea813e), // -1.118206274e-4
    f32::from_bits(0xb8aaab15), // -8.138098201e-5
    f32::from_bits(0xb86e42c1), // -5.680579125e-5
    f32::from_bits(0xb81e039a), // -3.767348971e-5
    f32::from_bits(0xb7c77ab3), // -2.377978490e-5
    f32::from_bits(0xb7956429), // -1.780882485e-5
    f32::from_bits(0x362dc5e3), // +2.589419410e-6
];
