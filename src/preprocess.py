"""Signal preprocessing: resample -> band-pass -> crop/pad to 10 s -> QC flags.

Returns TWO versions (see M1_IMPLEMENTATION_PLAN 2.4): `signal_mv` (filtered, still in mV, used for
amplitudes and delineation) and `signal_z` (per-lead z-score, used for spectral/wavelet and plots).
Short records are zero-padded to 10 s; `valid_len` says how many samples are real, and feature
code only ever looks at `[:valid_len]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from math import gcd

TARGET_FS = 500
TARGET_LEN = 10 * TARGET_FS
BAND = (0.5, 40.0)
FLAT_STD_MV = 0.01          # a lead whose std is below this is "flat"
NOISY_HF_RATIO = 0.25       # >25% of lead power above 40 Hz => "noisy"


@dataclass
class Preprocessed:
    signal_mv: np.ndarray    # (5000, n_leads) filtered, mV
    signal_z: np.ndarray     # (5000, n_leads) per-lead z-score
    valid_len: int           # real samples (<= 5000); rest is zero padding
    qc: dict                 # qc_* flags / ratios


def resample(sig: np.ndarray, fs: float, target: int = TARGET_FS) -> np.ndarray:
    if int(round(fs)) == target:
        return sig
    g = gcd(int(round(fs)), target)
    return resample_poly(sig, target // g, int(round(fs)) // g, axis=0)


def bandpass(sig: np.ndarray, fs: float = TARGET_FS, band=BAND, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth. The 0.5 Hz high-pass doubles as baseline-wander removal."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def _hf_baseline_ratios(raw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead fraction of power above 40 Hz and below 0.5 Hz in the (cropped) raw signal."""
    f, p = welch(raw, fs=fs, nperseg=min(1024, len(raw)), axis=0)
    tot = p.sum(axis=0) + 1e-12
    return p[f > BAND[1]].sum(axis=0) / tot, p[f < BAND[0]].sum(axis=0) / tot


def preprocess(sig: np.ndarray, fs: float) -> Preprocessed:
    sig = np.asarray(sig, dtype=np.float64)
    nan_frac = float(np.isnan(sig).mean())
    if nan_frac:
        sig = np.where(np.isnan(sig), 0.0, sig)
    sig = resample(sig, fs)
    sig = sig[:TARGET_LEN]                       # crop to the first 10 s
    valid = len(sig)
    hf, bw = _hf_baseline_ratios(sig, TARGET_FS) if valid >= 256 else (np.zeros(sig.shape[1]),) * 2
    filt = bandpass(sig) if valid > 30 else sig
    flat = filt.std(axis=0) < FLAT_STD_MV
    mv = np.zeros((TARGET_LEN, sig.shape[1]))
    mv[:valid] = filt
    sd = filt.std(axis=0)
    z = np.zeros_like(mv)
    z[:valid] = (filt - filt.mean(axis=0)) / np.where(sd < 1e-9, 1.0, sd)
    qc = {
        "qc_n_flat_leads": int(flat.sum()),
        "qc_n_noisy_leads": int((hf > NOISY_HF_RATIO).sum()),
        "qc_max_hf_ratio": float(hf.max()),
        "qc_max_baseline_ratio": float(bw.max()),
        "qc_nan_frac": nan_frac,
        "qc_orig_fs": float(fs),
        "qc_valid_s": valid / TARGET_FS,
        "qc_padded": bool(valid < TARGET_LEN),
        "qc_flat_leads": [int(i) for i in np.where(flat)[0]],
    }
    return Preprocessed(mv, z, valid, qc)
