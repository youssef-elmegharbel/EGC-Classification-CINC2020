"""Hand-crafted feature extraction (plan section 2.1): ~130 columns, one row per record.

* `FEATURES` is the registry (name, group, description, unit); the data dictionary is generated
  from it, and `extract_record` is asserted to emit exactly these columns.
* Delineation runs ONCE (neurokit2 DWT) on lead II; if it fails, retried on V1 then I. The fiducial
  points are reused to read amplitudes on all 12 leads. Failures give NaN, never a dropped record.
* `extract_record` never raises. Each feature group is wrapped separately.
* `extract_many` caches one json per record under results/cache/features_v{VERSION}/ (resumable).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.signal import welch

from .loaders import LEAD_NAMES, RESULTS_DIR, read_signal
from .preprocess import TARGET_FS, preprocess

VERSION = 2
CACHE_DIR = RESULTS_DIR / "cache" / f"features_v{VERSION}"
LEAD_PRIORITY = [1, 6, 0]  # II, V1, I
FS = TARGET_FS


@dataclass(frozen=True)
class Feature:
    name: str
    group: str
    description: str
    unit: str


def _build_registry() -> list[Feature]:
    f: list[Feature] = []
    r = lambda n, d, u: f.append(Feature(n, "rhythm", d, u))
    r("hr_mean", "Mean heart rate (60 / mean RR)", "bpm")
    r("rr_mean", "Mean RR interval", "ms")
    r("rr_median", "Median RR interval", "ms")
    r("rr_std", "Std of RR intervals (SDNN)", "ms")
    r("rr_min", "Shortest RR interval", "ms")
    r("rr_max", "Longest RR interval", "ms")
    r("rr_cv", "Coefficient of variation of RR (std / mean)", "ratio")
    r("rmssd", "Root mean square of successive RR differences", "ms")
    r("pnn50", "Fraction of successive RR differences > 50 ms", "ratio")
    r("rr_irregularity", "Mean |successive RR diff| / mean RR (AF marker)", "ratio")
    r("rr_short_frac", "Fraction of RR < 0.8 x median RR (premature beats, PAC marker)", "ratio")
    r("n_beats", "Number of detected R peaks in the 10 s window", "count")
    i = lambda n, d, u: f.append(Feature(n, "intervals", d, u))
    i("pr_interval", "PR interval (P onset to QRS onset), median over beats", "ms")
    i("qrs_duration", "QRS duration (R onset to R offset), median over beats", "ms")
    i("qt_interval", "QT interval (QRS onset to T offset), median over beats", "ms")
    i("qtc_bazett", "QT corrected by Bazett: QT / sqrt(RR in s)", "ms")
    i("p_duration", "P-wave duration (P onset to offset), median over beats", "ms")
    i("p_detected_frac", "Fraction of beats with a prominent P wave (baseline-corrected |amp| >= 0.025 mV; low in AF)", "ratio")
    for ln in LEAD_NAMES:
        for suffix, d, u in [
            ("r_amp", "max within QRS window (R wave)", "mV"),
            ("s_amp", "min within QRS window (S wave)", "mV"),
            ("t_amp", "signed amplitude at T peak", "mV"),
            ("p_amp", "signed amplitude at P peak", "mV"),
            ("std", "signal standard deviation", "mV"),
            ("skew", "signal skewness", ""),
            ("kurt", "signal excess kurtosis", ""),
            ("energy", "mean squared amplitude", "mV^2"),
        ]:
            f.append(Feature(f"{ln}_{suffix}", "per_lead", f"Lead {ln}: {d}"
                             + (" (median over beats)" if suffix.endswith("_amp") else ""), u))
    for k, lo, hi in [(0, 0.5, 5), (1, 5, 15), (2, 15, 25), (3, 25, 40)]:
        f.append(Feature(f"spec_band{k}_rel", "spectral", f"Lead II relative Welch power {lo}-{hi} Hz", "ratio"))
    f.append(Feature("spec_entropy", "spectral", "Lead II spectral entropy (0.5-40 Hz)", "ratio"))
    f.append(Feature("spec_dom_freq", "spectral", "Lead II dominant frequency (0.5-40 Hz)", "Hz"))
    f.append(Feature("spec_centroid", "spectral", "Lead II spectral centroid (0.5-40 Hz)", "Hz"))
    for lv in range(1, 6):
        f.append(Feature(f"wav_d{lv}_rel", "wavelet", f"Lead II db4 detail level {lv} relative energy", "ratio"))
    f.append(Feature("wav_a5_rel", "wavelet", "Lead II db4 level-5 approximation relative energy", "ratio"))
    f.append(Feature("wav_entropy", "wavelet", "Shannon entropy of the wavelet energy distribution", "nats"))
    f.append(Feature("spec_log_power", "spectral", "Lead II log10 total Welch power 0.5-40 Hz (z-scored signal)", "log10"))
    f.append(Feature("age", "patient", "Age from header (NaN if missing)", "years"))
    f.append(Feature("sex_male", "patient", "1 = male, 0 = female, NaN = unknown", "binary"))
    return f


FEATURES = _build_registry()
FEATURE_NAMES = [f.name for f in FEATURES]
DIAG_COLUMNS = ["delineation_ok", "lead_used", "rpeak_ok", "n_beats_diag", "pr_nan", "qrs_nan", "t_nan",
                "p_nan", "error"]


def data_dictionary() -> pd.DataFrame:
    return pd.DataFrame([f.__dict__ for f in FEATURES])


def _nan_row() -> dict:
    return {n: np.nan for n in FEATURE_NAMES}


# ----------------------------------------------------------------------------------------------
# groups
# ----------------------------------------------------------------------------------------------
def _detect_and_delineate(sig_mv: np.ndarray, valid: int):
    """Try II, V1, I. Returns (lead_idx, rpeaks, waves|None). waves None => delineation failed."""
    import neurokit2 as nk

    best = None
    for li in LEAD_PRIORITY:
        x = sig_mv[:valid, li]
        if x.std() < 1e-3:
            continue
        try:
            rp = np.asarray(nk.ecg_peaks(x, sampling_rate=FS, method="neurokit")[1]["ECG_R_Peaks"], int)
        except Exception:
            continue
        if len(rp) < 3:
            continue
        if best is None:
            best = (li, rp, None)
        try:
            _, w = nk.ecg_delineate(x, rp, sampling_rate=FS, method="dwt")
            waves = {k: np.asarray(v, dtype=float) for k, v in w.items()}
            n_ok = np.sum(~np.isnan(waves["ECG_R_Onsets"]) & ~np.isnan(waves["ECG_R_Offsets"]))
            if n_ok >= 2:
                return li, rp, waves
        except Exception:
            pass
    return best if best is not None else (None, None, None)


def _rhythm(rp: np.ndarray) -> dict:
    out = {k: np.nan for k in FEATURE_NAMES[:12]}
    out["n_beats"] = float(len(rp))
    rr = np.diff(rp) / FS * 1000.0
    rr = rr[(rr > 200) & (rr < 3000)]
    if len(rr) < 2:
        return out
    d = np.diff(rr)
    out.update(hr_mean=60000.0 / rr.mean(), rr_mean=rr.mean(), rr_median=float(np.median(rr)),
               rr_std=rr.std(ddof=1), rr_min=rr.min(), rr_max=rr.max(), rr_cv=rr.std(ddof=1) / rr.mean(),
               rmssd=float(np.sqrt(np.mean(d ** 2))), pnn50=float(np.mean(np.abs(d) > 50)),
               rr_irregularity=float(np.mean(np.abs(d)) / rr.mean()),
               rr_short_frac=float(np.mean(rr < 0.8 * np.median(rr))))
    return out


def _med(x: np.ndarray, lo: float, hi: float) -> float:
    x = x[~np.isnan(x)]
    x = x[(x >= lo) & (x <= hi)]
    return float(np.median(x)) if len(x) else np.nan


P_MIN_MV = 0.025


def _intervals(rp: np.ndarray, w: dict | None, rr_mean_ms: float, x: np.ndarray | None = None) -> dict:
    out = {k: np.nan for k in ["pr_interval", "qrs_duration", "qt_interval", "qtc_bazett", "p_duration",
                               "p_detected_frac"]}
    if w is None:
        return out
    ms = 1000.0 / FS
    g = lambda k: w.get(k, np.full(len(rp), np.nan))
    qrs_on, qrs_off = g("ECG_R_Onsets"), g("ECG_R_Offsets")
    p_on, p_off, p_pk = g("ECG_P_Onsets"), g("ECG_P_Offsets"), g("ECG_P_Peaks")
    t_off = g("ECG_T_Offsets")
    out["qrs_duration"] = _med((qrs_off - qrs_on) * ms, 40, 250)
    out["pr_interval"] = _med((qrs_on - p_on) * ms, 60, 600)
    out["qt_interval"] = _med((t_off - qrs_on) * ms, 200, 700)
    out["p_duration"] = _med((p_off - p_on) * ms, 20, 250)
    if len(p_pk) and x is not None:
        # DWT emits a P peak on nearly every beat (even AF), so require a real deflection.
        ok = []
        for pk, a, b in zip(p_pk, p_on, p_off):
            if np.isnan(pk) or np.isnan(a) or np.isnan(b) or not (0 <= a < len(x) and 0 <= b < len(x) and 0 <= pk < len(x)):
                ok.append(False)
            else:
                ok.append(abs(x[int(pk)] - 0.5 * (x[int(a)] + x[int(b)])) >= P_MIN_MV)
        out["p_detected_frac"] = float(np.mean(ok))
    if not np.isnan(out["qt_interval"]) and rr_mean_ms and not np.isnan(rr_mean_ms):
        out["qtc_bazett"] = out["qt_interval"] / np.sqrt(rr_mean_ms / 1000.0)
    return out


def _med_nan(a: np.ndarray) -> np.ndarray:
    """Column-wise nanmedian that returns NaN (no warning) for all-NaN columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(a, axis=0)


def _per_lead(sig_mv: np.ndarray, valid: int, rp: np.ndarray | None, w: dict | None) -> dict:
    x = sig_mv[:valid]
    out = {}
    std, skew, kurt = x.std(axis=0), stats.skew(x, axis=0), stats.kurtosis(x, axis=0)
    energy = (x ** 2).mean(axis=0)
    n_leads = x.shape[1]
    r_amp = s_amp = t_amp = p_amp = np.full(n_leads, np.nan)
    if rp is not None and len(rp):
        half = int(0.05 * FS)
        on = np.full(len(rp), np.nan) if w is None else w.get("ECG_R_Onsets", np.full(len(rp), np.nan))
        off = np.full(len(rp), np.nan) if w is None else w.get("ECG_R_Offsets", np.full(len(rp), np.nan))
        r_l, s_l = [], []
        for b, r0 in enumerate(rp):
            a = int(on[b]) if not np.isnan(on[b]) and 0.02 * FS < r0 - on[b] < 0.2 * FS else r0 - half
            e = int(off[b]) if not np.isnan(off[b]) and 0.02 * FS < off[b] - r0 < 0.2 * FS else r0 + half
            a, e = max(a, 0), min(e + 1, valid)
            if e - a >= 3:
                seg = x[a:e]
                r_l.append(seg.max(axis=0)); s_l.append(seg.min(axis=0))
        if r_l:
            r_amp, s_amp = np.median(r_l, axis=0), np.median(s_l, axis=0)
        if w is not None:
            def at(key):
                idx = w.get(key, np.array([]))
                idx = idx[~np.isnan(idx)].astype(int)
                idx = idx[(idx >= 0) & (idx < valid)]
                return _med_nan(x[idx]) if len(idx) else np.full(n_leads, np.nan)
            t_amp, p_amp = at("ECG_T_Peaks"), at("ECG_P_Peaks")
    for i, ln in enumerate(LEAD_NAMES[:n_leads]):
        out.update({f"{ln}_r_amp": r_amp[i], f"{ln}_s_amp": s_amp[i], f"{ln}_t_amp": t_amp[i],
                    f"{ln}_p_amp": p_amp[i], f"{ln}_std": std[i], f"{ln}_skew": skew[i],
                    f"{ln}_kurt": kurt[i], f"{ln}_energy": energy[i]})
    return out


def _spectral_wavelet(z2: np.ndarray, valid: int) -> dict:
    import pywt

    x = z2[:valid]
    out = {}
    f, p = welch(x, fs=FS, nperseg=min(1024, len(x)))
    m = (f >= 0.5) & (f <= 40)
    f, p = f[m], p[m]
    tot = p.sum() + 1e-12
    for k, (lo, hi) in enumerate([(0.5, 5), (5, 15), (15, 25), (25, 40)]):
        out[f"spec_band{k}_rel"] = p[(f >= lo) & (f < hi if hi < 40 else f <= hi)].sum() / tot
    pn = p / tot
    out["spec_entropy"] = float(-(pn * np.log(pn + 1e-12)).sum() / np.log(len(pn)))
    out["spec_dom_freq"] = float(f[np.argmax(p)])
    out["spec_centroid"] = float((f * p).sum() / tot)
    out["spec_log_power"] = float(np.log10(tot))
    coeffs = pywt.wavedec(x, "db4", level=5)  # [a5, d5, d4, d3, d2, d1]
    en = np.array([np.sum(c ** 2) for c in coeffs])
    rel = en / (en.sum() + 1e-12)
    out["wav_a5_rel"] = rel[0]
    for lv in range(1, 6):
        out[f"wav_d{lv}_rel"] = rel[6 - lv]
    out["wav_entropy"] = float(-(rel * np.log(rel + 1e-12)).sum())
    return out


# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def extract_record(path: str, age: float = np.nan, sex: str = "Unknown", fs_hint: float | None = None) -> dict:
    """Features for one record (path without extension). Never raises."""
    row = _nan_row()
    diag = {"delineation_ok": False, "lead_used": -1, "rpeak_ok": False, "n_beats_diag": 0, "pr_nan": True,
            "qrs_nan": True, "t_nan": True, "p_nan": True, "error": ""}
    row["age"] = float(age) if age is not None else np.nan
    row["sex_male"] = {"Male": 1.0, "Female": 0.0}.get(sex, np.nan)
    qc = {}
    try:
        sig, fs = read_signal(path)
        pp = preprocess(sig, fs)
        qc = pp.qc
        valid = pp.valid_len
    except Exception as e:  # unreadable record
        diag["error"] = f"load/preprocess: {type(e).__name__}: {e}"[:200]
        return {"features": row, "diag": diag, "qc": qc}
    rp = w = li = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            li, rp, w = _detect_and_delineate(pp.signal_mv, valid)
            if li is not None:
                diag.update(lead_used=int(li), rpeak_ok=True, n_beats_diag=int(len(rp)), delineation_ok=w is not None)
        except Exception as e:
            diag["error"] += f"delineate: {type(e).__name__}; "
        for name, fn in [
            ("rhythm", lambda: _rhythm(rp) if rp is not None else {}),
            ("intervals", lambda: _intervals(rp, w, row["rr_mean"], pp.signal_mv[:valid, li]) if rp is not None and li is not None else {}),
            ("per_lead", lambda: _per_lead(pp.signal_mv, valid, rp, w)),
            ("spectral", lambda: _spectral_wavelet(pp.signal_z[:, 1], valid)),
        ]:
            try:
                row.update(fn())
            except Exception as e:
                diag["error"] += f"{name}: {type(e).__name__}: {e}; "[:120]
    diag.update(pr_nan=bool(np.isnan(row["pr_interval"])), qrs_nan=bool(np.isnan(row["qrs_duration"])),
                t_nan=bool(np.isnan(row["qt_interval"])), p_nan=bool(np.isnan(row["p_duration"])))
    row = {k: (float(v) if v is not None and np.isfinite(v) else np.nan) for k, v in row.items()}
    assert list(row) == FEATURE_NAMES, "feature columns drifted from the registry"
    return {"features": row, "diag": diag, "qc": qc}


def _cached(rid: str, path: str, age: float, sex: str, cache_dir: Path) -> dict:
    cf = cache_dir / f"{rid}.json"
    if cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    res = extract_record(path, age, sex)
    res["record_id"] = rid
    tmp = cf.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(cf)
    return res


def extract_many(ids: list[str], meta: pd.DataFrame, n_jobs: int = 10, cache_dir: Path = CACHE_DIR,
                 verbose: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract (or load from cache) features for `ids`. Returns (features, diagnostics), both
    indexed by record_id in the order of `ids`. Interrupted runs resume from the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    m = meta.set_index("record_id").loc[ids]
    res = Parallel(n_jobs=n_jobs, batch_size=16, verbose=verbose)(
        delayed(_cached)(rid, r.path, r.age, r.sex, cache_dir) for rid, r in m.iterrows())
    feats = pd.DataFrame([r["features"] for r in res], index=pd.Index(ids, name="record_id"))[FEATURE_NAMES]
    diag = pd.DataFrame([{**r["diag"], **r["qc"]} for r in res], index=pd.Index(ids, name="record_id"))
    return feats, diag


def failure_report(diag: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Delineation/NaN failure rates overall and per class and source (noisy-data evidence)."""
    d = diag.join(labels.set_index("record_id")[["label", "source"]])
    d["no_peaks"] = ~d["rpeak_ok"]
    d["delin_failed"] = ~d["delineation_ok"]
    cols = ["no_peaks", "delin_failed", "pr_nan", "qrs_nan", "t_nan", "p_nan"]
    out = [d[cols].mean().rename("ALL").to_frame().T]
    out += [d.groupby("label")[cols].mean().add_prefix("")]
    out += [d.groupby("source")[cols].mean()]
    res = pd.concat(out)
    res.insert(0, "n", [len(d)] + d.groupby("label").size().tolist() + d.groupby("source").size().tolist())
    return res
