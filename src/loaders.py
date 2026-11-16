"""Record discovery and signal reading for the PhysioNet/CinC 2020 dataset.

`scan_records` parses headers only (no signal), so it is fast. `read_signal`
loads one record as (n_samples, 12) in mV.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "training"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"        # records_meta, labels, splits, features, diagnostics, dictionary
MODELS_DIR = RESULTS_DIR / "models"    # grids, comparison tables, OOF predictions, test results, curves
LOGS_DIR = RESULTS_DIR / "logs"        # run logs
FIG_DIR = RESULTS_DIR / "figures"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_header(hea_path: Path, root: Path = DATA_ROOT) -> dict:
    """Parse one .hea file into a flat dict. Never reads the signal."""
    hea_path = Path(hea_path)
    text = hea_path.read_text(errors="replace").splitlines()
    first = text[0].split()
    n_leads, fs, n_samples = int(first[1]), float(first[2]), int(first[3])
    age, sex, dx = np.nan, "Unknown", []
    for line in text:
        if not line.startswith("#"):
            continue
        key, _, val = line[1:].partition(":")
        key, val = key.strip().lower(), val.strip()
        if key in ("age", "age (years)"):
            try:
                a = float(val)
                age = a if 0 <= a <= 120 else np.nan
            except ValueError:
                age = np.nan
        elif key == "sex":
            v = val.lower()
            sex = "Male" if v.startswith("m") else "Female" if v.startswith("f") else "Unknown"
        elif key in ("dx", "diagnosis"):
            dx = [c.strip() for c in re.split(r"[,;]", val) if c.strip()]
    rel = hea_path.relative_to(root)
    return {
        "record_id": hea_path.stem,
        "source": rel.parts[0],
        "path": str(hea_path.with_suffix("")),
        "fs": fs,
        "n_samples": n_samples,
        "n_leads": n_leads,
        "age": age,
        "sex": sex,
        "dx_codes": dx,
    }


def scan_records(root: Path = DATA_ROOT, cache: Path | None = None, force: bool = False,
                 n_jobs: int = 10) -> pd.DataFrame:
    """Header metadata for every record. Cached to results/data/records_meta.parquet."""
    cache = Path(cache) if cache else DATA_DIR / "records_meta.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    files = sorted(Path(root).rglob("*.hea"))
    rows = Parallel(n_jobs=n_jobs, batch_size=256)(delayed(parse_header)(f, root) for f in files)
    df = pd.DataFrame(rows)
    assert df["record_id"].is_unique, "record ids are not unique across sources"
    df["duration_s"] = df["n_samples"] / df["fs"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def read_signal(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (signal[n_samples, n_leads] in mV, fs). `path` has no extension."""
    import wfdb

    rec = wfdb.rdrecord(str(path))
    sig = np.asarray(rec.p_signal, dtype=np.float64)
    return sig, float(rec.fs)
