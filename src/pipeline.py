"""Thin orchestration for building the shared artifacts (used by the notebook and scripts)."""
from __future__ import annotations

import time

import pandas as pd

from .features import data_dictionary, extract_many, failure_report
from .loaders import DATA_DIR, RESULTS_DIR, scan_records
from .splits import SEED


def v0_ids(labels: pd.DataFrame, n: int = 500) -> list[str]:
    """~n records, stratified by class x source (>=1 per non-empty stratum), subset of the final table."""
    frac = n / len(labels)
    parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=SEED)
             for _, g in labels.groupby(["label", "source"])]
    return sorted(pd.concat(parts)["record_id"].tolist())


def build_features(ids: list[str], labels: pd.DataFrame, name: str, n_jobs: int = 10) -> pd.DataFrame:
    """Extract (resumable), write results/data/{name}.parquet + extraction diagnostics + failure report."""
    meta = scan_records()
    t = time.time()
    feats, diag = extract_many(ids, meta, n_jobs=n_jobs)
    secs = time.time() - t
    feats.to_parquet(DATA_DIR / f"{name}.parquet")
    diag.to_parquet(DATA_DIR / f"{name.replace('features', 'extraction_diag')}.parquet")
    failure_report(diag, labels).round(4).to_csv(DATA_DIR / f"{name.replace('features', 'feature_failures')}.csv")
    data_dictionary().to_csv(DATA_DIR / "data_dictionary.csv", index=False)
    print(f"{name}: {len(ids)} records, {secs:.1f}s total, {secs / len(ids) * n_jobs:.2f} worker-s/record")
    return feats
