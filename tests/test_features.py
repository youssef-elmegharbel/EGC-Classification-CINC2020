"""Sanity checks for the feature pipeline on a handful of real records (needs data/training)."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_NAMES, FEATURES, extract_record
from src.loaders import DATA_DIR, DATA_ROOT

META = DATA_DIR / "records_meta.parquet"
pytestmark = pytest.mark.skipif(not META.exists() or not DATA_ROOT.exists(),
                                reason="needs results/data/records_meta.parquet and the raw data/training/ files")


@pytest.fixture(scope="module")
def rows():
    meta = pd.read_parquet(META).set_index("record_id")
    lab = pd.read_parquet(DATA_DIR / "labels.parquet")
    ids = lab.groupby("label").head(1).record_id.tolist()
    return {rid: extract_record(meta.loc[rid, "path"], meta.loc[rid, "age"], meta.loc[rid, "sex"]) for rid in ids}, lab


def test_columns_match_registry(rows):
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES) == len(FEATURES)
    for r, _ in [rows]:
        for res in r.values():
            assert list(res["features"]) == FEATURE_NAMES


def test_plausible_values_and_no_inf(rows):
    r, _ = rows
    for rid, res in r.items():
        f = res["features"]
        assert not any(np.isinf(v) for v in f.values())
        if not np.isnan(f["hr_mean"]):
            assert 20 < f["hr_mean"] < 250, rid
        if not np.isnan(f["qrs_duration"]):
            assert 40 <= f["qrs_duration"] <= 250, rid


def test_never_raises_on_bad_path():
    res = extract_record("does/not/exist")
    assert res["diag"]["error"] and all(np.isnan(v) for k, v in res["features"].items() if k not in ("age", "sex_male"))
