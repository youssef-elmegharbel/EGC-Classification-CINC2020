"""Stratified 80/20 split, 5 folds inside the 80%, and the shared plotting subsample."""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .loaders import DATA_DIR

SEED = 42


def make_splits(labels: pd.DataFrame, test_size: float = 0.2, n_folds: int = 5,
                write: bool = True) -> dict:
    """Stratify by class only. Order: 80/20 split, then k folds on the 80% only."""
    ids, y = labels["record_id"].to_numpy(), labels["label"].to_numpy()
    tr_ids, te_ids, y_tr, _ = train_test_split(ids, y, test_size=test_size, stratify=y,
                                               random_state=SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = [{"train_ids": tr_ids[a].tolist(), "val_ids": tr_ids[b].tolist()}
             for a, b in skf.split(tr_ids, y_tr)]
    splits = {"seed": SEED, "train_ids": tr_ids.tolist(), "test_ids": te_ids.tolist(), "folds": folds}
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "splits.json").write_text(json.dumps(splits))
    return splits


def make_subsample(labels: pd.DataFrame, train_ids: list[str], n: int = 2000, write: bool = True) -> list[str]:
    """Stratified (by class) subsample of the TRAINING set for 2D plots / O(n^2) methods."""
    tr = labels[labels["record_id"].isin(set(train_ids))]
    ids, _ = train_test_split(tr["record_id"].to_numpy(), train_size=n, stratify=tr["label"].to_numpy(),
                              random_state=SEED)
    ids = sorted(ids.tolist())
    if write:
        (DATA_DIR / "subsample_ids.json").write_text(json.dumps(ids))
    return ids


def source_mix(labels: pd.DataFrame, splits: dict, subsample: list[str] | None = None) -> pd.DataFrame:
    """Row-normalised source and class mix for train/test/each val fold (+ subsample)."""
    lab = labels.set_index("record_id")
    groups = {"train": splits["train_ids"], "test": splits["test_ids"]}
    groups.update({f"fold{i}_val": f["val_ids"] for i, f in enumerate(splits["folds"])})
    if subsample is not None:
        groups["subsample"] = subsample
    rows = {}
    for name, ids in groups.items():
        sub = lab.loc[ids]
        r = {"n": len(sub)}
        r.update({f"src:{k}": v for k, v in sub["source"].value_counts(normalize=True).items()})
        r.update({f"cls:{k}": v for k, v in sub["label"].value_counts(normalize=True).items()})
        rows[name] = r
    return pd.DataFrame(rows).T.fillna(0).round(3)


def load_splits() -> dict:
    return json.loads((DATA_DIR / "splits.json").read_text())


def load_subsample() -> list[str]:
    return json.loads((DATA_DIR / "subsample_ids.json").read_text())
