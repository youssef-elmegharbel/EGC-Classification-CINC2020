"""Label mapping and the 6-class single-label task table (plan section 2.2)."""
from __future__ import annotations

import pandas as pd

from .loaders import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

DX_MAP_PATH = PROJECT_ROOT / "data" / "dx_mapping_scored.csv"
SEED = 42

# Equivalent codes scored as the same diagnosis (see Notes column of the dx map).
EQUIVALENTS = {"713427006": "59118001",   # CRBBB -> RBBB
               "63593006": "284470004",   # SVPB  -> PAC
               "17338001": "427172004"}   # VPB   -> PVC

TASK_CLASSES = {  # SNOMED -> abbreviation, class order is the label order
    "426783006": "NSR", "59118001": "RBBB", "164889003": "AF",
    "270492004": "IAVB", "284470004": "PAC", "426177001": "SB",
}
PLAN_COUNTS = {"NSR": 9182, "RBBB": 1605, "AF": 1093, "IAVB": 754, "PAC": 538, "SB": 456}
NSR_TARGET = 2000


def load_dx_map() -> pd.DataFrame:
    df = pd.read_csv(DX_MAP_PATH, dtype={"SNOMED CT Code": str})
    return df.rename(columns={"SNOMED CT Code": "code", "Abbreviation": "abbr"})


def verify_task_codes() -> pd.DataFrame:
    """Check the 6 codes exist in dx_mapping_scored.csv with the expected abbreviation."""
    m = load_dx_map().set_index("code")
    rows = []
    for code, abbr in TASK_CLASSES.items():
        assert code in m.index, f"{code} ({abbr}) missing from dx_mapping_scored.csv"
        assert m.loc[code, "abbr"] == abbr, f"{code}: expected {abbr}, found {m.loc[code, 'abbr']}"
        rows.append({"code": code, "abbr": abbr, "dx": m.loc[code, "Dx"]})
    return pd.DataFrame(rows)


def merge_equivalents(codes: list[str]) -> list[str]:
    """Map equivalent codes onto their canonical code, de-duplicated, order kept."""
    out: list[str] = []
    for c in codes:
        c = EQUIVALENTS.get(c, c)
        if c not in out:
            out.append(c)
    return out


def scored_single_label(meta: pd.DataFrame) -> pd.DataFrame:
    """Records whose header carries exactly ONE diagnosis code (scored or not) after merging
    equivalents. This is the plan's "exactly one diagnosis" rule (19,466 raw / 19,555 merged)."""
    merged = meta["dx_codes"].map(lambda cs: merge_equivalents(list(cs)))
    out = meta.assign(scored_dx=merged)
    return out[out["scored_dx"].map(len) == 1].assign(code=lambda d: d["scored_dx"].str[0])


def build_task_table(meta: pd.DataFrame, nsr_target: int = NSR_TARGET,
                     write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (labels, counts). labels: record_id, label, source (NSR downsampled)."""
    verify_task_codes()
    single = scored_single_label(meta)
    task = single[single["code"].isin(TASK_CLASSES)].copy()
    task["label"] = task["code"].map(TASK_CLASSES)
    before = task["label"].value_counts()
    nsr = task[task["label"] == "NSR"]
    keep_nsr = nsr.sample(n=min(nsr_target, len(nsr)), random_state=SEED)
    task = pd.concat([task[task["label"] != "NSR"], keep_nsr]).sort_values("record_id")
    labels = task[["record_id", "label", "source"]].reset_index(drop=True)
    counts = pd.DataFrame({
        "plan": pd.Series(PLAN_COUNTS), "found": before, "after_downsample": labels["label"].value_counts(),
    }).loc[list(TASK_CLASSES.values())]
    counts["diff_pct_vs_plan"] = ((counts["found"] - counts["plan"]) / counts["plan"] * 100).round(1)
    if write:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(DATA_DIR / "labels.parquet", index=False)
    return labels, counts
