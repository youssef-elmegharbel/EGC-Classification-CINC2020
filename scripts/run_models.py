"""Grid-search the 6 models on one feature set with the shared 5 folds. Resumable.

usage: python scripts/run_models.py a            # all features
       python scripts/run_models.py b            # M2's best reducer (results/models/m2/best_reducer.json), fit per fold
       python scripts/run_models.py c            # M3's list in results/data/selected_features.json
Outputs: results/models/grid_results_<set>.csv (every grid point), results/models/model_comparison.csv (best per
model), results/models/best_params.json, results/models/oof_<set>_<model>.parquet (pooled out-of-fold predictions).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.evaluate import append_result, dummy_baseline, load_task, train_part
from src.loaders import DATA_DIR, MODELS_DIR, RESULTS_DIR
from src.models import GRIDS_R2, MODEL_NAMES, run_grid

fs = sys.argv[1] if len(sys.argv) > 1 else "a"
rnd = sys.argv[3] if len(sys.argv) > 3 else "r1"
only = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else MODEL_NAMES
X, y, splits = load_task()
Xtr, ytr = train_part(X, y, splits)
selector = None
reducer = None
if fs == "b":
    from src.reduction import best_reducer
    reducer = best_reducer()  # unfitted; fitted inside each training fold by the Pipeline (no leakage)
if fs == "c":
    selector = json.loads((DATA_DIR / "selected_features.json").read_text())["features"]
bp_path = MODELS_DIR / "best_params.json"
best_params = json.loads(bp_path.read_text()) if bp_path.exists() else {}

base_path = MODELS_DIR / "baselines.csv"
if not base_path.exists():
    for r in dummy_baseline(Xtr, ytr, splits["folds"]):
        append_result(r, base_path)
dummy_f1 = float(pd.read_csv(base_path).query("name=='dummy_stratified'")["f1_macro_val_mean"].iloc[0])
print(f"set={fs} train={Xtr.shape} dummy stratified macro-F1={dummy_f1:.3f}", flush=True)

for m in only:
    key = f"{m}|{fs}" + ("|r2" if rnd == "r2" else "")
    if rnd == "r2" and m not in GRIDS_R2:
        continue
    if key in best_params:
        print("skip (done):", key, flush=True)
        continue
    print(f"== {key}", flush=True)
    table, best = run_grid(m, Xtr, ytr, splits["folds"], fs, dummy_f1, selector=selector, reducer=reducer,
                           grid=GRIDS_R2[m] if rnd == "r2" else None,
                           log=lambda s: print(s, flush=True))
    for rec in table.assign(round=rnd).to_dict("records"):
        append_result(rec, MODELS_DIR / f"grid_results_{fs}.csv")
    append_result({**best["row"], "round": rnd}, MODELS_DIR / "model_comparison.csv")
    best["oof"].to_parquet(MODELS_DIR / f"oof_{fs}_{m}{'_r2' if rnd == 'r2' else ''}.parquet")
    best_params[key] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best["params"].items()}
    bp_path.write_text(json.dumps(best_params, indent=1))
print("DONE", flush=True)
