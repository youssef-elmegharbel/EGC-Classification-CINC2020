"""Score the best-CV config of each model on feature sets b and c on the held-out test set, ONCE each.

Best config = highest 5-fold CV macro-F1 across rounds (src.models.best_per_model); the test set never drives the choice.
fit_and_test refuses repeated names, so re-running this script is safe (already-scored names are skipped).
usage: PYTHONPATH=. python scripts/score_test_bc.py
Outputs appended to results/models/test_results.csv and test_log.csv.
"""
import json

import pandas as pd

import src.evaluate as ev
import src.models as md
from src.loaders import DATA_DIR, MODELS_DIR
from src.reduction import best_reducer

X, y, splits = ev.load_task()
selector_c = json.loads((DATA_DIR / "selected_features.json").read_text())["features"]
done = set(pd.read_csv(ev.TEST_LOG).name) if ev.TEST_LOG.exists() else set()
for fs in ("b", "c"):
    best = md.best_per_model(fs)
    for _, r in best.iterrows():
        name = f"{r.model}|{fs}"
        if name in done:
            continue
        pipe = md.make_model(str(r.model), r.params_dict,
                             selector=selector_c if fs == "c" else None,
                             reducer=best_reducer() if fs == "b" else None)
        out = ev.fit_and_test(pipe, X, y, splits, name)
        out["metrics"].update(name=name, model=str(r.model), feature_set=fs)
        ev.append_result({**out["metrics"], "cv_f1_macro": r.f1_macro_val_mean, "train_f1_macro": r.f1_macro_train_mean,
                          "params": r.params}, MODELS_DIR / "test_results.csv")
        print(name, round(out["metrics"]["f1_macro"], 3), flush=True)
