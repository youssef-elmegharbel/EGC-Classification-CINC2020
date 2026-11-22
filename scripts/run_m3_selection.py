"""M3: feature selection on the 80% training part (never touches the 20% test set).

Stages (each resumable; results land in results/models/m3/):
  select  [methods...]   run every selector on the full training part   -> m3/subsets/<method>.json
  nested  [methods...]   selection-bias check: rerun the whole procedure inside each outer fold
                                                                         -> m3/nested/<method>.csv
  classify               score every subset with 3 classifiers via src.evaluate.cv_evaluate (set c pipeline)
                                                                         -> m3/subsets_cv.csv
Run:  python scripts/run_m3_selection.py select
      python scripts/run_m3_selection.py nested
      python scripts/run_m3_selection.py classify
Then: python scripts/make_m3_outputs.py   (tables, figures, results/data/selected_features.json)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # workers are parallel already; avoid BLAS oversubscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd                                                           # noqa: E402
from src import selection as S                                                # noqa: E402
from src.evaluate import append_result, cv_evaluate, load_task, summary_columns, train_part  # noqa: E402
from src.loaders import MODELS_DIR                                            # noqa: E402
from src.models import make_model                                             # noqa: E402

OUT = MODELS_DIR / "m3"
(OUT / "subsets").mkdir(parents=True, exist_ok=True)
(OUT / "nested").mkdir(parents=True, exist_ok=True)
DUMMY_F1 = 0.166


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "select"
    only = sys.argv[2:] or list(S.METHODS)
    X, y, splits = load_task()
    Xtr, ytr = train_part(X, y, splits)
    if stage == "select":
        task = S.Task(Xtr, ytr, splits["folds"])
        for name in only:
            f = OUT / "subsets" / f"{name}.json"
            if f.exists():
                log("skip", name); continue
            log("select", name)
            fn = S.METHODS[name]
            try:
                res = fn(task, log=log) if name in ("wrapper_sfs", "wrapper_sbs", "meta_ga", "meta_pso", "meta_gwo") else fn(task)
            except Exception as e:                                             # keep going, report at the end
                log("FAILED", name, repr(e)); continue
            res["method"], res["family"] = name, S.FAMILY[name]
            f.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
            log(f"  -> {name}: k={res['k']} in {res['time_s']:.0f}s")
    elif stage == "nested":
        for name in only:
            f = OUT / "nested" / f"{name}.csv"
            if f.exists():
                log("skip", name); continue
            log("nested", name)
            df = S.nested_score(name, Xtr, ytr, splits["folds"], log=log)
            df.assign(features=df["features"].map(json.dumps)).to_csv(f, index=False)
    elif stage == "classify":
        csv = OUT / "subsets_cv.csv"
        done = set(pd.read_csv(csv)["name"]) if csv.exists() else set()
        subsets = {"all_131": list(X.columns)}
        for f in sorted((OUT / "subsets").glob("*.json")):
            subsets[f.stem] = json.loads(f.read_text())["features"]
        params = {"logreg": {"clf__C": 1}, "svm": {"clf__C": 3, "clf__gamma": "scale"}, "lgbm": {}}
        for sname, feats in subsets.items():
            for clf, p in params.items():
                key = f"{sname}|{clf}"
                if key in done:
                    continue
                pipe = make_model(clf, p, selector=None if sname == "all_131" else feats)
                res = cv_evaluate(pipe, Xtr, ytr, splits["folds"], key, n_jobs=1 if clf == "lgbm" else 5, dummy_f1=DUMMY_F1)
                append_result({"subset": sname, "clf": clf, "k": len(feats), **res["row"]}, csv)
                log(key, summary_columns(res["row"])["f1_macro_cv"], res["row"]["flag"])


if __name__ == "__main__":
    main()
