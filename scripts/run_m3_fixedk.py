"""M3 at a COMMON subset size: every selection method returns exactly K features, then the subsets are retrained.

Stages (each resumable; results land in results/models/m3_fixedk/):
  choosek                 pick K: mean CV curve of the ranking methods (ANOVA, MI, chi2, L2, RF) + 1-SE rule
                                                                         -> common_k.json
  select  [methods...]    run every selector with k_fixed=K on the training part  -> subsets/<method>.json
  nested  [methods...]    selection-bias check at the same K (selection repeated in each outer fold)
                                                                         -> nested/<method>.csv
  classify                retrain logreg / SVM / LightGBM on every K-feature subset (+ all 131 as reference)
                                                                         -> subsets_cv.csv
  summary                 one table: search-time CV, nested F1, and the 3 classifiers per method
                                                                         -> summary.csv
Run:  python scripts/run_m3_fixedk.py choosek
      python scripts/run_m3_fixedk.py select      (or:  select filter_anova meta_ga)
      python scripts/run_m3_fixedk.py nested
      python scripts/run_m3_fixedk.py classify
      python scripts/run_m3_fixedk.py summary
Override K with the env var M3_K (e.g. M3_K=40) instead of reading common_k.json. The test set is never used.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
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
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from src import selection as S                                                # noqa: E402
from src.evaluate import append_result, cv_evaluate, dummy_baseline, load_task, summary_columns, train_part  # noqa: E402
from src.loaders import MODELS_DIR                                            # noqa: E402
from src.models import make_model                                             # noqa: E402

OUT = MODELS_DIR / "m3_fixedk"
for d in ("subsets", "nested"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
WITH_LOG = ("wrapper_sfs", "wrapper_sbs", "meta_ga", "meta_pso", "meta_gwo")
RANKERS = ("filter_anova", "filter_mi", "filter_chi2", "embedded_l2", "embedded_rf")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_k() -> int:
    if os.environ.get("M3_K"):
        return int(os.environ["M3_K"])
    f = OUT / "common_k.json"
    if not f.exists():
        sys.exit("run `choosek` first (or set M3_K)")
    return int(json.loads(f.read_text())["k"])


def choosek(task):
    curves = {}
    for name in RANKERS:
        log("ranking", name)
        meta = S.METHODS[name](task)["meta"]
        rank = [task.names.index(n) for n in meta["ranking"]]
        res = [S._score_cols(task.fold_data, rank[:k]) for k in S.K_GRID if k <= len(rank)]
        curves[name] = pd.DataFrame({"k": [k for k in S.K_GRID if k <= len(rank)], "cv_f1": [r[0] for r in res], "cv_std": [r[1] for r in res]})
    ks = sorted(set.intersection(*[set(c.k) for c in curves.values()]))
    m = np.mean([c.set_index("k").loc[ks, "cv_f1"].to_numpy() for c in curves.values()], axis=0)
    s = np.mean([c.set_index("k").loc[ks, "cv_std"].to_numpy() for c in curves.values()], axis=0)
    k = ks[S.one_se_pick(m, s, ks, len(task.folds))]
    out = {"k": int(k), "rule": "mean CV curve over ANOVA/MI/chi2/L2/RF rankings; smallest k within 1 SE of the best",
           "k_grid": ks, "mean_cv_f1": [round(float(x), 4) for x in m], "mean_cv_std": [round(float(x), 4) for x in s]}
    (OUT / "common_k.json").write_text(json.dumps(out, indent=1) + "\n")
    log("common K =", k); print(pd.DataFrame({"k": ks, "mean_cv_f1": m.round(4), "mean_cv_std": s.round(4)}).to_string(index=False))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "select"
    only = sys.argv[2:] or list(S.METHODS)
    X, y, splits = load_task()
    Xtr, ytr = train_part(X, y, splits)
    task = S.Task(Xtr, ytr, splits["folds"])
    if stage == "choosek":
        return choosek(task)
    K = get_k(); log("K =", K)
    if stage == "select":
        for name in only:
            f = OUT / "subsets" / f"{name}.json"
            if f.exists():
                log("skip", name); continue
            log("select", name)
            fn = S.METHODS[name]
            try:
                res = fn(task, log=log, k_fixed=K) if name in WITH_LOG else fn(task, k_fixed=K)
            except Exception as e:
                log("FAILED", name, repr(e)); continue
            res["method"], res["family"] = name, S.FAMILY[name]
            f.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            log(f"  -> {name}: k={res['k']} in {res['time_s']:.0f}s")
    elif stage == "nested":
        for name in only:
            f = OUT / "nested" / f"{name}.csv"
            if f.exists():
                log("skip", name); continue
            log("nested", name)
            df = S.nested_score(name, Xtr, ytr, splits["folds"], log=log, k_fixed=K)
            df.assign(features=df["features"].map(json.dumps)).to_csv(f, index=False)
    elif stage == "classify":
        dummy = float(dummy_baseline(Xtr, ytr, splits["folds"])[0]["f1_macro_val_mean"])
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
                res = cv_evaluate(pipe, Xtr, ytr, splits["folds"], key, n_jobs=1 if clf == "lgbm" else 5, dummy_f1=dummy)
                append_result({"subset": sname, "clf": clf, "k": len(feats), **res["row"]}, csv)
                log(key, summary_columns(res["row"])["f1_macro_cv"], res["row"]["flag"])
    elif stage == "summary":
        rows = []
        cv = pd.read_csv(OUT / "subsets_cv.csv") if (OUT / "subsets_cv.csv").exists() else None
        for f in sorted((OUT / "subsets").glob("*.json")):
            d = json.loads(f.read_text()); name = f.stem
            r = {"method": name, "family": d["family"], "k": d["k"], "select_time_s": round(d["time_s"], 1),
                 "search_cv_f1": round(float(S.fold_scores(task.fold_data, task.cols(d["features"])).mean()), 4)}
            nf = OUT / "nested" / f"{name}.csv"
            if nf.exists():
                n = pd.read_csv(nf)
                r.update(nested_f1=round(n.f1_macro.mean(), 4), nested_std=round(n.f1_macro.std(ddof=1), 4),
                         stability_jaccard=round(float(np.mean([S.jaccard(json.loads(a), json.loads(b))
                                                                for i, a in enumerate(n.features) for b in list(n.features)[i + 1:]])), 3))
            if cv is not None:
                for clf in ("logreg", "svm", "lgbm"):
                    q = cv[(cv.subset == name) & (cv.clf == clf)]
                    if len(q):
                        r[f"cv_f1_{clf}"] = round(float(q.iloc[0].f1_macro_val_mean), 4); r[f"flag_{clf}"] = q.iloc[0].flag
            rows.append(r)
        df = pd.DataFrame(rows)
        if cv is not None:
            for clf in ("logreg", "svm", "lgbm"):
                q = cv[(cv.subset == "all_131") & (cv.clf == clf)]
                if len(q): df.attrs[f"all131_{clf}"] = float(q.iloc[0].f1_macro_val_mean)
        df.to_csv(OUT / "summary.csv", index=False)
        print(df.to_string(index=False)); print({k: round(v, 4) for k, v in df.attrs.items()})


if __name__ == "__main__":
    main()
