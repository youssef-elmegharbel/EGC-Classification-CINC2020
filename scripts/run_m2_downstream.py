"""M2 part 2: downstream classification with group-1 reducers INSIDE the pipeline (fit on training
folds only, shared 5 folds, src.evaluate.cv_evaluate). Never touches the 20% test set.

Stage 1: every group-1 method x dimension {2, 10, 20} (PCA/SVD: 95% variance, LDA: 5) x
         {logreg, knn, svm}.   Stage 2: LightGBM at each method's best dimension.
Outputs: results/models/m2/downstream_cv.csv, best_reducer.json, results/figures/28_*, 29_*
Run:  python scripts/run_m2_downstream.py              (full run, ~25 min)
      python scripts/run_m2_downstream.py --handoff-only (rebuild best_reducer.json from the CSV)
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import append_result, cv_evaluate, load_task, summary_columns, train_part  # noqa: E402
from src.loaders import FIG_DIR, MODELS_DIR                                              # noqa: E402
from src.models import make_model                                                         # noqa: E402
from src.reduction import make_reducer, preprocess_fit, with_clip                                    # noqa: E402

OUT = MODELS_DIR / "m2"; OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "downstream_cv.csv"
DUMMY_F1 = 0.166
CLF_PARAMS = {"logreg": {"clf__C": 1}, "knn": {"clf__n_neighbors": 11},
              "svm": {"clf__C": 3, "clf__gamma": "scale"}, "lgbm": {}}
KW = {"KernelPCA_rbf": {"gamma": 0.005}}


def configs(n95):
    yield "none", None, 131
    yield "PCA", 0.95, n95
    yield "SVD", n95, n95   # n95 from the full 80% (not per fold): small leak, noted in the report
    yield "LDA", 5, 5
    for m in ["KernelPCA_rbf", "KernelPCA_poly", "UMAP", "Isomap", "LLE", "LLE_modified"]:
        for d in (2, 10, 20):
            yield m, d, d


def run(method, ncomp, dim, clf, X, y, folds, cache):
    reducer = None if method == "none" else with_clip(make_reducer(method, ncomp, **KW.get(method, {})))
    pipe = make_model(clf, CLF_PARAMS[clf], reducer=reducer)
    if reducer is not None:
        pipe.memory = cache
    name = f"{method}|d={dim}|{clf}"
    res = cv_evaluate(pipe, X, y, folds, name, n_jobs=1 if clf == "lgbm" else 2, dummy_f1=DUMMY_F1)
    row = {"method": method, "dim": dim, "clf": clf, **res["row"]}
    append_result(row, CSV)
    s = summary_columns(res["row"])
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    return row


def write_handoff(df: pd.DataFrame) -> dict:
    """Choose M2's hand-off reducer from downstream_cv.csv and write best_reducer.json.

    Rule: among reducer runs (method != 'none') NOT flagged overfit by evaluate.py, take the highest
    CV macro-F1. kNN uses weights='distance', so its train score is always 1.0 and it is always flagged;
    excluding flagged runs keeps the hand-off honest. 'parsimonious' = smallest dimension within 0.01 of best.
    """
    red_runs = df[(df.method != "none") & (df.flag == "ok")]
    best = red_runs.sort_values("f1_macro_val_mean", ascending=False).iloc[0]
    near = (red_runs[red_runs.f1_macro_val_mean >= best.f1_macro_val_mean - 0.01]
            .sort_values(["dim", "f1_macro_val_mean"], ascending=[True, False]).iloc[0])
    none_best = df[df.method == "none"].f1_macro_val_mean.max()
    top_any = df[df.method != "none"].sort_values("f1_macro_val_mean", ascending=False).iloc[0]

    def entry(r):
        return {"method": r.method, "n_components": int(r.dim), "dim": int(r.dim), "classifier": r.clf,
                "cv_f1_macro": round(float(r.f1_macro_val_mean), 4),
                "cv_f1_macro_std": round(float(r.f1_macro_val_std), 4),
                "cv_accuracy": round(float(r.accuracy_val_mean), 4),
                "train_f1_macro": round(float(r.f1_macro_train_mean), 4),
                "flag": r.flag, "kwargs": KW.get(r.method, {})}

    handoff = {
        "best": entry(best),
        "parsimonious": entry(near),
        "note": (f"Chosen among runs NOT flagged overfit. kNN uses weights='distance', so its train score is always "
                 f"1.0 and it is always flagged; the top run overall is {top_any['name']} "
                 f"({top_any.f1_macro_val_mean:.4f}, flag={top_any.flag}). No reducer beats all 131 features "
                 f"(best {none_best:.3f}); {best.method} keeps {best.f1_macro_val_mean / none_best:.0%} of that with "
                 f"{int(best.dim)} dimensions instead of 131. Pipeline: impute -> scale -> ClipZ(5) -> "
                 f"{best.method}({int(best.dim)}) -> classifier."),
        "all_features_same_classifiers_cv_f1_macro": round(float(none_best), 4),
        "usage": "from src.reduction import best_reducer; make_model(clf, params, reducer=best_reducer())",
    }
    (OUT / "best_reducer.json").write_text(json.dumps(handoff, indent=2) + "\n")
    print(json.dumps(handoff, indent=2))
    return handoff


def main():
    X, y, splits = load_task()
    Xtr, ytr = train_part(X, y, splits)
    folds = splits["folds"]
    n95 = int(np.searchsorted(np.cumsum(make_reducer("PCA", None).fit(preprocess_fit(Xtr).transform(Xtr))
                                         .explained_variance_ratio_), 0.95) + 1)
    done = set()
    if CSV.exists():
        d = pd.read_csv(CSV); done = set(d["name"])
    cache = tempfile.mkdtemp(prefix="m2cache_")
    try:
        for method, ncomp, dim in configs(n95):
            for clf in ["logreg", "knn", "svm"]:
                if f"{method}|d={dim}|{clf}" not in done:
                    run(method, ncomp, dim, clf, Xtr, ytr, folds, cache)
        df = pd.read_csv(CSV)
        # best dim per method = highest mean macro-F1 over the three stage-1 classifiers
        s1 = df[df.clf.isin(["logreg", "knn", "svm"])]
        best_dim = (s1.groupby(["method", "dim"]).f1_macro_val_mean.mean().reset_index()
                    .sort_values("f1_macro_val_mean").groupby("method").tail(1).set_index("method").dim)
        for method, ncomp, dim in configs(n95):
            if dim == best_dim[method] and f"{method}|d={dim}|lgbm" not in done:
                run(method, ncomp, dim, "lgbm", Xtr, ytr, folds, cache)
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    df = pd.read_csv(CSV)
    best = df[df.apply(lambda r: r.dim == best_dim[r.method], axis=1)]
    tab = best.pivot_table(index="method", columns="clf", values="f1_macro_val_mean")
    tab["dim"] = best_dim; tab["best_f1"] = tab[["logreg", "knn", "svm", "lgbm"]].max(1)
    tab = tab.sort_values("best_f1", ascending=False); tab.to_csv(OUT / "downstream_best_dim.csv")
    print(tab.round(3), flush=True)

    write_handoff(df)

    # figures
    order = ["PCA", "SVD", "LDA", "KernelPCA_rbf", "KernelPCA_poly", "UMAP", "Isomap", "LLE", "LLE_modified"]
    fig, ax = plt.subplots(figsize=(11, 4))
    cols = {"logreg": "#2a78d6", "knn": "#eb6834", "svm": "#1baf7a", "lgbm": "#eda100"}
    w = 0.2
    for k, c in enumerate(["logreg", "knn", "svm", "lgbm"]):
        v = tab.reindex(order)[c]
        ax.bar(np.arange(len(order)) + (k - 1.5) * w, v, w * 0.9, color=c and cols[c], label=c)
    for k, c in enumerate(["logreg", "knn", "svm", "lgbm"]):
        ref = df[(df.method == "none") & (df.clf == c)].f1_macro_val_mean
        if len(ref):
            ax.axhline(ref.iloc[0], color=cols[c], ls=":", lw=1)
    ax.axhline(DUMMY_F1, color="#888", ls="--", lw=1); ax.text(len(order) - 0.6, DUMMY_F1 + 0.01, "dummy", fontsize=8, color="#666")
    ax.set_xticks(range(len(order)), [f"{m}\n(d={int(tab.loc[m, 'dim'])})" for m in order], fontsize=8)
    ax.set(ylabel="5-fold CV macro-F1", title="Downstream macro-F1 with each reducer inside the pipeline (dotted = all 131 features)", ylim=(0, 0.85))
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "28_downstream_f1.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, m in enumerate(["KernelPCA_rbf", "KernelPCA_poly", "UMAP", "Isomap", "LLE", "LLE_modified"]):
        d = s1[s1.method == m].groupby("dim").f1_macro_val_mean.mean()
        ax.plot(d.index, d.values, "o-", lw=2, ms=5, label=m,
                color=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"][i])
    ax.set(xticks=[2, 10, 20], xlabel="embedding dimension", ylabel="mean CV macro-F1 (logreg, kNN, SVM)",
           title="Choosing the dimension by CV")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG_DIR / "29_dimension_choice.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    if "--handoff-only" in sys.argv:          # rebuild best_reducer.json from the existing CSV (no refit)
        write_handoff(pd.read_csv(CSV))
    else:
        main()
