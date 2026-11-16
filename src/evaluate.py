"""Shared evaluation helpers (M1, M2, M3 all call these so every table is comparable).

Frozen interface (backward-compatible changes only):

    load_task(features="features")                      -> X, y, splits
    train_part(X, y, splits) / test_part(X, y, splits)  -> (X, y) of the 80% / 20%
    cv_evaluate(pipeline, X, y, folds, name, ...)       -> dict  (5-fold, train + validation scores)
    dummy_baseline(X, y, folds)                         -> list[dict]
    fit_and_test(pipeline, X, y, splits, name, ...)     -> dict  (TEST SET, once per name; guarded)
    flag_fit(train_f1, cv_f1, dummy_f1, ...)            -> "overfit" | "underfit" | "ok"
    append_result(row, path)                            -> writes one row to a results csv
    per_class_table(y_true, y_pred, classes)            -> DataFrame
    confusion_figure / roc_figure / learning_curve_figure / validation_curve_figure -> matplotlib Figure

`X` is a DataFrame and `y` a Series, both indexed by record_id. `folds` is `splits["folds"]`
(list of {"train_ids", "val_ids"}). Pipelines must expose `predict_proba` (ROC-AUC needs it).
Everything is rank-by macro-F1; accuracy is secondary. error_rate = 1 - accuracy.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

from .loaders import DATA_DIR, MODELS_DIR, RESULTS_DIR

OVERFIT_GAP = 0.10      # train - CV macro-F1 above this => overfit
UNDERFIT_MARGIN = 0.05  # both within this of the dummy score => underfit
TEST_LOG = MODELS_DIR / "test_log.csv"
CLASS_ORDER = ["NSR", "RBBB", "AF", "IAVB", "PAC", "SB"]


# ------------------------------------------------------------------ data access
def load_task(features: str = "features", results_dir: Path = DATA_DIR):
    """Return X (features), y (labels), splits. Rows are in labels.parquet order."""
    labels = pd.read_parquet(Path(results_dir) / "labels.parquet").set_index("record_id")
    X = pd.read_parquet(Path(results_dir) / f"{features}.parquet")
    X = X.loc[X.index.intersection(labels.index)]
    y = labels.loc[X.index, "label"]
    splits = json.loads((Path(results_dir) / "splits.json").read_text())
    return X, y, splits


def train_part(X, y, splits):
    ids = [i for i in splits["train_ids"] if i in X.index]
    return X.loc[ids], y.loc[ids]


def test_part(X, y, splits):
    ids = [i for i in splits["test_ids"] if i in X.index]
    return X.loc[ids], y.loc[ids]


# ------------------------------------------------------------------ metrics
def score_all(y_true, y_pred, proba=None, classes=None) -> dict:
    """accuracy, error rate, precision/recall/F1 (macro + weighted), OvR ROC-AUC (macro + weighted)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    out["error_rate"] = 1 - out["accuracy"]
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out.update({f"precision_{avg}": p, f"recall_{avg}": r, f"f1_{avg}": f})
    out["roc_auc_macro"] = out["roc_auc_weighted"] = np.nan
    if proba is not None and classes is not None:
        try:
            Y = label_binarize(y_true, classes=classes)
            keep = Y.sum(axis=0) > 0  # a class missing from y_true has no defined AUC
            for avg in ("macro", "weighted"):
                out[f"roc_auc_{avg}"] = roc_auc_score(Y[:, keep], np.asarray(proba)[:, keep], average=avg)
        except Exception:
            pass
    return out


def flag_fit(train_f1: float, cv_f1: float, dummy_f1: float, gap: float = OVERFIT_GAP,
             margin: float = UNDERFIT_MARGIN) -> str:
    """overfit: train - CV macro-F1 > gap.  underfit: both train and CV within `margin` of the dummy."""
    if train_f1 - cv_f1 > gap:
        return "overfit"
    if train_f1 - dummy_f1 < margin and cv_f1 - dummy_f1 < margin:
        return "underfit"
    return "ok"


def per_class_table(y_true, y_pred, classes=CLASS_ORDER) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(classes), zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=list(classes))


def confusion(y_true, y_pred, classes=CLASS_ORDER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(raw counts, row-normalised) confusion matrices as DataFrames (rows = true class)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))
    raw = pd.DataFrame(cm, index=list(classes), columns=list(classes))
    norm = raw.div(raw.sum(axis=1).replace(0, 1), axis=0)
    return raw, norm


# ------------------------------------------------------------------ cross-validation
def _fit_fold(pipeline, X, y, tr_ids, va_ids, classes):
    Xtr, ytr, Xva, yva = X.loc[tr_ids], y.loc[tr_ids], X.loc[va_ids], y.loc[va_ids]
    m = clone(pipeline)
    t = time.time()
    m.fit(Xtr, ytr)
    fit_s = time.time() - t
    res = {}
    for tag, (Xs, ys) in {"train": (Xtr, ytr), "val": (Xva, yva)}.items():
        pr = m.predict_proba(Xs) if hasattr(m, "predict_proba") else None
        if pr is not None and list(m.classes_) != list(classes):
            pr = pd.DataFrame(pr, columns=m.classes_).reindex(columns=classes, fill_value=0).to_numpy()
        res[tag] = (score_all(ys, m.predict(Xs), pr, classes), ys, m.predict(Xs), pr)
    return res, fit_s, va_ids


def cv_evaluate(pipeline, X, y, folds, name: str, n_jobs: int = 1, dummy_f1: float | None = None) -> dict:
    """5-fold evaluation on the shared folds. Returns a dict with:

    row     flat dict for results csv: name, <metric>_train_mean/_train_std/_val_mean/_val_std, fit_s, flag
    folds   per-fold validation scores (DataFrame)
    oof     DataFrame (index record_id): y_true, y_pred (+ proba_<class> columns) pooled over validation folds
    """
    classes = [c for c in CLASS_ORDER if c in set(y)]
    outs = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(pipeline, X, y, f["train_ids"], f["val_ids"], classes)
                                   for f in folds)
    tr = pd.DataFrame([o[0]["train"][0] for o in outs])
    va = pd.DataFrame([o[0]["val"][0] for o in outs])
    row = {"name": name}
    for metric in tr.columns:
        row[f"{metric}_train_mean"], row[f"{metric}_train_std"] = tr[metric].mean(), tr[metric].std(ddof=1)
        row[f"{metric}_val_mean"], row[f"{metric}_val_std"] = va[metric].mean(), va[metric].std(ddof=1)
    row["fit_s_mean"] = float(np.mean([o[1] for o in outs]))
    oof_parts = []
    for o in outs:
        _, ys, yp, pr = o[0]["val"]
        d = pd.DataFrame({"y_true": ys.to_numpy(), "y_pred": yp}, index=o[2])
        if pr is not None:
            for k, c in enumerate(classes):
                d[f"proba_{c}"] = pr[:, k]
        oof_parts.append(d)
    row["flag"] = (flag_fit(row["f1_macro_train_mean"], row["f1_macro_val_mean"], dummy_f1)
                   if dummy_f1 is not None else "n/a")
    row["gap_f1_macro"] = row["f1_macro_train_mean"] - row["f1_macro_val_mean"]
    return {"row": row, "folds": va, "oof": pd.concat(oof_parts)}


def dummy_baseline(X, y, folds) -> list[dict]:
    """Stratified-random and most-frequent baselines via the same 5 folds (always report these)."""
    rows = []
    for strat in ("stratified", "most_frequent"):
        r = cv_evaluate(DummyClassifier(strategy=strat, random_state=42), X, y, folds, f"dummy_{strat}")
        r["row"]["flag"] = "baseline"
        rows.append(r["row"])
    return rows


# ------------------------------------------------------------------ the test set (once)
def fit_and_test(pipeline, X, y, splits, name: str, allow_repeat: bool = False,
                 log_path: Path = TEST_LOG) -> dict:
    """Refit on the full 80% and score the 20% test set. Refuses to run twice for the same `name`
    unless allow_repeat=True. Every call is appended to results/models/test_log.csv."""
    log_path = Path(log_path)
    if log_path.exists() and not allow_repeat:
        if name in pd.read_csv(log_path)["name"].tolist():
            raise RuntimeError(f"Test set already used for '{name}'. The test set is scored once; "
                               f"pass allow_repeat=True only for a deliberate re-run.")
    Xtr, ytr = train_part(X, y, splits)
    Xte, yte = test_part(X, y, splits)
    classes = [c for c in CLASS_ORDER if c in set(ytr)]
    m = clone(pipeline).fit(Xtr, ytr)
    pr = m.predict_proba(Xte)
    if list(m.classes_) != classes:
        pr = pd.DataFrame(pr, columns=m.classes_).reindex(columns=classes, fill_value=0).to_numpy()
    yp = m.predict(Xte)
    metrics = score_all(yte, yp, pr, classes)
    raw, norm = confusion(yte, yp, classes)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), "name": name,
                   "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"]}]).to_csv(
        log_path, mode="a", header=not log_path.exists(), index=False)
    return {"name": name, "metrics": metrics, "model": m, "y_true": yte, "y_pred": yp, "proba": pr,
            "classes": classes, "confusion_raw": raw, "confusion_norm": norm,
            "per_class": per_class_table(yte, yp, classes)}


# ------------------------------------------------------------------ results csv
def append_result(row: dict, path: str | Path) -> None:
    """Append one row to a results csv, creating or widening it as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def summary_columns(row: dict) -> dict:
    """Compact 'mean ± std' view of a cv_evaluate row for display / report tables."""
    out = {"name": row["name"], "flag": row.get("flag")}
    for m in ("f1_macro", "accuracy", "roc_auc_macro", "f1_weighted"):
        out[f"{m}_cv"] = f"{row[f'{m}_val_mean']:.3f} ± {row[f'{m}_val_std']:.3f}"
        out[f"{m}_train"] = f"{row[f'{m}_train_mean']:.3f}"
    return out


# ------------------------------------------------------------------ figures
def confusion_figure(raw: pd.DataFrame, norm: pd.DataFrame, title: str = ""):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.heatmap(raw, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0])
    ax[0].set(title="Counts", xlabel="Predicted", ylabel="True")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax[1])
    ax[1].set(title="Row-normalised (recall)", xlabel="Predicted", ylabel="")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def roc_figure(y_true, proba, classes, title: str = ""):
    import matplotlib.pyplot as plt

    Y = label_binarize(y_true, classes=classes)
    proba = np.asarray(proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs, sup = [], []
    for k, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
        aucs.append(auc(fpr, tpr)); sup.append(Y[:, k].sum())
        ax.plot(fpr, tpr, label=f"{c} (AUC {aucs[-1]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    macro, weighted = np.mean(aucs), np.average(aucs, weights=sup)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"{title}\nOvR AUC macro {macro:.3f} | weighted {weighted:.3f}".strip())
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _positional_cv(X, folds):
    pos = {rid: i for i, rid in enumerate(X.index)}
    return [([pos[i] for i in f["train_ids"]], [pos[i] for i in f["val_ids"]]) for f in folds]


def learning_curve_data(pipeline, X, y, folds, sizes=(0.2, 0.4, 0.6, 0.8, 1.0), n_jobs: int = 1) -> pd.DataFrame:
    """Train/CV macro-F1 vs training-set size, on the shared folds."""
    from sklearn.model_selection import learning_curve

    n, tr, va = learning_curve(pipeline, X, y, cv=_positional_cv(X, folds), train_sizes=list(sizes),
                               scoring="f1_macro", n_jobs=n_jobs, shuffle=True, random_state=42)
    return pd.DataFrame({"n_train": n, "train_mean": tr.mean(1), "train_std": tr.std(1),
                         "val_mean": va.mean(1), "val_std": va.std(1)})


def learning_curve_figure(df: pd.DataFrame, title: str = "", dummy_f1: float | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for col, lab in (("train", "Train"), ("val", "5-fold CV")):
        ax.plot(df["n_train"], df[f"{col}_mean"], "o-", label=lab)
        ax.fill_between(df["n_train"], df[f"{col}_mean"] - df[f"{col}_std"],
                        df[f"{col}_mean"] + df[f"{col}_std"], alpha=0.2)
    if dummy_f1 is not None:
        ax.axhline(dummy_f1, color="gray", ls=":", label="Dummy baseline")
    ax.set(xlabel="Training samples", ylabel="Macro-F1", title=title, ylim=(0, 1.02))
    ax.legend(); fig.tight_layout()
    return fig


def validation_curve_figure(df: pd.DataFrame, param: str, title: str = "", log_x: bool = False):
    """df columns: value, train_mean, train_std, val_mean, val_std."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(df)) if df["value"].dtype == object else df["value"]
    for col, lab in (("train", "Train"), ("val", "5-fold CV")):
        ax.errorbar(x, df[f"{col}_mean"], yerr=df[f"{col}_std"], fmt="o-", capsize=3, label=lab)
    if df["value"].dtype == object:
        ax.set_xticks(x, df["value"].astype(str), rotation=20)
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=param, ylabel="Macro-F1", title=title, ylim=(0, 1.02))
    ax.legend(); fig.tight_layout()
    return fig
