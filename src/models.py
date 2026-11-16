"""M1 model factories, fixed small grids, and the grid runner (plan 2.7 / section 6).

Pipeline = [ColumnSelector (set c)] -> SimpleImputer(median) -> StandardScaler -> [reducer (set b)] -> clf.
Everything is fit on training folds only because the whole thing is one sklearn Pipeline.

Early stopping / complexity control (plan decision 8):
  LightGBM  : EarlyStoppingLGBM carves a stratified validation split from its OWN training data
  MLP       : early_stopping=True, validation_fraction=0.1, n_iter_no_change=10
  RF        : max_depth / min_samples_leaf grid
  LogReg/SVM: tuned on C
Class weights: LogReg, SVM, RF, LightGBM use class_weight="balanced". k-NN and sklearn's MLP
have no class_weight option, so they run unweighted (documented limitation).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate import cv_evaluate

SEED = 42
N_JOBS = 10


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Keep a fixed list of named columns (set c: M3's selected features)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)] if self.columns is not None else X


class EarlyStoppingLGBM(BaseEstimator, ClassifierMixin):
    """LightGBM with early stopping on a validation split carved from its own training data,
    so it works inside outer CV without ever touching the outer validation fold."""

    def __init__(self, num_leaves=31, learning_rate=0.1, n_estimators=600, min_child_samples=20,
                 reg_lambda=0.0, class_weight="balanced", val_fraction=0.1, early_stopping_rounds=30,
                 random_state=SEED, n_jobs=N_JOBS):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        import lightgbm as lgb

        X, y = np.asarray(X, dtype=float), np.asarray(y)
        Xa, Xv, ya, yv = train_test_split(X, y, test_size=self.val_fraction, stratify=y,
                                          random_state=self.random_state)
        self.model_ = lgb.LGBMClassifier(
            num_leaves=self.num_leaves, learning_rate=self.learning_rate, n_estimators=self.n_estimators,
            min_child_samples=self.min_child_samples, reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8)
        self.model_.fit(Xa, ya, eval_set=[(Xv, yv)],
                        callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        self.classes_ = self.model_.classes_
        self.best_iteration_ = self.model_.best_iteration_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def _clf(name: str):
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "knn": KNeighborsClassifier(weights="distance"),
        "svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=N_JOBS, random_state=SEED),
        "lgbm": EarlyStoppingLGBM(),
        "mlp": MLPClassifier(early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                             max_iter=400, random_state=SEED),
    }[name]


MODEL_NAMES = ["logreg", "knn", "svm", "rf", "lgbm", "mlp"]

GRIDS = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "knn": {"clf__n_neighbors": [3, 5, 11, 21]},
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01]},
    "rf": {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 5, 10]},
    "lgbm": {"clf__num_leaves": [15, 31], "clf__learning_rate": [0.05, 0.1]},
    "mlp": {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
}
# Round 2: stronger regularisation, run for models whose round-1 best was flagged overfit (plan decision 8).
GRIDS_R2 = {
    "svm": {"clf__C": [0.3, 1, 3], "clf__gamma": [0.003, 0.01, "scale"]},
    "rf": {"clf__max_depth": [10, 14], "clf__min_samples_leaf": [2, 3], "clf__max_features": [0.1, "sqrt"]},
    "lgbm": {"clf__num_leaves": [7, 15], "clf__learning_rate": [0.03], "clf__min_child_samples": [20, 50],
             "clf__reg_lambda": [1.0, 10.0]},
    "mlp": {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [0.1, 1.0]},
    "knn": {"clf__n_neighbors": [5, 11, 21], "clf__p": [1]},
}
# Models that parallelise internally keep folds sequential; single-threaded ones run folds in parallel.
FOLD_JOBS = {"logreg": 5, "knn": 5, "svm": 5, "rf": 1, "lgbm": 1, "mlp": 5}
# The "main" hyperparameter used for validation curves.
MAIN_PARAM = {"logreg": ("clf__C", [0.001, 0.01, 0.1, 1, 10, 100]),
              "knn": ("clf__n_neighbors", [1, 3, 5, 11, 21, 41]),
              "svm": ("clf__C", [0.1, 1, 10, 100, 1000]),
              "rf": ("clf__max_depth", [2, 4, 8, 16, 32]),
              "lgbm": ("clf__num_leaves", [4, 8, 15, 31, 63]),
              "mlp": ("clf__alpha", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])}


def make_model(name: str, params: dict | None = None, selector: list[str] | None = None,
               reducer=None) -> Pipeline:
    """Build the pipeline for one model. `params` use sklearn's `clf__<name>` keys."""
    steps = []
    if selector is not None:
        steps.append(("select", ColumnSelector(selector)))
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if reducer is not None:
        steps.append(("reduce", reducer))
    steps.append(("clf", _clf(name)))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe


def grid_combos(name: str, grid: dict | None = None) -> list[dict]:
    g = grid or GRIDS[name]
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def run_grid(name: str, X, y, folds, feature_set: str, dummy_f1: float, selector=None, reducer=None,
             grid: dict | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    """Evaluate every grid point with 5-fold CV. Returns (table of rows, best result dict).
    Best = highest CV macro-F1. Each row carries model, feature_set, params, scores, flag, runtime."""
    rows, best = [], None
    for params in grid_combos(name, grid):
        t = time.time()
        pipe = make_model(name, params, selector, reducer)
        res = cv_evaluate(pipe, X, y, folds, f"{name}|{feature_set}", n_jobs=FOLD_JOBS[name], dummy_f1=dummy_f1)
        row = {**res["row"], "model": name, "feature_set": feature_set, "params": str(params),
               "wall_s": time.time() - t}
        rows.append(row)
        log(f"  {name:6s} {str(params):60s} cvF1={row['f1_macro_val_mean']:.3f} trainF1={row['f1_macro_train_mean']:.3f} "
            f"{row['flag']:8s} {row['wall_s']:.0f}s")
        if best is None or row["f1_macro_val_mean"] > best["row"]["f1_macro_val_mean"]:
            best = {"row": row, "params": params, "oof": res["oof"]}
    return pd.DataFrame(rows), best


# ---------------------------------------------------------------------------------------------
# helpers used by the notebook (keep notebook cells thin)
# ---------------------------------------------------------------------------------------------
def best_per_model(feature_set: str, path=None) -> pd.DataFrame:
    """Best config per model for a feature set across all rounds, by 5-fold CV macro-F1 (never test)."""
    import ast
    from .loaders import MODELS_DIR

    df = pd.read_csv(path or MODELS_DIR / "model_comparison.csv")
    df = df[df["feature_set"] == feature_set]
    best = df.loc[df.groupby("model")["f1_macro_val_mean"].idxmax()].copy()
    best["params_dict"] = best["params"].map(ast.literal_eval)
    best["model"] = pd.Categorical(best["model"], MODEL_NAMES, ordered=True)
    return best.sort_values("model").reset_index(drop=True)


def validation_curve_data(name: str, base_params: dict, X, y, folds, selector=None, reducer=None) -> pd.DataFrame:
    """Train/CV macro-F1 as the model's main hyperparameter varies (other params at their best value)."""
    param, values = MAIN_PARAM[name]
    rows = []
    for v in values:
        pipe = make_model(name, {**base_params, param: v}, selector, reducer)
        r = cv_evaluate(pipe, X, y, folds, name, n_jobs=FOLD_JOBS[name])["row"]
        rows.append({"value": v, "train_mean": r["f1_macro_train_mean"], "train_std": r["f1_macro_train_std"],
                     "val_mean": r["f1_macro_val_mean"], "val_std": r["f1_macro_val_std"]})
    return pd.DataFrame(rows)
