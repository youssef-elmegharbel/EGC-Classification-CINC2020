"""M3 -- feature selection (filter, wrapper, embedded, metaheuristic) on the 131 hand-crafted features.

Design rules (PHASE1_PLAN decisions 3, 7, 8):

* Everything is developed on the 80% training part. The 20% test set is never touched here.
* `Task` holds the training rows and the folds. For every fold, imputation + standardisation are fitted on
  that fold's TRAINING rows only (same preprocessing as M1's pipeline: median impute -> StandardScaler), so a
  wrapper / GA fitness never sees the statistics of the rows it is scored on.
* Fast estimator for all search methods: class-balanced multinomial logistic regression (C = 1). Fitness / score
  = mean macro-F1 over the folds.
* Early stopping: forward/backward selection stop when the CV score stops improving; the metaheuristics stop
  after `patience` generations without improvement, and the fitness penalises the number of features.
* The CV score that drives a search is optimistically biased (the search has seen the validation folds), so
  `nested_score` re-runs the WHOLE selection procedure inside each outer training fold (with its own inner
  folds) and scores the chosen subset on the untouched outer validation fold.

Filter rankings (variance, correlation, ANOVA, MI, chi2) are likewise computed on all training rows of the
  task; the nested CV repeats them inside each outer training fold, which measures the resulting selection bias.

Every selector has the signature  fn(task, k_fixed=None) -> dict(features=[names], meta={...}).
With `k_fixed=K` every method returns EXACTLY K features, so methods can be compared at equal subset size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import RFECV, f_classif, mutual_info_classif, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SEED = 42
N_JOBS = 4
K_GRID = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 65, 80, 100, 131]


# ---------------------------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------------------------
def _prep(Xa: pd.DataFrame, others: list[pd.DataFrame]):
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = sc.fit_transform(imp.fit_transform(Xa))
    return [A] + [sc.transform(imp.transform(o)) for o in others]


@dataclass
class Task:
    """Training rows + folds (lists of record ids). Builds per-fold arrays lazily and caches them."""
    X: pd.DataFrame
    y: pd.Series
    folds: list[dict]
    names: list[str] = field(init=False)
    Z: np.ndarray = field(init=False)           # whole training part, imputed + standardised (unsupervised stats)
    fold_data: list = field(init=False)         # [(Xa, ya, Xb, yb)] preprocessed with fold-train statistics
    pos_folds: list = field(init=False)         # positional (train, val) index pairs over X's rows

    def __post_init__(self):
        self.names = list(self.X.columns)
        self.Z = _prep(self.X, [])[0]
        pos = {r: i for i, r in enumerate(self.X.index)}
        self.pos_folds = [([pos[i] for i in f["train_ids"]], [pos[i] for i in f["val_ids"]]) for f in self.folds]
        self.fold_data = []
        for tr, va in self.pos_folds:
            Xa, Xb = _prep(self.X.iloc[tr], [self.X.iloc[va]])
            self.fold_data.append((Xa, self.y.iloc[tr].to_numpy(), Xb, self.y.iloc[va].to_numpy()))

    @property
    def p(self) -> int:
        return len(self.names)

    def cols(self, features) -> np.ndarray:
        idx = {n: i for i, n in enumerate(self.names)}
        return np.array(sorted(idx[f] for f in features))


def make_inner_task(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> Task:
    """Task on `X`,`y` with its own stratified folds (used for nested evaluation)."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    ids = X.index.to_numpy()
    folds = [{"train_ids": ids[a].tolist(), "val_ids": ids[b].tolist()} for a, b in skf.split(ids, y.to_numpy())]
    return Task(X, y, folds)


# ---------------------------------------------------------------------------------------------
# the scoring estimator
# ---------------------------------------------------------------------------------------------
def _lr(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=300, class_weight="balanced", random_state=SEED)


def fold_scores(fold_data, cols, C: float = 1.0) -> np.ndarray:
    """Macro-F1 per fold for the column subset `cols` (positional indices)."""
    cols = np.asarray(cols)
    out = []
    for Xa, ya, Xb, yb in fold_data:
        m = _lr(C).fit(Xa[:, cols], ya)
        out.append(f1_score(yb, m.predict(Xb[:, cols]), average="macro"))
    return np.array(out)


def _score_cols(fold_data, cols, C=1.0):
    s = fold_scores(fold_data, cols, C)
    return s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0


def _par_scores(task: Task, col_sets: list, n_jobs: int = N_JOBS) -> list[tuple[float, float]]:
    if n_jobs == 1 or len(col_sets) < 4:
        return [_score_cols(task.fold_data, c) for c in col_sets]
    return Parallel(n_jobs=n_jobs)(delayed(_score_cols)(task.fold_data, c) for c in col_sets)


def one_se_pick(means, stds, sizes, n_folds: int):
    """Smallest model whose mean is within one standard error of the best mean (parsimony rule)."""
    means, stds, sizes = map(np.asarray, (means, stds, sizes))
    b = int(np.nanargmax(means))
    thr = means[b] - stds[b] / np.sqrt(n_folds)
    ok = np.where(means >= thr)[0]
    return int(ok[np.argmin(sizes[ok])])


def _result(task: Task, cols, t0: float, **meta) -> dict:
    cols = sorted(int(c) for c in cols)
    return {"features": [task.names[c] for c in cols], "k": len(cols), "time_s": time.perf_counter() - t0,
            "meta": meta}


def choose_k(task: Task, ranking: list[int], ks=K_GRID) -> tuple[int, pd.DataFrame]:
    """CV curve over the top-k of a ranking; returns (k by 1-SE rule, curve)."""
    ks = [k for k in ks if k <= task.p]
    res = _par_scores(task, [ranking[:k] for k in ks])
    curve = pd.DataFrame({"k": ks, "cv_f1": [r[0] for r in res], "cv_std": [r[1] for r in res]})
    pick = one_se_pick(curve.cv_f1, curve.cv_std, curve.k, len(task.fold_data))
    return int(curve.k[pick]), curve


def _cut_ranking(task: Task, ranking: list[int], k_fixed: int | None):
    """Choose the cut of a ranking: CV curve + 1-SE rule, or exactly `k_fixed` features if given.
    Returns (k, curve); the curve is always computed so plots keep working."""
    k, curve = choose_k(task, ranking)
    if k_fixed is not None:
        k = min(int(k_fixed), len(ranking))
    return k, curve


# ---------------------------------------------------------------------------------------------
# 1. FILTER methods
# ---------------------------------------------------------------------------------------------
def filter_scores(task: Task) -> pd.DataFrame:
    """Univariate scores for every feature: ANOVA F, mutual information, chi2 (on min-max scaled data)."""
    Xi = SimpleImputer(strategy="median").fit_transform(task.X)
    Xmm = MinMaxScaler().fit_transform(Xi)
    lo, hi = np.percentile(Xi, 1, axis=0), np.percentile(Xi, 99, axis=0)        # winsorised variance: heavy tails would
    Xw = MinMaxScaler().fit_transform(np.clip(Xi, lo, hi))                        # otherwise make every feature look flat
    y = task.y.to_numpy()
    F, _ = f_classif(Xi, y)
    chi, _ = chi2(Xmm, y)
    mi = mutual_info_classif(Xi, y, random_state=SEED)
    df = pd.DataFrame({"anova_f": F, "mutual_info": mi, "chi2": chi, "var_minmax": Xw.var(0)}, index=task.names)
    return df


def correlation_prune(Z: np.ndarray, names: list[str], priority: pd.Series, thr: float = 0.95) -> list[str]:
    """Greedy: walk features by descending priority (e.g. ANOVA F); drop any feature whose |r| with an already
    kept feature exceeds `thr`."""
    C = np.abs(np.corrcoef(np.nan_to_num(Z), rowvar=False))
    order = [names.index(n) for n in priority.sort_values(ascending=False).index]
    kept: list[int] = []
    for i in order:
        if all(C[i, j] < thr for j in kept):
            kept.append(i)
    return [names[i] for i in sorted(kept)]


def select_variance(task: Task, thr: float = 0.02, k_fixed: int | None = None) -> dict:
    t0 = time.perf_counter()
    sc = filter_scores(task)
    keep = sc.index[sc.var_minmax > thr]
    if k_fixed is not None:                                   # the k highest-variance features
        keep = sc.var_minmax.sort_values(ascending=False).index[:k_fixed]
    return _result(task, task.cols(keep), t0, threshold=thr, removed=[n for n in task.names if n not in set(keep)])


def select_corr(task: Task, thr: float = 0.95, k_fixed: int | None = None) -> dict:
    t0 = time.perf_counter()
    sc = filter_scores(task)
    keep = correlation_prune(task.Z, task.names, sc["anova_f"], thr)
    if k_fixed is not None:                                   # de-correlated set, then the k best by ANOVA F
        keep = sc.anova_f[keep].sort_values(ascending=False).index[:k_fixed]
    return _result(task, task.cols(keep), t0, threshold=thr)


def _select_ranked(task: Task, score_name: str, tag: str, prune: bool = True, k_fixed: int | None = None) -> dict:
    t0 = time.perf_counter()
    sc = filter_scores(task)[score_name]
    cand = correlation_prune(task.Z, task.names, sc, 0.95) if prune else task.names
    rank = [task.names.index(n) for n in sc[cand].sort_values(ascending=False).index]
    k, curve = choose_k(task, rank, [x for x in K_GRID if x <= len(rank)])
    if k_fixed is not None:
        k = min(int(k_fixed), len(rank))
    return _result(task, rank[:k], t0, score=score_name, curve=curve.to_dict("list"), ranking=[task.names[i] for i in rank])


def select_anova(task: Task, k_fixed: int | None = None) -> dict:
    return _select_ranked(task, "anova_f", "ANOVA", k_fixed=k_fixed)


def select_mi(task: Task, k_fixed: int | None = None) -> dict:
    return _select_ranked(task, "mutual_info", "MI", k_fixed=k_fixed)


def select_chi2(task: Task, k_fixed: int | None = None) -> dict:
    return _select_ranked(task, "chi2", "chi2", k_fixed=k_fixed)


# ---------------------------------------------------------------------------------------------
# 2. WRAPPER methods
# ---------------------------------------------------------------------------------------------
def _greedy(task: Task, forward: bool, start: list[int] | None = None, tol: float = 5e-4, patience: int = 3,
            max_k: int = 60, log=None, chunk: int = 1, chunk_above: int = 40, k_fixed: int | None = None) -> dict:
    """Sequential forward (add best) / backward (drop least harmful) selection with early stopping:
    stop when the CV score has not improved on the best seen (by `tol`) for `patience` consecutive steps."""
    t0 = time.perf_counter()
    p = task.p
    if k_fixed is not None:                              # run exactly to k_fixed features, no early stopping
        patience, max_k = 10 ** 9, k_fixed
    cur = [] if forward else list(range(p) if start is None else start)
    path = []                                           # (size, mean, std, subset)
    best, stale = -np.inf, 0
    if not forward:
        m, s = _score_cols(task.fold_data, cur)
        path.append((len(cur), m, s, list(cur))); best = m
    while True:
        pool = [c for c in range(p) if c not in cur] if forward else list(cur)
        if (forward and len(cur) >= max_k) or (not forward and len(cur) <= max(2, k_fixed or 0)) or not pool:
            break
        sets = [sorted(cur + [c]) if forward else sorted(set(cur) - {c}) for c in pool]
        res = _par_scores(task, sets)
        j = int(np.argmax([r[0] for r in res]))
        if not forward and chunk > 1 and len(cur) > chunk_above and len(cur) - chunk >= (k_fixed or 0):
            # large backward set: drop the `chunk` least useful features in one step (scored individually)
            top = np.argsort([-r[0] for r in res])[:chunk]
            cur = sorted(set(cur) - {pool[i] for i in top})
            res[j] = _score_cols(task.fold_data, cur)
        else:
            cur = sets[j]
        path.append((len(cur), res[j][0], res[j][1], list(cur)))
        if res[j][0] > best + tol:
            best, stale = res[j][0], 0
        else:
            stale += 1
        if log:
            log(f"    {'fwd' if forward else 'bwd'} k={len(cur):3d} cv={res[j][0]:.4f} best={best:.4f} stale={stale}")
        if stale >= patience:
            break
    sizes = [x[0] for x in path]
    if k_fixed is not None:                              # the path entry of size k_fixed (or the closest one)
        pick = int(np.argmin([abs(sz - k_fixed) for sz in sizes]))
        return _result(task, path[pick][3], t0, steps=[{"k": a, "cv_f1": b, "cv_std": c} for a, b, c, _ in path],
                       direction="forward" if forward else "backward", k_fixed=k_fixed)
    pick = one_se_pick([x[1] for x in path], [x[2] for x in path], sizes, len(task.fold_data))
    return _result(task, path[pick][3], t0, steps=[{"k": a, "cv_f1": b, "cv_std": c} for a, b, c, _ in path],
                   direction="forward" if forward else "backward")


def select_sfs(task: Task, log=None, **kw) -> dict:
    return _greedy(task, True, log=log, **kw)


def select_sbs(task: Task, log=None, prune_thr: float = 0.95, **kw) -> dict:
    """Backward elimination (5 features per step while >40 remain, then 1) starting from the correlation-pruned set (|r| < 0.95): starting from all 131 highly
    redundant columns is 2-3x slower and the near-duplicates are removed first anyway."""
    sc = filter_scores(task)["anova_f"]
    start = [task.names.index(n) for n in correlation_prune(task.Z, task.names, sc, prune_thr)]
    kw.setdefault("chunk", 5)
    return _greedy(task, False, start=start, log=log, **kw)


def select_rfecv(task: Task, estimator: str = "logreg", k_fixed: int | None = None) -> dict:
    """Recursive feature elimination with CV-chosen size (RFECV) on the shared folds. For stability and
    speed the imputation/scaling statistics come from the whole training part (label-free; the nested CV in
    `nested_score` re-runs it per outer fold, which covers the small leak)."""
    t0 = time.perf_counter()
    if estimator == "logreg":
        est, step = LogisticRegression(C=0.1, max_iter=300, class_weight="balanced", random_state=SEED), 3
    else:
        import lightgbm as lgb
        est = lgb.LGBMClassifier(n_estimators=120, num_leaves=15, learning_rate=0.1, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, class_weight="balanced", random_state=SEED, n_jobs=1, verbose=-1)
        step = 5
    if k_fixed is not None:                                   # plain RFE down to exactly k_fixed features
        from sklearn.feature_selection import RFE
        sel = RFE(est, n_features_to_select=k_fixed, step=step).fit(task.Z, task.y.to_numpy())
        return _result(task, np.where(sel.support_)[0], t0, estimator=estimator, k_fixed=k_fixed)
    sel = RFECV(est, step=step, cv=task.pos_folds, scoring="f1_macro", min_features_to_select=3, n_jobs=N_JOBS)
    sel.fit(task.Z, task.y.to_numpy())
    cv = sel.cv_results_
    return _result(task, np.where(sel.support_)[0], t0, estimator=estimator,
                   curve={"mean": [float(x) for x in cv["mean_test_score"]], "std": [float(x) for x in cv["std_test_score"]],
                          "step": step, "n_features_total": task.p})


def select_rfecv_lgbm(task: Task, k_fixed: int | None = None) -> dict:
    return select_rfecv(task, "lgbm", k_fixed=k_fixed)


# ---------------------------------------------------------------------------------------------
# 3. EMBEDDED methods (L1, L2, Elastic Net, tree importance)
# ---------------------------------------------------------------------------------------------
C_GRID = np.logspace(-3, 0, 9)


def _penalised(l1_ratio: float, C: float):
    return LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=400, tol=1e-3,
                              class_weight="balanced", random_state=SEED)


def _fit_pen(task, l1_ratio, C):
    import warnings
    warnings.filterwarnings("ignore")
    sc, nnz = [], []
    for Xa, ya, Xb, yb in task.fold_data:
        m = _penalised(l1_ratio, C).fit(Xa, ya)
        sc.append(f1_score(yb, m.predict(Xb), average="macro"))
    m = _penalised(l1_ratio, C).fit(task.Z, task.y.to_numpy())
    support = np.where(np.abs(m.coef_).max(axis=0) > 1e-8)[0]
    return float(np.mean(sc)), float(np.std(sc, ddof=1)), support, np.abs(m.coef_).mean(axis=0)


def regularisation_path(task: Task, l1_ratio: float, grid=C_GRID) -> pd.DataFrame:
    """For each C: CV macro-F1 of the penalised model and the number of features with a non-zero coefficient."""
    res = Parallel(n_jobs=N_JOBS)(delayed(_fit_pen)(task, l1_ratio, float(C)) for C in grid)
    return pd.DataFrame({"C": grid, "cv_f1": [r[0] for r in res], "cv_std": [r[1] for r in res],
                         "n_nonzero": [len(r[2]) for r in res],
                         "_support": [r[2] for r in res], "_absw": [r[3] for r in res]})


def _embedded_from_path(task, path: pd.DataFrame, t0, l1_ratio):
    pick = one_se_pick(path.cv_f1, path.cv_std, path.n_nonzero.where(path.n_nonzero > 0, 10 ** 6), len(task.fold_data))
    return _result(task, path._support[pick], t0, l1_ratio=l1_ratio, C=float(path.C[pick]),
                   path=path.drop(columns=["_support", "_absw"]).to_dict("list"))


def _embedded_fixed(task: Task, path: pd.DataFrame, k: int):
    """Sparsest C that still has >= k non-zero coefficients (else the weakest penalty), then the k features with
    the largest mean |coef|. Returns (cols, C)."""
    ok = path.index[path.n_nonzero >= k]
    i = int(path.loc[ok, "n_nonzero"].idxmin()) if len(ok) else int(path.n_nonzero.idxmax())
    w = path._absw[i]
    sup = path._support[i]
    cols = sup[np.argsort(-w[sup])][:k] if len(sup) >= k else np.argsort(-w)[:k]
    return cols, float(path.C[i])


def select_l1(task: Task, k_fixed: int | None = None) -> dict:
    t0 = time.perf_counter()
    path = regularisation_path(task, 1.0)
    if k_fixed is not None:
        cols, C = _embedded_fixed(task, path, k_fixed)
        return _result(task, cols, t0, l1_ratio=1.0, C=C, k_fixed=k_fixed,
                       path=path.drop(columns=["_support", "_absw"]).to_dict("list"))
    return _embedded_from_path(task, path, t0, 1.0)


def select_elasticnet(task: Task, ratios=(0.25, 0.5, 0.75), k_fixed: int | None = None) -> dict:
    """Grid over (l1_ratio, C); keep the sparsest model within one SE of the best CV macro-F1."""
    t0 = time.perf_counter()
    paths = {r: regularisation_path(task, r) for r in ratios}
    if k_fixed is not None:                              # per ratio: top-k at the sparsest C with >= k; keep best CV
        cands = {}
        for r in ratios:
            cols, C = _embedded_fixed(task, paths[r], k_fixed)
            cands[r] = (cols, C, S_mean(task, cols))
        r = max(cands, key=lambda q: cands[q][2])
        return _result(task, cands[r][0], t0, l1_ratio=r, C=cands[r][1], k_fixed=k_fixed,
                       grid={str(rr): paths[rr].drop(columns=["_support", "_absw"]).to_dict("list") for rr in ratios})
    rows = [(r, i) for r in ratios for i in range(len(C_GRID))]
    means = [paths[r].cv_f1[i] for r, i in rows]
    stds = [paths[r].cv_std[i] for r, i in rows]
    nnz = [max(paths[r].n_nonzero[i], 1) for r, i in rows]
    r, i = rows[one_se_pick(means, stds, nnz, len(task.fold_data))]
    return _result(task, paths[r]._support[i], t0, l1_ratio=r, C=float(C_GRID[i]),
                   grid={str(rr): paths[rr].drop(columns=["_support", "_absw"]).to_dict("list") for rr in ratios})


def S_mean(task: Task, cols) -> float:
    return float(fold_scores(task.fold_data, cols).mean())


def select_l2(task: Task, k_fixed: int | None = None) -> dict:
    """L2 never zeroes a coefficient, so rank features by mean |coef| (standardised inputs) at the CV-best C and
    cut the ranking with the CV curve (1-SE rule)."""
    t0 = time.perf_counter()
    path = regularisation_path(task, 0.0)
    b = int(path.cv_f1.idxmax())
    w = path._absw[b]
    rank = list(np.argsort(-w))
    k, curve = _cut_ranking(task, rank, k_fixed)
    return _result(task, rank[:k], t0, C=float(path.C[b]), curve=curve.to_dict("list"),
                   ranking=[task.names[i] for i in rank], weights=[float(w[i]) for i in rank])


def select_tree_importance(task: Task, kind: str = "rf", k_fixed: int | None = None) -> dict:
    """Bonus embedded view: impurity importance of a random forest, cut with the CV curve."""
    t0 = time.perf_counter()
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", class_weight="balanced",
                                n_jobs=N_JOBS, random_state=SEED).fit(task.Z, task.y.to_numpy())
    imp = rf.feature_importances_
    rank = list(np.argsort(-imp))
    k, curve = _cut_ranking(task, rank, k_fixed)
    return _result(task, rank[:k], t0, curve=curve.to_dict("list"),
                   ranking=[task.names[i] for i in rank], weights=[float(imp[i]) for i in rank])


# ---------------------------------------------------------------------------------------------
# 4. METAHEURISTICS: GA (DEAP), binary PSO, binary Grey Wolf
# ---------------------------------------------------------------------------------------------
LAMBDA = 0.05          # fitness = CV macro-F1 - LAMBDA * (#features / p)


def _fitness(fold_data, mask, lam: float, p: int):
    cols = np.flatnonzero(mask)
    if len(cols) == 0:
        return -1.0, 0.0
    m, s = _score_cols(fold_data, cols)
    return m - lam * len(cols) / p, m


class _Evaluator:
    """Cached, parallel fitness evaluation of binary masks."""

    def __init__(self, task: Task, lam: float = LAMBDA, n_jobs: int = N_JOBS):
        self.task, self.lam, self.n_jobs, self.cache = task, lam, n_jobs, {}
        self.n_evals = 0

    def __call__(self, masks: list[np.ndarray]) -> list[tuple[float, float]]:
        keys = [np.packbits(np.asarray(m, dtype=np.uint8)).tobytes() for m in masks]
        todo = {k: m for k, m in zip(keys, masks) if k not in self.cache}
        if todo:
            res = Parallel(n_jobs=self.n_jobs)(delayed(_fitness)(self.task.fold_data, np.asarray(m), self.lam, self.task.p)
                                               for m in todo.values())
            for k, r in zip(todo, res):
                self.cache[k] = r
            self.n_evals += len(todo)
        return [self.cache[k] for k in keys]


def _history_row(gen, fits, masks, best):
    f = np.array([x[0] for x in fits])
    return {"gen": gen, "best_fitness": float(best), "mean_fitness": float(f.mean()),
            "best_k": int(np.asarray(masks[int(f.argmax())]).sum()), "mean_k": float(np.mean([np.sum(m) for m in masks]))}


def select_ga(task: Task, pop: int = 40, ngen: int = 40, patience: int = 8, cxpb: float = 0.7, mutpb: float = 0.4,
              indpb: float = 0.03, init_p: float = 0.15, lam: float = LAMBDA, seed: int = SEED, log=None,
              k_fixed: int | None = None) -> dict:
    """Genetic algorithm (DEAP): binary chromosome of length p, fitness = CV macro-F1 - lam * k/p,
    tournament selection (size 3), two-point crossover, bit-flip mutation, elitism via a hall of fame.
    Stops after `patience` generations without improvement of the best fitness."""
    from deap import base, creator, tools
    import random

    t0 = time.perf_counter()
    random.seed(seed); np.random.seed(seed)
    rng_fix = np.random.default_rng(seed)
    if k_fixed is not None:
        lam = 0.0                                          # size is fixed, no size penalty needed
    if not hasattr(creator, "FitnessMaxM3"):
        creator.create("FitnessMaxM3", base.Fitness, weights=(1.0,))
        creator.create("IndividualM3", list, fitness=creator.FitnessMaxM3)
    tb = base.Toolbox()
    tb.register("attr", lambda: int(random.random() < init_p))
    if k_fixed is None:
        tb.register("individual", tools.initRepeat, creator.IndividualM3, tb.attr, n=task.p)
    else:
        tb.register("individual", lambda: creator.IndividualM3(_rand_masks(1, task.p, k_fixed, rng_fix)[0].tolist()))
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", tools.cxTwoPoint)
    tb.register("mutate", tools.mutFlipBit, indpb=indpb)
    ev = _Evaluator(task, lam)

    def evaluate(inds):
        res = ev([np.array(i) for i in inds])
        for i, r in zip(inds, res):
            i.fitness.values = (r[0],)
        return res

    popn = tb.population(n=pop)
    hist, best, stale = [], -np.inf, 0
    hof = tools.HallOfFame(1)
    res = evaluate(popn); hof.update(popn)
    hist.append(_history_row(0, res, popn, hof[0].fitness.values[0]))
    best = hof[0].fitness.values[0]
    for gen in range(1, ngen + 1):
        off = [tb.clone(i) for i in tb.select(popn, len(popn) - 1)]
        for a, b in zip(off[::2], off[1::2]):
            if random.random() < cxpb:
                tb.mate(a, b); del a.fitness.values, b.fitness.values
        for m in off:
            if random.random() < mutpb:
                tb.mutate(m); del m.fitness.values
        inv = [i for i in off if not i.fitness.valid]
        if k_fixed is not None:
            for i in inv:
                _repair(i, k_fixed, rng_fix)
        evaluate(inv)
        popn = off + [tb.clone(hof[0])]                     # elitism
        res = [(i.fitness.values[0], 0) for i in popn]
        hof.update(popn)
        cur = hof[0].fitness.values[0]
        hist.append(_history_row(gen, res, popn, cur))
        if cur > best + 1e-4:
            best, stale = cur, 0
        else:
            stale += 1
        if log:
            log(f"    GA gen {gen:2d} best={cur:.4f} k={sum(hof[0])} mean_k={hist[-1]['mean_k']:.1f} stale={stale}")
        if stale >= patience:
            break
    mask = np.array(hof[0])
    return _result(task, np.flatnonzero(mask), t0, history=hist, n_evals=ev.n_evals, lam=lam, pop=pop,
                   cv_f1=ev([mask])[0][1], stopped_gen=len(hist) - 1)


def _rand_masks(n: int, p: int, k: int, rng) -> np.ndarray:
    """n random binary masks with exactly k ones."""
    X = np.zeros((n, p), dtype=int)
    for i in range(n):
        X[i, rng.choice(p, k, replace=False)] = 1
    return X


def _topk_masks(score: np.ndarray, k: int) -> np.ndarray:
    """Binary masks keeping the k largest entries of every row of `score`."""
    idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
    X = np.zeros(score.shape, dtype=int)
    np.put_along_axis(X, idx, 1, axis=1)
    return X


def _repair(ind, k: int, rng) -> None:
    """In place: make the 0/1 list `ind` hold exactly k ones (drop / add random bits)."""
    ones = [i for i, b in enumerate(ind) if b]
    zeros = [i for i, b in enumerate(ind) if not b]
    if len(ones) > k:
        for i in rng.choice(ones, len(ones) - k, replace=False):
            ind[int(i)] = 0
    elif len(ones) < k:
        for i in rng.choice(zeros, k - len(ones), replace=False):
            ind[int(i)] = 1


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def select_pso(task: Task, n: int = 30, iters: int = 40, patience: int = 8, w_hi: float = 0.9, w_lo: float = 0.4,
               c1: float = 1.5, c2: float = 1.5, vmax: float = 4.0, init_p: float = 0.15, lam: float = LAMBDA,
               seed: int = SEED, log=None, k_fixed: int | None = None) -> dict:
    """Binary PSO (Kennedy & Eberhart 1997): velocity update as in continuous PSO with inertia decreasing
    from w_hi to w_lo; bit i is set with probability sigmoid(v_i)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    p = task.p
    if k_fixed is not None:
        lam = 0.0
    ev = _Evaluator(task, lam)
    X = (rng.random((n, p)) < init_p).astype(int) if k_fixed is None else _rand_masks(n, p, k_fixed, rng)
    V = rng.uniform(-1, 1, (n, p))
    fit = np.array([r[0] for r in ev(list(X))])
    pbest, pfit = X.copy(), fit.copy()
    g = int(fit.argmax()); gbest, gfit = X[g].copy(), fit[g]
    hist, best, stale = [_history_row(0, [(f, 0) for f in fit], list(X), gfit)], gfit, 0
    for it in range(1, iters + 1):
        w = w_hi - (w_hi - w_lo) * it / iters
        V = w * V + c1 * rng.random((n, p)) * (pbest - X) + c2 * rng.random((n, p)) * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        if k_fixed is None:
            X = (rng.random((n, p)) < _sigmoid(V)).astype(int)
            X[X.sum(1) == 0, rng.integers(p)] = 1
        else:                                              # stochastic top-k of sigmoid(V)
            X = _topk_masks(_sigmoid(V) * rng.random((n, p)), k_fixed)
        fit = np.array([r[0] for r in ev(list(X))])
        imp = fit > pfit
        pbest[imp], pfit[imp] = X[imp], fit[imp]
        g = int(pfit.argmax())
        if pfit[g] > gfit:
            gbest, gfit = pbest[g].copy(), pfit[g]
        hist.append(_history_row(it, [(f, 0) for f in fit], list(X), gfit))
        if gfit > best + 1e-4:
            best, stale = gfit, 0
        else:
            stale += 1
        if log:
            log(f"    PSO it {it:2d} best={gfit:.4f} k={int(gbest.sum())} stale={stale}")
        if stale >= patience:
            break
    return _result(task, np.flatnonzero(gbest), t0, history=hist, n_evals=ev.n_evals, lam=lam, swarm=n,
                   cv_f1=ev([gbest])[0][1], stopped_gen=len(hist) - 1)


def select_gwo(task: Task, n: int = 30, iters: int = 40, patience: int = 8, init_p: float = 0.15,
               lam: float = LAMBDA, seed: int = SEED, log=None, k_fixed: int | None = None) -> dict:
    """Binary Grey Wolf Optimizer (after Emary et al. 2016): the three best wolves (alpha, beta, delta) guide the
    pack; each leader proposes a bit vector from a sigmoid-squashed step, and the new position is sampled from the
    average of the three proposals. NOTE: this is a simplified variant -- the crossover step of Emary et al. is
    replaced by averaging the three leader proposals, so results are not a faithful reproduction of the paper. The coefficient `a` decays linearly from 2 to 0 (exploration -> exploitation)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    p = task.p
    ev = _Evaluator(task, lam)
    if k_fixed is not None:
        lam = 0.0
        ev = _Evaluator(task, lam)
    X = (rng.random((n, p)) < init_p).astype(int) if k_fixed is None else _rand_masks(n, p, k_fixed, rng)
    fit = np.array([r[0] for r in ev(list(X))])
    order = np.argsort(-fit)
    lead = [X[order[i]].copy() for i in range(3)]; lfit = [fit[order[i]] for i in range(3)]
    hist, best, stale = [_history_row(0, [(f, 0) for f in fit], list(X), lfit[0])], lfit[0], 0
    for it in range(1, iters + 1):
        a = 2 * (1 - it / iters)
        props = []
        for L in lead:
            A = 2 * a * rng.random((n, p)) - a
            C = 2 * rng.random((n, p))
            D = np.abs(C * L - X)
            bstep = (_sigmoid(10 * (A * D - 0.5)) >= rng.random((n, p))).astype(int)
            props.append(((L + bstep) >= 1).astype(int))
        prob = sum(props) / 3.0
        if k_fixed is None:
            X = (rng.random((n, p)) < prob).astype(int)
            X[X.sum(1) == 0, rng.integers(p)] = 1
        else:
            X = _topk_masks(prob * rng.random((n, p)), k_fixed)
        fit = np.array([r[0] for r in ev(list(X))])
        allX = np.vstack([X] + [l[None] for l in lead]); allf = np.concatenate([fit, lfit])
        top = np.argsort(-allf)[:3]
        lead = [allX[i].copy() for i in top]; lfit = [allf[i] for i in top]
        hist.append(_history_row(it, [(f, 0) for f in fit], list(X), lfit[0]))
        if lfit[0] > best + 1e-4:
            best, stale = lfit[0], 0
        else:
            stale += 1
        if log:
            log(f"    GWO it {it:2d} best={lfit[0]:.4f} k={int(lead[0].sum())} stale={stale}")
        if stale >= patience:
            break
    return _result(task, np.flatnonzero(lead[0]), t0, history=hist, n_evals=ev.n_evals, lam=lam, wolves=n,
                   cv_f1=ev([lead[0]])[0][1], stopped_gen=len(hist) - 1)


# ---------------------------------------------------------------------------------------------
# registry + nested evaluation
# ---------------------------------------------------------------------------------------------
METHODS = {
    # filter
    "filter_variance": select_variance, "filter_corr": select_corr,
    "filter_anova": select_anova, "filter_mi": select_mi, "filter_chi2": select_chi2,
    # wrapper
    "wrapper_sfs": select_sfs, "wrapper_sbs": select_sbs,
    "wrapper_rfecv_logreg": select_rfecv, "wrapper_rfecv_lgbm": select_rfecv_lgbm,
    # embedded
    "embedded_l1": select_l1, "embedded_l2": select_l2, "embedded_elasticnet": select_elasticnet,
    "embedded_rf": select_tree_importance,
    # metaheuristic
    "meta_ga": select_ga, "meta_pso": select_pso, "meta_gwo": select_gwo,
}
FAMILY = {k: k.split("_")[0] for k in METHODS}
# smaller budgets inside nested CV (3 inner folds, fewer generations)
NESTED_KW = {"wrapper_sfs": dict(max_k=30, tol=1e-3, patience=2), "wrapper_sbs": dict(tol=1e-3, patience=2),
             "meta_ga": dict(pop=30, ngen=25, patience=5), "meta_pso": dict(n=24, iters=25, patience=5),
             "meta_gwo": dict(n=24, iters=25, patience=5)}


def nested_score(name: str, X: pd.DataFrame, y: pd.Series, outer_folds: list[dict], inner_splits: int = 3,
                 log=print, k_fixed: int | None = None) -> pd.DataFrame:
    """Selection bias check. For each outer fold: run the complete selection procedure `name` on the outer
    TRAINING rows (own inner folds), then fit logistic regression on those rows with the chosen features and score
    it on the outer validation rows, which the search never saw. Returns one row per outer fold."""
    rows = []
    fn = METHODS[name]
    kw = dict(NESTED_KW.get(name, {}))
    if k_fixed is not None:
        kw["k_fixed"] = k_fixed                       # wrappers ignore max_k / patience when k_fixed is set
    for i, f in enumerate(outer_folds):
        Xa, ya = X.loc[f["train_ids"]], y.loc[f["train_ids"]]
        Xb, yb = X.loc[f["val_ids"]], y.loc[f["val_ids"]]
        t0 = time.perf_counter()
        inner = make_inner_task(Xa, ya, inner_splits)
        sel = fn(inner, **kw) if kw else fn(inner)
        feats = sel["features"]
        A, B = _prep(Xa[feats], [Xb[feats]])
        m = _lr().fit(A, ya.to_numpy())
        pred = m.predict(B)
        rows.append({"method": name, "outer_fold": i, "k": len(feats), "f1_macro": f1_score(yb, pred, average="macro"),
                     "accuracy": float((pred == yb.to_numpy()).mean()), "time_s": time.perf_counter() - t0,
                     "features": feats})
        log(f"  nested {name} fold {i}: k={len(feats)} F1={rows[-1]['f1_macro']:.3f} ({rows[-1]['time_s']:.0f}s)")
    return pd.DataFrame(rows)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0
