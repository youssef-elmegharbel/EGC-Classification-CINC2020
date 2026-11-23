"""Sanity checks for src/selection.py on a synthetic problem (fast, no real data needed)."""
import numpy as np
import pandas as pd
import pytest

from src import selection as S


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    n, p, inf = 300, 20, 4
    y = rng.integers(0, 3, n)
    X = rng.normal(size=(n, p))
    X[:, :inf] += y[:, None] * 1.5                       # features 0..3 informative
    X[:, 4] = X[:, 0] + rng.normal(scale=0.01, size=n)   # near-duplicate of feature 0
    X[rng.random((n, p)) < 0.01] = np.nan
    ids = [f"r{i}" for i in range(n)]
    Xd = pd.DataFrame(X, index=ids, columns=[f"f{i}" for i in range(p)])
    yd = pd.Series(np.array(["A", "B", "C"])[y], index=ids)
    return S.make_inner_task(Xd, yd, 3), Xd, yd


def test_one_se_prefers_smaller_model():
    assert S.one_se_pick([0.80, 0.808, 0.81], [0.01, 0.01, 0.01], [30, 10, 50], 5) == 1


def test_fold_preprocessing_is_per_fold(toy):
    task, *_ = toy
    Xa, ya, Xb, yb = task.fold_data[0]
    assert abs(Xa.mean()) < 1e-9 and not np.isnan(Xb).any()
    # Xb must equal the validation rows transformed with the TRAINING rows' median / mean / sd
    f = task.folds[0]
    tr, va = task.X.loc[f["train_ids"]], task.X.loc[f["val_ids"]]
    imp = tr.median()
    tr_i, va_i = tr.fillna(imp), va.fillna(imp)
    expected = (va_i - tr_i.mean()) / tr_i.std(ddof=0)
    np.testing.assert_allclose(Xb, expected.to_numpy(), atol=1e-8)


def test_l1_and_elasticnet_are_really_sparse(toy):
    """Guards against sklearn versions where l1_ratio is silently ignored (L1 behaving like L2)."""
    task, *_ = toy
    m = S._penalised(1.0, 0.05).fit(task.Z, task.y.to_numpy())
    assert (np.abs(m.coef_).max(axis=0) > 0).sum() < task.p
    m2 = S._penalised(0.5, 0.05).fit(task.Z, task.y.to_numpy())
    assert (np.abs(m2.coef_).max(axis=0) > 0).sum() < task.p


def test_filters_find_informative_and_prune_duplicates(toy):
    task, *_ = toy
    assert set(S.select_anova(task)["features"]) & {"f0", "f1", "f2", "f3"}
    kept = S.select_corr(task, 0.95)["features"]
    assert not {"f0", "f4"} <= set(kept)


@pytest.mark.parametrize("fn", [S.select_sfs, S.select_l1, S.select_ga, S.select_pso, S.select_gwo])
def test_selectors_return_valid_subsets(toy, fn):
    task, *_ = toy
    kw = {"select_ga": dict(pop=12, ngen=6), "select_pso": dict(n=10, iters=6), "select_gwo": dict(n=10, iters=6),
          "select_sfs": dict(max_k=6)}.get(fn.__name__, {})
    res = fn(task, **kw)
    assert 1 <= res["k"] <= task.p and set(res["features"]) <= set(task.names)
    assert len(set(res["features"]) & {"f0", "f1", "f2", "f3"}) >= 1


def test_nested_uses_unseen_validation_rows(toy):
    task, Xd, yd = toy
    S.METHODS["_t"] = S.select_anova
    try:
        df = S.nested_score("_t", Xd, yd, task.folds, inner_splits=3, log=lambda *a: None)
    finally:
        del S.METHODS["_t"]
    assert len(df) == len(task.folds) and (df.f1_macro > 0.5).all()
