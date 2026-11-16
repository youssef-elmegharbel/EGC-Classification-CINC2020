"""evaluate.py contract tests on a synthetic 3-class problem (no dataset needed)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src import evaluate as ev


@pytest.fixture()
def toy():
    rng = np.random.default_rng(0)
    classes = ["NSR", "AF", "SB"]
    y = pd.Series(rng.choice(classes, 300), index=[f"r{i}" for i in range(300)])
    X = pd.DataFrame(rng.normal(size=(300, 5)), index=y.index, columns=list("abcde"))
    X["a"] += y.map({"NSR": 0, "AF": 2, "SB": -2}).to_numpy()
    ids = y.index.to_numpy()
    folds = [{"train_ids": ids[np.arange(300) % 5 != k].tolist(), "val_ids": ids[np.arange(300) % 5 == k].tolist()}
             for k in range(5)]
    splits = {"train_ids": ids[:240].tolist(), "test_ids": ids[240:].tolist(), "folds": folds}
    return X, y, folds, splits


def test_cv_evaluate_outputs(toy):
    X, y, folds, _ = toy
    r = ev.cv_evaluate(LogisticRegression(max_iter=500), X, y, folds, "lr", dummy_f1=0.2)
    row = r["row"]
    for k in ("accuracy_val_mean", "accuracy_val_std", "f1_macro_train_mean", "f1_weighted_val_mean",
              "roc_auc_macro_val_mean", "roc_auc_weighted_val_mean", "error_rate_val_mean",
              "precision_macro_val_mean", "recall_weighted_val_mean"):
        assert k in row and np.isfinite(row[k]), k
    assert row["error_rate_val_mean"] == pytest.approx(1 - row["accuracy_val_mean"])
    assert len(r["oof"]) == 300 and r["oof"].index.is_unique  # every record validated exactly once
    assert row["flag"] == "ok"


def test_dummy_and_flags(toy):
    X, y, folds, _ = toy
    d = ev.dummy_baseline(X, y, folds)
    assert {r["name"] for r in d} == {"dummy_stratified", "dummy_most_frequent"}
    assert ev.flag_fit(1.0, 0.8, 0.2) == "overfit"
    assert ev.flag_fit(0.22, 0.21, 0.2) == "underfit"
    assert ev.flag_fit(0.9, 0.85, 0.2) == "ok"
    deep = ev.cv_evaluate(DecisionTreeClassifier(random_state=0), X, y, folds, "deep", dummy_f1=0.3)
    assert deep["row"]["gap_f1_macro"] > 0.1 and deep["row"]["flag"] == "overfit"


def test_test_once_guard(toy, tmp_path):
    X, y, _, splits = toy
    log = tmp_path / "test_log.csv"
    out = ev.fit_and_test(LogisticRegression(max_iter=500), X, y, splits, "m", log_path=log)
    assert out["confusion_raw"].to_numpy().sum() == 60
    assert out["confusion_norm"].sum(axis=1).round(6).eq(1).all()
    with pytest.raises(RuntimeError):
        ev.fit_and_test(LogisticRegression(max_iter=500), X, y, splits, "m", log_path=log)
    ev.fit_and_test(LogisticRegression(max_iter=500), X, y, splits, "m", allow_repeat=True, log_path=log)
    assert len(pd.read_csv(log)) == 2


def test_append_and_figures(toy, tmp_path):
    X, y, folds, splits = toy
    ev.append_result({"a": 1}, tmp_path / "r.csv"); ev.append_result({"a": 2, "b": 3}, tmp_path / "r.csv")
    assert list(pd.read_csv(tmp_path / "r.csv").columns) == ["a", "b"]
    out = ev.fit_and_test(LogisticRegression(max_iter=500), X, y, splits, "f", log_path=tmp_path / "l.csv")
    ev.confusion_figure(out["confusion_raw"], out["confusion_norm"])
    ev.roc_figure(out["y_true"], out["proba"], out["classes"])
    lc = ev.learning_curve_data(LogisticRegression(max_iter=500), X, y, folds)
    ev.learning_curve_figure(lc)
