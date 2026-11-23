"""M3 outputs: comparison tables, figures 30-38, related-work table and results/data/selected_features.json.
Reads results/models/m3/{subsets,nested,subsets_cv.csv}. No test-set access.   Run: python scripts/make_m3_outputs.py"""
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import selection as S                                           # noqa: E402
from src.evaluate import load_task, train_part                           # noqa: E402
from src.loaders import DATA_DIR, FIG_DIR, MODELS_DIR                    # noqa: E402

OUT = MODELS_DIR / "m3"
FAM_COL = {"filter": "#4C72B0", "wrapper": "#DD8452", "embedded": "#55A868", "meta": "#C44E52"}
LABEL = {"filter_variance": "Variance thr.", "filter_corr": "Corr. pruning", "filter_anova": "ANOVA F", "filter_mi": "Mutual info",
         "filter_chi2": "Chi-square", "wrapper_sfs": "Forward (SFS)", "wrapper_sbs": "Backward (SBS)",
         "wrapper_rfecv_logreg": "RFECV (LogReg)", "wrapper_rfecv_lgbm": "RFECV (LightGBM)", "embedded_l1": "L1 (Lasso)",
         "embedded_l2": "L2 (Ridge)", "embedded_elasticnet": "Elastic Net", "embedded_rf": "RF importance",
         "meta_ga": "Genetic Algorithm", "meta_pso": "Binary PSO", "meta_gwo": "Binary GWO"}


def load_subsets():
    return {f.stem: json.loads(f.read_text()) for f in sorted((OUT / "subsets").glob("*.json"))}


def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_filter(subs, X, y, splits):
    task = S.Task(*train_part(X, y, splits)[:2], splits["folds"])
    sc = S.filter_scores(task); sc.to_csv(OUT / "filter_scores.csv")
    dd = pd.read_csv(DATA_DIR / "data_dictionary.csv").set_index("name")["group"]
    pal = dict(zip(sorted(dd.unique()), sns.color_palette("deep", dd.nunique())))
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    for a, (col, t) in zip(ax, [("anova_f", "ANOVA F-statistic"), ("mutual_info", "Mutual information"), ("chi2", "Chi-square (min-max scaled)")]):
        top = sc[col].sort_values(ascending=False).head(20)[::-1]
        a.barh(top.index, top.values, color=[pal[dd[i]] for i in top.index]); a.set_title(f"Top 20: {t}")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()], labels=list(pal), loc="lower right", title="feature group")
    fig.tight_layout(); save(fig, "30_filter_rankings.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, lab in [("filter_anova", "ANOVA F"), ("filter_mi", "Mutual info"), ("filter_chi2", "Chi-square")]:
        if k in subs:
            c = pd.DataFrame(subs[k]["meta"]["curve"]); ax.errorbar(c.k, c.cv_f1, c.cv_std, marker="o", capsize=2, label=lab)
    ax.set(xscale="log", xlabel="top-k features (after |r|<0.95 pruning)", ylabel="5-fold CV macro-F1", title="Filter rankings: CV score vs k")
    ax.legend(); fig.tight_layout(); save(fig, "31_filter_k_curves.png")
    # correlation structure that motivates pruning
    Z = pd.DataFrame(task.Z, columns=task.names); r = Z.corr().abs().values[np.triu_indices(task.p, 1)]
    return {"n_pairs_gt_95": int((r > 0.95).sum()), "n_pairs_gt_90": int((r > 0.90).sum()), "n_pairs": int(len(r))}


def fig_wrapper(subs):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, lab in [("wrapper_sfs", "Forward (SFS)"), ("wrapper_sbs", "Backward (SBS)")]:
        if k in subs:
            st = pd.DataFrame(subs[k]["meta"]["steps"]); ax[0].plot(st.k, st.cv_f1, marker=".", label=lab)
            ax[0].fill_between(st.k, st.cv_f1 - st.cv_std, st.cv_f1 + st.cv_std, alpha=0.15)
            ax[0].axvline(subs[k]["k"], ls=":", color="gray")
    ax[0].set(xlabel="number of features", ylabel="5-fold CV macro-F1", title="Sequential selection (early-stopped; dotted = 1-SE pick)"); ax[0].legend()
    for k, lab in [("wrapper_rfecv_logreg", "RFECV LogReg"), ("wrapper_rfecv_lgbm", "RFECV LightGBM")]:
        if k in subs:
            c = subs[k]["meta"]["curve"]; m = np.array(c["mean"]); n = len(m)
            sizes = np.minimum(c["n_features_total"], 3 + c["step"] * np.arange(n)); ax[1].plot(sizes, m, marker=".", label=lab)
            ax[1].fill_between(sizes, m - np.array(c["std"]), m + np.array(c["std"]), alpha=0.15)
    ax[1].set(xlabel="number of features", ylabel="5-fold CV macro-F1", title="RFECV"); ax[1].legend()
    fig.tight_layout(); save(fig, "32_wrapper_paths.png")


def fig_embedded(subs):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for key, lab in [("embedded_l1", "L1"), ("embedded_elasticnet", "Elastic Net")]:
        if key not in subs: continue
        m = subs[key]["meta"]
        grids = {"": m["path"]} if "path" in m else {f" (l1_ratio {r})": g for r, g in m["grid"].items()}
        for suf, g in grids.items():
            g = pd.DataFrame(g)
            ax[0].plot(g.C, g.n_nonzero, marker="o", label=lab + suf); ax[1].plot(g.C, g.cv_f1, marker="o", label=lab + suf)
        ax[0].axvline(m["C"], ls=":", color="gray"); ax[1].axvline(m["C"], ls=":", color="gray")
    ax[0].set(xscale="log", xlabel="C (inverse regularisation)", ylabel="features with non-zero coefficient", title="Regularisation path: sparsity"); ax[0].legend(fontsize=7)
    ax[1].set(xscale="log", xlabel="C", ylabel="5-fold CV macro-F1", title="Regularisation path: score"); ax[1].legend(fontsize=7)
    if "embedded_l2" in subs:
        w = subs["embedded_l2"]["meta"]; top = pd.Series(w["weights"][:20], index=w["ranking"][:20])[::-1]
        ax[2].barh(top.index, top.values, color=FAM_COL["embedded"]); ax[2].set_title("L2 (ridge): mean |coef|, top 20")
    fig.tight_layout(); save(fig, "33_embedded_paths.png")
    if "embedded_rf" in subs:
        w = subs["embedded_rf"]["meta"]; top = pd.Series(w["weights"][:25], index=w["ranking"][:25])[::-1]
        fig, ax = plt.subplots(figsize=(6, 6)); ax.barh(top.index, top.values, color=FAM_COL["embedded"]); ax.set_title("Random-forest impurity importance, top 25")
        fig.tight_layout(); save(fig, "34_tree_importance.png")


def fig_meta(subs):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for k, lab in [("meta_ga", "GA"), ("meta_pso", "PSO"), ("meta_gwo", "GWO")]:
        if k not in subs: continue
        h = pd.DataFrame(subs[k]["meta"]["history"])
        l, = ax[0].plot(h.gen, h.best_fitness, marker=".", label=f"{lab} best"); ax[0].plot(h.gen, h.mean_fitness, ls="--", color=l.get_color(), alpha=0.6, label=f"{lab} mean")
        ax[1].plot(h.gen, h.best_k, marker=".", label=f"{lab} best")
        ax[2].plot(h.gen, h.mean_k, marker=".", label=lab)
    ax[0].set(xlabel="generation / iteration", ylabel="fitness = CV F1 - 0.05 k/p", title="Convergence"); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="generation / iteration", ylabel="#features of best individual", title="Size of best subset"); ax[1].legend()
    ax[2].set(xlabel="generation / iteration", ylabel="mean #features in population", title="Population size drift"); ax[2].legend()
    fig.tight_layout(); save(fig, "35_metaheuristic_convergence.png")


def comparison(subs):
    rows = []
    cv = pd.read_csv(OUT / "subsets_cv.csv") if (OUT / "subsets_cv.csv").exists() else None
    for name, s in subs.items():
        r = {"method": name, "family": s["family"], "label": LABEL[name], "k": s["k"], "time_s": round(s["time_s"], 1)}
        nf = OUT / "nested" / f"{name}.csv"
        if nf.exists():
            n = pd.read_csv(nf); r.update(nested_f1=n.f1_macro.mean(), nested_std=n.f1_macro.std(ddof=1), nested_k=n.k.mean(), n_outer=len(n),
                                          nested_acc=n.accuracy.mean(), nested_time_s=n.time_s.sum(),
                                          stability_jaccard=np.mean([S.jaccard(json.loads(a), json.loads(b)) for a, b in combinations(n.features, 2)]))
        if cv is not None:
            for clf in ("logreg", "svm", "lgbm"):
                q = cv[(cv.subset == name) & (cv.clf == clf)]
                if len(q):
                    q = q.iloc[0]; r[f"cv_f1_{clf}"] = q.f1_macro_val_mean; r[f"cv_std_{clf}"] = q.f1_macro_val_std
                    r[f"train_f1_{clf}"] = q.f1_macro_train_mean; r[f"flag_{clf}"] = q.flag
        rows.append(r)
    df = pd.DataFrame(rows)
    if cv is not None:
        base = cv[cv.subset == "all_131"].set_index("clf")
        for clf in base.index:
            df.attrs[f"all131_{clf}"] = float(base.loc[clf, "f1_macro_val_mean"])
    return df


def fig_compare(df, base):
    d = df.dropna(subset=["nested_f1"]).copy() if "nested_f1" in df else df.copy()
    if "nested_f1" not in d: return
    d = d.sort_values(["family", "nested_f1"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]})
    y = np.arange(len(d))
    ax[0].barh(y, d.nested_f1, xerr=d.nested_std, color=[FAM_COL[f] for f in d.family], alpha=0.9, capsize=2)
    if base is not None:
        ax[0].axvline(base, color="k", ls="--", lw=1, label=f"all 131 features ({base:.3f})")
    ax[0].set_yticks(y, [f"{l}  (k≈{k:.0f})" for l, k in zip(d.label, d.nested_k)]); ax[0].invert_yaxis(); ax[0].set_xlim(0.5, max(0.82, d.nested_f1.max() + 0.03))
    ax[0].set(xlabel="NESTED 5-fold macro-F1 (selection repeated inside each training fold)", title="Unbiased performance of each selection procedure")
    if base is not None: ax[0].legend(loc="lower right")
    for f, c in FAM_COL.items():
        s = d[d.family == f]; ax[1].scatter(s.nested_k, s.nested_f1, s=70, color=c, label=f)
    for _, r in d.iterrows(): ax[1].annotate(r.label, (r.nested_k, r.nested_f1), fontsize=6, xytext=(3, 3), textcoords="offset points")
    if base is not None:
        ax[1].axhline(base, color="k", ls="--", lw=1)
    ax[1].set(xlabel="mean #features chosen", ylabel="nested macro-F1", title="Accuracy vs parsimony"); ax[1].legend()
    fig.tight_layout(); save(fig, "36_method_comparison.png")


def fig_overlap(subs, X):
    names = list(subs); J = pd.DataFrame([[S.jaccard(subs[a]["features"], subs[b]["features"]) for b in names] for a in names],
                                         index=[LABEL[n] for n in names], columns=[LABEL[n] for n in names])
    J.round(3).to_csv(OUT / "jaccard.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5)); sns.heatmap(J, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, annot_kws={"size": 6}, ax=ax)
    ax.set_title("Jaccard overlap between selected subsets (full training part)"); fig.tight_layout(); save(fig, "37_jaccard_overlap.png")
    dd = pd.read_csv(DATA_DIR / "data_dictionary.csv").set_index("name")["group"]
    sel = [n for n in subs if n not in ("filter_variance", "filter_corr")]       # those two are not discriminative selectors
    votes = pd.Series({f: sum(f in subs[n]["features"] for n in sel) for f in X.columns}).sort_values(ascending=False)
    vdf = pd.DataFrame({"votes": votes, "group": dd[votes.index]}); vdf.to_csv(OUT / "feature_votes.csv")
    top = vdf.head(30)[::-1]; pal = dict(zip(sorted(dd.unique()), sns.color_palette("deep", dd.nunique())))
    fig, ax = plt.subplots(figsize=(7, 8)); ax.barh(top.index, top.votes, color=[pal[g] for g in top.group])
    ax.set(xlabel=f"number of the {len(sel)} discriminative selectors that keep the feature", title="Consensus: most-selected features")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()], labels=list(pal), loc="lower right", fontsize=7)
    fig.tight_layout(); save(fig, "38_feature_votes.png")
    grp = vdf.groupby("group").votes.mean().round(2); grp.to_csv(OUT / "group_votes.csv")
    return vdf


def choose_final(df, subs):
    """Highest nested macro-F1; among methods within one standard error of it, the one with fewest features."""
    d = df.dropna(subset=["nested_f1"]); b = d.nested_f1.idxmax()
    thr = d.nested_f1[b] - d.nested_std[b] / np.sqrt(d.n_outer[b])
    cand = d[d.nested_f1 >= thr].sort_values(["k", "nested_f1"], ascending=[True, False])
    pick = cand.iloc[0]
    s = subs[pick.method]
    out = {"method": pick.method, "label": pick.label, "k": s["k"], "features": s["features"],
           "nested_cv_f1_macro": round(float(pick.nested_f1), 4), "nested_std": round(float(pick.nested_std), 4),
           "note": "selection-adjusted: the method was chosen by its nested F1, so this number is slightly optimistic; confirm on CV of the final subset / another split",
           "rule": "highest nested 5-fold macro-F1; among methods within 1 SE of the best, the fewest features",
           "fit_on": "training 80% only (never the test set)"}
    (DATA_DIR / "selected_features.json").write_text(json.dumps(out, indent=1) + "\n")
    return out, cand


def related_work(our):
    rows = [
        ("Natarajan et al., CinC 2020 (1st place)", "PhysioNet/CinC 2020 (hidden test set)", "Wide (hand-crafted, RF-selected) + deep (CNN + Transformer)", 27, "Challenge score (generalised weighted accuracy)", "0.533"),
        ("Wong et al., CinC 2020 (arXiv 2010.13712)", "PhysioNet/CinC 2020 (hidden test set)", "Hand-crafted features + gradient-boosted trees", 27, "Challenge score", "-0.080 (rank 36/41; validation 0.476)"),
        ("Chen et al., iScience 2020", "CPSC 2018 (6,877 records, hidden test)", "CNN + RNN (challenge winner)", 9, "Mean F1", "0.84 (0.837)"),
        ("Ribeiro et al., Nat. Commun. 2020", "CODE, 2.3 M ECGs (Brazil)", "Residual 1-D CNN", 6, "F1 / specificity", "F1 > 0.80, specificity > 0.99 per class"),
        ("Strodthoff et al., IEEE JBHI 2021", "PTB-XL (21,837 records)", "xresnet1d101 / Inception-1D benchmarks", 71, "Macro AUC", "0.925 (PTB-XL benchmark, all-statements task)"),
        ("Hannun et al., Nat. Med. 2019", "Single-lead ambulatory ECGs (53,877 patients)", "34-layer 1-D CNN", 12, "Mean F1 / AUC", "F1 0.837 / AUC 0.97 (cardiologists F1 0.780)"),
        ("This work (M1 final model)", "CinC 2020 subset (6,541 single-diagnosis records)", "131 hand-crafted features + " + our["model"], 6, "Macro-F1 (5-fold CV / 20% test)", f"{our['cv']:.3f} / {our['test']:.3f}"),
        ("This work (M3 selected subset)", "same", f"{our['k']} selected features ({our['method']}) + {our['model']} (CV only)", 6, "Macro-F1 (5-fold CV)", f"{our['sel_cv']:.3f}"),
    ]
    df = pd.DataFrame(rows, columns=["Work", "Data", "Method", "Classes", "Metric", "Reported score"])
    df.to_csv(OUT / "related_work.csv", index=False)
    return df


def main():
    X, y, splits = load_task()
    subs = load_subsets(); print("methods:", list(subs))
    info = fig_filter(subs, X, y, splits); json.dump(info, open(OUT / "correlation_stats.json", "w"))
    fig_wrapper(subs); fig_embedded(subs); fig_meta(subs)
    df = comparison(subs)
    base = df.attrs.get("all131_logreg")
    vdf = fig_overlap(subs, X)
    if "nested_f1" in df:
        fig_compare(df, base)
        out, cand = choose_final(df, subs); print("FINAL:", out["method"], out["k"], out["nested_cv_f1_macro"])
    df.round(4).to_csv(OUT / "comparison.csv", index=False)
    t = pd.read_csv(MODELS_DIR / "test_results.csv").sort_values("cv_f1_macro", ascending=False).iloc[0]
    sel = json.loads((DATA_DIR / "selected_features.json").read_text()) if (DATA_DIR / "selected_features.json").exists() else None
    cvt = pd.read_csv(OUT / "subsets_cv.csv") if (OUT / "subsets_cv.csv").exists() else None
    sel_cv = float("nan")
    if sel is not None and cvt is not None:
        q = cvt[(cvt.subset == sel["method"]) & (cvt.clf == t.model)]; sel_cv = float(q.f1_macro_val_mean.iloc[0]) if len(q) else float("nan")
    related_work({"model": t.model, "cv": t.cv_f1_macro, "test": t.f1_macro, "k": sel["k"] if sel else 0,
                  "method": sel["label"] if sel else "-", "sel_cv": sel_cv})
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
