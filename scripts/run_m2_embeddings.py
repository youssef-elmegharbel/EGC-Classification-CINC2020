"""M2 part 1: linear-method analysis + 2D embeddings of every method on the shared 2,000-record
training subsample, with clustering (k-means, GMM) and structure scores.

Outputs: results/models/m2/{embeddings.parquet, embedding_scores.csv, pca_variance.csv,
         pca_loadings.csv, lda_info.csv, svd_vs_pca.csv, sne_kl.csv, sweeps.csv}
         results/figures/20_*.png ... 29_*.png
Run:  python scripts/run_m2_embeddings.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import CLASS_ORDER, load_task, train_part           # noqa: E402
from src.loaders import DATA_DIR, FIG_DIR, MODELS_DIR                 # noqa: E402
from src.reduction import (GROUP1, embed, make_reducer, preprocess_fit,  # noqa: E402
                           structure_scores)

OUT = MODELS_DIR / "m2"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
PALETTE = dict(zip(CLASS_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999", "axes.titleweight": "bold"})


def scatter(ax, E, labels, title, kind="class"):
    labels = np.asarray(labels)
    keys = CLASS_ORDER if kind == "class" else sorted(set(labels))
    for i, k in enumerate(keys):
        m = labels == k
        c = PALETTE[k] if kind == "class" else CLUSTER_COLORS[i % 6]
        ax.scatter(E[m, 0], E[m, 1], s=5, c=c, alpha=0.6, linewidths=0, label=str(k))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def legend_fig(fig):
    h = [plt.Line2D([], [], marker="o", ls="", color=PALETTE[c], label=c) for c in CLASS_ORDER]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))


def main():
    X, y, splits = load_task()
    Xtr, ytr = train_part(X, y, splits)
    sub = json.loads((DATA_DIR / "subsample_ids.json").read_text())
    prep = preprocess_fit(Xtr)                                   # fit on the 80% only
    Ztr = prep.transform(Xtr)
    Zs = prep.transform(Xtr.loc[sub]); ys = ytr.loc[sub].to_numpy()
    print("subsample", Zs.shape, pd.Series(ys).value_counts().to_dict(), flush=True)

    # ---------------- linear analysis on the full training set
    pca = make_reducer("PCA", n_components=None).fit(Ztr)
    ev = pca.explained_variance_ratio_; cum = np.cumsum(ev)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    pd.DataFrame({"pc": np.arange(1, len(ev) + 1), "explained": ev, "cumulative": cum}).to_csv(OUT / "pca_variance.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(np.arange(1, 41), ev[:40] * 100, color="#2a78d6", width=0.8)
    ax[0].set(title="PCA scree (first 40 PCs)", xlabel="principal component", ylabel="% variance explained")
    ax[1].plot(np.arange(1, len(cum) + 1), cum * 100, color="#2a78d6", lw=2)
    ax[1].axhline(95, color="#888", ls="--", lw=1); ax[1].axvline(n95, color="#888", ls="--", lw=1)
    ax[1].annotate(f"95% at {n95} PCs", (n95, 95), xytext=(n95 + 8, 80), fontsize=9)
    ax[1].set(title="Cumulative explained variance", xlabel="number of PCs", ylabel="%")
    fig.tight_layout(); fig.savefig(FIG_DIR / "20_pca_scree.png", dpi=150); plt.close(fig)
    print("PCA: n95 =", n95, "PC1-2 =", cum[1], flush=True)

    load = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"])
    load.to_csv(OUT / "pca_loadings.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, pc in zip(ax, ["PC1", "PC2"]):
        top = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index[:15])[::-1]
        a.barh(top.index, top.values, color=np.where(top.values > 0, "#2a78d6", "#eb6834"), height=0.7)
        a.axvline(0, color="#666", lw=0.8); a.set_title(f"Top-15 loadings on {pc} ({ev[int(pc[-1])-1]*100:.1f}% var)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "21_pca_loadings.png", dpi=150); plt.close(fig)

    # SVD vs PCA: on centred data they coincide; on uncentred (raw-scaled, here min-shifted) they don't
    from sklearn.decomposition import TruncatedSVD
    svd_c = TruncatedSVD(30, random_state=42).fit(Ztr)
    Zpos = Ztr - Ztr.min(0)                                       # non-centred copy
    svd_u = TruncatedSVD(30, random_state=42).fit(Zpos)
    pd.DataFrame({"component": np.arange(1, 31), "pca_explained": ev[:30],
                  "svd_centred_explained": svd_c.explained_variance_ratio_,
                  "svd_centred_singular": svd_c.singular_values_,
                  "svd_uncentred_explained": svd_u.explained_variance_ratio_,
                  "svd_uncentred_singular": svd_u.singular_values_}).to_csv(OUT / "svd_vs_pca.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(range(1, 31), svd_c.singular_values_, "o-", ms=3, color="#2a78d6", label="SVD, standardised (centred)")
    ax[0].plot(range(1, 31), svd_u.singular_values_, "s-", ms=3, color="#eb6834", label="SVD, uncentred")
    ax[0].set(yscale="log", title="Singular values", xlabel="component"); ax[0].legend(frameon=False)
    ax[1].plot(range(1, 31), ev[:30] * 100, "o-", ms=3, color="#2a78d6", label="PCA")
    ax[1].plot(range(1, 31), svd_c.explained_variance_ratio_ * 100, "x", color="#1baf7a", label="SVD (centred)")
    ax[1].plot(range(1, 31), svd_u.explained_variance_ratio_ * 100, "s", ms=3, color="#eb6834", label="SVD (uncentred)")
    ax[1].set(title="% variance per component", xlabel="component"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "22_svd_vs_pca.png", dpi=150); plt.close(fig)

    # ---------------- 2D embeddings on the shared subsample
    methods = {  # name -> kwargs
        "PCA": {}, "LDA": {}, "SVD": {}, "KernelPCA_rbf": {"gamma": 0.005}, "KernelPCA_poly": {"degree": 3},
        "UMAP": {}, "Isomap": {}, "LLE": {}, "LLE_modified": {},
        "SNE": {}, "tSNE_p5": {}, "tSNE_p30": {}, "tSNE_p50": {}, "MDS_metric": {}, "MDS_nonmetric": {},
    }
    rows, embs = [], {}
    rest = [i for i in Xtr.index if i not in set(sub)]
    for name, kw in methods.items():
        if name == "LDA":
            # supervised: fit on training rows OUTSIDE the subsample so the plot is not in-sample
            red = make_reducer("LDA", 2); t = time.perf_counter()
            red.fit(prep.transform(Xtr.loc[rest]), ytr.loc[rest]); E = red.transform(Zs); sec = time.perf_counter() - t
        else:
            E, sec, red = embed(name, Zs, ys, 2, **kw)
        if name == "SNE":
            pd.DataFrame(red.kl_history_, columns=["iter", "kl"]).to_csv(OUT / "sne_kl.csv", index=False)
        sc = structure_scores(Zs, E, ys)
        embs[name] = (E, sc["_kmeans"], sc["_gmm"])
        row = {"method": name, "group": 1 if name in GROUP1 else 2, "n_points": len(E), "runtime_s": sec,
               **{k: v for k, v in sc.items() if not k.startswith("_")}}
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    scores = pd.DataFrame(rows); scores.to_csv(OUT / "embedding_scores.csv", index=False)
    emb_df = pd.DataFrame({"record_id": sub, "label": ys})
    for n, (E, km, gm) in embs.items():
        emb_df[f"{n}_1"], emb_df[f"{n}_2"], emb_df[f"{n}_kmeans"], emb_df[f"{n}_gmm"] = E[:, 0], E[:, 1], km, gm
    emb_df.to_parquet(OUT / "embeddings.parquet", index=False)

    # overview grid coloured by true class
    names = list(methods)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9.4))
    for a, n in zip(axes.flat, names):
        s = scores.set_index("method").loc[n]
        scatter(a, embs[n][0], ys, f"{n}\nsil={s.silhouette_true:.2f}  kNN={s.knn_acc_2d:.2f}  T={s.trustworthiness:.2f}")
    legend_fig(fig); fig.suptitle("2D embeddings of 2,000 training records, coloured by true class", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97)); fig.savefig(FIG_DIR / "23_embeddings_true_class.png", dpi=150); plt.close(fig)

    # per method: true class | k-means | GMM
    for i, n in enumerate(names):
        E, km, gm = embs[n]; s = scores.set_index("method").loc[n]
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        scatter(ax[0], E, ys, "true class")
        scatter(ax[1], E, km, f"k-means (k=6)  ARI={s.ari_kmeans:.2f}  NMI={s.nmi_kmeans:.2f}", kind="cluster")
        scatter(ax[2], E, gm, f"GMM (6 comp.)  ARI={s.ari_gmm:.2f}  NMI={s.nmi_gmm:.2f}", kind="cluster")
        ax[0].legend(*ax[0].get_legend_handles_labels(), fontsize=7, frameon=False, markerscale=2, loc="best", title="true class", title_fontsize=7)
        fig.suptitle(f"{n}  (group {1 if n in GROUP1 else 2}, {s.runtime_s:.1f} s)", fontweight="bold")
        fig.tight_layout(rect=(0, 0.05, 1, 0.95)); fig.savefig(FIG_DIR / f"24_{i:02d}_{n}_clusters.png", dpi=130); plt.close(fig)

    # ---------------- hyperparameter sweeps (plots only)
    sweeps = [("UMAP", "n_neighbors", [5, 15, 50], {"min_dist": 0.1}),
              ("UMAP", "min_dist", [0.0, 0.1, 0.5], {"n_neighbors": 15}),
              ("KernelPCA_rbf", "gamma", [0.001, 0.005, 0.02], {}),
              ("Isomap", "n_neighbors", [5, 15, 50], {}),
              ("LLE", "n_neighbors", [10, 20, 50], {})]
    srows = []
    fig, axes = plt.subplots(len(sweeps), 3, figsize=(10, 3.2 * len(sweeps)))
    for r, (n, p, vals, base) in enumerate(sweeps):
        for c, v in enumerate(vals):
            E, sec, _ = embed(n, Zs, ys, 2, **{**base, p: v})
            sc = structure_scores(Zs, E, ys)
            srows.append({"method": n, "param": p, "value": v, "runtime_s": sec,
                          **{k: x for k, x in sc.items() if not k.startswith("_")}})
            scatter(axes[r, c], E, ys, f"{n}  {p}={v}\nsil={sc['silhouette_true']:.2f} kNN={sc['knn_acc_2d']:.2f} T={sc['trustworthiness']:.2f}")
        print("sweep done", n, p, flush=True)
    legend_fig(fig); fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(FIG_DIR / "25_hyperparameter_sweeps.png", dpi=130); plt.close(fig)
    pd.DataFrame(srows).to_csv(OUT / "sweeps.csv", index=False)

    # SNE convergence
    kl = pd.read_csv(OUT / "sne_kl.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2)); ax.plot(kl.iter, kl.kl, "o-", ms=3, color="#2a78d6", lw=2)
    ax.set(title="From-scratch SNE: KL(P||Q) vs iteration", xlabel="iteration", ylabel="sum of KL divergences")
    fig.tight_layout(); fig.savefig(FIG_DIR / "26_sne_convergence.png", dpi=150); plt.close(fig)

    # bar summary of structure scores
    s = scores.set_index("method")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a, col, t in zip(ax, ["trustworthiness", "silhouette_true", "knn_acc_2d"],
                         ["Trustworthiness (k=10)", "Silhouette by true class", "2D k-NN accuracy (5-fold)"]):
        v = s[col].sort_values()
        a.barh(v.index, v.values, color=["#2a78d6" if s.loc[i, "group"] == 1 else "#eb6834" for i in v.index], height=0.7)
        a.set_title(t)
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#2a78d6", label="group 1 (has transform)"),
                          plt.Rectangle((0, 0), 1, 1, color="#eb6834", label="group 2 (plot only)")],
                 frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "27_structure_scores.png", dpi=150); plt.close(fig)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
