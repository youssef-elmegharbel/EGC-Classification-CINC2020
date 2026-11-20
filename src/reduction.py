"""M2 -- dimensionality reduction helpers (linear + non-linear) and embedding-quality metrics.

Two groups (PHASE1_PLAN decision 4):

* Group 1 -- has an out-of-sample `transform`, so it can sit inside an sklearn Pipeline and feed a
  classifier (fit on training folds only): PCA, LDA, SVD, Kernel PCA, UMAP, Isomap, LLE.
* Group 2 -- optimises coordinates for a fixed point set, no mapping for new points; used only for 2D
  plots and structure scores on the shared training subsample: SNE, t-SNE, metric MDS, non-metric MDS.

`SNE` below is a from-scratch NumPy implementation of Hinton & Roweis (2002) symmetric-free SNE
(asymmetric Gaussian P_{j|i} and Q_{j|i}) so the report can show the maths directly.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
GROUP1 = ["PCA", "LDA", "SVD", "KernelPCA_rbf", "KernelPCA_poly", "UMAP", "Isomap", "LLE", "LLE_modified"]
GROUP2 = ["SNE", "tSNE_p5", "tSNE_p30", "tSNE_p50", "MDS_metric", "MDS_nonmetric"]


# ---------------------------------------------------------------- SNE from scratch
def _sq_dists(X: np.ndarray) -> np.ndarray:
    s = (X ** 2).sum(1)
    D = s[:, None] + s[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def conditional_p(X: np.ndarray, perplexity: float = 30.0, tol: float = 1e-5, max_iter: int = 60):
    """P_{j|i} = exp(-||xi-xj||^2 * beta_i) / sum_k..., beta_i = 1/(2 sigma_i^2) found by binary search
    so that the entropy H(P_i) = log(perplexity)."""
    D = _sq_dists(X)
    n = D.shape[0]
    P = np.zeros((n, n))
    beta = np.ones(n)
    logU = np.log(perplexity)
    for i in range(n):
        lo, hi = -np.inf, np.inf
        Di = np.delete(D[i], i)
        for _ in range(max_iter):
            Pi = np.exp(-(Di - Di.min()) * beta[i])          # shift for numerical stability
            sP = Pi.sum()
            H = np.log(sP) + beta[i] * ((Di - Di.min()) * Pi).sum() / sP
            diff = H - logU
            if abs(diff) < tol:
                break
            if diff > 0:                                     # entropy too high -> sharpen
                lo = beta[i]
                beta[i] = beta[i] * 2 if hi == np.inf else (beta[i] + hi) / 2
            else:
                hi = beta[i]
                beta[i] = beta[i] / 2 if lo == -np.inf else (beta[i] + lo) / 2
        P[i, np.arange(n) != i] = Pi / sP
    return P, np.sqrt(1 / (2 * beta))


class SNE(BaseEstimator):
    """Original (asymmetric) SNE. Cost C = sum_i KL(P_i || Q_i);
    gradient dC/dy_i = 2 sum_j (p_{j|i} - q_{j|i} + p_{i|j} - q_{i|j}) (y_i - y_j).
    Q uses a Gaussian with fixed sigma = 1/sqrt(2). Optimised by gradient descent with momentum and
    (as in the paper) a little Gaussian jitter in the early iterations to escape poor minima."""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=10.0, n_iter=600,
                 momentum=0.5, final_momentum=0.8, jitter=0.05, jitter_iters=100, random_state=SEED):
        self.n_components, self.perplexity, self.learning_rate, self.n_iter = n_components, perplexity, learning_rate, n_iter
        self.momentum, self.final_momentum, self.jitter, self.jitter_iters = momentum, final_momentum, jitter, jitter_iters
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, float)
        P, self.sigmas_ = conditional_p(X, self.perplexity)
        n = X.shape[0]
        Y = rng.normal(scale=1e-4, size=(n, self.n_components))
        V = np.zeros_like(Y)
        self.kl_history_ = []
        for it in range(self.n_iter):
            D = _sq_dists(Y)
            np.fill_diagonal(D, np.inf)
            Q = np.exp(-(D - D.min(1, keepdims=True)))       # row-wise shift: softmax-stable
            Q /= Q.sum(1, keepdims=True)
            M = (P - Q) + (P - Q).T                          # symmetric coefficient matrix
            grad = 2 * (M.sum(1)[:, None] * Y - M @ Y)
            mom = self.momentum if it < 250 else self.final_momentum
            V = mom * V - self.learning_rate * grad
            step = np.linalg.norm(V, axis=1, keepdims=True)  # cap per-point step (stability)
            V = np.where(step > 0.5, V * 0.5 / np.maximum(step, 1e-12), V)
            Y = Y + V
            Y -= Y.mean(0)
            if it < self.jitter_iters:
                Y += rng.normal(scale=self.jitter * (1 - it / self.jitter_iters), size=Y.shape)
            if it % 25 == 0 or it == self.n_iter - 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    kl = np.nansum(np.where(P > 0, P * np.log(P / np.maximum(Q, 1e-12)), 0.0))
                self.kl_history_.append((it, float(kl)))
        self.embedding_ = Y
        return Y


class UnsupervisedUMAP(BaseEstimator, TransformerMixin):
    """umap.UMAP that ignores y (a Pipeline passes y to fit, which would switch UMAP to supervised mode)."""

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED):
        self.n_components, self.n_neighbors, self.min_dist, self.random_state = n_components, n_neighbors, min_dist, random_state

    def fit(self, X, y=None):
        import umap
        self.model_ = umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbors,
                                min_dist=self.min_dist, random_state=self.random_state, n_jobs=1).fit(X)
        return self

    def transform(self, X):
        return self.model_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X).model_.embedding_


# ---------------------------------------------------------------- method registry
def make_reducer(name: str, n_components: int = 2, **kw):
    """Return an (unfitted) reducer. Group-1 reducers are sklearn transformers usable in a Pipeline."""
    r = SEED
    if name == "PCA":
        return PCA(n_components=n_components, random_state=r)
    if name == "LDA":
        return LinearDiscriminantAnalysis(n_components=min(n_components, 5))
    if name == "SVD":
        return TruncatedSVD(n_components=n_components, random_state=r)
    if name == "KernelPCA_rbf":
        return KernelPCA(n_components=n_components, kernel="rbf", gamma=kw.get("gamma"),
                         eigen_solver="randomized", random_state=r)
    if name == "KernelPCA_poly":
        return KernelPCA(n_components=n_components, kernel="poly", degree=kw.get("degree", 3),
                         gamma=kw.get("gamma"), eigen_solver="randomized", random_state=r)
    if name == "UMAP":
        return UnsupervisedUMAP(n_components=n_components, n_neighbors=kw.get("n_neighbors", 15),
                         min_dist=kw.get("min_dist", 0.1))
    if name == "Isomap":
        return Isomap(n_components=n_components, n_neighbors=kw.get("n_neighbors", 15))
    if name == "LLE":
        return LocallyLinearEmbedding(n_components=n_components, n_neighbors=kw.get("n_neighbors", 20),
                                      method="standard", eigen_solver="dense", reg=1e-3, random_state=r)
    if name == "LLE_modified":
        return LocallyLinearEmbedding(n_components=n_components, n_neighbors=kw.get("n_neighbors", 20),
                                      method="modified", eigen_solver="dense", random_state=r)
    if name == "SNE":
        return SNE(n_components=n_components, perplexity=kw.get("perplexity", 30))
    if name.startswith("tSNE"):
        return TSNE(n_components=n_components, perplexity=kw.get("perplexity", float(name.split("_p")[1])),
                    init="pca", random_state=r, n_jobs=2)
    if name == "MDS_metric":
        return MDS(n_components=n_components, metric_mds=True, n_init=1, init="classical_mds",
                   max_iter=300, random_state=r, normalized_stress="auto")
    if name == "MDS_nonmetric":
        return MDS(n_components=n_components, metric_mds=False, n_init=1, init="classical_mds",
                   max_iter=300, random_state=r, normalized_stress="auto")
    raise ValueError(name)


class ClipZ(BaseEstimator, TransformerMixin):
    """Winsorise standardised features to [-c, c]. Stateless (nothing learned), so no leakage.
    A few noisy records reach |z| ~ 70 and would otherwise dominate every distance/variance-based
    method (PCA, SVD, MDS, poly-KPCA, LLE collapse the other 99% of points into one blob)."""

    def __init__(self, c: float = 5.0):
        self.c = c

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), -self.c, self.c)


def with_clip(reducer, c: float = 5.0):
    """Reducer preceded by ClipZ, for use as `reducer=` in src.models.make_model (after impute+scale)."""
    return Pipeline([("clip", ClipZ(c)), ("red", reducer)])


def preprocess_fit(X_fit: pd.DataFrame, clip: float | None = 5.0):
    """Median imputation + standardisation (+ winsorise at +-clip sd), fit on the given training rows only."""
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if clip:
        steps.append(("clip", ClipZ(clip)))
    return Pipeline(steps).fit(X_fit)


def embed(name: str, Z: np.ndarray, y=None, n_components: int = 2, **kw):
    """Fit `name` on Z (already imputed+scaled) and return (embedding, seconds, fitted reducer)."""
    red = make_reducer(name, n_components, **kw)
    t = time.perf_counter()
    E = red.fit_transform(Z, y) if name == "LDA" else red.fit_transform(Z)
    return np.asarray(E), time.perf_counter() - t, red


# ---------------------------------------------------------------- structure scores
def structure_scores(Z: np.ndarray, E: np.ndarray, y, n_clusters: int = 6) -> dict:
    """trustworthiness (k=10), silhouette by true class, 2D k-NN accuracy (5-fold within the subsample),
    ARI / NMI of k-means and GMM clusters vs the true class."""
    y = np.asarray(y)
    Es = StandardScaler().fit_transform(E)
    km = KMeans(n_clusters, n_init=10, random_state=SEED).fit_predict(Es)
    gm = GaussianMixture(n_clusters, covariance_type="full", random_state=SEED).fit(Es).predict(Es)
    knn = cross_val_score(KNeighborsClassifier(10), Es, y, scoring="accuracy",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    knn_f1 = cross_val_score(KNeighborsClassifier(10), Es, y, scoring="f1_macro",
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return {"trustworthiness": trustworthiness(Z, E, n_neighbors=10),
            "silhouette_true": silhouette_score(Es, y),
            "knn_acc_2d": knn.mean(), "knn_f1_2d": knn_f1.mean(),
            "ari_kmeans": adjusted_rand_score(y, km), "nmi_kmeans": normalized_mutual_info_score(y, km),
            "ari_gmm": adjusted_rand_score(y, gm), "nmi_gmm": normalized_mutual_info_score(y, gm),
            "_kmeans": km, "_gmm": gm}


# ---------------------------------------------------------------- hand-off to M1
def best_reducer(which: str = "best"):
    """Unfitted reducer chosen by M2's 5-fold CV (results/models/m2/best_reducer.json).
    Put it in M1's pipeline: make_model(clf, params, reducer=best_reducer()). `which`: best | parsimonious."""
    import json
    from .loaders import MODELS_DIR
    h = json.loads((MODELS_DIR / "m2" / "best_reducer.json").read_text())[which]
    n = h.get("n_components", h["dim"])
    return with_clip(make_reducer(h["method"], n, **h.get("kwargs", {})))
