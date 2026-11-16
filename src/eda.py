"""EDA figures for plan section 4 (each function saves to results/figures/ and returns the Figure)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .features import FEATURES, _detect_and_delineate
from .labels import EQUIVALENTS, TASK_CLASSES, load_dx_map
from .loaders import DATA_DIR, FIG_DIR as _FIG, MODELS_DIR, RESULTS_DIR, read_signal
from .preprocess import TARGET_FS, preprocess

FIG_DIR = _FIG
CLASS_ORDER = ["NSR", "RBBB", "AF", "IAVB", "PAC", "SB"]
PALETTE = dict(zip(CLASS_ORDER, sns.color_palette("tab10", 6)))
sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


# 1 ------------------------------------------------------------------------------------------
def class_frequencies(meta: pd.DataFrame, labels: pd.DataFrame):
    from .labels import scored_single_label

    single = scored_single_label(meta)
    abbr = load_dx_map().set_index("code")["abbr"]
    before = single["code"].map(abbr).value_counts()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    allc = pd.Series([abbr[EQUIVALENTS.get(c, c)] for cs in meta["dx_codes"]
                      for c in set(EQUIVALENTS.get(x, x) for x in cs) if c in abbr]).value_counts()
    allc.plot.bar(ax=ax[0], color="gray"); ax[0].set(title="All 43k records: occurrences of the 27 scored classes\n(multi-label)")
    b = before[before.index.isin(CLASS_ORDER)].reindex(CLASS_ORDER)
    b.plot.bar(ax=ax[1], color=[PALETTE[c] for c in CLASS_ORDER]); ax[1].set(title=f"Single-diagnosis, 6 classes (n={int(b.sum())})")
    a = labels["label"].value_counts().reindex(CLASS_ORDER)
    a.plot.bar(ax=ax[2], color=[PALETTE[c] for c in CLASS_ORDER]); ax[2].set(title=f"After NSR downsampling (n={int(a.sum())})")
    for axis, s in zip(ax[1:], (b, a)):
        for i, v in enumerate(s):
            axis.text(i, v, int(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return _save(fig, "01_class_frequencies")


# 2 ------------------------------------------------------------------------------------------
def cooccurrence(meta: pd.DataFrame):
    dm = load_dx_map()
    codes = [c for c in dm["code"] if c not in EQUIVALENTS]
    abbr = dm.set_index("code")["abbr"]
    idx = {c: i for i, c in enumerate(codes)}
    M = np.zeros((len(meta), len(codes)), dtype=np.int32)
    for r, cs in enumerate(meta["dx_codes"]):
        for c in cs:
            c = EQUIVALENTS.get(c, c)
            if c in idx:
                M[r, idx[c]] = 1
    co = M.T @ M
    names = [abbr[c] for c in codes]
    co = pd.DataFrame(co, index=names, columns=names)
    cond = co.div(np.diag(co), axis=0)  # P(col | row)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cond, cmap="viridis", ax=ax, cbar_kws={"label": "P(column class | row class)"})
    ax.set(title="Label co-occurrence on all 43,101 records (27 scored classes)\n"
                 "Most diagnoses co-occur with others, which motivates the single-label task")
    fig.tight_layout()
    return _save(fig, "02_cooccurrence")


# 3 ------------------------------------------------------------------------------------------
def source_distributions(meta: pd.DataFrame, labels: pd.DataFrame):
    m = meta[meta["sex"] != "Unknown"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    sns.violinplot(data=meta.dropna(subset=["age"]), x="source", y="age", ax=ax[0, 0], cut=0, inner="quartile")
    ax[0, 0].set(title="Age by source"); ax[0, 0].tick_params(axis="x", rotation=30)
    sx = m.groupby("source")["sex"].value_counts(normalize=True).unstack()
    sx.plot.bar(stacked=True, ax=ax[0, 1]); ax[0, 1].set(title="Sex by source", ylabel="fraction")
    fs = meta.groupby("source")["fs"].agg(lambda s: ", ".join(f"{int(v)}" for v in sorted(s.unique())))
    ax[0, 2].axis("off"); ax[0, 2].table(cellText=[[k, v] for k, v in fs.items()], colLabels=["source", "fs (Hz)"],
                                         loc="center").scale(1, 2)
    ax[0, 2].set_title("Sampling rate by source")
    d = meta.assign(dur=meta["duration_s"].clip(upper=40))
    sns.boxplot(data=d, x="source", y="dur", ax=ax[1, 0]); ax[1, 0].set(title="Record length (s, clipped at 40)", ylabel="s")
    ax[1, 0].tick_params(axis="x", rotation=30)
    mix = pd.crosstab(labels["source"], labels["label"], normalize="index")[CLASS_ORDER]
    mix.plot.bar(stacked=True, ax=ax[1, 1], color=[PALETTE[c] for c in CLASS_ORDER])
    ax[1, 1].set(title="Class mix within each source (6-class task)", ylabel="fraction"); ax[1, 1].tick_params(axis="x", rotation=30)
    mix2 = pd.crosstab(labels["label"], labels["source"], normalize="index").loc[CLASS_ORDER]
    mix2.plot.bar(stacked=True, ax=ax[1, 2]); ax[1, 2].set(title="Source mix within each class (SB = Georgia only!)", ylabel="fraction")
    ax[1, 2].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _save(fig, "03_source_distributions")


# 4 ------------------------------------------------------------------------------------------
def missingness(X: pd.DataFrame, labels: pd.DataFrame, meta: pd.DataFrame):
    grp = {f.name: f.group for f in FEATURES}
    na = X.isna()
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    top = na.mean().sort_values(ascending=False).head(10)
    top.iloc[::-1].plot.barh(ax=ax[0]); ax[0].set(title="Top-10 feature NaN rates (all records)", xlabel="fraction NaN")
    key = ["pr_interval", "qrs_duration", "qt_interval", "p_duration", "age"]
    by_c = na[key].join(labels.set_index("record_id")["label"]).groupby("label")[key].mean().reindex(CLASS_ORDER)
    sns.heatmap(by_c * 100, annot=True, fmt=".2f", cmap="Reds", ax=ax[1], cbar_kws={"label": "% NaN"})
    ax[1].set(title="Delineation / age NaN rate by class (%)")
    by_s = na[key].join(labels.set_index("record_id")["source"]).groupby("source")[key].mean()
    sns.heatmap(by_s * 100, annot=True, fmt=".2f", cmap="Reds", ax=ax[2], cbar_kws={"label": "% NaN"})
    ax[2].set(title="... by source (%)")
    fig.tight_layout()
    return _save(fig, "04_missingness")


# 5 ------------------------------------------------------------------------------------------
def _plot_before_after(ax, path, lead=1, seconds=4, title=""):
    sig, fs = read_signal(path)
    pp = preprocess(sig, fs)
    n = int(seconds * fs)
    t_raw = np.arange(min(n, len(sig))) / fs
    ax.plot(t_raw, sig[:n, lead], color="0.6", lw=0.8, label="raw")
    n2 = seconds * TARGET_FS
    ax.plot(np.arange(n2) / TARGET_FS, pp.signal_mv[:n2, lead], color="C0", lw=0.9, label="filtered 0.5-40 Hz")
    ax.set(title=title, xlabel="s", ylabel="mV")
    ax.legend(loc="upper right", fontsize=7)


def noise_examples(meta: pd.DataFrame, diag: pd.DataFrame, labels: pd.DataFrame):
    """Clean, baseline-wander, high-frequency noise, flat-lead examples (picked from qc_* columns)."""
    d = diag.join(meta.set_index("record_id")[["path", "source"]])
    picks = {
        "clean": d.sort_values(["qc_max_hf_ratio", "qc_max_baseline_ratio"]).index[len(d) // 4],
        "baseline wander": d["qc_max_baseline_ratio"].idxmax(),
        "high-frequency noise": d["qc_max_hf_ratio"].idxmax(),
    }
    flat = d[d["qc_n_flat_leads"] > 0]
    if len(flat):
        picks["flat lead(s)"] = flat.index[0]
    fig, ax = plt.subplots(len(picks), 1, figsize=(11, 2.8 * len(picks)))
    lab = labels.set_index("record_id")["label"]
    from .loaders import LEAD_NAMES
    for a, (k, rid) in zip(np.atleast_1d(ax), picks.items()):
        li = int(d.loc[rid, "qc_flat_leads"][0]) if k.startswith("flat") else 1
        _plot_before_after(a, d.loc[rid, "path"], lead=li,
                           title=f"{k}: {rid} ({d.loc[rid, 'source']}, {lab.get(rid, '?')}), lead {LEAD_NAMES[li]}")
    fig.tight_layout()
    return _save(fig, "05_noise_examples")


def qc_rates(diag: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    d = diag.join(labels.set_index("record_id")[["label", "source"]])
    d["any_flat"] = d["qc_n_flat_leads"] > 0
    d["any_noisy"] = d["qc_n_noisy_leads"] > 0
    cols = ["any_flat", "any_noisy", "qc_padded"]
    out = pd.concat([d[cols].mean().rename("ALL").to_frame().T, d.groupby("label")[cols].mean(),
                     d.groupby("source")[cols].mean()])
    return (out * 100).round(2).rename(columns={"any_flat": "% with flat lead", "any_noisy": "% with noisy lead (>25% HF power)",
                                                 "qc_padded": "% shorter than 10 s (zero-padded)"})


# 6 ------------------------------------------------------------------------------------------
def delineation_overlay(meta: pd.DataFrame, labels: pd.DataFrame, per_class: int = 1, seconds: float = 3.5):
    m = meta.set_index("record_id")
    fig, ax = plt.subplots(len(CLASS_ORDER), 1, figsize=(11, 2.5 * len(CLASS_ORDER)))
    marks = {"ECG_P_Peaks": ("P", "tab:green"), "ECG_R_Peaks": ("R", "tab:red"), "ECG_T_Peaks": ("T", "tab:purple"),
             "ECG_R_Onsets": ("QRS on", "k"), "ECG_R_Offsets": ("QRS off", "k")}
    for a, cls in zip(ax, CLASS_ORDER):
        rid = labels[labels["label"] == cls].sample(1, random_state=7)["record_id"].iloc[0]
        sig, fs = read_signal(m.loc[rid, "path"])
        pp = preprocess(sig, fs)
        li, rp, w = _detect_and_delineate(pp.signal_mv, pp.valid_len)
        li = 1 if li is None else li
        n = int(seconds * TARGET_FS)
        x = pp.signal_mv[:n, li]
        a.plot(np.arange(n) / TARGET_FS, x, lw=0.9)
        if rp is not None:
            pts = {"ECG_R_Peaks": rp, **(w or {})}
            for k, (lab, col) in marks.items():
                idx = np.asarray(pts.get(k, []), float)
                idx = idx[~np.isnan(idx)].astype(int)
                idx = idx[idx < n]
                a.scatter(idx / TARGET_FS, x[idx], s=14, color=col, zorder=3, label=lab if k != "ECG_R_Offsets" else None)
        a.set(title=f"{cls}: {rid}, lead {['I','II','III','aVR','aVL','aVF','V1'][li] if li < 7 else li}", ylabel="mV")
        if cls == CLASS_ORDER[0]:
            a.legend(ncol=5, fontsize=7, loc="upper right")
    fig.tight_layout()
    return _save(fig, "06_delineation_overlay")


def before_after_preprocessing(meta: pd.DataFrame, labels: pd.DataFrame):
    m = meta.set_index("record_id")
    fig, ax = plt.subplots(3, 1, figsize=(11, 8))
    for a, cls in zip(ax, ["NSR", "AF", "SB"]):
        rid = labels[labels["label"] == cls].sample(1, random_state=3)["record_id"].iloc[0]
        _plot_before_after(a, m.loc[rid, "path"], title=f"{cls}: {rid} (raw vs filtered, lead II)")
    fig.tight_layout()
    return _save(fig, "06b_before_after_preprocessing")


# 7 ------------------------------------------------------------------------------------------
KEY_FEATURES = [("hr_mean", "SB: HR < 60"), ("pr_interval", "IAVB: PR > 200 ms"),
                ("rr_irregularity", "AF: irregular RR"), ("p_detected_frac", "AF: no P waves"),
                ("rr_short_frac", "PAC: premature beats"), ("qrs_duration", "RBBB: wide QRS")]


def key_feature_distributions(X: pd.DataFrame, labels: pd.DataFrame):
    d = X.join(labels.set_index("record_id")["label"])
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for a, (f, hyp) in zip(ax.ravel(), KEY_FEATURES):
        sns.boxplot(data=d, x="label", y=f, order=CLASS_ORDER, palette=PALETTE, ax=a, showfliers=False, hue="label", legend=False)
        a.set(title=f"{f}   ({hyp})", xlabel="")
        if f == "hr_mean": a.axhline(60, color="k", ls="--", lw=0.8)
        if f == "pr_interval": a.axhline(200, color="k", ls="--", lw=0.8)
    fig.tight_layout()
    return _save(fig, "07b_key_feature_distributions")


def correlation_heatmap(X: pd.DataFrame, thresh: float = 0.9):
    c = X.corr(method="spearman")
    order = sns.clustermap(c, cmap="coolwarm", vmin=-1, vmax=1, figsize=(1, 1)).dendrogram_row.reordered_ind
    plt.close("all")
    c = c.iloc[order, order]
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(c, cmap="coolwarm", vmin=-1, vmax=1, xticklabels=False, yticklabels=False, ax=ax,
                cbar_kws={"label": "Spearman r"})
    n_hi = int(((c.abs() > thresh).sum().sum() - len(c)) / 2)
    ax.set(title=f"Feature correlation (Spearman, clustered): {n_hi} pairs with |r| > {thresh}")
    fig.tight_layout()
    return _save(fig, "07_feature_correlation")


def baseline_table(path=MODELS_DIR / "baselines.csv") -> pd.DataFrame:
    b = pd.read_csv(path)
    return b[["name", "accuracy_val_mean", "accuracy_val_std", "f1_macro_val_mean", "f1_macro_val_std"]].round(3)
