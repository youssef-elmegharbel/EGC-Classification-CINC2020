"""Redraw the per-method cluster figures (24_*) from results/models/m2/embeddings.parquet (no recompute)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import pandas as pd
from run_m2_embeddings import scatter, FIG_DIR, OUT
from src.reduction import GROUP1
e = pd.read_parquet(OUT / "embeddings.parquet"); sc = pd.read_csv(OUT / "embedding_scores.csv").set_index("method")
for i, n in enumerate(sc.index):
    E = e[[f"{n}_1", f"{n}_2"]].to_numpy(); s = sc.loc[n]
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    scatter(ax[0], E, e.label, "true class")
    scatter(ax[1], E, e[f"{n}_kmeans"], f"k-means (k=6)  ARI={s.ari_kmeans:.2f}  NMI={s.nmi_kmeans:.2f}", kind="cluster")
    scatter(ax[2], E, e[f"{n}_gmm"], f"GMM (6 comp.)  ARI={s.ari_gmm:.2f}  NMI={s.nmi_gmm:.2f}", kind="cluster")
    ax[0].legend(fontsize=7, frameon=False, markerscale=2, title="true class", title_fontsize=7)
    fig.suptitle(f"{n}  (group {1 if n in GROUP1 else 2}, {s.runtime_s:.1f} s); cluster colours are arbitrary IDs", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG_DIR / f"24_{i:02d}_{n}_clusters.png", dpi=130); plt.close(fig)
print("ok")
