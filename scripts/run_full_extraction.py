import pandas as pd
from src.pipeline import build_features
lab = pd.read_parquet("results/data/labels.parquet")
f = build_features(lab.record_id.tolist(), lab, "features")
print(f.shape)
