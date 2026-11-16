# M1 -> M2 / M3 hand-off (frozen 2026-11-17)

## Artifacts (inputs in `results/data/`, experiment outputs in `results/models/`)
| File | Content |
|---|---|
| `data/features.parquet` | 6,541 rows x 131 hand-crafted columns, index `record_id`. NaN = failed delineation (0.3-0.6% for PR/QT), impute inside your pipeline. |
| `data/labels.parquet` | `record_id, label, source`. Classes: NSR, RBBB, AF, IAVB, PAC, SB. |
| `data/splits.json` | `train_ids` (5,232), `test_ids` (1,309), `folds` = 5 x `{train_ids, val_ids}` inside the 80%. **Do not touch the test ids.** |
| `data/subsample_ids.json` | 2,000 stratified TRAINING ids for the shared 2D plots. |
| `data/data_dictionary.csv` | name, group, description, unit for every feature. |
| `data/extraction_diag.parquet`, `data/feature_failures.csv` | per-record QC + delineation failure rates. |

## evaluate.py (frozen signatures, see module docstring)
```python
from src.evaluate import (load_task, train_part, cv_evaluate, dummy_baseline, append_result,
                          summary_columns, flag_fit, fit_and_test)
X, y, splits = load_task()  # reads results/data/                    # X: DataFrame, y: Series (index = record_id)
Xtr, ytr = train_part(X, y, splits)           # the 80% -- develop EVERYTHING here
res = cv_evaluate(pipeline, Xtr, ytr, splits["folds"], "my_method", dummy_f1=0.166)
append_result(res["row"], "results/models/my_table.csv")
```
`pipeline` must be an sklearn `Pipeline` ending in a classifier with `predict_proba`. Put imputer, scaler,
your reducer / selector INSIDE it so everything is fit on training folds only. Dummy baseline stratified
macro-F1 = 0.166 (see `results/models/baselines.csv`). Flags: overfit if train - CV macro-F1 > 0.10.
Use `src.models.ColumnSelector(columns=[...])` as a pipeline step for a feature subset.

## What M1 needs back (Fri night)
* **M2:** a fitted-in-pipeline transformer (sklearn-compatible, e.g. `PCA(0.95)`, `LDA`) plus the chosen dimension. No precomputed embeddings (leakage).
* **M3:** `results/data/selected_features.json` = `{"method": "...", "k": N, "features": [names]}`, selection fit on training folds only.

## Dataset facts worth knowing
* **SB comes only from Georgia, NSR mostly from PTB-XL, RBBB/IAVB/PAC mostly CPSC.** Class and source are confounded; report it as a limitation.
* Single-diagnosis rule = exactly one code in the header (19,555 records after merging CRBBB->RBBB, SVPB->PAC, VPB->PVC). Counts vs plan: all identical except RBBB 1,611 (+6) and PAC 627 (+89, SVPB-only records).
* Two PTB-XL records (HR09972, HR11696) have truncated `.mat` files: all-NaN signal features, kept in the table.
* `p_detected_frac` counts only P waves with baseline-corrected amplitude >= 0.025 mV (DWT alone finds a "P" on nearly every beat).

## Path constants
`from src.loaders import DATA_DIR, MODELS_DIR, FIG_DIR` (results/data, results/models, results/figures). `load_task()` already reads from `DATA_DIR`. Put your own tables in `results/models/` (or your own subfolder) and figures in `results/figures/`.
