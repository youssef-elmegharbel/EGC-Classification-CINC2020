# results/

All generated artifacts. Code reads and writes these through the constants in `src/loaders.py`
(`DATA_DIR`, `MODELS_DIR`, `FIG_DIR`), so paths are never hard-coded.

## data/  (frozen inputs shared by all members)
| File | Content |
|---|---|
| `records_meta.parquet` | header metadata for all 43,101 records (source, fs, length, age, sex, dx codes) |
| `labels.parquet` | task table: `record_id, label, source` (6,541 records) |
| `splits.json` | `train_ids`, `test_ids`, 5 `folds`; do not regenerate |
| `subsample_ids.json` | 2,000 stratified training ids for shared 2D plots |
| `features.parquet` | 6,541 x 131 hand-crafted features (index `record_id`) |
| `data_dictionary.csv` | feature name, group, description, unit |
| `extraction_diag.parquet`, `feature_failures.csv` | per-record QC and delineation failure rates |
| `selected_features.json` | (to come from M3) `{"method", "k", "features": [...]}` |

## models/  (experiment outputs)
| File | Content |
|---|---|
| `baselines.csv` | dummy baselines (stratified, most-frequent), 5-fold |
| `grid_results_<set>.csv` | every grid point: CV and train scores, flag, runtime (`round` r1 / r2) |
| `model_comparison.csv` | best configuration per model and feature set |
| `best_params.json` | best parameters per `model|set[|r2]` (round-1 / round-2 keys) |
| `oof_<set>_<model>[_r2].parquet` | pooled out-of-fold predictions and probabilities |
| `test_log.csv` | guard log: one row per model scored on the 20% test set (never repeated) |
| `test_results.csv`, `per_class_final.csv` | test metrics (all models) and per-class table of the final model |
| `learning_curves.csv`, `validation_curves.csv`, `table_cv_set_a.csv` | diagnostics and report tables |

## figures/
`NN_name.png`: 01-07 EDA, 08 model comparison, 10 final evaluation, 11 learning and validation curves.

## logs/  (git-ignored)   cache/  (git-ignored)
Run logs, and the per-record feature cache `cache/features_v<VERSION>/` (resumable; bump `VERSION` in `src/features.py`
when feature definitions change).

`HANDOFF_M2_M3.md` describes the interface for M2 and M3.
