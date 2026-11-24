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
| `selected_features.json` | M3's set (c): RFECV (LogReg), k = 59; method, nested-CV score, feature list |

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
| `learning_curves.csv`, `validation_curves.csv`, `table_cv_set_a.csv`, `table_cv_test_sets_abc.csv` | diagnostics and report tables |
| `m2/` | M2: `best_reducer.json` (LDA, 5 dims = set b), `downstream_cv.csv`, `embeddings.parquet`, `embedding_scores*.csv`, `pca_*.csv`, `svd_vs_pca.csv`, `sne_kl.csv`, `sweeps.csv` |
| `m3/` | M3: `subsets/<method>.json` (16 selectors), `nested/<method>.csv`, `subsets_cv.csv`, `comparison.csv`, `feature_votes.csv`, `group_votes.csv`, `jaccard.csv`, `filter_scores.csv`, `related_work.csv` |
| `m3_fixedk/` | M3 at a common K = 65 for all 16 methods: `common_k.json`, `subsets/`, `subsets_cv.csv`, `summary.csv` (not used for set c) |

Feature sets: `a` all 131, `b` M2's LDA(5) reducer, `c` M3's 59 selected features; `_r2` = round 2 of the grid.

## figures/
`NN_name.png`: 01-07 EDA, 08-09 model comparison (set a; sets a/b/c), 10 final evaluation, 11 learning and validation curves, 20-29 M2 (reduction, embeddings, clustering, downstream), 30-38 M3 (selection methods, overlap, votes).

## logs/  (git-ignored)   cache/  (git-ignored)
Run logs, and the per-record feature cache `cache/features_v<VERSION>/` (resumable; bump `VERSION` in `src/features.py`
when feature definitions change).
