# AID427 Project 1: ECG-Net (PhysioNet/CinC 2020)

Math for Data Science / New Trends in Data Science course project. Phase 1 is a classical-ML pipeline on the
PhysioNet/CinC 2020 12-lead ECG dataset: a 6-class single-diagnosis task (NSR, RBBB, AF, IAVB, PAC, SB) with
hand-crafted features, dimensionality reduction, feature selection, and model evaluation.
Team: Youssef Elmegharbel (M1: data, EDA, models, evaluation), Nour Ashry (M2: reduction), Malak Abuelgheit (M3: feature selection).

## Layout

```
data/            SHA256SUMS.txt and dx_mapping_scored.csv tracked; data/training/ (raw 43k records) is git-ignored
src/             Shared, importable logic (notebooks stay thin)
  loaders.py       header scan, signal reading, path constants (DATA_DIR, MODELS_DIR, FIG_DIR)
  labels.py        label mapping, equivalent-code merge, 6-class task table
  splits.py        stratified 80/20 + 5 folds + shared plotting subsample
  preprocess.py    resample, band-pass, crop/pad, QC flags
  features.py      feature registry + per-record extraction (cached, resumable)
  pipeline.py      orchestration for building the feature table
  evaluate.py      SHARED by all members: CV, metrics, test-once guard, figures
  models.py        M1 model factories, grids, grid runner
  eda.py           EDA figures
  reduction.py     M2: PCA/SVD/LDA/kernel/manifold reducers, clustering scores, best_reducer() for set (b)
  selection.py     M3: filter, wrapper, embedded and metaheuristic (GA/PSO/GWO) selectors, nested CV
  report_m1.py     builds report/sections/M1_sections.docx from results/
  report_m3.py     builds report/sections/M3_sections.docx (section 6) from results/models/m3*
scripts/         M1: run_full_extraction, run_models, score_test_bc, merge_report
                 M2: run_m2_embeddings, run_m2_downstream, replot_m2_clusters
                 M3: run_m3_selection, make_m3_outputs, run_m3_fixedk
notebooks/       M1_data_eda_models, M2_reduction, M3_feature_selection, M3_selection_k65 (executed, outputs kept)
tests/           pytest, 17 tests (features, evaluate.py contract, selection)
results/         see results/README.md (data/, models/ incl. m2/ m3/, figures/, logs/, cache/)
report/          sections/ (one .docx per member) -> phase1_report.docx; paper/ (bonus, empty)
docs/            phase0/, phase1/ (plan, cover sheet); tracked although `docs/` is listed in .gitignore
```

## Setup

Use the shared env `C:\envs\ml` (Jupyter kernel "Python (ML GPU)"), or any env with `pip install -r requirements.txt`.
Phase 1 is CPU-only. Never `pip install torch` on Windows without `--index-url https://download.pytorch.org/whl/cu128`.

Download the PhysioNet/CinC 2020 training set into `data/training/` (cpsc_2018, cpsc_2018_extra, georgia, ptb, ptb-xl,
st_petersburg_incart) and verify with `sha256sum -c data/SHA256SUMS.txt` from inside `data/`.

## Reproduce (from the project root, `random_state=42` everywhere)

Set `PYTHONPATH` to the project root. M2 and M3 only read the frozen M1 tables; sets (b) and (c) of M1 need their output,
so the order below is the dependency order.

```
python -m pytest tests                              # 17 sanity checks (~45 s)
python scripts/run_full_extraction.py               # M1: features for the 6,541 task records (~2 min, cached per record)
python scripts/run_models.py a all r1               # M1: set (a) all features, 6 models x grids (round 1)
python scripts/run_models.py a all r2               #     stronger-regularisation round for overfit models
python scripts/run_m2_embeddings.py                 # M2: linear analysis + 2D embeddings (figs 20-27)
python scripts/run_m2_downstream.py                 # M2: reducers inside the pipeline -> m2/best_reducer.json (figs 28-29)
python scripts/run_m3_selection.py select           # M3: 16 selectors on the training 80% (long: wrappers, GA/PSO/GWO)
python scripts/run_m3_selection.py nested           # M3: nested-CV selection-bias check
python scripts/run_m3_selection.py classify         # M3: score every subset with 3 classifiers -> m3/subsets_cv.csv
python scripts/make_m3_outputs.py                   # M3: comparison tables, figs 30-38, data/selected_features.json
python scripts/run_m3_fixedk.py choosek|select|nested|classify|summary   # M3: all 16 methods at a common K=65 -> m3_fixedk/
python scripts/run_models.py b all r1               # M1: set (b) = M2's LDA(5); also r2
python scripts/run_models.py c all r1               # M1: set (c) = M3's 59 features; also r2
python scripts/score_test_bc.py                     # M1: score best-CV config of sets b, c on the test set once each
python -m src.report_m1                             # M1: build report/sections/M1_sections.docx
python -m src.report_m3                             # M3: build report/sections/M3_sections.docx (section 6)
python scripts/merge_report.py                      # merge all sections -> report/phase1_report.docx
```

Generated results are committed, so nothing needs re-running to inspect them. The label table, splits and feature table are
frozen in `results/data/` (do not regenerate). The notebooks run top to bottom from cached results;
`M1_data_eda_models.ipynb` has `RUN_HEAVY = True` in its setup cell to re-run the model sweeps. `fit_and_test` refuses to
score the same model name on the test set twice, so re-running `score_test_bc.py` is safe.

## Using the shared code (M2 / M3)

```python
from src.evaluate import load_task, train_part, cv_evaluate, append_result
X, y, splits = load_task()                          # reads results/data/
Xtr, ytr = train_part(X, y, splits)                 # develop everything on the 80%
res = cv_evaluate(pipeline, Xtr, ytr, splits["folds"], "my_method", dummy_f1=0.166)
append_result(res["row"], "results/models/my_table.csv")
```

Rules: imputation, scaling, reduction and selection live inside the `Pipeline` (fit on training folds only); never touch
`splits["test_ids"]` except through `evaluate.fit_and_test`, which scores the test set once per name. `pipeline` must be an sklearn `Pipeline` ending in a classifier with `predict_proba`;
use `src.models.ColumnSelector(columns=[...])` as a step for a feature subset. Full signatures are in the `src/evaluate.py` docstring.

## Key facts

* Task table: 6,541 records (NSR downsampled to 2,000); 5,232 train / 1,309 test; 5 folds inside train.
* "Exactly one diagnosis" = exactly one code in the header, after merging CRBBB to RBBB, SVPB to PAC, VPB to PVC.
* 131 hand-crafted features (see `results/data/data_dictionary.csv`).
* Feature sets: (a) all 131; (b) M2's best reducer, LDA to 5 dimensions; (c) M3's best subset, 59 features (RFECV with logistic regression).
* Dummy baseline macro-F1 (5-fold): 0.166 stratified, 0.078 most-frequent.

## Results (20% test set, each model scored once; CV scores in `results/models/`)

| Set | Best model on test | Accuracy | Macro-F1 |
|---|---|---|---|
| (a) all 131 | LightGBM | 0.832 | **0.803** |
| (b) LDA(5) | MLP | 0.817 | 0.778 |
| (c) 59 selected | SVM (RBF) | 0.810 | 0.778 |

LightGBM on set (c) reaches 0.796 macro-F1 with 45% fewer features. Overfit/underfit flags (train - CV macro-F1 > 0.10)
are logged for every grid point in `results/models/grid_results_<set>.csv`; k-NN is always flagged because distance
weighting gives a train score of 1.0. All test metrics: `results/models/test_results.csv`.

## Known limitations

No patient IDs (no patient-grouped splits, so a patient may appear in both train and test); noisy records; multi-source
shift; class/source confounding (SB only in Georgia, NSR mostly PTB-XL). Hyperparameters and feature subsets were chosen on
the same 5 folds used to report CV, so CV scores are slightly optimistic and the test set is the unbiased figure. k-NN and
sklearn's MLP have no class weights.

## Report

`report/phase1_report.docx` is the merged Phase 1 report (sections 1-8), rebuilt by `python scripts/merge_report.py` from
`report/sections/`. The script splits each file at its numbered headings and orders them by section number. Section map:
1-3, 7, 8 = M1 (`M1_sections.docx`, `python -m src.report_m1`); 4-5 = M2 (`M2_sections.docx`); 6 = M3
(`M3_sections.docx`, generated from the M3 results by `python -m src.report_m3`; replace it with the member's own Word
section if there is one). Tables are read from `results/`, so rebuild the sections, then merge, after any result changes.

## Submission status

Code, results, notebooks, tests and the merged report are complete. The bonus paper (`report/paper/`) was not started.
