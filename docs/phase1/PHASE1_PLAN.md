# ECG-Net — Phase 1 Plan (AID427)

**Project:** ECG-Net: Multi-label Cardiac Abnormality Detection from 12-Lead ECGs
**Dataset:** PhysioNet/CinC 2020 (43,101 records, all 6 sources downloaded under `data/training/`)
**Team:** Youssef Elmegharbel (M1) · Nour Ashry (M2) · Malak Abuelgheit (M3)
**Deadline:** Monday 23 Nov 2026 (7 days from Mon 16 Nov)

---

## 1. What Phase 1 is graded on (20 marks + 7 bonus)

| # | Rubric item | Marks | Owner |
|---|---|---|---|
| 1 | Problem introduction | 0 (required) | M1 |
| 2 | Dataset description (noisy, difficult dataset) | 0 (required) | M1 |
| 3 | Data analysis, preprocessing, visualization before/after reduction and clustering | 2 | M1 (+ M2 for the "after" plots) |
| 4 | Linear reduction: PCA, LDA, SVD | 1 | M2 |
| 5 | Non-linear reduction: Kernel PCA, SNE, t-SNE, UMAP, metric + non-metric MDS, Isomap, LLE | 6 | M2 |
| 6 | Feature selection: filter methods | 1 | M3 |
| 7 | Feature selection: wrapper methods (forward, backward, RFE, ...) | 2 | M3 |
| 8 | Feature selection: embedded methods (L1, L2, Elastic Net) | 2 | M3 |
| 9 | Feature selection: GA and other metaheuristics | 2 | M3 |
| 10 | Model implementations | 1 | M1 |
| 11 | Evaluation: 80/20 split + all metrics, K-fold CV + mean accuracy, confusion matrix (accuracy, error rate, precision, recall, F1, ROC-AUC), over/underfitting interpretation | 3 | M1 |
| 11b | Related-work comparison table (part of item 11) | — | M3 |
| B | Bonus: research paper, 15+ pages, IEEE format | +7 | All (M3 leads) — **stretch goal only** |

**Load per person:** M1 = 6 marks + the shared data pipeline (critical path) · M2 = 7 · M3 = 7 + related-work table + paper lead.

---

## 2. Key design decisions (agree on these on day 1)

1. **Feature table — DECIDED (16 Nov): hand-crafted features only (no raw-signal features).** The cover sheet is classical ML (reduction, selection, confusion matrices), and our 6 classes are defined by measurable properties (SB: HR < 60; IAVB: PR > 200 ms; AF: irregular RR + no P waves; PAC: premature beats / RR outliers; RBBB: wide QRS + V1 shape), so selection methods should rediscover clinically meaningful features. M1 produces `features.parquet`, one row per record, ~130 columns:
   - **Rhythm, lead II (~10):** HR, RR mean/std/min/max, RMSSD, pNN50, RR-irregularity score.
   - **Intervals, global (~6):** PR, QRS duration, QT, QTc, P-wave duration, fraction of beats with a detected P wave.
   - **Per-lead, all 12 leads (~96):** R/S/T/P amplitudes, std, skewness, kurtosis, energy (cross-lead redundancy is intentional: gives correlation pruning and PCA something to show).
   - **Spectral + wavelet, lead II (~15):** band powers, wavelet level energies.
   - **Patient (2):** age, sex.
   - If neurokit2 delineation fails on a noisy record, store **NaN** (don't drop the record); impute inside the training pipeline. Report the failure rate as evidence of noise.
   - Raw-signal / median-beat features were rejected: wrapper selection over 720+ columns won't finish in a week, and they're not interpretable. Deep models (1D-CNN) are out of scope for Phase 1.
2. **Target definition — DECIDED (16 Nov): Option C (hybrid).** Phase 0 promised multi-label (27 scored classes), but the cover sheet's items (LDA, a single confusion matrix, coloured 2D embeddings, filter tests, GA fitness) all assume one class per record. So:
   - **Main task (all graded Phase 1 work):** 6-class single-label classification on records that carry **exactly one** diagnosis (19,466 of the 43,101 records). Classes, all from the 27 scored ones:

     | SNOMED | Class | Single-label records |
     |---|---|---|
     | 426783006 | Normal sinus rhythm (NSR) | 9,182 → **downsample to ~2,000** |
     | 59118001 | Right bundle branch block (RBBB) | 1,605 |
     | 164889003 | Atrial fibrillation (AF) | 1,093 |
     | 270492004 | 1st-degree AV block (IAVB) | 754 |
     | 284470004 | Premature atrial contraction (PAC) | 538 |
     | 426177001 | Sinus bradycardia (SB) | 456 |

     ≈ 6.4k records after downsampling NSR (random, `random_state=42`). Without it NSR is 67% of the data and accuracy is meaningless. Keep class weights and report **macro-F1** alongside accuracy for the remaining imbalance.
   - Unscored classes (ST depression, nonspecific ST-T, MI, LVH, ...) are left out to stay consistent with Phase 0's 27 scored classes.
   - **Report wording:** "Phase 1 focuses on the single-diagnosis subset; the full multi-label problem is evaluated as an extension."
   - **Extension (optional, Saturday only if on schedule):** full 27-class multi-label one-vs-rest on all records, with per-class metrics and the official challenge score. Also reusable for the bonus paper.
   - M1 to verify the codes and names above against the challenge's `dx_mapping_scored.csv` and merge equivalent codes (e.g. 713427006 CRBBB → RBBB, 63593006 SVPB → PAC) before counting.
3. **Splits — DECIDED (16 Nov).** M1 writes one `splits.json` that everyone uses:
   - **Order:** (1) build the 6-class subset and downsample NSR, (2) stratified 80/20 train/test split, (3) 5 stratified folds on the 80% only. `random_state=42` everywhere.
   - **K-fold lives inside the training 80%.** All tuning, feature selection and method comparison use the 5 folds. The 20% test set is touched **once**, at the end, for the final confusion matrices and metrics. K = 5 (10 would double the cost of wrapper selection and the GA).
   - **Stratify by class only;** print the source-database mix per split to show it's similar. (Class × source strata are too small for some classes.)
   - **No leakage:** imputation, scaling, reduction and selection are always fit on training folds only, inside a `sklearn.Pipeline`. Final models are refit on the full 80% and evaluated once on the 20%.
   - **Known limitation (report):** headers have no patient IDs, so recordings can't be grouped by patient. Most sources have one recording per patient (PTB is the exception, with few records in our classes).
   - **Held-out-database test (from Phase 0) is not in Phase 1;** optional extra for the bonus paper.
4. **Reduction methods — DECIDED (16 Nov).**
   - **Two groups.** Group 1 can map new data (PCA, LDA, SVD, Kernel PCA, UMAP, Isomap, LLE): used for 2D plots **and** as classifier input (fit on train, apply to test). Group 2 cannot (SNE, t-SNE, metric MDS, non-metric MDS): 2D plots and structure scores only. The report explains why (they optimize coordinates for a fixed point set and have no out-of-sample mapping).
   - **Group 1 classifier tests use the full training set** (~5.1k records is small enough for Kernel PCA, Isomap, LLE), with the shared 5 folds.
   - **One shared plotting subsample:** 2,000 records, stratified, drawn from the **training set only**, saved as `subsample_ids.json`. All 11 methods plot the same points. If SNE or non-metric MDS is too slow, drop to 1,000 for that method only and note it in the table.
   - **Comparison table (one row per method):** trustworthiness, silhouette by true class, 2D k-NN accuracy (5-fold within the subsample), runtime; ARI/NMI of k-means (k = 6) and GMM clusters vs true class; group 1 also gets downstream macro-F1 with 2–3 classifiers.
   - **Clustering plots:** every 2D embedding shown twice, coloured by true class and by cluster.
   - **Dimensions:** plots in 2D (optional 3D for PCA/LDA). For classification: LDA = 5 (classes − 1); PCA/SVD keep 95% variance; Kernel PCA, UMAP, Isomap, LLE try 2/10/20 and pick by CV.
   - Imputation + standardization before any reduction, fit on train only.
5. **Environment.** Shared env `C:\envs\ml` (kernel "Python (ML GPU)"). Extra packages: `wfdb neurokit2 umap-learn mlxtend deap pyarrow`. Install into the shared env only.
6. **Submission + teamwork — DECIDED (16 Nov).**
   - **Deliverables:** the code (GitHub repo with notebooks) + a written report. No presentation required so far.
   - **GitHub repo:** to be initialized later (by M1). Until then, keep everything in the folder layout below so it can be committed as-is.
   - **One notebook per person;** shared code lives in `src/` so nobody copies helpers between notebooks.
   - **Report:** a local Word (.docx) file. A single .docx can't be merged when three people edit it, so each member writes their own section file in `report/sections/` and M1 merges them into `phase1_report.docx` on Sunday.
   ```
   src/        loaders.py, preprocess.py, features.py, evaluate.py (shared helpers)
   notebooks/  M1_data_eda_models.ipynb · M2_reduction.ipynb · M3_feature_selection.ipynb
   outputs/    features.parquet, splits.json, subsample_ids.json, figures/, results/*.csv
   report/     sections/ (one .docx per member) → phase1_report.docx; paper/ (IEEE template, bonus only)
   ```
7. **Evaluation rules — DECIDED (16 Nov).** Every results table, from every member, uses `src/evaluate.py` so numbers are comparable:
   - **Scores:** report **both accuracy and macro-F1** everywhere. Rank methods/models by macro-F1 (accuracy is misleading with the remaining class imbalance); accuracy is the secondary column.
   - **Precision / recall / F1:** report **both macro and weighted** averages, plus the per-class table.
   - **ROC-AUC:** one-vs-rest ROC curve per class, with **both macro and weighted** AUC.
   - **Confusion matrix:** **both raw counts and row-normalized** (recall per class), side by side. Error rate = 1 − accuracy.
   - **K-fold:** mean ± std over the 5 folds for every score.
8. **Overfitting: detect it and stop it early — DECIDED (16 Nov), high priority.**
   - **Detect:** every model/method logs train score and 5-fold CV score. Flag **overfitting** when train − CV macro-F1 > 10 points; flag **underfitting** when both train and CV macro-F1 are low (close to a majority/stratified dummy baseline, which `evaluate.py` always reports). Learning curves (score vs training size) and validation curves (score vs main hyperparameter) for the final models.
   - **Stop early, in training:** gradient boosting uses early stopping on a validation split carved from the training fold (`early_stopping_rounds`); MLP uses `early_stopping=True` with a validation fraction; trees are capped (`max_depth`, `min_samples_leaf`); linear/SVM models are tuned on regularization strength `C`.
   - **Stop early, in feature selection (for M3 to confirm):** forward/backward selection stops when CV score stops improving (not a fixed feature count); the GA stops after N generations without improvement and penalizes feature count, so it doesn't overfit the CV folds.
   - **The test set never drives any of these choices** (decision 3).
9. **Still open (owners decide later):**
   - Feature-selection protocol (estimator inside wrappers/GA, GA fitness penalty and budget, how to pick the final subset) — **M1 + M3**.
   - Final model list and tuning budget per model — **M1**, when implementation starts.

---

## 3. Task breakdown per member

### M1 — Youssef: data pipeline, EDA, models and evaluation

**Mon–Tue (blocking — others depend on it)**
- [ ] Loader: read all `.hea/.mat` (wfdb/scipy), parse age, sex, SNOMED Dx, source DB, sampling rate, length.
- [ ] Label mapping to the 27 scored classes (use the challenge's `dx_mapping_scored.csv`, merge equivalent codes: CRBBB/RBBB, PAC/SVPB, PVC/VPB).
- [ ] Signal preprocessing: resample to 500 Hz, 0.5–40 Hz band-pass, baseline-wander removal, crop/pad to 10 s, per-lead z-score; flag flat/noisy leads.
- [ ] Feature extraction (neurokit2 + numpy/scipy) per lead or on lead II + summary across leads:
  RR/HRV (mean, std, RMSSD, pNN50), heart rate, QRS width, PR/QT intervals, P/R/T amplitudes, statistical (mean, std, skew, kurtosis, energy), spectral band powers, wavelet energies, + age, sex.
- [ ] Deliver `features.parquet`, `labels.parquet`, `splits.json`, `subsample_ids.json` + a short data dictionary. Extract features only for the ~6.4k records of the 6-class task first (multi-label extension features can run in the background later). **Target: v0 on CPSC 2018 only by Tue night; full table frozen Wed 18 Nov noon.** Keep the feature list modest (~100–150) — extraction time on 43k records is the biggest schedule risk.

**Wed**
- [ ] EDA (2 marks): class frequencies, co-occurrence heatmap, per-source distributions (age, sex, sampling rate, length), missing values, noise examples (raw vs filtered ECG plots), feature correlation heatmap.
- [ ] Problem introduction + dataset description sections (reuse Phase 0 text).
- [ ] `src/evaluate.py` shared helper, following section 2 items 7–8: given a pipeline → confusion matrix (raw + row-normalized), accuracy, error rate, precision/recall/F1 (macro + weighted + per class), one-vs-rest ROC curves + AUC (macro + weighted), train vs CV vs test scores, 5-fold mean ± std, dummy baseline, and an over/underfitting flag. M2 and M3 call it too, so all results look the same.

**Thu–Sat**
- [ ] Models (1 mark): Logistic Regression, k-NN, SVM (RBF), Random Forest, Gradient Boosting (XGBoost/LightGBM), simple MLP; class weights for imbalance.
- [ ] Evaluation (3 marks): run every model on (a) all features, (b) best reduced set from M2, (c) best selected set from M3. 80/20 test metrics + 5-fold CV accuracy.
- [ ] Over/underfitting analysis: train vs test vs CV gap, learning curves for 2 models.
- [ ] Add our final numbers into M3's related-work table.

### M2 — Nour: dimensionality reduction (linear + non-linear) and clustering plots

**Mon–Tue (while waiting for features)**
- [ ] Write each method's math summary (objective, what it preserves, complexity, key hyperparameters) — this goes straight into the report and paper.
- [ ] Prototype all code on a public toy dataset (sklearn `load_digits`) so it only needs the data swapped in.
- [ ] Implement **SNE from scratch** in NumPy (sklearn only has t-SNE): Gaussian P with perplexity binary search, Gaussian Q, gradient descent. Good "math" content for the report.

**Wed–Fri (hand-off to M1 by Fri night)**
- [ ] Linear (1 mark): PCA (scree + cumulative variance, 2D/3D plots, loadings), LDA (≤ C−1 components, class separation), Truncated SVD (singular values, compare to PCA).
- [ ] Non-linear (6 marks): Kernel PCA (RBF, poly; tune γ), SNE, t-SNE (perplexity 5/30/50), UMAP (n_neighbors, min_dist), metric MDS, non-metric MDS, Isomap (n_neighbors), LLE (standard + modified).
- [ ] For every method: 2D embedding coloured by true class **and** by k-means / GMM clusters (gives the "visualization after reduction and clustering" marks), runtime, and a quantitative score: trustworthiness, silhouette, and k-NN accuracy in the embedded space.
- [ ] Downstream test (group 1 only: PCA, LDA, SVD, KPCA, UMAP, Isomap, LLE): feed components into 2–3 classifiers via M1's `evaluate.py`. Hand M1 the best reduced representation.
- [ ] Summary comparison table: method × (silhouette, trustworthiness, k-NN acc, downstream F1, runtime).

### M3 — Malak: feature selection + paper lead

**Mon–Tue (while waiting for features)**
- [ ] Write the theory for each selection family (filter vs wrapper vs embedded vs metaheuristic, pros/cons, cost).
- [ ] Prototype on a toy dataset (`load_breast_cancer`), including the GA.
- [ ] Related-work comparison table (5–8 papers on CPSC 2018 / PhysioNet 2020; e.g. challenge winners, Ribeiro et al. 2020, Strodthoff et al. 2021 PTB-XL benchmark): dataset, method, #classes, metric, score. M1 adds our row on Sat.
- [ ] Set up the IEEE paper template (IEEE conference LaTeX on Overleaf or Word) and outline (only if the bonus is attempted).

**Wed–Fri (hand-off to M1 by Fri night)**
- [ ] Filter (1 mark): variance threshold, correlation pruning, ANOVA F-test, mutual information, chi² (on min-max scaled data); ranked feature bar plots.
- [ ] Wrapper (2 marks): Sequential Forward Selection, Sequential Backward Selection (mlxtend), RFE / RFECV with a fast estimator (Logistic Regression or LightGBM); performance vs number of features curve.
- [ ] Embedded (2 marks): L1 (Lasso-logistic), L2 (Ridge-logistic, coefficient magnitudes), Elastic Net (grid over l1_ratio, C); plus tree importance as a bonus view. Regularization path plot.
- [ ] Metaheuristics (2 marks): **Genetic Algorithm** feature selection (DEAP or own code: binary chromosome, fitness = CV F1 − λ·(#features), tournament selection, crossover, mutation) + one more for comparison: **PSO** (binary) or **Grey Wolf Optimizer**. Convergence plots (best/mean fitness per generation).
- [ ] Comparison table: method × (#features kept, CV F1, test F1, runtime) + overlap/Jaccard between the selected subsets. Hand M1 the best subset.

---

## 4. Timeline (Mon 16 → Mon 23 Nov)

| Day | M1 Youssef | M2 Nour | M3 Malak | Sync point |
|---|---|---|---|---|
| **Mon 16** | Repo + env setup, loader, label mapping | Theory notes, toy prototypes (digits) | Theory notes, toy prototypes (breast cancer), start related-work search | Agree on the decisions in section 2 tonight |
| **Tue 17** | Preprocessing + feature extraction; **v0 table on CPSC 2018 by night** | SNE from scratch, all methods running on toy data | GA + PSO/GWO on toy data, related-work table | M2/M3 switch to v0 table |
| **Wed 18** | **Full table + splits frozen at noon**; EDA; `evaluate.py` | PCA, LDA, SVD on real data; Kernel PCA, t-SNE, UMAP | Filter + embedded on real data | Everyone runs `evaluate.py` once |
| **Thu 19** | All 6 models on all features | MDS (metric + non-metric), Isomap, LLE, SNE on subsample | Wrapper (SFS, SBS, RFECV) | — |
| **Fri 20** | Evaluation on all features; write sections 1–3 | Clustering plots, comparison table, downstream test | GA + PSO/GWO on real data, comparison table | **Night: best reduced set (M2) + best subset (M3) to M1** |
| **Sat 21** | Final evaluation on reduced/selected sets, over/underfitting, add our row to related work | Write sections 4–5 | Write section 6 + related-work text | Draft report complete by night |
| **Sun 22** | Assemble report, cover sheet | Proofread, fix figures | Proofread; paper only if time | **Final review together** |
| **Mon 23** | **Submit** | | | |

Rules: each member writes their own report section (with figures) as they go; M1 assembles. Freeze the feature table on Wednesday — no changes after that or everyone's results go stale. If something is running late on Friday, cut depth (fewer hyperparameter settings), not methods: every listed method carries marks.

---

## 5. Report structure (follows the cover sheet order)

1. Problem introduction — M1
2. Dataset description and difficulty (noise, imbalance, multi-source shift) — M1
3. Data analysis and preprocessing (before/after plots) — M1
4. Linear feature reduction — M2
5. Non-linear feature reduction + clustering visualizations — M2
6. Feature selection: filter, wrapper, embedded, metaheuristics — M3
7. Model implementations — M1
8. Evaluation: 80/20 metrics, confusion matrices, ROC-AUC, K-fold CV, over/underfitting, comparisons — M1
9. Comparison with related work (table) — M3 (M1 adds our results)
10. Conclusion — all

## 6. Bonus paper (+7)

Angle that reuses Phase 1 work directly: **"Comparative analysis of dimensionality reduction and metaheuristic feature selection for 12-lead ECG arrhythmia classification"**, with the novel part being a **GA (or hybrid GA-PSO) that jointly selects features and tunes classifier hyperparameters**, optionally multi-objective (maximize F1, minimize #features → Pareto front with NSGA-II in DEAP).
- M3: method + experiments sections, overall structure.
- M2: related work on reduction, reduction results and figures.
- M1: introduction, dataset, preprocessing, evaluation, related-work comparison.
- Target 15+ pages in IEEE format. **With a one-week deadline this is a stretch goal:** do it only if the 20-mark report is done by Saturday night; otherwise check with the TA whether the paper can come in later.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Feature extraction slow | Main task needs only ~6.4k records (decision 2); run with `joblib` in parallel; cache to parquet; all 43k only for the optional multi-label extension |
| neurokit2 fails on noisy records | Wrap in try/except, return NaN, impute later; log failure rate (good "noisy data" evidence) |
| O(n²) methods out of memory/time | Fixed stratified subsample (section 2.4) |
| Data leakage from scaling/selection on full data | Everything inside `sklearn.Pipeline`, fit on train folds only |
| Blocking on M1's feature table | M2/M3 prototype on toy data Mon–Tue; M1 ships a v0 table on CPSC 2018 Tuesday night |
| Running out of time on the weekend | Report sections written daily, not on Sunday; bonus paper dropped first |
