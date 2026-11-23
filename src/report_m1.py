"""Builds report/sections/M1_sections.docx (report sections 1-3, 7, 8) from results/ artifacts.

Numbers in tables are read from results/*.csv so the report cannot drift from the experiments.
Run:  python -m src.report_m1
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from .loaders import DATA_DIR, FIG_DIR, MODELS_DIR, PROJECT_ROOT, scan_records
from .models import best_per_model

FIG = FIG_DIR
OUT = PROJECT_ROOT / "report" / "sections" / "M1_sections.docx"
_fig_n = [0]
_tab_n = [0]


def _table(doc, df: pd.DataFrame, caption: str, widths=None, font=8):
    _tab_n[0] += 1
    p = doc.add_paragraph(f"Table {_tab_n[0]}. {caption}")
    p.runs[0].bold = True
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for par in c.paragraphs:
                for run in par.runs:
                    run.font.size = Pt(font)
    doc.add_paragraph()


def _figure(doc, name: str, caption: str, width=6.3):
    _fig_n[0] += 1
    doc.add_picture(str(FIG / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(f"Figure {_fig_n[0]}. {caption}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True


def _pm(row, metric, tag="val"):
    return f"{row[f'{metric}_{tag}_mean']:.3f} ± {row[f'{metric}_{tag}_std']:.3f}"


def build() -> "Path":

    meta = scan_records()
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")
    diag = pd.read_parquet(DATA_DIR / "extraction_diag.parquet")
    fails = pd.read_csv(DATA_DIR / "feature_failures.csv", index_col=0)
    dd = pd.read_csv(DATA_DIR / "data_dictionary.csv")
    base = pd.read_csv(MODELS_DIR / "baselines.csv")
    best = best_per_model("a")
    test = pd.read_csv(MODELS_DIR / "test_results.csv")
    pc = pd.read_csv(MODELS_DIR / "per_class_final.csv", index_col=0)
    grid = pd.read_csv(MODELS_DIR / "grid_results_a.csv")
    vc = pd.read_csv(MODELS_DIR / "validation_curves.csv")
    final = test.sort_values("cv_f1_macro", ascending=False).iloc[0]
    import json
    n_tr = len(json.loads((DATA_DIR / "splits.json").read_text())["train_ids"])
    cls_n = labels["label"].value_counts()

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ------------------------------------------------------------------ 1
    doc.add_heading("1. Problem Introduction", 1)
    doc.add_paragraph(
        "Cardiovascular disease is the leading cause of death worldwide, and the 12-lead electrocardiogram (ECG) is one of "
        "the cheapest, fastest and most widely available tests for detecting it. Reading an ECG, however, needs a trained "
        "cardiologist, which is slow and scarce in many hospitals and regions. ECG-Net asks whether classical machine "
        "learning on hand-crafted, clinically interpretable features can recognise common cardiac abnormalities "
        "automatically from a 12-lead recording and basic patient data (age and sex).")
    doc.add_paragraph(
        "Phase 0 proposed a 27-class multi-label problem. Phase 1 needs one class per record (LDA, a single confusion "
        "matrix, coloured 2D embeddings, filter tests and GA fitness all assume it), so Phase 1 focuses on the "
        "single-diagnosis subset: a 6-class single-label task over normal sinus rhythm (NSR), right bundle branch block "
        "(RBBB), atrial fibrillation (AF), first-degree AV block (IAVB), premature atrial contraction (PAC) and sinus "
        "bradycardia (SB). The full multi-label problem is left as an extension. These six classes are defined by "
        "measurable properties (SB: heart rate < 60 bpm; IAVB: PR interval > 200 ms; AF: irregular RR intervals and no "
        "P waves; PAC: premature beats; RBBB: wide QRS and a characteristic V1 shape), so interpretable features should "
        "carry most of the signal.")
    doc.add_paragraph(
        "Stakeholders are patients with undiagnosed or intermittent arrhythmias, cardiologists, emergency physicians and "
        "nurses who benefit from triage and a second opinion, and developers of wearable and remote-monitoring systems.")

    # ------------------------------------------------------------------ 2
    doc.add_heading("2. Dataset Description and Difficulty", 1)
    doc.add_paragraph(
        "We use the training set of the PhysioNet/Computing in Cardiology Challenge 2020 (Perez Alday et al., Physiol. "
        f"Meas. 2020): {len(meta):,} recordings of 12-lead ECGs from six databases. Each record has a WFDB header with "
        "age, sex and one or more SNOMED-CT diagnosis codes, and a signal file of 12 channels. The challenge scores 27 "
        "diagnoses, three pairs of which are equivalent codes (CRBBB/RBBB, SVPB/PAC, VPB/PVC) and are merged before counting.")
    src = meta.groupby("source").agg(Records=("record_id", "size"), fs_Hz=("fs", lambda s: ", ".join(str(int(v)) for v in sorted(s.unique()))),
                                     min_len_s=("duration_s", "min"), max_len_s=("duration_s", "max"),
                                     age_missing=("age", lambda s: int(s.isna().sum()))).reset_index()
    src[["min_len_s", "max_len_s"]] = src[["min_len_s", "max_len_s"]].round(0).astype(int)
    _table(doc, src.rename(columns={"source": "Source"}), "The six source databases (header statistics).")
    doc.add_heading("2.1 Task subset", 2)
    doc.add_paragraph(
        "A record is in the task if its header carries exactly one diagnosis code (19,555 of 43,101 after merging "
        "equivalents) and that code is one of the six classes. NSR is then randomly downsampled to 2,000 records "
        "(random_state = 42), because otherwise it would be 67% of the data and accuracy would be meaningless. "
        f"The final table has {len(labels):,} records.")
    plan = {"NSR": 9182, "RBBB": 1605, "AF": 1093, "IAVB": 754, "PAC": 538, "SB": 456}
    tab = pd.DataFrame({"Class": list(plan), "Single-dx records (plan)": list(plan.values()),
                        "After merging equivalents": [{"NSR": 9182, "RBBB": 1611, "AF": 1093, "IAVB": 754, "PAC": 627, "SB": 456}[c] for c in plan],
                        "In task table": [int(cls_n[c]) for c in plan]})
    _table(doc, tab, "Class counts. RBBB and PAC grow slightly because records whose only code was CRBBB or SVPB are merged in.")
    _figure(doc, "01_class_frequencies.png", "Class frequencies: the full multi-label picture, the 6-class single-diagnosis subset, and after NSR downsampling.")
    doc.add_heading("2.2 Why the dataset is difficult", 2)
    for t in [
        "Class imbalance: even after downsampling NSR, PAC (627) and SB (456) are much smaller than NSR (2,000) or RBBB (1,611). We use class weights where the model supports them and rank by macro-F1.",
        "Multi-label overlap: 55% of records carry two or more diagnoses (Figure 2). Restricting to single-diagnosis records removes this overlap, and is a selection effect that the report states as a limitation.",
        "Multi-source shift: the databases differ in hospital, device, sampling rate (257/500/1000 Hz), record length (5 s to 30 min) and labelling style. Class and source are strongly confounded: all 456 SB records come from the Georgia database, NSR is mostly PTB-XL, and RBBB, IAVB and PAC are mostly CPSC (Figure 3). A model can therefore partly learn the source instead of the diagnosis, and accuracy on a new hospital will be lower than our cross-validated numbers.",
        "Noise and label errors: baseline wander, high-frequency noise and flat leads occur (Section 3), and the challenge organisers state that some labels are debatable.",
        "Missing values and corrupt files: age is missing for a few hundred records; two PTB-XL records (HR09972, HR11696) have truncated signal files and cannot be read. They stay in the table with NaN signal features rather than being dropped.",
    ]:
        doc.add_paragraph(t, style="List Bullet")
    _figure(doc, "02_cooccurrence.png", "Conditional label co-occurrence on all 43,101 records (27 scored classes).", 5.2)
    _figure(doc, "03_source_distributions.png", "Per-source age, sex, sampling rate, length and class mix. Note that SB occurs only in Georgia.")

    # ------------------------------------------------------------------ 3
    doc.add_heading("3. Data Analysis and Preprocessing", 1)
    doc.add_heading("3.1 Signal preprocessing", 2)
    doc.add_paragraph(
        "Each record is resampled to 500 Hz, band-pass filtered 0.5-40 Hz with a zero-phase 3rd-order Butterworth filter "
        "(the 0.5 Hz high-pass also removes baseline wander), and cropped to the first 10 s (shorter records are "
        "zero-padded; features only use the real samples). Two outputs are kept: the filtered signal in mV, which is used "
        "for delineation and amplitudes (z-scoring would destroy their meaning), and a per-lead z-scored copy, used for "
        "spectral and wavelet features. Quality-control flags are computed per record: flat leads (std < 0.01 mV), noisy "
        "leads (more than 25% of power above 40 Hz), and the baseline-wander power ratio.")
    _figure(doc, "06b_before_after_preprocessing.png", "Raw versus filtered lead II for three records.")
    _figure(doc, "05_noise_examples.png", "Typical quality problems: baseline wander, high-frequency noise and a flat lead, with the filtered result.")
    qc = (diag["qc_n_noisy_leads"] > 0).mean() * 100
    doc.add_paragraph(
        f"In the task table {qc:.1f}% of records have at least one noisy lead, {(diag['qc_n_flat_leads'] > 0).mean() * 100:.1f}% have a flat lead "
        f"and {diag['qc_padded'].mean() * 100:.1f}% are shorter than 10 s. The filter removes most of this noise, so these rates are low after "
        "filtering but they confirm the data are not clean.")
    doc.add_heading("3.2 Hand-crafted features", 2)
    g = dd.groupby("group").size()
    doc.add_paragraph(
        f"Feature extraction produces one row per record with {len(dd)} hand-crafted columns (no raw-signal features, as decided "
        "for Phase 1). R peaks are detected on lead II with neurokit2, and its discrete-wavelet delineation runs once on lead II "
        "(retried on V1, then I) to find the P, QRS and T fiducial points, which are reused to read amplitudes on all 12 leads. "
        "If delineation fails a feature is stored as NaN, and the record is kept; imputation happens inside the training pipeline.")
    grp = pd.DataFrame({"Group": ["Rhythm (lead II)", "Intervals (global)", "Per-lead x 12 leads", "Spectral (lead II)", "Wavelet (lead II)", "Patient"],
                        "Columns": [int(g.get("rhythm", 0)), int(g.get("intervals", 0)), int(g.get("per_lead", 0)), int(g.get("spectral", 0)), int(g.get("wavelet", 0)), int(g.get("patient", 0))],
                        "Content": ["HR, RR mean/median/std/min/max, RR CV, RMSSD, pNN50, RR irregularity, short-RR fraction, beat count",
                                    "PR, QRS, QT, QTc (Bazett), P duration, fraction of beats with a prominent P wave",
                                    "R, S, T, P amplitude, std, skewness, kurtosis, energy for each lead",
                                    "Welch band powers, entropy, dominant frequency, centroid, log power",
                                    "db4 level 1-5 detail and approximation relative energies, wavelet entropy",
                                    "Age, sex"]})
    _table(doc, grp, "Feature groups (full dictionary in results/data/data_dictionary.csv).")
    doc.add_paragraph(
        "One design detail matters: DWT delineation reports a P peak on almost every beat, even in AF, so a plain 'P detected' fraction "
        "was constant at about 1.0 across classes. We redefined it as the fraction of beats whose baseline-corrected P amplitude is "
        "at least 0.025 mV, which separates AF (median 0.62) from the other classes (about 1.0).")
    a = fails.loc["ALL"]
    doc.add_paragraph(
        f"Delineation failure is rare: R peaks were found on {100 * (1 - a['no_peaks']):.1f}% of records, full delineation succeeded on "
        f"{100 * (1 - a['delin_failed']):.1f}%, and the NaN rate is {100 * a['pr_nan']:.1f}% for PR and {100 * a['t_nan']:.1f}% for QT (the two least "
        "reliable intervals). Feature extraction for all 6,541 records took about 100 s on 10 CPU cores.")
    _figure(doc, "06_delineation_overlay.png", "Delineation overlay on lead II, one record per class (P green, R red, T purple, QRS onset black).")
    _figure(doc, "04_missingness.png", "Missing values: top NaN rates and delineation NaN rate by class and source.")
    doc.add_heading("3.3 Feature analysis", 2)
    doc.add_paragraph(
        "The class-conditional distributions match the clinical definitions (Figure 8): SB sits below 60 bpm, IAVB has a longer PR "
        "interval, AF shows high RR irregularity and fewer prominent P waves, PAC shows a share of short RR intervals, and RBBB is "
        "visible in the V1 amplitudes. A caveat: the measured QRS duration is not clearly longer for RBBB, and is widest for SB, "
        "which we attribute to delineation of smooth low-amplitude Georgia recordings; the V1 R and S amplitudes carry the RBBB signal instead. "
        "The Spearman correlation matrix shows strong redundancy between leads and between related features, which is intended, "
        "and gives PCA and correlation pruning (M2, M3) something to work with.")
    _figure(doc, "07b_key_feature_distributions.png", "Key features by class (outliers hidden).")
    _figure(doc, "07_feature_correlation.png", "Feature correlation (Spearman, hierarchically clustered).", 5.0)
    doc.add_paragraph(
        "Splitting protocol: stratified 80/20 train/test split by class (random_state = 42), then 5 stratified folds inside the 80% only "
        f"({n_tr:,} training and {len(labels) - n_tr:,} test records). The test set is used once at the end. Source mix per split and fold is "
        "close to the overall mix. Because the headers carry no patient identifiers, recordings cannot be grouped by patient; most "
        "databases have one recording per patient, but this remains a limitation. Visualisation after dimensionality reduction and "
        "clustering is covered in Sections 4 and 5.")

    # ------------------------------------------------------------------ 7
    doc.add_heading("7. Model Implementations", 1)
    doc.add_paragraph(
        "Six classifiers are implemented, each as one scikit-learn Pipeline: median imputer, standard scaler, classifier. Imputation, "
        "scaling and (in later experiments) reduction or selection are therefore fit on training folds only, with no leakage.")
    mt = pd.DataFrame({
        "Model": ["Logistic regression", "k-NN", "SVM (RBF)", "Random forest", "LightGBM", "MLP"],
        "Class imbalance": ["class_weight=balanced", "none (distance-weighted)", "class_weight=balanced", "class_weight=balanced", "class_weight=balanced", "none (sklearn MLP has no class weights)"],
        "Overfitting control": ["tuned on C", "tuned on k, L1 distance in round 2", "tuned on C and gamma",
                                "max_depth, min_samples_leaf, max_features", "early stopping on a validation split carved from each training fold; num_leaves, min_child_samples, L2",
                                "early_stopping=True (validation_fraction 0.1, patience 10), L2 alpha"],
        "Grid (round 1 / round 2)": ["C in {0.01, 0.1, 1, 10}", "k in {3, 5, 11, 21} / p=1", "C in {1, 10, 100} x gamma / smaller C and gamma",
                                    "depth {8, 16, None} x leaf {1, 5, 10} / depth {10, 14} x leaf {2, 3} x max_features",
                                    "leaves {15, 31} x lr {0.05, 0.1} / leaves {7, 15}, lr 0.03, min_child {20, 50}, lambda {1, 10}",
                                    "layers {(64), (128, 64)} x alpha {1e-4, 1e-2} / alpha {0.1, 1}"]})
    _table(doc, mt, "Models, imbalance handling, overfitting control and search grids.", font=7.5)
    doc.add_paragraph(
        "Tuning uses small fixed grids over the shared 5 folds inside the 80% training set; the configuration with the best mean CV macro-F1 "
        "is chosen. Round 1 covers the plan's grids; round 2 adds stronger regularisation for models whose best round-1 configuration was "
        "flagged overfit. The grid scores themselves are the CV estimates, so the best configuration is slightly optimistic (no nested loop); "
        "the untouched test set provides the unbiased number. LightGBM was chosen over XGBoost for native early stopping, NaN handling and CPU speed. "
        f"In total {len(grid)} configurations x 5 folds were evaluated.")

    # ------------------------------------------------------------------ 8
    doc.add_heading("8. Evaluation", 1)
    doc.add_heading("8.1 Protocol and metrics", 2)
    doc.add_paragraph(
        "Every table is produced by the shared module src/evaluate.py. We report accuracy and error rate (1 - accuracy), precision, "
        "recall and F1 as macro and weighted averages and per class, one-vs-rest ROC-AUC (macro and weighted), raw and row-normalised "
        "confusion matrices, and 5-fold results as mean ± std. Models are ranked by macro-F1. Each experiment also logs the training score, "
        "and a model is flagged overfit if train macro-F1 minus CV macro-F1 exceeds 0.10, and underfit if both are within 0.05 of the "
        "dummy baseline. The test set is scored once per model through a guarded function that refuses repeats.")
    bt = base.set_index("name")
    doc.add_paragraph(
        f"Dummy baselines: a stratified-random classifier reaches accuracy {bt.loc['dummy_stratified', 'accuracy_val_mean']:.3f} and macro-F1 "
        f"{bt.loc['dummy_stratified', 'f1_macro_val_mean']:.3f}; always predicting the most frequent class (NSR) gives accuracy "
        f"{bt.loc['dummy_most_frequent', 'accuracy_val_mean']:.3f} but macro-F1 only {bt.loc['dummy_most_frequent', 'f1_macro_val_mean']:.3f}, "
        "which is why macro-F1 is the ranking metric.")
    doc.add_heading("8.2 Results on feature set (a): all 131 features", 2)
    cv = pd.DataFrame({
        "Model": best["model"].astype(str).map({"logreg": "Logistic regression", "knn": "k-NN", "svm": "SVM (RBF)", "rf": "Random forest", "lgbm": "LightGBM", "mlp": "MLP"}),
        "CV macro-F1": best.apply(lambda r: _pm(r, "f1_macro"), axis=1),
        "CV accuracy": best.apply(lambda r: _pm(r, "accuracy"), axis=1),
        "CV ROC-AUC (macro)": best.apply(lambda r: _pm(r, "roc_auc_macro"), axis=1),
        "Train macro-F1": best["f1_macro_train_mean"].map("{:.3f}".format),
        "Gap": best["gap_f1_macro"].map("{:.3f}".format),
        "Flag": best["flag"]})
    _table(doc, cv, "5-fold cross-validation on the 80% training set (mean ± std over folds), best configuration per model.", font=8)
    tt = test.set_index("name")
    order = [f"{m}|a" for m in best["model"].astype(str)]
    tst = pd.DataFrame({
        "Model": [m.split("|")[0] for m in order],
        "Accuracy": [f"{tt.loc[o, 'accuracy']:.3f}" for o in order],
        "Error rate": [f"{tt.loc[o, 'error_rate']:.3f}" for o in order],
        "Macro P": [f"{tt.loc[o, 'precision_macro']:.3f}" for o in order],
        "Macro R": [f"{tt.loc[o, 'recall_macro']:.3f}" for o in order],
        "Macro F1": [f"{tt.loc[o, 'f1_macro']:.3f}" for o in order],
        "Weighted F1": [f"{tt.loc[o, 'f1_weighted']:.3f}" for o in order],
        "AUC macro": [f"{tt.loc[o, 'roc_auc_macro']:.3f}" for o in order],
        "AUC weighted": [f"{tt.loc[o, 'roc_auc_weighted']:.3f}" for o in order]})
    _table(doc, tst, "Held-out 20% test set (1,309 records), each model scored once. Test numbers did not influence any choice.", font=8)
    doc.add_paragraph(
        f"The best model by CV macro-F1 is {final['name'].split('|')[0]} (CV {final['cv_f1_macro']:.3f}, test macro-F1 {final['f1_macro']:.3f}, "
        f"test accuracy {final['accuracy']:.3f}, weighted AUC {final['roc_auc_weighted']:.3f}). Test scores are close to CV scores for all six models, "
        "so model selection on the folds did not overfit the folds. All models beat the dummy baseline by a wide margin, so none underfits.")
    pcd = pc.round(3).reset_index().rename(columns={"index": "Class"})
    _table(doc, pcd, f"Per-class precision, recall and F1 of the final model ({final['name'].split('|')[0]}) on the test set.")
    _figure(doc, "10_confusion_final.png", "Confusion matrix of the final model on the test set: counts (left) and row-normalised recall (right).")
    _figure(doc, "10_roc_final.png", "One-vs-rest ROC curves of the final model on the test set, with macro and weighted AUC.", 4.6)
    doc.add_paragraph(
        "Per-class behaviour matches the clinical picture. AF (F1 0.91) and RBBB (0.88) are recognised best because their markers (RR irregularity, "
        "absent P waves; V1 morphology) are strong. PAC is weakest (recall 0.63): premature beats are intermittent, so a 10 s window may contain "
        "only one or none, and PAC is confused with AF, IAVB and NSR. IAVB (F1 0.70) depends on a PR interval that delineation measures less reliably. "
        "SB is recognised well (recall 0.86), but part of this may come from the source confound: all SB records are from Georgia.")
    _figure(doc, "08_models_set_a.png", "Macro-F1 per model: 5-fold CV (bars, ± std) versus train (red) and the dummy baseline.", 4.8)
    doc.add_heading("8.3 Over- and underfitting analysis", 2)
    nflag = int((best["flag"] == "overfit").sum())
    doc.add_paragraph(
        f"{nflag} of 6 best configurations are flagged overfit (train - CV macro-F1 > 0.10): k-NN (train 1.00, CV 0.62), random forest, SVM and LightGBM "
        "(train about 0.94-0.98, CV 0.75-0.80) and the MLP (gap 0.11); logistic regression is the only model that is not flagged (gap 0.07). "
        "Round 2 applied stronger regularisation, but CV macro-F1 changed by at most about 0.015 while the gap stayed large. The validation curves "
        "explain why: for LightGBM, SVM and the MLP, CV macro-F1 is flat over a wide range of the regularisation parameter while the training score "
        "rises, so the limit is the information in the features (and class overlap and source noise) rather than the model capacity. "
        "Overfit by this definition therefore does not mean the CV score would improve with simpler models. It does mean that complex models "
        "memorise the training folds, which is why CV and test scores, not train scores, are reported.")
    _figure(doc, "11_validation_curves.png", "Validation curves: train and 5-fold CV macro-F1 versus the main hyperparameter of each model.")
    doc.add_paragraph(
        "Learning curves for the best model (LightGBM) and the most overfit model (k-NN) are shown next. For LightGBM, CV macro-F1 rises from 0.72 at "
        "837 training records to 0.80 at 4,185 and starts to flatten, while the training score stays near 0.97-0.98: more data would help only a little. "
        "For k-NN the gap stays at about 0.4 at every size, a sign of high variance in 131 dimensions with many redundant, differently scaled features.")
    _figure(doc, "11_learning_lgbm.png", "Learning curve of LightGBM (set a).", 4.5)
    if (FIG / "11_learning_knn.png").exists():
        _figure(doc, "11_learning_knn.png", "Learning curve of k-NN (set a), the most overfit model.", 4.5)
    doc.add_heading("8.4 Reduced and selected feature sets", 2)
    names = {"logreg": "Logistic regression", "knn": "k-NN", "svm": "SVM (RBF)", "rf": "Random forest", "lgbm": "LightGBM", "mlp": "MLP"}
    bb, bc = best_per_model("b").set_index("model"), best_per_model("c").set_index("model")
    ba = best.assign(model=best["model"].astype(str)).set_index("model")
    bb.index, bc.index = bb.index.astype(str), bc.index.astype(str)
    ms = list(ba.index)
    doc.add_paragraph(
        "The same six models, grids, folds and evaluation code were run on two further inputs. Set (b) is M2's best reduced representation: "
        "impute, scale, clip at 5 standard deviations, then LDA with 5 components (selected by M2 with 5-fold CV among runs not flagged overfit). "
        "Set (c) is M3's best selected subset: 59 of the 131 features chosen by RFECV with logistic regression. In both cases the reducer or selector "
        "sits inside the Pipeline and is fit on training folds only. Hyperparameters were chosen by CV macro-F1, and each model was scored once on the test set.")
    cmp_ = pd.DataFrame({
        "Model": [names[m] for m in ms],
        "CV F1 (a, 131)": [f"{ba.loc[m, 'f1_macro_val_mean']:.3f}" for m in ms],
        "CV F1 (b, LDA-5)": [f"{bb.loc[m, 'f1_macro_val_mean']:.3f}" for m in ms],
        "CV F1 (c, 59)": [f"{bc.loc[m, 'f1_macro_val_mean']:.3f}" for m in ms],
        "Flag a/b/c": [f"{ba.loc[m, 'flag']}/{bb.loc[m, 'flag']}/{bc.loc[m, 'flag']}" for m in ms],
        "Test F1 a": [f"{tt.loc[m + '|a', 'f1_macro']:.3f}" for m in ms],
        "Test F1 b": [f"{tt.loc[m + '|b', 'f1_macro']:.3f}" for m in ms],
        "Test F1 c": [f"{tt.loc[m + '|c', 'f1_macro']:.3f}" for m in ms],
        "Test acc. a/b/c": [f"{tt.loc[m + '|a', 'accuracy']:.3f}/{tt.loc[m + '|b', 'accuracy']:.3f}/{tt.loc[m + '|c', 'accuracy']:.3f}" for m in ms]})
    _table(doc, cmp_, "Best configuration per model on each feature set: 5-fold CV macro-F1, overfit flag (train - CV macro-F1 > 0.10) and one-time test scores.", font=7)
    doc.add_paragraph(
        "Neither reduction nor selection beats the full feature set for the best model. LightGBM on all features has CV macro-F1 0.796 and test macro-F1 0.803; "
        "on the 59 selected features it reaches 0.793 and 0.796, a difference far smaller than the fold standard deviation (about 0.01), "
        "so M3's subset keeps the accuracy with 45% of the features. LightGBM on the 5 LDA components drops to 0.761 CV and 0.763 test, because LDA is a linear, "
        "supervised projection that discards non-linear structure that the boosted trees can use.")
    doc.add_paragraph(
        "Reduction helps the models that suffer in high dimensions. k-NN improves from CV 0.617 (test 0.613) on 131 features to 0.767 (test 0.767) on the LDA components, "
        "and the SVM, MLP and logistic regression reach CV 0.763-0.768 with a small train-CV gap. "
        "Four of six models are not flagged overfit on set (b) (logistic regression, SVM, LightGBM, MLP), against one of six on set (a); k-NN is flagged "
        "only because distance weighting makes its training score exactly 1.0, and the random forest keeps a gap of about 0.16. "
        "On set (c) the picture is close to set (a): only logistic regression is not flagged. For the weaker models, selection helps modestly "
        "(logistic regression CV 0.762 -> 0.781, SVM 0.766 -> 0.780, MLP 0.764 -> 0.776), while the random forest is unchanged within noise. "
        "No model is near the dummy baseline (macro-F1 0.166), so none underfits.")
    doc.add_paragraph(
        "Test scores agree with CV scores to within about 0.02 for every model and set, with the largest differences for the random forest on set (b) "
        "(CV 0.763, test 0.777) and the MLP on set (b) (0.768, 0.778). The recommended final model therefore remains LightGBM on all features "
        "(best test macro-F1 0.803); LightGBM on M3's 59 features is an equally accurate and cheaper alternative, and the LDA representation is the best choice "
        "when a low-dimensional, less overfit input is wanted (for example for k-NN or for visualisation). "
        "Set (c) CV scores are slightly optimistic because the subset was chosen on the same training data; M3's nested estimate for this subset with logistic regression "
        "is 0.767, close to the 0.766 test score of logistic regression on set (c).")
    doc.add_heading("8.5 Limitations", 2)
    for t in [
        "No patient identifiers: recordings cannot be grouped by patient, so folds may contain several recordings of one patient, which would make scores slightly optimistic.",
        "Class-source confounding (SB only in Georgia, NSR mostly PTB-XL) and multi-source shift: results measure within-database discrimination and are not a guarantee on a new hospital. A held-out-database test is left for future work.",
        "Single-diagnosis subset only: the 6-class task excludes records with several diagnoses and unscored diagnoses, so it is easier than the real multi-label problem.",
        "Noisy records and delineation: failures are rare (0.1% full delineation failure) but the intervals depend on automatic lead-II delineation, whose QRS duration looks unreliable for some sources.",
        "Hyperparameters were selected on the same 5 folds used for reporting CV scores (no nested loop), so CV scores are slightly optimistic. The test set is the unbiased number. M3's feature-selection step, if scored on the same folds, adds a further small optimism to set (c).",
        "k-NN and the MLP run without class weights, which is a handicap for the minority classes.",
    ]:
        doc.add_paragraph(t, style="List Bullet")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
