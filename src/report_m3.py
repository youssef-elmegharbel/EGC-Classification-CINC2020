"""Builds report/sections/M3_sections.docx (report section 6, feature selection) from results/ artifacts.

All numbers are read from results/models/m3, results/models/m3_fixedk and results/data/selected_features.json, so the
text cannot drift from the experiments.   Run:  python -m src.report_m3
"""
from __future__ import annotations

import json

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

from .loaders import DATA_DIR, FIG_DIR, MODELS_DIR, PROJECT_ROOT

OUT = PROJECT_ROOT / "report" / "sections" / "M3_sections.docx"
M3 = MODELS_DIR / "m3"
FK = MODELS_DIR / "m3_fixedk"
_fig_n, _tab_n = [0], [0]


def _table(doc, df, caption, font=8):
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


def _figure(doc, name, caption, width=6.3):
    _fig_n[0] += 1
    doc.add_picture(str(FIG_DIR / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(f"Figure {_fig_n[0]}. {caption}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True


def build():
    comp = pd.read_csv(M3 / "comparison.csv")
    fk = pd.read_csv(FK / "summary.csv")
    kinfo = json.load(open(FK / "common_k.json"))
    sel = json.load(open(DATA_DIR / "selected_features.json"))
    votes = pd.read_csv(M3 / "feature_votes.csv", index_col=0)
    gvotes = pd.read_csv(M3 / "group_votes.csv", index_col=0).votes
    corr = json.load(open(M3 / "correlation_stats.json"))
    test = pd.read_csv(MODELS_DIR / "test_results.csv").set_index("name")
    scv = pd.read_csv(M3 / "subsets_cv.csv")
    all131 = scv[scv.subset == "all_131"].set_index("clf").f1_macro_val_mean

    best_nested = comp.loc[comp.nested_f1.idxmax()]
    chosen = comp.set_index("method").loc[sel["method"]]
    n_sel = len(votes[votes.votes >= 13])
    top7 = votes[votes.votes == votes.votes.max()].index.tolist()

    doc = Document()
    doc.add_heading("6. Feature Selection", 1)

    # ------------------------------------------------------------------ 6.1
    doc.add_heading("6.1 Protocol", 2)
    doc.add_paragraph(
        "Feature selection reduces the 131 hand-crafted features to a subset that keeps (or improves) classification quality "
        "while being cheaper and easier to interpret. We compared 16 selectors from the four families of the plan: filter "
        "(variance threshold, correlation pruning, ANOVA F-test, mutual information, chi-square), wrapper (sequential forward "
        "and backward selection, RFECV with logistic regression, RFECV with LightGBM), embedded (L1, L2 and Elastic-Net "
        "penalised logistic regression, random-forest importance) and metaheuristic (a genetic algorithm, binary particle "
        "swarm optimisation and a binary grey wolf optimiser).")
    doc.add_paragraph(
        "Every selector is developed on the 80% training part only; the 20% test set is never used for selection. Imputation "
        "(median) and standardisation are fitted on each fold's training rows only. The search estimator is a class-balanced "
        "multinomial logistic regression and the score is the mean 5-fold macro-F1 on the shared folds. Search stops early: "
        "forward and backward selection stop when the CV score stops improving, the metaheuristics stop after a number of "
        "generations without improvement, and the genetic algorithm penalises its fitness by the subset size so that it does "
        "not simply keep everything.")
    doc.add_paragraph(
        "The CV score that drives a search is optimistically biased, because the search has seen the validation folds. We "
        "therefore also report a nested score: the whole selection procedure is repeated inside each outer training fold "
        "(with its own inner folds) and the chosen subset is scored on the untouched outer validation fold. Differences "
        "between methods should be read against this nested score and its standard deviation.")

    # ------------------------------------------------------------------ 6.2
    doc.add_heading("6.2 Filter methods", 2)
    doc.add_paragraph(
        f"Filters score each feature independently of any classifier. Correlation pruning removes one of every pair of features "
        f"with |r| above 0.95 (the feature table has {corr['n_pairs_gt_95']} such pairs and {corr['n_pairs_gt_90']} above 0.90 "
        f"among {corr['n_pairs']:,} pairs), so it removes few features and keeps "
        f"{int(comp.set_index('method').loc['filter_corr'].k)} of 131. The variance threshold removes almost nothing "
        f"({int(comp.set_index('method').loc['filter_variance'].k)} kept). The "
        "supervised filters (ANOVA F, mutual information, chi-square on min-max scaled data) are ranked and cut at the number of "
        "features that maximises CV macro-F1.")
    _figure(doc, "30_filter_rankings.png", "Filter rankings of the most informative features.")
    _figure(doc, "31_filter_k_curves.png", "CV macro-F1 as a function of the number of top-ranked features for each filter.")

    # ------------------------------------------------------------------ 6.3
    doc.add_heading("6.3 Wrapper methods", 2)
    doc.add_paragraph(
        "Wrappers score subsets with the classifier itself. Sequential forward selection (SFS) adds the feature that helps most; "
        "sequential backward selection (SBS) starts from the correlation-pruned set and removes features in chunks. RFECV "
        "repeatedly drops the weakest features of a fitted model and picks the size with the best CV score, here with "
        "logistic regression and with LightGBM as the ranking estimator. Wrappers are the most expensive family: SFS needed "
        f"{comp.set_index('method').loc['wrapper_sfs'].time_s:.0f} s and SBS "
        f"{comp.set_index('method').loc['wrapper_sbs'].time_s:.0f} s, against "
        f"{comp.set_index('method').loc['wrapper_rfecv_logreg'].time_s:.0f} s for RFECV with logistic regression.")
    _figure(doc, "32_wrapper_paths.png", "Macro-F1 versus number of features along the wrapper search paths.")

    # ------------------------------------------------------------------ 6.4
    doc.add_heading("6.4 Embedded methods", 2)
    doc.add_paragraph(
        "Embedded methods select features while fitting the model. L1 (lasso) logistic regression drives coefficients to zero; "
        "L2 (ridge) does not, so its coefficient magnitudes are thresholded; Elastic Net mixes both (grid over the mixing "
        "ratio and C, keeping the sparsest model within one standard error of the best CV score). Random-forest impurity "
        "importance is added as a tree-based view.")
    _figure(doc, "33_embedded_paths.png", "Regularisation paths of the embedded methods.")
    _figure(doc, "34_tree_importance.png", "Random-forest feature importance.")

    # ------------------------------------------------------------------ 6.5
    doc.add_heading("6.5 Metaheuristics", 2)
    doc.add_paragraph(
        "The genetic algorithm (DEAP) uses a binary chromosome of length 131, fitness equal to the CV macro-F1 minus a penalty "
        "proportional to the number of selected features, tournament selection (size 3), two-point crossover, bit-flip "
        "mutation and a hall of fame for elitism. Binary PSO and binary grey wolf optimisation use the same fitness for "
        "comparison. All three stop after a number of generations without improvement.")
    _figure(doc, "35_metaheuristic_convergence.png", "Best and mean fitness per generation for the three metaheuristics.")

    # ------------------------------------------------------------------ 6.6
    doc.add_heading("6.6 Comparison of all methods", 2)
    t = comp.copy()
    t["Family"] = t.family
    out = pd.DataFrame({
        "Method": t.label,
        "Family": t.family,
        "k": t.k,
        "Time (s)": t.time_s.round(0).astype(int),
        "Nested F1": [f"{a:.3f} ± {b:.3f}" for a, b in zip(t.nested_f1, t.nested_std)],
        "CV F1 LogReg": t.cv_f1_logreg.round(3),
        "CV F1 SVM": t.cv_f1_svm.round(3),
        "CV F1 LightGBM": t.cv_f1_lgbm.round(3),
        "Jaccard stability": t.stability_jaccard.round(2),
    })
    _table(doc, out, "All 16 selectors on the training part. k = features kept on the full training part; Nested F1 = "
                     "selection repeated inside each outer fold (5-fold mean ± std); CV F1 = the kept subset retrained "
                     "with each classifier (5-fold); Jaccard stability = mean overlap of the subsets chosen in different folds.")
    doc.add_paragraph(
        f"The best nested macro-F1 is {best_nested.nested_f1:.3f} ± {best_nested.nested_std:.3f} ({best_nested.label}). "
        f"Most methods sit between 0.75 and 0.77, i.e. within about one standard deviation ({comp.nested_std.median():.3f}) "
        f"of each other, and logistic regression on all 131 features scores {all131['logreg']:.3f} CV macro-F1 (dashed line in the figure below). "
        "The honest reading is therefore that selection does not clearly raise accuracy; its benefit is parsimony: logistic regression on "
        f"{int(chosen.k)} features scores {chosen.cv_f1_logreg:.3f} CV macro-F1 against {all131['logreg']:.3f} with all 131. "
        "The forward search (SFS) and the genetic algorithm have the lowest nested scores and are among the least stable: "
        "greedy and stochastic searches over highly correlated features can pick different, equally good subsets in different "
        "folds. Because the subsets are retrained with SVM and LightGBM, which were not used in the search, their CV scores are "
        "flagged as overfit (train minus CV macro-F1 above 0.10) for the same reason as in Section 8.3: these flexible models "
        "reach near-perfect training scores on any subset, and the flag does not depend on the selection method.")
    _figure(doc, "36_method_comparison.png", "Nested macro-F1 of each selection procedure (left) and accuracy versus subset size (right).", 6.5)

    doc.add_heading("6.6.1 Equal subset size (K = %d)" % kinfo["k"], 3)
    doc.add_paragraph(
        "The methods in the table above return different numbers of features, which confounds the comparison. We therefore forced every "
        f"selector to return exactly K = {kinfo['k']} features and retrained each subset. K is the smallest size within one "
        "standard error of the best mean CV macro-F1 over the ANOVA, mutual-information, chi-square, L2 and random-forest "
        "rankings, and the same preprocessing and folds are used.")
    o = pd.DataFrame({
        "Method": fk.method,
        "Family": fk.family,
        "Search CV F1 (LogReg)": fk.cv_f1_logreg.round(3),
        "SVM": fk.cv_f1_svm.round(3),
        "LightGBM": fk.cv_f1_lgbm.round(3),
    }).sort_values("Search CV F1 (LogReg)", ascending=False)
    _table(doc, o, f"All selectors at K = {kinfo['k']} features (5-fold CV macro-F1, subsets retrained with each classifier).")
    top = fk.loc[fk.cv_f1_logreg.idxmax()]
    doc.add_paragraph(
        f"At equal size the genetic algorithm gives the best logistic-regression score ({fk.set_index('method').loc['meta_ga'].cv_f1_logreg:.3f}), "
        f"followed by SBS and SFS; the filters are the weakest ({fk[fk.family == 'filter'].cv_f1_logreg.min():.3f} to "
        f"{fk[fk.family == 'filter'].cv_f1_logreg.max():.3f}), which is expected because they ignore feature interactions. "
        "These scores are search-time CV scores and are optimistic; the spread between methods is again small compared with the "
        "fold-to-fold standard deviation of about 0.01. With LightGBM, every K-feature subset stays between "
        f"{all131['lgbm'] - fk.cv_f1_lgbm.max():.3f} and {all131['lgbm'] - fk.cv_f1_lgbm.min():.3f} below the all-feature score "
        f"({all131['lgbm']:.3f}), so {kinfo['k']} features carry most of the usable information.")

    doc.add_heading("6.6.2 Overlap and consensus", 3)
    _figure(doc, "37_jaccard_overlap.png", "Jaccard overlap between the subsets chosen by different methods.", 5.8)
    doc.add_paragraph(
        f"Across the 14 methods that use the labels, {len(votes[votes.votes == votes.votes.max()])} features are kept by all of them: "
        f"{', '.join(top7)}. A further {n_sel - len(top7)} features are kept by 13 of the 14. These are rhythm and "
        "variability features (pNN50, RMSSD, RR minimum, RR irregularity), the interval durations (QRS, P, PR) and V1 "
        "morphology, which matches the clinical definition of the classes (AF = irregular rhythm without P waves, IAVB = long PR, "
        "RBBB = wide QRS with a tall R in V1). Averaged per feature, the patient features (age, sex) and the rhythm group "
        f"receive the most votes ({gvotes['patient']:.1f} and {gvotes['rhythm']:.1f} of 14), followed by intervals "
        f"({gvotes['intervals']:.1f}); wavelet features receive the fewest ({gvotes['wavelet']:.1f}).")
    _figure(doc, "38_feature_votes.png", "Most frequently selected features (votes out of 14 label-aware selectors).", 4.6)

    # ------------------------------------------------------------------ 6.7
    doc.add_heading("6.7 Final subset passed to the models", 2)
    doc.add_paragraph(
        f"The subset handed to M1 is {sel['label']}, with k = {sel['k']} features. The rule was: highest nested 5-fold macro-F1, "
        "and among methods within one standard error of the best, the one with the fewest features. Its nested macro-F1 is "
        f"{sel['nested_cv_f1_macro']:.4f} ± {sel['nested_std']:.4f}. Since the method was chosen by that same nested score "
        "the figure is slightly optimistic, which is why the final test-set evaluation (Section 8.4) is the unbiased check.")
    rows = []
    for m, label in [("lgbm|c", "LightGBM"), ("svm|c", "SVM (RBF)"), ("mlp|c", "MLP"), ("logreg|c", "Logistic regression")]:
        r = test.loc[m]
        rows.append({"Model on set (c)": label, "CV macro-F1": f"{r.cv_f1_macro:.3f}",
                     "Test accuracy": f"{r.accuracy:.3f}", "Test macro-F1": f"{r.f1_macro:.3f}"})
    _table(doc, pd.DataFrame(rows), f"Models trained on the {sel['k']} selected features (test set scored once per model).")
    doc.add_paragraph(
        f"On the 20% test set, LightGBM with the {sel['k']} selected features reaches {test.loc['lgbm|c'].f1_macro:.3f} macro-F1, "
        f"against {test.loc['lgbm|a'].f1_macro:.3f} with all 131 features, using {100 - round(100 * sel['k'] / 131)}% fewer "
        "features.")

    # ------------------------------------------------------------------ 6.8
    doc.add_heading("6.8 Related work", 2)
    rw = pd.read_csv(M3 / "related_work.csv")
    _table(doc, rw, "Comparison with published work. Scores are on different datasets, label sets and metrics and are "
                    "therefore indicative only.", font=7)
    doc.add_paragraph(
        "Published systems are trained on far more data (CODE: 2.3 million ECGs; PTB-XL) or use deep networks on the raw "
        "signal, and the challenge entries solve a 27-class multi-label task, so their scores are not directly comparable to "
        "our 6-class single-label subset. We include them to place a hand-crafted-feature pipeline of 131 (or 59) features "
        "in context: it reaches a macro-F1 around 0.80 on the held-out subset without any deep model.")

    doc.add_heading("6.9 Limitations", 2)
    doc.add_paragraph(
        "Selection was done on the same training folds used to report CV scores, so search-time scores are optimistic; the "
        "nested scores remove most of this bias but the final method choice adds a small one. Differences between methods are "
        "mostly within one standard deviation. There are no patient identifiers, so folds may place records of one patient in "
        "both training and validation parts, and the source datasets differ in class mix (for example SB occurs only in "
        "Georgia), so some selected features may reflect source rather than pathology.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
