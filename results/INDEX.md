# Result files — what each one is, and which script produces it

Every JSON file in this directory is written by exactly one script in `../scripts/`. Each file carries
the audit result plus a `generated_by` / `generated_at` field where the script sets one.

| Result file | Produced by | What it contains |
|---|---|---|
| `gcs_increment_audit.json` | `58_gcs_increment_audit.py` | Does adding the Glasgow Coma Scale to the 28-predictor model change anything? ΔAUC, DeLong z/p, category-free NRI, IDI and bootstrap CIs for **+total GCS** and **+motor GCS**; single-centre descriptives and single-predictor AUCs |
| `external_current.json` | `59_rebuild_external_figures.py` | **Single source of truth for every externally validated number** (discrimination, calibration, O:E, Brier) in the current analysis run. Also writes `figure_number_manifest.json` |
| `figure_number_manifest.json` | `59_rebuild_external_figures.py` | Every number printed inside the external-validation figures, each with the `source` path in `external_current.json` it was read from |
| `leakage_sensitivity.json` | `60_leakage_sensitivity.py` | Sensitivity analyses removing all aetiology diagnosis codes (the strongest of which could plausibly encode events occurring *after* the prediction time point): ΔAUC for the 1-year, in-hospital, survivor-subset and external scenarios, with DeLong tests |
| `calibration_ci_and_severity.json` | `62_calib_ci_and_severity.py` | Bootstrap (B = 1000) confidence intervals for calibration slope, intercept, O:E, ECE and Brier, per database × model; plus the head-to-head **SOFA / OASIS / SAPS-II vs the full model** comparison |
| `slope_epsilon_audit.json` | `63_check_slope_epsilon.py` | Calibration slope as a function of the clipping constant ε (1e-9, 1e-6, 1e-2). Demonstrates that the logistic-regression slope is **ε-dependent** and must not be reported as a single point estimate; XGBoost is stable |
| `v8_harmonisation.json` | `66_v8_audit_figures.py` | Unit and scale discordance variants (the "what if the local cohort is not unit-harmonised?" audit), with ΔAUC and O:E per learner |
| `v8_figure_number_manifest.json` | `66_v8_audit_figures.py` | **The figure-number manifest.** Every number printed inside every figure, as a `(figure, item, value, source)` record where `value` is *derived* from `source` — never hand-entered |
| `v8_appendix_data.json` | see note below | Aggregated numbers behind the supplementary tables (completeness, ε grid, unit conversions, harmonisation variants, leakage scenarios, external cohorts) |
| `internal_learners.json` | `68_internal_learners.py` | Internal comparison of the candidate learners |
| `cohort_descriptives.json` | `69_cohort_descriptives.py` | Descriptives of the analysis cohorts |
| `nwicu_vasopressor_audit.json` | `74_recover_nwicu_vasopressor.py` | Forensic audit of the NWICU vasopressor variable: the extraction layer had hard-coded a constant 0 while the source database does hold vasopressor prescriptions. Records the recovered prevalence under three route definitions |
| `nwicu_vasopressor_recovery.json` / `nwicu_vasopressor_variants.json` / `nwicu_vasopressor_forensics.json` | `74_recover_nwicu_vasopressor.py` | Supporting detail for the above |
| `revision2.json`, `revision_internal.json`, `revision_external.json` | `16_revision2_analyses.py` (and the round-1 revision scripts) | Results of the earlier revision rounds, retained because later scripts read them |

> **Note on `v8_appendix_data.json`**: it is assembled by the appendix-table builder, which is *not* included
> here (see the README section "What this repository deliberately does not contain"). The aggregated
> numbers themselves are included because they are referenced by the audits above.

## How to use these files to check the manuscript

1. Take any number from a table in the paper.
2. Find it in the relevant JSON above (the file names map onto the analyses one-to-one).
3. To trace a number that appears **inside a figure**, use `v8_figure_number_manifest.json`: each entry
   names the figure, the item, the printed value, and the JSON path it was derived from. If the two ever
   disagree, the manifest is wrong — that is the failure mode these audits were written to catch.

## Provenance and reproducibility conventions

* Every script reads its workspace root from the `ABI_BASE` environment variable (falling back to the
  original development path), and results are written to `$ABI_BASE/output/`. This directory is a snapshot
  of that output directory, restricted to the analysis results.
* All stochastic steps use fixed seeds, so re-running a script reproduces the same numbers.
* **No intermediate file stores a rounded value.** Rounding happens once, in the presentation layer.
  Storing an already-rounded value and rounding it again produces different answers for values that sit
  on a rounding boundary — the classic example being an AUC of 0.8464648 that becomes 0.8465 and then
  prints as 0.847 instead of 0.846.
* No patient-level identifiers appear anywhere in this directory.
