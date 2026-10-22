# Result files — what each one is, and which script produces it

Every JSON file in this directory is written by exactly one script in `../scripts/`, and each file
carries the audit result plus a `generated_by` / `generated_at` field where the script sets one.

## ⚠️ Snapshot version (read this first)

The analysis behind this repository has been run more than once, so this directory holds **two
vintages side by side**: twenty files describe the **current state** — nineteen snapshots of the
**current analysis run** plus one audit of the frozen submission package — and eight are snapshots
of the **earlier run** that it superseded. Nothing here is hand-edited — every file is a verbatim
copy of the output directory of the run that produced it, and the *Vintage* column below says which.
The earlier-run files are kept because they are still the correct reference for the archived audit
figures and because later scripts read them.

What the current run changed, and where it shows up in these files:

| Change in the current run | Where it lands |
|---|---|
| The NWICU outcome labels were rebuilt as the union of three sources (death time inside the stay, discharge disposition, hospital-expiry flag) with a lower time bound; the unbounded variant is kept as a sensitivity analysis | NWICU one-year events **699** and in-hospital **376** — the earlier snapshots carried **635** and **16**. See `external_current.json`, `cohort_descriptives.json`, `leakage_sensitivity_v9.json`, `nwicu_vasopressor_recovery.json` |
| External validation was re-aggregated at the source layer, and this became the main analysis (the placeholder-cleaned variant is the sensitivity analysis) | NWICU one-year logistic **AUC 0.7652, calibration slope 0.872, O:E 0.994** — the earlier snapshots carried 0.6971 / 0.039 / 0.535 |
| The tree learners' hyper-parameters were re-selected and re-fitted on the current training split | `internal_learners.json` (five learners) and the optimism-correction block of `revision_internal.json` |
| The NWICU vasopressor recovery was re-run on the current data (prevalence unchanged at 11.1%), and `sql/11_nwicu_cohort.sql` was rewritten so that the cohort SQL itself reproduces it | `nwicu_vasopressor_recovery.json` |

The fourteen current-run files are written by two archived scripts: `98_build_v9_results.py`
(thirteen files) translates the current run's analysis outputs — the files whose names carry a
`_v2` / `_v9` suffix — into the schema the rest of the pipeline reads, and
`91_leakage_sensitivity_v9.py` writes the leakage grid directly. The header of
`98_build_v9_results.py` lists every input it reads and every field it carries over unchanged from
the earlier run.

Two consequences are worth knowing before comparing a file against the paper:

* The `recalibration` sub-block of `external_current.json` is a **carried-over placeholder** from the
  earlier run and is *not* read by the manuscript builder. The current-run calibration numbers are in
  `calibration_ci_and_severity.json` → `calibration_ci`.
* `leakage_sensitivity.json` is the **earlier** grid. The current grid — the one the manuscript's
  appendix note points at — is `leakage_sensitivity_v9.json`; its `summary` and `external_delta`
  blocks are exactly the blocks the manuscript builder reads.

## The files

| Result file | Vintage | Produced by | What it contains |
|---|---|---|---|
| `gcs_increment_audit.json` | current run | `98_build_v9_results.py` | Does adding the Glasgow Coma Scale to the 28-predictor model change anything? ΔAUC, DeLong z/p, category-free NRI, IDI and bootstrap CIs for **+total GCS** and **+motor GCS**; single-centre descriptives and single-predictor AUCs |
| `external_current.json` | current run | `98_build_v9_results.py` | **Single source of truth for every externally validated number** (discrimination, calibration, O:E, Brier) in the current analysis run. ⚠ Its `recalibration` sub-block is a carried-over earlier-run placeholder (see above) |
| `leakage_sensitivity_v9.json` | current run | `91_leakage_sensitivity_v9.py` | **The leakage-sensitivity grid for the current analysis run.** The same five feature sets and the same three scenarios (A one-year, B in-hospital, C one-year among hospital survivors) as `leakage_sensitivity.json`, **plus the joint deletion of the seven aetiology indicators with the Charlson index** (`NO_AET_CHARLSON`, 28 → 20 predictors). For every scenario × database × learner it records the full-model AUC, the reduced-model AUC and ΔAUC, and the same quantities for the joint deletion; `delong` holds the paired DeLong comparisons and `external_delta` the external cohorts. **The internal one-year cell is the one summarized in the manuscript** (note to Table S7 in Multimedia Appendix 1); the remaining scenarios and external cohorts are in this file |
| `calibration_ci_and_severity.json` | current run | `98_build_v9_results.py` | Bootstrap (B = 1000) confidence intervals for calibration slope, intercept, O:E, ECE and Brier, per database × model; plus the head-to-head **SOFA / OASIS / SAPS-II vs the full model** comparison |
| `slope_epsilon_audit.json` | current run | `98_build_v9_results.py` | Calibration slope as a function of the clipping constant ε (1e-9, 1e-6, 1e-2). Demonstrates that the logistic-regression slope is **ε-dependent** and must not be reported as a single point estimate; XGBoost is stable |
| `v8_harmonisation.json` | current run | `98_build_v9_results.py` | Unit and scale discordance variants (the "what if the local cohort is not unit-harmonised?" audit), with ΔAUC and O:E per learner |
| `v8_appendix_data.json` | current run | `98_build_v9_results.py` | Aggregated numbers behind the supplementary tables (completeness, ε grid, unit conversions, harmonisation variants, leakage scenarios, external cohorts) — see note below |
| `internal_learners.json` | current run | `98_build_v9_results.py` | Internal comparison of the candidate learners |
| `cohort_descriptives.json` | current run | `98_build_v9_results.py` | Descriptives of the analysis cohorts (n, events, event rate, age, sex, treatment intensity per database) |
| `nwicu_vasopressor_recovery.json` | current run | `98_build_v9_results.py` | Recovery of the NWICU vasopressor variable: the extraction layer had hard-coded a constant 0 while the source database does hold vasopressor prescriptions. Records the prevalence under three route definitions (chosen variant: 11.1%) and the effect of using the recovered variable |
| `official_score_provenance.json` | current run | `98_build_v9_results.py` | Provenance of the official SOFA / OASIS / SAPS-II tables: the local database carries no build metadata, so equivalence to the official formulas is established row by row instead of naming a commit. **Carried over unchanged** from the earlier run — its counts are structure counts, which the current run did not touch |
| `revision2.json`, `revision_external.json`, `revision_internal.json` | current run | `98_build_v9_results.py` | Results of the earlier revision rounds, retained because later scripts read them |
| `figure_number_manifest.json` | **earlier run** | `59_rebuild_external_figures.py` | Every number printed inside the external-validation figures, each with the `source` path in `external_current.json` it was read from. **V8-legacy snapshot**: its NWICU / INSPIRE values (one-year events 635, logistic AUC 0.6971, slope 0.039) predate the label rebuild and must not be used to check the current figures. The audit-figure manifest is `v8_figure_number_manifest.json`. For the manuscript figures of the current run use `v9_number_master.json` (below), which is archived here since v1.4.3 |
| `leakage_sensitivity.json` | **earlier run** | `60_leakage_sensitivity.py` | Sensitivity analyses removing all aetiology diagnosis codes (the strongest of which could plausibly encode events occurring *after* the prediction time point): ΔAUC for the 1-year, in-hospital, survivor-subset and external scenarios, with DeLong tests. *Superseded by `leakage_sensitivity_v9.json` for the current manuscript* |
| `nwicu_vasopressor_audit.json` / `nwicu_vasopressor_forensics.json` / `nwicu_vasopressor_variants.json` | **earlier run** | `74_recover_nwicu_vasopressor.py` | Forensic audit of the NWICU vasopressor variable: the extraction layer had hard-coded a constant 0 while the source database does hold vasopressor prescriptions. Records the recovered prevalence under three route definitions |
| `panel_pixel_repro.json` | **earlier run** | `82_panel_pixel_repro.py` | Pixel-level reproducibility of every panel and composite figure against an independent regeneration |
| `v8_figure_number_manifest.json` | **earlier run** | `66_v8_audit_figures.py` | **The figure-number manifest for the audit figures.** Every number printed inside every audit figure, as a `(figure, item, value, source)` record where `value` is *derived* from `source` — never hand-entered |
| `v8_panel_provenance.json` | **earlier run** | `80_panel_provenance.py` | Provenance of every figure panel, plus number manifests for the internal panels |
| `v9_number_master.json` | current run | `97_v9_vs_v8_diff.py` | **The number master for the figures of this submission.** Every number printed inside every figure of the current run, each entry pointing at the result field it was derived from. This is the file the response letter refers to when it says every figure number is registered machine-readably; the two `*_figure_number_manifest.json` files above cover the audit figures of earlier runs |
| `figure_inventory.json` | current run | `99_prep_v9_package.py` (inventory written by `92_v9_audit_figures.py`) | Which panels each of the composite figures is assembled from, with the panel list and ordering used for this submission |
| `v9_panel_pixel_repro.json` | current run | `96_v9_pixel_repro.py` | Pixel-level reproducibility of every panel and composite figure of the current run against an independent regeneration of the same generator |
| `treatment_rule_check_v1.json` | current run | `125_treatment_rule_check.py` | **The extraction windows and coding rules of the three treatment-intensity indicators, re-derived cohort by cohort.** For each database and each indicator it recomputes six definitions (the value in the analysis file, the rule as coded, earliest start time inside the window, any qualifying record inside the window, overlap with the window, and the wording the manuscript previously used) and reports the pairwise differences in patient counts. Archived since v1.4.4, when the manuscript stopped describing all predictors as measured "inside the first 24 hours of the ICU stay" |
| `treatment_rule_sensitivity_v1.json` | current run | `126_treatment_rule_sensitivity.py` | What the choice of coding rule does to the **frozen** models: for each cohort where a variant exists, the AUC / O:E / calibration slope / Brier / ECE under the rule as coded and under the alternative rule, with the differences. The `as_coded` block must reproduce the published external-validation AUCs digit for digit; the script asserts this against `external_validation_v9.csv` before it reports anything. Requires the three frozen model artefacts written by `118_v10_retrain.py`, which are derived from credentialed data and are therefore not redistributed |
| `freeze_numeric_audit_v13.json` | package audit | `127_freeze_numeric_audit.py` | Read-only audit of the **frozen submission package**: SHA256 and size of the fifteen deliverables, of the eight analysis inputs and of the chain scripts; an independent recomputation of the ten external-validation cohort–outcome combinations (own preparation, prediction and metric code) compared against the numbers **printed in Table 4** rather than against intermediate files; the appendix cells of Tables S3, S5 and S19 and the abstract's Results sentences traced back to the result file that holds each; a character-identity check for every quantity printed more than once; and the abstract / page / table counts. 304 checks in the deposited run: the four recorded as *needs a decision* rather than as passes are listed in the README. Needs the patient-level CSVs, the three frozen model artefacts and the submission package; any layer whose inputs are absent is reported as skipped, not as a failure |

> **Note on `v8_appendix_data.json`**: it is written by the manuscript's appendix builder, which is *not*
> deposited because it embeds the unpublished appendix text. It is therefore **the one current-run file
> this repository does not regenerate**; the file itself is archived verbatim, it is what Appendix
> Table S4 points at, and its numbers are also referenced by several audit scripts here.

> **Note on `sql/11_nwicu_cohort.sql`**: re-running it on a credentialed NWICU copy reproduces the
> cohort the current run used — n = 3,420, 699 one-year deaths, 376 in-hospital deaths, vasopressor
> prevalence 11.1%. The script emits the bounded label columns (`death_hosp`, `death_365`) together
> with the unbounded variants (`death_hosp_unbounded`, `death_365_unbounded`), which are the
> sensitivity analysis, and the legacy `death_90` column, which is kept only for backwards
> compatibility with the archived earlier-run files.

## How to use these files to check the manuscript

1. Take any number from a table in the paper.
2. Find it in the relevant JSON above (the file names map onto the analyses one-to-one). For any
   number that belongs to the **current** analysis run, use the files marked *current run* — the
   earlier-run files still carry the pre-rebuild values for the same quantities.
3. To trace a number that appears **inside an audit figure**, use `v8_figure_number_manifest.json`:
   each entry names the figure, the item, the printed value, and the JSON path it was derived from.
   If the two ever disagree, the manifest is wrong — that is the failure mode these audits were
   written to catch.

## Provenance and reproducibility conventions

* Every script reads its workspace root from the `ABI_BASE` environment variable (falling back to the
  original development path), and results are written to `$ABI_BASE/output/`. This directory is a
  snapshot of that output directory, restricted to the analysis results; the *Vintage* column says
  which run each file is a snapshot of.
* All stochastic steps use fixed seeds, so re-running a script reproduces the same numbers.
* **No intermediate file stores a rounded value.** Rounding happens once, in the presentation layer.
  Storing an already-rounded value and rounding it again produces different answers for values that
  sit on a rounding boundary — the classic example being an AUC of 0.8464648 that becomes 0.8465 and
  then prints as 0.847 instead of 0.846.
* No patient-level identifiers appear anywhere in this directory.
