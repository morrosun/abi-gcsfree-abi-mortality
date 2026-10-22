# Analysis code and de-identified local cohort for a GCS-free long-term mortality model in critically ill patients with acute brain injury

Companion repository for the manuscript:

> **A transportable, Glasgow Coma Scale–free model for predicting long-term mortality in critically ill
> patients with acute brain injury: development and multi-scenario external validation across four
> international databases and an independent single-centre cohort**

The deposit title above (the one recorded on Zenodo) names the contents rather than repeating the
manuscript title, which is quoted in full in the block above.

The model is deliberately built **without the Glasgow Coma Scale (GCS)**, from routine variables that can be
harmonised across heterogeneous ICU databases, so that a genuine
*development → external-validation* loop can be completed across five cohorts — one development cohort
and four validation cohorts, in three countries.

---

## Repository contents

```
sql/                                  Cohort extraction (PostgreSQL, local credentialed copies)
  01_mimic_cohort.sql                 MIMIC-IV   (development cohort)
  10_eicu_cohort.sql                  eICU-CRD
  11_nwicu_cohort.sql                 Northwestern ICU (NWICU)
  12_inspire_cohort.sql               INSPIRE
scripts/                              Python analysis pipeline
  ---- development and first validation round ------------------------------
  02_table1_missing.py                Cohort description + missingness
  03_model_dev.py                     Model development (28 predictors, logistic + XGBoost)
  04_figures.py                       Development-cohort figures
  05_external_validation.py           External validation in eICU / NWICU / INSPIRE
  06_external_figures.py              External-validation figures
  07_recalibration.py                 Recalibration analyses
  08_recal_figures.py                 Recalibration figures
  12_revision_figures.py              Additional figures (revision round 1)
  13_revision_figures2.py             Additional figures (revision round 1, part 2)
  16_revision2_analyses.py            Additional analyses (revision round 2)
  20_local_qc.py                      Local single-centre cohort: data quality control
  22_missingness_diag.py              Local cohort: missingness diagnostics
  23_inspect_inhosp_model.py          Inspect the frozen in-hospital model object
  24_local_inhosp_validation.py       **Local single-centre external validation** (in-hospital model)
  25_local_part1_stats.py             Local validation statistics + sensitivity analyses 1–4
  27_part1_figures.py                 Local validation figures
  ---- transportability / harmonisation audits ------------------------------
  58_gcs_increment_audit.py           Does adding the GCS change anything? (ΔAUC, DeLong, NRI, IDI)
  59_rebuild_external_figures.py      Rebuild every external-validation figure **from the current results**
  60_leakage_sensitivity.py           Aetiology-code leakage sensitivity analysis
  61_extract_severity_inputs.py       Extract the inputs needed for SOFA / OASIS / SAPS-II
  62_calib_ci_and_severity.py         Bootstrap CIs for calibration metrics + severity head-to-head
  63_check_slope_epsilon.py           Calibration slope as a function of the clipping constant ε
  66_v8_audit_figures.py              Audit figures + the figure-number manifest
  68_internal_learners.py             Internal learner comparison
  69_cohort_descriptives.py           Cohort descriptives
  74_recover_nwicu_vasopressor.py     Forensic audit + recovery of the NWICU vasopressor variable
  91_leakage_sensitivity_v9.py        Leakage-sensitivity grid for the **current** analysis run (adds the joint
                                       deletion of the seven aetiology indicators **with the Charlson index**)
  ---- input-generating chain of the current analysis run (v1.4.3) ---------
  84_death_label_crosscheck.py       Re-derive and cross-check the death labels of every database
  85_treatment_window_audit.py       Treatment-window audit; writes the NWICU cohort file the rebuild reads
  86_rebuild_datasets.py             **Rebuild the four analysis datasets** on the first-day treatment window
                                     and the corrected labels (writes `data/*_v2.csv`)
  118_v10_retrain.py                 **Retrain** the frozen models on the rebuilt data; re-selects the tree
                                     hyper-parameters inside the current training split; window-exclusion
                                     sensitivity
  102_nwicu_source_reaggregation.py  NWICU source-layer re-aggregation (the main external analysis) + the
                                     sentinel accounting behind Appendix S10
  88_eta_calibration_audit.py        Calibration on the unclipped linear predictor (η)
  89_sentinel_clean_and_reeval.py    Placeholder-cleaned variant used as the sensitivity analysis
  90_downstream_v9_severity_gcs.py   Severity-score head-to-head and the GCS increment
  92_v9_audit_figures.py             Audit figures, harmonisation table and their number records
  93_slope_epsilon_v9.py             Calibration slope as a function of the clipping constant
  94_v9_main_figures.py              Main figures of the current run + the number master
  95_v9_external_figures.py          External-validation figures of the current run
  96_v9_pixel_repro.py               Pixel-level reproducibility of the current-run figures
  97_v9_vs_v8_diff.py                Cell-by-cell diff against the previous run
  108_split_overlap_sensitivity.py   No-overlap subset analysis (Appendix S11)
  124_death_label_bounds_primary.py  Bounded-label primary analysis and its sensitivity grid
  98_build_v9_results.py              Last step of the **current** analysis run: translates that run's
                                       `*_v2` / `*_v9` outputs into the result schema the rest of the pipeline
                                       reads, and writes the current-run files in `results/`
  79_extract_official_scores.py       Extract the **official** OASIS / SAPS-II / SOFA scores from mimiciv_derived
  80_panel_provenance.py              Provenance of every figure panel + number manifests for the internal panels
  81_official_score_provenance.py     Provenance of the official severity-score tables (equivalence, not a commit)
  82_panel_pixel_repro.py             Pixel-level reproducibility of panels/composites vs an independent regeneration
results/                              Machine-readable output of the audit scripts (+ INDEX.md)
data/
  huaian_local_cohort_deidentified.csv   De-identified local validation cohort (n = 239 rows)
  DATA_DICTIONARY.md                     Variable definitions and units
```

`results/INDEX.md` maps every result file to the script that produced it and to the table it supports.

The leakage-sensitivity grid for the **current** analysis run — the five feature sets and three scenarios
of `leakage_sensitivity.json`, plus the joint deletion of the seven aetiology indicators **with the Charlson
comorbidity index** (28 → 20 predictors) — is archived as **`results/leakage_sensitivity_v9.json`**
(produced by `scripts/91_leakage_sensitivity_v9.py`) and indexed in `results/INDEX.md`. The manuscript
summarizes only its internal one-year cell; the remaining scenarios and the external cohorts are in that
file.

`results/` holds **two vintages side by side**: the files of the current analysis run, and the files of the
earlier run it superseded. The current run rebuilt the NWICU outcome labels (three-source union with a lower
time bound: 699 one-year and 376 in-hospital deaths, where the earlier snapshots carried 635 and 16),
re-aggregated the external validation at the source layer as the main analysis, re-selected the tree
learners' hyper-parameters on the current training split, and re-ran the vasopressor recovery.
`results/INDEX.md` marks every file with its vintage and spells out each change — read it before comparing
any archived number against the manuscript.

## Data availability and access

| Dataset | Source | Redistributable here? |
|---|---|---|
| MIMIC-IV (v3.x) | PhysioNet, credentialed access | **No** — run `sql/01_mimic_cohort.sql` on your own approved copy |
| eICU-CRD | PhysioNet, credentialed access | **No** — `sql/10_eicu_cohort.sql` |
| NWICU | PhysioNet, `10.13026/s84w-1829` | **No** — `sql/11_nwicu_cohort.sql` |
| INSPIRE | PhysioNet, credentialed access | **No** — `sql/12_inspire_cohort.sql` |
| Huaian local cohort | The Affiliated Huaian No.1 People's Hospital | **Yes, de-identified** — `data/huaian_local_cohort_deidentified.csv` |

PhysioNet data-use agreements forbid redistribution of the source data **and of objects derived from it**
(e.g. the extracted cohort CSVs, feature medians, and the trained model files). Those are therefore **not**
included.

**Both vintages of `results/` are re-derivable from this repository.** Since **v1.4.3** the scripts that
*produce the inputs* of the current analysis run are archived alongside the scripts that consume them, so
the chain `SQL → 84/85 → 86 → 118 → 102 → 88/89/90 → 92/93/94/95 → 96 → 97 → 98 → results/` can be executed
end to end against your own credentialed copy of each database. The run order and the inputs each step
expects are given under "Running the pipeline" below. The **single exception is
`results/v8_appendix_data.json`**: it is written by the manuscript's appendix builder, which is not
deposited because it embeds the unpublished appendix text. The file itself is archived, it is what
Appendix Table S4 points at, and `results/INDEX.md` says so.

`results/INDEX.md` labels every file with its vintage and states what each change was, so an archived
number can never be mistaken for a current one.

### De-identification of the local cohort

The shared local-cohort file was de-identified before release:

1. The hospital record number (`patient_id`) was **removed** and replaced by a sequential `case_id`.
2. All **absolute timestamps** (hospital admission, ICU admission/discharge, hospital discharge, death,
   last follow-up) were **removed** and replaced by intervals in days relative to ICU admission
   (`pre_icu_los_days`, `icu_los_days`, `hosp_los_days`, `time_to_death_days`, `followup_days`).
3. Age was **top-coded**: values > 89 years are recorded as 90.
4. No names, contact details, admission numbers, free-text notes or calendar dates are retained.

Applying the manuscript's inclusion rule (`n_etiology_flags >= 1`, `age >= 18`, discharge disposition known)
to this file yields **n = 225 with 55 in-hospital deaths**, i.e. the analysis set reported in the paper.

*Note:* `death_365` (one-year mortality) is **incomplete by design** — it is missing for a substantial
fraction of the cohort, and the mechanism behind that incompleteness could not be established from the
released variables, so it is released as-is with no imputation. The manuscript's one-year analyses do
**not** use this local cohort.

## Running the pipeline

```bash
export ABI_BASE="/path/to/your/analysis/workspace"   # must contain sql/, data/, output/
python scripts/02_table1_missing.py
python scripts/03_model_dev.py                        # fits and saves the models
python scripts/04_figures.py
python scripts/05_external_validation.py
python scripts/06_external_figures.py
python scripts/07_recalibration.py
python scripts/08_recal_figures.py
python scripts/16_revision2_analyses.py
python scripts/12_revision_figures.py
python scripts/13_revision_figures2.py
# independent single-centre (Huaian) validation
python scripts/20_local_qc.py
python scripts/22_missingness_diag.py
python scripts/24_local_inhosp_validation.py
python scripts/25_local_part1_stats.py
python scripts/27_part1_figures.py
# ---- transportability / harmonisation audits -------------------------------
# 74 MUST run first: it writes results/nwicu_vasopressor_recovery.json, which
# 58 and 62 read. An earlier revision of this README listed it last, which
# left those two scripts running without the recovered variable.
python scripts/74_recover_nwicu_vasopressor.py        # needs PostgreSQL
# 79 pulls the official severity scores out of the database into
# data/mimic_severity_official.csv, which 62 then reads — so it precedes 62.
python scripts/79_extract_official_scores.py          # needs PostgreSQL
python scripts/61_extract_severity_inputs.py          # needs PostgreSQL
python scripts/58_gcs_increment_audit.py
python scripts/60_leakage_sensitivity.py
python scripts/91_leakage_sensitivity_v9.py           # writes results/leakage_sensitivity_v9.json
# 98 is the last step of the *current* analysis run. It translates that run's `*_v2` / `*_v9`
# outputs into the result schema the rest of the pipeline reads, and writes the current-run
# files in results/. Its inputs are produced by the chain listed under "Current analysis run"
# below, which is archived in this repository since v1.4.3.
python scripts/98_build_v9_results.py
python scripts/62_calib_ci_and_severity.py            # reads the CSV written by 79
python scripts/63_check_slope_epsilon.py
python scripts/59_rebuild_external_figures.py         # writes results/external_current.json
python scripts/66_v8_audit_figures.py                 # writes the figure-number manifest
python scripts/68_internal_learners.py
python scripts/69_cohort_descriptives.py
# 81 checks the provenance of the official severity-score tables (needs PostgreSQL).
# It records that the local database carries NO build metadata, and therefore proves
# equivalence to the official formulas row by row instead of naming a commit.
python scripts/81_official_score_provenance.py        # needs PostgreSQL
# 80 needs the figure inventory written by 66, and it consumes the pixel-level
# reproducibility report produced by 82, so it runs after 82.
python scripts/80_panel_provenance.py
# 82 is a *verification* step, not a generator: it compares each panel and each of the
# eight composite figures against the version produced by an INDEPENDENT regeneration of
# the same generator. Procedure: copy the artefacts aside, re-run 04 / 12 / 59 / 66,
# then run 82. It refuses to report a pass if the backup is missing.
python scripts/82_panel_pixel_repro.py
```

### Current analysis run — exact order

This is the chain that regenerates every file `results/INDEX.md` marks *current run* from a credentialed copy
of each database. Each step names what it reads, because the two most common ways to get a wrong answer here
are to run a step before its input exists, or to run it against the previous run's file.

```bash
# 1–2  labels and treatment windows (PostgreSQL). 85 writes the NWICU cohort file that 86 reads.
python scripts/84_death_label_crosscheck.py           # needs PostgreSQL
python scripts/85_treatment_window_audit.py           # needs PostgreSQL
# 3    rebuild all four analysis datasets on the first-day window + corrected labels
#      -> data/mimic_abi_cohort_v2.csv, eicu_abi_final_v2.csv, inspire_abi_final_v2.csv
python scripts/86_rebuild_datasets.py                 # needs PostgreSQL
# 4    retrain: frozen models refitted, tree hyper-parameters re-selected inside the current
#      training split, window-exclusion sensitivity  -> calibration_ci_and_severity_v2.json
python scripts/118_v10_retrain.py                     # needs PostgreSQL
# 5    NWICU source-layer re-aggregation, which is the main external analysis, plus the
#      sentinel accounting behind Appendix S10      -> nwicu_abi_final_v5.csv
python scripts/102_nwicu_source_reaggregation.py      # needs PostgreSQL
# 6–8  evaluation layers
python scripts/88_eta_calibration_audit.py            # calibration on the unclipped margin
python scripts/89_sentinel_clean_and_reeval.py        # placeholder-cleaned sensitivity variant
python scripts/90_downstream_v9_severity_gcs.py       # severity head-to-head + GCS increment
python scripts/91_leakage_sensitivity_v9.py           # leakage-sensitivity grid
python scripts/108_split_overlap_sensitivity.py       # no-overlap subset (Appendix S11)
python scripts/124_death_label_bounds_primary.py      # bounded-label primary + sensitivity grid
# 9–13 figures of this run, their number master, and the pixel check
python scripts/92_v9_audit_figures.py                 # audit figures + harmonisation table
python scripts/93_slope_epsilon_v9.py                 # slope vs clipping constant
python scripts/94_v9_main_figures.py                  # main figures + results/v9_number_master.json
python scripts/95_v9_external_figures.py              # external-validation figures
python scripts/96_v9_pixel_repro.py                   # pixel-level reproducibility
python scripts/97_v9_vs_v8_diff.py                    # cell-by-cell diff against the previous run
# 14   translate into the result schema and write results/
python scripts/98_build_v9_results.py
```

`results/v9_number_master.json` — the machine-readable record of every number printed inside every figure of
this run, each entry pointing at the result field it was derived from — is written by step 12 and archived
here. `results/v8_appendix_data.json` is the one current-run file this chain does not produce: it is written
by the manuscript's appendix builder, which is not deposited.

Two things worth knowing before you run the audits:

- `58` reads the de-identified local cohort from **`data/huaian_local_cohort_deidentified.csv`**,
  which is published here. The two inputs were checked to be equivalent item by item
  (n=225, 55 events, 24.4%, and every aetiology count identical). It only falls back to a
  local, non-public spreadsheet if that CSV is absent.
- `62` computes the SOFA / OASIS / SAPS-II head-to-head from the **official** `mimiciv_derived`
  tables (`first_day_sofa`, `oasis`, `sapsii`) rather than from a re-implementation. A
  hand-written version is still in the script for comparison; it assigns 0 points where the
  official score assigns 4 (heart rate < 33: 86 of the 16,597 included stays, 389 of all
  94,458 stays in the table) and misses the high-systolic-blood-pressure branch entirely
  (347 of the included stays, 1,186 of all stays), so it is **not** used for the reported
  numbers. Counts are given both within the included cohort and table-wide because the two
  differ by more than four-fold and the scope matters when reproducing them.

### How the provenance claims are verified

Two claims in the accompanying manuscript are about provenance rather than about results, and
each is checked by a dedicated script so that a reader can re-run the check:

- **The figures come from the current analysis, not from an earlier run.** Composite figures
  are assembled from PNG panels, so a composite is always *newer* than the panels it was built
  from even when those panels are stale — a timestamp proves nothing. `82` therefore compares
  every panel and every composite against the output of an independent regeneration, at the
  level of **decoded pixels** (and records a pixel hash that is independent of PNG metadata).
  In the reference run, 26 of 27 compared files were pixel-identical; the single exception,
  `figR2_inhosp_parallel.png`, is written by two scripts from different data vintages, is
  documented as such, and is not used by any of the eight composite figures.
  `80` restricts its verdict to the 15 panels the figures actually use and emits a
  per-file table (`output/_panel_repro_table.md`) plus an explicit count identity, so the
  "panels used" and "panels tested" sets cannot be confused with each other.
- **The severity scores come from the official pipeline.** `81` first records that the local
  PostgreSQL database stores no DDL provenance at all (no table comments, no `reloptions`, no
  metadata table), so *the build commit cannot be recovered from the database* and is not
  claimed. It then establishes equivalence instead: the published total is exactly
  `SUM(COALESCE(component, 0))` and the published probability exactly follows the official
  logistic formula (SAPS-II includes its `ln` term) for all 94,458 rows of both tables, and the
  two branch behaviours that a re-implementation typically gets wrong are confirmed. The
  upstream revision compared against is pinned by blob SHA in
  `results/official_score_provenance.json`.


Every script reads its workspace root from the `ABI_BASE` environment variable; if the variable is unset the
original development path is used as a fallback. Extracted cohort CSVs are expected under `$ABI_BASE/data/`
and results are written to `$ABI_BASE/output/`.

Database credentials are **not** stored in the code. Scripts that query PostgreSQL read them from the
environment, falling back to an empty password:

```bash
export ABI_DB_HOST="localhost"  ABI_DB_PORT="5432"
export ABI_DB_USER="postgres"   ABI_DB_PASSWORD="<your password>"
```

### Environment

Python 3.13.14 with: numpy 2.4.6, pandas 2.3.3, scikit-learn 1.9.0, xgboost 3.3.0, matplotlib 3.11.0,
joblib 1.5.3, scipy 1.18.0, statsmodels 0.14.6, lifelines 0.30.3, openpyxl 3.1.5.
Cohort extraction requires PostgreSQL with local, credentialed copies of the four databases.

## ⚠️ Two conventions that silently break external validation

Reproducing the external validations requires matching two conventions that are easy to get wrong and that
produce plausible-looking but wrong numbers:

1. **Units.** The local cohort reports laboratory values in **SI units**, whereas the MIMIC-trained model
   expects **conventional units**. Convert *before* scoring:
   creatinine ÷ 88.4, urea nitrogen ÷ 0.357, glucose × 18, haemoglobin ÷ 10.
2. **Feature scaling.** The two learners in the frozen model were fitted on *different* inputs:
   the **logistic regression expects standardised features**, the **XGBoost expects raw features**.
   Feeding the wrong one is not harmless — in the single-centre cohort it moved the logistic AUC from
   0.869 to 0.567 (ΔAUC −0.301) and the XGBoost AUC from 0.826 to 0.448 (ΔAUC −0.378), both P<.001
   by DeLong. Both numbers are in `results/v8_harmonisation.json`, under `variants.scale_discordant`
   and `delta.scale_discordant`, which is also where the audit figure is drawn from.

## ⚠️ A third convention: the clipping constant behind any probability-logit calibration slope

A calibration slope is fitted on the log-odds scale, so a predicted probability of exactly 0 or 1 has
to be handled by an explicit convention, and that convention is not free. In the single-centre cohort
the logistic slope computed on the ε-clipped probability path ranges from **0.9399 to 1.1885** across
ε ∈ {1e-6, 1e-4, 1e-3, 1e-2} — a spread of 0.2486 in the smallest cohort — while the XGBoost slopes
move by at most 0.0084 in any cohort. **The primary calibration slopes reported in the manuscript are
therefore fitted on the unclipped linear predictor** (`logit P = α + βm`, with `m` the logistic
model's decision function or the raw XGBoost margin), and the ε-grid is reported as a sensitivity
analysis, because an estimate that moves with an arbitrary constant is a property of the convention
rather than of the model. `scripts/63_check_slope_epsilon.py` reproduces the whole grid and writes
`results/slope_epsilon_audit.json`, so the sensitivity can be checked rather than taken on trust;
that file also carries the unclipped `slope_eta` beside the per-ε values.

## What this repository deliberately does not contain

Two things are left out on purpose, and neither is an oversight:

1. **Patient-level source data and objects derived from it.** The extracted cohort CSVs, the feature
   medians and the trained model objects are all covered by the PhysioNet data-use agreements and cannot be
   redistributed. Trained model files (`*.joblib`, `*.pkl`) are excluded by `.gitignore`.
2. **The manuscript and appendix production scripts.** Those files contain the full text of a manuscript
   that is under review, so publishing them would amount to posting the manuscript. They are therefore not
   part of this repository.

If you need anything in the second category in order to check a specific claim, please contact the
corresponding author and it will be provided.

## How to check that the figures match the tables

The failure mode this repository is designed to make impossible is a figure annotated with numbers from an
earlier analysis run. Two manifests make that checkable:

* `results/figure_number_manifest.json` — every number printed in the external-validation figures, with the
  key in the run's `external_current.json` it was read from. **V8-legacy snapshot**: its values predate the
  current run (NWICU one-year events 635, logistic AUC 0.6971) and are kept for the archived V8 figures only;
  the current figure numbers are not tracked by any file in this repository.
* `results/v8_figure_number_manifest.json` — the same idea for the audit figures.

In both files, `value` is **derived** from `source` at build time; no value is typed in by hand. So a
manifest entry can always be recomputed from the result files in this repository, and any discrepancy
between a printed figure number and a table number is mechanically detectable. The scripts also abort if any
figure is older than the result files it was drawn from.

## License

Code: **MIT** (see `LICENSE`). The de-identified dataset is released under
**CC BY 4.0**; please cite the version DOI below when re-using it.

## Citation / DOI

Archived on Zenodo — cite the **version-specific DOI** of the release you used:

> Pan Q, Zhu X, Li C, Wang K. *Analysis code and de-identified local validation cohort for a GCS-free
> long-term mortality model in critically ill patients with acute brain injury* (v1.4.2).
> Zenodo, 2026. **doi: [10.5281/zenodo.23535852](https://doi.org/10.5281/zenodo.23535852)**

- v1.4.2 (current): `10.5281/zenodo.23535852` — corrects three statements in this README that did not
  match the analysis files they describe, and adds no new analysis. (i) The feature-scaling example now
  quotes the `results/v8_harmonisation.json` values that the manuscript reports — logistic AUC
  0.869 → 0.567 and XGBoost 0.826 → 0.448, ΔAUC −0.301 and −0.378 — in place of an illustration that
  mixed an AUC with a calibration slope and could not be reproduced from any file in this repository.
  (ii) The clipping-constant section now states that the primary calibration slopes are fitted on the
  **unclipped linear predictor** and that the ε-grid is the sensitivity analysis, which is what the
  manuscript reports; the ε values quoted are the four ε actually present in
  `results/slope_epsilon_audit.json`. (iii) The note on the local one-year outcome no longer asserts a
  missing-not-at-random mechanism, which the released variables cannot establish. The deposit title
  recorded on Zenodo names the contents rather than repeating the manuscript title, and this is now
  stated explicitly.
- v1.4.1 (previous): `10.5281/zenodo.23485896` — replaces the stale 2026-10-02 result snapshot with the
  **current analysis run**, and makes the two vintages explicit. Thirteen result files are re-sourced
  from that run (NWICU one-year events 699 and in-hospital 376 against 635 and 16; logistic AUC
  0.7652, calibration slope 0.872 and O:E 0.994 against 0.6971 / 0.039 / 0.535), because it rebuilt
  the NWICU outcome labels as the union of in-stay death time, discharge disposition and the
  hospital-expiry flag with a lower time bound, re-aggregated the external validation at the source
  layer as the main analysis, and re-selected the tree learners' hyper-parameters on the current
  training split. Adds `scripts/98_build_v9_results.py`, the last step of that run, rewrites
  `sql/11_nwicu_cohort.sql` so the cohort SQL itself reproduces the labels and the recovered
  vasopressor variable (11.1%), and adds a "Snapshot version" section plus a vintage column to
  `results/INDEX.md`
- v1.4.0 (previous): `10.5281/zenodo.23240062` — adds the leakage-sensitivity grid of the **current**
  analysis run (`scripts/91_leakage_sensitivity_v9.py`, `results/leakage_sensitivity_v9.json`), which
  extends the five feature sets and three label-timing scenarios with the joint deletion of the seven
  aetiology indicators **together with the Charlson index** (28 → 20 predictors), refits the reduced
  models in MIMIC-IV and applies them without refitting to the external cohorts, and records the paired
  DeLong comparison for every scenario × database × learner cell; also syncs four files that had drifted
  behind the local source (the versioned `figure_inventory_V8.json` name and the archived status of the
  unversioned figure-number manifest)
- v1.3.0 (previous): `10.5281/zenodo.23112164` — adds pixel-level reproducibility evidence for every
  panel and all eight composite figures (`results/panel_pixel_repro.json`), provenance for the
  official SOFA / OASIS / SAPS-II tables with the explicit statement that the local build commit
  cannot be recovered (`results/official_score_provenance.json`), and a comparability assertion
  for the two reported GCS AUC increments (`results/gcs_increment_audit.json`, key `comparability`)
- v1.2.0 (previous): `10.5281/zenodo.23092310` — recomputes the SOFA / OASIS / SAPS-II head-to-head
  from the official `mimiciv_derived` tables (OASIS-alone AUC 0.718 → 0.720, SAPS-II-alone
  0.785 → 0.780), adds panel provenance and number manifests for the internal panels, removes the
  remaining intermediate rounding, and makes the single-centre analysis reproducible from the
  published de-identified cohort
- v1.1.0 (previous): `10.5281/zenodo.23077740` — adds the transportability/harmonisation audit layer,
  the machine-readable `results/` directory, and the figure-number manifests
- v1.0.1 (previous): `10.5281/zenodo.22820024` — development and first validation round only
- Concept DOI (always resolves to the latest version): `10.5281/zenodo.22820023`
- Repository: <https://github.com/morrosun/abi-gcsfree-abi-mortality>

> **This README is a living document.** Each Zenodo archive is a frozen snapshot taken at its release tag;
> a version-specific DOI points at that snapshot, not at this file. Release metadata
> (title, author list, keywords, licence) is declared in `.zenodo.json`.
