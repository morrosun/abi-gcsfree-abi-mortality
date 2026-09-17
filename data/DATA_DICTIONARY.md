# Data dictionary — Huaian local validation cohort (de-identified)
- Rows: 239 (all consecutive screened admissions; the manuscript analysis set is a subset, see below)
- Re-identification: no record numbers, no calendar dates (only relative intervals), age top-coded at 90.
- Reproducing the manuscript analysis set: keep `n_etiology_flags >= 1` AND `age >= 18` AND `hospital_discharge_location` not missing -> n=225, in-hospital deaths=55.

| Variable | Definition |
|---|---|
| `case_id` | Sequential de-identified case number (1..N); replaces the hospital record number. |
| `pre_icu_los_days` | Days from hospital admission to ICU admission. |
| `icu_los_days` | ICU length of stay, days (ICU discharge - ICU admission). |
| `hosp_los_days` | Days from ICU admission to hospital discharge. |
| `time_to_death_days` | Days from ICU admission to death; empty if alive/not recorded. |
| `followup_days` | Days from ICU admission to last follow-up; empty if not recorded. |
| `age` | Age at ICU admission, years; values >89 top-coded to 90. |
| `female` | 1 = female, 0 = male. |
| `hospital_discharge_location` | 1=Home, 2=Rehabilitation, 3=SNF/LTC, 4=Hospice, 5=Died in hospital, 6=Transfer, 9=Other/unknown. |
| `death_365` | 1-year all-cause mortality after ICU admission (1=dead, 0=alive). Missing where follow-up incomplete (MNAR - see manuscript Limitations). |
| `death_inhosp` | In-hospital mortality, derived as hospital_discharge_location == 5. |
| `anoxic/ich/sah/ais/seizure/cns_inf/tbi` | Aetiology flags for acute brain injury (1=present). |
| `n_etiology_flags` | Count of positive aetiology flags. |
| `charlson_comorbidity_index` | Charlson comorbidity index. |
| `sofa_day1` | SOFA score on ICU day 1. |
| `gcs` | Glasgow Coma Scale at ICU admission (3-15). |
| `mech_vent/vasopressor/rrt` | Organ support on day 1 (1=yes). |
| `heart_rate_mean/mbp_mean/resp_rate_mean/temperature_mean/spo2_mean` | Day-1 mean vital signs. |
| `wbc_max/hemoglobin_min/platelets_min/sodium_min/potassium_max/creatinine_max/bun_max/glucose_max/bicarbonate_min/inr_max` | Day-1 laboratory extremes. **SI units**: creatinine umol/L, urea mmol/L, glucose mmol/L, haemoglobin g/L. Scripts convert SI -> conventional (MIMIC) units before scoring: creatinine/88.4, urea/0.357, glucose*18, haemoglobin/10. |
