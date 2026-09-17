-- =====================================================================
-- A1 | MIMIC-IV ABI development cohort + GCS-independent predictors
-- Index time = first ICU admission; primary outcome = 365-day all-cause death
-- =====================================================================

DROP TABLE IF EXISTS mimiciv_derived.abi_a1_cohort;
CREATE TABLE mimiciv_derived.abi_a1_cohort AS
WITH abi_dx AS (   -- ABI aetiology from hospital diagnoses (any position)
  SELECT hadm_id,
         max(case when icd_code ~ '^(S06|8[0-4][0-9][0-9]|850|851|852|853|854)' then 1 else 0 end) as tbi,
         max(case when icd_code ~ '^(I60|430)' then 1 else 0 end) as sah,
         max(case when icd_code ~ '^(I61|431)' then 1 else 0 end) as ich,
         max(case when icd_code ~ '^(I63|43[34])' then 1 else 0 end) as ais,
         max(case when icd_code ~ '^(G0[0-5]|32[0-4])' then 1 else 0 end) as cns_inf,
         max(case when icd_code ~ '^(G4[01]|345)' then 1 else 0 end) as seizure,
         max(case when icd_code ~ '^(G931|3481)' then 1 else 0 end) as anoxic
  FROM mimiciv_hosp.diagnoses_icd
  GROUP BY hadm_id
),
abi_hadm AS (
  SELECT * FROM abi_dx
  WHERE (tbi+sah+ich+ais+cns_inf+seizure+anoxic) > 0
),
first_icu AS (   -- first ICU stay per subject
  SELECT DISTINCT ON (i.subject_id)
         i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.los
  FROM mimiciv_icu.icustays i
  JOIN abi_hadm a ON a.hadm_id = i.hadm_id
  ORDER BY i.subject_id, i.intime
),
base AS (
  SELECT f.subject_id, f.hadm_id, f.stay_id, f.intime, f.outtime, f.los as icu_los_days,
         p.gender, p.dod, p.anchor_age,
         (p.anchor_age + (extract(year from f.intime) - p.anchor_year))::int as age,
         adm.admittime, adm.dischtime, adm.hospital_expire_flag,
         adm.discharge_location, adm.admission_type, adm.race, adm.insurance,
         d.tbi, d.sah, d.ich, d.ais, d.cns_inf, d.seizure, d.anoxic
  FROM first_icu f
  JOIN mimiciv_hosp.patients p   ON p.subject_id = f.subject_id
  JOIN mimiciv_hosp.admissions adm ON adm.hadm_id = f.hadm_id
  JOIN abi_hadm d ON d.hadm_id = f.hadm_id
)
SELECT b.*,
  -- ===== outcomes =====
  CASE WHEN b.dod IS NOT NULL AND b.dod <= (b.intime + interval '365 days') THEN 1 ELSE 0 END as death_365,
  CASE WHEN b.dod IS NOT NULL AND b.dod <= (b.intime + interval '90 days')  THEN 1 ELSE 0 END as death_90,
  b.hospital_expire_flag as death_hosp,
  -- survival time (days) censored at 365
  LEAST(365, GREATEST(0, EXTRACT(epoch FROM (COALESCE(b.dod, b.intime + interval '400 days') - b.intime))/86400))::numeric(6,1) as time_365,
  -- ===== comorbidity =====
  ch.charlson_comorbidity_index,
  -- ===== first-day vitals =====
  v.heart_rate_mean, v.sbp_mean, v.dbp_mean, v.mbp_mean, v.resp_rate_mean,
  v.temperature_mean, v.spo2_mean,
  -- ===== first-day labs =====
  l.wbc_max, l.hemoglobin_min, l.platelets_min, l.sodium_min, l.potassium_max,
  l.creatinine_max, l.bun_max, l.glucose_max, l.albumin_min, l.bicarbonate_min,
  l.inr_max,
  -- ===== first-day bg =====
  bg.lactate_max, bg.ph_min,
  -- ===== organ support (first day) =====
  CASE WHEN vent.stay_id IS NOT NULL THEN 1 ELSE 0 END as mech_vent,
  CASE WHEN vaso.stay_id IS NOT NULL THEN 1 ELSE 0 END as vasopressor,
  CASE WHEN rrt.stay_id  IS NOT NULL THEN 1 ELSE 0 END as rrt,
  -- ===== severity (reference only, not model input) =====
  sofa.sofa as sofa_day1
FROM base b
LEFT JOIN mimiciv_derived.charlson ch ON ch.hadm_id = b.hadm_id
LEFT JOIN mimiciv_derived.first_day_vitalsign v ON v.stay_id = b.stay_id
LEFT JOIN mimiciv_derived.first_day_lab l ON l.stay_id = b.stay_id
LEFT JOIN mimiciv_derived.first_day_bg bg ON bg.stay_id = b.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa sofa ON sofa.stay_id = b.stay_id
LEFT JOIN (SELECT DISTINCT stay_id FROM mimiciv_derived.ventilation
           WHERE ventilation_status IN ('InvasiveVent','Tracheostomy')) vent ON vent.stay_id = b.stay_id
LEFT JOIN (SELECT DISTINCT stay_id FROM mimiciv_derived.vasoactive_agent
           WHERE norepinephrine IS NOT NULL OR epinephrine IS NOT NULL
              OR dopamine IS NOT NULL OR phenylephrine IS NOT NULL
              OR vasopressin IS NOT NULL) vaso ON vaso.stay_id = b.stay_id
LEFT JOIN (SELECT DISTINCT stay_id FROM mimiciv_derived.first_day_rrt WHERE dialysis_present=1) rrt ON rrt.stay_id = b.stay_id
WHERE b.age >= 18;
