-- =====================================================================
-- A1 external validation | NWICU (Northwestern)  1-year all-cause mortality
-- MIMIC-structured but non-standard itemids; temp in Fahrenheit; no vasopressor source.
-- =====================================================================
DROP TABLE IF EXISTS icu.abi_a1_ext;
CREATE TABLE icu.abi_a1_ext AS
WITH abi_dx AS (   -- ABI aetiology (NWICU is ICD-10 only)
  SELECT hadm_id,
    max(case when icd_code ~ '^S06' then 1 else 0 end) tbi,
    max(case when icd_code ~ '^I60' then 1 else 0 end) sah,
    max(case when icd_code ~ '^I61' then 1 else 0 end) ich,
    max(case when icd_code ~ '^I63' then 1 else 0 end) ais,
    max(case when icd_code ~ '^G0[0-5]' then 1 else 0 end) cns_inf,
    max(case when icd_code ~ '^G4[01]' then 1 else 0 end) seizure,
    max(case when icd_code ~ '^G931' then 1 else 0 end) anoxic
  FROM hosp.diagnoses_icd GROUP BY hadm_id
),
abi_hadm AS (SELECT * FROM abi_dx WHERE (tbi+sah+ich+ais+cns_inf+seizure+anoxic)>0),
first_icu AS (
  SELECT DISTINCT ON (i.subject_id) i.subject_id,i.hadm_id,i.stay_id,i.intime,i.outtime
  FROM icu.icustays i JOIN abi_hadm a ON a.hadm_id=i.hadm_id
  ORDER BY i.subject_id, i.intime
),
base AS (
  SELECT f.subject_id,f.hadm_id,f.stay_id,f.intime,f.outtime,
    p.gender,p.dod,p.anchor_age,
    (p.anchor_age + (extract(year from f.intime)-p.anchor_year))::int as age,
    a.dischtime, a.hospital_expire_flag, a.discharge_location,
    d.tbi,d.sah,d.ich,d.ais,d.cns_inf,d.seizure,d.anoxic
  FROM first_icu f
  JOIN hosp.patients p ON p.subject_id=f.subject_id
  JOIN hosp.admissions a ON a.hadm_id=f.hadm_id
  JOIN abi_hadm d ON d.hadm_id=f.hadm_id
),
vit AS (   -- first-day vitals; MBP=(SBP+2*DBP)/3; temp F->C
  SELECT b.stay_id,
    avg(case when ce.itemid=320045 and ce.valuenum between 20 and 250 then ce.valuenum end) hr,
    avg(case when ce.itemid=320210 and ce.valuenum between 4 and 60 then ce.valuenum end) rr,
    avg(case when ce.itemid=320277 and ce.valuenum between 40 and 100 then ce.valuenum end) spo2,
    (avg(case when ce.itemid=323761 and ce.valuenum between 90 and 110 then ce.valuenum end)-32)*5.0/9 temp_c,
    avg(case when ce.itemid=320179 and ce.valuenum between 40 and 300 then ce.valuenum end) sbp,
    avg(case when ce.itemid=320180 and ce.valuenum between 10 and 200 then ce.valuenum end) dbp
  FROM base b JOIN icu.chartevents ce ON ce.stay_id=b.stay_id
    AND ce.charttime BETWEEN b.intime AND b.intime + interval '1 day'
  GROUP BY b.stay_id
),
lab AS (   -- first-day labs via hadm_id + time window (no bicarbonate item in NWICU)
  SELECT b.hadm_id,
    max(case when le.itemid=100016 then le.valuenum end) wbc_max,
    min(case when le.itemid=100007 then le.valuenum end) hemoglobin_min,
    min(case when le.itemid=100014 then le.valuenum end) platelets_min,
    min(case when le.itemid=100010 then le.valuenum end) sodium_min,
    max(case when le.itemid=100011 then le.valuenum end) potassium_max,
    max(case when le.itemid=100002 then le.valuenum end) creatinine_max,
    max(case when le.itemid=100004 then le.valuenum end) bun_max,
    max(case when le.itemid in (100001,100062,100045) then le.valuenum end) glucose_max,
    max(case when le.itemid=100030 then le.valuenum end) inr_max
  FROM base b JOIN hosp.labevents le ON le.hadm_id=b.hadm_id
    AND le.charttime BETWEEN b.intime - interval '6 hours' AND b.intime + interval '1 day'
  GROUP BY b.hadm_id
),
vent AS (SELECT DISTINCT b.stay_id FROM base b JOIN icu.procedureevents pe ON pe.stay_id=b.stay_id
         WHERE pe.itemid IN (787541,753461)),
rrt AS (SELECT DISTINCT b.stay_id FROM base b JOIN icu.procedureevents pe ON pe.stay_id=b.stay_id
        WHERE pe.itemid IN (704890,772042,724671,798351))
SELECT b.*,
  CASE WHEN b.dod IS NOT NULL AND b.dod <= (b.intime + interval '365 days') THEN 1 ELSE 0 END as death_365,
  CASE WHEN b.dod IS NOT NULL AND b.dod <= (b.intime + interval '90 days') THEN 1 ELSE 0 END as death_90,
  b.hospital_expire_flag as death_hosp,
  v.hr as heart_rate_mean,
  case when v.sbp is not null and v.dbp is not null then (v.sbp+2*v.dbp)/3.0 end as mbp_mean,
  v.rr as resp_rate_mean, v.temp_c as temperature_mean, v.spo2 as spo2_mean,
  l.wbc_max,l.hemoglobin_min,l.platelets_min,l.sodium_min,l.potassium_max,
  l.creatinine_max,l.bun_max,l.glucose_max, NULL::numeric as bicarbonate_min, l.inr_max,
  case when vent.stay_id is not null then 1 else 0 end as mech_vent,
  0 as vasopressor,   -- NWICU has no vasopressor/infusion source
  case when rrt.stay_id is not null then 1 else 0 end as rrt
FROM base b
LEFT JOIN vit v ON v.stay_id=b.stay_id
LEFT JOIN lab l ON l.hadm_id=b.hadm_id
LEFT JOIN vent ON vent.stay_id=b.stay_id
LEFT JOIN rrt ON rrt.stay_id=b.stay_id
WHERE b.age >= 18;
