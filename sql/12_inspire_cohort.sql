-- =====================================================================
-- A1 external validation | INSPIRE (Korea, surgical/anesthesia) 1-yr mortality
-- Times are TEXT minutes from a per-subject anchor. Index = ICU admission (icuin_time).
-- =====================================================================
DROP TABLE IF EXISTS inspire.abi_a1_ext;
CREATE TABLE inspire.abi_a1_ext AS
WITH tt AS (   -- cast text-minute times to bigint
  SELECT op_id, subject_id, hadm_id, age, sex, race,
    nullif(trim(icuin_time),'')::bigint icuin,
    nullif(trim(icuout_time),'')::bigint icuout,
    nullif(trim(discharge_time),'')::bigint disch,
    nullif(trim(inhosp_death_time),'')::bigint inhosp_death,
    nullif(trim(allcause_death_time),'')::bigint death
  FROM inspire.operations
),
abi_dx AS (   -- ABI aetiology (subject level; strip dots from icd10_cm)
  SELECT subject_id,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^S06' then 1 else 0 end) tbi,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^I60' then 1 else 0 end) sah,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^I61' then 1 else 0 end) ich,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^I63' then 1 else 0 end) ais,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^G0[0-5]' then 1 else 0 end) cns_inf,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^G4[01]' then 1 else 0 end) seizure,
    max(case when regexp_replace(icd10_cm,'\.','','g') ~ '^G931' then 1 else 0 end) anoxic
  FROM inspire.diagnosis GROUP BY subject_id
),
abi_subj AS (SELECT * FROM abi_dx WHERE (tbi+sah+ich+ais+cns_inf+seizure+anoxic)>0),
base AS (   -- first ABI operation with ICU admission per subject
  SELECT DISTINCT ON (t.subject_id) t.op_id, t.subject_id, t.hadm_id, t.age, t.sex,
    t.icuin, t.icuout, t.disch, t.death,
    d.tbi,d.sah,d.ich,d.ais,d.cns_inf,d.seizure,d.anoxic
  FROM tt t JOIN abi_subj d ON d.subject_id=t.subject_id
  WHERE t.icuin IS NOT NULL
  ORDER BY t.subject_id, t.icuin
),
vit AS (   -- ICU first-day vitals from ward_vitals (subject level); mbp direct or from sbp/dbp
  SELECT b.op_id,
    avg(case when w.item_name='hr' then nullif(trim(w.value),'')::numeric end) hr,
    avg(case when w.item_name='rr' then nullif(trim(w.value),'')::numeric end) rr,
    avg(case when w.item_name='spo2' then nullif(trim(w.value),'')::numeric end) spo2,
    avg(case when w.item_name='bt' then nullif(trim(w.value),'')::numeric end) bt,
    coalesce(
      avg(case when w.item_name='nibp_mbp' then nullif(trim(w.value),'')::numeric end),
      (avg(case when w.item_name='nibp_sbp' then nullif(trim(w.value),'')::numeric end)
       + 2*avg(case when w.item_name='nibp_dbp' then nullif(trim(w.value),'')::numeric end))/3.0
    ) mbp,
    max(case when w.item_name='vent' then 1 else 0 end) vented,
    max(case when w.item_name='crrt' then 1 else 0 end) crrt
  FROM base b JOIN inspire.ward_vitals w ON w.subject_id=b.subject_id
    AND nullif(trim(w.chart_time),'')::bigint BETWEEN b.icuin-60 AND b.icuin+1440
  GROUP BY b.op_id
),
lab AS (   -- ICU first-day labs (subject level)
  SELECT b.op_id,
    max(case when l.item_name='wbc' then nullif(trim(l.value),'')::numeric end) wbc_max,
    min(case when l.item_name='hb' then nullif(trim(l.value),'')::numeric end) hemoglobin_min,
    min(case when l.item_name='platelet' then nullif(trim(l.value),'')::numeric end) platelets_min,
    min(case when l.item_name='sodium' then nullif(trim(l.value),'')::numeric end) sodium_min,
    max(case when l.item_name='potassium' then nullif(trim(l.value),'')::numeric end) potassium_max,
    max(case when l.item_name='creatinine' then nullif(trim(l.value),'')::numeric end) creatinine_max,
    max(case when l.item_name='bun' then nullif(trim(l.value),'')::numeric end) bun_max,
    max(case when l.item_name='glucose' then nullif(trim(l.value),'')::numeric end) glucose_max,
    min(case when l.item_name='hco3' then nullif(trim(l.value),'')::numeric end) bicarbonate_min,
    max(case when l.item_name='ptinr' then nullif(trim(l.value),'')::numeric end) inr_max
  FROM base b JOIN inspire.labs l ON l.subject_id=b.subject_id
    AND nullif(trim(l.chart_time),'')::bigint BETWEEN b.icuin-360 AND b.icuin+1440
  GROUP BY b.op_id
),
vaso AS (SELECT DISTINCT b.op_id FROM base b JOIN inspire.medications m ON m.subject_id=b.subject_id
  AND nullif(trim(m.chart_time),'')::bigint BETWEEN b.icuin AND b.icuin+1440
  WHERE lower(m.drug_name) ~ 'norepineph|epinephrine|dopamine|phenyleph|vasopressin|dobutamine|milrinone')
SELECT b.op_id, b.subject_id, b.hadm_id, b.age, b.sex,
  b.tbi,b.sah,b.ich,b.ais,b.cns_inf,b.seizure,b.anoxic,
  CASE WHEN b.death IS NOT NULL AND (b.death - b.icuin) <= 525600 AND (b.death-b.icuin)>=0 THEN 1 ELSE 0 END as death_365,
  CASE WHEN b.death IS NOT NULL AND (b.death - b.icuin) <= 129600 AND (b.death-b.icuin)>=0 THEN 1 ELSE 0 END as death_90,
  v.hr as heart_rate_mean, v.mbp as mbp_mean, v.rr as resp_rate_mean,
  v.bt as temperature_mean, v.spo2 as spo2_mean,
  l.wbc_max,l.hemoglobin_min,l.platelets_min,l.sodium_min,l.potassium_max,
  l.creatinine_max,l.bun_max,l.glucose_max,l.bicarbonate_min,l.inr_max,
  coalesce(v.vented,0) as mech_vent,
  case when vaso.op_id is not null then 1 else 0 end as vasopressor,
  coalesce(v.crrt,0) as rrt   -- crrt flag from ward_vitals
FROM base b
LEFT JOIN vit v ON v.op_id=b.op_id
LEFT JOIN lab l ON l.op_id=b.op_id
LEFT JOIN vaso ON vaso.op_id=b.op_id
WHERE b.age >= 18;
