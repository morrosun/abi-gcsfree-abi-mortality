-- =====================================================================
-- A1 external validation | eICU-CRD  (in-hospital mortality)
-- Aligns to MIMIC A1 predictor set (GCS-independent). Index = first ICU unit stay.
-- =====================================================================
DROP TABLE IF EXISTS eicu_crd.abi_a1_ext;
CREATE TABLE eicu_crd.abi_a1_ext AS
WITH abi AS (   -- ABI aetiology per unit stay (ICD10 in icd9code col OR diagnosisstring)
  SELECT patientunitstayid,
    max(case when icd9code ~ '(^|,)\s*S06' or lower(diagnosisstring) ~ 'traumatic brain|head injury|cerebral contusion' then 1 else 0 end) tbi,
    max(case when icd9code ~ '(^|,)\s*I60' or lower(diagnosisstring) ~ 'subarachnoid' then 1 else 0 end) sah,
    max(case when icd9code ~ '(^|,)\s*I61' or lower(diagnosisstring) ~ 'intracerebral|intraparenchymal hemorrhage' then 1 else 0 end) ich,
    max(case when icd9code ~ '(^|,)\s*I63' or lower(diagnosisstring) ~ 'ischemic stroke|cerebrovascular accident|acute cva' then 1 else 0 end) ais,
    max(case when icd9code ~ '(^|,)\s*G0[0-5]' or lower(diagnosisstring) ~ 'meningitis|encephalitis' then 1 else 0 end) cns_inf,
    max(case when icd9code ~ '(^|,)\s*G4[01]' or lower(diagnosisstring) ~ 'seizure|status epilepticus' then 1 else 0 end) seizure,
    max(case when icd9code ~ '(^|,)\s*G93\.1' or lower(diagnosisstring) ~ 'anoxic|hypoxic' then 1 else 0 end) anoxic
  FROM eicu_crd.diagnosis
  GROUP BY patientunitstayid
),
abi_stays AS (
  SELECT * FROM abi WHERE (tbi+sah+ich+ais+cns_inf+seizure+anoxic) > 0
),
pat AS (   -- attach patient info + parse age; pick first ABI unit stay per person
  SELECT DISTINCT ON (p.uniquepid)
    p.patientunitstayid, p.uniquepid, p.gender,
    case when p.age ~ '89' then 90 when p.age ~ '^[0-9]+$' then p.age::int else null end as age,
    p.hospitaldischargestatus, p.unitvisitnumber, p.hospitaladmitoffset,
    a.tbi,a.sah,a.ich,a.ais,a.cns_inf,a.seizure,a.anoxic
  FROM eicu_crd.patient p
  JOIN abi_stays a ON a.patientunitstayid = p.patientunitstayid
  ORDER BY p.uniquepid, p.hospitaladmitoffset, p.unitvisitnumber
),
aps AS (   -- APACHE worst-24h physiology (-1 = missing)
  SELECT patientunitstayid,
    nullif(heartrate,-1) heartrate, nullif(meanbp,-1) meanbp, nullif(respiratoryrate,-1) resp,
    nullif(temperature,-1) temp, nullif(sodium,-1) sodium, nullif(creatinine,-1) creat,
    nullif(bun,-1) bun, nullif(glucose,-1) glucose, nullif(wbc,-1) wbc,
    greatest(vent,intubated) vent, dialysis
  FROM eicu_crd.apacheapsvar
),
spo2 AS (   -- mean SpO2 first 24h
  SELECT patientunitstayid, avg(nullif(sao2,-1)) spo2_mean
  FROM eicu_crd.vitalperiodic
  WHERE observationoffset BETWEEN 0 AND 1440 AND sao2 IS NOT NULL
  GROUP BY patientunitstayid
),
lab AS (   -- first-day labs (worst dir mirrors MIMIC): min hgb/plt/hco3, max k/inr
  SELECT patientunitstayid,
    min(case when labname='Hgb' then labresult end) hemoglobin_min,
    min(case when labname='platelets x 1000' then labresult end) platelets_min,
    max(case when labname='potassium' then labresult end) potassium_max,
    min(case when labname in ('bicarbonate','HCO3') then labresult end) bicarbonate_min,
    max(case when labname='PT - INR' then labresult end) inr_max
  FROM eicu_crd.lab
  WHERE labresultoffset BETWEEN -60 AND 1440
  GROUP BY patientunitstayid
),
vaso AS (
  SELECT DISTINCT patientunitstayid FROM eicu_crd.infusiondrug
  WHERE lower(drugname) ~ 'norepineph|epineph|dopamine|phenyleph|vasopressin|levophed|neosynephrine|dobutamine|milrinone'
)
SELECT p.patientunitstayid, p.uniquepid, p.gender, p.age,
  p.tbi,p.sah,p.ich,p.ais,p.cns_inf,p.seizure,p.anoxic,
  case when p.hospitaldischargestatus='Expired' then 1 else 0 end as death_hosp,
  aps.heartrate as heart_rate_mean, aps.meanbp as mbp_mean, aps.resp as resp_rate_mean,
  aps.temp as temperature_mean, s.spo2_mean,
  aps.wbc as wbc_max, l.hemoglobin_min, l.platelets_min, aps.sodium as sodium_min,
  l.potassium_max, aps.creat as creatinine_max, aps.bun as bun_max, aps.glucose as glucose_max,
  l.bicarbonate_min, l.inr_max,
  case when aps.vent=1 then 1 else 0 end as mech_vent,
  case when v.patientunitstayid is not null then 1 else 0 end as vasopressor,
  case when aps.dialysis=1 then 1 else 0 end as rrt
FROM pat p
LEFT JOIN aps ON aps.patientunitstayid = p.patientunitstayid
LEFT JOIN spo2 s ON s.patientunitstayid = p.patientunitstayid
LEFT JOIN lab l ON l.patientunitstayid = p.patientunitstayid
LEFT JOIN vaso v ON v.patientunitstayid = p.patientunitstayid
WHERE p.age >= 18;
