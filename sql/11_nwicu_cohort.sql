-- =====================================================================
-- A1 external validation | NWICU (Northwestern)  1-year all-cause mortality
-- MIMIC-structured but non-standard itemids; temp in Fahrenheit.
--
-- ★ v1.4.1（2026-10-19）三处修正，与现行分析口径（scripts/112 有下界标签 +
--   74/100 血管活性药恢复）逐条一致：
--   ① 死亡标签改用**三源并集**（deathtime / discharge_location /
--      hospital_expire_flag），不再单靠 dod 或 hospital_expire_flag。
--      旧口径下院内死亡只数到 16/3,420，而 discharge-disposition 字段记录了
--      370 例死亡 —— 该库自带的 death flag 漏报了 96% 以上的院内死亡。
--   ② 并集加**时间下界**：deathtime 必须 ≥ admittime（本队列 1,035/3,420 例
--      intime < admittime，deathtime 可能早于入院时刻）。主分析 = 有下界口径
--      （院内 376 / 一年 699）；无下界值（院内 378 / 一年 701）保留为敏感性，
--      由 death_hosp_unbounded / death_365_unbounded 两列给出。
--   ③ vasopressor 不再是常数 0 —— NWICU 的 hosp.prescriptions 确有升压药记录
--      （五种升压药命中 84,398 行）。按 D 口径（intravenous/injection 给药途径，
--      排除 lidocaine–epinephrine 局麻复方与 cocoa butter）在预测窗
--      [intime−6h, intime+1d] 内抽取，队列患病率 11.1%。
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
    -- ★ v1.4.1：补 admittime / deathtime —— 三源并集与时间下界都要用
    a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag, a.discharge_location,
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
        WHERE pe.itemid IN (704890,772042,724671,798351)),
-- ★ v1.4.1：血管活性药恢复（74/100 号脚本的 D 口径）。原写死 0 as vasopressor。
vp AS (
  SELECT DISTINCT b.hadm_id
  FROM base b JOIN hosp.prescriptions p ON p.hadm_id=b.hadm_id
  WHERE ( lower(p.drug) LIKE '%norepinephrine%' OR lower(p.drug) LIKE '%epinephrine%'
       OR lower(p.drug) LIKE '%dopamine%' OR lower(p.drug) LIKE '%phenylephrine%'
       OR lower(p.drug) LIKE '%vasopressin%' )
    AND lower(p.route) IN ('intravenous','injection')
    AND lower(p.drug) NOT LIKE '%lidocaine%'
    AND lower(p.drug) NOT LIKE '%cocoa butter%'
    AND p.starttime BETWEEN b.intime - interval '6 hours' AND b.intime + interval '1 day'
),
-- ★ v1.4.1：三源并集死亡标签。death_hosp / death_hosp_unbounded 的差只在
--   "deathtime < admittime" 的那 2 例（有下界者剔除）。
labels AS (
  SELECT b.*,
    CASE WHEN (b.deathtime IS NOT NULL AND b.deathtime >= b.admittime
               AND b.deathtime <= b.dischtime)
           OR upper(b.discharge_location) ~ 'EXPIR|DEAD|DEATH|DIED|MORT'
           OR b.hospital_expire_flag = 1
         THEN 1 ELSE 0 END AS death_hosp,
    CASE WHEN (b.deathtime IS NOT NULL AND b.deathtime <= b.dischtime)
           OR upper(b.discharge_location) ~ 'EXPIR|DEAD|DEATH|DIED|MORT'
           OR b.hospital_expire_flag = 1
         THEN 1 ELSE 0 END AS death_hosp_unbounded,
    -- 90 天列为遗留口径（仅 84 号标签交叉核验用），仍按无下界 dod 单规则计
    CASE WHEN b.dod IS NOT NULL AND b.dod <= (b.intime + interval '90 days')
         THEN 1 ELSE 0 END AS death_90
  FROM base b
),
death365 AS (
  SELECT l.*,
    -- 主分析（有下界）：院内并集 ∪ {dod 或 deathtime 落在 [intime, intime+365d]}
    CASE WHEN l.death_hosp = 1
           OR (l.dod IS NOT NULL AND l.dod >= l.intime
               AND l.dod <= l.intime + interval '365 days')
           OR (l.deathtime IS NOT NULL AND l.deathtime >= l.intime
               AND l.deathtime <= l.intime + interval '365 days')
         THEN 1 ELSE 0 END AS death_365,
    -- 敏感性（无下界）
    CASE WHEN l.death_hosp_unbounded = 1
           OR (l.dod IS NOT NULL AND l.dod >= l.intime
               AND l.dod <= l.intime + interval '365 days')
           OR (l.deathtime IS NOT NULL AND l.deathtime >= l.intime
               AND l.deathtime <= l.intime + interval '365 days')
         THEN 1 ELSE 0 END AS death_365_unbounded
  FROM labels l
)
SELECT b.*,   -- death_365 / death_365_unbounded / death_90 / death_hosp / death_hosp_unbounded 已由 labels/death365 两个 CTE 给出
  v.hr as heart_rate_mean,
  case when v.sbp is not null and v.dbp is not null then (v.sbp+2*v.dbp)/3.0 end as mbp_mean,
  v.rr as resp_rate_mean, v.temp_c as temperature_mean, v.spo2 as spo2_mean,
  l.wbc_max,l.hemoglobin_min,l.platelets_min,l.sodium_min,l.potassium_max,
  l.creatinine_max,l.bun_max,l.glucose_max, NULL::numeric as bicarbonate_min, l.inr_max,
  case when vent.stay_id is not null then 1 else 0 end as mech_vent,
  case when vp.hadm_id is not null then 1 else 0 end as vasopressor,
  case when rrt.stay_id is not null then 1 else 0 end as rrt
FROM death365 b
LEFT JOIN vit v ON v.stay_id=b.stay_id
LEFT JOIN lab l ON l.hadm_id=b.hadm_id
LEFT JOIN vent ON vent.stay_id=b.stay_id
LEFT JOIN vp ON vp.hadm_id=b.hadm_id
LEFT JOIN rrt ON rrt.stay_id=b.stay_id
WHERE b.age >= 18;
