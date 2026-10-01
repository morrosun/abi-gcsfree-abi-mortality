# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
61_extract_severity_inputs.py —— 为 SOFA / OASIS / SAPS-II 头对头比较抽取首日变量

本地队列 CSV 里只有 sofa_day1；OASIS 与 SAPS-II 需要的
  尿量、胆红素、PaO2/FiO2、GCS、择期手术、慢性病史
都不在本地 CSV 里 —— 直接从本机 MIMIC-IV（mimiciv_derived / mimiciv_hosp）抽取。

OASIS 切点来源（已逐条核对原文 Table 1）：
  Awad A et al. OASIS+: BMC Med Inform Decis Mak. 2021;21:126. doi:10.1186/s12911-021-01517-7
SAPS-II 切点来源：Le Gall JR et al. JAMA. 1993;270:2957-2963.

产出：data/mimic_severity_inputs.csv（新增文件，不覆盖任何既有文件）
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(ABI_BASE)
DATA = BASE / "data"
OUTC = DATA / "mimic_severity_inputs.csv"


def log(*a):
    print(*a)
    sys.stdout.flush()


def main():
    log("=" * 74)
    log("61 | 抽取 SOFA / OASIS / SAPS-II 的缺失输入")
    coh = pd.read_csv(DATA / "mimic_abi_cohort.csv",
                      usecols=['subject_id', 'hadm_id', 'stay_id', 'age', 'admittime', 'intime'])
    log(f"  cohort stays = {len(coh):,}")

    conn = psycopg2.connect(host=_os.environ.get("ABI_DB_HOST", "localhost"),
                            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
                            user=_os.environ.get("ABI_DB_USER", "postgres"),
                            password=_os.environ.get("ABI_DB_PASSWORD", ""),
                            dbname="mimiciv")
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '600s'")

    def q(sql, label):
        t0 = time.time()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        log(f"  [{label}] {len(df):,} rows  ({time.time()-t0:.1f}s)")
        return df

    gcs = q("""SELECT stay_id, gcs_min, gcs_motor FROM mimiciv_derived.first_day_gcs""", 'first_day_gcs')
    uo = q("""SELECT stay_id, urineoutput FROM mimiciv_derived.first_day_urine_output""", 'urine_output')
    vs = q("""SELECT stay_id, heart_rate_min, heart_rate_max, sbp_min, sbp_max,
                     resp_rate_mean, temperature_min, temperature_max
              FROM mimiciv_derived.first_day_vitalsign""", 'vitalsign')
    lab = q("""SELECT stay_id, bilirubin_total_max, bun_max, wbc_max, potassium_max,
                      sodium_min, bicarbonate_min
               FROM mimiciv_derived.first_day_lab""", 'lab')
    bg = q("""SELECT stay_id, pao2fio2ratio_min, pao2fio2ratio_max
              FROM mimiciv_derived.first_day_bg_art""", 'bg_art')

    # 慢性病史（SAPS-II 三项）
    dx = q(r"""
        SELECT d.hadm_id,
               MAX(CASE WHEN d.icd_version=10 AND d.icd_code LIKE 'B2%'
                        THEN 1
                        WHEN d.icd_version=9  AND d.icd_code LIKE '042%'
                        THEN 1 ELSE 0 END) AS aids,
               MAX(CASE WHEN d.icd_version=10 AND (d.icd_code LIKE 'C77%' OR d.icd_code LIKE 'C78%'
                        OR d.icd_code LIKE 'C79%' OR d.icd_code LIKE 'C80%')
                        THEN 1
                        WHEN d.icd_version=9  AND (d.icd_code LIKE '196%' OR d.icd_code LIKE '197%'
                        OR d.icd_code LIKE '198%' OR d.icd_code LIKE '199%')
                        THEN 1 ELSE 0 END) AS mets,
               MAX(CASE WHEN d.icd_version=10 AND (d.icd_code LIKE 'C81%' OR d.icd_code LIKE 'C82%'
                        OR d.icd_code LIKE 'C83%' OR d.icd_code LIKE 'C84%' OR d.icd_code LIKE 'C85%'
                        OR d.icd_code LIKE 'C86%' OR d.icd_code LIKE 'C88%' OR d.icd_code LIKE 'C90%'
                        OR d.icd_code LIKE 'C91%' OR d.icd_code LIKE 'C92%' OR d.icd_code LIKE 'C93%'
                        OR d.icd_code LIKE 'C94%' OR d.icd_code LIKE 'C95%' OR d.icd_code LIKE 'C96%')
                        THEN 1
                        WHEN d.icd_version=9  AND (d.icd_code LIKE '200%' OR d.icd_code LIKE '201%'
                        OR d.icd_code LIKE '202%' OR d.icd_code LIKE '203%' OR d.icd_code LIKE '204%'
                        OR d.icd_code LIKE '205%' OR d.icd_code LIKE '206%' OR d.icd_code LIKE '207%'
                        OR d.icd_code LIKE '208%')
                        THEN 1 ELSE 0 END) AS hem_malign
        FROM mimiciv_hosp.diagnoses_icd d
        GROUP BY d.hadm_id
    """, 'chronic_dx')

    # 手术入院代理（SAPS-II 入院类型）：该次住院是否登记过手术操作码
    proc = q("""SELECT hadm_id, COUNT(*) AS n_proc
                FROM mimiciv_hosp.procedures_icd GROUP BY hadm_id""", 'procedures')
    cur.close()
    conn.close()

    d = coh.copy()
    for df in [gcs, uo, vs, lab, bg]:
        df2 = df.drop_duplicates(subset=['stay_id'])
        d = d.merge(df2, on='stay_id', how='left')
    d = d.merge(dx, on='hadm_id', how='left')
    d = d.merge(proc, on='hadm_id', how='left')
    for c in ['aids', 'mets', 'hem_malign', 'n_proc']:
        d[c] = d[c].fillna(0).astype(int)

    d['pre_icu_los_hours'] = (pd.to_datetime(d['intime']) -
                              pd.to_datetime(d['admittime'])).dt.total_seconds() / 3600.0
    n_neg = int((d['pre_icu_los_hours'] < 0).sum())
    d['pre_icu_los_hours'] = d['pre_icu_los_hours'].clip(lower=0.0)   # MIMIC admittime 匿名化会造出负值
    d.to_csv(OUTC, index=False)
    log(f"\n  wrote {OUTC}  n={len(d):,}")

    cov = {c: round(float(d[c].notna().mean()), 4) for c in
           ['gcs_min', 'urineoutput', 'bilirubin_total_max', 'pao2fio2ratio_min',
            'heart_rate_min', 'sbp_min', 'temperature_max', 'bun_max', 'wbc_max',
            'potassium_max', 'sodium_min', 'bicarbonate_min']}
    cov['pre_icu_los_hours_clipped_negative_n'] = n_neg
    json.dump(cov, open(DATA / "mimic_severity_input_coverage.json", 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    # ---------------- 自检 ----------------
    # 说明：胆红素与 PaO2/FiO2 的覆盖率天然偏低（MIMIC-IV 并非所有病人都会查），
    # 这不是抽取失败，而是"未查即按正常计 0 分"的前提 —— 必须在论文里写明。
    log("\nSELF-CHECK")
    ok = True
    checks = [
        ('rows == cohort', len(d) == len(coh), f"{len(d)} == {len(coh)}"),
        ('stay_id unique', d['stay_id'].is_unique, str(d['stay_id'].duplicated().sum()) + ' dup'),
        ('gcs_min coverage >0.98', d['gcs_min'].notna().mean() > 0.98, f"{cov['gcs_min']:.4f}"),
        ('urineoutput coverage >0.95', d['urineoutput'].notna().mean() > 0.95, f"{cov['urineoutput']:.4f}"),
        ('vitals coverage >0.95', min(cov['heart_rate_min'], cov['sbp_min'],
                                      cov['temperature_max']) > 0.95,
         f"hr {cov['heart_rate_min']:.4f} sbp {cov['sbp_min']:.4f} temp {cov['temperature_max']:.4f}"),
        ('labs coverage >0.95', min(cov['bun_max'], cov['wbc_max'], cov['potassium_max'],
                                    cov['sodium_min'], cov['bicarbonate_min']) > 0.95,
         f"bun {cov['bun_max']:.4f} wbc {cov['wbc_max']:.4f} k {cov['potassium_max']:.4f}"),
        ('bilirubin coverage >0.50 (low, by design)', d['bilirubin_total_max'].notna().mean() > 0.50,
         f"{cov['bilirubin_total_max']:.4f}  -> 未查者按正常计 0 分，须在稿件中声明"),
        ('pao2fio2 coverage >0.30 (only if ventilated)',
         d['pao2fio2ratio_min'].notna().mean() > 0.30,
         f"{cov['pao2fio2ratio_min']:.4f}  -> 仅通气患者计分"),
        ('temp in Celsius 30-45', bool(d['temperature_max'].dropna().between(30, 45).mean() > 0.99),
         f"median {d['temperature_max'].median():.2f}"),
        ('pre_icu_los clipped >=0', bool((d['pre_icu_los_hours'] >= 0).all()),
         f"clipped {n_neg} negative rows"),
        ('chronic dx flags non-null', d[['aids', 'mets', 'hem_malign']].notna().all().all(), 'ok'),
    ]
    for name, good, detail in checks:
        ok &= bool(good)
        log(f"  [{'OK ' if good else 'FAIL'}] {name:<40} {detail}")
    log("=" * 74)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
