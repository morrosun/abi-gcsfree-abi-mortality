# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
84 —— 死亡标签「日期级」交叉核验（评审意见 A0 / A1）

背景
----
外部评审指出 Table S7 的事件数关系不成立：
  * NWICU   全队列 3420，一年死亡 635，院内死亡 16，院内存活者一年死亡仍是 635
            → 16 例院内死亡无一计入一年死亡
  * INSPIRE 全队列一年死亡 262，院内死亡 126，院内存活者一年死亡 151
            → 126-?=136 才对，表里是 151，差 15
本脚本不预设结论，回到四个库的原始死亡字段做**日期级**核验。

核验原则
--------
1. 每个库的院内死亡至少取两个独立源互校；一年死亡取「院内死亡 ∨ 死亡时间落在
   [index, index+365d]」。
2. 只读取、不改写任何既有文件；新标签另存为 *_labels_v2.csv（不覆盖 *_final*.csv）。
3. 所有结果落盘：output/death_label_crosscheck_v1.json + output/_84_death_label_crosscheck.md。

统一口径定义（本脚本输出的 v2 标签）
-----------------------------------
  death_inhosp_v2 = deathtime 非空
                  ∨ hospital_expire_flag = 1
                  ∨ discharge_location 命中死亡语义（EXPIRED / DEAD / DEATH / EXPIRED…）
                  ∨ [INSPIRE] inhosp_death_time 非空
                  ∨ [eICU]    hospitaldischargestatus = 'Expired'
  death_365_v2    = death_inhosp_v2 ∨ (死亡时间在 [index, index+365d] 内)
  死亡时间优先级：deathtime > dod > inhosp_death_time > allcause_death_time
"""
import json
import os
import sys
from datetime import timedelta

import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")
PGKW = dict(host=_os.environ.get("ABI_DB_HOST", "localhost"),
            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
            user=_os.environ.get("ABI_DB_USER", "postgres"),
            password=_os.environ.get("ABI_DB_PASSWORD", ""))

LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def pg(db):
    return psycopg2.connect(dbname=db, **PGKW)


def death_loc_series(s):
    """discharge_location 是否命中死亡语义"""
    if s is None:
        return pd.Series([False] * 0)
    u = s.astype(str).str.upper().str.strip()
    return u.str.contains(r"EXPIR|DEAD|DEATH|DIED|MORT", regex=True, na=False)


def crosstab_md(a, b, na, nb):
    t = pd.crosstab(a, b, rownames=[na], colnames=[nb])
    return t


# ----------------------------------------------------------------- NWICU
def audit_nwicu():
    log("=" * 78)
    log("NWICU —— 院内死亡标签三源互校（评审 A0 的核心）")
    log("=" * 78)
    df = pd.read_csv(os.path.join(DATA, "nwicu_abi_final_vp2.csv"),
                     parse_dates=["intime", "outtime", "dischtime", "dod"])
    n = len(df)
    c = pg("nwicu")
    cur = c.cursor()
    cur.execute("set statement_timeout=120000")
    adm = pd.read_sql("SELECT hadm_id, admittime, dischtime, deathtime, "
                      "hospital_expire_flag, discharge_location FROM hosp.admissions", c)
    # 全库级：字段本身是否可用
    adm_all = adm.copy()
    adm_all["dl_dead"] = death_loc_series(adm_all["discharge_location"])
    adm_all["dt_dead"] = adm_all["deathtime"].notna()
    full = pd.crosstab(adm_all["hospital_expire_flag"], adm_all["dl_dead"])
    full2 = pd.crosstab(adm_all["hospital_expire_flag"], adm_all["dt_dead"])
    log("")
    log("[全库 hosp.admissions n=%d]  hospital_expire_flag × discharge_location=死亡" % len(adm_all))
    log(full.to_string())
    log("")
    log("[全库]  hospital_expire_flag × deathtime 非空")
    log(full2.to_string())

    # ★ deathtime 语义判定：是「院内死亡时间」还是「全因死亡时间」？
    cur.execute("""SELECT count(*) n, count(deathtime) n_dt,
      sum(case when deathtime<=dischtime then 1 else 0 end) dt_le_disch,
      sum(case when deathtime>dischtime  then 1 else 0 end) dt_gt_disch,
      sum(case when hospital_expire_flag=1 then 1 else 0 end) flag1,
      sum(case when hospital_expire_flag=1 and deathtime is null then 1 else 0 end) flag1_dt_null
      FROM hosp.admissions""")
    sem = cur.fetchone()
    log("")
    log("[全库]  ★ deathtime 语义判定")
    log("  admissions=%d ；deathtime 非空=%d ；deathtime<=dischtime=%d ；deathtime>dischtime=%d" %
        (sem[0], sem[1], sem[2], sem[3]))
    log("  hospital_expire_flag=1 共 %d 例，其中 deathtime 为空 %d 例  → 两源%s" %
        (sem[4], sem[5], "完全互斥" if sem[4] == sem[5] else "有重叠"))
    cur.execute("""SELECT count(*) n,
      sum(case when a.deathtime::date = p.dod then 1 else 0 end) same_day,
      sum(case when a.deathtime is not null and p.dod is null then 1 else 0 end) dt_only,
      sum(case when a.deathtime is null and p.dod is not null then 1 else 0 end) dod_only
      FROM hosp.admissions a JOIN hosp.patients p USING(subject_id)""")
    sem2 = cur.fetchone()
    log("  deathtime 与 patients.dod 同一天=%d ；仅 deathtime 有值=%d ；仅 dod 有值=%d" %
        (sem2[1], sem2[2], sem2[3]))
    log("  → 判定：deathtime 是「全因死亡时间」（与 dod 同源且覆盖更全），")
    log("    因此院内死亡必须限定为 deathtime<=dischtime，不能直接用 deathtime 非空。")
    c.close()

    m = df.merge(adm[["hadm_id", "admittime", "deathtime"]].rename(
        columns={"dischtime": "_x"}), on="hadm_id", how="left", suffixes=("", "_adm"))
    m["dl_dead"] = death_loc_series(m["discharge_location"])
    m["dt_dead"] = m["deathtime"].notna()
    m["dt_inhosp"] = m["dt_dead"] & (m["deathtime"] <= m["dischtime"])
    m["fl_dead"] = m["hospital_expire_flag"].fillna(0).astype(int).eq(1)

    log("")
    log("[队列 n=%d]  三源阳性数" % n)
    log("  hospital_expire_flag=1              : %d" % m["fl_dead"].sum())
    log("  deathtime 非空（全因死亡）          : %d" % m["dt_dead"].sum())
    log("    其中 deathtime<=dischtime（院内） : %d" % m["dt_inhosp"].sum())
    log("  discharge_location=死亡             : %d" % m["dl_dead"].sum())
    log("  现行 death_hosp（= flag）   : %d" % m["death_hosp"].sum())
    log("  现行 death_365（dod≤365d）  : %d" % m["death_365"].sum())
    log("")
    log("[队列]  flag × discharge_location=死亡")
    log(pd.crosstab(m["fl_dead"], m["dl_dead"]).to_string())
    log("")
    log("[队列]  flag × deathtime 非空")
    log(pd.crosstab(m["fl_dead"], m["dt_dead"]).to_string())
    log("")
    log("[队列]  deathtime 非空 × discharge_location=死亡")
    log(pd.crosstab(m["dt_dead"], m["dl_dead"]).to_string())

    # 日期级矛盾
    hosp_v2 = m["dt_inhosp"] | m["dl_dead"] | m["fl_dead"]
    dod = m["dod"]
    win_hi = m["intime"] + timedelta(days=365)
    in_win_dod = dod.notna() & (dod >= m["intime"]) & (dod <= win_hi)
    in_win_dt = m["dt_dead"] & (m["deathtime"] >= m["intime"]) & (m["deathtime"] <= win_hi)
    d365_v2 = hosp_v2 | in_win_dod | in_win_dt

    contra = {}
    contra["dod 早于 ICU 入组（dod < intime）"] = int((dod.notna() & (dod < m["intime"])).sum())
    contra["deathtime 早于 ICU 入组"] = int(
        (m["dt_dead"] & (m["deathtime"] < m["intime"])).sum())
    contra["deathtime 非空但 flag=0"] = int((m["dt_dead"] & ~m["fl_dead"]).sum())
    contra["flag=1 但 deathtime 为空"] = int((m["fl_dead"] & ~m["dt_dead"]).sum())
    contra["出院去向=死亡但 flag=0"] = int((m["dl_dead"] & ~m["fl_dead"]).sum())
    contra["deathtime<=dischtime 但出院去向非死亡"] = int((m["dt_inhosp"] & ~m["dl_dead"]).sum())
    contra["出院去向=死亡但 deathtime>dischtime 或缺失"] = int(
        (m["dl_dead"] & ~m["dt_inhosp"]).sum())
    contra["院内死亡(v2)=1 但 death_365(现行)=0"] = int((hosp_v2 & m["death_365"].eq(0)).sum())
    contra["death_365(现行)=1 但 dod 缺失（不应发生）"] = int((m["death_365"].eq(1) & dod.isna()).sum())
    contra["deathtime 在365d窗口内但 death_365(现行)=0"] = int(
        (in_win_dt & m["death_365"].eq(0)).sum())
    log("")
    log("[队列]  日期级矛盾计数")
    for k, v in contra.items():
        log("  %-42s : %d" % (k, v))

    log("")
    log("[队列]  统一口径后的事件数")
    log("  death_inhosp : %d  →  %d   (Δ %+d)" %
        (int(m["death_hosp"].sum()), int(hosp_v2.sum()), int(hosp_v2.sum() - m["death_hosp"].sum())))
    log("  death_365    : %d  →  %d   (Δ %+d)" %
        (int(m["death_365"].sum()), int(d365_v2.sum()), int(d365_v2.sum() - m["death_365"].sum())))
    log("  院内存活者一年死亡 : 现行 %d  →  v2 %d" %
        (int(m.loc[m["death_hosp"].eq(0), "death_365"].sum()),
         int(d365_v2[~hosp_v2].sum())))

    # 不符预期逐例（院内死亡但现行一年标签=0）
    bad = m[hosp_v2 & m["death_365"].eq(0)].copy()
    def _why(r):
        if pd.isna(r["deathtime"]):
            return "deathtime 与 dod 均缺失"
        if r["deathtime"] < r["intime"]:
            return "deathtime 早于 ICU 入组"
        if r["deathtime"] > r["intime"] + timedelta(days=365):
            return "deathtime 晚于 365 天窗口"
        return "其他（窗口内却未计入）"
    bad["原因"] = bad.apply(_why, axis=1)
    log("")
    log("[队列]  院内死亡(v2)=1 但 death_365(现行)=0 的 %d 例 —— 原因分布" % len(bad))
    log(bad["原因"].value_counts().to_string())
    cols = ["subject_id", "hadm_id", "stay_id", "intime", "dischtime", "dod", "deathtime",
            "hospital_expire_flag", "discharge_location", "death_365"]
    log("")
    log("  前 15 例明细：")
    log(bad[cols].head(15).to_string())

    out = m[["subject_id", "hadm_id", "stay_id", "death_hosp", "death_365", "death_90"]].copy()
    out["death_inhosp_v2"] = hosp_v2.astype(int).values
    out["death_365_v2"] = d365_v2.astype(int).values
    out.to_csv(os.path.join(DATA, "nwicu_abi_labels_v2.csv"), index=False)

    return dict(
        database="NWICU", n=n,
        full_db=dict(n_admissions=int(len(adm_all)),
                     flag1=int(adm_all["hospital_expire_flag"].fillna(0).eq(1).sum()),
                     dl_dead=int(adm_all["dl_dead"].sum()),
                     dt_dead=int(adm_all["dt_dead"].sum()),
                     flag1_and_dl_dead=int((adm_all["hospital_expire_flag"].fillna(0).eq(1)
                                            & adm_all["dl_dead"]).sum()),
                     flag0_and_dl_dead=int((adm_all["hospital_expire_flag"].fillna(0).eq(0)
                                            & adm_all["dl_dead"]).sum())),
        cohort=dict(death_hosp_current=int(m["death_hosp"].sum()),
                    death_365_current=int(m["death_365"].sum()),
                    flag1=int(m["fl_dead"].sum()),
                    deathtime_notnull=int(m["dt_dead"].sum()),
                    deathtime_inhosp=int(m["dt_inhosp"].sum()),
                    discharge_location_dead=int(m["dl_dead"].sum()),
                    death_inhosp_v2=int(hosp_v2.sum()),
                    death_365_v2=int(d365_v2.sum()),
                    survivor_1yr_current=int(m.loc[m["death_hosp"].eq(0), "death_365"].sum()),
                    survivor_1yr_v2=int(d365_v2[~hosp_v2].sum())),
        contradictions=contra,
        bad_case_reasons={k: int(v) for k, v in bad["原因"].value_counts().items()},
    )


# ----------------------------------------------------------------- MIMIC-IV
def audit_mimic():
    log("")
    log("=" * 78)
    log("MIMIC-IV —— 开发队列标签自证（评审待查项 1）")
    log("=" * 78)
    df = pd.read_csv(os.path.join(DATA, "mimic_abi_cohort.csv"),
                     parse_dates=["intime", "outtime", "dod", "admittime", "dischtime"])
    c = pg("mimiciv")
    cur = c.cursor()
    cur.execute("set statement_timeout=120000")
    adm = pd.read_sql("SELECT hadm_id, deathtime, hospital_expire_flag, discharge_location "
                      "FROM mimiciv_hosp.admissions", c)
    c.close()
    m = df.merge(adm, on="hadm_id", how="left", suffixes=("", "_adm"))
    n = len(m)
    m["dl_dead"] = death_loc_series(m["discharge_location"])
    m["dt_dead"] = m["deathtime"].notna()
    m["fl_dead"] = m["hospital_expire_flag"].fillna(0).astype(int).eq(1)

    log("")
    log("[队列 n=%d]  三源阳性数" % n)
    log("  hospital_expire_flag=1  : %d" % m["fl_dead"].sum())
    log("  deathtime 非空          : %d" % m["dt_dead"].sum())
    log("  discharge_location=死亡 : %d" % m["dl_dead"].sum())
    log("  现行 death_hosp         : %d" % m["death_hosp"].sum())
    log("  现行 death_365          : %d" % m["death_365"].sum())
    log("")
    log("[队列]  flag × deathtime 非空")
    log(pd.crosstab(m["fl_dead"], m["dt_dead"]).to_string())

    hosp_v2 = m["dt_dead"] | m["dl_dead"] | m["fl_dead"]
    dod = m["dod"]
    in_win = dod.notna() & (dod >= m["intime"]) & (dod <= m["intime"] + timedelta(days=365))
    d365_v2 = hosp_v2 | in_win

    contra = {}
    contra["deathtime 非空但 flag=0"] = int((m["dt_dead"] & ~m["fl_dead"]).sum())
    contra["flag=1 但 deathtime 为空"] = int((m["fl_dead"] & ~m["dt_dead"]).sum())
    contra["院内死亡(v2)=1 但 dod 缺失"] = int((hosp_v2 & dod.isna()).sum())
    contra["院内死亡(v2)=1 且 death_365(现行)=0"] = int((hosp_v2 & m["death_365"].eq(0)).sum())
    contra["dod 早于 ICU 入组"] = int((dod.notna() & (dod < m["intime"])).sum())
    contra["deathtime 晚于出院时间（语义异常）"] = int(
        (m["dt_dead"] & (m["deathtime"] > m["dischtime"])).sum())
    log("")
    log("[队列]  日期级矛盾计数")
    for k, v in contra.items():
        log("  %-42s : %d" % (k, v))
    log("")
    log("[队列]  统一口径后：death_inhosp %d → %d (Δ%+d)；death_365 %d → %d (Δ%+d)" %
        (int(m["death_hosp"].sum()), int(hosp_v2.sum()), int(hosp_v2.sum() - m["death_hosp"].sum()),
         int(m["death_365"].sum()), int(d365_v2.sum()), int(d365_v2.sum() - m["death_365"].sum())))

    out = m[["subject_id", "hadm_id", "stay_id", "death_hosp", "death_365"]].copy()
    out["death_inhosp_v2"] = hosp_v2.astype(int).values
    out["death_365_v2"] = d365_v2.astype(int).values
    out.to_csv(os.path.join(DATA, "mimic_abi_labels_v2.csv"), index=False)
    return dict(database="MIMIC-IV", n=n,
                cohort=dict(death_hosp_current=int(m["death_hosp"].sum()),
                            death_365_current=int(m["death_365"].sum()),
                            flag1=int(m["fl_dead"].sum()),
                            deathtime_notnull=int(m["dt_dead"].sum()),
                            discharge_location_dead=int(m["dl_dead"].sum()),
                            death_inhosp_v2=int(hosp_v2.sum()),
                            death_365_v2=int(d365_v2.sum())),
                contradictions=contra)


# ----------------------------------------------------------------- INSPIRE
def audit_inspire():
    log("")
    log("=" * 78)
    log("INSPIRE —— 院内死亡与全因死亡时间双源互校（评审 A1 的 15 例差异）")
    log("=" * 78)
    c = pg("inspire")
    cur = c.cursor()
    cur.execute("set statement_timeout=120000")
    op = pd.read_sql("""SELECT op_id, subject_id, hadm_id,
        nullif(trim(icuin_time),'')::bigint        AS icuin,
        nullif(trim(icuout_time),'')::bigint       AS icuout,
        nullif(trim(discharge_time),'')::bigint    AS disch,
        nullif(trim(inhosp_death_time),'')::bigint AS inhosp_death,
        nullif(trim(allcause_death_time),'')::bigint AS allcause_death
        FROM inspire.operations""", c)
    ext = pd.read_sql("SELECT op_id, death_365, death_90 FROM inspire.abi_a1_ext", c)
    c.close()
    dh = pd.read_csv(os.path.join(DATA, "inspire_deathhosp.csv"))
    m = ext.merge(op, on="op_id", how="left").merge(dh, on="op_id", how="left")
    n = len(m)
    DAY = 1440  # 分钟/天
    m["inhosp_dead"] = m["inhosp_death"].notna()
    m["allcause_dead"] = m["allcause_death"].notna()
    m["dt_in_win"] = m["allcause_death"].notna() & (m["allcause_death"] - m["icuin"]).between(0, 365 * DAY)
    m["ih_in_win"] = m["inhosp_death"].notna() & (m["inhosp_death"] - m["icuin"]).between(0, 365 * DAY)

    log("")
    log("[队列 n=%d]  源字段阳性数" % n)
    log("  inhosp_death_time 非空   : %d" % m["inhosp_dead"].sum())
    log("  allcause_death_time 非空 : %d" % m["allcause_dead"].sum())
    log("  death_hosp（现行）       : %d" % m["death_hosp"].sum())
    log("  death_365（现行）        : %d" % m["death_365"].sum())
    log("")
    log("[队列]  death_hosp × death_365（Table S7 的原始关系）")
    log(pd.crosstab(m["death_hosp"], m["death_365"]).to_string())
    log("")
    log("[队列]  inhosp_death_time 非空 × death_365")
    log(pd.crosstab(m["inhosp_dead"], m["death_365"]).to_string())

    hosp_v2 = m["inhosp_dead"] | m["death_hosp"].fillna(0).astype(int).eq(1)
    d365_v2 = hosp_v2 | m["dt_in_win"] | m["ih_in_win"]

    contra = {}
    contra["inhosp_death_time 非空但 death_hosp=0"] = int(
        (m["inhosp_dead"] & m["death_hosp"].fillna(0).eq(0)).sum())
    contra["death_hosp=1 但 inhosp_death_time 缺失"] = int(
        (m["death_hosp"].fillna(0).eq(1) & ~m["inhosp_dead"]).sum())
    contra["院内死亡(v2)=1 但 death_365(现行)=0"] = int((hosp_v2 & m["death_365"].eq(0)).sum())
    bad = m[hosp_v2 & m["death_365"].eq(0)].copy()
    contra["  其中 allcause_death_time 缺失"] = int(bad["allcause_death"].isna().sum())
    contra["  其中 allcause 在窗口外（<0 或 >365d）"] = int(
        (~bad["allcause_death"].isna() & ~bad["dt_in_win"]).sum())
    contra["death-inhosp 时间晚于出院时间（疑似错配）"] = int(
        (m["inhosp_death"].notna() & m["disch"].notna() & (m["inhosp_death"] > m["disch"] + 60)).sum())
    log("")
    log("[队列]  日期级矛盾计数")
    for k, v in contra.items():
        log("  %-44s : %d" % (k, v))

    log("")
    log("[队列]  统一口径后：death_inhosp %d → %d (Δ%+d)；death_365 %d → %d (Δ%+d)" %
        (int(m["death_hosp"].fillna(0).sum()), int(hosp_v2.sum()),
         int(hosp_v2.sum() - m["death_hosp"].fillna(0).sum()),
         int(m["death_365"].sum()), int(d365_v2.sum()), int(d365_v2.sum() - m["death_365"].sum())))
    log("  院内存活者一年死亡：现行 %d → v2 %d" %
        (int(m.loc[m["death_hosp"].fillna(0).eq(0), "death_365"].sum()),
         int(d365_v2[~hosp_v2].sum())))
    cols = ["op_id", "icuin", "icuout", "disch", "inhosp_death", "allcause_death",
            "death_hosp", "death_365"]
    log("")
    log("  不符预期样本（院内死亡 v2=1 但 death_365=0）前 15 例：")
    log(bad[cols].head(15).to_string())

    out = m[["op_id", "death_hosp", "death_365", "death_90"]].copy()
    out["death_inhosp_v2"] = hosp_v2.astype(int).values
    out["death_365_v2"] = d365_v2.astype(int).values
    out.to_csv(os.path.join(DATA, "inspire_abi_labels_v2.csv"), index=False)
    return dict(database="INSPIRE", n=n,
                cohort=dict(death_hosp_current=int(m["death_hosp"].fillna(0).sum()),
                            death_365_current=int(m["death_365"].sum()),
                            inhosp_death_time_notnull=int(m["inhosp_dead"].sum()),
                            allcause_death_time_notnull=int(m["allcause_dead"].sum()),
                            death_inhosp_v2=int(hosp_v2.sum()),
                            death_365_v2=int(d365_v2.sum()),
                            survivor_1yr_current=int(
                                m.loc[m["death_hosp"].fillna(0).eq(0), "death_365"].sum()),
                            survivor_1yr_v2=int(d365_v2[~hosp_v2].sum())),
                contradictions=contra)


# ----------------------------------------------------------------- eICU
def audit_eicu():
    log("")
    log("=" * 78)
    log("eICU —— 院内死亡标签（该库无 365 天随访，只做院内自证）")
    log("=" * 78)
    df = pd.read_csv(os.path.join(DATA, "eicu_abi_final.csv"))
    c = pg("eicu")
    cur = c.cursor()
    cur.execute("set statement_timeout=120000")
    p = pd.read_sql("SELECT patientunitstayid, hospitaldischargestatus, unitdischargestatus, "
                    "hospitaldischargeoffset, unitdischargeoffset FROM eicu_crd.patient", c)
    c.close()
    m = df.merge(p, on="patientunitstayid", how="left")
    n = len(m)
    m["hs_exp"] = m["hospitaldischargestatus"].astype(str).str.strip().str.lower().eq("expired")
    m["us_exp"] = m["unitdischargestatus"].astype(str).str.strip().str.lower().eq("expired")
    log("")
    log("[队列 n=%d]  hospitaldischargestatus 分布（前 8）" % n)
    log(m["hospitaldischargestatus"].value_counts(dropna=False).head(8).to_string())
    log("")
    log("  现行 death_hosp=%d ；源字段 hospitaldischargestatus='Expired'=%d ；"
        "unitdischargestatus='Expired'=%d" %
        (int(m["death_hosp"].sum()), int(m["hs_exp"].sum()), int(m["us_exp"].sum())))
    log("")
    log("[队列]  death_hosp × hospitaldischargestatus='Expired'")
    log(pd.crosstab(m["death_hosp"], m["hs_exp"]).to_string())
    contra = dict(
        **{"ICU 内死亡但医院状态非 Expired": int((m["us_exp"] & ~m["hs_exp"]).sum()),
           "医院 Expired 但 ICU 状态非 Expired": int((m["hs_exp"] & ~m["us_exp"]).sum()),
           "两源不一致合计": int((m["hs_exp"] ^ m["us_exp"]).sum())})
    log("")
    log("  %-40s : %d" % ("两源不一致合计", contra["两源不一致合计"]))
    return dict(database="eICU", n=n,
                cohort=dict(death_hosp_current=int(m["death_hosp"].sum()),
                            hospitaldischargestatus_expired=int(m["hs_exp"].sum()),
                            unitdischargestatus_expired=int(m["us_exp"].sum())),
                contradictions=contra)


# ----------------------------------------------------------------- 淮安
def audit_huaian():
    log("")
    log("=" * 78)
    log("淮安本地队列 —— 外部验证用（仅有汇总级时间字段）")
    log("=" * 78)
    d = pd.read_csv(os.path.join(DATA, "huaian_local_cohort_deidentified.csv"))
    n = len(d)
    log("")
    log("[n=%d]  death_inhosp=%d ；death_365=%d ；death_365 缺失=%d" %
        (n, int(d["death_inhosp"].sum()), int(d["death_365"].sum()), int(d["death_365"].isna().sum())))
    ttd, fu = d["time_to_death_days"], d["followup_days"]
    contra = {}
    contra["death_365=1 但 time_to_death_days > 365"] = int(
        ((d["death_365"] == 1) & (ttd > 365)).sum())
    contra["death_inhosp=1 但 death_365=0"] = int(((d["death_inhosp"] == 1) & (d["death_365"] == 0)).sum())
    contra["death_365=1 但 time_to_death_days 缺失"] = int(
        ((d["death_365"] == 1) & ttd.isna()).sum())
    contra["time_to_death_days > followup_days"] = int((ttd > fu).sum())
    for k, v in contra.items():
        log("  %-42s : %d" % (k, v))
    if "hospital_discharge_location" in d.columns:
        dl = death_loc_series(d["hospital_discharge_location"])
        log("")
        log("  discharge_location 死亡语义=%d ；与 death_inhosp 不一致=%d" %
            (int(dl.sum()), int((dl ^ d["death_inhosp"].eq(1)).sum())))
    return dict(database="Huaian", n=n,
                cohort=dict(death_inhosp=int(d["death_inhosp"].sum()),
                            death_365=int(d["death_365"].sum()),
                            death_365_missing=int(d["death_365"].isna().sum())),
                contradictions=contra)


def main():
    res = {"generated_by": "scripts/84_death_label_crosscheck.py",
           "definition": {
               "death_inhosp_v2": "deathtime NOT NULL OR hospital_expire_flag=1 OR "
                                  "discharge_location matches death semantics OR "
                                  "[INSPIRE] inhosp_death_time NOT NULL OR "
                                  "[eICU] hospitaldischargestatus='Expired'",
               "death_365_v2": "death_inhosp_v2 OR death-time within [index, index+365d]",
               "death_time_priority": "deathtime > dod > inhosp_death_time > allcause_death_time"}}
    res["nwicu"] = audit_nwicu()
    res["mimic"] = audit_mimic()
    res["inspire"] = audit_inspire()
    res["eicu"] = audit_eicu()
    res["huaian"] = audit_huaian()

    with open(os.path.join(OUT, "death_label_crosscheck_v1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "_84_death_label_crosscheck.md"), "w", encoding="utf-8") as f:
        f.write("# 84 · 死亡标签日期级交叉核验\n\n```\n")
        f.write("\n".join(LOG))
        f.write("\n```\n")
    log("")
    log("→ 已写出 output/death_label_crosscheck_v1.json 与 output/_84_death_label_crosscheck.md")
    log("→ 新标签（不覆盖原文件）：data/nwicu_abi_labels_v2.csv, data/mimic_abi_labels_v2.csv, "
        "data/inspire_abi_labels_v2.csv")


if __name__ == "__main__":
    main()
