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
86 —— V9 分析数据集重建（用户决策已下：MIMIC 同步改首日口径；INSPIRE 一年死亡取保守 265）

修改内容（四个库逐一定性后的结论）
----------------------------------
| 库      | 治疗指标            | 现行口径 | 处理                              |
|---------|---------------------|----------|-----------------------------------|
| MIMIC   | mech_vent           | 全程     | → 首日（ventilation 起止时间限定） |
| MIMIC   | vasopressor         | 全程     | → 首日（vasoactive_agent 限定）    |
| MIMIC   | rrt                 | 首日     | 不变（first_day_rrt）              |
| NWICU   | mech_vent / rrt     | 全程     | 85 已重建（nwicu_abi_final_v3.csv）|
| NWICU   | vasopressor         | 首日     | 74 已修，不变                      |
| eICU    | mech_vent / rrt     | 首日     | APACHE 首个 24h，不变              |
| eICU    | vasopressor         | 全程     | → 首日（infusiondrug offset 限定） |
| INSPIRE | vent / crrt / vaso  | 首日     | 不变                               |

时间窗定义（与 NWICU vasopressor 一致，四库统一）
------------------------------------------------
  start ∈ [index − 6h, index + 1day]
  eICU 用 offset（分钟，相对 ICU 入组）：startoffset ∈ [−360, 1440]

结局标签
--------
  MIMIC   death_hosp 2,799 → 2,858；death_365 5,301 → 5,303（84 号脚本 v2 标签）
  NWICU   death_hosp 16 → 378；death_365 635 → 701
  INSPIRE death_365 262 → **265（保守）**：仅当 inhosp_death_time 或 allcause_death_time
          落在 [icuin, icuin+365d] 内才计 1。12 例 inhosp−icuin 超 365 天且 disch < icuin
          → 时间锚点存疑，**不计入**，另设 `time_anchor_suspect` 标记列供附录说明。
  eICU    院内死亡 1,894，与源字段完全一致，不变。

产物（一律新建，不覆盖任何既有文件）
------------------------------------
  data/mimic_abi_cohort_v2.csv
  data/eicu_abi_final_v2.csv
  data/inspire_abi_final_v2.csv
  output/dataset_rebuild_v1.json、output/_86_rebuild_datasets.md
"""
import json
import os
from pathlib import Path

import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
DATA = ROOT / "data"
PGKW = dict(host=_os.environ.get("ABI_DB_HOST", "localhost"),
            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
            user=_os.environ.get("ABI_DB_USER", "postgres"),
            password=_os.environ.get("ABI_DB_PASSWORD", ""))

LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


class Pg:
    def __init__(self, db):
        self.c = psycopg2.connect(dbname=db, **PGKW)
        self.cur = self.c.cursor()
        self.cur.execute("set statement_timeout=600000")

    def idx(self, keys, times):
        self.cur.execute("DROP TABLE IF EXISTS _idx")
        self.cur.execute("CREATE TEMP TABLE _idx(k bigint, t timestamp)")
        self.cur.executemany("INSERT INTO _idx VALUES (%s,%s)",
                             [(int(a), None if pd.isna(b) else b.to_pydatetime())
                              for a, b in zip(keys, times)])
        self.c.commit()

    def q(self, sql):
        return pd.read_sql(sql, self.c)

    def close(self):
        self.c.close()


WIN_START = "e.st BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'"


def first_day_flag(p, src_sql, keycol, df):
    """按首日窗口计算 0/1 标志，并按 keycol 对齐回原顺序"""
    sql = """
      WITH e AS (%s)
      SELECT b.k AS %s,
        max(CASE WHEN %s THEN 1 ELSE 0 END) AS w_start,
        max(CASE WHEN e.st IS NOT NULL THEN 1 ELSE 0 END) AS w_any
      FROM _idx b LEFT JOIN e ON e.%s = b.k GROUP BY b.k
    """ % (src_sql, keycol, WIN_START.replace("e.st", "e.st").replace("b.t", "b.t"), keycol)
    sql = """
      WITH e AS (%s)
      SELECT b.k AS %s,
        max(CASE WHEN e.st BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'
                 THEN 1 ELSE 0 END) AS w_start,
        max(CASE WHEN e.st IS NOT NULL THEN 1 ELSE 0 END) AS w_any
      FROM _idx b LEFT JOIN e ON e.%s = b.k GROUP BY b.k
    """ % (src_sql, keycol, keycol)
    d = p.q(sql)
    m = df[[keycol]].merge(d, on=keycol, how="left").fillna(0)
    return m["w_start"].astype(int).values, m["w_any"].astype(int).values


# ============================================================ MIMIC-IV
def rebuild_mimic():
    log("=" * 78)
    log("MIMIC-IV —— 首日治疗指标 + 修正死亡标签")
    log("=" * 78)
    df = pd.read_csv(DATA / "mimic_abi_cohort.csv", parse_dates=["intime"])
    lab = pd.read_csv(DATA / "mimic_abi_labels_v2.csv")
    p = Pg("mimiciv")
    p.idx(df["stay_id"], df["intime"])
    out = {}

    vent_sql = """SELECT v.stay_id, min(v.starttime) st, max(v.endtime) et
                  FROM mimiciv_derived.ventilation v JOIN _idx b ON b.k = v.stay_id
                  WHERE v.ventilation_status IN ('InvasiveVent','Tracheostomy')
                  GROUP BY v.stay_id"""
    vaso_sql = """SELECT v.stay_id, min(v.starttime) st, max(v.endtime) et
                  FROM mimiciv_derived.vasoactive_agent v JOIN _idx b ON b.k = v.stay_id
                  WHERE v.norepinephrine IS NOT NULL OR v.epinephrine IS NOT NULL
                     OR v.dopamine IS NOT NULL OR v.phenylephrine IS NOT NULL
                     OR v.vasopressin IS NOT NULL
                  GROUP BY v.stay_id"""
    for name, sql in (("mech_vent", vent_sql), ("vasopressor", vaso_sql)):
        w_start, w_any = first_day_flag(p, sql, "stay_id", df)
        cur = df[name].astype(int).values
        assert (cur == w_any).all(), "%s：现行值与全程口径不一致，需复核" % name
        out[name] = dict(before=int(cur.sum()), before_pct=round(100 * float(cur.mean()), 1),
                         after=int(w_start.sum()), after_pct=round(100 * float(w_start.sum() / len(df) * 100), 1),
                         reclassified=int((cur - w_start > 0).sum()))
        log("  %-12s 全程 %5d (%.1f%%) → 首日 %5d (%.1f%%) ；1→0 改判 %d 例" %
            (name, out[name]["before"], out[name]["before_pct"], out[name]["after"],
             100 * w_start.sum() / len(df), out[name]["reclassified"]))
        df[name + "_orig"] = cur
        df[name] = w_start
    p.close()

    m = df.merge(lab[["stay_id", "death_inhosp_v2", "death_365_v2"]], on="stay_id", how="left")
    assert m["death_inhosp_v2"].notna().all(), "标签 merge 缺失"
    out["labels"] = dict(death_hosp_before=int(m["death_hosp"].sum()),
                         death_hosp_after=int(m["death_inhosp_v2"].sum()),
                         death_365_before=int(m["death_365"].sum()),
                         death_365_after=int(m["death_365_v2"].sum()))
    m["death_hosp_orig"] = m["death_hosp"].astype(int)
    m["death_365_orig"] = m["death_365"].astype(int)
    m["death_hosp"] = m["death_inhosp_v2"].astype(int)
    m["death_365"] = m["death_365_v2"].astype(int)
    m = m.drop(columns=["death_inhosp_v2", "death_365_v2"])
    m.to_csv(DATA / "mimic_abi_cohort_v2.csv", index=False)
    log("  标签：death_hosp %d→%d ；death_365 %d→%d" %
        (out["labels"]["death_hosp_before"], out["labels"]["death_hosp_after"],
         out["labels"]["death_365_before"], out["labels"]["death_365_after"]))
    log("  → data/mimic_abi_cohort_v2.csv (n=%d)" % len(m))
    return out


# ============================================================ eICU
def rebuild_eicu():
    log("")
    log("=" * 78)
    log("eICU —— 血管活性药改首日窗口（offset ∈ [-360, 1440] 分钟）")
    log("=" * 78)
    df = pd.read_csv(DATA / "eicu_abi_final.csv")
    p = Pg("eicu")
    # eICU infusiondrug 只有 infusionoffset（单一时点，无起止），故首日窗口只能用该时点判定
    sql = """
      WITH e AS (SELECT d.patientunitstayid, min(d.infusionoffset) st
                 FROM eicu_crd.infusiondrug d JOIN eicu_crd.abi_a1_ext b
                   ON b.patientunitstayid = d.patientunitstayid
                 WHERE lower(d.drugname) ~ 'norepineph|epineph|dopamine|phenyleph|vasopressin|'
                                           'levophed|neosynephrine|dobutamine|milrinone'
                 GROUP BY d.patientunitstayid)
      SELECT b.patientunitstayid,
        CASE WHEN e.st IS NULL THEN 0 ELSE 1 END AS w_any,
        CASE WHEN e.st BETWEEN -360 AND 1440 THEN 1 ELSE 0 END AS w_start
      FROM eicu_crd.abi_a1_ext b LEFT JOIN e ON e.patientunitstayid = b.patientunitstayid
    """
    d = p.q(sql)
    p.close()
    m = df[["patientunitstayid"]].merge(d, on="patientunitstayid", how="left").fillna(0)
    cur = df["vasopressor"].astype(int).values
    res = dict(before=int(cur.sum()), before_pct=round(100 * float(cur.mean()), 1),
               w_any=int(m["w_any"].sum()),
               after=int(m["w_start"].sum()),
               after_pct=round(100 * float(m["w_start"].mean()), 1),
               reclassified=int((cur - m["w_start"].values > 0).sum()))
    log("  vasopressor 全程 %d (%.1f%%) → 首日 %d (%.1f%%) ；1→0 改判 %d 例" %
        (res["before"], res["before_pct"], res["after"], res["after_pct"], res["reclassified"]))
    log("  （现行值与全程口径%s）" % ("一致，确认无时间窗" if res["before"] == res["w_any"] else "不一致，需复核"))
    df["vasopressor_orig"] = cur
    df["vasopressor"] = m["w_start"].astype(int).values
    df.to_csv(DATA / "eicu_abi_final_v2.csv", index=False)
    log("  → data/eicu_abi_final_v2.csv (n=%d)" % len(df))
    return res


# ============================================================ INSPIRE
DAY = 1440


def rebuild_inspire():
    log("")
    log("=" * 78)
    log("INSPIRE —— 一年死亡取保守口径 265 + 时间锚点存疑标记")
    log("=" * 78)
    p = Pg("inspire")
    op = p.q("""SELECT op_id,
        nullif(trim(icuin_time),'')::bigint AS icuin,
        nullif(trim(icuout_time),'')::bigint AS icuout,
        nullif(trim(discharge_time),'')::bigint AS disch,
        nullif(trim(inhosp_death_time),'')::bigint AS inhosp_death,
        nullif(trim(allcause_death_time),'')::bigint AS allcause_death
        FROM inspire.operations""")
    p.close()
    df = pd.read_csv(DATA / "inspire_abi_final.csv")
    dh = pd.read_csv(DATA / "inspire_deathhosp.csv")
    m = df.merge(dh, on="op_id", how="left").merge(op, on="op_id", how="left")
    m["death_hosp"] = m["death_hosp"].fillna(0).astype(int)

    ih = m["inhosp_death"]
    ac = m["allcause_death"]
    win = 365 * DAY
    ih_in = ih.notna() & (ih - m["icuin"]).between(0, win)
    ac_in = ac.notna() & (ac - m["icuin"]).between(0, win)
    new = (ih_in | ac_in).astype(int)

    # ★ 锚点存疑只取"与死亡时间直接冲突"的两类；
    #   disch < icuin 属 discharge_time 字段语义问题（不影响标签），单列不计入。
    suspect = (ih.notna() & ((ih - m["icuin"]) > win)) | \
              (ih.notna() & m["disch"].notna() & (ih > m["disch"] + 60))

    res = dict(before=int(m["death_365"].sum()), after=int(new.sum()),
               n_suspect=int(suspect.sum()),
               hosp_death_not_in_365=int((m["death_hosp"].eq(1) & new.eq(0)).sum()),
               allcause_missing=int(ac.isna().sum()),
               ih_beyond_window=int((ih.notna() & ((ih - m["icuin"]) > win)).sum()),
               disch_before_icuin=int((m["disch"].notna() & (m["disch"] < m["icuin"])).sum()))
    log("  death_365 现行 %d → 保守 %d (Δ%+d)" % (res["before"], res["after"],
                                                 res["after"] - res["before"]))
    log("  时间锚点存疑（inhosp 超 365 天 / inhosp 晚于出院）合计 %d 例" % res["n_suspect"])
    log("    ├ inhosp−icuin 超 365 天   : %d" % res["ih_beyond_window"])
    log("    ├ disch < icuin           : %d  ← discharge_time 字段语义问题，不影响标签，单列"
        % res["disch_before_icuin"])
    log("    └ 院内死亡未计入一年死亡   : %d（即那 12 例，附录须说明）"
        % res["hosp_death_not_in_365"])
    log("  allcause_death_time 缺失 %d 例（其中落在院内死亡中的由 inhosp 补齐）"
        % res["allcause_missing"])

    m["death_365_orig"] = m["death_365"].astype(int)
    m["death_365"] = new
    m["time_anchor_suspect"] = suspect.astype(int)
    m["disch_before_icuin"] = (m["disch"].notna() & (m["disch"] < m["icuin"])).astype(int)
    cols = [c for c in m.columns if c not in
            ("icuin", "icuout", "disch", "inhosp_death", "allcause_death")]
    m[cols].to_csv(DATA / "inspire_abi_final_v2.csv", index=False)
    log("  → data/inspire_abi_final_v2.csv (n=%d)" % len(m))
    return res


def main():
    payload = dict(generated_by="scripts/86_rebuild_datasets.py",
                   window="start in [index-6h, index+1day]; eICU offset in [-360,1440] min",
                   mimic=rebuild_mimic(), eicu=rebuild_eicu(), inspire=rebuild_inspire())
    nw = DATA / "nwicu_abi_final_v3.csv"
    payload["nwicu_v3_present"] = bool(nw.exists())
    log("")
    log("  NWICU 重建数据已存在：%s（85 号脚本产出）" % nw.exists())
    with open(OUT / "dataset_rebuild_v1.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with open(OUT / "_86_rebuild_datasets.md", "w", encoding="utf-8") as f:
        f.write("# 86 · V9 分析数据集重建\n\n```\n" + "\n".join(LOG) + "\n```\n")
    log("→ output/dataset_rebuild_v1.json / _86_rebuild_datasets.md 已写出")


if __name__ == "__main__":
    main()
