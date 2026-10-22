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
85 —— 器官支持（治疗强度）指标的「时间窗」审计与重算（评审意见 A2，并外推至全部数据库）

背景
----
评审抽查 sql/11_nwicu_cohort.sql，指出 vent / rrt 两个 CTE 只按 stay_id+itemid 取记录、
没有首日时间条件，而同一脚本的 vit / lab 都有首日窗口 → 与正文「first-day predictors」冲突。

本脚本把该审计**外推到全部四个库**，因为一旦只有部分库修正，跨库可比性会更差：

  MIMIC-IV  mech_vent  ← mimiciv_derived.ventilation       （全程，无窗）  ★ 开发集
            vasopressor← mimiciv_derived.vasoactive_agent  （全程，无窗）  ★ 开发集
            rrt        ← mimiciv_derived.first_day_rrt     （首日，OK）
  NWICU     mech_vent  ← icu.procedureevents 787541/753461 （全程，无窗）
            rrt        ← icu.procedureevents 704890/...    （全程，无窗）
            vasopressor← 74 号脚本已修正为 intime-6h~+1d   （首日，OK）
  eICU      mech_vent/rrt ← apacheapsvar（APACHE 入 ICU 首个 24h，OK）
  INSPIRE   vent/crrt  ← ward_vitals，窗口 icuin-60~icuin+1440（首日，OK）

三种口径
--------
  W_any      ：现行口径，整个住院/ICU 期间（无时间条件）
  W_start    ：starttime ∈ [index-6h, index+1day]                     —— 与 NWICU vasopressor 一致
  W_ongoing  ：starttime <= index+1day 且 (endtime 缺失 或 endtime >= index)  —— 含跨窗持续治疗

产物（不覆盖既有文件）
----------------------
  data/nwicu_abi_final_v3.csv   NWICU 重建数据：新死亡标签 + 三种口径的 vent/rrt
  output/treatment_window_audit_v1.json
  output/_85_treatment_window.md
"""
import json
import os

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


class Pg:
    def __init__(self, db):
        self.c = psycopg2.connect(dbname=db, **PGKW)
        self.cur = self.c.cursor()
        self.cur.execute("set statement_timeout=600000")

    def temp_index(self, keys, times):
        """把 (key, index_time) 灌进临时表，供 SQL join 使用（避免超长 IN 列表）"""
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


# --------------------------------------------------------------- NWICU
VENT_NW = (787541, 753461)
RRT_NW = (704890, 772042, 724671, 798351)


def nwicu_variants():
    log("=" * 78)
    log("NWICU —— vent / rrt 时间窗重算（评审 A2）")
    log("=" * 78)
    df = pd.read_csv(os.path.join(DATA, "nwicu_abi_final_vp2.csv"), parse_dates=["intime"])
    p = Pg("nwicu")
    p.temp_index(df["stay_id"], df["intime"])
    base = p.q("SELECT _idx.k AS stay_id FROM _idx")
    log("")
    log("  队列 n=%d ；procedureevents 中相关 itemid 命中情况：" % len(df))
    for name, ids in (("vent", VENT_NW), ("rrt", RRT_NW)):
        r = p.q("SELECT itemid, count(*) n FROM icu.procedureevents WHERE itemid IN %s GROUP BY 1"
                % (tuple(ids),))
        log("    %-5s itemid=%s → %s" % (name, list(ids),
                                         ", ".join("%d:%d" % (a, b) for a, b in r.values)))

    res = {}
    for var, ids in (("mech_vent", VENT_NW), ("rrt", RRT_NW)):
        ids_s = "(%s)" % ",".join(str(i) for i in ids)
        sql = """
        WITH e AS (
          SELECT pe.stay_id, min(pe.starttime) st, max(pe.endtime) et
          FROM icu.procedureevents pe JOIN _idx b ON b.k = pe.stay_id
          WHERE pe.itemid IN %s GROUP BY pe.stay_id)
        SELECT b.k AS stay_id,
          max(CASE WHEN e.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS w_any,
          max(CASE WHEN e.st BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'
                   THEN 1 ELSE 0 END) AS w_start,
          max(CASE WHEN e.st <= b.t + interval '1 day'
                        AND (e.et IS NULL OR e.et >= b.t) THEN 1 ELSE 0 END) AS w_ongoing
        FROM _idx b LEFT JOIN e ON e.stay_id = b.k GROUP BY b.k
        """ % ids_s
        d = p.q(sql)
        m = df[["stay_id"]].merge(d, on="stay_id", how="left").fillna(0)
        cur_val = df[var].astype(int)
        res[var] = dict(
            current_any=int(cur_val.sum()),
            current_pct=round(100 * float(cur_val.mean()), 1),
            w_any=int(m["w_any"].sum()),
            w_start=int(m["w_start"].sum()),
            w_start_pct=round(100 * float(m["w_start"].mean()), 1),
            w_ongoing=int(m["w_ongoing"].sum()),
            w_ongoing_pct=round(100 * float(m["w_ongoing"].mean()), 1),
            reclassified_any_minus_start=int((cur_val.values - m["w_start"].values > 0).sum()),
        )
        log("")
        log("  %s（现行=全住院期口径）" % var)
        log("    现行 w_any        : %5d (%.1f%%)" % (res[var]["current_any"], res[var]["current_pct"]))
        log("    重算 w_any        : %5d" % res[var]["w_any"])
        log("    首日 w_start      : %5d (%.1f%%)   ← 主口径（与 vasopressor 一致）"
            % (res[var]["w_start"], res[var]["w_start_pct"]))
        log("    含跨窗 w_ongoing  : %5d (%.1f%%)" % (res[var]["w_ongoing"], res[var]["w_ongoing_pct"]))
        log("    由 1 改判为 0 的例数 : %d" % res[var]["reclassified_any_minus_start"])
        df[var + "_w_start"] = m["w_start"].astype(int).values
        df[var + "_w_ongoing"] = m["w_ongoing"].astype(int).values
        df[var + "_anytime"] = cur_val.values
    p.close()
    return df, res


def build_nwicu_v3(df):
    lab = pd.read_csv(os.path.join(DATA, "nwicu_abi_labels_v2.csv"))
    m = df.merge(lab[["stay_id", "death_inhosp_v2", "death_365_v2"]], on="stay_id", how="left")
    assert m["death_inhosp_v2"].notna().all(), "标签 merge 有缺失"
    out = m.copy()
    # 主口径：治疗强度取首日；结局取修正标签；旧值保留以便回溯
    out["death_hosp_orig"] = out["death_hosp"].astype(int)
    out["death_365_orig"] = out["death_365"].astype(int)
    out["death_hosp"] = out["death_inhosp_v2"].astype(int)
    out["death_365"] = out["death_365_v2"].astype(int)
    out["mech_vent_orig"] = out["mech_vent"].astype(int)
    out["rrt_orig"] = out["rrt"].astype(int)
    out["mech_vent"] = out["mech_vent_w_start"].astype(int)
    out["rrt"] = out["rrt_w_start"].astype(int)
    path = os.path.join(DATA, "nwicu_abi_final_v3.csv")
    out.to_csv(path, index=False)
    log("")
    log("  → 已写出 %s （n=%d）" % (path, len(out)))
    log("    death_hosp %d→%d ；death_365 %d→%d ；mech_vent %d→%d ；rrt %d→%d" %
        (int(out["death_hosp_orig"].sum()), int(out["death_hosp"].sum()),
         int(out["death_365_orig"].sum()), int(out["death_365"].sum()),
         int(out["mech_vent_orig"].sum()), int(out["mech_vent"].sum()),
         int(out["rrt_orig"].sum()), int(out["rrt"].sum())))
    return path


# --------------------------------------------------------------- MIMIC-IV（只量化）
def mimic_variants():
    log("")
    log("=" * 78)
    log("MIMIC-IV —— 开发队列 mech_vent / vasopressor 是否也是全程口径（外推审计）")
    log("=" * 78)
    df = pd.read_csv(os.path.join(DATA, "mimic_abi_cohort.csv"), parse_dates=["intime"])
    p = Pg("mimiciv")
    p.temp_index(df["stay_id"], df["intime"])
    res = {}
    spec = [("mech_vent", """SELECT v.stay_id, min(v.starttime) st, max(v.endtime) et
                             FROM mimiciv_derived.ventilation v JOIN _idx b ON b.k=v.stay_id
                             WHERE v.ventilation_status IN ('InvasiveVent','Tracheostomy')
                             GROUP BY v.stay_id"""),
            ("vasopressor", """SELECT v.stay_id, min(v.starttime) st, max(v.endtime) et
                              FROM mimiciv_derived.vasoactive_agent v JOIN _idx b ON b.k=v.stay_id
                              WHERE v.norepinephrine IS NOT NULL OR v.epinephrine IS NOT NULL
                                 OR v.dopamine IS NOT NULL OR v.phenylephrine IS NOT NULL
                                 OR v.vasopressin IS NOT NULL
                              GROUP BY v.stay_id""")]
    for name, src in spec:
        d = p.q("""
          WITH e AS (%s)
          SELECT b.k AS stay_id,
            max(CASE WHEN e.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS w_any,
            max(CASE WHEN e.st BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'
                     THEN 1 ELSE 0 END) AS w_start,
            max(CASE WHEN e.st <= b.t + interval '1 day'
                          AND (e.et IS NULL OR e.et >= b.t) THEN 1 ELSE 0 END) AS w_ongoing
          FROM _idx b LEFT JOIN e ON e.stay_id=b.k GROUP BY b.k""" % src)
        m = df[["stay_id"]].merge(d, on="stay_id", how="left").fillna(0)
        cur = df[name].astype(int)
        res[name] = dict(
            current=int(cur.sum()), current_pct=round(100 * float(cur.mean()), 1),
            w_any=int(m["w_any"].sum()),
            w_start=int(m["w_start"].sum()), w_start_pct=round(100 * float(m["w_start"].mean()), 1),
            w_ongoing=int(m["w_ongoing"].sum()),
            w_ongoing_pct=round(100 * float(m["w_ongoing"].mean()), 1),
            reclassified=int((cur.values - m["w_start"].values > 0).sum()),
            newly_positive=int((m["w_start"].values - cur.values > 0).sum()))
        log("")
        log("  %s" % name)
        log("    现行（全程）      : %5d (%.1f%%)" % (res[name]["current"], res[name]["current_pct"]))
        log("    重算 全程         : %5d" % res[name]["w_any"])
        log("    首日 w_start      : %5d (%.1f%%)" % (res[name]["w_start"], res[name]["w_start_pct"]))
        log("    含跨窗 w_ongoing  : %5d (%.1f%%)" % (res[name]["w_ongoing"], res[name]["w_ongoing_pct"]))
        log("    1→0 改判 %d 例 ；0→1 改判 %d 例" %
            (res[name]["reclassified"], res[name]["newly_positive"]))
        log("    ★ 现行值与全程口径%s" %
            ("一致（确认无时间窗）" if res[name]["current"] == res[name]["w_any"] else "不一致，需另查"))
    p.close()
    return res


# --------------------------------------------------------------- eICU / INSPIRE 确认
def others():
    log("")
    log("=" * 78)
    log("eICU / INSPIRE —— 确认治疗指标本身已是首日口径")
    log("=" * 78)
    log("""
  eICU     mech_vent = apacheapsvar.vent/intubated（APACHE 入 ICU 首个 24h）      → 首日
           rrt       = apacheapsvar.dialysis                                     → 首日
  INSPIRE  vent/crrt = ward_vitals，chart_time ∈ [icuin-60, icuin+1440] 分钟      → 首日
  结论：这两个库无需重算；只有 MIMIC-IV（开发）与 NWICU（外部）需要修正。
""")
    return dict(eicu="first-day (APACHE 24h)", inspire="first-day (icuin-60 ~ icuin+1440 min)")


def main():
    nw_df, nw_res = nwicu_variants()
    v3 = build_nwicu_v3(nw_df)
    mim = mimic_variants()
    oth = others()

    payload = dict(generated_by="scripts/85_treatment_window_audit.py",
                   nwicu=nw_res, mimiciv=mim, others=oth,
                   nwicu_v3_path=os.path.relpath(v3, ROOT).replace("\\", "/"))
    with open(os.path.join(OUT, "treatment_window_audit_v1.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "_85_treatment_window.md"), "w", encoding="utf-8") as f:
        f.write("# 85 · 器官支持指标时间窗审计\n\n```\n" + "\n".join(LOG) + "\n```\n")
    log("")
    log("→ output/treatment_window_audit_v1.json / output/_85_treatment_window.md 已写出")


if __name__ == "__main__":
    main()
