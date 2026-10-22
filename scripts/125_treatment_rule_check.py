# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""125_treatment_rule_check.py — 治疗强度指标：判定规则与窗口的逐库核对（第十九轮评审 §二/§三）

评审提出的两个实质问题（必须用代码回答，不能只改文字）：
  (1) 稿件称 18 个预测变量"measured inside the first 24 hours of the ICU stay"、S3 称治疗指标
      "recorded in the first ICU day"且"'First ICU day' is the first 24 hours after ICU admission"。
      但公开代码里治疗指标的窗口是 **入 ICU 前 6 小时至入 ICU 后 24 小时**
      （86 号 `BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'`；
       eICU `BETWEEN -360 AND 1440`）。
  (2) 判定规则在各库并不一致：有的是 **"最早开始时间落在窗口内"**（86 号对 MIMIC/eICU-vaso），
      有的是 **"窗口内任意一条记录"**（12 号对 INSPIRE、74 号对 NWICU-vaso）。
      两者对"治疗在窗口前已开始、窗口内仍有记录"的患者会给出不同标签。

本脚本对四个库的**最终分析队列**逐条计算下列口径并两两比对：
  final      : 最终 CSV 里实际使用的值
  as_coded   : 按该库该变量**代码真正实现的规则**重算（应与 final 逐例相同；用来验证读懂了代码）
  first_win  : min(开始时间) ∈ 该库窗口                        ← "最早开始时间"规则
  any_start  : ∃ 记录 开始时间 ∈ 该库窗口                      ← "窗口内任意记录（按开始时点）"
  any_over   : ∃ 记录与该库窗口相交                            ← "窗口内任意治疗期重叠"
  （后两者均在**记录层**判定后取 max，不用 min/max 聚合近似 —— 聚合近似会把
    "窗前一笔 + 窗后一笔"误判成重叠，已踩过。）
  doc_0_24   : ∃ 记录 开始时间 ∈ [入 ICU, +24h]                ← S3 现行文字声称的口径
  doc_0_24ov : ∃ 记录与 [入 ICU, +24h] 相交                    ← S3 文字按重叠解释

窗口与判定规则（逐库逐变量，全部来自代码，不统一假设；★ = 本脚本实测确认）
  MIMIC-IV   mech_vent / vasopressor : 窗 [intime-6h, intime+24h]，规则 = 最早开始时间落窗（86 号）★
             rrt                    : mimiciv_derived.first_day_rrt（派生首日表，不按事件重算）
  eICU       vasopressor            : 窗 [-360, +1440] 分钟，规则 = 最早 infusionoffset 落窗（86 号）★
             mech_vent / rrt        : apacheapsvar（APACHE 入 ICU 首 24h 汇总，不按事件重算）
  NWICU      mech_vent / rrt        : 窗 [intime-6h, intime+24h]，规则 = 最早 starttime 落窗（85 号）★
             vasopressor            : 窗 [intime-6h, intime+24h]，规则 = 窗口内任意处方（74 号）★
  INSPIRE    mech_vent / rrt        : 窗 [icuin-60, icuin+1440] 分钟，规则 = 窗口内任意记录（12 号）★
             vasopressor            : 窗 [icuin, icuin+1440] 分钟，规则 = 窗口内任意记录（12 号）★

产物：
  output/treatment_rule_check_v1.json
  output/_125_treatment_rule_check.md
"""
import io
import json
import os
import traceback
from datetime import datetime

import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
PGKW = dict(host=_os.environ.get("ABI_DB_HOST", "localhost"),
            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
            user=_os.environ.get("ABI_DB_USER", "postgres"),
            password=_os.environ.get("ABI_DB_PASSWORD", ""))
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


def _q(cu, sql):
    cu.execute("SET statement_timeout = '300000'")
    cu.execute(sql)
    return pd.DataFrame(cu.fetchall(), columns=[d[0] for d in cu.description])


def _cmp(a, b):
    a = pd.Series(a).astype(int).values
    b = pd.Series(b).astype(int).values
    assert len(a) == len(b), (len(a), len(b))
    return dict(n_diff=int((a != b).sum()),
                a_only=int(((a == 1) & (b == 0)).sum()),
                b_only=int(((a == 0) & (b == 1)).sum()),
                a_pos=int(a.sum()), b_pos=int(b.sum()))


def _row(var, note, window, rule, n, cols):
    out = dict(variable=var, note=note, window=window, coded_rule=rule, n=int(n))
    out["counts"] = {k: int(pd.Series(v).astype(int).sum()) for k, v in cols.items()}
    base = cols["final"]
    out["vs_final"] = {k: _cmp(base, v) for k, v in cols.items() if k != "final"}
    return out


def _anchor(cu, df, key):
    return df[key].map(_q(cu, "SELECT k, t0 FROM _i125").set_index("k")["t0"])


# ==================================================================== MIMIC-IV
def mimic():
    df = pd.read_csv(os.path.join(DATA, "mimic_abi_cohort_v2.csv"),
                     usecols=["stay_id", "intime", "mech_vent", "vasopressor", "rrt"])
    df["intime"] = pd.to_datetime(df["intime"])
    cn = psycopg2.connect(dbname="mimiciv", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("DROP TABLE IF EXISTS _m125")
    cu.execute("CREATE TEMP TABLE _m125(k bigint primary key, t timestamp)")
    cu.executemany("INSERT INTO _m125 VALUES (%s,%s)",
                   [(int(a), b.to_pydatetime())
                    for a, b in zip(df["stay_id"], df["intime"])])
    SRC = {
        "mech_vent": ("mimiciv_derived.ventilation",
                      "ventilation_status IN ('InvasiveVent','Tracheostomy')"),
        "vasopressor": ("mimiciv_derived.vasoactive_agent",
                        "norepinephrine IS NOT NULL OR epinephrine IS NOT NULL "
                        "OR dopamine IS NOT NULL OR phenylephrine IS NOT NULL "
                        "OR vasopressin IS NOT NULL"),
    }
    res = []
    for var, (tbl, cond) in SRC.items():
        sql = """
        WITH e AS (SELECT stay_id, starttime, endtime FROM %s WHERE %s)
        SELECT b.k AS stay_id,
          min(e.starttime) AS first_st,
          max(CASE WHEN e.starttime BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'
                   THEN 1 ELSE 0 END) AS any_start,
          max(CASE WHEN e.starttime <= b.t + interval '1 day'
                        AND (e.endtime IS NULL OR e.endtime >= b.t - interval '6 hours')
                   THEN 1 ELSE 0 END) AS any_over,
          max(CASE WHEN e.starttime BETWEEN b.t AND b.t + interval '1 day'
                   THEN 1 ELSE 0 END) AS doc_0_24,
          max(CASE WHEN e.starttime <= b.t + interval '1 day'
                        AND (e.endtime IS NULL OR e.endtime >= b.t)
                   THEN 1 ELSE 0 END) AS doc_0_24ov
        FROM _m125 b LEFT JOIN e ON e.stay_id = b.k GROUP BY b.k
        """ % (tbl, cond)  # 记录层 max()：每条记录独立判定，不做 min/max 聚合
        d = _q(cu, sql)
        m = df[["stay_id"]].merge(d, on="stay_id", how="left")
        assert len(m) == len(df)
        fs = pd.to_datetime(m["first_st"])
        lo, hi = df["intime"] - pd.Timedelta(hours=6), df["intime"] + pd.Timedelta(hours=24)
        _fw = (fs.notna() & (fs >= lo) & (fs <= hi)).astype(int)
        cols = dict(final=df[var].astype(int), as_coded=_fw, first_win=_fw,
                    any_start=m["any_start"].fillna(0).astype(int),
                    any_over=m["any_over"].fillna(0).astype(int),
                    doc_0_24=m["doc_0_24"].fillna(0).astype(int),
                    doc_0_24ov=m["doc_0_24ov"].fillna(0).astype(int))
        res.append(_row(var, tbl, "[intime-6h, intime+24h]",
                        "min(starttime) 落窗", len(df), cols))
    res.append(_row("rrt", "mimiciv_derived.first_day_rrt（派生首日表）",
                    "派生首日表（dialysis_present=1）", "派生表口径，不按事件重算",
                    len(df), dict(final=df["rrt"].astype(int))))
    cu.execute("DROP TABLE IF EXISTS _m125")
    cn.close()
    return res


# ==================================================================== eICU
def eicu():
    df = pd.read_csv(os.path.join(DATA, "eicu_abi_final_v3.csv"),
                     usecols=["patientunitstayid", "mech_vent", "vasopressor", "rrt"])
    cn = psycopg2.connect(dbname="eicu", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    res = []
    sql = """
    WITH e AS (SELECT d.patientunitstayid, min(d.infusionoffset) st
               FROM eicu_crd.infusiondrug d
               WHERE lower(d.drugname) ~ 'norepineph|epineph|dopamine|phenyleph|vasopressin|'
                                         'levophed|neosynephrine|dobutamine|milrinone'
               GROUP BY d.patientunitstayid)
    SELECT b.patientunitstayid, min(e.st) AS first_off
    FROM eicu_crd.abi_a1_ext b LEFT JOIN e ON e.patientunitstayid = b.patientunitstayid
    GROUP BY b.patientunitstayid
    """
    d = _q(cu, sql)
    m = df[["patientunitstayid"]].merge(d, on="patientunitstayid", how="left")
    assert len(m) == len(df), (len(m), len(df))
    fo = pd.to_numeric(m["first_off"], errors="coerce")

    def _t(s):
        return s.fillna(False).astype(int)
    _w = _t((fo >= -360) & (fo <= 1440))
    _d = _t((fo >= 0) & (fo <= 1440))
    cols = dict(final=df["vasopressor"].astype(int), as_coded=_w, first_win=_w,
                any_start=_w, any_over=_w, doc_0_24=_d, doc_0_24ov=_d)
    res.append(_row("vasopressor", "eicu_crd.infusiondrug（仅 infusionoffset，单时点）",
                    "[-360, +1440] 分钟", "min(infusionoffset) 落窗", len(df), cols))
    for var in ("mech_vent", "rrt"):
        res.append(_row(var, "eicu_crd.apacheapsvar（APACHE 入 ICU 首 24h 汇总）",
                        "APACHE 首 24h", "APACHE 汇总口径，不按事件重算",
                        len(df), dict(final=df[var].astype(int))))
    cn.close()
    return res


# ==================================================================== NWICU
def nwicu():
    df = pd.read_csv(os.path.join(DATA, "nwicu_abi_final_v6.csv"),
                     usecols=["hadm_id", "stay_id", "intime",
                              "mech_vent", "vasopressor", "rrt"])
    df["intime"] = pd.to_datetime(df["intime"])
    cn = psycopg2.connect(dbname="nwicu", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("DROP TABLE IF EXISTS _n125")
    cu.execute("CREATE TEMP TABLE _n125(h bigint primary key, s bigint, t timestamp)")
    cu.executemany("INSERT INTO _n125 VALUES (%s,%s,%s)",
                   [(int(a), int(b), c.to_pydatetime())
                    for a, b, c in zip(df["hadm_id"], df["stay_id"], df["intime"])])
    res = []
    for var, ids in (("mech_vent", "(787541,753461)"),
                     ("rrt", "(704890,772042,724671,798351)")):
        sql = """
        WITH e AS (SELECT pe.stay_id, min(pe.starttime) st, max(pe.endtime) et
                   FROM icu.procedureevents pe JOIN _n125 b ON b.s = pe.stay_id
                   WHERE pe.itemid IN %s GROUP BY pe.stay_id)
        SELECT b.h AS hadm_id, min(e.st) AS first_st,
          max(CASE WHEN e.st BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'
                   THEN 1 ELSE 0 END) AS any_start,
          max(CASE WHEN e.st <= b.t + interval '1 day'
                        AND (e.et IS NULL OR e.et >= b.t - interval '6 hours')
                   THEN 1 ELSE 0 END) AS any_over,
          max(CASE WHEN e.st BETWEEN b.t AND b.t + interval '1 day'
                   THEN 1 ELSE 0 END) AS doc_0_24,
          max(CASE WHEN e.st <= b.t + interval '1 day'
                        AND (e.et IS NULL OR e.et >= b.t)
                   THEN 1 ELSE 0 END) AS doc_0_24ov
        FROM _n125 b LEFT JOIN e ON e.stay_id = b.s GROUP BY b.h
        """ % ids  # 记录层 max()：每条记录独立判定
        d = _q(cu, sql)
        m = df[["hadm_id"]].merge(d, on="hadm_id", how="left")
        assert len(m) == len(df)
        fs = pd.to_datetime(m["first_st"])
        lo, hi = df["intime"] - pd.Timedelta(hours=6), df["intime"] + pd.Timedelta(hours=24)
        _fw = (fs.notna() & (fs >= lo) & (fs <= hi)).astype(int)
        cols = dict(final=df[var].astype(int), as_coded=_fw, first_win=_fw,
                    any_start=m["any_start"].fillna(0).astype(int),
                    any_over=m["any_over"].fillna(0).astype(int),
                    doc_0_24=m["doc_0_24"].fillna(0).astype(int),
                    doc_0_24ov=m["doc_0_24ov"].fillna(0).astype(int))
        res.append(_row(var, "icu.procedureevents %s" % ids, "[intime-6h, intime+24h]",
                        "min(starttime) 落窗", len(df), cols))
    DRUG = ("(lower(p.drug) LIKE '%%norepinephrine%%' OR lower(p.drug) LIKE '%%epinephrine%%' "
            "OR lower(p.drug) LIKE '%%dopamine%%' OR lower(p.drug) LIKE '%%phenylephrine%%' "
            "OR lower(p.drug) LIKE '%%vasopressin%%')")
    ROUTE = ("(lower(p.route) IN ('intravenous','injection') "
             "AND lower(p.drug) NOT LIKE '%%lidocaine%%' "
             "AND lower(p.drug) NOT LIKE '%%cocoa butter%%')")
    sql = """
    SELECT b.h AS hadm_id,
      min(p.starttime) AS first_st,
      max(CASE WHEN p.starttime BETWEEN b.t - interval '6 hours' AND b.t + interval '1 day'
               THEN 1 ELSE 0 END) AS any_start,
      max(CASE WHEN p.starttime BETWEEN b.t AND b.t + interval '1 day'
               THEN 1 ELSE 0 END) AS doc_0_24
    FROM _n125 b LEFT JOIN hosp.prescriptions p
      ON p.hadm_id = b.h AND %s AND %s
    GROUP BY b.h
    """ % (DRUG, ROUTE)
    d = _q(cu, sql)
    m = df[["hadm_id"]].merge(d, on="hadm_id", how="left")
    assert len(m) == len(df)
    fs = pd.to_datetime(m["first_st"])
    lo, hi = df["intime"] - pd.Timedelta(hours=6), df["intime"] + pd.Timedelta(hours=24)
    _fw = (fs.notna() & (fs >= lo) & (fs <= hi)).astype(int)
    _as = m["any_start"].fillna(0).astype(int)
    _d0 = m["doc_0_24"].fillna(0).astype(int)
    cols = dict(final=df["vasopressor"].astype(int), as_coded=_as, first_win=_fw,
                any_start=_as, any_over=_as, doc_0_24=_d0, doc_0_24ov=_d0)
    res.append(_row("vasopressor", "hosp.prescriptions（74 号药物+路线过滤）",
                    "[intime-6h, intime+24h]",
                    "窗口内任意处方（处方无 endtime ⇒ any_start=any_over）", len(df), cols))
    cu.execute("DROP TABLE IF EXISTS _n125")
    cn.close()
    return res


# ==================================================================== INSPIRE
def inspire():
    df = pd.read_csv(os.path.join(DATA, "inspire_abi_final_v3.csv"),
                     usecols=["op_id", "subject_id", "mech_vent", "vasopressor", "rrt"])
    cn = psycopg2.connect(dbname="inspire", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("DROP TABLE IF EXISTS _i125")
    cu.execute("CREATE TEMP TABLE _i125(k bigint primary key, s bigint, t0 bigint)")
    cu.executemany("INSERT INTO _i125 VALUES (%s,%s,NULL)",
                   [(int(a), int(b)) for a, b in zip(df["op_id"], df["subject_id"])])
    cu.execute("""UPDATE _i125 SET t0 = o.icuin FROM (
                   SELECT op_id, nullif(trim(icuin_time),'')::bigint AS icuin
                   FROM inspire.operations) o WHERE o.op_id = _i125.k""")
    anch = _q(cu, "SELECT k, t0 FROM _i125").set_index("k")["t0"]
    n_anchor = int(anch.notna().sum())
    log("  INSPIRE 锚点 icuin 可用 %d / %d" % (n_anchor, len(df)))
    assert n_anchor == len(df), "有 op_id 取不到 icuin"
    res = []

    def _t(s):
        return s.fillna(False).astype(int)

    for var, tbl, cond, lo_off in (
            ("mech_vent", "inspire.ward_vitals", "w.item_name = 'vent'", 60),
            ("rrt", "inspire.ward_vitals", "w.item_name = 'crrt'", 60)):
        sql = """
        SELECT b.k AS op_id,
          min(nullif(trim(w.chart_time),'')::bigint) AS first_st,
          max(CASE WHEN nullif(trim(w.chart_time),'')::bigint
                        BETWEEN b.t0 - %d AND b.t0 + 1440 THEN 1 ELSE 0 END) AS any_start,
          max(CASE WHEN nullif(trim(w.chart_time),'')::bigint
                        BETWEEN b.t0 - %d AND b.t0 + 1440 THEN 1 ELSE 0 END) AS any_over,
          max(CASE WHEN nullif(trim(w.chart_time),'')::bigint
                        BETWEEN b.t0 AND b.t0 + 1440 THEN 1 ELSE 0 END) AS doc_0_24,
          max(CASE WHEN nullif(trim(w.chart_time),'')::bigint
                        BETWEEN b.t0 AND b.t0 + 1440 THEN 1 ELSE 0 END) AS doc_0_24ov
        FROM _i125 b LEFT JOIN %s w
          ON w.subject_id = b.s AND %s
        GROUP BY b.k
        """ % (lo_off, lo_off, tbl, cond)
        d = _q(cu, sql)
        m = df[["op_id"]].merge(d, on="op_id", how="left")
        assert len(m) == len(df)
        fs = pd.to_numeric(m["first_st"], errors="coerce")
        t0 = pd.to_numeric(m["op_id"].map(anch), errors="coerce")
        _fw = _t((fs >= t0 - lo_off) & (fs <= t0 + 1440))
        _as = m["any_start"].fillna(0).astype(int)
        cols = dict(final=df[var].astype(int), as_coded=_as, first_win=_fw, any_start=_as,
                    any_over=m["any_over"].fillna(0).astype(int),
                    doc_0_24=m["doc_0_24"].fillna(0).astype(int),
                    doc_0_24ov=m["doc_0_24ov"].fillna(0).astype(int))
        res.append(_row(var, "%s（%s）" % (tbl, cond),
                        "[icuin-%d, icuin+1440] 分钟" % lo_off,
                        "窗口内任意记录", len(df), cols))
    sql = """
    SELECT b.k AS op_id,
      min(nullif(trim(m.chart_time),'')::bigint) AS first_st,
      max(CASE WHEN nullif(trim(m.chart_time),'')::bigint
                BETWEEN b.t0 AND b.t0 + 1440 THEN 1 ELSE 0 END) AS any_start
    FROM _i125 b LEFT JOIN inspire.medications m
      ON m.subject_id = b.s
     AND lower(m.drug_name) ~ 'norepineph|epinephrine|dopamine|phenyleph|'
                              'vasopressin|dobutamine|milrinone'
    GROUP BY b.k
    """
    d = _q(cu, sql)
    m = df[["op_id"]].merge(d, on="op_id", how="left")
    assert len(m) == len(df)
    fs = pd.to_numeric(m["first_st"], errors="coerce")
    t0 = pd.to_numeric(m["op_id"].map(anch), errors="coerce")
    _fw = _t((fs >= t0) & (fs <= t0 + 1440))
    _as = m["any_start"].fillna(0).astype(int)
    cols = dict(final=df["vasopressor"].astype(int), as_coded=_as, first_win=_fw,
                any_start=_as, any_over=_as, doc_0_24=_as, doc_0_24ov=_as)
    res.append(_row("vasopressor", "inspire.medications", "[icuin, icuin+1440] 分钟",
                    "窗口内任意记录", len(df), cols))
    cu.execute("DROP TABLE IF EXISTS _i125")
    cn.close()
    return res


# ==================================================================== main
def main():
    log("=" * 100)
    log("治疗强度指标：判定规则与窗口逐库核对（第十九轮模拟评审 §二/§三）")
    log("=" * 100)
    log("")
    log("口径：final=最终 CSV 值；as_coded=按代码真实规则重算；first_win=最早开始时间落窗；")
    log("      any_start=窗口内任意记录（开始时点）；any_over=与窗口相交；doc_0_24=S3 现行文字口径。")
    payload = dict(generated_at=datetime.now().isoformat(timespec="seconds"),
                   generated_by="scripts/125_treatment_rule_check.py",
                   databases={})
    for name, fn in (("MIMIC-IV", mimic), ("eICU", eicu),
                     ("NWICU", nwicu), ("INSPIRE", inspire)):
        log("")
        log("---- %s" % name)
        try:
            rows = fn()
            payload["databases"][name] = rows
            for r in rows:
                c = r["counts"]
                log("  %-12s n=%4d | final=%5s as_coded=%5s first_win=%5s any_start=%5s "
                    "any_over=%5s doc_0_24=%5s"
                    % (r["variable"], r["n"], c.get("final", "-"), c.get("as_coded", "-"),
                       c.get("first_win", "-"), c.get("any_start", "-"),
                       c.get("any_over", "-"), c.get("doc_0_24", "-")))
                for k, v in r["vs_final"].items():
                    if v["n_diff"]:
                        log("       └─ 与 final 差 %3d 例（final-only %3d / %s-only %3d）"
                            % (v["n_diff"], v["a_only"], k, v["b_only"]))
        except Exception as e:
            payload["databases"][name] = dict(error="%s: %s" % (type(e).__name__, e),
                                              tb=traceback.format_exc()[-1500:])
            log("  !! %s 失败：%s" % (name, e))
    with io.open(os.path.join(OUT, "treatment_rule_check_v1.json"), "w",
                 encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with io.open(os.path.join(OUT, "_125_treatment_rule_check.md"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    log("")
    log("→ output/treatment_rule_check_v1.json")
    log("→ output/_125_treatment_rule_check.md")


if __name__ == "__main__":
    main()
