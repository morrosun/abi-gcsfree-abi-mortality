# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""126_treatment_rule_sensitivity.py — 治疗指标判定规则变体对外部验证的影响（第十九轮评审 §三）

背景：125 号已证实各库治疗指标的**窗口与判定规则并不统一**（有"最早开始时间落窗"，
也有"窗口内任意记录"），且两种规则会给出不同患者标签（差异集中在 INSPIRE：mech_vent
−185 / rrt −11 / vasopressor −50；NWICU vasopressor −6；MIMIC +1~+2）。

本脚本回答评审的下一问：**若把每个库的规则换成另一种，已发表的外部验证指标会变多少？**
做法：冻结模型不变（不重训），只把受影响队列的治疗标志换成"另一规则"重算，
与 95 号产出的 external_validation_v9.csv 逐项对比（as-coded 一栏必须逐位复现，作为自检）。

口径：
  as_coded : 最终 CSV 的值（= 各库代码真实规则）
  alt      : 该库该变量的"另一规则"
              MIMIC-IV   mech_vent/vasopressor : 最早开始时间 → 窗口内任意记录(any_start)
                                                 （另报 any_over = 与窗口相交）
              NWICU      vasopressor           : 窗口内任意处方 → 最早开始时间落窗
                        mech_vent/rrt          : 最早开始时间 → 窗口内任意记录（125 实测=0 差异）
              INSPIRE    mech_vent/rrt/vasopressor : 窗口内任意记录 → 最早开始时间落窗
              eICU       vasopressor           : infusiondrug 仅单时点 ⇒ 两规则恒等（无变体）
              淮安        : 数据来自单中心 Excel，无事件层来源 ⇒ 不适用
产物：
  output/treatment_rule_sensitivity_v1.json
  output/_126_treatment_rule_sensitivity.md
"""
import io
import json
import os
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
PGKW = dict(host=_os.environ.get("ABI_DB_HOST", "localhost"),
            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
            user=_os.environ.get("ABI_DB_USER", "postgres"),
            password=_os.environ.get("ABI_DB_PASSWORD", ""))
EPS = 1e-6
SEED = 42
LOG = []


def log(s=""):
    print(s)
    LOG.append(str(s))


def cal_slope_intercept(y, p):
    lp = np.log(np.clip(np.asarray(p, float), EPS, 1 - EPS)).reshape(-1, 1)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver="lbfgs",
                           max_iter=2000).fit(lp, np.asarray(y, int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def ece(y, p, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(e)


def metrics(y, p):
    s, i = cal_slope_intercept(y, p)
    return {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "slope": s, "intercept": i, "oe": float(np.mean(y) / np.mean(p)),
            "ece": ece(y, p)}


def prep(art, df, feats, cont):
    Z = df[feats].copy()
    for c in cont:
        Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(art["medians"][c])
    for c in feats:
        if c not in cont:
            Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(0)
    Zs = Z.copy()
    Zs[cont] = art["scaler"].transform(Z[cont])
    return Z, Zs


def predict(art, df, feats, cont):
    Z, Zs = prep(art, df, feats, cont)
    if "logit_params" in art:
        lp = art["logit_params"]
        a = 1 / (1 + np.exp(-(lp["const"] + Zs[feats].values @
                              np.array([lp[f] for f in feats]))))
    else:
        a = art["lr"].predict_proba(Zs[feats])[:, 1]
    xg_ = art.get("xg", art.get("xgb"))
    return a, xg_.predict_proba(Z[feats].values)[:, 1]


def _q(cu, sql):
    cu.execute("SET statement_timeout = '600000'")
    cu.execute(sql)
    return pd.DataFrame(cu.fetchall(), columns=[d[0] for d in cu.description])


# ==================================================================== 各库变体旗标
def mimic_alt():
    """MIMIC-IV：返回 per-stay 的 any_start / any_over（mech_vent 与 vasopressor）。"""
    df = pd.read_csv(os.path.join(DATA, "mimic_abi_cohort_v2.csv"),
                     usecols=["stay_id", "intime", "mech_vent", "vasopressor", "rrt"])
    df["intime"] = pd.to_datetime(df["intime"])
    cn = psycopg2.connect(dbname="mimiciv", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("DROP TABLE IF EXISTS _m126")
    cu.execute("CREATE TEMP TABLE _m126(k bigint primary key, t timestamp)")
    cu.executemany("INSERT INTO _m126 VALUES (%s,%s)",
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
    out = {}
    for var, (tbl, cond) in SRC.items():
        sql = """
        SELECT b.k AS stay_id,
          max(CASE WHEN e.starttime BETWEEN b.t - interval '6 hours'
                                        AND b.t + interval '1 day' THEN 1 ELSE 0 END) AS any_start,
          max(CASE WHEN e.starttime <= b.t + interval '1 day'
                    AND (e.endtime IS NULL OR e.endtime >= b.t - interval '6 hours')
               THEN 1 ELSE 0 END) AS any_over
        FROM _m126 b LEFT JOIN (SELECT stay_id, starttime, endtime FROM %s WHERE %s) e
          ON e.stay_id = b.k GROUP BY b.k
        """ % (tbl, cond)
        d = _q(cu, sql)
        m = df[["stay_id"]].merge(d, on="stay_id", how="left")
        assert len(m) == len(df)
        out[var] = dict(any_start=m["any_start"].fillna(0).astype(int).values,
                        any_over=m["any_over"].fillna(0).astype(int).values)
    cu.execute("DROP TABLE IF EXISTS _m126")
    cn.close()
    return df, out


def nwicu_alt():
    """NWICU：vasopressor 的"最早开始时间落窗"旗标（mech_vent/rrt 的 any_start=0 差异）。"""
    df = pd.read_csv(os.path.join(DATA, "nwicu_abi_final_v6.csv"),
                     usecols=["hadm_id", "stay_id", "intime",
                              "mech_vent", "vasopressor", "rrt"])
    df["intime"] = pd.to_datetime(df["intime"])
    cn = psycopg2.connect(dbname="nwicu", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("DROP TABLE IF EXISTS _n126")
    cu.execute("CREATE TEMP TABLE _n126(h bigint primary key, s bigint, t timestamp)")
    cu.executemany("INSERT INTO _n126 VALUES (%s,%s,%s)",
                   [(int(a), int(b), c.to_pydatetime())
                    for a, b, c in zip(df["hadm_id"], df["stay_id"], df["intime"])])
    DRUG = ("(lower(p.drug) LIKE '%%norepinephrine%%' OR lower(p.drug) LIKE '%%epinephrine%%' "
            "OR lower(p.drug) LIKE '%%dopamine%%' OR lower(p.drug) LIKE '%%phenylephrine%%' "
            "OR lower(p.drug) LIKE '%%vasopressin%%')")
    ROUTE = ("(lower(p.route) IN ('intravenous','injection') "
             "AND lower(p.drug) NOT LIKE '%%lidocaine%%' "
             "AND lower(p.drug) NOT LIKE '%%cocoa butter%%')")
    sql = """
    SELECT b.h AS hadm_id,
      max(CASE WHEN p.starttime BETWEEN b.t - interval '6 hours'
                                   AND b.t + interval '1 day' THEN 1 ELSE 0 END) AS any_start
    FROM _n126 b LEFT JOIN hosp.prescriptions p
      ON p.hadm_id = b.h AND %s AND %s
    GROUP BY b.h
    """ % (DRUG, ROUTE)
    d = _q(cu, sql)
    m = df[["hadm_id"]].merge(d, on="hadm_id", how="left")
    assert len(m) == len(df)
    out = dict(vasopressor=dict(first_win=None,
                                any_start=m["any_start"].fillna(0).astype(int).values))
    # 最早开始时间落窗
    sql2 = """
    SELECT b.h AS hadm_id, min(p.starttime) AS first_st
    FROM _n126 b LEFT JOIN hosp.prescriptions p
      ON p.hadm_id = b.h AND %s AND %s
    GROUP BY b.h
    """ % (DRUG, ROUTE)
    d2 = _q(cu, sql2)
    m2 = df[["hadm_id"]].merge(d2, on="hadm_id", how="left")
    fs = pd.to_datetime(m2["first_st"])
    lo = df["intime"] - pd.Timedelta(hours=6)
    hi = df["intime"] + pd.Timedelta(hours=24)
    out["vasopressor"]["first_win"] = (fs.notna() & (fs >= lo) & (fs <= hi)).astype(int).values
    # mech_vent / rrt 的 any_start（125 实测与 final 相同，这里再取一次用于自检）
    for var, ids in (("mech_vent", "(787541,753461)"),
                     ("rrt", "(704890,772042,724671,798351)")):
        sql3 = """
        SELECT b.h AS hadm_id,
          max(CASE WHEN pe.starttime BETWEEN b.t - interval '6 hours'
                                        AND b.t + interval '1 day' THEN 1 ELSE 0 END) AS any_start
        FROM _n126 b LEFT JOIN icu.procedureevents pe
          ON pe.stay_id = b.s AND pe.itemid IN %s
        GROUP BY b.h
        """ % ids
        d3 = _q(cu, sql3)
        m3 = df[["hadm_id"]].merge(d3, on="hadm_id", how="left")
        assert len(m3) == len(df)
        out[var] = dict(any_start=m3["any_start"].fillna(0).astype(int).values)
    cu.execute("DROP TABLE IF EXISTS _n126")
    cn.close()
    return df, out


def inspire_alt():
    """INSPIRE：mech_vent/rrt/vasopressor 的"最早记录时间落窗"旗标。"""
    df = pd.read_csv(os.path.join(DATA, "inspire_abi_final_v3.csv"),
                     usecols=["op_id", "subject_id", "mech_vent", "vasopressor", "rrt"])
    cn = psycopg2.connect(dbname="inspire", **PGKW)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("DROP TABLE IF EXISTS _i126")
    cu.execute("CREATE TEMP TABLE _i126(k bigint primary key, s bigint, t0 bigint)")
    cu.executemany("INSERT INTO _i126 VALUES (%s,%s,NULL)",
                   [(int(a), int(b)) for a, b in zip(df["op_id"], df["subject_id"])])
    cu.execute("""UPDATE _i126 SET t0 = o.icuin FROM (
                   SELECT op_id, nullif(trim(icuin_time),'')::bigint AS icuin
                   FROM inspire.operations) o WHERE o.op_id = _i126.k""")
    anch = _q(cu, "SELECT k, t0 FROM _i126").set_index("k")["t0"]
    t0 = pd.to_numeric(df["op_id"].map(anch), errors="coerce")
    assert int(t0.notna().sum()) == len(df)
    out = {}

    def _fw(series):
        s = pd.to_numeric(series, errors="coerce")
        return (s.notna() & (s >= t0 - 0) & (s <= t0 + 1440)).astype(int).values

    for var, tbl, cond, lo_off in (("mech_vent", "inspire.ward_vitals", "w.item_name = 'vent'", 60),
                                   ("rrt", "inspire.ward_vitals", "w.item_name = 'crrt'", 60)):
        sql = """
        SELECT b.k AS op_id, min(nullif(trim(w.chart_time),'')::bigint) AS first_st
        FROM _i126 b LEFT JOIN %s w ON w.subject_id = b.s AND %s
        GROUP BY b.k
        """ % (tbl, cond)
        d = _q(cu, sql)
        m = df[["op_id"]].merge(d, on="op_id", how="left")
        assert len(m) == len(df)
        s = pd.to_numeric(m["first_st"], errors="coerce")
        out[var] = dict(first_win=(s.notna() & (s >= t0 - lo_off) &
                                   (s <= t0 + 1440)).astype(int).values)
    sql = """
    SELECT b.k AS op_id, min(nullif(trim(m.chart_time),'')::bigint) AS first_st
    FROM _i126 b LEFT JOIN inspire.medications m
      ON m.subject_id = b.s
     AND lower(m.drug_name) ~ 'norepineph|epinephrine|dopamine|phenyleph|'
                              'vasopressin|dobutamine|milrinone'
    GROUP BY b.k
    """
    d = _q(cu, sql)
    m = df[["op_id"]].merge(d, on="op_id", how="left")
    assert len(m) == len(df)
    out["vasopressor"] = dict(first_win=_fw(m["first_st"]))
    cu.execute("DROP TABLE IF EXISTS _i126")
    cn.close()
    return df, out


# ==================================================================== main
def main():
    log("=" * 100)
    log("治疗指标判定规则变体 → 外部验证指标敏感性（第十九轮模拟评审 §三）")
    log("=" * 100)

    pp = joblib.load(os.path.join(OUT, "preproc_v2.joblib"))
    md = joblib.load(os.path.join(OUT, "models_v2.joblib"))
    ih = joblib.load(os.path.join(OUT, "model_inhosp_v2.joblib"))
    a1 = dict(medians=pp["medians"], scaler=pp["scaler"],
              logit_params=md["logit_params"], xgb=md["xgb"])
    a2 = dict(medians=ih["medians"], scaler=ih["scaler"], lr=ih["lr"], xgb=ih["xg"])
    FEATS, CONT = pp["feats"], pp["cont"]
    ref = pd.read_csv(os.path.join(OUT, "external_validation_v9.csv"))
    refm = {(r["cohort"], r["model"]): r for _, r in ref.iterrows()}

    payload = dict(generated_at=datetime.now().isoformat(timespec="seconds"),
                   generated_by="scripts/126_treatment_rule_sensitivity.py",
                   cohorts={})

    def run(tag, df, y, art, alt_fn, note):
        """alt_fn(df) -> dict(变量 -> 变体旗标数组)；None 表示该库无事件层变体。"""
        base = {}
        for nm in ("Logistic", "XGBoost"):
            plr, pxg = predict(art, df, FEATS, CONT)
            base[nm] = metrics(y, plr if nm == "Logistic" else pxg)
            r = refm.get((tag, nm))
            if r is not None:
                d = abs(base[nm]["auc"] - float(r["auc"]))
                assert d < 1e-9, "自检失败 %s/%s: %.6f vs %.6f" % (tag, nm, base[nm]["auc"], r["auc"])
        entry = dict(note=note, n=int(len(y)), events=int(np.sum(y)),
                     as_coded={k: {kk: round(vv, 6) for kk, vv in v.items()}
                               for k, v in base.items()}, alt=None)
        if alt_fn is None:
            log("  %-16s %s（无变体）" % (tag, note))
        else:
            flags = alt_fn(df)
            df2 = df.copy()
            for var, arr in flags.items():
                df2[var] = np.asarray(arr, dtype=int)
            entry["alt_flags"] = {var: int(np.sum(np.asarray(a, dtype=int)))
                                  for var, a in flags.items()}
            alt = {}
            for nm in ("Logistic", "XGBoost"):
                plr, pxg = predict(art, df2, FEATS, CONT)
                alt[nm] = metrics(y, plr if nm == "Logistic" else pxg)
            entry["alt"] = {k: {kk: round(vv, 6) for kk, vv in v.items()}
                            for k, v in alt.items()}
            entry["delta"] = {k: {kk: round(alt[k][kk] - base[k][kk], 6)
                                  for kk in base[k]} for k in base}
            log("  %-16s %s" % (tag, note))
            for nm in ("Logistic", "XGBoost"):
                log("      %-8s AUC %.4f → %.4f (Δ%+.4f) | O:E %.3f → %.3f | slope %.3f → %.3f"
                    % (nm, base[nm]["auc"], alt[nm]["auc"],
                       alt[nm]["auc"] - base[nm]["auc"],
                       base[nm]["oe"], alt[nm]["oe"],
                       base[nm]["slope"], alt[nm]["slope"]))
        payload["cohorts"][tag] = entry

    # ---- 读库 + 变体旗标 ----
    log("")
    log("[1/3] 读取队列与变体旗标")
    nw = pd.read_csv(os.path.join(DATA, "nwicu_abi_final_v6.csv"))
    ins = pd.read_csv(os.path.join(DATA, "inspire_abi_final_v3.csv"))
    e = pd.read_csv(os.path.join(DATA, "eicu_abi_final_v3.csv"))

    def nw_alt(df):
        _, out = nwicu_alt()
        f = {"vasopressor": out["vasopressor"]["first_win"]}
        f["mech_vent"] = out["mech_vent"]["any_start"]
        f["rrt"] = out["rrt"]["any_start"]
        return f

    def ins_alt(df):
        _, out = inspire_alt()
        return {v: out[v]["first_win"] for v in ("mech_vent", "rrt", "vasopressor")}

    log("[2/3] 逐队列计算（as_coded 一栏须与 external_validation_v9.csv 逐位一致）")
    log("")
    run("NWICU_1yr", nw, nw["death_365"].values.astype(int), a1, nw_alt,
        "NWICU：vasopressor 换'最早开始时间落窗'；mech_vent/rrt 换'窗口内任意记录'（实测 0 差异）")
    run("INSPIRE_1yr", ins, ins["death_365"].values.astype(int), a1, ins_alt,
        "INSPIRE：三个治疗指标换'最早记录时间落窗'")
    run("eICU_inhosp", e, e["death_hosp"].values.astype(int), a2, None,
        "eICU：infusiondrug 仅 infusionoffset 单时点 ⇒ 两规则恒等；mech_vent/rrt 来自 APACHE 汇总")
    run("NWICU_inhosp", nw, nw["death_hosp"].values.astype(int), a2, nw_alt,
        "NWICU：同上")
    run("INSPIRE_inhosp", ins, ins["death_hosp"].values.astype(int), a2, ins_alt,
        "INSPIRE：同上")
    try:
        sys_path = os.path.join(ROOT, "scripts")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "v95", os.path.join(sys_path, "95_v9_external_figures.py"))
        # 不在 95 的 main 里跑，只复用 huaian_frame
        src = io.open(os.path.join(sys_path, "95_v9_external_figures.py"),
                      encoding="utf-8").read()
        ns = {}
        exec(compile(src.split("def main(")[0], "v95_partial", "exec"), ns)  # noqa: S102
        Xl, yl = ns["huaian_frame"]()
        run("Huaian_inhosp", Xl, yl, a2, None,
            "淮安：单中心 Excel 台账，无事件层时间来源 ⇒ 规则变体不适用")
    except Exception as ex:  # noqa: BLE001
        log("  !! 淮安跳过：%s" % ex)
        payload["cohorts"]["Huaian_inhosp"] = dict(
            note="单中心 Excel 台账，无事件层时间来源 ⇒ 规则变体不适用",
            error="%s: %s" % (type(ex).__name__, ex))

    log("")
    log("[3/3] 落盘")
    with io.open(os.path.join(OUT, "treatment_rule_sensitivity_v1.json"), "w",
                 encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with io.open(os.path.join(OUT, "_126_treatment_rule_sensitivity.md"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    log("→ output/treatment_rule_sensitivity_v1.json")
    log("→ output/_126_treatment_rule_sensitivity.md")


if __name__ == "__main__":
    main()
