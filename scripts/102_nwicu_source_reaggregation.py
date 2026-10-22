# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""102_nwicu_source_reaggregation.py —— 在**源记录层面**清理哨兵值后重新聚合。

问题（2026-10-07 第三轮评审 §一.1，评审判断正确）：
  上一轮（脚本 89）只在**聚合后的分析表**上把越界值置 NaN。评审指出：
    "真实值不能从聚合表恢复，不等于不能从源记录恢复" —— 属实。
  NWICU 的 `glucose_max` 由 `sql/11_nwicu_cohort.sql` 写成
    `max(case when itemid in (100001,100062,100045) then le.valuenum end)`
  即**对首日窗口内的原始检验值取最大值**。哨兵值 9,999,999 因此会**顶掉真实最大值**，
  之后再把该格置 NaN，等于把"有真实测量"的入院变成"缺失"。
  抽样核验（NWICU 库 hosp.labevents）：分析表中 glucose_max 缺失的 675 例，
  抽查 200 例 → **195 例在源层排除哨兵值后能取回真实最大值**。

本脚本（只读库、不改任何已交付文件）：
  1. 在源层排除哨兵值，按**与原 SQL 完全相同**的口径重新聚合首日窗口
     （intime-6h ~ intime+1d，同 itemid 集合，同 max/min 方向）；
  2. 与 `data/nwicu_abi_final_v4.csv` 逐变量逐行对比，报告
     「能救回」「本来就缺」「口径不一致」三类计数；
  3. 用**冻结产物**（不重训）在「源层重聚合」的 glucose_max 上重算 NWICU 的
     判别力与校准，与现值对照。
     ⚠ 冻结模型仍按「训练集中位数」插补，故若真实值与NaN 的取舍影响预测，
        效应会真实体现出来——这正是本脚本要量化的东西。

输出：output/nwicu_source_reaggregation_v1.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(ABI_BASE)
OUT = BASE / "output"
V9R = OUT / "v9_results"
SENTINEL = 999999

# 与 sql/11_nwicu_cohort.sql 的 lab CTE 完全一致的 itemid 与聚合方向
LAB_SPEC = [
    ("wbc_max", "max", [100016]),
    ("hemoglobin_min", "min", [100007]),
    ("platelets_min", "min", [100014]),
    ("sodium_min", "min", [100010]),
    ("potassium_max", "max", [100011]),
    ("creatinine_max", "max", [100002]),
    ("bun_max", "max", [100004]),
    ("glucose_max", "max", [100001, 100062, 100045]),
    ("inr_max", "max", [100030]),
]


def reaggregate():
    """在源层排除哨兵值后重算首日窗口极值。

    队列表 = `icu.abi_a1_ext`（n=3420，与 data/nwicu_abi_final_v4.csv 逐行对应）。
    时间窗与 `sql/11_nwicu_cohort.sql` 的 lab CTE 完全一致：
        le.charttime BETWEEN intime - interval '6 hours' AND intime + interval '1 day'
    """
    conn = psycopg2.connect(host=_os.environ.get("ABI_DB_HOST", "localhost"),
                            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
                            user=_os.environ.get("ABI_DB_USER", "postgres"),
                            password=_os.environ.get("ABI_DB_PASSWORD", ""),
                            dbname="nwicu")
    cur = conn.cursor()
    cur.execute("SELECT hadm_id, intime FROM icu.abi_a1_ext")
    base = {int(h): t for h, t in cur.fetchall()}
    log("  icu.abi_a1_ext 队列 hadm 数: %d" % len(base))

    items = sorted({i for _, _, ids in LAB_SPEC for i in ids})
    ph = ",".join(str(i) for i in items)
    placeholders = ",".join(str(int(h)) for h in base)
    # ⚠ 时间窗必须与原 SQL 一致，否则不是"同一口径重算"而是换了口径。
    cur.execute(
        "SELECT le.hadm_id, le.itemid, le.valuenum "
        "FROM hosp.labevents le JOIN icu.abi_a1_ext b ON le.hadm_id = b.hadm_id "
        "WHERE le.itemid IN (%s) "
        "  AND le.charttime BETWEEN b.intime - interval '6 hours' "
        "                     AND b.intime + interval '1 day'" % ph)
    rows = cur.fetchall()
    conn.close()
    log("  首日窗口内候选检验行: %d" % len(rows))

    # 按 hadm 聚合：(真实极值, 哨兵数, 真实值是否存在)
    acc = {}
    for hadm, itemid, val in rows:
        if val is None:
            continue
        h = int(hadm)
        d = acc.setdefault(h, {})
        s = d.setdefault(itemid, {"real": [], "sent": 0})
        if abs(float(val)) >= SENTINEL:
            s["sent"] += 1
        else:
            s["real"].append(float(val))

    out = {}
    for name, direc, ids in LAB_SPEC:
        for h in acc:
            vals = []
            sent = 0
            for i in ids:
                if i in acc[h]:
                    vals.extend(acc[h][i]["real"])
                    sent += acc[h][i]["sent"]
            if not vals and not sent:
                out.setdefault(name, {})[h] = (None, 0)
                continue
            out.setdefault(name, {})[h] = (
                (max(vals) if direc == "max" else min(vals)) if vals else None, sent)
    return base, out


def cal_slope_intercept(y, p, eps=1e-6):
    """⚠ 两种口径必须分开报告，否则会与正文数字对不上：
      · clipped_grid：先把 p 裁剪到 [eps,1-eps] 再取 logit（脚本 89/93 的口径，
        正文 Table 4 的 calibration slope 用它 → NWICU 1年 = 0.878）；
      · unclipped  ：直接对**原始线性预测子**做回归（V9 新主分析口径，无裁剪常数）。
      本函数给出 unclipped；clipped 由调用方另算。
    """
    lg = np.log(np.asarray(p, dtype=float) / (1.0 - np.asarray(p, dtype=float)))
    X = np.column_stack([lg, np.ones_like(lg)])
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, dtype=float), rcond=None)
    return float(beta[0]), float(beta[1])


def cal_slope_clipped(y, p, eps=1e-6):
    """与正文 Table 4 同口径：先裁剪再取 logit。"""
    from sklearn.linear_model import LogisticRegression
    pc = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lg = np.log(pc / (1 - pc))
    m = LogisticRegression(fit_intercept=True, C=1e12, solver="lbfgs",
                           max_iter=5000).fit(lg.reshape(-1, 1),
                                              np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def log(s):
    print(s)


def main():
    log("#" * 78)
    log("# 102_nwicu_source_reaggregation.py —— NWICU 哨兵值**源层**清理并重新聚合")
    log("#" * 78)

    df = pd.read_csv(BASE / "data" / "nwicu_abi_final_v5.csv")
    log("  分析表 n = %d" % len(df))

    base, reagg = reaggregate()

    # ---- 逐变量对比 ----
    report = {}
    for name, direc, ids in LAB_SPEC:
        if name not in df.columns:
            continue
        cur_col = pd.to_numeric(df[name], errors="coerce")
        src = reagg.get(name, {})
        cur_by_hadm = dict(zip(df["hadm_id"].astype(int), cur_col))
        n_cur_na = n_rescued = n_still_na = n_diff_val = n_same = n_not_in_src = 0
        rescued_examples, diff_examples = [], []
        for h, cur_v in cur_by_hadm.items():
            src_v, sent = src.get(h, ("ABSENT", 0))
            if src_v == "ABSENT":
                n_not_in_src += 1
                continue
            if pd.isna(cur_v):
                n_cur_na += 1
                if src_v is not None:
                    n_rescued += 1
                    if len(rescued_examples) < 5:
                        rescued_examples.append(
                            {"hadm_id": h, "was": None, "now": src_v, "n_sentinel_rows": sent})
                else:
                    n_still_na += 1
            else:
                if src_v is None:
                    n_still_na += 1
                elif abs(float(cur_v) - float(src_v)) > 1e-9:
                    n_diff_val += 1
                    if len(diff_examples) < 5:
                        diff_examples.append(
                            {"hadm_id": h, "analysis_table": float(cur_v),
                             "source_reagg": float(src_v), "n_sentinel_rows": sent})
                else:
                    n_same += 1
        report[name] = dict(
            n_analysis_missing=int(n_cur_na), n_rescued=int(n_rescued),
            n_still_missing=int(n_still_na), n_value_differs=int(n_diff_val),
            n_identical=int(n_same), n_hadm_not_in_source=int(n_not_in_src),
            rescued_examples=rescued_examples, value_diff_examples=diff_examples)
        log("  %-18s 分析表缺失 %4d → 源层救回 %4d / 仍缺 %4d / 数值不同 %4d / 相同 %4d"
            % (name, n_cur_na, n_rescued, n_still_na, n_diff_val, n_same))

    # ---- 构造源层重聚合版分析表并重算 NWICU 性能（冻结模型，不重训）----
    pre = joblib.load(V9R / "preproc.joblib")
    mo = joblib.load(V9R / "models.joblib")
    FEATS = list(pre["feats"]); CONT = list(pre["cont"])
    SCALER, MED = pre["scaler"], pre["medians"]
    LP, XGB = mo["logit_params"], mo["xgb"]

    def predict(d):
        X = d[FEATS].copy()
        for c in CONT:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(MED[c]) if not pd.isna(MED[c]) else 0.0
        for c in FEATS:
            if c not in CONT:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
        Xv = X.copy().astype(float)
        Xv.loc[:, CONT] = SCALER.transform(Xv.loc[:, CONT])
        C = np.asarray([float(LP[c]) for c in FEATS]); B = float(LP["const"])
        lr = Xv[FEATS].to_numpy() @ C + B
        return 1.0 / (1.0 + np.exp(-lr)), XGB.predict_proba(X.to_numpy())[:, 1]

    d_new = df.copy()
    for name, direc, ids in LAB_SPEC:
        if name not in d_new.columns:
            continue
        src = reagg.get(name, {})
        newcol = []
        for h in d_new["hadm_id"].astype(int):
            v = src.get(h, ("ABSENT", 0))[0]
            newcol.append(np.nan if v in (None, "ABSENT") else float(v))
        d_new[name] = newcol

    y = d_new["death_365"].values.astype(int)
    yv = d_new["death_hosp"].values.astype(int)
    # ---- 哨兵值计数口径对账（第三轮评审 §三：605 与 611 并非同一口径）----
    #   605 = 清洗**前**分析表里glucose_max 等于哨兵值的**格数**（= 受影响的入院数，
    #         因为每格对应一个入院）。
    #   611 = 清洗**后** glucose_max 为**缺失**的格数 = 605（被置空）+ 6（本就没有该项检查）。
    #   → 两个数都对，但放在同一句里必然被读成矛盾。必须分开命名并给出完整小表。
    _SA = json.load(open(OUT / "v9_results" / "sentinel_audit_v1.json",
                         encoding="utf-8"))["nwicu"]
    acct = {"_definitions": {
        "n_sentinel_cells": "清洗前等于哨兵值的格数（每格 = 一个入院）",
        "n_missing_after_clean": "清洗后为缺失的格数 =哨兵置空 + 本来就没有该项检查",
        "n_rescued_at_source": "回到源记录排除哨兵行后能重算出真实值的格数",
        "n_still_missing": "源层重聚合后仍缺失（源库该窗内确实没有有效值）"},
        "rows": []}
    _uniq = set()
    for name, direc, ids in LAB_SPEC:
        if name not in df.columns:
            continue
        v = report[name]
        nsent = int(_SA.get(name, {}).get("n_sentinel", 0))
        acct["rows"].append(dict(
            variable=name, n_missing_after_clean=v["n_analysis_missing"],
            n_sentinel_cells=nsent, n_rescued_at_source=v["n_rescued"],
            n_still_missing=v["n_still_missing"],
            rescue_rate_pct=(round(100.0 * v["n_rescued"] / v["n_analysis_missing"], 1)
                             if v["n_analysis_missing"] else None)))
        if nsent:
            _uniq.add(name)
    acct["n_variables_affected"] = len(_uniq)
    acct["n_admissions_with_any_placeholder_upper_bound"] = int(
        sum(r["n_sentinel_cells"] for r in acct["rows"]))

    # 独立入院数（同一入院可能多个变量都含哨兵）
    import psycopg2 as _pg
    _c = _pg.connect(host=_os.environ.get("ABI_DB_HOST", "localhost"),
                     port=int(_os.environ.get("ABI_DB_PORT", "5432")),
                     user=_os.environ.get("ABI_DB_USER", "postgres"),
                     password=_os.environ.get("ABI_DB_PASSWORD", ""), dbname="nwicu")
    _q = _c.cursor()
    _items = ",".join(str(i) for _, _, ids in LAB_SPEC for i in ids)
    _q.execute("SELECT count(DISTINCT le.hadm_id) FROM hosp.labevents le "
               "WHERE le.itemid IN (%s) AND le.valuenum >= %d" % (_items, SENTINEL))
    acct["n_unique_admissions_with_any_sentinel_in_source"] = int(_q.fetchone()[0])
    _c.close()

    res = {"_note": "源层排除哨兵值后重算首日极值；冻结模型、不重训。",
           "sentinel_accounting": acct,
           "variables": report,
           "n": int(len(d_new)),
           "one_year_events": int(y.sum()),
           "inhospital_events": int(yv.sum()),
           "performance": {}}
    for tag, yy in [("one_year", y), ("inhospital", yv)]:
        if len(np.unique(yy)) < 2:
            continue
        for label, d in [("as_is", df), ("source_reaggregated", d_new)]:
            plr, pxg = predict(d)
            s, icpt = cal_slope_intercept(yy, plr)          # 未裁剪线性预测子
            sc, icptc = cal_slope_clipped(yy, plr)          # 裁剪后取logit（正文口径）
            pc = np.clip(plr, 1e-6, 1 - 1e-6)
            res["performance"]["%s/%s" % (tag, label)] = dict(
                n=int(len(yy)), events=int(yy.sum()),
                auc_lr=round(float(roc_auc_score(yy, plr)), 4),
                auc_xgb=round(float(roc_auc_score(yy, pxg)), 4),
                slope_lr_unclipped=round(s, 4), intercept_lr_unclipped=round(icpt, 4),
                slope_lr_clipped_grid=round(sc, 4),
                oe_lr=round(float(np.mean(yy) / np.mean(plr)), 4),
                brier_lr=round(float(brier_score_loss(yy, plr)), 4))

    log("")
    log("  NWICU 性能对照（冻结模型，未重训）")
    log("  slope_unclipped = 原始线性预测子回归；slope_clipped_grid = 裁剪后取 logit（正文口径）")
    log("  %-34s %6s %7s %8s %8s %10s %8s"
        % ("outcome / variant", "n", "events", "AUC_LR", "AUC_XGB", "slope_clip", "O:E"))
    for k, v in res["performance"].items():
        log("  %-34s %6d %7d %8.4f %8.4f %10.4f %8.4f"
            % (k, v["n"], v["events"], v["auc_lr"], v["auc_xgb"],
               v["slope_lr_clipped_grid"], v["oe_lr"]))
    log("")
    for k, v in res["performance"].items():
        log("  %-34s slope_unclipped=%.4f  intercept=%.4f"
            % (k, v["slope_lr_unclipped"], v["intercept_lr_unclipped"]))

    json.dump(res, open(OUT / "nwicu_source_reaggregation_v1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("")
    log("哨兵值计数口径对账（清洗前 vs 清洗后）")
    log("  %-18s %12s %12s %12s %10s" % ("变量", "清洗后缺失", "含哨兵格数", "源层救回", "仍缺"))
    for r in acct["rows"]:
        log("  %-18s %12d %12d %12d %10d"
            % (r["variable"], r["n_missing_after_clean"], r["n_sentinel_cells"],
               r["n_rescued_at_source"], r["n_still_missing"]))
    log("  受影响变量数 = %d；源库中含哨兵值的**独立入院** = %d（同一入院可含多个变量）"
        % (acct["n_variables_affected"],
           acct["n_unique_admissions_with_any_sentinel_in_source"]))
    log("\n→ output/nwicu_source_reaggregation_v1.json")


if __name__ == "__main__":
    main()