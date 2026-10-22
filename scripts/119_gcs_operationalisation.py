# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
119_gcs_operationalisation.py
=============================
第十二轮（模拟评审 §三）—— GCS 单独预测的**操作化核查**。

评审要求回答：
  ① 用第一条 / 最低值 / 平均值？  ② 原始总分还是风险方向的 −GCS？
  ③ "GCS alone" 是原始分数的 ROC，还是训练集拟合的单变量模型？
  ④ 气管插管时 verbal 分量如何编码？  ⑤ 总分与 motor 是否来自相近时间点？
  ⑥ 能否识别镇静或肌松状态？

本脚本只做**核对与描述**，不改任何既有结果文件；输出
`output/gcs_operationalisation_v11.json`，供正文与 Additional file 1 引用。

要点（回答 ③）：AUC 对**单调变换不变**，故 "原始分数的 ROC" 与 "在训练集上拟合
单变量 logistic 后取 AUC" 在数值上必然相等。本脚本两种都算，作为互相验证。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"
DATA = BASE / "data"
SEED = 42
TEST = 0.30

LOG: list[str] = []


def log(m: str = "") -> None:
    print(m)
    LOG.append(m)


def iqr(s: pd.Series) -> list:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return [float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))]


def main() -> int:
    log("[119] GCS 单独预测 —— 操作化与记录质量核查")
    log("=" * 78)

    coh = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv",
                      usecols=["stay_id", "mech_vent", "death_365", "hadm_id"])
    gcs = pd.read_csv(DATA / "mimic_gcs.csv")
    m = coh.merge(gcs, on="stay_id", how="left")
    assert len(m) == 16597, "行数不符：%d" % len(m)
    y = m["death_365"].astype(int).to_numpy()

    # 与主分析同一切分（118 号：seeded 7:3，按结局分层）
    idx = np.arange(len(m))
    _, te = train_test_split(idx, test_size=TEST, stratify=y, random_state=SEED)
    log("[OK] 测试集 n=%d events=%d（与主分析同一切分）" % (len(te), int(y[te].sum())))

    # ---------------------------------------------------------- 1 库内分量与 unable 标记
    conn = psycopg2.connect(host=_os.environ.get("ABI_DB_HOST", "localhost"),
                            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
                            user=_os.environ.get("ABI_DB_USER", "postgres"),
                            password=_os.environ.get("ABI_DB_PASSWORD", ""),
                            dbname="mimiciv")
    ids = tuple(int(x) for x in m["stay_id"].dropna().unique())
    q = ("SELECT stay_id, gcs_min, gcs_motor, gcs_verbal, gcs_eyes, gcs_unable "
         "FROM mimiciv_derived.first_day_gcs WHERE stay_id IN %s")
    db = pd.read_sql(q, conn, params=(ids,))
    conn.close()
    log("[OK] first_day_gcs 命中 %d / %d 个 stay_id" % (len(db), len(ids)))
    m = m.merge(db, on="stay_id", how="left", suffixes=("", "_db"))

    # ★★ 2026-10-21（第十二轮模拟评审 §三，评审正确）：**原注释写错了**。
    #   旧注释说 "derived 表里 min 是对各分量分别取最小值" —— 事实相反：
    #   first_day_gcs 先按总分排序取最低的那一条记录，再把该记录的 motor/verbal/eyes
    #   整行返回，四个字段同出一处；而且当语言分量无法评估时，派生总分被**直接置为 15**。
    #   因此"分量和 ≠ 总分"恰恰是那条置 15 规则留下的指纹，不是"同一时点"的解释。
    #   下面在全表（94,458 行）上验证这两条，而不是只读我们的 stay_id。
    comp = m[["gcs_eyes", "gcs_verbal", "gcs_motor"]].sum(axis=1, min_count=3)
    both = comp.notna() & m["gcs_min"].notna()
    eq = int(np.sum(np.isclose(comp[both], m["gcs_min"][both])))
    log("  （分析集内）分量和 == 首日总分：%d / %d（%.1f%%）"
        % (eq, int(both.sum()), 100.0 * eq / max(1, int(both.sum()))))

    conn = psycopg2.connect(host=_os.environ.get("ABI_DB_HOST", "localhost"),
                            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
                            user=_os.environ.get("ABI_DB_USER", "postgres"),
                            password=_os.environ.get("ABI_DB_PASSWORD", ""),
                            dbname="mimiciv")
    full = pd.read_sql("""
        SELECT count(*) AS n_rows,
               count(*) FILTER (WHERE gcs_unable = 1) AS n_unable,
               count(*) FILTER (WHERE gcs_unable = 1 AND gcs_min = 15) AS n_unable_total15,
               count(*) FILTER (WHERE gcs_unable = 1 AND gcs_motor = 1) AS n_unable_motor1,
               count(*) FILTER (WHERE gcs_unable = 1 AND gcs_motor <= 4) AS n_unable_motor_le4,
               count(*) FILTER (WHERE gcs_min <> gcs_eyes + gcs_verbal + gcs_motor)
                   AS n_sum_ne_total
        FROM mimiciv_derived.first_day_gcs""", conn)
    conn.close()
    _f = {k: int(v) for k, v in full.iloc[0].items()}
    log("  （全表 %d 行）unable=%d，其中总分=15 的 %d 行、motor=1 的 %d 行、"
        "分量和≠总分 %d 行"
        % (_f["n_rows"], _f["n_unable"], _f["n_unable_total15"],
           _f["n_unable_motor1"], _f["n_sum_ne_total"]))

    vent = m["mech_vent"].astype(int).to_numpy() == 1
    res: dict = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "table": "mimiciv_derived.first_day_gcs",
            "aggregation": ("取窗口（ICU 入科前 6 小时至入科后 24 小时）内**总分最低的那一条"
                            "记录**，并把该记录的总分、运动、语言、睁眼四个字段整行返回；"
                            "**不是**对每个分量分别取最小值"),
            "total": "gcs_min（所选记录的总分；语言无法评估时派生值直接置 15）",
            "motor": "gcs_motor（**同一**所选记录的运动分量，非当日最低运动分）",
        },
        "n_total": int(len(m)),
        "n_test": int(len(te)),
        "events_test": int(y[te].sum()),
        "missing_pct": {
            "gcs_min": round(100.0 * float(m["gcs_min"].isna().mean()), 2),
            "gcs_motor": round(100.0 * float(m["gcs_motor"].isna().mean()), 2),
        },
        "unable_flag": {
            "n_flagged": int((m["gcs_unable"] == 1).sum()),
            "pct_overall": round(100.0 * float((m["gcs_unable"] == 1).mean()), 1),
            "pct_ventilated": (round(100.0 * float((m.loc[vent, "gcs_unable"] == 1).mean()), 1)
                               if vent.any() else None),
            "pct_nonventilated": (round(100.0 * float((m.loc[~vent, "gcs_unable"] == 1).mean()), 1)
                                  if (~vent).any() else None),
            "meaning": ("gcs_unable=1 由派生表在**无法评估**时置位（插管、镇静、"
                        "语言障碍等），不能单独用来区分镇静与肌松"),
        },
        "component_consistency": {
            "n_with_all_components": int(both.sum()),
            "n_sum_equals_min_total": eq,
            "pct_equal": round(100.0 * eq / max(1, int(both.sum())), 1),
            "note": ("四个字段同出所选记录，故正常情况下分量和应等于总分；不等的那些行"
                     "正是【语言无法评估时总分被置 15】这条派生规则留下的痕迹。"),
        },
        "derivation_check_full_table": dict(_f, table="mimiciv_derived.first_day_gcs",
                                            note=("全表核查：unable 标记为 1 的行**全部**"
                                                  "被置为总分 15，其中包括大量 motor=1 的行；"
                                                  "这说明派生总分在插管/无法评估时被钉在最高值，"
                                                  "而不是床旁总分的忠实记录。")),
        "distribution": {},
        "alone_auc": {},
    }

    for lab, mask in (("all", np.ones(len(m), bool)), ("ventilated", vent),
                      ("nonventilated", ~vent)):
        sub = m.loc[mask]
        res["distribution"][lab] = {
            "n": int(len(sub)),
            "gcs_min_median_iqr": iqr(sub["gcs_min"]),
            "gcs_motor_median_iqr": iqr(sub["gcs_motor"]),
            "pct_gcs_min_15": round(100.0 * float((sub["gcs_min"] == 15).mean()), 1),
            "pct_gcs_min_le8": round(100.0 * float((sub["gcs_min"] <= 8).mean()), 1),
        }

    # ---------------------------------------------------------- 2 GCS-alone AUC（两种口径）
    gmed = float(np.nanmedian(m["gcs_min"].values[
        np.setdiff1d(idx, te)]))          # 训练集内中位数（冻结口径）

    def alone(col: str, mask: np.ndarray) -> dict:
        # ★ 每个分数用它**自己**的训练集中位数插补（total→14.0，motor→6.0）。
        #   早先用 total 的中位数去补 motor，会让 motor 的 AUC 掉到 0.681（错口径）。
        _tr = np.setdiff1d(idx, te)
        med = float(np.nanmedian(m[col].values[_tr]))
        s = np.intersect1d(te, np.where(mask)[0])
        yy = y[s]
        gg = m[col].values[s].astype(float)
        gg = np.where(np.isnan(gg), med, gg)
        raw = float(roc_auc_score(yy, gg))          # 原始分数（未定向）
        neg = float(roc_auc_score(yy, -gg))         # 低 GCS ⇒ 高风险
        # 交叉验证：训练集拟合单变量 logistic，测试集取 AUC（应与 -GCS 的 ROC 相同）
        tr = np.setdiff1d(idx, te)
        gtr = m[col].values[tr].astype(float)
        gtr = np.where(np.isnan(gtr), med, gtr)
        lr = LogisticRegression(max_iter=1000).fit(gtr.reshape(-1, 1), y[tr])
        p = lr.predict_proba(gg.reshape(-1, 1))[:, 1]
        return {"n": int(len(s)), "events": int(yy.sum()),
                "auc_raw_score": round(raw, 4),
                "auc_directed": round(neg, 4),
                "auc_univariate_model": round(float(roc_auc_score(yy, p)), 4),
                "coef_sign": ("低 GCS 高风险" if lr.coef_[0][0] < 0 else "方向相反")}

    res["alone_auc"] = {
        "note": ("auc_directed 与 auc_univariate_model 在数值上必然一致 —— AUC 对单调"
                 "变换不变，故两者不是两套估计，而是同一估计的两种写法。"),
        "total": {"all": alone("gcs_min", np.ones(len(m), bool)),
                  "ventilated": alone("gcs_min", vent),
                  "nonventilated": alone("gcs_min", ~vent)},
        "motor": {"all": alone("gcs_motor", np.ones(len(m), bool)),
                  "ventilated": alone("gcs_motor", vent),
                  "nonventilated": alone("gcs_motor", ~vent)},
        "imputation": {"train_median": gmed,
                       "n_missing_test_total": int(m["gcs_min"].iloc[te].isna().sum())},
    }

    log("")
    for k, v in res["alone_auc"]["total"].items():
        log("  total %-14s AUC=%.4f (raw %.4f)  n=%d ev=%d"
            % (k, v["auc_directed"], v["auc_raw_score"], v["n"], v["events"]))
    for k, v in res["alone_auc"]["motor"].items():
        log("  motor %-14s AUC=%.4f  n=%d ev=%d" % (k, v["auc_directed"], v["n"], v["events"]))
    log("")
    log("  unable 标记：%d 例（%.1f%%）；通气亚组 %.1f%% / 非通气 %.1f%%"
        % (res["unable_flag"]["n_flagged"], res["unable_flag"]["pct_overall"],
           res["unable_flag"]["pct_ventilated"] or 0.0,
           res["unable_flag"]["pct_nonventilated"] or 0.0))

    (OUT / "gcs_operationalisation_v11.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[OK] 写出 output/gcs_operationalisation_v11.json")
    (OUT / "_119_gcs_operationalisation.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
