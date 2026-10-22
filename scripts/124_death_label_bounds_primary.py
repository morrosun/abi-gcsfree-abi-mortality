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
124_death_label_bounds_primary.py
=================================
第十三轮模拟评审 §三.2 的回应脚本。

问题
----
Table S12 写的 "bounded primary rule" 对照（一年 Logistic slope 0.874 → 0.878、
O:E 1.000 → 1.003）出自 110 号脚本，而 110 读的是 `nwicu_abi_final_v4.csv` ——
即**占位值未在源层重聚合**的那一代输入。主分析（Table 4）早已换到
`nwicu_abi_final_v6.csv`（117 号，源层重聚合）。于是同一张表里出现两个输入版本，
读者会把 0.874 当成表 4 的当前值（实际 0.872）。

本脚本
------
在同一套冻结模型（118 号：preproc_v2 / models_v2 / model_inhosp_v2）与**当前主分析
输入**（v6）上，重算「有界 / 无界」两套标签下的判别与校准，并把有界一侧与
`v9_number_master.json` 的主分析值逐位对齐（断言），从而证明：
  ① 有界一侧就是表 4 的数字；
  ② 有界 → 无界的差值是在**同一输入版本**上测得的，可以放心写进 S12。

只读，不训练，不改任何既有数据。

产物：output/death_label_bounds_impact_primary_v1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"
DATA = BASE / "data"
EPS = 1e-6

LOG: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def cal_slope(y, p, eta=None):
    """与 122 号一致的斜率实现：有 η 就在 η 上拟合，否则退回概率-logit。"""
    y = np.asarray(y, int)
    if eta is not None:
        z = np.asarray(eta, float).reshape(-1, 1)
    else:
        pc = np.clip(np.asarray(p, float), EPS, 1 - EPS)
        z = np.log(pc / (1 - pc)).reshape(-1, 1)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver="lbfgs", max_iter=5000)
    m.fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def metrics(y, p, eta):
    y = np.asarray(y, int)
    s, i = cal_slope(y, p, eta)
    return {"n": int(len(y)), "events": int(y.sum()),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "slope": s, "intercept": i,
            "oe": float(np.mean(y) / np.mean(p))}


def _design(pp, df):
    feats, cont = pp["feats"], pp["cont"]
    Z = df[feats].copy()
    for c in cont:
        Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(pp["medians"][c])
    for c in feats:
        if c not in cont:
            Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(0)
    Zs = Z.copy()
    Zs[cont] = pp["scaler"].transform(Z[cont])
    return Z, Zs, feats


def main() -> int:
    log("=" * 78)
    log("124 —— 有界 / 无界死亡标签对照：当前主分析输入（v6）")
    log("=" * 78)

    df = pd.read_csv(DATA / "nwicu_abi_final_v6.csv")
    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    ih = joblib.load(OUT / "model_inhosp_v2.joblib")
    master = json.load(open(OUT / "v9_number_master.json", encoding="utf-8"))["cohorts"]

    Z, Zs, feats = _design(pp, df)

    # ---- 一年模型（LR 吃标准化、XGB 吃原始；与 87/118/119 完全同口径）
    lp = md["logit_params"]
    eta_1y_lr = lp["const"] + Zs[feats].values @ np.array([lp[f] for f in feats])
    p_1y_lr = 1.0 / (1.0 + np.exp(-eta_1y_lr))
    eta_1y_xg = md["xgb"].predict(Z[feats].values, output_margin=True)
    p_1y_xg = 1.0 / (1.0 + np.exp(-eta_1y_xg))

    # ---- 院内平行模型
    ih_feats, ih_cont = ih["feats"], ih["cont"]
    Zi = df[ih_feats].copy()
    for c in ih_cont:
        Zi[c] = pd.to_numeric(Zi[c], errors="coerce").fillna(ih["medians"][c])
    for c in ih_feats:
        if c not in ih_cont:
            Zi[c] = pd.to_numeric(Zi[c], errors="coerce").fillna(0)
    Zis = Zi.copy()
    Zis[ih_cont] = ih["scaler"].transform(Zi[ih_cont])
    eta_ih_lr = ih["lr"].decision_function(Zis[ih_feats].values)
    p_ih_lr = ih["lr"].predict_proba(Zis[ih_feats].values)[:, 1]
    eta_ih_xg = ih["xg"].predict(Zi[ih_feats].values, output_margin=True)
    p_ih_xg = 1.0 / (1.0 + np.exp(-eta_ih_xg))

    res = {
        "_note": ("有界 / 无界死亡标签对照，在当前主分析输入 nwicu_abi_final_v6.csv 上重算，"
                  "冻结模型 = 118 号（preproc_v2 / models_v2 / model_inhosp_v2）。"
                  "旧的 death_label_bounds_impact_v1.json 出自 110 号，读的是 v4 输入，"
                  "只能作为历史对照。"),
        "input_file": "data/nwicu_abi_final_v6.csv",
        "n": int(len(df)),
        "calibration_route": "raw model margin (unclipped)",
        "one_year": {}, "inhospital": {},
    }

    pairs = [
        ("one_year", "death_365", "death_365_unbounded", p_1y_lr, p_1y_xg, eta_1y_lr, eta_1y_xg),
        ("inhospital", "death_hosp", "death_hosp_unbounded", p_ih_lr, p_ih_xg,
         eta_ih_lr, eta_ih_xg),
    ]
    for tag, ycol, ycol_u, pl, px, el, ex in pairs:
        y_b = df[ycol].values.astype(int)
        y_u = df[ycol_u].values.astype(int)
        log("")
        log("[%s]  有界 %d 例事件 / 无界 %d 例事件" % (tag, y_b.sum(), y_u.sum()))
        for mdl, p, e in (("Logistic", pl, el), ("XGBoost", px, ex)):
            b = metrics(y_b, p, e)
            u = metrics(y_u, p, e)
            res[tag][mdl] = {"bounded": b, "unbounded": u,
                             "delta": {k: u[k] - b[k] for k in ("auc", "brier", "slope", "oe")}}
            log("  %-8s 有界 AUC %.4f  slope %.4f  O:E %.4f | 无界 AUC %.4f  slope %.4f  O:E %.4f"
                % (mdl, b["auc"], b["slope"], b["oe"], u["auc"], u["slope"], u["oe"]))

    # ---- 断言：有界一侧必须复现主分析（表 4）
    checks = [("one_year", "Logistic", "NWICU_1yr", "Logistic"),
              ("one_year", "XGBoost", "NWICU_1yr", "XGBoost"),
              ("inhospital", "Logistic", "MIMIC_internal_inhosp", None),
              ("inhospital", "XGBoost", "MIMIC_internal_inhosp", None)]
    for tag, mdl, master_key, _ in checks[:2]:
        got = res[tag][mdl]["bounded"]
        exp = master[master_key][mdl]["point"]
        for f in ("auc", "slope", "oe", "brier"):
            assert abs(got[f] - exp[f]) < 5e-6, \
                "有界 %s/%s 的 %s 与主分析不符：%.8f vs %.8f" % (tag, mdl, f, got[f], exp[f])
        log("  [OK] %s/%s 有界一侧与 v9_number_master 逐位一致" % (tag, mdl))

    # 院内模型在主分析里没有 NWICU 行（只有 MIMIC 内部），故改用内部一致性检查：
    # 主分析 NWICU 院内 XGBoost AUC 0.8644997 来自 leakage_sensitivity.json 的 full 模型。
    # ⚠ 必须读 v9_results 里的那一份：根级 leakage_sensitivity.json 是 V8 时代的旧件，
    #   其 NWICU 行用的是未重导的死亡标签（AUC 0.684），不能用来对账。
    _lk = json.load(open(OUT / "v9_results" / "leakage_sensitivity.json", encoding="utf-8"))
    xg_inh = res["inhospital"]["XGBoost"]["bounded"]["auc"]
    xg_lk = _lk["external"]["B_inhosp|NWICU"]["XGBoost"]["full"]
    assert abs(xg_inh - xg_lk) < 1e-9, \
        "NWICU 院内 XGBoost AUC 与 leakage 敏感性文件的 full 模型不一致：%.10f vs %.10f" \
        % (xg_inh, xg_lk)
    log("  [OK] NWICU 院内 XGBoost AUC = %.10f，与 leakage 文件 full 模型逐位一致" % xg_inh)

    # 3 位小数上是否有任何已报数字改变
    changed = []
    for tag in ("one_year", "inhospital"):
        for mdl in ("Logistic", "XGBoost"):
            b, u = res[tag][mdl]["bounded"], res[tag][mdl]["unbounded"]
            for f in ("auc", "slope", "oe"):
                if round(b[f], 3) != round(u[f], 3):
                    changed.append("%s.%s.%s: %.3f → %.3f" % (tag, mdl, f, b[f], u[f]))
    res["_changed_at_3dp"] = changed
    log("")
    log("3 位小数上发生变化的已报数字：%d 处" % len(changed))
    for c in changed:
        log("   " + c)

    with open(OUT / "death_label_bounds_impact_primary_v1.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    # 附录构建器（65）在 V9 口径下把 OUT 指向 output/v9_results/，故两处各写一份，
    # 内容完全相同 —— 避免"正文读到的是另一份"这类口径分裂。
    (OUT / "v9_results").mkdir(exist_ok=True)
    with open(OUT / "v9_results" / "death_label_bounds_impact_primary_v1.json",
              "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    log("")
    log("[OK] 写出 death_label_bounds_impact_primary_v1.json（output/ 与 output/v9_results/）")
    (OUT / "_124_impact_primary.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
