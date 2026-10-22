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
88 —— 用原始线性预测值 η 重估校准，并追查极端 η 的来源（评审 A4 / A5）

评审原话
--------
  「对于 Logistic 模型，可以直接使用原始线性预测值 η = β0 + Σβj·zj，然后拟合
    logit P(Y=1) = α + γη —— 不需要先把 η 转成概率、裁剪、再转回 logit。」
  「应报告每个 ε 下有多少预测概率被裁剪；检查极端 η 由哪些特征贡献；
    尤其核查 NWICU 的极端标准化值、单位、变量映射及异常值。」

87 号脚本重训后出现的硬事实
----------------------------
  NWICU  ● Logistic slope = 0.0382（修正死亡标签 16→378、治疗指标改首日之后**仍近乎不变**）
         ● XGBoost  slope = 0.9060（同一队列、同一标签，完全正常）
  → 校准失败不是标签缺陷造成的，而是**Logistic 特有的**；
    且 87 在 NWICU 上触发了 np.exp 溢出 → 线性预测值 η 极端。本脚本把它查到底。

本脚本回答四个问题
------------------
  Q1 用原始 η 直接拟合得到的校准斜率是多少？（不裁剪，无需 logit 变换）
  Q2 现行"概率裁剪后转 logit"的做法在 ε 取不同值时斜率如何变化？各 ε 下有多少预测被裁剪？
  Q3 NWICU 的 η 分布相对 MIMIC 内部测试集偏离多少？
  Q4 极端 η 由哪些特征的标准化值贡献？（逐特征分解 βj·zj）

产物：output/eta_calibration_audit_v1.json、output/_88_eta_calibration.md
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression  # noqa: E402

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
FEATS = ['age', 'female'] + AET + ['charlson_comorbidity_index'] + \
        [c for c in CONT if c not in ('age', 'charlson_comorbidity_index')] + BINV
EPS_LIST = [1e-6, 1e-4, 1e-3, 1e-2]
SEED = 42

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def fit_slope(y, x):
    """logit P(Y=1) = α + γ·x 的最小二乘式 logistic 拟合（无正则）"""
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=5000).fit(
        np.asarray(x, dtype=float).reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def prep(art, df):
    Z = df[FEATS].copy()
    for c in CONT:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(art['medians'][c])
    for c in FEATS:
        if c not in CONT:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
    Zs = Z.copy()
    Zs[CONT] = art['scaler'].transform(Z[CONT])
    return Z, Zs


def main():
    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    art = dict(feats=pp['feats'], cont=pp['cont'], scaler=pp['scaler'],
               medians=pp['medians'], logit_params=md['logit_params'], xgb=md['xgb'])
    lp = art['logit_params']
    beta = np.array([lp[f] for f in FEATS])

    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    from sklearn.model_selection import train_test_split
    _, Xte, _, yte = train_test_split(mim[FEATS], mim['death_365'].values.astype(int),
                                      test_size=0.30, stratify=mim['death_365'], random_state=SEED)
    nw = pd.read_csv(DATA / "nwicu_abi_final_v3.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v2.csv")

    cohorts = [('MIMIC_internal_1yr', mim.loc[Xte.index], yte, 'death_365'),
               ('NWICU_1yr', nw, None, 'death_365'),
               ('INSPIRE_1yr', ins, None, 'death_365')]

    res = {}
    log("=" * 78)
    log("Q1 / Q2 —— η 直接拟合 vs 概率裁剪后转 logit（各 ε）")
    log("=" * 78)
    for tag, df, yv, yc in cohorts:
        Z, Zs = prep(art, df)
        y = yv if yv is not None else df[yc].values.astype(int)
        eta_lr = lp['const'] + Zs[FEATS].values @ beta
        eta_xg = art['xgb'].predict(Z[FEATS].values, output_margin=True)
        blk = {}
        for mname, eta in (("Logistic", eta_lr), ("XGBoost", eta_xg)):
            s_eta, i_eta = fit_slope(y, eta)
            d = {'slope_eta': s_eta, 'intercept_eta': i_eta,
                 'eta_min': float(np.min(eta)), 'eta_max': float(np.max(eta)),
                 'eta_sd': float(np.std(eta)), 'eta_p1': float(np.percentile(eta, 1)),
                 'eta_p99': float(np.percentile(eta, 99)),
                 'by_eps': {}}
            for eps in EPS_LIST:
                p = 1 / (1 + np.exp(-np.clip(eta, -700, 700)))
                pc = np.clip(p, eps, 1 - eps)
                nclip = int(((p < eps) | (p > 1 - eps)).sum())
                s_e, i_e = fit_slope(y, np.log(pc / (1 - pc)))
                d['by_eps'][str(eps)] = dict(slope=s_e, intercept=i_e,
                                             n_clipped=nclip,
                                             pct_clipped=round(100 * nclip / len(y), 2))
            blk[mname] = d
            log("")
            log("  %s · %s   η ∈ [%.2f, %.2f]  sd=%.2f" % (tag, mname, d['eta_min'],
                                                           d['eta_max'], d['eta_sd']))
            log("    直接拟合 η        : slope %.4f  intercept %.4f" % (s_eta, i_eta))
            for eps in EPS_LIST:
                b = d['by_eps'][str(eps)]
                log("    ε=%-6g 裁剪后 : slope %.4f intercept %.4f  被裁剪 %5d 例 (%.2f%%)"
                    % (eps, b['slope'], b['intercept'], b['n_clipped'], b['pct_clipped']))
        res[tag] = dict(n=int(len(y)), events=int(np.sum(y)), models=blk)

    # ---------------------------------------------------------- Q3 / Q4
    log("")
    log("=" * 78)
    log("Q3 —— η 分布：NWICU 相对 MIMIC 内部测试集偏离多少")
    log("=" * 78)
    for tag in ('MIMIC_internal_1yr', 'NWICU_1yr', 'INSPIRE_1yr'):
        b = res[tag]['models']['Logistic']
        log("  %-20s η 中位数区间 p1..p99 = [%.2f, %.2f] ；sd=%.2f ；min/max = %.2f / %.2f"
            % (tag, b['eta_p1'], b['eta_p99'], b['eta_sd'], b['eta_min'], b['eta_max']))
    spread = res['NWICU_1yr']['models']['Logistic']['eta_sd'] / \
        res['MIMIC_internal_1yr']['models']['Logistic']['eta_sd']
    log("")
    log("  NWICU / MIMIC 的 η 标准差之比 = %.3f   ← η 越分散，校准斜率越趋于 0"
        % spread)

    log("")
    log("=" * 78)
    log("Q4 —— NWICU 极端 η 的逐特征归因（Logistic：贡献 = βj · zj）")
    log("=" * 78)
    Z, Zs = prep(art, nw)
    eta = lp['const'] + Zs[FEATS].values @ beta
    contrib = pd.DataFrame(Zs[FEATS].values * beta, columns=FEATS)
    contrib['eta'] = eta
    hi = contrib.nlargest(300, 'eta')
    lo = contrib.nsmallest(300, 'eta')
    summ = pd.DataFrame({
        'beta': pd.Series(beta, index=FEATS),
        'mean_contrib_all': contrib[FEATS].mean(),
        'mean_contrib_top300': hi[FEATS].mean(),
        'mean_contrib_bottom300': lo[FEATS].mean(),
        'abs_mean_top300': hi[FEATS].abs().mean(),
        'z_abs_p99': pd.Series([np.percentile(np.abs(Zs[f].values), 99) for f in FEATS], index=FEATS),
        'missing_pct_nwicu': pd.Series([100 * nw[f].isna().mean() if f in nw.columns else np.nan
                                        for f in FEATS], index=FEATS),
    }).sort_values('abs_mean_top300', ascending=False)
    log("")
    log(summ.round(3).to_string())
    res['nwicu_feature_attribution'] = {
        f: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        for f, row in summ.iterrows()}
    res['nwicu_eta_spread_ratio_vs_mimic'] = float(spread)

    # 特征缺失率（NWICU 上插补为中位数 → 全部落在 z=0）
    log("")
    log("  NWICU 中 100%% 缺失、被整体插补为训练集中位数的特征：")
    miss = {f: round(100 * float(nw[f].isna().mean()), 1) for f in FEATS
            if f in nw.columns and nw[f].isna().mean() > 0.5}
    log("   " + (", ".join("%s %.0f%%" % (k, v) for k, v in miss.items()) or "（无）"))
    res['nwicu_high_missing_features'] = miss

    json.dump(res, open(OUT / "eta_calibration_audit_v1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(OUT / "_88_eta_calibration.md", "w", encoding="utf-8") as f:
        f.write("# 88 · η 校准审计\n\n```\n" + "\n".join(LOG) + "\n```\n")
    log("")
    log("→ output/eta_calibration_audit_v1.json / _88_eta_calibration.md")


if __name__ == "__main__":
    main()
