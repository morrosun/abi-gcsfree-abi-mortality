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
89 —— 哨兵值 / 超生理范围值清洗，并重算外部验证（评审 A5「追查极端 η 的来源」）

88 号脚本的发现（本脚本的起因）
------------------------------
  NWICU 的 Logistic 线性预测值 η ∈ [−81,785, 211,788]，sd = 19,563
  （MIMIC 内部测试集 η ∈ [−4.91, 6.33]，sd = 1.51）—— **相差 12,980 倍**。
  逐特征归因显示 glucose_max 的标准化 z 值 99 分位达 104,959。
  核查原始值后确认：NWICU 抽取时**未清洗哨兵值 9,999,999**：

      glucose_max    605 例（17.7% 的队列）
      potassium_max   29 例
      wbc_max          2 例
      creatinine_max   2 例
      bun_max          1 例

  这些值标准化后 z ≈ 2.5×10^5，直接把 Logistic 的线性预测值推到 ±10^5，
  于是校准斜率被压缩到 0.038；而 XGBoost 不吃标准化、只依赖变量序关系，
  所以同一队列上 XGBoost 的斜率是 0.906（正常）—— 这个"一个崩、一个不崩"的
  分裂现象正是「标准化 + 未清洗极值」的指纹。

清洗规则（保守：只剔除临床上不可能出现的值，不做任意缩尾）
------------------------------------------------------------
  各变量取生理可 plausible 区间，区间外一律置为缺失（随后按既有流程中位数插补）。
  区间刻意放宽，宁可少清洗，也不误删真实危重值。

产物
----
  data/nwicu_abi_final_v4.csv、data/eicu_abi_final_v3.csv、data/inspire_abi_final_v3.csv
  （仅当有值被清洗时才另存新版本；MIMIC 若需清洗则报错提示需重跑 87）
  output/sentinel_audit_v1.json、output/_89_sentinel_clean.md
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

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
EPS = 1e-6

# 生理可 plausible 区间（单位沿用 MIMIC-IV）
# ★ 刻意放宽：ICU 里体温 25℃、Hb 1.5 g/dL、SpO2 30% 都是真实的危重值，不能洗。
#   本区间只为剔除「荒谬值」（哨兵 9,999,999、负值、量级错误），不做临床缩尾。
#   （首版把下界设得偏紧，误判 MIMIC 40 例真实低值，已修正。）
RANGE = {
    'heart_rate_mean': (10, 300), 'mbp_mean': (5, 300), 'resp_rate_mean': (1, 100),
    'temperature_mean': (20, 45), 'spo2_mean': (20, 100),
    'wbc_max': (0.01, 1000), 'hemoglobin_min': (0.5, 30), 'platelets_min': (1, 3000),
    # sodium 74 mEq/L、creatinine 80 mg/dL 在 ICU 均属真实可见的危重值 → 上/下界继续放宽，
    # 只让哨兵值（9,999,999）与量级错误落入清洗范围
    'sodium_min': (60, 200), 'potassium_max': (0.5, 15), 'creatinine_max': (0.01, 150),
    'bun_max': (0.5, 300), 'glucose_max': (10, 5000), 'bicarbonate_min': (1, 60),
    'inr_max': (0.1, 30),
}
SENTINEL = 999999  # NWICU 抽取中观察到的哨兵值量级下限

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def logit_f(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_slope_intercept(y, p):
    lp = logit_f(p)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=5000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def metrics(y, p, eps=EPS):
    s, i = cal_slope_intercept(y, p)
    pc = np.clip(p, eps, 1 - eps)
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                slope=s, intercept=i, oe=float(np.mean(y) / np.mean(p)),
                n_clipped=int(((p < eps) | (p > 1 - eps)).sum()))


def clean(df, tag):
    """按生理区间清洗，返回 (清洗后 df, 逐变量清洗统计)"""
    d = df.copy()
    stat = {}
    for c, (lo, hi) in RANGE.items():
        if c not in d.columns:
            continue
        s = pd.to_numeric(d[c], errors='coerce')
        bad = (s < lo) | (s > hi)
        n_bad = int(bad.sum())
        n_sent = int((s.abs() >= SENTINEL).sum())
        if n_bad:
            d.loc[bad, c] = np.nan
        stat[c] = dict(n_out_of_range=n_bad, n_sentinel=n_sent,
                       pct_out_of_range=round(100 * n_bad / len(d), 2),
                       min=float(s.min()) if s.notna().any() else None,
                       max=float(s.max()) if s.notna().any() else None,
                       median=float(s.median()) if s.notna().any() else None)
    log("  [%s] 清洗统计（仅列出有清洗的变量）" % tag)
    any_bad = False
    for c, v in stat.items():
        if v['n_out_of_range']:
            any_bad = True
            log("    %-20s 越界 %5d 例 (%.2f%%)  哨兵值 %5d 例  原 max=%.1f  med=%.2f"
                % (c, v['n_out_of_range'], v['pct_out_of_range'], v['n_sentinel'],
                   v['max'], v['median']))
    if not any_bad:
        log("    无越界值")
    return d, stat


def apply_frozen(art, df):
    Z = df[FEATS].copy()
    for c in CONT:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(art['medians'][c])
    for c in FEATS:
        if c not in CONT:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
    Zs = Z.copy()
    Zs[CONT] = art['scaler'].transform(Z[CONT])
    lp = art['logit_params']
    eta = lp['const'] + Zs[FEATS].values @ np.array([lp[f] for f in FEATS])
    eta_c = np.clip(eta, -700, 700)
    a = 1 / (1 + np.exp(-eta_c))
    b = art['xgb'].predict_proba(Z[FEATS].values)[:, 1]
    return a, b, eta


def main():
    log("=" * 78)
    log("89 —— 哨兵值 / 超生理范围值审计与清洗")
    log("=" * 78)

    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    nw = pd.read_csv(DATA / "nwicu_abi_final_v3.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v2.csv")
    ei = pd.read_csv(DATA / "eicu_abi_final_v2.csv")

    mim_c, st_mim = clean(mim, "MIMIC-IV 开发集")
    nw_c, st_nw = clean(nw, "NWICU")
    ins_c, st_ins = clean(ins, "INSPIRE")
    ei_c, st_ei = clean(ei, "eICU")

    n_mim = sum(v['n_out_of_range'] for v in st_mim.values())
    if n_mim:
        log("")
        log("  ⚠ MIMIC-IV 开发集也有 %d 个越界值 → 清洗后必须重跑 87 重训，本脚本停止"
            % n_mim)
        json.dump(dict(mimic=st_mim, nwicu=st_nw, inspire=st_ins, eicu=st_ei,
                       stopped="MIMIC needs cleaning; rerun 87 first"),
                  open(OUT / "sentinel_audit_v1.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        return

    nw_c.to_csv(DATA / "nwicu_abi_final_v4.csv", index=False)
    ins_c.to_csv(DATA / "inspire_abi_final_v3.csv", index=False)
    ei_c.to_csv(DATA / "eicu_abi_final_v3.csv", index=False)
    log("")
    log("  → data/nwicu_abi_final_v4.csv、inspire_abi_final_v3.csv、eicu_abi_final_v3.csv")

    # ---------------------------------------------------- 清洗前后对比
    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    art1 = dict(feats=pp['feats'], cont=pp['cont'], scaler=pp['scaler'],
                medians=pp['medians'], logit_params=md['logit_params'], xgb=md['xgb'])
    mi = joblib.load(OUT / "model_inhosp_v2.joblib")
    art2 = dict(feats=mi['feats'], cont=mi['cont'], scaler=mi['scaler'],
                medians=mi['medians'], logit_params=None, lr=mi['lr'], xg=mi['xg'])

    log("")
    log("=" * 78)
    log("清洗前后 —— 外部验证指标对比（冻结模型不变，只换清洗后的输入）")
    log("=" * 78)
    res = {}
    for tag, old, new, yc in [('NWICU_1yr', nw, nw_c, 'death_365'),
                              ('INSPIRE_1yr', ins, ins_c, 'death_365')]:
        y = new[yc].values.astype(int)
        res[tag] = {}
        for label, df in (("before", old), ("after", new)):
            a, b, eta = apply_frozen(art1, df)
            ma, mb = metrics(y, a), metrics(y, b)
            res[tag][label] = {'Logistic': ma, 'XGBoost': mb,
                               'eta_sd': float(np.std(eta)),
                               'eta_min': float(np.min(eta)), 'eta_max': float(np.max(eta))}
        for mname in ('Logistic', 'XGBoost'):
            o = res[tag]['before'][mname]
            n_ = res[tag]['after'][mname]
            log("  %-14s %-8s  AUC %.4f→%.4f   slope %.4f→%.4f   O:E %.3f→%.3f   "
                "Brier %.4f→%.4f  被裁剪 %d→%d"
                % (tag, mname, o['auc'], n_['auc'], o['slope'], n_['slope'],
                   o['oe'], n_['oe'], o['brier'], n_['brier'],
                   o['n_clipped'], n_['n_clipped']))
        log("    η sd %.1f → %.2f ；η 范围 [%.0f, %.0f] → [%.2f, %.2f]"
            % (res[tag]['before']['eta_sd'], res[tag]['after']['eta_sd'],
               res[tag]['before']['eta_min'], res[tag]['before']['eta_max'],
               res[tag]['after']['eta_min'], res[tag]['after']['eta_max']))

    # eICU（院内模型）
    def apply_inhosp(df_):
        Z = df_[FEATS].copy()
        for c in CONT:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(art2['medians'][c])
        for c in FEATS:
            if c not in CONT:
                Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
        Zs = Z.copy()
        Zs[CONT] = art2['scaler'].transform(Z[CONT])
        return art2['lr'].predict_proba(Zs[FEATS])[:, 1], \
            art2['xg'].predict_proba(Z[FEATS])[:, 1]

    y = ei_c['death_hosp'].values.astype(int)
    res['eICU_inhosp'] = {}
    for label, df in (("before", ei), ("after", ei_c)):
        a, b = apply_inhosp(df)
        res['eICU_inhosp'][label] = {'Logistic': metrics(y, a), 'XGBoost': metrics(y, b)}
    for mname in ('Logistic', 'XGBoost'):
        o = res['eICU_inhosp']['before'][mname]
        n_ = res['eICU_inhosp']['after'][mname]
        log("  %-14s %-8s  AUC %.4f→%.4f   slope %.4f→%.4f   O:E %.3f→%.3f"
            % ('eICU_inhosp', mname, o['auc'], n_['auc'], o['slope'], n_['slope'],
               o['oe'], n_['oe']))

    json.dump(dict(mimic=st_mim, nwicu=st_nw, inspire=st_ins, eicu=st_ei, compare=res),
              open(OUT / "sentinel_audit_v1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(OUT / "_89_sentinel_clean.md", "w", encoding="utf-8") as f:
        f.write("# 89 · 哨兵值清洗与重算\n\n```\n" + "\n".join(LOG) + "\n```\n")
    log("")
    log("→ output/sentinel_audit_v1.json / _89_sentinel_clean.md")


if __name__ == "__main__":
    main()
