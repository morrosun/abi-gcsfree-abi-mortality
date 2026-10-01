# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
63_check_slope_epsilon.py —— 审计：校准斜率是否依赖"裁剪常数"这一任意选择

触发：62 脚本算出淮安队列 Logistic 校准斜率 = 0.899，而 V7 稿件 Table 5 写的是 0.80。
两者用的是同一批预测概率（AUC 完全一致 0.8646），差别只可能来自
  24_local_inhosp_validation.py : np.clip(p, 1e-9, 1-1e-9)   -> slope 0.800
  62_calib_ci_and_severity.py  : np.clip(p, 1e-6, 1-1e-6)   -> slope 0.899
即 logit 变换前的裁剪常数。n=225、55 事件的小队列里，个别极端预测值会主导斜率。

本脚本逐队列量化：换裁剪常数，斜率动多少？
产出：output/slope_epsilon_audit.json（新增文件）
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression   # noqa: E402

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
EPS_LIST = [1e-9, 1e-6, 1e-4, 1e-3, 1e-2]


def log(*a):
    print(*a)
    sys.stdout.flush()


def slope_intercept(y, p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    m = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def main():
    log("=" * 78)
    log("63 | 校准斜率对裁剪常数 ε 的敏感性审计")
    nw = pd.read_csv(DATA / "nwicu_abi_final_vp2.csv")   # 74：恢复血管活性药
    ins = pd.read_csv(DATA / "inspire_abi_final.csv")
    dh = pd.read_csv(DATA / "inspire_deathhosp.csv")
    ins = ins.merge(dh, on='op_id', how='left')
    ins['death_hosp'] = ins['death_hosp'].fillna(0).astype(int)
    e = pd.read_csv(DATA / "eicu_abi_final.csv")

    pp = joblib.load(OUT / "preproc.joblib")
    models = joblib.load(OUT / "models.joblib")
    feats, cont = pp['feats'], pp['cont']
    scaler, med1 = pp['scaler'], pd.Series(pp['medians'])
    logit = pd.Series(models['logit_params'])
    xgbm = models['xgb']

    def p1(df):
        Z = df[feats].copy()
        for c in cont:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(med1[c])
        for c in AET + ['mech_vent', 'vasopressor', 'rrt']:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0).astype(int)
        Zs = Z.copy()
        Zs[cont] = scaler.transform(Z[cont])
        return (1 / (1 + np.exp(-(logit['const'] + Zs[feats].values @ logit[feats].values))),
                xgbm.predict_proba(Z[feats].values)[:, 1])

    m = joblib.load(OUT / "model_inhosp.joblib")
    uf, uc = m['feats'], m['cont']

    def p2(df_):
        Z = df_[uf].copy()
        for c in uc:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(m['medians'][c])
        for c in uf:
            if c not in uc:
                Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
        Zs = Z.copy()
        Zs[uc] = m['scaler'].transform(Z[uc])
        return m['lr'].predict_proba(Zs[uf])[:, 1], m['xg'].predict_proba(Z[uf])[:, 1]

    PRED = {}
    for tag, df, yc, fn in [('NWICU_1yr', nw, 'death_365', p1), ('INSPIRE_1yr', ins, 'death_365', p1),
                            ('eICU_inhosp', e, 'death_hosp', p2)]:
        a, b = fn(df)
        PRED[tag] = {'y': df[yc].values.astype(int), 'Logistic': a, 'XGBoost': b}

    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')     # noqa: E731
    d = pd.DataFrame(index=raw.index)
    d['age'] = num('age')
    d['y'] = (num('hospital_discharge_location') == 5).astype(int)
    for c in AET:
        d[c] = num(c).fillna(0)
    keep = (d[AET].sum(axis=1) >= 1) & (d['age'] >= 18) & d['age'].notna() & \
        num('hospital_discharge_location').notna()
    dx = d[keep].copy()
    Xl = pd.DataFrame(index=dx.index)
    for c in uf:
        Xl[c] = num(c) if c in raw.columns else np.nan
    CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
            'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
    for c, f in CONV.items():
        Xl[c] = f(Xl[c])
    for c in uc:
        Xl[c] = Xl[c].fillna(m['medians'][c])
    for c in uf:
        if c not in uc:
            Xl[c] = Xl[c].fillna(0)
    Xls = Xl.copy()
    Xls[uc] = m['scaler'].transform(Xl[uc])
    PRED['Huaian_inhosp'] = {'y': dx['y'].values.astype(int),
                             'Logistic': m['lr'].predict_proba(Xls[uf])[:, 1],
                             'XGBoost': m['xg'].predict_proba(Xl[uf])[:, 1]}

    RES = {}
    all_ok = True
    for tag, d0 in PRED.items():
        RES[tag] = {'n': int(len(d0['y'])), 'events': int(d0['y'].sum())}
        for mname in ['Logistic', 'XGBoost']:
            p = d0[mname]
            row = {'pred_min': round(float(np.min(p)), 6), 'pred_max': round(float(np.max(p)), 6),
                   'pred_p01': round(float(np.percentile(p, 1)), 6),
                   'pred_p99': round(float(np.percentile(p, 99)), 6),
                   'by_eps': {}}
            for eps in EPS_LIST:
                s, i = slope_intercept(d0['y'], p, eps)
                row['by_eps'][str(eps)] = {'slope': round(s, 4), 'intercept': round(i, 4)}
            slopes = [row['by_eps'][str(x)]['slope'] for x in EPS_LIST]
            row['slope_range'] = round(max(slopes) - min(slopes), 4)
            RES[tag][mname] = row
            log(f"  {tag:<16} {mname:<8} n={RES[tag]['n']:>6,}  "
                f"p∈[{row['pred_min']:.4f},{row['pred_max']:.4f}]  "
                + "  ".join(f"ε={x:g}:{row['by_eps'][str(x)]['slope']:.3f}" for x in EPS_LIST)
                + f"   |range|={row['slope_range']:.4f}")

    # 判定：区分「可复现」与「脆弱」。脆弱不是失败，而是必须在稿件中声明。
    log("\n 判定：ε 敏感性分级（STABLE ≤0.01 / FRAGILE >0.01）")
    for tag in RES:
        for mname in ['Logistic', 'XGBoost']:
            r = RES[tag][mname]['slope_range']
            grade = 'STABLE ' if r <= 0.01 else 'FRAGILE'
            RES[tag][mname]['grade'] = grade.strip()
            log(f"  [{grade}] {tag:<16} {mname:<8} |slope range over ε| = {r:.4f}")
    fragile = [(t, m) for t in RES for m in ['Logistic', 'XGBoost']
               if RES[t][m]['slope_range'] > 0.01]
    log(f"\n  FRAGILE 组合 {len(fragile)} 个: {fragile}")
    log("  → 这些组合的校准斜率被『预测概率接近 0/1 的少数样本』主导，")
    log("     稿件必须写明所用 ε，并同时报告 bootstrap CI。")

    # 与稿件数字对账（各按原脚本所用 ε）
    log("\n 与 V7 稿件对账")
    for tag, mname, eps, exp_s, exp_i, src in [
            ('Huaian_inhosp', 'Logistic', '1e-09', 0.80, -0.08, 'Table 5'),
            ('Huaian_inhosp', 'XGBoost', '1e-09', 0.85, -0.22, 'Table 5'),
            ('NWICU_1yr', 'Logistic', '1e-06', 0.04, None, '正文 (slope 0.04)'),
            # 74：NWICU 血管活性药恢复真值后 XGBoost 斜率 0.832 → 0.838
            ('NWICU_1yr', 'XGBoost', '1e-06', 0.838, None, '正文 (O:E 0.96 段)'),
            ('eICU_inhosp', 'Logistic', '1e-06', 0.675, None, '正文 (slope 0.68)'),
            ('eICU_inhosp', 'XGBoost', '1e-06', 0.703, None, '正文 (slope 0.70)'),
            ('INSPIRE_1yr', 'Logistic', '1e-06', 0.807, None, 'recalibration_v2'),
            ('INSPIRE_1yr', 'XGBoost', '1e-06', 0.802, None, 'recalibration_v2')]:
        got = RES[tag][mname]['by_eps'][eps]['slope']
        good = abs(got - exp_s) <= 0.006
        all_ok &= good
        extra = f"  intercept {RES[tag][mname]['by_eps'][eps]['intercept']:.4f} vs {exp_i}" \
            if exp_i is not None else ''
        log(f"  [{'OK ' if good else 'FAIL'}] {src:<18} {tag:<15} {mname:<8} ε={eps} "
            f"slope {got:.4f} vs {exp_s}{extra}")

    json.dump({'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'eps_list': EPS_LIST,
               'conclusion': ('Calibration slope is insensitive to the clipping constant in the '
                              'large cohorts (range <=0.01) but NOT in the 225-patient Huaian '
                              'cohort; the manuscript must state the epsilon used.'),
               'results': RES},
              open(OUT / "slope_epsilon_audit.json", 'w', encoding='utf-8'), indent=2)
    log("\nwrote output/slope_epsilon_audit.json")
    log("=" * 78)
    log("ALL CHECKS PASSED" if all_ok else "!!! SELF-CHECK FAILED")
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
