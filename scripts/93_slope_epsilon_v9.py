# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
93_slope_epsilon_v9.py —— V9 终态：校准斜率的 ε 敏感性 + η 直接拟合对照

回应评审第 5 条（A5）：
  ① 校准斜率应当**直接用原始线性预测值 η** 拟合 logit P(Y=1)=α+γη，
     不必先转概率、裁剪、再转回 logit；
  ② 若仍用概率裁剪法，必须报告每个 ε 下**有多少预测被裁剪**；
  ③ 稿件须写明：斜率与基于 logit 的截距依赖 ε；O:E、ECE 不经过 logit 变换，
     不依赖 ε —— 不要把"每个校准统计量都依赖 ε"的错误印象留给读者。

★ 与 88 的区别：88 是在**哨兵值清洗之前**跑的（那次跑出了根因：NWICU η sd=19,563、
  18.4% 预测被裁剪、斜率完全由 ε 决定）。本脚本在 89 清洗之后的终态数据上跑，
  产出的是**要进稿件的数字**。88 的证据作为"发现过程"单独保留。

★ 对账：ε=1e-6 的斜率必须等于 87 主结果 JSON 的 slope（87 用 EPS=1e-6）。

产出：output/slope_epsilon_audit_v9.json + _93_slope_epsilon_v9.md
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.model_selection import train_test_split          # noqa: E402

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


def huaian_frame():
    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')       # noqa: E731
    d = pd.DataFrame(index=raw.index)
    d['y'] = (num('hospital_discharge_location') == 5).astype(int)
    for c in AET:
        d[c] = num(c).fillna(0)
    keep = (d[AET].sum(axis=1) >= 1) & (num('age') >= 18) & \
        num('age').notna() & num('hospital_discharge_location').notna()
    dx = d[keep].copy()
    Xl = pd.DataFrame(index=dx.index)
    for c in FEATS:
        Xl[c] = num(c) if c in raw.columns else np.nan
    CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
            'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
    for c, f in CONV.items():
        Xl[c] = f(Xl[c])
    return Xl, dx['y'].values.astype(int)


def block(tag, art, df, y, ref=None):
    Z, Zs = prep(art, df)
    # 1 年模型 artefacts 存的是 logit_params（系数向量）；院内模型存的是 lr 对象 → 两条路径取 η
    if 'logit_params' in art:
        lp = art['logit_params']
        beta = np.array([lp[f] for f in FEATS])
        eta_lr = lp['const'] + Zs[FEATS].values @ beta
    else:
        eta_lr = art['lr'].decision_function(Zs[FEATS])
    xg_ = art.get('xg', art.get('xgb'))
    eta_xg = xg_.predict(Z[FEATS].values, output_margin=True)
    out = {'n': int(len(y)), 'events': int(np.sum(y))}
    log("")
    log("  %s  n=%d  events=%d" % (tag, len(y), int(np.sum(y))))
    for mname, eta in [('Logistic', eta_lr), ('XGBoost', eta_xg)]:
        s_eta, i_eta = fit_slope(y, eta)
        d = {'slope_eta': s_eta, 'intercept_eta': i_eta,
             'eta_min': float(np.min(eta)), 'eta_max': float(np.max(eta)),
             'eta_sd': float(np.std(eta)), 'by_eps': {}}
        for eps in EPS_LIST:
            p = 1 / (1 + np.exp(-np.clip(eta, -700, 700)))
            pc = np.clip(p, eps, 1 - eps)
            nclip = int(((p < eps) | (p > 1 - eps)).sum())
            s_e, i_e = fit_slope(y, np.log(pc / (1 - pc)))
            d['by_eps'][str(eps)] = {'slope': s_e, 'intercept': i_e, 'n_clipped': nclip,
                                     'pct_clipped': round(100 * nclip / len(y), 2)}
        d['slope_range_over_eps'] = round(
            max(v['slope'] for v in d['by_eps'].values()) -
            min(v['slope'] for v in d['by_eps'].values()), 4)
        d['grade'] = 'STABLE' if d['slope_range_over_eps'] <= 0.01 else 'FRAGILE'
        out[mname] = d
        log("    %-8s η∈[%.2f,%.2f] sd=%.2f | 直接拟合 η slope %.4f" %
            (mname, d['eta_min'], d['eta_max'], d['eta_sd'], s_eta))
        for eps in EPS_LIST:
            b = d['by_eps'][str(eps)]
            log("      ε=%-6g slope %.4f  被裁剪 %5d 例 (%.2f%%)" %
                (eps, b['slope'], b['n_clipped'], b['pct_clipped']))
        log("      |range over ε| = %.4f  → %s" % (d['slope_range_over_eps'], d['grade']))
        if ref is not None:
            # 容差 1e-6（非 1e-9）：lbfgs 迭代收敛存在 ~1e-9 级浮点差异，
            # 属求解器精度，不是口径差异。真正要防的是"换了一批数据/换了一个模型"。
            want = ref[mname]['point']['slope']
            got = d['by_eps']['1e-06']['slope']
            assert abs(got - want) < 1e-6, \
                f"RECONCILE FAIL: {tag}/{mname} ε=1e-6 slope {got!r} != 87 的 {want!r}"
            log("      [reconcile OK] ε=1e-6 slope %.6f ≈ 87 的 %.6f (Δ%.2e)" %
                (got, want, abs(got - want)))
    return out


def main():
    log("#" * 78)
    log("# 93_slope_epsilon_v9.py | 校准斜率的 ε 敏感性（89 清洗后终态）+ η 直接拟合对照")
    log("#" * 78)
    ref = json.load(open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))

    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    a1 = dict(feats=FEATS, cont=CONT, scaler=pp['scaler'], medians=pp['medians'],
              logit_params=md['logit_params'], xgb=md['xgb'])
    a2 = joblib.load(OUT / "model_inhosp_v2.joblib")
    a2 = dict(feats=a2['feats'], cont=a2['cont'], scaler=a2['scaler'],
              medians=a2['medians'], lr=a2['lr'], xg=a2['xg'])

    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    nw = pd.read_csv(DATA / "nwicu_abi_final_v6.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v3.csv")
    e = pd.read_csv(DATA / "eicu_abi_final_v3.csv")

    _, Xte, _, yte = train_test_split(mim[FEATS], mim['death_365'].values.astype(int),
                                      test_size=0.30, stratify=mim['death_365'],
                                      random_state=SEED)
    _, Xte2, _, yte2 = train_test_split(mim[FEATS], mim['death_hosp'].values.astype(int),
                                        test_size=0.30, stratify=mim['death_hosp'],
                                        random_state=SEED)

    R = {}
    log("\n=== 1 年死亡模型 ===")
    R['MIMIC_internal_1yr'] = block('MIMIC_internal_1yr', a1, mim.loc[Xte.index], yte,
                                    ref['MIMIC_internal_1yr'])
    R['NWICU_1yr'] = block('NWICU_1yr', a1, nw, nw['death_365'].values.astype(int),
                           ref['NWICU_1yr'])
    R['INSPIRE_1yr'] = block('INSPIRE_1yr', a1, ins, ins['death_365'].values.astype(int),
                             ref['INSPIRE_1yr'])
    log("\n=== 院内死亡模型 ===")
    R['MIMIC_internal_inhosp'] = block('MIMIC_internal_inhosp', a2, mim.loc[Xte2.index], yte2,
                                       ref['MIMIC_internal_inhosp'])
    R['eICU_inhosp'] = block('eICU_inhosp', a2, e, e['death_hosp'].values.astype(int),
                             ref['eICU_inhosp'])
    Xl, yl = huaian_frame()
    R['Huaian_inhosp'] = block('Huaian_inhosp', a2, Xl, yl, ref['Huaian_inhosp'])

    frag = [(t, m) for t in R for m in ('Logistic', 'XGBoost') if R[t][m]['grade'] == 'FRAGILE']
    log("")
    log("  FRAGILE（|slope range over ε| > 0.01）组合 %d 个：%s" % (len(frag), frag))
    log("  η 直接拟合 vs ε=1e-6 的最大绝对差：%.4f" %
        max(abs(R[t][m]['slope_eta'] - R[t][m]['by_eps']['1e-06']['slope'])
            for t in R for m in ('Logistic', 'XGBoost')))

    json.dump({'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'eps_list': EPS_LIST,
               'scope_note': (
                   'Only the calibration slope and the logit-based calibration intercept pass '
                   'through a logit transform and therefore depend on the clipping constant. '
                   'The observed:expected ratio and the ECE are computed on the probability '
                   'scale and do not depend on epsilon.'),
               'results': R, 'fragile': frag},
              open(OUT / "slope_epsilon_audit_v9.json", 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    (OUT / "_93_slope_epsilon_v9.md").write_text(
        "# 93 —— V9 校准斜率 ε 敏感性（终态）\n\n```\n" + "\n".join(LOG) + "\n```\n",
        encoding='utf-8')
    log("\n→ output/slope_epsilon_audit_v9.json / _93_slope_epsilon_v9.md")


if __name__ == "__main__":
    main()
