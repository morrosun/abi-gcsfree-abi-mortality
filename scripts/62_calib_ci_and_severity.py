# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
62_calib_ci_and_severity.py —— 修 Decision E 第 E 条（方法学细节缺失）

编辑原话：
  "These include the optimism correction, the definition of calibration error,
   uncertainty estimates for calibration measures, and comparison with established
   ICU severity scores."

本脚本补两项：
  PART A  校准指标的不确定性估计 —— 对每个队列 × 每个模型，bootstrap（B=1000，百分位法）
          给出 slope / intercept / O:E / ECE / Brier 的 95% CI。
          同时落盘"校准误差的明确定义"（ECE = 等宽 10 分箱、按箱内样本数加权的
          |观测率 − 平均预测概率| 之和），供 Methods 直接引用。
  PART B  与既有 ICU 严重度评分头对头 —— SOFA（本地已有 sofa_day1）、OASIS、SAPS-II
          （本次从 mimiciv_derived / mimiciv_hosp 新抽取），在**同一个 30% 内部测试集**上
          与全模型比较，配 DeLong 检验。
          ⚠ 三个外部库均不发布 SOFA/OASIS/SAPS-II 的构成变量，故头对头只能在 MIMIC 内部做。

OASIS 切点：Awad A et al. OASIS+. BMC Med Inform Decis Mak. 2021;21:126（Table 1，已逐条核对）
SAPS-II 切点：Le Gall JR et al. JAMA. 1993;270:2957-2963.

产出：output/calibration_ci_and_severity.json（新增，不覆盖任何已交付文件）
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
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split         # noqa: E402
from sklearn.preprocessing import StandardScaler             # noqa: E402
from scipy import stats                                      # noqa: E402
import xgboost as xgb                                        # noqa: E402

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
SEED = 42
B_BOOT = 1000


def log(*a):
    print(*a)
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
#  校准指标（定义与 07/59 完全一致）
# --------------------------------------------------------------------------- #
def logit_f(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_slope_intercept(y, p):
    lp = logit_f(p)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=2000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def oe_ratio(y, p):
    return float(np.mean(y) / np.mean(p))


def ece(y, p, bins=10):
    """ECE 定义：等宽 10 分箱；ECE = Σ_b (n_b / N) · |mean(y_b) − mean(p_b)|。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    n = len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * m.sum() / n
    return float(e)


def all_metrics(y, p):
    s, i = cal_slope_intercept(y, p)
    return {'auc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
            'slope': s, 'intercept': i, 'oe': oe_ratio(y, p), 'ece': ece(y, p)}


def boot_ci(y, p, B=B_BOOT, seed=42):
    """bootstrap 百分位 CI：slope / intercept / O:E / ECE / Brier / AUC。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    acc = {k: [] for k in ['auc', 'brier', 'slope', 'intercept', 'oe', 'ece']}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2 or len(np.unique(pb)) < 2:
            continue
        try:
            m = all_metrics(yb, pb)
        except Exception:                                     # noqa: BLE001
            continue
        for k in acc:
            acc[k].append(m[k])
    out = {}
    for k, v in acc.items():
        # ★ 第五轮终审（2026-10-01）：不再对 CI 上下限做 round(,4)。
        #   下游一律用 r3()/ci3() 呈现；先舍到 4 位、再舍到 3 位是一段双重舍入，
        #   会让 0.7595 这类边界值多进一位，使正文数字无法从全精度值复算。
        #   真值只应有一个精度：全精度。呈现层（3 位/4 位）交给下游脚本。
        out[k] = {'n_boot': len(v),
                  'ci': [float(np.percentile(v, 2.5)),
                         float(np.percentile(v, 97.5))]}
    return out


# --------------------------------------------------------------------------- #
#  DeLong（与 10 / 60 同一实现）
# --------------------------------------------------------------------------- #
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_var(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return auc, v01, v10, m, n


def delong_test(y, p1, p2):
    y = np.asarray(y)
    a1, v01a, v10a, m, n = delong_var(y, np.asarray(p1))
    a2, v01b, v10b, _, _ = delong_var(y, np.asarray(p2))
    S = np.cov(np.vstack([v01a, v01b])) / m + np.cov(np.vstack([v10a, v10b])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return a1, a2, float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


# =========================================================================== #
#  PART A
# =========================================================================== #
def part_a():
    log("=" * 78)
    log("PART A | 校准指标的 bootstrap 95%% CI (B=%d, 百分位法)" % B_BOOT)
    log("   ECE 定义 = Σ_b (n_b/N)·|mean(y_b) − mean(p_b)|，等宽 10 分箱")
    mim = pd.read_csv(DATA / "mimic_abi_cohort.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    nw = pd.read_csv(DATA / "nwicu_abi_final_vp2.csv")   # 74：恢复血管活性药
    ins = pd.read_csv(DATA / "inspire_abi_final.csv")
    dh = pd.read_csv(DATA / "inspire_deathhosp.csv")
    ins = ins.merge(dh, on='op_id', how='left')
    ins['death_hosp'] = ins['death_hosp'].fillna(0).astype(int)
    e = pd.read_csv(DATA / "eicu_abi_final.csv")

    # ---- MIMIC 内部测试集（重建 1 年模型，口径与 03/10/60 一致） ----
    X = mim[FEATS].copy()
    y = mim['death_365'].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    med = Xtr[CONT].median()
    for c in CONT:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xte[c] = Xte[c].fillna(med[c])
    for c in FEATS:
        if c not in CONT:
            Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').fillna(0)
            Xte[c] = pd.to_numeric(Xte[c], errors='coerce').fillna(0)
    sc = StandardScaler().fit(Xtr[CONT])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[CONT] = sc.transform(Xtr[CONT])
    Xte_s[CONT] = sc.transform(Xte[CONT])
    lr = LogisticRegression(max_iter=2000, C=1e6).fit(Xtr_s[FEATS], ytr)
    xg_ = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric='logloss', random_state=SEED,
                            tree_method='hist', n_jobs=-1).fit(Xtr[FEATS], ytr)
    PRED = {'MIMIC_internal_1yr': {
        'y': yte,
        'Logistic': lr.predict_proba(Xte_s[FEATS])[:, 1],
        'XGBoost': xg_.predict_proba(Xte[FEATS])[:, 1]}}

    # ---- 冻结 1 年模型 → NWICU / INSPIRE（1 年） ----
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
        for c in AET + BINV:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0).astype(int)
        Zs = Z.copy()
        Zs[cont] = scaler.transform(Z[cont])
        return (1 / (1 + np.exp(-(logit['const'] + Zs[feats].values @ logit[feats].values))),
                xgbm.predict_proba(Z[feats].values)[:, 1])

    for tag, df, yc in [('NWICU_1yr', nw, 'death_365'), ('INSPIRE_1yr', ins, 'death_365')]:
        a, b = p1(df)
        PRED[tag] = {'y': df[yc].values.astype(int), 'Logistic': a, 'XGBoost': b}

    # ---- 冻结院内平行模型 → eICU（院内）+ 淮安单中心（院内） ----
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

    a, b = p2(e)
    PRED['eICU_inhosp'] = {'y': e['death_hosp'].values.astype(int), 'Logistic': a, 'XGBoost': b}

    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    if P.exists():
        raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
        raw = raw.dropna(how='all').reset_index(drop=True)
        num = lambda c: pd.to_numeric(raw[c], errors='coerce')      # noqa: E731
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
    for tag, d in PRED.items():
        RES[tag] = {'n': int(len(d['y'])), 'events': int(d['y'].sum()),
                    'prevalence_pct': round(100 * float(np.mean(d['y'])), 1)}
        for mname in ['Logistic', 'XGBoost']:
            pt = d[mname]
            m0 = all_metrics(d['y'], pt)
            ci = boot_ci(d['y'], pt)
            RES[tag][mname] = {
                # ★ 第五轮终审（2026-10-01）：point 值不再 round(,4)。
                #   本 JSON 是 64/65 的数字真源，存舍入值 = 下游再做一次舍入（双重舍入）。
                #   实测因此把 NWICU logistic slope 0.0394810 显示成 0.040（应为 0.039）、
                #   eICU logistic slope 0.6745356 显示成 0.674（应为 0.675）、
                #   MIMIC XGBoost AUC 0.8464649 显示成 0.847（应为 0.846）。
                'point': {k: float(m0[k]) for k in m0},
                'ci95': {k: ci[k]['ci'] for k in ci},
                'n_boot': ci['auc']['n_boot']}
            log(f"  {tag:<20} {mname:<8} n={RES[tag]['n']:>6,} slope {m0['slope']:.3f} "
                f"[{ci['slope']['ci'][0]:.3f},{ci['slope']['ci'][1]:.3f}]  "
                f"O:E {m0['oe']:.3f} [{ci['oe']['ci'][0]:.3f},{ci['oe']['ci'][1]:.3f}]  "
                f"ECE {m0['ece']:.4f} [{ci['ece']['ci'][0]:.4f},{ci['ece']['ci'][1]:.4f}]  "
                f"Brier {m0['brier']:.3f} [{ci['brier']['ci'][0]:.3f},{ci['brier']['ci'][1]:.3f}]")
    return RES, PRED


# =========================================================================== #
#  PART B
# =========================================================================== #
def oasis(df):
    """OASIS（OASIS+ 原文 Table 1，BMC Med Inform Decis Mak 2021;21:126）"""
    s = np.zeros(len(df))
    age = df['age'].values
    s += np.select([age < 24, age <= 53, age <= 77, age <= 89], [0, 3, 6, 9], default=7)
    g = df['gcs_min'].fillna(15).values
    s += np.select([g <= 7, g <= 13, g <= 14], [10, 4, 3], default=0)
    hr = df['heart_rate_mean'].values
    s += np.select([hr < 33, hr <= 88, hr <= 106, hr <= 125], [0, 0, 1, 3], default=6)
    mbp = df['mbp_mean'].values
    s += np.select([mbp < 20.65, mbp <= 50.99, mbp <= 61.32, mbp <= 143.44],
                   [4, 3, 2, 0], default=3)
    rr = df['resp_rate_mean'].values
    s += np.select([rr < 6, rr <= 12, rr <= 22, rr <= 30, rr <= 44], [10, 1, 0, 1, 6], default=9)
    t = df['temperature_mean'].values
    s += np.select([t < 33.22, t <= 35.93, t <= 36.39, t <= 36.88, t <= 39.88],
                   [3, 4, 2, 0, 2], default=6)
    uo = df['urineoutput'].fillna(df['urineoutput'].median()).values
    s += np.select([uo < 671, uo <= 1426.99, uo <= 2543.99, uo <= 6896], [10, 5, 1, 0], default=8)
    los = df['pre_icu_los_hours'].values
    s += np.select([los < 0.17, los <= 4.94, los <= 24.0, los <= 311.80], [5, 3, 0, 2], default=1)
    s += np.where(df['mech_vent'].values.astype(float) > 0, 9, 0)
    elective = df['admission_type'].astype(str).str.upper().eq('ELECTIVE').values
    s += np.where(elective, 0, 6)
    return s


def saps2(df):
    """SAPS II（Le Gall 1993）。缺失项按"正常"计 0 分；HR 取首日 min/max 中赋分高者。"""
    s = np.zeros(len(df))
    a = df['age'].values
    s += np.select([a < 40, a < 60, a < 70, a < 75, a < 80], [0, 7, 12, 15, 16], default=18)
    hr_min = df['heart_rate_min'].fillna(80).values
    hr_max = df['heart_rate_max'].fillna(80).values

    def hr_pts(h):
        return np.select([h < 40, h < 70, h < 120, h < 160], [11, 2, 0, 4], default=7)
    s += np.maximum(hr_pts(hr_min), hr_pts(hr_max))
    sbp = df['sbp_min'].fillna(120).values
    s += np.select([sbp < 70, sbp < 100, sbp < 200], [13, 5, 0], default=2)
    t = df['temperature_max'].fillna(37).values
    s += np.where(t >= 39.0, 3, 0)
    g = df['gcs_min'].fillna(15).values
    s += np.select([g < 6, g < 9, g < 11, g < 14], [26, 13, 7, 5], default=0)
    # PaO2/FiO2 仅对机械通气者计分
    pf = df['pao2fio2ratio_min'].values
    vent = df['mech_vent'].values.astype(float) > 0
    pf_pts = np.select([pf < 100, pf < 200], [11, 9], default=6)
    s += np.where(vent & ~np.isnan(pf), pf_pts, 0)
    uo = df['urineoutput'].fillna(df['urineoutput'].median()).values
    s += np.select([uo < 500, uo < 1000], [11, 4], default=0)
    bun = df['bun_max'].fillna(15).values
    s += np.select([bun < 28, bun < 84], [0, 6], default=10)
    wbc = df['wbc_max'].fillna(10).values
    s += np.select([wbc < 1.0, wbc < 20.0], [12, 0], default=3)
    k = df['potassium_max'].fillna(4).values
    s += np.select([k < 3.0, k < 5.0], [3, 0], default=3)
    na = df['sodium_min'].fillna(140).values
    s += np.select([na < 125, na < 145], [5, 0], default=1)
    hco3 = df['bicarbonate_min'].fillna(24).values
    s += np.select([hco3 < 15, hco3 < 20], [6, 3], default=0)
    bili = df['bilirubin_total_max'].fillna(0.5).values   # 未查者按正常计 0 分
    s += np.select([bili < 4.0, bili < 6.0], [0, 4], default=9)
    s += df['aids'].values * 17 + df['mets'].values * 9 + df['hem_malign'].values * 10
    elective = df['admission_type'].astype(str).str.upper().eq('ELECTIVE').values
    surgical = df['n_proc'].values > 0
    s += np.select([elective & surgical, surgical], [0, 8], default=6)
    return s


def part_b(pred_internal):
    log("\n" + "=" * 78)
    log("PART B | 与既有 ICU 严重度评分头对头（MIMIC-IV 内部 30% 测试集）")
    mim = pd.read_csv(DATA / "mimic_abi_cohort.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    sv = pd.read_csv(DATA / "mimic_severity_inputs.csv",
                     usecols=['stay_id', 'gcs_min', 'urineoutput', 'pao2fio2ratio_min',
                              'heart_rate_min', 'heart_rate_max', 'sbp_min', 'temperature_max',
                              'bilirubin_total_max', 'aids', 'mets', 'hem_malign', 'n_proc',
                              'pre_icu_los_hours'])
    mim = mim.merge(sv, on='stay_id', how='left')
    # 与 PART A 完全同一个切分
    idx = np.arange(len(mim))
    _, te_idx = train_test_split(idx, test_size=0.30, stratify=mim['death_365'].values,
                                 random_state=SEED)
    te = mim.iloc[te_idx].copy()

    te['sofa'] = te['sofa_day1'].fillna(te['sofa_day1'].median())
    te['oasis'] = oasis(te)
    te['saps2'] = saps2(te)
    yte = te['death_365'].values.astype(int)

    p_lr, p_xg = pred_internal['Logistic'], pred_internal['XGBoost']
    assert len(p_lr) == len(yte), f"PART A/B test-set mismatch {len(p_lr)} vs {len(yte)}"

    RES = {'n': int(len(te)), 'events': int(yte.sum()),
           'score_descriptives': {k: {'median': round(float(te[k].median()), 2),
                                      'IQR': [round(float(te[k].quantile(.25)), 2),
                                              round(float(te[k].quantile(.75)), 2)],
                                      'min': round(float(te[k].min()), 2),
                                      'max': round(float(te[k].max()), 2)}
                                  for k in ['sofa', 'oasis', 'saps2']},
           'comparison': {}}
    for k in ['sofa', 'oasis', 'saps2']:
        # ★ 第五轮终审：全精度（下游 r3 呈现），避免双重舍入
        RES['score_descriptives'][k]['auc_alone'] = float(roc_auc_score(yte, te[k]))
    log(f"  test set n={len(te):,} events={yte.sum():,}")
    for k in ['sofa', 'oasis', 'saps2']:
        sd = RES['score_descriptives'][k]
        log(f"    {k:<6} median {sd['median']:>6.1f} IQR {sd['IQR']} range "
            f"[{sd['min']},{sd['max']}]  AUC-alone {sd['auc_alone']:.3f}")

    log("\n  DeLong: 全模型 vs 严重度评分（同一批样本）")
    for score in ['sofa', 'oasis', 'saps2']:
        RES['comparison'][score] = {}
        for mname, pfull in [('Logistic', p_lr), ('XGBoost', p_xg)]:
            a1, a2, z, pv = delong_test(yte, pfull, te[score].values)
            RES['comparison'][score][mname] = {
                # ★ 第五轮终审：全精度（下游 r3/sg/r2 呈现），避免双重舍入
                'auc_model': float(a1), 'auc_score': float(a2),
                'dAUC': float(a1 - a2), 'z': float(z), 'p': float(f'{pv:.4g}')}
            log(f"    {score:<6} vs {mname:<8} model {a1:.3f} | score {a2:.3f} | "
                f"Δ{a1-a2:+.4f} | DeLong p={pv:.3g}")

    # 严重度评分的校准（供参考）
    for score in ['sofa', 'oasis', 'saps2']:
        lp = np.log(np.clip(te[score].values, 1e-6, None) + 1.0)
        m = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), yte)
        p = m.predict_proba(lp.reshape(-1, 1))[:, 1]
        mt = all_metrics(yte, p)
        RES['comparison'][score]['logistic_on_score'] = {
            # ★ 第五轮终审：全精度，避免双重舍入
            k: float(v) for k, v in mt.items()}
    return RES


# =========================================================================== #
def main():
    log("#" * 78)
    log("# 62_calib_ci_and_severity.py | Decision-E item E: calibration uncertainty +")
    log("#                                head-to-head vs SOFA / OASIS / SAPS-II")
    log("#" * 78)
    A, PRED = part_a()
    B = part_b(PRED['MIMIC_internal_1yr'])

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ece_definition': ('ECE = sum over 10 equal-width probability bins of '
                           '(n_b/N) * |mean(observed in bin b) - mean(predicted in bin b)|'),
        'bootstrap': {'B': B_BOOT, 'method': 'percentile', 'seed': 42,
                      'resampled': 'patient-level, with replacement'},
        'calibration_ci': A,
        'severity_headtohead': B,
        'external_scores_note': ('None of eICU, NWICU or INSPIRE release the components of '
                                 'SOFA / OASIS / SAPS-II, so the head-to-head comparison can '
                                 'only be performed inside MIMIC-IV.'),
    }
    json.dump(payload, open(OUT / "calibration_ci_and_severity.json", 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    log(f"\nwrote output/calibration_ci_and_severity.json")

    # ---------------- 自检 ----------------
    log("\n" + "=" * 78)
    log("SELF-CHECK")
    ok = True
    mi = A['MIMIC_internal_1yr']
    for mname, exp in [('Logistic', 0.816), ('XGBoost', 0.846)]:
        got = mi[mname]['point']['auc']
        good = abs(got - exp) <= 0.006
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] MIMIC internal 1-yr {mname:<8} {got:.4f} "
            f"vs Table 2 {exp:.3f}")
    for tag, mname, exp in [('NWICU_1yr', 'XGBoost', 0.796), ('INSPIRE_1yr', 'XGBoost', 0.759),
                            ('eICU_inhosp', 'Logistic', 0.811), ('eICU_inhosp', 'XGBoost', 0.850),
                            ('Huaian_inhosp', 'Logistic', 0.865), ('Huaian_inhosp', 'XGBoost', 0.800)]:
        got = A[tag][mname]['point']['auc']
        good = abs(got - exp) <= 0.006
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] {tag:<20} {mname:<8} {got:.4f} vs Table 4/5 {exp:.3f}")
    # SOFA alone 应复现稿件 0.702
    got = B['score_descriptives']['sofa']['auc_alone']
    good = abs(got - 0.702) <= 0.01
    ok &= good
    log(f"  [{'OK ' if good else 'FAIL'}] SOFA alone AUC {got:.4f} vs manuscript 0.702 (tol 0.01)")
    # bootstrap 次数
    nb = A['MIMIC_internal_1yr']['Logistic']['n_boot']
    good = nb >= 900
    ok &= good
    log(f"  [{'OK ' if good else 'FAIL'}] bootstrap completed reps = {nb} (>=900)")
    # 严重度评分分布合理
    for k, lo, hi in [('oasis', 0, 90), ('saps2', 0, 120)]:
        med = B['score_descriptives'][k]['median']
        good = lo <= med <= hi
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] {k:<6} median {med} within [{lo},{hi}]")
    log("=" * 78)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
