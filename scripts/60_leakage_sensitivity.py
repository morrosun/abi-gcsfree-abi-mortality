# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
60_leakage_sensitivity.py —— 修 Decision E 第 D 条（诊断码时间泄漏）

编辑原话：
  "The diagnostic predictors, including hypoxic brain injury, come from discharge codes
   that may capture events occurring after the stated prediction time."

客观事实：MIMIC-IV 的 diagnoses_icd **没有 POA（present-on-admission）标记，也没有诊断时间戳**，
因此"某个病因码是在 ICU 第 1 天之后才发生的"这件事**在本数据里无法直接证伪**。
能做的只有把它"量化"——即把病因指标拿掉，看模型掉多少。掉得少 => 结论对泄漏不敏感。

本脚本量化四件事：
  Q1  MIMIC 内部：删掉 7 个病因哑变量，1 年模型的 AUC 掉多少？（ΔAUC + DeLong）
  Q2  MIMIC 内部：只删 anoxic（缺氧性脑损伤，OR 9.84 的最强因子），掉多少？
  Q3  泄漏最严重的场景是"院内死亡"——出院诊断码在结局之后才写下。因此做一个强度对照：
      同一个特征集在 1 年结局 vs 院内结局上，病因指标的贡献各是多少？
      若院内结局下病因贡献明显更大，与"码在结局之后"的泄漏机制一致（仍非证明）。
  Q4  更干净的结局：只看"活着出院的病人"的 1 年死亡（此时出院码必然早于结局）。
      在这批人里删病因掉多少？——外部库 NWICU / INSPIRE 同样可算，一并做。
  Q5  病因哑变量单独（7 个）与 anoxic 单独（1 个）的判别力：
      若 anoxic 单独就能接近全模型，说明模型确实"搭在诊断码上"。

产出：output/leakage_sensitivity.json（不覆盖任何已交付文件）
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score                    # noqa: E402
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

FEATSETS = {
    'FULL_28': FEATS,
    'NO_AET': [f for f in FEATS if f not in AET],
    'NO_ANOXIC': [f for f in FEATS if f != 'anoxic'],
    'AET_ONLY': AET,
    'ANOXIC_ONLY': ['anoxic'],
}

SEED = 42


def log(*a):
    print(*a)
    sys.stdout.flush()


# ---------------- DeLong (Sun & Xu 2014)，与 10_internal_revision.py 同一实现 ----------------
def _compute_midrank(x):
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
    pos = p[y == 1]
    neg = p[y == 0]
    m, n = len(pos), len(neg)
    tx = _compute_midrank(pos)
    ty = _compute_midrank(neg)
    tz = _compute_midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    s = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    return auc, s, v01, v10, m, n


def delong_test(y, p1, p2):
    y = np.asarray(y)
    a1, _, v01a, v10a, m, n = delong_var(y, np.asarray(p1))
    a2, _, v01b, v10b, _, _ = delong_var(y, np.asarray(p2))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    pval = 2 * (1 - stats.norm.cdf(abs(z)))
    return a1, a2, z, pval


def boot_auc_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    idx = np.arange(len(y))
    a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# --------------------------------------------------------------------------- #
def prep(df, cols, y, seed=SEED):
    X = df[cols].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    imp = [c for c in CONT if c in cols]
    med = Xtr[imp].median()
    for c in imp:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xte[c] = Xte[c].fillna(med[c])
    for c in cols:
        if c not in imp:
            Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').fillna(0)
            Xte[c] = pd.to_numeric(Xte[c], errors='coerce').fillna(0)
    sc = StandardScaler().fit(Xtr[imp]) if imp else None
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    if imp:
        Xtr_s[imp] = sc.transform(Xtr[imp])
        Xte_s[imp] = sc.transform(Xte[imp])
    return dict(Xtr=Xtr, Xte=Xte, Xtr_s=Xtr_s, Xte_s=Xte_s, ytr=np.asarray(ytr),
                yte=np.asarray(yte), cols=cols, imp=imp, med=med, sc=sc)


def fit_models(P):
    lr = LogisticRegression(max_iter=2000, C=1e6).fit(P['Xtr_s'][P['cols']], P['ytr'])
    xg_ = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric='logloss', random_state=SEED,
                            tree_method='hist', n_jobs=-1).fit(P['Xtr'][P['cols']], P['ytr'])
    return lr, xg_


def apply_models(lr, xg_, P, df, ycol):
    cols, imp = P['cols'], P['imp']
    X = df[cols].copy()
    for c in imp:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(P['med'][c])
    for c in cols:
        if c not in imp:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    Xs = X.copy()
    if imp:
        Xs[imp] = P['sc'].transform(X[imp])
    y = df[ycol].values.astype(int)
    return (lr.predict_proba(Xs[cols])[:, 1], xg_.predict_proba(X[cols])[:, 1], y)


# --------------------------------------------------------------------------- #
def run_block(tag, mim_df, cols, ycol, externals, survivor_only=False):
    """在一批数据上跑一个特征集；返回 dict。"""
    y = mim_df[ycol].values.astype(int)
    P = prep(mim_df, cols, y)
    lr, xg_ = fit_models(P)
    p_lr_te = lr.predict_proba(P['Xte_s'][P['cols']])[:, 1]
    p_xg_te = xg_.predict_proba(P['Xte'][P['cols']])[:, 1]
    res = {'internal': {'n': int(len(P['yte'])), 'events': int(P['yte'].sum())}}
    for nm, p in [('Logistic', p_lr_te), ('XGBoost', p_xg_te)]:
        res['internal'][nm] = {'auc': float(roc_auc_score(P['yte'], p)),
                               'ci': [float(v) for v in boot_auc_ci(P['yte'], p)]}
    res['_pred_internal'] = {'y': P['yte'], 'lr': p_lr_te, 'xg': p_xg_te}
    for db, ddf in externals:
        q = ddf[0].copy()
        if survivor_only and 'death_hosp' in q.columns:
            q = q[q['death_hosp'] == 0]
        yy = q[ycol].values.astype(int)
        if len(np.unique(yy)) < 2:
            res[db] = {'n': len(q), 'events': int(yy.sum()), 'skipped': 'single-class outcome'}
            continue
        plr, pxg, yv = apply_models(lr, xg_, P, q, ycol)
        res[db] = {'n': int(len(q)), 'events': int(yv.sum())}
        for nm, p in [('Logistic', plr), ('XGBoost', pxg)]:
            res[db][nm] = {'auc': float(roc_auc_score(yv, p)),
                           'ci': [float(v) for v in boot_auc_ci(yv, p)]}
    log(f"  [{tag}] {ycol:<11} internal n={res['internal']['n']:,} "
        f"LR={res['internal']['Logistic']['auc']:.3f} XGB={res['internal']['XGBoost']['auc']:.3f}")
    return res


def main():
    log("#" * 76)
    log("# 60_leakage_sensitivity.py | Decision-E item D: discharge-code temporal leakage")
    log("#" * 76)
    mim = pd.read_csv(DATA / "mimic_abi_cohort.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    nw = pd.read_csv(DATA / "nwicu_abi_final_vp2.csv")   # 74：恢复血管活性药
    ins = pd.read_csv(DATA / "inspire_abi_final.csv")
    dh = pd.read_csv(DATA / "inspire_deathhosp.csv")
    ins = ins.merge(dh, on='op_id', how='left')
    ins['death_hosp'] = ins['death_hosp'].fillna(0).astype(int)
    e = pd.read_csv(DATA / "eicu_abi_final.csv")

    hosp_surv = mim[mim['death_hosp'] == 0]
    log(f"MIMIC total n={len(mim):,}  1-yr deaths={int(mim['death_365'].sum()):,}  "
        f"in-hosp deaths={int(mim['death_hosp'].sum()):,}")
    log(f"MIMIC hospital survivors n={len(hosp_surv):,}  "
        f"1-yr deaths among them={int(hosp_surv['death_365'].sum()):,} "
        f"({100*hosp_surv['death_365'].mean():.1f}%)")

    R = {}

    # ---------------- 场景 A：1 年死亡（主结局，全人群） ----------------
    log("\n=== A. 1-year mortality, full MIMIC cohort (primary outcome) ===")
    EXT_365 = [('NWICU', [nw]), ('INSPIRE', [ins])]
    R['A_1yr_full'] = {}
    for fs, cols in FEATSETS.items():
        R['A_1yr_full'][fs] = run_block(f"A/{fs}", mim, cols, 'death_365', EXT_365)

    # ---------------- 场景 B：院内死亡（泄漏最严重：出院码在结局之后） ----------------
    log("\n=== B. In-hospital mortality (discharge codes written AFTER the outcome) ===")
    EXT_HOSP = [('eICU', [e]), ('NWICU', [nw]), ('INSPIRE', [ins])]
    R['B_inhosp'] = {}
    for fs, cols in FEATSETS.items():
        R['B_inhosp'][fs] = run_block(f"B/{fs}", mim, cols, 'death_hosp', EXT_HOSP)

    # ---------------- 场景 C：出院存活者的 1 年死亡（码必然早于结局） ----------------
    log("\n=== C. 1-year mortality among hospital survivors (codes precede the outcome) ===")
    EXT_SURV = [('NWICU', [nw]), ('INSPIRE', [ins])]
    R['C_survivors'] = {}
    for fs, cols in FEATSETS.items():
        R['C_survivors'][fs] = run_block(f"C/{fs}", hosp_surv, cols, 'death_365',
                                         EXT_SURV, survivor_only=True)

    # ---------------- DeLong：各特征集 vs FULL ----------------
    log("\n=== DeLong: feature set vs FULL (ΔAUC, paired on identical samples) ===")
    DEL = {}
    for scene in ['A_1yr_full', 'B_inhosp', 'C_survivors']:
        DEL[scene] = {}
        for fs in FEATSETS:
            if fs == 'FULL_28':
                continue
            DEL[scene][fs] = {}
            for site in ['internal', 'NWICU', 'INSPIRE', 'eICU']:
                pass
            # internal
            ye = R[scene]['FULL_28']['_pred_internal']['y']
            for nm, key in [('Logistic', 'lr'), ('XGBoost', 'xg')]:
                p1 = R[scene]['FULL_28']['_pred_internal'][key]
                p2 = R[scene][fs]['_pred_internal'][key]
                a1, a2, z, pv = delong_test(ye, p1, p2)
                DEL[scene][fs][f"internal_{nm}"] = {
                    'auc_full': float(a1), 'auc_reduced': float(a2),
                    'dAUC': float(a2 - a1), 'z': float(z), 'p': float(f'{pv:.4g}')}
                log(f"  {scene:<12} {fs:<12} internal {nm:<8} "
                    f"{a1:.3f} -> {a2:.3f}  Δ{a2-a1:+.4f}  DeLong p={pv:.3g}")

    # ---------------- 汇总表：病因指标的贡献 ----------------
    log("\n=== SUMMARY: cost of the discharge-code (aetiology) predictors ===")
    summary = {}
    for scene, label in [('A_1yr_full', '1-year death (all patients)'),
                         ('B_inhosp', 'in-hospital death (max leakage)'),
                         ('C_survivors', '1-year death among hospital survivors')]:
        f = R[scene]['FULL_28']['internal']
        n = R[scene]['NO_AET']['internal']
        nx = R[scene]['NO_ANOXIC']['internal']
        ao = R[scene]['AET_ONLY']['internal']
        ax = R[scene]['ANOXIC_ONLY']['internal']
        summary[scene] = {
            'outcome': label,
            'internal_n': f['n'], 'internal_events': f['events'],
            'Logistic': {'full': f['Logistic']['auc'], 'no_aet': n['Logistic']['auc'],
                         'delta_no_aet': n['Logistic']['auc'] - f['Logistic']['auc'],
                         'no_anoxic': nx['Logistic']['auc'],
                         'delta_no_anoxic': nx['Logistic']['auc'] - f['Logistic']['auc'],
                         'aet_only': ao['Logistic']['auc'], 'anoxic_only': ax['Logistic']['auc']},
            'XGBoost': {'full': f['XGBoost']['auc'], 'no_aet': n['XGBoost']['auc'],
                        'delta_no_aet': n['XGBoost']['auc'] - f['XGBoost']['auc'],
                        'no_anoxic': nx['XGBoost']['auc'],
                        'delta_no_anoxic': nx['XGBoost']['auc'] - f['XGBoost']['auc'],
                        'aet_only': ao['XGBoost']['auc'], 'anoxic_only': ax['XGBoost']['auc']},
        }
        for nm in ['Logistic', 'XGBoost']:
            s = summary[scene][nm]
            log(f"  {label:<44} {nm:<8} full {s['full']:.3f} | no-aet {s['no_aet']:.3f} "
                f"({s['delta_no_aet']:+.4f}) | no-anoxic {s['no_anoxic']:.3f} "
                f"({s['delta_no_anoxic']:+.4f}) | aet-only {s['aet_only']:.3f} | "
                f"anoxic-only {s['anoxic_only']:.3f}")

    # 外部：删病因后掉多少
    log("\n=== External cohorts: ΔAUC when aetiology dummies are removed ===")
    ext_tab = {}
    for scene, sites in [('A_1yr_full', ['NWICU', 'INSPIRE']),
                         ('B_inhosp', ['eICU', 'NWICU', 'INSPIRE']),
                         ('C_survivors', ['NWICU', 'INSPIRE'])]:
        for db in sites:
            if db not in R[scene]['FULL_28'] or 'XGBoost' not in R[scene]['FULL_28'][db]:
                continue
            row = {'n': R[scene]['FULL_28'][db]['n'], 'events': R[scene]['FULL_28'][db]['events']}
            for nm in ['Logistic', 'XGBoost']:
                a_full = R[scene]['FULL_28'][db][nm]['auc']
                a_no = R[scene]['NO_AET'][db][nm]['auc']
                row[nm] = {'full': a_full, 'no_aet': a_no, 'delta': a_no - a_full}
            ext_tab[f"{scene}|{db}"] = row
            log(f"  {scene:<12} {db:<9} n={row['n']:>6,} LR {row['Logistic']['full']:.3f}->"
                f"{row['Logistic']['no_aet']:.3f} ({row['Logistic']['delta']:+.4f})   "
                f"XGB {row['XGBoost']['full']:.3f}->{row['XGBoost']['no_aet']:.3f} "
                f"({row['XGBoost']['delta']:+.4f})")

    # ---------------- 落盘 ----------------
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != '_pred_internal'}
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    payload = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'note': ('MIMIC-IV diagnoses_icd carries no present-on-admission flag and no '
                        'diagnosis timestamp; leakage therefore cannot be disproven directly. '
                        'What is reported is the magnitude of the model dependence on the '
                        'discharge-coded aetiology indicators.'),
               'feature_sets': {k: len(v) for k, v in FEATSETS.items()},
               'scenarios': clean(R), 'delong': DEL, 'summary': summary, 'external': ext_tab}
    json.dump(payload, open(OUT / "leakage_sensitivity.json", 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    log(f"\nwrote output/leakage_sensitivity.json")

    # ---------------- 自检 ----------------
    log("\n" + "=" * 76)
    log("SELF-CHECK")
    ok = True
    a_lr = R['A_1yr_full']['FULL_28']['internal']['Logistic']['auc']
    a_xg = R['A_1yr_full']['FULL_28']['internal']['XGBoost']['auc']
    for nm, got, exp in [('Logistic', a_lr, 0.816), ('XGBoost', a_xg, 0.846)]:
        good = abs(got - exp) <= 0.006
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] FULL_28 1-yr internal {nm:<8} {got:.4f} "
            f"vs manuscript {exp:.3f} (tol 0.006)")
    # ★ 第六轮：3 位呈现必须与 Table 2/3/4/S4 一致（禁止双重舍入把 0.846 抬成 0.847）
    for nm, got, exp in [('Logistic', a_lr, '0.816'), ('XGBoost', a_xg, '0.846')]:
        shown = '%.3f' % got
        good = shown == exp
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] FULL_28 1-yr internal {nm:<8} "
            f"shown {shown} == Table 2 {exp}")
    b_lr = R['B_inhosp']['FULL_28']['eICU']['Logistic']['auc']
    b_xg = R['B_inhosp']['FULL_28']['eICU']['XGBoost']['auc']
    for nm, got, exp in [('Logistic', b_lr, 0.811), ('XGBoost', b_xg, 0.850)]:
        good = abs(got - exp) <= 0.006
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] FULL_28 in-hosp eICU  {nm:<8} {got:.4f} "
            f"vs Table 4 {exp:.3f} (tol 0.006)")
    n_full = len(FEATS)
    good = len(FEATSETS['NO_AET']) == n_full - 7 and len(FEATSETS['NO_ANOXIC']) == n_full - 1
    ok &= good
    log(f"  [{'OK ' if good else 'FAIL'}] feature-set sizes: FULL={n_full} "
        f"NO_AET={len(FEATSETS['NO_AET'])} NO_ANOXIC={len(FEATSETS['NO_ANOXIC'])}")
    log("=" * 76)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
