# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
91_leakage_sensitivity_v9.py —— V9 重算：出院诊断码时间泄漏的敏感性分析

评审第 7 条要求：把"泄漏上限（bounding leakage）"改为"对出院编码预测变量的依赖分析
（dependence on discharge-coded predictors）"。本脚本量化的是**依赖程度**，不是泄漏量。

客观边界（不可绕过）：MIMIC-IV 的 diagnoses_icd 既无 POA 标记、也无诊断时间戳，
"某个病因码是否在预测时点之后写下"在本数据里无法证伪。因此本脚本的比较结论只能表述为
"模型性能对这 7 个病因哑变量的依赖程度"，不能写成"时间泄漏造成的偏倚上限"。

设计（沿用 60 的五个特征集 × 三个场景）：
  FULL_28     28 特征全模型
  NO_AET      去掉 7 个病因哑变量              ← 主要依赖度指标
  NO_ANOXIC   只去掉 anoxic（最强单因子）
  AET_ONLY    只保留 7 个病因哑变量
  ANOXIC_ONLY 只保留 anoxic
  A 1 年死亡（全人群）  B 院内死亡（出院码必然在结局之后，泄漏最严重）
  C 出院存活者的 1 年死亡（出院码必然早于结局，最干净）

★ 对账：场景 A 的 FULL_28 内部 AUC 必须等于 87 的主结果（0.8176 / 0.8445），
  场景 B 的 FULL_28 内部 AUC 必须等于 87 的院内模型（0.8474 / 0.8772）。

产出：output/leakage_sensitivity_v9.json + _91_leakage_sensitivity.md
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
    # ★ P1（2026-10-07 第二轮评审 §4.2-21）：Charlson 与 7 个病因哑变量同属"出院编码派生、
    #   预测时点不可验证"的一组。此前只测了"去掉 7 个病因"，把 Charlson 留在模型里 ——
    #   若读者据此认为时点不可验证的只占 7/28，是低估。此场景把两者**一起**删除
    #   （28 → 20 个预测因子），给出联合删除的代价。
    'NO_AET_CHARLSON': [f for f in FEATS if f not in AET
                        and f != 'charlson_comorbidity_index'],
    'NO_ANOXIC': [f for f in FEATS if f != 'anoxic'],
    'AET_ONLY': AET,
    'ANOXIC_ONLY': ['anoxic'],
}
SEED = 42

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


# ---------------------------------------------------------------- DeLong
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
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    return auc, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m, m, n


def delong_test(y, p1, p2):
    a1, v01a, v10a, m, n = delong_var(y, np.asarray(p1))
    a2, v01b, v10b, _, _ = delong_var(y, np.asarray(p2))
    S = np.cov(np.vstack([v01a, v01b])) / m + np.cov(np.vstack([v10a, v10b])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1), float(a2), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def boot_auc_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def prep(df, cols, y, seed=SEED):
    X = df[cols].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    imp = [c for c in CONT if c in cols]
    med = Xtr[imp].median() if imp else None
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


def run_block(tag, mim_df, cols, ycol, externals, survivor_only=False):
    y = mim_df[ycol].values.astype(int)
    P = prep(mim_df, cols, y)
    lr, xg_ = fit_models(P)
    p_lr_te = lr.predict_proba(P['Xte_s'][P['cols']])[:, 1]
    p_xg_te = xg_.predict_proba(P['Xte'][P['cols']])[:, 1]
    res = {'internal': {'n': int(len(P['yte'])), 'events': int(P['yte'].sum())}}
    for nm, p in [('Logistic', p_lr_te), ('XGBoost', p_xg_te)]:
        res['internal'][nm] = {'auc': float(roc_auc_score(P['yte'], p)),
                               'ci': boot_auc_ci(P['yte'], p)}
    res['_pred_internal'] = {'y': P['yte'], 'lr': p_lr_te, 'xg': p_xg_te}
    for db, ddf in externals:
        q = ddf.copy()
        if survivor_only and 'death_hosp' in q.columns:
            q = q[q['death_hosp'] == 0]
        yy = q[ycol].values.astype(int)
        if len(np.unique(yy)) < 2:
            res[db] = {'n': int(len(q)), 'events': int(yy.sum()),
                       'skipped': 'single-class outcome'}
            continue
        plr, pxg, yv = apply_models(lr, xg_, P, q, ycol)
        res[db] = {'n': int(len(q)), 'events': int(yv.sum())}
        for nm, p in [('Logistic', plr), ('XGBoost', pxg)]:
            res[db][nm] = {'auc': float(roc_auc_score(yv, p)), 'ci': boot_auc_ci(yv, p)}
    log(f"  [{tag:<22}] internal n={res['internal']['n']:,}  "
        f"LR={res['internal']['Logistic']['auc']:.4f}  "
        f"XGB={res['internal']['XGBoost']['auc']:.4f}")
    return res


def main():
    log("#" * 78)
    log("# 91_leakage_sensitivity_v9.py | 对出院编码预测变量的依赖度分析（非泄漏上限）")
    log("#" * 78)
    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    nw = pd.read_csv(DATA / "nwicu_abi_final_v5.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v3.csv")
    e = pd.read_csv(DATA / "eicu_abi_final_v3.csv")
    log(f"  MIMIC n={len(mim):,}  NWICU n={len(nw):,}  INSPIRE n={len(ins):,}  eICU n={len(e):,}")

    ref = json.load(open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))
    hosp_surv = mim[mim['death_hosp'] == 0]
    log(f"  MIMIC 1 年死亡={int(mim.death_365.sum()):,}  院内死亡={int(mim.death_hosp.sum()):,}  "
        f"出院存活者 n={len(hosp_surv):,}（其中 1 年死亡 {int(hosp_surv.death_365.sum()):,}）")

    R = {}
    # ---------------- A：1 年死亡（主结局，全人群） ----------------
    log("\n=== A. 1-year mortality, full cohort (primary outcome) ===")
    R['A_1yr_full'] = {}
    for fs, cols in FEATSETS.items():
        R['A_1yr_full'][fs] = run_block(f"A/{fs}", mim, cols, 'death_365',
                                        [('NWICU', nw), ('INSPIRE', ins)])
    # ---------------- B：院内死亡（出院码在结局之后 → 依赖最强） ----------------
    log("\n=== B. In-hospital mortality (discharge codes written AFTER the outcome) ===")
    R['B_inhosp'] = {}
    for fs, cols in FEATSETS.items():
        R['B_inhosp'][fs] = run_block(f"B/{fs}", mim, cols, 'death_hosp',
                                      [('eICU', e), ('NWICU', nw), ('INSPIRE', ins)])
    # ---------------- C：出院存活者的 1 年死亡（码必然早于结局 → 最干净） ----------------
    log("\n=== C. 1-year mortality among hospital survivors (codes precede the outcome) ===")
    R['C_survivors'] = {}
    for fs, cols in FEATSETS.items():
        R['C_survivors'][fs] = run_block(f"C/{fs}", hosp_surv, cols, 'death_365',
                                         [('NWICU', nw), ('INSPIRE', ins)], survivor_only=True)

    # ★ 对账：FULL_28 必须复现 87 的主结果
    for scene, refkey in [('A_1yr_full', 'MIMIC_internal_1yr'),
                          ('B_inhosp', 'MIMIC_internal_inhosp')]:
        for nm in ['Logistic', 'XGBoost']:
            got = R[scene]['FULL_28']['internal'][nm]['auc']
            want = ref[refkey][nm]['point']['auc']
            assert abs(got - want) < 1e-9, \
                f"RECONCILE FAIL: {scene}/{nm} {got!r} != 87 的 {want!r}"
            log(f"\n  [reconcile OK] {scene}/{nm:<8} AUC {got:.6f} == 87 的 {want:.6f}")

    # ---------------- DeLong：各特征集 vs FULL ----------------
    log("\n=== DeLong: reduced feature set vs FULL (paired, identical samples) ===")
    DEL = {}
    for scene in ['A_1yr_full', 'B_inhosp', 'C_survivors']:
        DEL[scene] = {}
        for fs in FEATSETS:
            if fs == 'FULL_28':
                continue
            DEL[scene][fs] = {}
            ye = R[scene]['FULL_28']['_pred_internal']['y']
            for nm, key in [('Logistic', 'lr'), ('XGBoost', 'xg')]:
                p1 = R[scene]['FULL_28']['_pred_internal'][key]
                p2 = R[scene][fs]['_pred_internal'][key]
                a1, a2, z, pv = delong_test(ye, p1, p2)
                DEL[scene][fs][f"internal_{nm}"] = {'auc_full': a1, 'auc_reduced': a2,
                                                    'dAUC': a2 - a1, 'z': z,
                                                    'p': float(f"{pv:.4g}")}
                log(f"  {scene:<12} {fs:<12} {nm:<8} {a1:.4f} -> {a2:.4f}  "
                    f"Δ{a2 - a1:+.4f}  DeLong p={pv:.3g}")

    # ---------------- 汇总 ----------------
    log("\n=== SUMMARY: dependence on the discharge-coded (aetiology) predictors ===")
    summary = {}
    for scene, label in [('A_1yr_full', '1-year death (all patients)'),
                         ('B_inhosp', 'in-hospital death (codes after outcome)'),
                         ('C_survivors', '1-year death among hospital survivors')]:
        f = R[scene]['FULL_28']['internal']
        n = R[scene]['NO_AET']['internal']
        nc = R[scene]['NO_AET_CHARLSON']['internal']
        nx = R[scene]['NO_ANOXIC']['internal']
        ao = R[scene]['AET_ONLY']['internal']
        ax = R[scene]['ANOXIC_ONLY']['internal']
        summary[scene] = {'outcome': label, 'internal_n': f['n'],
                          'internal_events': f['events']}
        for nm in ['Logistic', 'XGBoost']:
            summary[scene][nm] = {
                'full': f[nm]['auc'], 'no_aet': n[nm]['auc'],
                'delta_no_aet': n[nm]['auc'] - f[nm]['auc'],
                # ★ P1：7 病因 + Charlson 联合删除（20 个预测因子）
                'no_aet_charlson': nc[nm]['auc'],
                'delta_no_aet_charlson': nc[nm]['auc'] - f[nm]['auc'],
                'no_anoxic': nx[nm]['auc'],
                'delta_no_anoxic': nx[nm]['auc'] - f[nm]['auc'],
                'aet_only': ao[nm]['auc'], 'anoxic_only': ax[nm]['auc']}
            s = summary[scene][nm]
            log(f"  {label:<44} {nm:<8} full {s['full']:.4f} | no-aet {s['no_aet']:.4f} "
                f"({s['delta_no_aet']:+.4f}) | no-aet+charlson "
                f"{s['no_aet_charlson']:.4f} ({s['delta_no_aet_charlson']:+.4f}) | "
                f"no-anoxic {s['no_anoxic']:.4f} "
                f"({s['delta_no_anoxic']:+.4f}) | aet-only {s['aet_only']:.4f} | "
                f"anoxic-only {s['anoxic_only']:.4f}")

    # 外部：去掉病因后掉多少
    log("\n=== External cohorts: ΔAUC when aetiology dummies are removed ===")
    ext_tab = {}
    for scene, sites in [('A_1yr_full', ['NWICU', 'INSPIRE']),
                         ('B_inhosp', ['eICU', 'NWICU', 'INSPIRE']),
                         ('C_survivors', ['NWICU', 'INSPIRE'])]:
        for db in sites:
            if db not in R[scene]['FULL_28'] or 'XGBoost' not in R[scene]['FULL_28'][db]:
                continue
            row = {'n': R[scene]['FULL_28'][db]['n'],
                   'events': R[scene]['FULL_28'][db]['events']}
            for nm in ['Logistic', 'XGBoost']:
                a_full = R[scene]['FULL_28'][db][nm]['auc']
                a_no = R[scene]['NO_AET'][db][nm]['auc']
                a_nc = R[scene]['NO_AET_CHARLSON'][db][nm]['auc']
                row[nm] = {'full': a_full, 'no_aet': a_no, 'delta': a_no - a_full,
                           'no_aet_charlson': a_nc, 'delta_no_aet_charlson': a_nc - a_full}
            ext_tab[f"{scene}|{db}"] = row
            log(f"  {scene:<12} {db:<9} n={row['n']:>6,}  LR {row['Logistic']['full']:.4f}->"
                f"{row['Logistic']['no_aet']:.4f} ({row['Logistic']['delta']:+.4f})"
                f"[+charlson {row['Logistic']['no_aet_charlson']:.4f} "
                f"({row['Logistic']['delta_no_aet_charlson']:+.4f})]   "
                f"XGB {row['XGBoost']['full']:.4f}->{row['XGBoost']['no_aet']:.4f} "
                f"({row['XGBoost']['delta']:+.4f})"
                f"[+charlson {row['XGBoost']['no_aet_charlson']:.4f} "
                f"({row['XGBoost']['delta_no_aet_charlson']:+.4f})]")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != '_pred_internal'}
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o

    payload = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
               'interpretation_bound': (
                   'MIMIC-IV diagnoses_icd carries no present-on-admission flag and no '
                   'diagnosis timestamp; the timing of the aetiology codes relative to the '
                   'prediction time cannot be verified. What is quantified here is the '
                   'dependence of model performance on discharge-coded predictors, NOT an '
                   'upper bound on bias attributable to temporal leakage.'),
               'datasets': {'MIMIC': 'mimic_abi_cohort_v2.csv', 'NWICU': 'nwicu_abi_final_v5.csv',
                            'INSPIRE': 'inspire_abi_final_v3.csv', 'eICU': 'eicu_abi_final_v3.csv'},
               'feature_sets': {k: len(v) for k, v in FEATSETS.items()},
               'results': clean(R), 'delong': DEL, 'summary': summary,
               'external_delta': ext_tab}
    json.dump(payload, open(OUT / "leakage_sensitivity_v9.json", 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    (OUT / "_91_leakage_sensitivity.md").write_text(
        "# 91 —— V9 泄漏敏感性（对出院编码预测变量的依赖度）\n\n```\n" + "\n".join(LOG) + "\n```\n",
        encoding='utf-8')
    log("\n→ output/leakage_sensitivity_v9.json / _91_leakage_sensitivity.md")


if __name__ == "__main__":
    main()
