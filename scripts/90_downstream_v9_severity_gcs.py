# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
90_downstream_v9_severity_gcs.py —— V9 下游重算（第 1 批 / 下游）

数据层已于 86（重建）+ 89（哨兵值清洗）修正，模型层于 87 重训。
本脚本在**同一套 v2/v3/v4 数据 + 同一冻结模型**上重跑两项下游分析：

  PART 1  与既有 ICU 严重度评分头对头（SOFA / OASIS / SAPS-II，MIMIC-IV 内部 30% 测试集）
           - OASIS / SAPS-II 取 MIMIC-IV 官方派生表（MIT-LCP mimic-code），SOFA 取 first_day_sofa
           - 配 DeLong 检验；附各评分单独 AUC、分布、以及"把评分当单变量做 logistic"的校准
  PART 2  GCS 增量（base / +总分 / +运动项），配 cNRI / IDI / DeLong
           - ★ 可比性硬断言：三个变体必须同一样本、同一划分、同一插补与标度
  PART 3  单中心（淮安）审计：病因构成、单预测因子单独 AUC、冻结模型 AUC

★ 口径对账（防漂移）：本脚本从 87 产出的 artefacts 重建内部测试集预测，
  并断言 AUC 与 calibration_ci_and_severity_v2.json 完全一致（容差 1e-9）。
  不一致即中止 —— 绝不允许"下游数字"与"主结果数字"来自两套口径。

产出：output/downstream_severity_gcs_v9.json + _90_downstream_severity_gcs.md
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from scipy import stats                                        # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.metrics import roc_auc_score, brier_score_loss    # noqa: E402
from sklearn.model_selection import train_test_split           # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402

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
SEED = 42
EPS = 1e-6

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


def delong_test(y, p_new, p_old):
    a1, v01a, v10a, m, n = delong_var(y, np.asarray(p_new))
    a2, v01b, v10b, _, _ = delong_var(y, np.asarray(p_old))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1), float(a2), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def boot_ci(y, p, n=1000, seed=7):
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


def nri_idi(y, p_old, p_new):
    y = np.asarray(y)
    p_old, p_new = np.asarray(p_old), np.asarray(p_new)
    ev, ne = y == 1, y == 0
    nri_e = np.mean(p_new[ev] > p_old[ev]) - np.mean(p_new[ev] < p_old[ev])
    nri_ne = np.mean(p_new[ne] < p_old[ne]) - np.mean(p_new[ne] > p_old[ne])
    idi = (p_new[ev].mean() - p_old[ev].mean()) - (p_new[ne].mean() - p_old[ne].mean())
    return float(nri_e + nri_ne), float(nri_e), float(nri_ne), float(idi)


def cal_slope_intercept(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    lp = np.log(p / (1 - p))
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs',
                           max_iter=2000).fit(lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def ece(y, p, bins=10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(e)


# ---------------------------------------------------------------- 对账
def load_and_reconcile():
    """从 87 的 artefacts 重建内部测试集预测，并与 87 的 JSON 对账。"""
    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    ref = json.load(open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))

    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    X = mim[FEATS].copy()
    y = mim['death_365'].values.astype(int)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xte = Xte.copy()
    for c in CONT:
        Xte[c] = Xte[c].fillna(pp['medians'][c])
    for c in FEATS:
        if c not in CONT:
            Xte[c] = pd.to_numeric(Xte[c], errors='coerce').fillna(0)
    Xte_s = Xte.copy()
    Xte_s[CONT] = pp['scaler'].transform(Xte[CONT])

    lp = md['logit_params']
    p_lr = 1 / (1 + np.exp(-(lp['const'] + Xte_s[FEATS].values @
                             np.array([lp[f] for f in FEATS]))))
    p_xg = md['xgb'].predict_proba(Xte[FEATS])[:, 1]

    got = {'Logistic': roc_auc_score(yte, p_lr), 'XGBoost': roc_auc_score(yte, p_xg)}
    for nm, v in got.items():
        want = ref['MIMIC_internal_1yr'][nm]['point']['auc']
        assert abs(v - want) < 1e-9, \
            f"RECONCILE FAIL: MIMIC_internal_1yr {nm} AUC {v!r} != 87 的 {want!r}"
        log(f"  [reconcile OK] MIMIC_internal_1yr {nm:<8} AUC {v:.6f} == 87 产出 {want:.6f}")
    return mim, Xte, Xte_s, yte, p_lr, p_xg, md, pp


# ================================================================ PART 1
def part1_severity(mim, Xte, yte, p_lr, p_xg):
    log("")
    log("=" * 78)
    log("PART 1 | 与既有 ICU 严重度评分头对头（MIMIC-IV 内部 30% 测试集，1 年死亡）")
    log("=" * 78)
    # 与 PART 0 完全同一个切分：用测试集的 stay_id 取回评分
    sv = pd.read_csv(DATA / "mimic_severity_official.csv")
    te = mim.loc[Xte.index].copy()
    te = te.merge(sv, on='stay_id', how='left', validate='one_to_one')
    for c in ['oasis_official', 'sapsii_official']:
        assert te[c].notna().all(), f"official {c} 存在缺失"
    te['sofa'] = te['sofa_day1'].fillna(te['sofa_day1'].median())
    te['oasis'] = te['oasis_official']
    te['saps2'] = te['sapsii_official']
    y = te['death_365'].values.astype(int)
    assert len(y) == len(yte) and np.array_equal(y, yte), "PART1 标签与 PART0 测试集不一致"

    R = {'n': int(len(te)), 'events': int(y.sum()),
         'outcome': 'death within 365 days of ICU admission',
         'scores': {}, 'comparison': {}}
    log(f"  test set n={len(te):,}  events={y.sum():,}")
    for k, label in [('sofa', 'SOFA (first-day)'), ('oasis', 'OASIS (official derived)'),
                     ('saps2', 'SAPS-II (official derived)')]:
        auc = float(roc_auc_score(y, te[k].values))
        R['scores'][k] = {'label': label,
                          'median': round(float(te[k].median()), 2),
                          'IQR': [round(float(te[k].quantile(.25)), 2),
                                  round(float(te[k].quantile(.75)), 2)],
                          'min': round(float(te[k].min()), 2),
                          'max': round(float(te[k].max()), 2),
                          'auc_alone': auc,
                          'auc_alone_ci': boot_ci(y, te[k].values)}
        s = R['scores'][k]
        log(f"    {k:<6} median {s['median']:>6.1f}  IQR {s['IQR']}  "
            f"range [{s['min']},{s['max']}]  AUC-alone {auc:.4f} "
            f"[{s['auc_alone_ci'][0]:.3f},{s['auc_alone_ci'][1]:.3f}]")

    log("")
    log("  DeLong：全模型 vs 严重度评分（同一批样本、配对比较）")
    for k in ['sofa', 'oasis', 'saps2']:
        R['comparison'][k] = {}
        for nm, pf in [('Logistic', p_lr), ('XGBoost', p_xg)]:
            a1, a2, z, pv = delong_test(y, pf, te[k].values)
            R['comparison'][k][nm] = {'auc_model': a1, 'auc_score': a2,
                                      'dAUC': a1 - a2, 'z': z,
                                      'p': float(f"{pv:.4g}")}
            log(f"    {k:<6} vs {nm:<8} model {a1:.4f} | score {a2:.4f} | "
                f"Δ{a1 - a2:+.4f} | DeLong p={pv:.3g}")
        # 评分当单变量做 logistic 的校准（供参考）
        lp = np.log(np.clip(te[k].values, 1e-6, None) + 1.0)
        m = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), y)
        p = m.predict_proba(lp.reshape(-1, 1))[:, 1]
        sl, ic = cal_slope_intercept(y, p)
        R['comparison'][k]['logistic_on_score'] = {
            'auc': float(roc_auc_score(y, p)), 'brier': float(brier_score_loss(y, p)),
            'slope': sl, 'intercept': ic,
            'oe': float(y.mean() / p.mean()), 'ece': ece(y, p)}
        c = R['comparison'][k]['logistic_on_score']
        log(f"    {k:<6} 单变量 logistic 化后：AUC {c['auc']:.4f} slope {c['slope']:.3f} "
            f"O:E {c['oe']:.3f} ECE {c['ece']:.4f}")
    return R


# ================================================================ PART 2
def part2_gcs(mim):
    log("")
    log("=" * 78)
    log("PART 2 | GCS 增量：base(28 特征) / +总分 / +运动项（MIMIC-IV，30% hold-out，seed=42）")
    log("=" * 78)
    gcsf = pd.read_csv(DATA / "mimic_gcs.csv")
    df = mim.merge(gcsf, on='stay_id', how='left')
    for c in ['gcs_min', 'gcs_motor']:
        assert c in df.columns, f"missing {c}"
    y = df['death_365'].values
    miss = {'gcs_min_pct': round(100 * float(df['gcs_min'].isna().mean()), 2),
            'gcs_motor_pct': round(100 * float(df['gcs_motor'].isna().mean()), 2)}
    log(f"  缺失率：gcs_min {miss['gcs_min_pct']}% | gcs_motor {miss['gcs_motor_pct']}%")

    DIAG = {}

    def prep_split(extra=None):
        cols = FEATS + ([extra] if extra else [])
        X = df[cols].copy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                              random_state=SEED)
        imp = CONT + ([extra] if extra else [])
        med = Xtr[imp].median()
        for c in imp:
            Xtr[c] = Xtr[c].fillna(med[c])
            Xte[c] = Xte[c].fillna(med[c])
        sc = StandardScaler().fit(Xtr[imp])
        Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
        Xtr_s[imp] = sc.transform(Xtr[imp])
        Xte_s[imp] = sc.transform(Xte[imp])
        idx = np.asarray(Xte.index, dtype=np.int64)
        DIAG[extra or 'base'] = {
            'n_test': int(len(Xte)), 'events_test': int(np.asarray(yte).sum()),
            'test_index_fp': hashlib.sha1(idx.tobytes()).hexdigest()[:16],
            'yte_fp': hashlib.sha1(np.asarray(yte).tobytes()).hexdigest()[:16],
            'med_cont': {c: float(med[c]) for c in CONT},
            'mean_cont': {c: float(sc.mean_[i]) for i, c in enumerate(imp)},
            'scale_cont': {c: float(sc.scale_[i]) for i, c in enumerate(imp)},
            'Xte_s_cont': Xte_s[CONT].to_numpy(dtype=float)}
        return Xtr, Xte, Xtr_s, Xte_s, np.asarray(ytr), np.asarray(yte), cols

    def fit_lr(Xtr_s, ytr, Xte_s, cols):
        lr = LogisticRegression(max_iter=2000, C=1e6).fit(Xtr_s[cols], ytr)
        return lr.predict_proba(Xte_s[cols])[:, 1]

    def fit_xgb(Xtr, ytr, Xte, cols):
        import xgboost as xgb
        # ★ 2026-10-19（118 号脚本）：GCS 增量分析中的 XGBoost 也改用主分析的选中超参
        #   （tuned_hyperparameters_v1.json），保证"加 GCS 前后"与主结果同源。
        _hp = json.load(open(OUT / "tuned_hyperparameters_v1.json",
                             encoding="utf-8"))["one_year"]["XGBoost"]["best_params"]
        m = xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                              random_state=SEED, tree_method='hist',
                              n_jobs=-1, **_hp).fit(Xtr[cols], ytr)
        return m.predict_proba(Xte[cols])[:, 1]

    variants = {'base': None, 'plus_total_gcs': 'gcs_min', 'plus_motor_gcs': 'gcs_motor'}
    preds = {}
    for tag, extra in variants.items():
        Xtr, Xte, Xtr_s, Xte_s, ytr, yte, cols = prep_split(extra)
        preds[tag] = {'Logistic': fit_lr(Xtr_s, ytr, Xte_s, cols),
                      'XGBoost': fit_xgb(Xtr, ytr, Xte, cols)}
        log(f"  fitted {tag:16s} n_feat={len(cols):2d}  "
            f"LR={roc_auc_score(yte, preds[tag]['Logistic']):.4f}  "
            f"XGB={roc_auc_score(yte, preds[tag]['XGBoost']):.4f}")

    b = DIAG['base']
    CMP = {'verified': True, 'n_test': b['n_test'], 'events_test': b['events_test'],
           'test_index_fp': b['test_index_fp'], 'checks': {}}
    for extra, lab in [('gcs_min', 'plus_total_gcs'), ('gcs_motor', 'plus_motor_gcs')]:
        d = DIAG[extra]
        assert d['n_test'] == b['n_test'], f"COMPARABILITY FAIL: {lab} 测试集大小不同"
        assert d['test_index_fp'] == b['test_index_fp'], \
            f"COMPARABILITY FAIL: {lab} 测试行不是同一批"
        assert d['yte_fp'] == b['yte_fp'], f"COMPARABILITY FAIL: {lab} 测试标签不同"
        assert d['events_test'] == b['events_test'], f"COMPARABILITY FAIL: {lab} 事件数不同"
        for c in CONT:
            assert d['med_cont'][c] == b['med_cont'][c], \
                f"COMPARABILITY FAIL: {lab} 列 {c} 插补中位数不同"
            assert d['mean_cont'][c] == b['mean_cont'][c], \
                f"COMPARABILITY FAIL: {lab} 列 {c} 标度均值不同"
            assert d['scale_cont'][c] == b['scale_cont'][c], \
                f"COMPARABILITY FAIL: {lab} 列 {c} 标度尺度不同"
        assert np.array_equal(d['Xte_s_cont'], b['Xte_s_cont']), \
            f"COMPARABILITY FAIL: {lab} 标准化后的测试矩阵不同"
        CMP['checks'][lab] = {'same_test_rows': True, 'same_labels': True,
                              'same_imputation_medians': True, 'same_scaler': True,
                              'n_shared_continuous': len(CONT)}
        log(f"  [comparability OK] {lab:16s} 与 base 同一测试集（n={b['n_test']}, "
            f"events={b['events_test']}），{len(CONT)} 个共有连续列插补与标度一致")

    inc = {'comparability': CMP, 'missingness_pct': miss}
    for learner in ['Logistic', 'XGBoost']:
        pb = preds['base'][learner]
        inc[learner] = {'base_auc': float(roc_auc_score(yte, pb)),
                        'base_ci': boot_ci(yte, pb)}
        log(f"    {learner:8s} base AUC {inc[learner]['base_auc']:.4f}")
        for tag in ['plus_total_gcs', 'plus_motor_gcs']:
            pg = preds[tag][learner]
            a_new, a_old, z, pv = delong_test(yte, pg, pb)
            cnri, nrie, nrine, idi = nri_idi(yte, pb, pg)
            d = {'auc': float(roc_auc_score(yte, pg)),
                 'dAUC': float(roc_auc_score(yte, pg) - roc_auc_score(yte, pb)),
                 'delong_z': z, 'delong_p': float(pv), 'cNRI': cnri,
                 'NRI_event': nrie, 'NRI_nonevent': nrine, 'IDI': idi,
                 'ci': boot_ci(yte, pg)}
            inc[learner][tag] = d
            log(f"    {learner:8s} {tag:16s} AUC {d['auc']:.4f}  Δ{d['dAUC']:+.4f}  "
                f"DeLong p={pv:.4f}  cNRI {cnri:+.3f}  IDI {idi:+.4f}")

    # 单独判别力（单变量 logistic）
    def alone_auc(col):
        Xtr, Xte_, _, _, ytr, yte_, _ = prep_split(col)
        p = LogisticRegression(max_iter=1000).fit(Xtr[[col]], ytr).predict_proba(Xte_[[col]])[:, 1]
        return float(roc_auc_score(yte_, p))
    inc['alone_AUC_logit'] = {'gcs_total': alone_auc('gcs_min'),
                              'gcs_motor': alone_auc('gcs_motor')}
    log(f"  单变量 logistic 判别力：总分 {inc['alone_AUC_logit']['gcs_total']:.4f} | "
        f"运动项 {inc['alone_AUC_logit']['gcs_motor']:.4f}")

    # 运动项相对总分回收了多少（避免"几乎无代价"式误读）
    for learner in ['Logistic', 'XGBoost']:
        dt = inc[learner]['plus_total_gcs']['dAUC']
        dm = inc[learner]['plus_motor_gcs']['dAUC']
        inc[learner]['motor_over_total_ratio'] = (dm / dt) if dt else None
    log(f"  ΔAUC 倍数（运动项 / 总分）：LR "
        f"{inc['Logistic']['motor_over_total_ratio']:.2f}× | XGB "
        f"{inc['XGBoost']['motor_over_total_ratio']:.2f}×")
    return inc


# ================================================================ PART 3
def part3_single_site():
    """★ 单中心是院内死亡结局 → 必须用**院内模型**的冻结 artefacts
    （87 的 art2：medians/scaler/lr/xg 均来自 stratify=death_hosp 的 70% 训练集），
    与 1 年模型的 split 不同，混用会给出错误数字。"""
    log("")
    log("=" * 78)
    log("PART 3 | 单中心（淮安）审计 —— 院内结局，使用院内模型冻结 artefacts")
    log("=" * 78)
    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    if not P.exists():
        log("  ⚠ 淮安原始表不存在，跳过")
        return {'skipped': str(P)}
    ih = joblib.load(OUT / "model_inhosp_v2.joblib")
    pp = {'medians': ih['medians'], 'scaler': ih['scaler']}
    raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')          # noqa: E731
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
    Zl = Xl.copy()
    for c in CONT:
        Zl[c] = Zl[c].fillna(pp['medians'][c])
    for c in FEATS:
        if c not in CONT:
            Zl[c] = Zl[c].fillna(0)
    Zls = Zl.copy()
    Zls[CONT] = pp['scaler'].transform(Zl[CONT])
    p_lr = ih['lr'].predict_proba(Zls[FEATS])[:, 1]
    p_xg = ih['xg'].predict_proba(Zl[FEATS])[:, 1]
    y = dx['y'].values.astype(int)
    # 对账：必须复现 87 报出的淮安院内 AUC
    ref = json.load(open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))
    for nm, p in [('Logistic', p_lr), ('XGBoost', p_xg)]:
        want = ref['Huaian_inhosp'][nm]['point']['auc']
        got = float(roc_auc_score(y, p))
        assert abs(got - want) < 1e-9, \
            f"RECONCILE FAIL: Huaian_inhosp {nm} AUC {got!r} != 87 的 {want!r}"
        log(f"  [reconcile OK] Huaian_inhosp {nm:<8} AUC {got:.6f} == 87 产出 {want:.6f}")

    R = {'n': int(len(dx)), 'events': int(y.sum()),
         'aetiology_mix': {c: int(dx[c].sum()) for c in AET},
         'aetiology_mix_pct': {c: round(100 * float(dx[c].mean()), 1) for c in AET},
         'model': {}}
    log(f"  n={len(dx)}  院内死亡={int(y.sum())}")
    log("  病因构成：" + ", ".join(f"{c}={int(dx[c].sum())} ({100*dx[c].mean():.1f}%)"
                                 for c in AET))
    for nm, p in [('Logistic', p_lr), ('XGBoost', p_xg)]:
        sl, ic = cal_slope_intercept(y, p)
        R['model'][nm] = {'auc': float(roc_auc_score(y, p)), 'ci': boot_ci(y, p),
                          'brier': float(brier_score_loss(y, p)), 'slope': sl,
                          'intercept': ic, 'oe': float(y.mean() / p.mean()),
                          'ece': ece(y, p)}
        r = R['model'][nm]
        log(f"    {nm:<8} AUC {r['auc']:.4f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]  "
            f"slope {r['slope']:.3f}  O:E {r['oe']:.3f}  ECE {r['ece']:.4f}")
    # 单预测因子单独 AUC（重点：anoxic 是否能单独复现模型判别力）
    alone = {}
    for c in FEATS:
        v = Zl[c]
        if v.nunique(dropna=True) < 2:
            continue
        p = v.fillna(v.median()).values.astype(float)
        a = roc_auc_score(y, p)
        # 方向反转后取较大者（单变量 AUC 通常按"风险越高越大"理解）
        alone[c] = float(max(a, 1 - a))
    R['single_predictor_auc'] = alone
    top = sorted(alone.items(), key=lambda kv: -kv[1])[:6]
    log("  单预测因子 |AUC-0.5| 前 6：" + ", ".join(f"{k}={v:.3f}" for k, v in top))
    R['top_single_predictors'] = [{'feature': k, 'auc': v} for k, v in top]
    return R


def main():
    log("#" * 78)
    log("# 90_downstream_v9_severity_gcs.py | V9 下游重算：严重度头对头 / GCS 增量 / 单中心")
    log("#" * 78)
    log("")
    log("[PART 0] 与 87 的冻结模型对账（防口径漂移）")
    mim, Xte, Xte_s, yte, p_lr, p_xg, md, pp = load_and_reconcile()

    R = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
         'data': {'MIMIC': 'data/mimic_abi_cohort_v2.csv',
                  'severity': 'data/mimic_severity_official.csv (MIT-LCP official derived)',
                  'sofa': 'sql/01 first_day_sofa → sofa_day1',
                  'gcs': 'data/mimic_gcs.csv'},
         'severity_headtohead': part1_severity(mim, Xte, yte, p_lr, p_xg),
         'gcs_increment': part2_gcs(mim),
         'single_site_huaian': part3_single_site()}

    json.dump(R, open(OUT / "downstream_severity_gcs_v9.json", 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    (OUT / "_90_downstream_severity_gcs.md").write_text(
        "# 90 —— V9 下游重算（严重度 / GCS / 单中心）\n\n```\n" + "\n".join(LOG) + "\n```\n",
        encoding='utf-8')
    log("")
    log("→ output/downstream_severity_gcs_v9.json / _90_downstream_severity_gcs.md")


if __name__ == "__main__":
    main()
