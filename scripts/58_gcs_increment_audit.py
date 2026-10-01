# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""Decision-E 审计 · 两个决定性补分析（2026-09-30）

 [1] GCS 运动项增量（决定"省掉 GCS 几乎无代价"这一核心主张的生死）
     编辑原话：GCS 总分被插管/镇静污染，而 motor component alone 表现明显更好，
     因此"神经系统状态预后价值不大"并未被证实。
     我们此前只跑了「+总分 GCS」(ΔAUC 0.005–0.006)，**从未跑「+运动项 GCS」**。
     本脚本在同一队列、同一 split(seed=42)、同一预处理下同时跑三种特征集：
         base(28 特征, 无 GCS) / +gcs_min(总分) / +gcs_motor(运动项)
     两个学习器(Logistic / XGBoost)，配 DeLong、cNRI、IDI、bootstrap CI。
     自检：+gcs_min 的 ΔAUC 必须复现稿件已报的 0.005–0.006，否则说明口径跑偏。

 [2] 单中心队列审计（核编辑"小样本 + 几乎全 TBI + 缺氧标志单独即可复现模型判别力"）
     n=225、病因构成、各单预测因子单独 AUC（重点 anoxic）、与冻结模型 AUC 对比。

只做计算与取证，不改任何稿件。产物 output/gcs_increment_audit.json
"""
import pandas as pd, numpy as np, json, sys, os
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

def log(*a):
    print(*a); sys.stdout.flush()

BASE = Path(ABI_BASE)
OUT = BASE / "output"; DATA = BASE / "data"
np.random.seed(42)
rep = {}

aet = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
cont = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
binv = ['mech_vent', 'vasopressor', 'rrt']
feats = ['age', 'female'] + aet + ['charlson_comorbidity_index'] + \
        [c for c in cont if c not in ('age', 'charlson_comorbidity_index')] + binv

# ---------------- DeLong (Sun & Xu 2014) ----------------
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_var(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]; m, n = len(pos), len(neg)
    tx = _midrank(pos); ty = _midrank(neg); tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n; v10 = 1.0 - (tz[m:] - ty) / m
    return auc, v01, v10, m, n

def delong_test(y, p_new, p_old):
    y = np.asarray(y)
    a1, v01a, v10a, m, n = delong_var(y, np.asarray(p_new))
    a2, v01b, v10b, _, _ = delong_var(y, np.asarray(p_old))
    s01 = np.cov(np.vstack([v01a, v01b])); s10 = np.cov(np.vstack([v10a, v10b]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return a1, a2, z, 2 * (1 - stats.norm.cdf(abs(z)))

def boot_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    idx = np.arange(len(y)); a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        a.append(roc_auc_score(y[b], p[b]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def nri_idi(y, p_old, p_new):
    y = np.asarray(y); p_old = np.asarray(p_old); p_new = np.asarray(p_new)
    ev = y == 1; ne = y == 0
    nri_e = np.mean(p_new[ev] > p_old[ev]) - np.mean(p_new[ev] < p_old[ev])
    nri_ne = np.mean(p_new[ne] < p_old[ne]) - np.mean(p_new[ne] > p_old[ne])
    idi = (p_new[ev].mean() - p_old[ev].mean()) - (p_new[ne].mean() - p_old[ne].mean())
    return float(nri_e + nri_ne), float(nri_e), float(nri_ne), float(idi)

# ==================================================================
# [1] GCS 增量：base / +总分 / +运动项
# ==================================================================
log("=" * 78)
log("[1] GCS incremental value  (MIMIC-IV, 30% hold-out, seed=42)")
log("=" * 78)

df = pd.read_csv(DATA / "mimic_abi_cohort.csv")
gcs = pd.read_csv(DATA / "mimic_gcs.csv")
df = df.merge(gcs, on='stay_id', how='left')
df['female'] = (df['gender'] == 'F').astype(int)
y = df['death_365'].values

for c in ['gcs_min', 'gcs_motor']:
    assert c in df.columns, f"missing column {c} in mimic_gcs.csv"

miss = {'gcs_min_pct': round(100 * float(df['gcs_min'].isna().mean()), 2),
        'gcs_motor_pct': round(100 * float(df['gcs_motor'].isna().mean()), 2)}
log(f"  缺失率: gcs_min {miss['gcs_min_pct']}% | gcs_motor {miss['gcs_motor_pct']}%")

def prep_split(extra=None):
    cols = feats + ([extra] if extra else [])
    X = df[cols].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    imp_cols = cont + ([extra] if extra else [])
    med = Xtr[imp_cols].median()
    for c in imp_cols:
        Xtr[c] = Xtr[c].fillna(med[c]); Xte[c] = Xte[c].fillna(med[c])
    sc = StandardScaler().fit(Xtr[imp_cols])
    Xtr_s = Xtr.copy(); Xte_s = Xte.copy()
    Xtr_s[imp_cols] = sc.transform(Xtr[imp_cols]); Xte_s[imp_cols] = sc.transform(Xte[imp_cols])
    return Xtr, Xte, Xtr_s, Xte_s, np.asarray(ytr), np.asarray(yte), cols

def fit_lr(Xtr_s, ytr, Xte_s, cols):
    lr = LogisticRegression(max_iter=2000, C=1e6).fit(Xtr_s[cols], ytr)
    return lr.predict_proba(Xte_s[cols])[:, 1]

def fit_xgb(Xtr, ytr, Xte, cols):
    mdl = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric='logloss', random_state=42,
                            tree_method='hist', n_jobs=-1).fit(Xtr[cols], ytr)
    return mdl.predict_proba(Xte[cols])[:, 1]

variants = {'base': None, 'plus_total_gcs': 'gcs_min', 'plus_motor_gcs': 'gcs_motor'}
preds = {}
for tag, extra in variants.items():
    Xtr, Xte, Xtr_s, Xte_s, ytr, yte, cols = prep_split(extra)
    preds[tag] = {'Logistic': fit_lr(Xtr_s, ytr, Xte_s, cols),
                  'XGBoost': fit_xgb(Xtr, ytr, Xte, cols)}
    log(f"  fitted {tag:16s} n_feat={len(cols)}  "
        f"LR={roc_auc_score(yte, preds[tag]['Logistic']):.4f}  "
        f"XGB={roc_auc_score(yte, preds[tag]['XGBoost']):.4f}")

inc = {}
for learner in ['Logistic', 'XGBoost']:
    inc[learner] = {}
    pb = preds['base'][learner]
    inc[learner]['base_auc'] = round(float(roc_auc_score(yte, pb)), 4)
    inc[learner]['base_ci'] = boot_ci(yte, pb)
    for tag in ['plus_total_gcs', 'plus_motor_gcs']:
        pg = preds[tag][learner]
        a_new, a_old, z, pv = delong_test(yte, pg, pb)
        cnri, nrie, nrine, idi = nri_idi(yte, pb, pg)
        d = {'auc': round(float(roc_auc_score(yte, pg)), 4),
             'dAUC': round(float(roc_auc_score(yte, pg) - roc_auc_score(yte, pb)), 4),
             'delong_z': round(float(z), 3), 'delong_p': float(pv),
             'cNRI': round(cnri, 4), 'NRI_event': round(nrie, 4),
             'NRI_nonevent': round(nrine, 4), 'IDI': round(idi, 4),
             'ci': boot_ci(yte, pg)}
        inc[learner][tag] = d
        log(f"    {learner:8s} {tag:16s} AUC {d['auc']:.4f}  dAUC {d['dAUC']:+.4f}  "
            f"DeLong p={pv:.4f}  cNRI {cnri:+.3f}  IDI {idi:+.4f}")

# 自检：+总分 GCS 的 ΔAUC 必须落在稿件已报的 0.005–0.006 区间（容差 0.002）
for learner, lo, hi in [('Logistic', 0.003, 0.008), ('XGBoost', 0.004, 0.009)]:
    v = inc[learner]['plus_total_gcs']['dAUC']
    assert lo <= v <= hi, f"SELF-CHECK FAIL: {learner} +total-GCS dAUC={v} 不在 {lo}-{hi}（稿件已报 0.005-0.006）"
    log(f"  [selfcheck OK] {learner} +total-GCS dAUC={v:+.4f} 复现稿件口径")

# 运动项 / 总分 单独判别力（复核稿件已报：总分 0.551、运动项 0.679）
gmed = float(df.loc[np.arange(len(df))[:int(len(df) * 0.7)], 'gcs_min'].median())
def alone_auc(col):
    Xtr, Xte, _, _, ytr, yte, _ = prep_split(col)
    p = LogisticRegression(max_iter=1000).fit(Xtr[[col]], ytr).predict_proba(Xte[[col]])[:, 1]
    return round(float(roc_auc_score(yte, p)), 4)
inc['alone_AUC_logit'] = {'gcs_total': alone_auc('gcs_min'), 'gcs_motor': alone_auc('gcs_motor')}
log(f"  单独判别力(Logistic 单变量): 总分 {inc['alone_AUC_logit']['gcs_total']} | "
    f"运动项 {inc['alone_AUC_logit']['gcs_motor']}")
inc['missingness_pct'] = miss
rep['gcs_increment_decisionE'] = inc

# ==================================================================
# [2] 单中心队列审计
# ==================================================================
log("")
log("=" * 78)
log("[2] Single-centre (Huaian) cohort audit — n / aetiology / single-flag AUC")
log("=" * 78)

P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
raw = raw.dropna(how='all').reset_index(drop=True)
num = lambda c: pd.to_numeric(raw[c], errors='coerce')
d = pd.DataFrame(index=raw.index)
d['age'] = num('age'); d['dl'] = num('hospital_discharge_location')
d['y'] = (d['dl'] == 5).astype(int); d['y_known'] = d['dl'].notna()
for c in aet:
    d[c] = num(c).fillna(0)
nflag = d[aet].sum(axis=1)
keep = (nflag >= 1) & (d['age'] >= 18) & d['age'].notna() & d['y_known']
dx = d[keep].copy()
log(f"  原始 {len(raw)} → 纳入 {int(keep.sum())} | 院内死亡 {int(dx['y'].sum())} "
    f"({100 * dx['y'].mean():.1f}%)")

comp = {c: int(dx[c].sum()) for c in aet}
rep['single_centre'] = {'n': int(len(dx)), 'events': int(dx['y'].sum()),
                        'event_rate_pct': round(100 * float(dx['y'].mean()), 1),
                        'aetiology_counts': comp,
                        'tbi_pct': round(100 * float(dx['tbi'].mean()), 1),
                        'anoxic_pct': round(100 * float(dx['anoxic'].mean()), 1)}
log(f"  病因构成: {comp}")
log(f"  TBI {rep['single_centre']['tbi_pct']}% | anoxic {rep['single_centre']['anoxic_pct']}%")

# 冻结模型在该队列的 AUC（复核 0.865 / 0.800）
import joblib
m = joblib.load(OUT / "model_inhosp.joblib")
ufeats, ucont, umed, uscal = m['feats'], m['cont'], m['medians'], m['scaler']
Xl = pd.DataFrame(index=dx.index)
for c in ufeats:
    Xl[c] = num(c) if c in raw.columns else np.nan
CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
        'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
for c, f in CONV.items():
    Xl[c] = f(Xl[c])
for c in ucont:
    Xl[c] = Xl[c].fillna(umed[c])
for c in ufeats:
    if c not in ucont:
        Xl[c] = Xl[c].fillna(0)
Xls = Xl.copy(); Xls[ucont] = uscal.transform(Xl[ucont])
p_lr = m['lr'].predict_proba(Xls[ufeats])[:, 1]
p_xg = m['xg'].predict_proba(Xl[ufeats])[:, 1]
yv = dx['y'].values
def auc_ci(p):
    a = roc_auc_score(yv, p)
    rng = np.random.default_rng(42); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[i])) < 2: continue
        bs.append(roc_auc_score(yv[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return round(float(a), 4), [round(float(lo), 4), round(float(hi), 4)]
rep['single_centre']['model_AUC'] = {'Logistic': auc_ci(p_lr), 'XGBoost': auc_ci(p_xg)}
log(f"  冻结模型 AUC: Logistic {rep['single_centre']['model_AUC']['Logistic'][0]} "
    f"{rep['single_centre']['model_AUC']['Logistic'][1]} | "
    f"XGBoost {rep['single_centre']['model_AUC']['XGBoost'][0]} {rep['single_centre']['model_AUC']['XGBoost'][1]}")

# 各单预测因子单独 AUC（重点：anoxic 是否"单独即可复现模型判别力"）
single = {}
for c in ufeats:
    v = Xl[c].values.astype(float)
    if np.nanstd(v) == 0:
        single[c] = {'auc': None, 'note': 'constant in this cohort'}
        continue
    a = roc_auc_score(yv, v)
    single[c] = {'auc': round(float(a), 4), 'abs_from_050': round(abs(float(a) - 0.5), 4)}
rep['single_centre']['single_predictor_AUC'] = single
top = sorted([(k, v['auc']) for k, v in single.items() if v.get('auc') is not None],
             key=lambda t: abs(t[1] - 0.5), reverse=True)[:8]
log("  单预测因子 |AUC-0.5| 排名前 8:")
for k, a in top:
    log(f"      {k:26s} AUC={a:.4f}")
anx = single.get('anoxic', {}).get('auc')
log(f"  ★ anoxic 单独 AUC = {anx}  vs 模型 Logistic "
    f"{rep['single_centre']['model_AUC']['Logistic'][0]} / XGBoost {rep['single_centre']['model_AUC']['XGBoost'][0]}")
rep['single_centre']['editor_claim_anoxic_reproduces_model'] = {
    'anoxic_alone_auc': anx,
    'model_logistic_auc': rep['single_centre']['model_AUC']['Logistic'][0],
    'model_xgb_auc': rep['single_centre']['model_AUC']['XGBoost'][0],
    'gap_logistic': (round(rep['single_centre']['model_AUC']['Logistic'][0] - anx, 4) if anx else None),
    'verdict': ('anoxic 单独远低于模型 → 编辑该子条不成立' if anx and
                rep['single_centre']['model_AUC']['Logistic'][0] - anx > 0.1 else
                '需人工判读：anoxic 单独接近模型')}

json.dump(rep, open(OUT / "gcs_increment_audit.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
log("")
log(f"WROTE {OUT / 'gcs_increment_audit.json'}")
