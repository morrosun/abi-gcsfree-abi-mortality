# -*- coding: utf-8 -*-
"""Part 1 | 本地队列外部验证（院内死亡）: 基线表 + 主分析 + 敏感性分析
   输出 output/local_part1_stats.json （供 HTML 手稿组装，零占位符）"""
import pandas as pd, numpy as np, joblib, json, os
# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from scipy import stats

BASE = ABI_BASE
P = os.path.join(BASE, 'ABI3', 'output', 'ABI本地数据采集模板_已录入检验结果.xlsx')
OUTJ = os.path.join(BASE, 'ABI3', 'output', 'local_part1_stats.json')

m = joblib.load(os.path.join(BASE, 'output', 'model_inhosp.joblib'))
feats, cont, medians, scaler = m['feats'], m['cont'], m['medians'], m['scaler']
binv = [c for c in feats if c not in cont]

# ---------------- 载入 ----------------
df = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
df = df.dropna(how='all').reset_index(drop=True)
raw_n = len(df)
num = lambda c: pd.to_numeric(df[c], errors='coerce')

d = pd.DataFrame(index=df.index)
for c in feats:
    d[c] = num(c)
d['age'] = num('age')
d['gcs'] = num('gcs')
d['dl'] = num('hospital_discharge_location')
d['y'] = (d['dl'] == 5).astype(int)
d['y_known'] = d['dl'].notna()
# LOS 由时间字段计算（模板未直接提供）
_tin = pd.to_datetime(df['icu_in_time'], errors='coerce')
_tout = pd.to_datetime(df['icu_out_time'], errors='coerce')
d['icu_los_days'] = (_tout - _tin).dt.total_seconds() / 86400.0
_hin = pd.to_datetime(df['hospital_admission_time'], errors='coerce')
_hout = pd.to_datetime(df['hospital_discharge_time'], errors='coerce')
d['hospital_los_days'] = (_hout - _hin).dt.total_seconds() / 86400.0

PRIO = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
nflag = d[PRIO].fillna(0).sum(axis=1)
keep = (nflag >= 1) & (d['age'] >= 18) & d['age'].notna() & d['y_known']

flow = {
    'raw': int(raw_n),
    'excl_no_etiology': int((nflag < 1).sum()),
    'excl_age_lt18': int(((nflag >= 1) & (d['age'] < 18)).sum()),
    'excl_outcome_missing': int(((nflag >= 1) & (d['age'] >= 18) & (~d['y_known'])).sum()),
    'included': int(keep.sum()),
    'excluded_inhosp_deaths_after_icu_unknown': None,
}
dfx = d[keep].copy()

# ---------------- SI → 传统单位（模型训练口径） ----------------
CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
        'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
for c, f in CONV.items():
    dfx[c] = f(dfx[c])

# ---------------- 冻结预处理 ----------------
def frozen_matrix(dd):
    X = dd[feats].copy()
    for c in cont:
        X[c] = X[c].fillna(medians[c])
    for c in binv:
        X[c] = X[c].fillna(0)
    Xs = X.copy(); Xs[cont] = scaler.transform(X[cont])
    return X, Xs

def eval_metrics(dd, tag, nboot=2000):
    X, Xs = frozen_matrix(dd)
    y = dd['y'].values
    out = {'tag': tag, 'n': int(len(dd)), 'events': int(y.sum()),
           'prev': round(100 * y.mean(), 1)}
    if len(np.unique(y)) < 2:
        out['note'] = '仅单一结局，无法估计判别力'
        return out
    for nm, p in [('Logistic', m['lr'].predict_proba(Xs)[:, 1]),
                  ('XGBoost', m['xg'].predict_proba(X)[:, 1])]:
        auc = roc_auc_score(y, p)
        rng = np.random.default_rng(42); bs = []
        for _ in range(nboot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2: continue
            bs.append(roc_auc_score(y[i], p[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        oe = y.mean() / p.mean()
        br = brier_score_loss(y, p)
        lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
        cal = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000
                                 ).fit(lp.reshape(-1, 1), y)
        out[nm] = {'AUC': round(float(auc), 3), 'CI': [round(float(lo), 3), round(float(hi), 3)],
                   'pred': round(100 * float(p.mean()), 1), 'OE': round(float(oe), 2),
                   'brier': round(float(br), 3),
                   'slope': round(float(cal.coef_[0][0]), 2),
                   'intercept': round(float(cal.intercept_[0]), 2)}
    return out

main = eval_metrics(dfx, '主分析（全纳入 n=%d）' % len(dfx))

# ---------- 敏感性 1：完整病例（模型特征全观测，无插补） ----------
cc = dfx.dropna(subset=cont)
sens_cc = eval_metrics(cc, '敏感性分析 1：完整病例（无中位数插补）')

# ---------- 敏感性 2：排除含越界值记录（体温/血压/血小板/血钠） ----------
RANGE = {'mbp_mean': (20, 150), 'temperature_mean': (30, 43),
         'platelets_min': (5, 1000), 'sodium_min': (100, 180)}
bad = pd.Series(False, index=dfx.index)
for c, (lo, hi) in RANGE.items():
    v = dfx[c]
    bad |= (v < lo) | (v > hi)
    # 体温华氏度未换算者（>50）
    if c == 'temperature_mean':
        bad |= (v > 50)
sens_range = eval_metrics(dfx[~bad], '敏感性分析 2：排除越界/未换算值（n 剔除 %d）' % int(bad.sum()))

# ---------- 敏感性 3：仅机械通气亚组（与训练人群同质） ----------
sens_mv = eval_metrics(dfx[dfx['mech_vent'] == 1], '敏感性分析 3：仅机械通气者')

# ---------- 敏感性 4：XGBoost 本地重校准（logistic recalibration） ----------
X, Xs = frozen_matrix(dfx); y = dfx['y'].values
p_xg = m['xg'].predict_proba(X)[:, 1]
lp = np.log(np.clip(p_xg, 1e-9, 1 - 1e-9) / (1 - np.clip(p_xg, 1e-9, 1 - 1e-9)))
rc = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(lp.reshape(-1, 1), y)
p_rc = rc.predict_proba(lp.reshape(-1, 1))[:, 1]
sens_rc = {'tag': '敏感性分析 4：XGBoost 本地重校准（截距+斜率）', 'n': int(len(dfx)), 'events': int(y.sum()),
           'AUC': round(float(roc_auc_score(y, p_rc)), 3),
           'OE': round(float(y.mean() / p_rc.mean()), 2),
           'brier': round(float(brier_score_loss(y, p_rc)), 3),
           'slope': round(float(rc.coef_[0][0]), 2), 'intercept': round(float(rc.intercept_[0]), 2)}

# ---------------- Table 1：基线特征（按院内死亡分组） ----------------
def pval_num(a, b):
    a = a.dropna(); b = b.dropna()
    if len(a) < 3 or len(b) < 3: return None
    try: return float(stats.mannwhitneyu(a, b).pvalue)
    except Exception: return None

def pval_cat(x, y):
    tab = pd.crosstab(x, y)
    if tab.shape[1] < 2 or tab.shape[0] < 2: return None
    try:
        if tab.values.min() < 5 and tab.size == 4:
            return float(stats.fisher_exact(tab.values)[1])
        return float(stats.chi2_contingency(tab)[1])
    except Exception: return None

def med_iqr(s):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0: return '—'
    return '%.1f (%.1f–%.1f)' % (s.median(), s.quantile(.25), s.quantile(.75))

def n_pct(mask):
    n = int(mask.sum()); tot = int(mask.notna().sum())
    if tot == 0: return '—'
    return '%d (%.1f%%)' % (n, 100 * n / tot)

surv = dfx[dfx['y'] == 0]; died = dfx[dfx['y'] == 1]
T1 = []
def si(col, s):
    """把已换算为传统单位的列还原成采集时的 SI 单位用于展示"""
    if col == 'creatinine_max': return s * 88.4
    if col == 'bun_max':        return s * 0.357
    if col == 'glucose_max':    return s / 18.0
    if col == 'hemoglobin_min': return s * 10.0
    return s

T1.append({'item': '年龄（岁）', 'all': med_iqr(dfx['age']), 'surv': med_iqr(surv['age']),
           'died': med_iqr(died['age']), 'p': pval_num(surv['age'], died['age'])})
T1.append({'item': '女性', 'all': n_pct(dfx['female'] == 1), 'surv': n_pct(surv['female'] == 1),
           'died': n_pct(died['female'] == 1), 'p': pval_cat(dfx['female'], dfx['y'])})
T1.append({'item': 'Charlson 合并症指数', 'all': med_iqr(dfx['charlson_comorbidity_index']),
           'surv': med_iqr(surv['charlson_comorbidity_index']), 'died': med_iqr(died['charlson_comorbidity_index']),
           'p': pval_num(surv['charlson_comorbidity_index'], died['charlson_comorbidity_index'])})
T1.append({'item': 'GCS（入ICU）', 'all': med_iqr(dfx['gcs']), 'surv': med_iqr(surv['gcs']),
           'died': med_iqr(died['gcs']), 'p': pval_num(surv['gcs'], died['gcs'])})
for nm, col in [('心率（次/分）', 'heart_rate_mean'), ('平均动脉压（mmHg）', 'mbp_mean'),
                ('呼吸频率（次/分）', 'resp_rate_mean'), ('体温（℃）', 'temperature_mean'),
                ('SpO₂（%）', 'spo2_mean'), ('白细胞（×10⁹/L）', 'wbc_max'),
                ('血红蛋白（g/L）', 'hemoglobin_min'), ('血小板（×10⁹/L）', 'platelets_min'),
                ('血钠（mmol/L）', 'sodium_min'), ('血钾（mmol/L）', 'potassium_max'),
                ('肌酐（μmol/L）*', 'creatinine_max'), ('尿素氮（mmol/L）*', 'bun_max'),
                ('血糖（mmol/L）*', 'glucose_max'), ('INR', 'inr_max')]:
    T1.append({'item': nm, 'all': med_iqr(si(col, dfx[col])),
               'surv': med_iqr(si(col, surv[col])), 'died': med_iqr(si(col, died[col])),
               'p': pval_num(surv[col], died[col])})
for nm, col in [('创伤性脑损伤（TBI）', 'tbi'), ('蛛网膜下腔出血（SAH）', 'sah'),
                ('脑出血（ICH）', 'ich'), ('缺氧性脑损伤', 'anoxic'),
                ('急性缺血性卒中（AIS）', 'ais'), ('中枢神经系统感染', 'cns_inf'),
                ('癫痫持续状态', 'seizure'), ('机械通气', 'mech_vent'),
                ('血管活性药', 'vasopressor'), ('肾脏替代治疗（RRT）', 'rrt')]:
    T1.append({'item': nm, 'all': n_pct(dfx[col] == 1), 'surv': n_pct(surv[col] == 1),
               'died': n_pct(died[col] == 1), 'p': pval_cat(dfx[col], dfx['y'])})
T1.append({'item': 'ICU 住院时长（天）', 'all': med_iqr(dfx['icu_los_days']), 'surv': med_iqr(surv['icu_los_days']),
           'died': med_iqr(died['icu_los_days']), 'p': pval_num(surv['icu_los_days'], died['icu_los_days'])})

res = {'flow': flow, 'main': main, 'sensitivity': [sens_cc, sens_range, sens_mv, sens_rc],
       'table1': T1,
       'training': {'mimic_n': 16597, 'mimic_inhosp_events': 2799, 'mimic_inhosp_prev': 16.9,
                    'internal_LR': 0.845, 'internal_XGB': 0.8754},
       'prior_external_inhosp': {
           'eICU': {'n': 14852, 'events': 1894, 'prev': 12.8, 'LR': 0.811, 'XGB': 0.850},
           'INSPIRE': {'n': 1543, 'events': 126, 'prev': 8.2, 'LR': 0.644, 'XGB': 0.720},
           'NWICU': {'n': 3420, 'events': 16, 'prev': 0.5, 'LR': 0.524, 'XGB': 0.683}},
       'unit_conversion': {'creatinine_max': '÷88.4 μmol/L→mg/dL', 'bun_max': '÷0.357 mmol/L→mg/dL',
                           'glucose_max': '×18 mmol/L→mg/dL', 'hemoglobin_min': '÷10 g/L→g/dL'}}

json.dump(res, open(OUTJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print('原始 n =', raw_n, '→ 纳入 n =', int(keep.sum()))
print('流动:', json.dumps(flow, ensure_ascii=False))
print('\n主分析:', json.dumps(main, ensure_ascii=False, indent=1))
for s in res['sensitivity']:
    print('\n' + json.dumps(s, ensure_ascii=False))
print('\nTable1 行数:', len(T1))
print('WROTE', OUTJ)
