# -*- coding: utf-8 -*-
"""本地队列对外部验证：院内死亡模型（model_inhosp.joblib）
   关键：本地为 SI 单位，模型为 MIMIC 传统单位，须先换算"""
import pandas as pd, numpy as np, joblib, json, os
# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = ABI_BASE
P = os.path.join(BASE, 'ABI3', 'output', 'ABI本地数据采集模板_已录入检验结果.xlsx')
OUTJ = os.path.join(BASE, 'ABI3', 'output', 'local_inhosp_validation.json')
OUTF = os.path.join(BASE, 'ABI3', 'figs', 'local_calibration_inhosp.png')
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
try:
    font_manager.fontManager.addfont(r"C:/Windows/Fonts/simhei.ttf")
    plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception as e:
    print('font warn:', e)
plt.rcParams['axes.unicode_minus'] = False

m = joblib.load(os.path.join(BASE, 'output', 'model_inhosp.joblib'))
feats, cont, medians, scaler = m['feats'], m['cont'], m['medians'], m['scaler']

# ---------- 载入并清洗 ----------
df = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
df = df.dropna(how='all').reset_index(drop=True)
n0 = len(df)
num = lambda c: pd.to_numeric(df[c], errors='coerce')

d = pd.DataFrame(index=df.index)
d['patient_id'] = df['patient_id']
for c in feats:
    d[c] = num(c)
d['age'] = num('age')
d['dl'] = num('hospital_discharge_location')
d['y'] = (d['dl'] == 5).astype(int)          # 院内死亡结局
d['y_known'] = d['dl'].notna()

# 排除：病因全阴 / 年龄<18（模型训练人群为成人 ABI）
PRIO = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
nflag = d[PRIO].fillna(0).sum(axis=1)
keep = (nflag >= 1) & (d['age'] >= 18) & d['age'].notna() & d['y_known']
print(f'原始 {n0} → 纳入 {int(keep.sum())} （排除 病因全阴|年龄<18|结局缺失）')
dfx = d[keep].copy()

# ---------- 单位换算：SI → 传统(MIMIC) ----------
CONV = {'creatinine_max': ('÷88.4 (μmol/L→mg/dL)', lambda s: s / 88.4),
        'bun_max':        ('÷0.357 (mmol/L→mg/dL)', lambda s: s / 0.357),
        'glucose_max':    ('×18 (mmol/L→mg/dL)',    lambda s: s * 18.0),
        'hemoglobin_min': ('÷10 (g/L→g/dL)',        lambda s: s / 10.0)}
print('\n单位换算（本地 SI → 模型传统单位）:')
for c, (desc, f) in CONV.items():
    before = dfx[c].median()
    dfx[c] = f(dfx[c])
    print(f'  {c:16s} {desc:26s} 中位 {before:.2f} → {dfx[c].median():.2f}'
          f'  (模型训练中位 {medians.get(c)})')

# ---------- 冻结预处理：训练集中位数插补 + 标准化 ----------
X = dfx[feats].copy()
for c in cont:
    X[c] = X[c].fillna(medians[c])
for c in feats:
    if c not in cont:
        X[c] = X[c].fillna(0)
Xs = X.copy()
Xs[cont] = scaler.transform(X[cont])

print(f'\n可用于验证: n = {len(dfx)} | 院内死亡 = {int(dfx["y"].sum())} '
      f'({100*dfx["y"].mean():.1f}%)')

# ---------- 打分 & 指标 ----------
def metrics(p, y, name):
    auc = roc_auc_score(y, p)
    # bootstrap CI
    rng = np.random.default_rng(42); bs = []
    yv, pv = np.asarray(y), np.asarray(p)
    for _ in range(2000):
        i = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[i])) < 2: continue
        bs.append(roc_auc_score(yv[i], pv[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    oe = y.mean() / p.mean()
    brier = brier_score_loss(y, p)
    # 校准斜率/截距
    from sklearn.linear_model import LogisticRegression
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    cal = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(lp.reshape(-1, 1), y)
    print(f'\n  ▸ {name}')
    print(f'    AUC = {auc:.3f}  (95%CI {lo:.3f}–{hi:.3f})')
    print(f'    观察死亡率 = {100*y.mean():.1f}% | 平均预测 = {100*p.mean():.1f}% | O:E = {oe:.2f}')
    print(f'    Brier = {brier:.3f} | 校准斜率 = {cal.coef_[0][0]:.2f} | 校准截距 = {cal.intercept_[0]:.2f}')
    return dict(AUC=float(auc), CI=[float(lo), float(hi)], obs=float(y.mean()),
                pred=float(p.mean()), OE=float(oe), brier=float(brier),
                slope=float(cal.coef_[0][0]), intercept=float(cal.intercept_[0]))

p_lr = m['lr'].predict_proba(Xs)[:, 1]          # LR 训练于标准化特征 → 必须标准化
p_xg = m['xg'].predict_proba(X[feats])[:, 1]     # XGB 训练于未标准化特征 → 不得标准化
res = {}
print('=' * 74); print('本地外部验证结果（结局 = 院内死亡）'); print('=' * 74)
res['Logistic'] = metrics(p_lr, dfx['y'].values, 'Logistic（MIMIC 训练内 AUC 0.845）')
res['XGBoost'] = metrics(p_xg, dfx['y'].values, 'XGBoost （MIMIC 训练内 AUC 0.875）')

# ---------- 校准图 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
ax1.plot([0, 1], [0, 1], '--', color='#9aa7b4', lw=1, label='Ideal calibration')
for p, lab, col in [(p_xg, 'XGBoost', '#c0392b'), (p_lr, 'Logistic', '#2a7ab0')]:
    q = pd.qcut(pd.Series(p), 5, duplicates='drop')
    g = pd.DataFrame({'p': p, 'y': dfx['y'].values}).groupby(q, observed=True).mean()
    ax1.plot(g['p'], g['y'], 'o-', color=col, lw=1.8, ms=6, label=lab)
ax1.set_xlabel('Mean predicted probability (quintile)'); ax1.set_ylabel('Observed mortality')
ax1.set_title('Calibration in the single-centre cohort (in-hospital death)', fontsize=11)
ax1.legend(); ax1.grid(alpha=.25)

from sklearn.metrics import roc_curve
for p, lab, col in [(p_xg, f'XGBoost AUC={res["XGBoost"]["AUC"]:.3f}', '#c0392b'),
                    (p_lr, f'Logistic AUC={res["Logistic"]["AUC"]:.3f}', '#2a7ab0')]:
    fpr, tpr, _ = roc_curve(dfx['y'], p)
    ax2.plot(fpr, tpr, color=col, lw=2, label=lab)
ax2.plot([0, 1], [0, 1], '--', color='#9aa7b4', lw=1)
ax2.set_xlabel('1 - Specificity'); ax2.set_ylabel('Sensitivity'); ax2.set_title('ROC curves (single-centre cohort)', fontsize=11)
ax2.legend(loc='lower right'); ax2.grid(alpha=.25)
plt.tight_layout(); plt.savefig(OUTF, dpi=300, bbox_inches='tight')
print(f'\nWROTE {OUTF}')

res['n'] = int(len(dfx)); res['events'] = int(dfx['y'].sum())
res['unit_conversion'] = {k: v[0] for k, v in CONV.items()}
json.dump(res, open(OUTJ, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'WROTE {OUTJ}')
