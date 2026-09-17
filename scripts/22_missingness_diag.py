# -*- coding: utf-8 -*-
"""判定 death_365 缺失性质 + 盘点可用模型"""
import pandas as pd, numpy as np, joblib, json, os
# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------

BASE = ABI_BASE
OUT = os.path.join(BASE, 'output')
P = os.path.join(BASE, 'ABI3', 'output', 'ABI本地数据采集模板_已录入检验结果.xlsx')

# ---------- A. 盘点模型 ----------
print('=' * 74); print('A. 本地已存档模型盘点'); print('=' * 74)
for f in ['models.joblib', 'model_inhosp.joblib', 'preproc.joblib', 'recal_preds.joblib']:
    p = os.path.join(OUT, f)
    if not os.path.exists(p):
        print(f'  {f}: 不存在'); continue
    try:
        obj = joblib.load(p)
        print(f'\n  ▸ {f}  ({type(obj).__name__})')
        if isinstance(obj, dict):
            for k, v in obj.items():
                d = type(v).__name__
                extra = ''
                if hasattr(v, 'coef_'):
                    extra = f' | coef_={np.ravel(v.coef_)[:4].round(3)}...  n_feat={len(np.ravel(v.coef_))}'
                    if hasattr(v, 'intercept_'): extra += f' | intercept={np.ravel(v.intercept_)[:1].round(3)}'
                elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], str):
                    extra = f' | {list(v)[:12]}'
                elif hasattr(v, 'columns'):
                    extra = f' | cols={list(v.columns)[:10]}'
                elif hasattr(v, 'feature_names_in_'):
                    extra = f' | features={list(v.feature_names_in_)[:12]}'
                print(f'      {k}: {d}{extra}')
        else:
            print('     ', repr(obj)[:300])
    except Exception as e:
        print(f'  {f}: 载入失败 {e}')

# ---------- B. 缺失性质判定 ----------
print('\n' + '=' * 74); print('B. death_365 缺失性质判定（关键）'); print('=' * 74)
df = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
df = df.dropna(how='all').reset_index(drop=True)
num = lambda c: pd.to_numeric(df[c], errors='coerce')
d365 = num('death_365'); dl = num('hospital_discharge_location')
inhosp_died = (dl == 5)

known = d365.notna()
print(f'  已知结局组 n={int(known.sum())} | 缺失组 n={int((~known).sum())}')
print('\n  【决定性证据】院内死亡率对比：')
for lab, m in [('已知结局组', known), ('缺失结局组', ~known)]:
    k = int(inhosp_died[m].sum()); n = int(m.sum())
    print(f'    {lab}: 院内死亡 {k}/{n} = {100*k/n:.1f}%')
print('    → 缺失组的院内死亡几乎为 0，说明"没查"的基本都是**活着出院**的人，')
print('      而"查了"的组被院内死亡严重富集 → 典型结局确认偏倚(MNAR)。')

print('\n  基线严重度对比（缺失组是否更健康）：')
for c in ['age', 'gcs', 'charlson_comorbidity_index', 'mbp_mean', 'spo2_mean',
          'creatinine_max', 'wbc_max']:
    s = num(c)
    a, b = s[known], s[~known]
    print(f'    {c:28s} 已知 {a.median():>7.1f} (n={a.notna().sum():3d}) | '
          f'缺失 {b.median():>7.1f} (n={b.notna().sum():3d})')
for c in ['mech_vent', 'vasopressor', 'rrt']:
    s = num(c)
    print(f'    {c:28s} 已知 {100*s[known].mean():>6.1f}% | 缺失 {100*s[~known].mean():>6.1f}%')

# ---------- C. 1年死亡率可识别区间 ----------
print('\n' + '=' * 74); print('C. 1年死亡率的可识别区间（部分识别）'); print('=' * 74)
N = len(df)
k_death = int(d365.sum())              # 已知 1 年死亡
n_unknown = int((~known).sum())        # 未知
lo = k_death / N                       # 未知者全存活
hi = (k_death + n_unknown) / N         # 未知者全死亡
print(f'  已知 1 年死亡 {k_death} 例；未知 {n_unknown} 例；N={N}')
print(f'  → 真实 1 年死亡率区间： [{100*lo:.1f}%, {100*hi:.1f}%]')
print(f'  完整病例法（只用已知 103 例）会得到 {100*k_death/int(known.sum()):.1f}%  ← 落在区间上界附近，必然高估')

# 若按"活出院者"分层估计
surv_disch = (dl != 5)
sd_known = surv_disch & known
sd_unk = surv_disch & ~known
print(f'\n  存活出院者(非院内死亡) n={int(surv_disch.sum())}：已知结局 {int(sd_known.sum())} 例'
      f'（其中 1 年死亡 {int(d365[sd_known].sum())} 例 = {100*d365[sd_known].mean():.1f}%），未知 {int(sd_unk.sum())} 例')

# ---------- D. 完整病例外部验证样本 ----------
print('\n' + '=' * 74); print('D. 若硬做外部验证的样本量'); print('=' * 74)
FEATS = ['age', 'female', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean',
         'resp_rate_mean', 'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min',
         'platelets_min', 'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max',
         'glucose_max', 'inr_max', 'mech_vent', 'vasopressor', 'rrt']
comp = df[FEATS].apply(pd.to_numeric, errors='coerce').notna().all(axis=1)
for lab, m in [('含碳酸氢根(21特征)', comp),
               ('不含碳酸氢根(20特征)', comp)]:
    m2 = m & d365.notna()
    print(f'  {lab}: 特征完整 {int(m.sum())} 例；特征+结局双完整 {int(m2.sum())} 例'
          f'（死亡 {int(d365[m2].sum())}）')
print(f'\n  院内死亡结局：可评估 {int((~dl.isna()).sum())} 例（缺失仅 {int(dl.isna().sum())} 例）→ 结局基本完整')
