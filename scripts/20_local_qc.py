# -*- coding: utf-8 -*-
"""本地已录入数据 系统性质控 v2（稳健版）"""
import pandas as pd, numpy as np, json
# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------

P = ABI_BASE + "/ABI3/output/ABI本地数据采集模板_已录入检验结果.xlsx"
OUT = ABI_BASE + "/ABI3/output/local_qc_report.json"
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 60)

df = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
df = df.dropna(how='all').reset_index(drop=True)
N = len(df)
rep = {'N': N}

# ---------- 5'. 结局交叉核查 ----------
print('=' * 70); print('【5】结局一致性（交叉核查）'); print('=' * 70)
df['death_time'] = pd.to_datetime(df['death_time'], errors='coerce')
df['icu_in_time'] = pd.to_datetime(df['icu_in_time'], errors='coerce')
df['hospital_discharge_time'] = pd.to_datetime(df['hospital_discharge_time'], errors='coerce')
d365 = pd.to_numeric(df['death_365'], errors='coerce')
dl = pd.to_numeric(df['hospital_discharge_location'], errors='coerce')

print('  death_365：', d365.value_counts(dropna=False).to_dict())
print('  death_time 非空：', int(df['death_time'].notna().sum()))
print('  出院去向=5(院内死亡)：', int((dl == 5).sum()))
print('  存活组(female 无关)检查 —— 交叉表 death_365 × death_time是否存在：')
ct = pd.crosstab(d365.fillna('缺失'), df['death_time'].notna().map({True: '有death_time', False: '无death_time'}))
print(ct.to_string())
print('  交叉表 death_365 × 出院去向为5：')
ct2 = pd.crosstab(d365.fillna('缺失'), (dl == 5).map({True: '院内死亡', False: '非院内死亡'}))
print(ct2.to_string())

# 逻辑冲突
conf_a = ((d365 == 1) & df['death_time'].isna()).sum()          # 标死亡但无死亡日期
conf_b = ((d365 == 0) & df['death_time'].notna()).sum()          # 标存活但有死亡日期
conf_c = ((d365 == 0) & (dl == 5)).sum()                         # 标存活但院内死亡
print(f'  ⚠ 标死亡(1)但无死亡日期: {int(conf_a)} 例')
print(f'  ⚠ 标存活(0)但有死亡日期: {int(conf_b)} 例')
print(f'  ⚠ 标存活(0)但出院去向=院内死亡: {int(conf_c)} 例')
rep.update(conf_a=int(conf_a), conf_b=int(conf_b), conf_c=int(conf_c))

# 派生 death_365（需 death_time）
has = df['death_time'].notna() & df['icu_in_time'].notna()
der = pd.Series(np.nan, index=df.index)
der[has] = ((df.loc[has, 'death_time'] - df.loc[has, 'icu_in_time']).dt.total_seconds() <= 365 * 86400).astype(int)
both = d365.notna() & der.notna()
print(f'  可派生者: {int(der.notna().sum())} | 与录入值不一致: {int((d365[both] != der[both]).sum())}')
if (d365[both] != der[both]).sum():
    ex = df.loc[both & (d365 != der), ['patient_id', 'icu_in_time', 'death_time', 'death_365']].head(8)
    ex['派生值'] = der[both & (d365 != der)]
    print(ex.to_string())

# ---------- 6. 数值范围 ----------
print('\n' + '=' * 70); print('【6】数值范围（越界者）'); print('=' * 70)
RNG = {
    'heart_rate_mean': (20, 200), 'mbp_mean': (30, 200), 'resp_rate_mean': (5, 60),
    'temperature_mean': (32, 42), 'spo2_mean': (60, 100), 'wbc_max': (0.1, 100),
    'hemoglobin_min': (30, 200), 'platelets_min': (1, 1000), 'sodium_min': (100, 180),
    'potassium_max': (1.0, 9.0), 'creatinine_max': (10, 2000), 'bun_max': (1, 100),
    'glucose_max': (1, 50), 'bicarbonate_min': (5, 40), 'inr_max': (0.5, 15),
    'gcs': (3, 15), 'sofa_day1': (0, 24), 'charlson_comorbidity_index': (0, 24),
}
oor = {}
for c, (lo, hi) in RNG.items():
    if c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        bad = int((((s < lo) | (s > hi)) & s.notna()).sum())
        if bad:
            oor[c] = bad
            print(f'  {c}: {bad} 例越界 (允许 {lo}–{hi})，实际 {s.min()}–{s.max()}')
if not oor:
    print('  无越界')
rep['out_of_range'] = oor

# ---------- 7. 描述统计 ----------
print('\n' + '=' * 70); print('【7】关键连续变量描述'); print('=' * 70)
keys = ['age', 'charlson_comorbidity_index', 'gcs', 'heart_rate_mean', 'mbp_mean',
        'resp_rate_mean', 'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min',
        'platelets_min', 'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max',
        'glucose_max', 'bicarbonate_min', 'inr_max']
desc = df[keys].apply(pd.to_numeric, errors='coerce').describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
print(desc.round(2).to_string())
rep['describe'] = json.loads(desc.round(3).to_json(orient='index'))

# ---------- 8. 二分类 ----------
print('\n' + '=' * 70); print('【8】二分类与治疗措施'); print('=' * 70)
for c in ['mech_vent', 'vasopressor', 'rrt', 'female']:
    print(f'  {c}: {df[c].value_counts(dropna=False).to_dict()}')

# ---------- 9. 分层：主要病因（按优先级派生） ----------
print('\n' + '=' * 70); print('【9】主要病因（按模板互斥优先级派生）'); print('=' * 70)
PRIO = ['anoxic', 'ich', 'sah', 'ais', 'seizure', 'cns_inf', 'tbi']
etd = df[PRIO].fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
def main_etiology(row):
    for c in PRIO:
        if row[c] == 1:
            return c
    return '无'
df['main_etiology'] = etd.apply(main_etiology, axis=1)
print(df['main_etiology'].value_counts().to_string())
rep['main_etiology'] = df['main_etiology'].value_counts().to_dict()

# 各主要病因的 1 年死亡率
tmp = df.assign(d=d365)
g = tmp.groupby('main_etiology')['d'].agg(['count', 'sum', 'mean'])
print('\n  各主要病因 death_365（count=可评估数, sum=死亡数, mean=缺失率外的死亡比）:')
print(g.round(3).to_string())

json.dump(rep, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
print(f'\nWROTE {OUT}')
