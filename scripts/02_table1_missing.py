# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | MIMIC-IV baseline table (Table 1) + missingness report"""
import pandas as pd, numpy as np
from pathlib import Path

BASE = Path(ABI_BASE)
df = pd.read_csv(BASE/"data"/"mimic_abi_cohort.csv")
print(f"N = {len(df)}")

# ---- single aetiology label (priority order) ----
def aet(r):
    for k,lab in [('tbi','TBI'),('sah','SAH'),('ich','ICH'),('ais','Ischemic'),
                  ('anoxic','Anoxic'),('cns_inf','CNS_inf'),('seizure','Seizure')]:
        if r[k]==1: return lab
    return 'Other'
df['aetiology'] = df.apply(aet, axis=1)

cont = ['age','charlson_comorbidity_index','heart_rate_mean','sbp_mean','mbp_mean',
        'resp_rate_mean','temperature_mean','spo2_mean','wbc_max','hemoglobin_min',
        'platelets_min','sodium_min','potassium_max','creatinine_max','bun_max',
        'glucose_max','albumin_min','bicarbonate_min','inr_max','lactate_max','ph_min',
        'sofa_day1','icu_los_days']
binv = ['mech_vent','vasopressor','rrt']

# ---- missingness ----
miss = (df[cont+binv].isna().mean()*100).round(1).sort_values(ascending=False)
miss.to_csv(BASE/"output"/"missingness.csv", header=['missing_pct'])
print("\n=== Missingness (%) top ===")
print(miss.head(25).to_string())

# ---- Table 1 by 365-day mortality ----
g = df.groupby('death_365')
rows=[]
def fmt_c(s): 
    return f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]"
for v in cont:
    rows.append([v,'median[IQR]', fmt_c(df[v].dropna()),
                 fmt_c(g.get_group(0)[v].dropna()), fmt_c(g.get_group(1)[v].dropna())])
for v in binv:
    rows.append([v,'n(%)', f"{int(df[v].sum())} ({100*df[v].mean():.1f})",
                 f"{int(g.get_group(0)[v].sum())} ({100*g.get_group(0)[v].mean():.1f})",
                 f"{int(g.get_group(1)[v].sum())} ({100*g.get_group(1)[v].mean():.1f})"])
# gender + aetiology
for lab in ['F']:
    m=df['gender']=='F'
    rows.append(['female','n(%)',f"{m.sum()} ({100*m.mean():.1f})",
                 f"{(g.get_group(0)['gender']=='F').sum()}",
                 f"{(g.get_group(1)['gender']=='F').sum()}"])
t1 = pd.DataFrame(rows, columns=['variable','stat','overall',
                                 f'alive_1yr (n={len(g.get_group(0))})',
                                 f'dead_1yr (n={len(g.get_group(1))})'])
t1.to_csv(BASE/"output"/"table1.csv", index=False)
print("\n=== Table 1 (by 1-year mortality) ===")
print(t1.to_string(index=False))

# aetiology distribution + mortality
print("\n=== Aetiology: n and 1yr mortality ===")
print(df.groupby('aetiology').agg(n=('subject_id','count'),
      d365=('death_365','mean')).assign(d365=lambda x:(x.d365*100).round(1)).to_string())
print("\nOverall 1yr mortality: %.1f%%"%(100*df['death_365'].mean()))
