# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 REVISION | figures for must-fix analyses (light theme, English titles)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path(ABI_BASE + "/output")
plt.rcParams.update({'font.size':11,'axes.linewidth':0.8,'figure.dpi':150,
                     'font.family':'DejaVu Sans'})
ri=json.load(open(OUT/"revision_internal.json"))
re=json.load(open(OUT/"revision_external.json"))
C={'base':'#4C72B0','gcs':'#C44E52','xgb':'#55A868','lr':'#4C72B0','ref':'#8C8C8C'}

# ---------- Fig R1: GCS incremental value ----------
fig,ax=plt.subplots(figsize=(7,4.6))
g=ri['M1_gcs_increment']
groups=['Logistic','XGBoost']; x=np.arange(2); w=0.35
base=[g['Logistic']['auc_base'],g['XGBoost']['auc_base']]
gcs=[g['Logistic']['auc_gcs'],g['XGBoost']['auc_gcs']]
b1=ax.bar(x-w/2,base,w,label='Without GCS (main model)',color=C['base'])
b2=ax.bar(x+w/2,gcs,w,label='With GCS added',color=C['gcs'])
for xi,(bb,gg,gi) in enumerate(zip(base,gcs,['Logistic','XGBoost'])):
    ax.text(xi-w/2,bb+0.004,f'{bb:.3f}',ha='center',fontsize=9)
    ax.text(xi+w/2,gg+0.004,f'{gg:.3f}',ha='center',fontsize=9)
    d=g[gi]['dAUC']; ax.text(xi,max(bb,gg)+0.02,f'ΔAUC {d:+.3f}',ha='center',fontsize=9,color='#333')
ax.axhline(g['GCS_alone_auc'],ls='--',color=C['ref'],lw=1)
ax.text(1.45,g['GCS_alone_auc']+0.005,f"GCS alone {g['GCS_alone_auc']:.3f}",ha='right',fontsize=8.5,color=C['ref'])
ax.axhline(g['SOFA_alone_auc'],ls=':',color='#B07AA1',lw=1)
ax.text(1.45,g['SOFA_alone_auc']+0.005,f"SOFA alone {g['SOFA_alone_auc']:.3f}",ha='right',fontsize=8.5,color='#B07AA1')
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel('AUROC (30% hold-out)')
ax.set_ylim(0.5,0.90); ax.set_title('Incremental value of GCS for 1-year mortality')
ax.legend(loc='lower right',fontsize=9,frameon=False)
plt.tight_layout(); plt.savefig(OUT/"figR1_gcs_increment.png",bbox_inches='tight'); plt.close()

# ---------- Fig R2: in-hospital parallel validation (same outcome) ----------
fig,ax=plt.subplots(figsize=(7.2,4.6))
m2=re['M2_inhosp_parallel']
order=['MIMIC_internal','eICU','INSPIRE']  # NWICU excluded (only 16 in-hosp events)
labels=['MIMIC-IV\n(internal)','eICU','INSPIRE']
lr=[m2['MIMIC_internal']['Logistic']['auc'],m2['eICU']['Logistic']['auc'],m2['INSPIRE']['Logistic']['auc']]
xg=[m2['MIMIC_internal']['XGBoost']['auc'],m2['eICU']['XGBoost']['auc'],m2['INSPIRE']['XGBoost']['auc']]
x=np.arange(3); w=0.35
ax.bar(x-w/2,lr,w,label='Logistic',color=C['lr'])
ax.bar(x+w/2,xg,w,label='XGBoost',color=C['xgb'])
for xi,(a,b) in enumerate(zip(lr,xg)):
    ax.text(xi-w/2,a+0.005,f'{a:.3f}',ha='center',fontsize=9)
    ax.text(xi+w/2,b+0.005,f'{b:.3f}',ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('AUROC (in-hospital mortality)')
ax.set_ylim(0.5,0.95); ax.set_title('Parallel validation on a common outcome (in-hospital death)')
ax.legend(loc='lower left',fontsize=9,frameon=False)
ax.text(0.5,0.52,'NWICU excluded (only 16 in-hospital deaths; retained for 1-year validation)',
        fontsize=8,color='#777')
plt.tight_layout(); plt.savefig(OUT/"figR2_inhosp_parallel.png",bbox_inches='tight'); plt.close()

# ---------- Fig R3: root-cause panel (vasopressor prevalence + INSPIRE aetiology gap) ----------
fig,axes=plt.subplots(1,2,figsize=(10,4.4))
vp=re['M3_nwicu_slope']['vasopressor_prevalence']
dbs=list(vp.keys()); vals=[float(vp[k]) for k in dbs]
cols=['#4C72B0','#55A868','#C44E52','#DD8452']
axes[0].bar(range(len(dbs)),vals,color=cols)
for i,v in enumerate(vals): axes[0].text(i,v+0.4,f'{v:.1f}%',ha='center',fontsize=9)
axes[0].set_xticks(range(len(dbs))); axes[0].set_xticklabels(dbs,rotation=15,fontsize=9)
axes[0].set_ylabel('Vasopressor use (%)'); axes[0].set_ylim(0,28)
axes[0].set_title('(A) Vasopressor recording across databases')
# INSPIRE aetiology gap
aet=['tbi','sah','ich','ais','cns_inf','seizure','anoxic']
nw=re['M3_aetiology_strata']['NWICU']; ins=re['M3_aetiology_strata']['INSPIRE']
nwv=[nw[a]['n'] for a in aet]; inv=[ins[a]['n'] for a in aet]
x=np.arange(len(aet)); w=0.4
axes[1].bar(x-w/2,nwv,w,label='NWICU',color='#4C72B0')
axes[1].bar(x+w/2,inv,w,label='INSPIRE',color='#DD8452')
axes[1].set_xticks(x); axes[1].set_xticklabels([a.upper() for a in aet],rotation=40,fontsize=8)
axes[1].set_ylabel('Patients (n)'); axes[1].legend(fontsize=9,frameon=False)
axes[1].set_title('(B) Aetiology mix: INSPIRE lacks anoxic/seizure')
plt.tight_layout(); plt.savefig(OUT/"figR3_rootcause.png",bbox_inches='tight'); plt.close()

print("Saved figR1_gcs_increment.png, figR2_inhosp_parallel.png, figR3_rootcause.png")
