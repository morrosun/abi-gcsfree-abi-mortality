# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | Publication figures: ROC, calibration, DCA, OR forest"""
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spontaneous':False} if False else {'font.size':10})

BASE=Path(ABI_BASE); OUT=BASE/"output"
d=np.load(OUT/"roc_calib.npz")
perf=pd.read_csv(OUT/"model_performance.csv",index_col=0)

# ---- Fig1: ROC comparison ----
fig,ax=plt.subplots(figsize=(5.2,5))
colors={'Logistic':'#185FA5','XGBoost':'#A32D2D','RandomForest':'#0F6E56'}
for m in ['Logistic','RandomForest','XGBoost']:
    auc=perf.loc[m,'AUC (95% CI)']
    ax.plot(d[f'roc_{m}_fpr'],d[f'roc_{m}_tpr'],color=colors[m],lw=2,
            label=f"{m}: {auc}")
ax.plot([0,1],[0,1],'--',color='gray',lw=1)
ax.set_xlabel('1 - Specificity');ax.set_ylabel('Sensitivity')
ax.set_title('ROC: 1-year mortality (MIMIC-IV internal test)')
ax.legend(loc='lower right',fontsize=8.5);ax.set_aspect('equal')
plt.tight_layout();plt.savefig(OUT/"fig1_roc.png",dpi=200);plt.close()

# ---- Fig2: Calibration ----
fig,ax=plt.subplots(figsize=(5.2,5))
for m,c in [('Logistic','#185FA5'),('XGBoost','#A32D2D')]:
    ax.plot(d[f'cal_{m}_pred'],d[f'cal_{m}_true'],'o-',color=c,label=m,ms=5)
ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Ideal')
ax.set_xlabel('Predicted probability');ax.set_ylabel('Observed frequency')
ax.set_title('Calibration (10 quantile bins)')
ax.legend(fontsize=9);plt.tight_layout();plt.savefig(OUT/"fig2_calibration.png",dpi=200);plt.close()

# ---- Fig3: DCA ----
dca=pd.read_csv(OUT/"dca_logistic.csv")
fig,ax=plt.subplots(figsize=(5.6,4.6))
ax.plot(dca.threshold,dca.model,color='#185FA5',lw=2,label='Logistic model')
ax.plot(dca.threshold,dca.treat_all,color='#854F0B',lw=1,label='Treat all')
ax.plot(dca.threshold,dca.treat_none,color='gray',lw=1,ls='--',label='Treat none')
ax.set_ylim(-0.05,dca.model.max()*1.1);ax.set_xlim(0,0.8)
ax.set_xlabel('Threshold probability');ax.set_ylabel('Net benefit')
ax.set_title('Decision curve analysis');ax.legend(fontsize=9)
plt.tight_layout();plt.savefig(OUT/"fig3_dca.png",dpi=200);plt.close()

# ---- Fig4: OR forest (top significant) ----
orr=pd.read_csv(OUT/"logit_OR.csv",index_col=0)
orr=orr[orr.index!='const']
label_map={'anoxic':'Anoxic injury','mech_vent':'Mechanical ventilation','ich':'ICH',
 'charlson_comorbidity_index':'Charlson index','sah':'SAH','age':'Age','seizure':'Seizure/SE',
 'tbi':'TBI','resp_rate_mean':'Resp rate','heart_rate_mean':'Heart rate','bun_max':'BUN',
 'inr_max':'INR','wbc_max':'WBC','vasopressor':'Vasopressor','glucose_max':'Glucose',
 'hemoglobin_min':'Hemoglobin (low)','bicarbonate_min':'Bicarbonate (low)',
 'spo2_mean':'SpO2','temperature_mean':'Temperature'}
sig=orr[orr['p']<0.05].copy().sort_values('OR')
sig=sig[sig.index.isin(label_map)]
sig['lab']=[label_map[i] for i in sig.index]
fig,ax=plt.subplots(figsize=(6.2,7))
yv=np.arange(len(sig))
ax.errorbar(sig['OR'],yv,xerr=[sig['OR']-sig['CI_low'],sig['CI_high']-sig['OR']],
    fmt='o',color='#185FA5',ecolor='#85B7EB',capsize=3,ms=6)
ax.axvline(1,color='gray',ls='--',lw=1)
ax.set_yticks(yv);ax.set_yticklabels(sig['lab'],fontsize=9)
ax.set_xscale('log');ax.set_xlabel('Odds ratio (per 1-SD for continuous), log scale')
ax.set_title('Adjusted odds ratios for 1-year mortality')
plt.tight_layout();plt.savefig(OUT/"fig4_forest.png",dpi=200);plt.close()

# ---- Fig5: XGBoost feature importance ----
import joblib
mdl=joblib.load(OUT/"models.joblib")
pp=joblib.load(OUT/"preproc.joblib");feats=pp['feats']
imp=pd.Series(mdl['xgb'].feature_importances_,index=feats).sort_values()[-15:]
imp.index=[label_map.get(i,i) for i in imp.index]
fig,ax=plt.subplots(figsize=(5.8,6))
ax.barh(range(len(imp)),imp.values,color='#A32D2D')
ax.set_yticks(range(len(imp)));ax.set_yticklabels(imp.index,fontsize=9)
ax.set_xlabel('XGBoost feature importance (gain)')
ax.set_title('Top predictors (XGBoost)')
plt.tight_layout();plt.savefig(OUT/"fig5_xgb_importance.png",dpi=200);plt.close()
print("Saved fig1-5 to output/")
