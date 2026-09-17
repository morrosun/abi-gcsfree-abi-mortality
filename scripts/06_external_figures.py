# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | External validation figures: ROC (3 DBs) + calibration (1-yr DBs)."""
import pandas as pd, numpy as np, joblib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

BASE=Path(ABI_BASE); OUT=BASE/"output"; DATA=BASE/"data"
plt.rcParams.update({'font.size':11,'axes.linewidth':0.8})

pp=joblib.load(OUT/"preproc.joblib"); models=joblib.load(OUT/"models.joblib")
feats,cont=pp['feats'],pp['cont']; scaler=pp['scaler']; medians=pd.Series(pp['medians'])
logit=pd.Series(models['logit_params']); xgbm=models['xgb']
aet=['tbi','sah','ich','ais','cns_inf','seizure','anoxic']; binv=['mech_vent','vasopressor','rrt']
def sig(x): return 1/(1+np.exp(-x))
def preds(df):
    X=df[feats].copy()
    for c in cont: X[c]=pd.to_numeric(X[c],errors='coerce').fillna(medians[c])
    for c in aet+binv: X[c]=pd.to_numeric(X[c],errors='coerce').fillna(0).astype(int)
    Xs=X.copy(); Xs[cont]=scaler.transform(X[cont])
    p_lr=sig(logit['const']+Xs[feats].values@logit[feats].values)
    p_xg=xgbm.predict_proba(X[feats].values)[:,1]
    return p_lr,p_xg

dbs=[('eICU (in-hospital)','eicu_abi_final.csv','death_hosp','#378ADD'),
     ('NWICU (1-year)','nwicu_abi_final.csv','death_365','#1D9E75'),
     ('INSPIRE (1-year)','inspire_abi_final.csv','death_365','#7F77DD')]

# ---- Fig A: ROC (XGBoost, primary) across 3 external DBs ----
fig,ax=plt.subplots(figsize=(5.2,5.2))
for name,fn,ycol,col in dbs:
    d=pd.read_csv(DATA/fn); y=d[ycol].values; _,pxg=preds(d)
    fpr,tpr,_=roc_curve(y,pxg); auc=roc_auc_score(y,pxg)
    ax.plot(fpr,tpr,color=col,lw=2,label=f"{name}: AUC {auc:.3f}")
ax.plot([0,1],[0,1],'--',color='#888780',lw=1)
ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title('External validation ROC (XGBoost)')
ax.legend(loc='lower right',fontsize=9,frameon=False); ax.set_aspect('equal')
plt.tight_layout(); plt.savefig(OUT/"fig6_ext_roc.png",dpi=200); plt.close()

# ---- Fig B: ROC compare Logistic vs XGBoost per DB (small multiples) ----
fig,axes=plt.subplots(1,3,figsize=(13,4.3))
for ax,(name,fn,ycol,col) in zip(axes,dbs):
    d=pd.read_csv(DATA/fn); y=d[ycol].values; plr,pxg=preds(d)
    for p,c,lab in [(plr,'#A32D2D','Logistic'),(pxg,'#185FA5','XGBoost')]:
        fpr,tpr,_=roc_curve(y,p); ax.plot(fpr,tpr,color=c,lw=2,label=f"{lab} {roc_auc_score(y,p):.3f}")
    ax.plot([0,1],[0,1],'--',color='#888780',lw=1)
    ax.set_title(name,fontsize=11); ax.set_xlabel('1 - Specificity')
    ax.legend(loc='lower right',fontsize=9,frameon=False); ax.set_aspect('equal')
axes[0].set_ylabel('Sensitivity')
plt.tight_layout(); plt.savefig(OUT/"fig7_ext_roc_bymodel.png",dpi=200); plt.close()

# ---- Fig C: calibration for 1-year DBs (XGBoost) ----
fig,ax=plt.subplots(figsize=(5.2,5.2))
for name,fn,ycol,col in dbs[1:]:
    d=pd.read_csv(DATA/fn); y=d[ycol].values; _,pxg=preds(d)
    ct,cp=calibration_curve(y,pxg,n_bins=8,strategy='quantile')
    ax.plot(cp,ct,'o-',color=col,lw=1.8,ms=5,label=name)
ax.plot([0,1],[0,1],'--',color='#888780',lw=1)
ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed frequency')
ax.set_title('Calibration on 1-year mortality (XGBoost)')
ax.legend(loc='upper left',fontsize=9,frameon=False)
plt.tight_layout(); plt.savefig(OUT/"fig8_ext_calibration.png",dpi=200); plt.close()
print("Saved fig6_ext_roc / fig7_ext_roc_bymodel / fig8_ext_calibration")
