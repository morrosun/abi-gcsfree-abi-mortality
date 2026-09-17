# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | Recalibration figures: calibration curves before vs after (CV logistic recal),
plus ECE/Brier improvement summary."""
import numpy as np, joblib, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

BASE = Path(ABI_BASE); OUT=BASE/"output"
plt.rcParams.update({'font.size':10,'axes.linewidth':0.8,'font.family':'DejaVu Sans'})
store=joblib.load(OUT/"recal_preds.joblib")
tab=pd.read_csv(OUT/"recalibration.csv")

dbs=['eICU (in-hospital)','NWICU (1-year)','INSPIRE (1-year)']
mods=['Logistic','XGBoost']
C_ORIG='#c0392b'; C_RECAL='#1f77b4'

# ---------- Fig 9: 2x3 calibration curves before vs after ----------
fig,axes=plt.subplots(2,3,figsize=(13.5,9))
for i,m in enumerate(mods):
    for j,db in enumerate(dbs):
        ax=axes[i,j]; d=store[(db,m)]; y=d['y']
        ax.plot([0,1],[0,1],'--',color='#888',lw=1,zorder=1)
        for p,lab,col in [(d['orig'],'Original',C_ORIG),(d['log'],'Recalibrated',C_RECAL)]:
            nb=8
            frac,mean=calibration_curve(y,p,n_bins=nb,strategy='quantile')
            ax.plot(mean,frac,'o-',color=col,lw=1.8,ms=5,label=lab,zorder=3)
        row=tab[(tab.Database==db)&(tab.Model==m)].iloc[0]
        ax.text(0.03,0.97,f"Brier {row.Brier_orig}\u2192{row.Brier_log}\nECE {row.ECE_orig}\u2192{row.ECE_log}",
                transform=ax.transAxes,va='top',ha='left',fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='#ccc',alpha=0.9))
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"{db} \u00b7 {m}",fontsize=10,fontweight='bold')
        if j==0: ax.set_ylabel('Observed frequency')
        if i==1: ax.set_xlabel('Predicted probability')
        if i==0 and j==0: ax.legend(loc='lower right',fontsize=8.5,frameon=True)
        ax.grid(alpha=0.25,lw=0.5)
fig.suptitle('External calibration before vs after logistic recalibration (5-fold CV)',
             fontsize=12.5,fontweight='bold',y=0.995)
fig.tight_layout(rect=[0,0,1,0.98])
fig.savefig(OUT/"fig9_recal_calibration.png",dpi=200,bbox_inches='tight'); plt.close(fig)

# ---------- Fig 10: ECE before/after grouped bar ----------
fig,ax=plt.subplots(figsize=(11,5.2))
labels=[f"{db.split(' (')[0]}\n{m}" for db in dbs for m in mods]
eo=[float(tab[(tab.Database==db)&(tab.Model==m)].ECE_orig) for db in dbs for m in mods]
ei=[float(tab[(tab.Database==db)&(tab.Model==m)].ECE_int) for db in dbs for m in mods]
el=[float(tab[(tab.Database==db)&(tab.Model==m)].ECE_log) for db in dbs for m in mods]
x=np.arange(len(labels)); w=0.26
ax.bar(x-w,eo,w,label='Original',color='#c0392b')
ax.bar(x  ,ei,w,label='Intercept-only',color='#e6a817')
ax.bar(x+w,el,w,label='Logistic recal.',color='#1f77b4')
for xi,(a,b,c) in enumerate(zip(eo,ei,el)):
    for off,v in [(-w,a),(0,b),(w,c)]:
        ax.text(xi+off,v+0.003,f"{v:.3f}",ha='center',va='bottom',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylabel('Expected Calibration Error (ECE)')
ax.set_title('Calibration error before vs after recalibration',fontsize=12,fontweight='bold')
ax.legend(frameon=True); ax.grid(axis='y',alpha=0.25,lw=0.5)
ax.set_ylim(0,max(eo)*1.18)
fig.tight_layout()
fig.savefig(OUT/"fig10_recal_ece.png",dpi=200,bbox_inches='tight'); plt.close(fig)
print("Saved fig9_recal_calibration.png + fig10_recal_ece.png")
