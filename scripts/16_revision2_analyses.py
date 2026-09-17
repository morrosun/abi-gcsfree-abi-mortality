# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 REVISION ROUND 2 | must-clear analyses for the Minor-Revision critique.
 N2: eICU IN-HOSPITAL PARALLEL model calibration (O:E, ECE) pre/post recalibration,
     to replace the stale 1-year-model calibration numbers in Table 7.
 N3: GCS operationalization + GCS-alone discrimination sensitivity analyses
     (full test set, non-ventilated vs ventilated subgroup, motor-GCS, complete-case)
     to defend the very-low GCS-alone AUC (0.551).
 N4: event count in the lab complete-case subset (n=3,499).
All numbers come from local CSVs + frozen models; saved to output/revision2.json.
"""
import pandas as pd, numpy as np, json, sys, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
def log(*a): print(*a); sys.stdout.flush()

BASE=Path(ABI_BASE); OUT=BASE/"output"; DATA=BASE/"data"
np.random.seed(42)
rep={}

aet=['tbi','sah','ich','ais','cns_inf','seizure','anoxic']
cont=['age','charlson_comorbidity_index','heart_rate_mean','mbp_mean','resp_rate_mean',
      'temperature_mean','spo2_mean','wbc_max','hemoglobin_min','platelets_min',
      'sodium_min','potassium_max','creatinine_max','bun_max','glucose_max',
      'bicarbonate_min','inr_max']
binv=['mech_vent','vasopressor','rrt']
feats=['age','female']+aet+['charlson_comorbidity_index']+[c for c in cont if c not in ('age','charlson_comorbidity_index')]+binv

def ece(y,p,nb=10):
    y=np.asarray(y); p=np.asarray(p); e=0.0; N=len(y)
    edges=np.linspace(0,1,nb+1)
    for i in range(nb):
        m=(p>=edges[i])&(p<edges[i+1] if i<nb-1 else p<=edges[i+1])
        if m.sum()==0: continue
        e+=abs(y[m].mean()-p[m].mean())*m.sum()/N
    return float(e)

def oe(y,p): return float(np.mean(y)/np.mean(p))

def full_recal_cv(y,p,seed=42):
    """5-fold CV full logistic recalibration (slope+intercept); return recalibrated probs."""
    y=np.asarray(y); p=np.asarray(p); eps=1e-6
    lo=np.log(np.clip(p,eps,1-eps)/(1-np.clip(p,eps,1-eps))).reshape(-1,1)
    pr=np.zeros_like(p)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    for tr,te in skf.split(lo,y):
        m=LogisticRegression(C=1e6,max_iter=1000).fit(lo[tr],y[tr])
        pr[te]=m.predict_proba(lo[te])[:,1]
    return pr

# ============================================================ N2 eICU in-hosp calibration
log("="*60,"\nN2  eICU in-hospital parallel model calibration (O:E / ECE)")
mdl=joblib.load(OUT/"model_inhosp.joblib")
e=pd.read_csv(DATA/"eicu_abi_final.csv"); y=e['death_hosp'].values
uf=mdl['feats']; c_in=mdl['cont']
X=e[uf].copy()
for c in c_in: X[c]=pd.to_numeric(X[c],errors='coerce').fillna(mdl['medians'][c])
for c in [f for f in uf if f not in c_in]: X[c]=pd.to_numeric(X[c],errors='coerce').fillna(0)
Xs=X.copy(); Xs[c_in]=mdl['scaler'].transform(X[c_in])
preds={'Logistic':mdl['lr'].predict_proba(Xs[uf])[:,1],'XGBoost':mdl['xg'].predict_proba(X[uf])[:,1]}
n2={'obs_rate':round(100*float(np.mean(y)),1)}
for nm,p in preds.items():
    pr=full_recal_cv(y,p)
    n2[nm]={'auc':round(roc_auc_score(y,p),3),
            'brier_pre':round(brier_score_loss(y,p),3),'brier_post':round(brier_score_loss(y,pr),3),
            'oe_pre':round(oe(y,p),2),'oe_post':round(oe(y,pr),2),
            'ece_pre':round(ece(y,p),3),'ece_post':round(ece(y,pr),3),
            'exp_rate_pre':round(100*float(np.mean(p)),1)}
    log(f"  eICU {nm}: AUC {n2[nm]['auc']}  exp {n2[nm]['exp_rate_pre']}% vs obs {n2['obs_rate']}%  "
        f"O:E {n2[nm]['oe_pre']}->{n2[nm]['oe_post']}  ECE {n2[nm]['ece_pre']}->{n2[nm]['ece_post']}  "
        f"Brier {n2[nm]['brier_pre']}->{n2[nm]['brier_post']}")
rep['N2_eicu_inhosp_calibration']=n2

# ============================================================ N3 GCS operationalization + sensitivity
log("="*60,"\nN3  GCS-alone discrimination sensitivity")
m=pd.read_csv(DATA/"mimic_abi_cohort.csv"); m['female']=(m['gender']=='F').astype(int)
g=pd.read_csv(DATA/"mimic_gcs.csv"); m=m.merge(g,on='stay_id',how='left')
yb=m['death_365'].values
# same seed=42 30% test split used across the study
idx=np.arange(len(m))
tr,te=train_test_split(idx,test_size=0.30,stratify=yb,random_state=42)
gmed=float(np.nanmedian(m['gcs_min'].values[tr]))     # frozen training median for imputation
def gcs_auc(mask_bool, col='gcs_min', impute=True):
    sub=np.intersect1d(te, np.where(mask_bool)[0])
    yy=yb[sub]; gg=m[col].values[sub].astype(float)
    if impute: gg=np.where(np.isnan(gg), gmed, gg)
    else:
        keep=~np.isnan(gg); yy=yy[keep]; gg=gg[keep]
    if len(np.unique(yy))<2: return None,len(yy),int(yy.sum())
    return round(roc_auc_score(yy,-gg),3),int(len(yy)),int(yy.sum())  # lower GCS -> higher risk

vent=m['mech_vent'].values==1
full_auc,fn,fe=gcs_auc(np.ones(len(m),bool))
nv_auc,nvn,nve=gcs_auc(~vent)
v_auc,vn,ve=gcs_auc(vent)
motor_auc,mn,me=gcs_auc(np.ones(len(m),bool),col='gcs_motor')
cc_auc,ccn,cce=gcs_auc(np.ones(len(m),bool),impute=False)  # complete-case (drop missing GCS)
n3={'operationalization':{'source':'mimiciv_derived.first_day_gcs (ICU 首日最差/最低 GCS 总分)',
        'measure':'首日最差值(worst/min)','missing_rate_pct':round(100*float(m['gcs_min'].isna().mean()),2),
        'imputation':'训练集中位数插补','train_median':gmed,
        'gcs_min_median_iqr':[float(m['gcs_min'].median()),float(m['gcs_min'].quantile(.25)),float(m['gcs_min'].quantile(.75))],
        'vent_pct':round(100*float(m['mech_vent'].mean()),1)},
    'gcs_alone_auc':{'full_test':{'auc':full_auc,'n':fn,'events':fe},
        'nonventilated':{'auc':nv_auc,'n':nvn,'events':nve},
        'ventilated':{'auc':v_auc,'n':vn,'events':ve},
        'motor_gcs_full':{'auc':motor_auc,'n':mn,'events':me},
        'complete_case_full':{'auc':cc_auc,'n':ccn,'events':cce}}}
log(f"  GCS-alone AUC: full={full_auc} (n={fn}) | non-vent={nv_auc} (n={nvn}) | vent={v_auc} (n={vn})")
log(f"  motor-GCS full={motor_auc} | complete-case={cc_auc} | missing={n3['operationalization']['missing_rate_pct']}%")
rep['N3_gcs_sensitivity']=n3

# ============================================================ N4 lab complete-case event count
log("="*60,"\nN4  lab complete-case subset event count")
labcols=[c for c in ['albumin_min','lactate_max','ph_min','albumin','lactate','ph'] if c in m.columns]
log(f"  candidate lab cols present: {labcols}")
if labcols:
    cc=m.dropna(subset=labcols)
    n4={'lab_cols':labcols,'n':int(len(cc)),'events':int(cc['death_365'].sum()),
        'event_rate_pct':round(100*float(cc['death_365'].mean()),1)}
    log(f"  complete-case n={n4['n']} events={n4['events']} ({n4['event_rate_pct']}%)")
else:
    n4={'note':'lab columns not in cohort csv; see 10_internal_revision.py M5b subset'}
rep['N4_lab_completecase']=n4

json.dump(rep, open(OUT/"revision2.json","w"), indent=2, default=str)
log("\nSaved output/revision2.json")
