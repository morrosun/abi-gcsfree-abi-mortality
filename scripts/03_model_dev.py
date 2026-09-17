# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | Model development on MIMIC-IV (1-year mortality)
Logistic (main) + LASSO selection + ML comparison + internal validation.
Saves fitted objects + train medians for external validation.
"""
import pandas as pd, numpy as np, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import statsmodels.api as sm

BASE = Path(ABI_BASE)
OUT = BASE/"output"; OUT.mkdir(exist_ok=True)
np.random.seed(42)

df = pd.read_csv(BASE/"data"/"mimic_abi_cohort.csv")

# ---------- feature set (GCS-independent, drop high-missing albumin/lactate/ph) ----------
aet = ['tbi','sah','ich','ais','cns_inf','seizure','anoxic']
cont = ['age','charlson_comorbidity_index','heart_rate_mean','mbp_mean','resp_rate_mean',
        'temperature_mean','spo2_mean','wbc_max','hemoglobin_min','platelets_min',
        'sodium_min','potassium_max','creatinine_max','bun_max','glucose_max',
        'bicarbonate_min','inr_max']
binv = ['mech_vent','vasopressor','rrt']
df['female'] = (df['gender']=='F').astype(int)
feats = ['age','female'] + aet + ['charlson_comorbidity_index'] + \
        [c for c in cont if c not in ('age','charlson_comorbidity_index')] + binv
y = df['death_365'].values
X = df[feats].copy()

# ---------- split (stratified) ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# ---------- impute with TRAIN medians (saved for external validation) ----------
medians = Xtr[cont].median()
for c in cont:
    Xtr[c] = Xtr[c].fillna(medians[c]); Xte[c] = Xte[c].fillna(medians[c])
medians.to_json(OUT/"train_medians.json")

# ---------- standardize continuous (for logistic/lasso; save scaler) ----------
scaler = StandardScaler().fit(Xtr[cont])
Xtr_s = Xtr.copy(); Xte_s = Xte.copy()
Xtr_s[cont] = scaler.transform(Xtr[cont]); Xte_s[cont] = scaler.transform(Xte[cont])
joblib.dump({'scaler':scaler,'cont':cont,'feats':feats,'medians':medians.to_dict()},
            OUT/"preproc.joblib")

def boot_auc(y_true, y_prob, n=1000):
    rng = np.random.default_rng(1); aucs=[]
    y_true=np.asarray(y_true); y_prob=np.asarray(y_prob); idx=np.arange(len(y_true))
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(y_true[b]))<2: continue
        aucs.append(roc_auc_score(y_true[b], y_prob[b]))
    return np.percentile(aucs,[2.5,97.5])

results={}; roc_store={}; calib_store={}

# ===== 1. Logistic main (statsmodels for OR) =====
Xtr_sm = sm.add_constant(Xtr_s[feats])
logit = sm.Logit(ytr, Xtr_sm).fit(disp=0, maxiter=200)
or_tab = pd.DataFrame({'coef':logit.params,'OR':np.exp(logit.params),
    'CI_low':np.exp(logit.conf_int()[0]),'CI_high':np.exp(logit.conf_int()[1]),
    'p':logit.pvalues}).round(4)
or_tab.to_csv(OUT/"logit_OR.csv")
p_lr = logit.predict(sm.add_constant(Xte_s[feats]))
auc=roc_auc_score(yte,p_lr); ci=boot_auc(yte,p_lr)
results['Logistic']={'auc':auc,'ci':list(ci),'brier':brier_score_loss(yte,p_lr)}
fpr,tpr,_=roc_curve(yte,p_lr); roc_store['Logistic']=(fpr,tpr)
calib_store['Logistic']=calibration_curve(yte,p_lr,n_bins=10,strategy='quantile')

# ===== 2. LASSO logistic (variable selection) =====
lasso=LogisticRegressionCV(Cs=20,cv=5,penalty='l1',solver='saga',
    scoring='roc_auc',max_iter=5000,random_state=42).fit(Xtr_s[feats],ytr)
p_la=lasso.predict_proba(Xte_s[feats])[:,1]
sel=pd.Series(lasso.coef_[0],index=feats)
sel[sel!=0].round(4).to_csv(OUT/"lasso_selected.csv",header=['coef'])
results['LASSO']={'auc':roc_auc_score(yte,p_la),'ci':list(boot_auc(yte,p_la)),
    'brier':brier_score_loss(yte,p_la),'n_selected':int((sel!=0).sum())}

# ===== 3. Random Forest =====
rf=RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_leaf=20,
    n_jobs=-1,random_state=42).fit(Xtr[feats],ytr)
p_rf=rf.predict_proba(Xte[feats])[:,1]
results['RandomForest']={'auc':roc_auc_score(yte,p_rf),'ci':list(boot_auc(yte,p_rf)),
    'brier':brier_score_loss(yte,p_rf)}
fpr,tpr,_=roc_curve(yte,p_rf); roc_store['RandomForest']=(fpr,tpr)

# ===== 4. XGBoost =====
xgbm=xgb.XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',random_state=42,
    scale_pos_weight=1).fit(Xtr[feats],ytr)
p_xg=xgbm.predict_proba(Xte[feats])[:,1]
auc_xg=roc_auc_score(yte,p_xg)
results['XGBoost']={'auc':auc_xg,'ci':list(boot_auc(yte,p_xg)),
    'brier':brier_score_loss(yte,p_xg)}
fpr,tpr,_=roc_curve(yte,p_xg); roc_store['XGBoost']=(fpr,tpr)
calib_store['XGBoost']=calibration_curve(yte,p_xg,n_bins=10,strategy='quantile')

# ===== 5. GBM =====
gbm=GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,
    random_state=42).fit(Xtr[feats],ytr)
p_gb=gbm.predict_proba(Xte[feats])[:,1]
results['GBM']={'auc':roc_auc_score(yte,p_gb),'ci':list(boot_auc(yte,p_gb)),
    'brier':brier_score_loss(yte,p_gb)}

# save models
joblib.dump({'logit_params':logit.params.to_dict(),'lasso':lasso,'rf':rf,
    'xgb':xgbm,'gbm':gbm},OUT/"models.joblib")

# ---------- performance table ----------
perf=pd.DataFrame(results).T
perf['AUC (95% CI)']=[f"{results[m]['auc']:.3f} ({results[m]['ci'][0]:.3f}-{results[m]['ci'][1]:.3f})" for m in perf.index]
perf['Brier']=[f"{results[m]['brier']:.3f}" for m in perf.index]
perf[['AUC (95% CI)','Brier']].to_csv(OUT/"model_performance.csv")
print("=== Internal validation (30% hold-out) ===")
print(perf[['AUC (95% CI)','Brier']].to_string())

# ---------- DCA data (net benefit) for best model ----------
def net_benefit(y_true,y_prob,thresholds):
    y_true=np.asarray(y_true); N=len(y_true); nb=[]
    for pt in thresholds:
        pred=y_prob>=pt
        tp=np.sum((pred==1)&(y_true==1)); fp=np.sum((pred==1)&(y_true==0))
        nb.append(tp/N-(fp/N)*(pt/(1-pt)))
    return np.array(nb)
th=np.arange(0.01,0.80,0.01)
prev=yte.mean()
dca=pd.DataFrame({'threshold':th,'model':net_benefit(yte,p_lr,th),
    'treat_all':prev-(1-prev)*(th/(1-th)),'treat_none':0})
dca.to_csv(OUT/"dca_logistic.csv",index=False)

# save ROC + calibration for plotting
np.savez(OUT/"roc_calib.npz",
    **{f"roc_{k}_fpr":v[0] for k,v in roc_store.items()},
    **{f"roc_{k}_tpr":v[1] for k,v in roc_store.items()},
    **{f"cal_{k}_true":v[0] for k,v in calib_store.items()},
    **{f"cal_{k}_pred":v[1] for k,v in calib_store.items()})
print("\nSaved: logit_OR.csv, lasso_selected.csv, model_performance.csv, dca, models.joblib")
print(f"Train n={len(ytr)} (events={ytr.sum()}), Test n={len(yte)} (events={yte.sum()})")
