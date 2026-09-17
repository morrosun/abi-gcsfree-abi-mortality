# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | External validation on eICU / NWICU / INSPIRE
Applies the frozen MIMIC preprocessing (train medians + scaler) and models
(Logistic main + XGBoost). Charlson computed uniformly (Quan-Deyo + age points).
"""
import pandas as pd, numpy as np, joblib, re
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

BASE = Path(ABI_BASE)
OUT = BASE/"output"; DATA = BASE/"data"

pp = joblib.load(OUT/"preproc.joblib")
models = joblib.load(OUT/"models.joblib")
feats, cont = pp['feats'], pp['cont']
scaler = pp['scaler']
medians = pd.Series(pp['medians'])
logit = pd.Series(models['logit_params'])
xgbm = models['xgb']
aet = ['tbi','sah','ich','ais','cns_inf','seizure','anoxic']
binv = ['mech_vent','vasopressor','rrt']

def sigmoid(x): return 1/(1+np.exp(-x))

# ---------- Charlson (Quan 2005 ICD-10, + ICD-9 fallback) + age points ----------
ICD10 = {
 'mi':(['I21','I22','I252'],1),
 'chf':(['I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290'],1),
 'pvd':(['I70','I71','I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959'],1),
 'cevd':(['G45','G46','H340','I60','I61','I62','I63','I64','I65','I66','I67','I68','I69'],1),
 'dementia':(['F00','F01','F02','F03','F051','G30','G311'],1),
 'copd':(['J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','I278','I279','J684','J701','J703'],1),
 'rheum':(['M05','M06','M32','M33','M34','M315','M351','M353','M360'],1),
 'pud':(['K25','K26','K27','K28'],1),
 'mildliver':(['B18','K70','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944'],1),
 'dm':(['E100','E101','E106','E108','E109','E110','E111','E116','E118','E119','E120','E121','E126','E128','E129','E130','E131','E136','E138','E139','E140','E141','E146','E148','E149'],1),
 'dmcx':(['E102','E103','E104','E105','E107','E112','E113','E114','E115','E117','E122','E123','E124','E125','E127','E132','E133','E134','E135','E137','E142','E143','E144','E145','E147'],2),
 'plegia':(['G041','G114','G801','G802','G81','G82','G830','G831','G832','G833','G834','G839'],2),
 'renal':(['I120','I131','N032','N033','N034','N035','N036','N037','N052','N053','N054','N055','N056','N057','N18','N19','N250','Z490','Z491','Z492','Z940','Z992'],2),
 'malig':(['C0','C1','C2','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C5','C6','C70','C71','C72','C73','C74','C75','C76','C81','C82','C83','C84','C85','C88','C90','C91','C92','C93','C94','C95','C96','C97'],2),
 'sevliver':(['I850','I859','I864','I982','K704','K711','K721','K729','K765','K766','K767'],3),
 'mets':(['C77','C78','C79','C80'],6),
 'hiv':(['B20','B21','B22','B24'],6),
}
def norm(code):
    return re.sub(r'[^A-Z0-9]','',str(code).upper())
def charlson_score(codes):
    codes = [norm(c) for c in codes if c and str(c)!='nan']
    score, seen = 0, set()
    for name,(prefixes,w) in ICD10.items():
        for c in codes:
            if any(c.startswith(p) for p in prefixes):
                seen.add(name); break
    # hierarchy: severe overrides mild (Quan) -- keep both per MIMIC derived (additive), so just sum
    for name in seen:
        score += ICD10[name][1]
    return score
def age_points(a):
    if a<50: return 0
    if a<60: return 1
    if a<70: return 2
    if a<80: return 3
    return 4

def charlson_from_diag(diag, idcol, codecol, cohort_ids, ages):
    g = diag.groupby(idcol)[codecol].apply(list)
    out = {}
    for i in cohort_ids:
        cs = charlson_score(g.get(i, []))
        out[i] = cs
    return out

def evaluate(df, y, tag):
    X = df[feats].copy()
    # impute cont with TRAIN medians; binaries/aet already 0/1
    for c in cont:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(medians[c])
    for c in aet+binv:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype(int)
    # standardized copy for logistic
    Xs = X.copy()
    Xs[cont] = scaler.transform(X[cont])
    lp = logit['const'] + Xs[feats].values @ logit[feats].values
    p_lr = sigmoid(lp)
    p_xg = xgbm.predict_proba(X[feats].values)[:,1]
    res = {}
    for name,p in [('Logistic',p_lr),('XGBoost',p_xg)]:
        auc = roc_auc_score(y,p)
        ci = boot_auc(y,p)
        br = brier_score_loss(y,p)
        # calibration slope/intercept
        eps=1e-6; lo=np.log(np.clip(p,eps,1-eps)/(1-np.clip(p,eps,1-eps)))
        lr = LogisticRegression(fit_intercept=True, C=1e6, solver='lbfgs', max_iter=1000).fit(lo.reshape(-1,1),y)
        slope=float(lr.coef_[0][0]); inter=float(lr.intercept_[0])
        res[name]={'auc':auc,'ci':ci,'brier':br,'slope':slope,'intercept':inter}
    res['_meta']={'n':int(len(y)),'events':int(y.sum()),'prev':float(y.mean())}
    return res, p_lr

def boot_auc(y,p,n=1000):
    rng=np.random.default_rng(7); y=np.asarray(y); p=np.asarray(p); idx=np.arange(len(y)); a=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(y[b]))<2: continue
        a.append(roc_auc_score(y[b],p[b]))
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]

def female_from(series):
    return series.astype(str).str.upper().str.startswith('F').astype(int)

all_res={}; roc_export={}

# ===== eICU (in-hospital) =====
e = pd.read_csv(DATA/"eicu_abi_cohort.csv")
ed = pd.read_csv(DATA/"eicu_diag.csv")
# split multi-code icd strings
ed['code']=ed['icd9code'].astype(str).str.split(',')
ed=ed.explode('code')
e['female']=female_from(e['gender'])
ch=charlson_from_diag(ed,'patientunitstayid','code',e['patientunitstayid'].tolist(),None)
e['charlson_comorbidity_index']=e['patientunitstayid'].map(ch)+e['age'].apply(age_points)
all_res['eICU (in-hospital)'],_=evaluate(e, e['death_hosp'].values, 'eicu')

# ===== NWICU (1-year) =====
n = pd.read_csv(DATA/"nwicu_abi_cohort.csv")
nd = pd.read_csv(DATA/"nwicu_diag.csv")
n['female']=female_from(n['gender'])
ch=charlson_from_diag(nd,'hadm_id','icd_code',n['hadm_id'].tolist(),None)
n['charlson_comorbidity_index']=n['hadm_id'].map(ch)+n['age'].apply(age_points)
all_res['NWICU (1-year)'],p_nw=evaluate(n, n['death_365'].values, 'nwicu')

# ===== INSPIRE (1-year) =====
ins = pd.read_csv(DATA/"inspire_abi_cohort.csv")
ind = pd.read_csv(DATA/"inspire_diag.csv")
ins['female']=female_from(ins['sex'])
ch=charlson_from_diag(ind,'subject_id','icd10_cm',ins['subject_id'].tolist(),None)
ins['charlson_comorbidity_index']=ins['subject_id'].map(ch)+ins['age'].apply(age_points)
all_res['INSPIRE (1-year)'],p_in=evaluate(ins, ins['death_365'].values, 'inspire')

# ---------- report ----------
rows=[]
for db,r in all_res.items():
    for m in ['Logistic','XGBoost']:
        x=r[m]
        rows.append({'Database':db,'Model':m,'N':r['_meta']['n'],'Events':r['_meta']['events'],
            'Prevalence':f"{r['_meta']['prev']*100:.1f}%",
            'AUC (95% CI)':f"{x['auc']:.3f} ({x['ci'][0]:.3f}-{x['ci'][1]:.3f})",
            'Brier':f"{x['brier']:.3f}",'Cal.slope':f"{x['slope']:.2f}",'Cal.intercept':f"{x['intercept']:.2f}"})
tab=pd.DataFrame(rows)
tab.to_csv(OUT/"external_validation.csv",index=False)
print(tab.to_string(index=False))

# save predictions for figures
np.savez(OUT/"external_preds.npz",
    nw_y=n['death_365'].values, nw_p=p_nw,
    in_y=ins['death_365'].values, in_p=p_in)
# save enriched cohorts (with charlson+female) for later Table1
e.to_csv(DATA/"eicu_abi_final.csv",index=False)
n.to_csv(DATA/"nwicu_abi_final.csv",index=False)
ins.to_csv(DATA/"inspire_abi_final.csv",index=False)
print("\nSaved external_validation.csv + external_preds.npz + *_final.csv")
