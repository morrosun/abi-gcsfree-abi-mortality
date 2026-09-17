# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""A1 | Recalibration analysis for miscalibrated external cohorts
For each database x model, applies two standard recalibration methods:
  (A) Intercept-only update  (calibration-in-the-large / offset correction)
  (B) Logistic recalibration (intercept + slope; a.k.a. Cox / Platt)
Honest performance via 5-fold CV recalibration (fit on train folds, apply to held-out
fold) to avoid in-sample optimism. Reports Brier, calibration slope/intercept, O:E ratio,
and ECE before vs after. AUC is unchanged by monotonic recalibration (reported once).
"""
import pandas as pd, numpy as np, joblib, re
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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
EPS = 1e-6

def sigmoid(x): return 1/(1+np.exp(-x))
def logit_f(p):
    p = np.clip(p, EPS, 1-EPS); return np.log(p/(1-p))

# ---------- Charlson (Quan 2005 ICD-10) + age points (same as 05) ----------
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
def norm(code): return re.sub(r'[^A-Z0-9]','',str(code).upper())
def charlson_score(codes):
    codes=[norm(c) for c in codes if c and str(c)!='nan']; seen=set()
    for name,(prefixes,w) in ICD10.items():
        for c in codes:
            if any(c.startswith(p) for p in prefixes): seen.add(name); break
    return sum(ICD10[n][1] for n in seen)
def age_points(a):
    return 0 if a<50 else 1 if a<60 else 2 if a<70 else 3 if a<80 else 4
def charlson_from_diag(diag, idcol, codecol, ids):
    g=diag.groupby(idcol)[codecol].apply(list)
    return {i:charlson_score(g.get(i,[])) for i in ids}
def female_from(s): return s.astype(str).str.upper().str.startswith('F').astype(int)

def predict(df):
    """Return (p_logistic, p_xgb, lp_logistic) with frozen preprocessing."""
    X=df[feats].copy()
    for c in cont: X[c]=pd.to_numeric(X[c],errors='coerce').fillna(medians[c])
    for c in aet+binv: X[c]=pd.to_numeric(X[c],errors='coerce').fillna(0).astype(int)
    Xs=X.copy(); Xs[cont]=scaler.transform(X[cont])
    lp=logit['const']+Xs[feats].values@logit[feats].values
    p_lr=sigmoid(lp)
    p_xg=xgbm.predict_proba(X[feats].values)[:,1]
    return p_lr, p_xg, lp

# ---------- calibration metrics ----------
def cal_slope_intercept(y,p):
    lp=logit_f(p)
    m=LogisticRegression(fit_intercept=True,C=1e12,solver='lbfgs',max_iter=2000).fit(lp.reshape(-1,1),y)
    return float(m.coef_[0][0]), float(m.intercept_[0])
def oe_ratio(y,p): return float(y.mean()/p.mean())
def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); e=0.0; n=len(y)
    for i in range(bins):
        m=(p>=edges[i])&(p<edges[i+1] if i<bins-1 else p<=edges[i+1])
        if m.sum()==0: continue
        e+=abs(y[m].mean()-p[m].mean())*m.sum()/n
    return float(e)

# ---------- recalibration fitters (on a training subset) ----------
def fit_intercept_only(y,lp):
    # MLE intercept with slope fixed at 1: minimize deviance; Newton on scalar a
    a=0.0
    for _ in range(100):
        p=sigmoid(a+lp); g=np.sum(y-p); h=-np.sum(p*(1-p))
        if abs(h)<1e-12: break
        step=g/h; a-=step
        if abs(step)<1e-10: break
    return a  # p_new = sigmoid(a + lp)
def fit_logistic_recal(y,lp):
    m=LogisticRegression(fit_intercept=True,C=1e12,solver='lbfgs',max_iter=2000).fit(lp.reshape(-1,1),y)
    return float(m.intercept_[0]), float(m.coef_[0][0])  # a,b -> sigmoid(a+b*lp)

def cv_recalibrate(y,p,method,seed=7):
    """5-fold CV: fit recal on train folds, apply to held-out. Returns OOF recal probs."""
    lp=logit_f(p); oof=np.zeros_like(p,dtype=float)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    for tr,te in skf.split(lp,y):
        if method=='intercept':
            a=fit_intercept_only(y[tr],lp[tr]); oof[te]=sigmoid(a+lp[te])
        else:
            a,b=fit_logistic_recal(y[tr],lp[tr]); oof[te]=sigmoid(a+b*lp[te])
    return oof

def metrics(y,p):
    s,i=cal_slope_intercept(y,p)
    return {'brier':brier_score_loss(y,p),'slope':s,'intercept':i,
            'oe':oe_ratio(y,p),'ece':ece(y,np.asarray(p))}

# ================= build cohorts + predictions =================
cohorts={}
# eICU (in-hospital)
e=pd.read_csv(DATA/"eicu_abi_cohort.csv"); ed=pd.read_csv(DATA/"eicu_diag.csv")
ed['code']=ed['icd9code'].astype(str).str.split(','); ed=ed.explode('code')
e['female']=female_from(e['gender'])
ch=charlson_from_diag(ed,'patientunitstayid','code',e['patientunitstayid'].tolist())
e['charlson_comorbidity_index']=e['patientunitstayid'].map(ch)+e['age'].apply(age_points)
cohorts['eICU (in-hospital)']=(e,'death_hosp')
# NWICU (1-year)
n=pd.read_csv(DATA/"nwicu_abi_cohort.csv"); nd=pd.read_csv(DATA/"nwicu_diag.csv")
n['female']=female_from(n['gender'])
ch=charlson_from_diag(nd,'hadm_id','icd_code',n['hadm_id'].tolist())
n['charlson_comorbidity_index']=n['hadm_id'].map(ch)+n['age'].apply(age_points)
cohorts['NWICU (1-year)']=(n,'death_365')
# INSPIRE (1-year)
ins=pd.read_csv(DATA/"inspire_abi_cohort.csv"); ind=pd.read_csv(DATA/"inspire_diag.csv")
ins['female']=female_from(ins['sex'])
ch=charlson_from_diag(ind,'subject_id','icd10_cm',ins['subject_id'].tolist())
ins['charlson_comorbidity_index']=ins['subject_id'].map(ch)+ins['age'].apply(age_points)
cohorts['INSPIRE (1-year)']=(ins,'death_365')

# ================= run recalibration =================
rows=[]; plot_store={}
for db,(df,ycol) in cohorts.items():
    y=df[ycol].values.astype(int)
    p_lr,p_xg,_=predict(df)
    for mname,p in [('Logistic',p_lr),('XGBoost',p_xg)]:
        auc=roc_auc_score(y,p)
        base=metrics(y,p)
        p_int=cv_recalibrate(y,p,'intercept')
        p_log=cv_recalibrate(y,p,'logistic')
        m_int=metrics(y,p_int); m_log=metrics(y,p_log)
        # also store full-sample recal params (for reporting the correction applied)
        a_int=fit_intercept_only(y,logit_f(p))
        a_log,b_log=fit_logistic_recal(y,logit_f(p))
        rows.append({'Database':db,'Model':mname,'N':len(y),'Events':int(y.sum()),
            'AUC':f"{auc:.3f}",
            'Brier_orig':f"{base['brier']:.3f}",'Brier_int':f"{m_int['brier']:.3f}",'Brier_log':f"{m_log['brier']:.3f}",
            'Slope_orig':f"{base['slope']:.2f}",'Slope_log':f"{m_log['slope']:.2f}",
            'Intcpt_orig':f"{base['intercept']:.2f}",'Intcpt_log':f"{m_log['intercept']:.2f}",
            'OE_orig':f"{base['oe']:.2f}",'OE_int':f"{m_int['oe']:.2f}",'OE_log':f"{m_log['oe']:.2f}",
            'ECE_orig':f"{base['ece']:.3f}",'ECE_int':f"{m_int['ece']:.3f}",'ECE_log':f"{m_log['ece']:.3f}",
            'a_int':round(a_int,3),'a_log':round(a_log,3),'b_log':round(b_log,3)})
        plot_store[(db,mname)]={'y':y,'orig':p,'int':p_int,'log':p_log}

tab=pd.DataFrame(rows)
tab.to_csv(OUT/"recalibration.csv",index=False)
pd.set_option('display.width',240,'display.max_columns',40)
print(tab[['Database','Model','AUC','Brier_orig','Brier_int','Brier_log',
           'Slope_orig','Slope_log','Intcpt_orig','Intcpt_log',
           'OE_orig','OE_int','OE_log','ECE_orig','ECE_int','ECE_log']].to_string(index=False))
joblib.dump(plot_store, OUT/"recal_preds.joblib")
print("\nSaved output/recalibration.csv + recal_preds.joblib")
