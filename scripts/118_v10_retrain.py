# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
118_v10_retrain.py —— V10 重建：NWICU 源层重聚合主数据 + 树超参在当前训练划分重选
                    + 「预测窗内死亡」量化与内部敏感性

三项修改（对应上一轮期刊评审意见 #4 / #3 / #1，2026-10-19）
-----------------------------------------------------------
#4 glucose 主分析切换（编辑：*"the fully corrected data … should replace the flawed
   data in your primary analysis"*）
   · NWICU 主分析数据由 nwicu_abi_final_v5.csv（占位值置缺失）切换到
     nwicu_abi_final_v6.csv（117 号脚本：源层排除哨兵行后按同一窗口重聚合，
     救回 602 例血糖 / 27 例血钾 / 1 例尿素氮，再过 89 的同一套可信区间筛查）；
   · 旧版（v5）降为敏感性分析，数字由 102 号脚本的报告承载。

#1 树超参污染（编辑：*"hyperparameters were selected using a cross validation grid
   search on an earlier data split that overlaps heavily with the final held out
   test set"*）
   · 旧管线：RF/GBM/XGB 超参取自**标签修正前**的开发期网格搜索，修正后重画划分，
     新测试集 45.7%（2,276/4,980）落入旧训练划分；
   · 本脚本：三个树学习器的超参一律在**当前**（修正后）训练划分内做 5 折 CV 重选，
     再用选出的值在完整训练划分上重拟合，冻结产物随之更新；
   · 选中值落盘 output/tuned_hyperparameters_v1.json，94 / 98 只读该文件，
     保证「选参 → 拟合 → 报表」单一真源。

#3 窗内死亡（编辑：*"the model retains patients who die before the 24 hour predictor
   window closes"*）
   · 用 mimiciv_hosp.admissions.deathtime（精确到分钟）算「入 ICU 24 h 内死亡」，
     而不是用 date-only 的 dod 估；
   · 报告 n / 占队列比 / 占结局事件比；
   · 内部敏感性：把窗内死亡从训练与测试中一并剔除，重拟合 LR + XGB
     （插补中位数、标准化、超参全部在缩减后的训练划分内重推），比较 AUC 变化。

口径与 87 严格一致（保证下游 90–99 无需改代码）
------------------------------------------------
  * 7:3 分层划分，stratify=y，random_state=42
  * 连续变量训练集中位数插补；指示变量填 0
  * StandardScaler 只在训练集拟合，只喂 Logistic/LASSO
  * LR: LogisticRegression(max_iter=2000, C=1e6)；LASSO: LogisticRegressionCV(Cs=20, cv=5, l1, saga)
  * bootstrap B=1000 百分位 95% CI（seed=42），与 87/62 同一实现
  * 淮安单位换算：肌酐 ÷88.4、尿素氮 ÷0.357、血糖 ×18、血红蛋白 ÷10

产物（一律先备份再覆写；V8 文件一律不碰）
-----------------------------------------
  output/calibration_ci_and_severity_v2.json   （覆写，旧件备份为 _bak_118_*.json）
  output/preproc_v2.joblib / models_v2.joblib / model_inhosp_v2.joblib（同上）
  output/tuned_hyperparameters_v1.json         （新增：两结局 × 三树学习器的选中超参）
  output/window_exclusion_sensitivity_v1.json  （新增：窗内死亡量化 + 剔除后重拟合）
  output/_118_retrain.md                       （运行日志）
"""
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV        # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score                     # noqa: E402
from sklearn.model_selection import GridSearchCV, train_test_split              # noqa: E402
from sklearn.preprocessing import StandardScaler                                # noqa: E402
import xgboost as xgb                                                           # noqa: E402
import psycopg2                                                                 # noqa: E402

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
FEATS = ['age', 'female'] + AET + ['charlson_comorbidity_index'] + \
        [c for c in CONT if c not in ('age', 'charlson_comorbidity_index')] + BINV
EPS = 1e-6
SEED = 42
B_BOOT = 1000
WINDOW_H = 24.0

# ---- 树学习器搜索网格（围绕旧开发期选中值设小网格，控制算力与变动幅度）----
GRIDS = {
    "RandomForest": dict(
        n_estimators=[500], max_depth=[6, 8, 12], min_samples_leaf=[10, 20]),
    "GradientBoosting": dict(
        n_estimators=[200, 300], max_depth=[2, 3], learning_rate=[0.05, 0.10]),
    "XGBoost": dict(
        n_estimators=[200, 400], max_depth=[3, 4, 6],
        learning_rate=[0.05, 0.10]),
}
XGB_FIXED = dict(subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                 random_state=SEED, tree_method='hist', n_jobs=-1)

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


# ---------------------------------------------------------------- 指标（与 87 同一实现）
def logit_f(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cal_slope_intercept(y, p):
    lp = logit_f(p)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=2000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def oe_ratio(y, p):
    return float(np.mean(y) / np.mean(p))


def ece(y, p, bins=10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    n = len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += abs(y[m].mean() - p[m].mean()) * m.sum() / n
    return float(e)


def all_metrics(y, p):
    s, i = cal_slope_intercept(y, p)
    return {'auc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
            'slope': s, 'intercept': i, 'oe': oe_ratio(y, p), 'ece': ece(y, p)}


def boot_ci(y, p, B=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    acc = {k: [] for k in ['auc', 'brier', 'slope', 'intercept', 'oe', 'ece']}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2 or len(np.unique(pb)) < 2:
            continue
        try:
            m = all_metrics(yb, pb)
        except Exception:                                     # noqa: BLE001
            continue
        for k in acc:
            acc[k].append(m[k])
    return {k: {'n_boot': len(v),
                'ci': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in acc.items()}


# ---------------------------------------------------------------- 构建（含超参重选）
def make_tree(name, params, outcome_key):
    if name == "RandomForest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=SEED, **params)
    if name == "XGBoost":
        return xgb.XGBClassifier(**XGB_FIXED, **params)
    raise KeyError(name)


def tune_and_fit(name, Xtr, ytr):
    """在当前训练划分内 5 折 CV 选参 → 在完整训练划分上重拟合。"""
    grid = GRIDS[name]
    base = make_tree(name, {}, None)
    gs = GridSearchCV(base, grid, cv=5, scoring='roc_auc', n_jobs=-1, refit=True)
    gs.fit(Xtr[FEATS], ytr)
    best = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in gs.best_params_.items()}
    log("    %-17s CV AUC %.4f  选中 %s" % (name, gs.best_score_, best))
    return gs.best_estimator_, best, float(gs.best_score_)


def build_outcome(X, y, tag):
    """7:3 划分 + 插补 + 标准化 + LR/LASSO/三树学习器（树超参当前划分内重选）。"""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    med = Xtr[CONT].median()
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in CONT:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xte[c] = Xte[c].fillna(med[c])
    for c in FEATS:
        if c not in CONT:
            Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').fillna(0)
            Xte[c] = pd.to_numeric(Xte[c], errors='coerce').fillna(0)
    sc = StandardScaler().fit(Xtr[CONT])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[CONT] = sc.transform(Xtr[CONT])
    Xte_s[CONT] = sc.transform(Xte[CONT])

    lr = LogisticRegression(max_iter=2000, C=1e6).fit(Xtr_s[FEATS], ytr)
    lasso = LogisticRegressionCV(Cs=20, cv=5, penalty='l1', solver='saga',
                                 scoring='roc_auc', max_iter=5000,
                                 random_state=SEED).fit(Xtr_s[FEATS], ytr)
    log("  [%s] 树超参在当前训练划分内重选（5 折 CV, scoring=roc_auc）:" % tag)
    tuned = {}
    trees = {}
    for nm in ["RandomForest", "GradientBoosting", "XGBoost"]:
        est, best, score = tune_and_fit(nm, Xtr, ytr)
        trees[nm] = est
        tuned[nm] = dict(best_params=best, cv_auc=round(score, 4))
    art = dict(feats=FEATS, cont=CONT, scaler=sc, medians=med.to_dict(),
               lr=lr, xg=trees["XGBoost"], trees=trees, tuned=tuned,
               logit_params={'const': float(lr.intercept_[0]),
                             **{f: float(v) for f, v in zip(FEATS, lr.coef_[0])}})
    return art, Xtr, Xte, Xtr_s, Xte_s, ytr, yte, lasso


def apply_frozen(art, df):
    """把冻结模型应用到外部数据：LR 吃标准化，XGB 吃原始。"""
    Z = df[art['feats']].copy()
    for c in art['cont']:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(art['medians'][c])
    for c in FEATS:
        if c not in art['cont']:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
    Zs = Z.copy()
    Zs[art['cont']] = art['scaler'].transform(Z[art['cont']])
    lp = art['logit_params']
    a = 1 / (1 + np.exp(-(lp['const'] + Zs[FEATS].values @ np.array([lp[f] for f in FEATS]))))
    b = art['xg'].predict_proba(Z[FEATS].values)[:, 1]
    return a, b


def in_window_deaths(mim):
    """精确到分钟的窗内死亡：deathtime - intime ∈ [0, 24h]。

    ⚠ 不能用 date-only 的 dod 估（会把同日不同时的死亡算错，本次实测两种口径
      差 192 例：674 vs 482）；deathtime 只覆盖院内死亡，而窗内死亡必为院内死亡，
      故口径闭合。
    """
    conn = psycopg2.connect(host=_os.environ.get("ABI_DB_HOST", "localhost"),
                            port=int(_os.environ.get("ABI_DB_PORT", "5432")),
                            user=_os.environ.get("ABI_DB_USER", "postgres"),
                            password=_os.environ.get("ABI_DB_PASSWORD", ""),
                            dbname="mimiciv")
    cur = conn.cursor()
    ids = tuple(int(x) for x in mim['hadm_id'].unique())
    cur.execute("SELECT hadm_id, deathtime FROM mimiciv_hosp.admissions "
                "WHERE hadm_id IN %s", (ids,))
    dt = pd.DataFrame(cur.fetchall(), columns=['hadm_id', 'deathtime'])
    conn.close()
    m = mim.merge(dt, on='hadm_id', how='left')
    m['deathtime'] = pd.to_datetime(m['deathtime'])
    m['intime'] = pd.to_datetime(m['intime'])
    hrs = (m['deathtime'] - m['intime']).dt.total_seconds() / 3600.0
    m['hours_to_death'] = hrs
    m['died_in_window'] = ((hrs >= 0) & (hrs <= WINDOW_H)).astype(int)
    return m


def window_sensitivity(X, y, tag, art):
    """剔除窗内死亡后从头重拟合（插补/标准化/模型全部在缩减训练划分内重推）。"""
    w = X['died_in_window'].values.astype(bool)
    Xr = X.loc[~w, FEATS].copy()
    yr = y[~w]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.30, stratify=yr,
                                          random_state=SEED)
    med = Xtr[CONT].median()
    for c in CONT:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xte[c] = Xte[c].fillna(med[c])
    for c in FEATS:
        if c not in CONT:
            Xtr[c] = pd.to_numeric(Xtr[c], errors='coerce').fillna(0)
            Xte[c] = pd.to_numeric(Xte[c], errors='coerce').fillna(0)
    sc = StandardScaler().fit(Xtr[CONT])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[CONT] = sc.transform(Xtr[CONT])
    Xte_s[CONT] = sc.transform(Xte[CONT])
    out = {'n_excluded_total': int(w.sum()),
           'n_train_after': int(len(ytr)), 'events_train_after': int(ytr.sum()),
           'n_test_after': int(len(yte)), 'events_test_after': int(yte.sum())}
    for nm, scaled, mk in [
            ('Logistic', True, lambda: LogisticRegression(max_iter=2000, C=1e6)),
            ('XGBoost', False, lambda: make_tree('XGBoost', art['tuned']['XGBoost']['best_params'], tag))]:
        clf = mk().fit(Xtr_s[FEATS] if scaled else Xtr[FEATS], ytr)
        p = clf.predict_proba(Xte_s[FEATS] if scaled else Xte[FEATS])[:, 1]
        mm = all_metrics(yte, p)
        ci = boot_ci(yte, p)
        out[nm] = {'auc': float(mm['auc']), 'ci95': ci['auc']['ci'],
                   'brier': float(mm['brier']), 'slope': float(mm['slope']),
                   'intercept': float(mm['intercept']), 'oe': float(mm['oe']),
                   'ece': float(mm['ece'])}
        log("    %-9s 剔除窗内死亡后 AUC %.4f (%.3f-%.3f)  slope %.3f  O:E %.3f"
            % (nm, mm['auc'], ci['auc']['ci'][0], ci['auc']['ci'][1], mm['slope'], mm['oe']))
    return out


def backup(path):
    if path.exists():
        b = path.with_name("_bak_118_" + path.name)
        if not b.exists():
            shutil.copy2(path, b)
            log("  [backup] %s → %s" % (path.name, b.name))


def main():
    log("=" * 78)
    log("118 —— V10 重建：重聚合主数据 + 树超参重选 + 窗内死亡敏感性")
    log("=" * 78)

    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    nw = pd.read_csv(DATA / "nwicu_abi_final_v6.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v3.csv")
    e = pd.read_csv(DATA / "eicu_abi_final_v3.csv")
    log("  MIMIC n=%d（death_365=%d, death_hosp=%d）" %
        (len(mim), int(mim.death_365.sum()), int(mim.death_hosp.sum())))
    log("  NWICU n=%d（death_365=%d, death_hosp=%d） ← v6 源层重聚合版" %
        (len(nw), int(nw.death_365.sum()), int(nw.death_hosp.sum())))
    log("  INSPIRE n=%d（death_365=%d）  eICU n=%d（death_hosp=%d）" %
        (len(ins), int(ins.death_365.sum()), len(e), int(e.death_hosp.sum())))

    # ---- 窗内死亡（精确 deathtime）----
    mim = in_window_deaths(mim)
    nw_ = int(mim['died_in_window'].sum())
    log("")
    log("  [窗内死亡] 入 ICU 24h 内死亡 n=%d（%.2f%% 的开发队列；占院内死亡 %.1f%%；"
        "占一年死亡 %.1f%%）" %
        (nw_, 100 * nw_ / len(mim), 100 * nw_ / int(mim.death_hosp.sum()),
         100 * nw_ / int(mim.death_365.sum())))

    X = mim[FEATS + ['died_in_window']].copy()

    # ---------------- 1 年模型 ----------------
    art1, Xtr, Xte, Xtr_s, Xte_s, ytr, yte, lasso1 = build_outcome(
        X, mim['death_365'].values.astype(int), "1yr")
    log("  [1 年模型] 内部测试集 n=%d, events=%d" % (len(yte), int(yte.sum())))
    PRED = {'MIMIC_internal_1yr': {
        'y': yte,
        'Logistic': art1['lr'].predict_proba(Xte_s[FEATS])[:, 1],
        'XGBoost': art1['xg'].predict_proba(Xte[FEATS])[:, 1]}}
    for tag, df, yc in [('NWICU_1yr', nw, 'death_365'), ('INSPIRE_1yr', ins, 'death_365')]:
        a, b = apply_frozen(art1, df)
        PRED[tag] = {'y': df[yc].values.astype(int), 'Logistic': a, 'XGBoost': b}

    # ---------------- 院内平行模型 ----------------
    art2, Xtr2, Xte2, Xtr2_s, Xte2_s, ytr2, yte2, lasso2 = build_outcome(
        X, mim['death_hosp'].values.astype(int), "inhosp")
    log("  [院内模型] 内部测试集 n=%d, events=%d" % (len(yte2), int(yte2.sum())))
    PRED['MIMIC_internal_inhosp'] = {
        'y': yte2,
        'Logistic': art2['lr'].predict_proba(Xte2_s[FEATS])[:, 1],
        'XGBoost': art2['xg'].predict_proba(Xte2[FEATS])[:, 1]}
    a, b = apply_frozen(art2, e)
    PRED['eICU_inhosp'] = {'y': e['death_hosp'].values.astype(int), 'Logistic': a, 'XGBoost': b}

    # ---------------- 淮安单中心（院内） ----------------
    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    if P.exists():
        raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
        raw = raw.dropna(how='all').reset_index(drop=True)
        num = lambda c: pd.to_numeric(raw[c], errors='coerce')      # noqa: E731
        d = pd.DataFrame(index=raw.index)
        d['age'] = num('age')
        d['y'] = (num('hospital_discharge_location') == 5).astype(int)
        for c in AET:
            d[c] = num(c).fillna(0)
        keep = (d[AET].sum(axis=1) >= 1) & (d['age'] >= 18) & d['age'].notna() & \
            num('hospital_discharge_location').notna()
        dx = d[keep].copy()
        Xl = pd.DataFrame(index=dx.index)
        for c in FEATS:
            Xl[c] = num(c) if c in raw.columns else np.nan
        CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
                'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
        for c, f in CONV.items():
            Xl[c] = f(Xl[c])
        Zl = Xl.copy()
        for c in CONT:
            Zl[c] = Zl[c].fillna(art2['medians'][c])
        for c in FEATS:
            if c not in CONT:
                Zl[c] = Zl[c].fillna(0)
        Zls = Zl.copy()
        Zls[CONT] = art2['scaler'].transform(Zl[CONT])
        PRED['Huaian_inhosp'] = {
            'y': dx['y'].values.astype(int),
            'Logistic': art2['lr'].predict_proba(Zls[FEATS])[:, 1],
            'XGBoost': art2['xg'].predict_proba(Zl[FEATS])[:, 1]}
    else:
        log("  ⚠ 淮安原始表不存在：%s" % P)

    # ---------------- 汇总 ----------------
    RES = {}
    log("")
    for tag, d in PRED.items():
        RES[tag] = {'n': int(len(d['y'])), 'events': int(d['y'].sum()),
                    'prevalence_pct': round(100 * float(np.mean(d['y'])), 1)}
        for mname in ['Logistic', 'XGBoost']:
            pt = d[mname]
            m0 = all_metrics(d['y'], pt)
            ci = boot_ci(d['y'], pt)
            RES[tag][mname] = {'point': {k: float(m0[k]) for k in m0},
                               'ci95': {k: ci[k]['ci'] for k in ci},
                               'n_boot': ci['auc']['n_boot']}
            log("  %-22s %-8s n=%6d AUC %.4f  slope %.4f [%.3f,%.3f]  O:E %.3f  ECE %.4f" %
                (tag, mname, RES[tag]['n'], m0['auc'], m0['slope'],
                 ci['slope']['ci'][0], ci['slope']['ci'][1], m0['oe'], m0['ece']))

    # ---------------- 窗内死亡敏感性 ----------------
    log("")
    log("  [窗内死亡敏感性] 剔除 24h 内死亡后从头重拟合（训练划分内重推插补/标准化/模型）")
    WS = {'_definition': "deathtime - intime ∈ [0, 24h]，取自 mimiciv_hosp.admissions.deathtime"
                         "（分钟级）；dod 为 date-only，两种口径实测差 192 例，故不用 dod。",
          'n_died_in_window': nw_,
          'pct_of_cohort': round(100 * nw_ / len(mim), 2),
          'pct_of_inhospital_deaths': round(100 * nw_ / int(mim.death_hosp.sum()), 1),
          'pct_of_one_year_deaths': round(100 * nw_ / int(mim.death_365.sum()), 1),
          'hours_to_death_median': float(np.nanmedian(mim.loc[mim.died_in_window == 1,
                                                              'hours_to_death'])),
          'one_year': {}, 'inhospital': {}}
    WS['one_year'] = window_sensitivity(X, mim['death_365'].values.astype(int), "1yr", art1)
    WS['inhospital'] = window_sensitivity(X, mim['death_hosp'].values.astype(int), "inhosp", art2)

    # ---------------- 落盘（先备份） ----------------
    for f in ["calibration_ci_and_severity_v2.json", "preproc_v2.joblib",
              "models_v2.joblib", "model_inhosp_v2.joblib"]:
        backup(OUT / f)

    joblib.dump(dict(feats=FEATS, cont=CONT, scaler=art1['scaler'],
                     medians=art1['medians']), OUT / "preproc_v2.joblib")
    joblib.dump(dict(logit_params=art1['logit_params'], xgb=art1['xg']),
                OUT / "models_v2.joblib")
    joblib.dump(dict(lr=art2['lr'], xg=art2['xg'], scaler=art2['scaler'], cont=CONT,
                     medians=art2['medians'], feats=FEATS,
                     internal={'Logistic': {'auc': RES['MIMIC_internal_inhosp']['Logistic']['point']['auc']},
                               'XGBoost': {'auc': RES['MIMIC_internal_inhosp']['XGBoost']['point']['auc']}}),
                OUT / "model_inhosp_v2.joblib")
    json.dump(RES, open(OUT / "calibration_ci_and_severity_v2.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(WS, open(OUT / "window_exclusion_sensitivity_v1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    HP = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
          "_method": ("三个树学习器的超参在**当前**（标签修正后）训练划分内做 5 折 "
                      "GridSearchCV（scoring=roc_auc），选中值在完整训练划分上重拟合；"
                      "取代旧开发期（修正前划分）网格搜索。94 / 98 只读本文件。"),
          "grids": {k: {kk: [type(x).__name__ for x in vv] for kk, vv in v.items()}
                    for k, v in GRIDS.items()},
          "grids_values": GRIDS,
          "one_year": art1['tuned'], "inhosp": art2['tuned']}
    json.dump(HP, open(OUT / "tuned_hyperparameters_v1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(OUT / "_118_retrain.md", "w", encoding="utf-8") as f:
        f.write("# 118 · V10 重建（重聚合主数据 + 树超参重选 + 窗内死亡敏感性）\n\n```\n"
                + "\n".join(LOG) + "\n```\n")
    log("")
    log("→ calibration_ci_and_severity_v2.json（覆写，旧件 _bak_118_*）")
    log("→ preproc_v2.joblib / models_v2.joblib / model_inhosp_v2.joblib（覆写，旧件备份）")
    log("→ tuned_hyperparameters_v1.json / window_exclusion_sensitivity_v1.json / _118_retrain.md")


if __name__ == "__main__":
    main()
