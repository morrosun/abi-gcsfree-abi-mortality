# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
98_build_v9_results.py —— 把 V8-schema 的结果文件用 V9 数据/数字重建，写入 output/v9_results/

背景
----
V9 重建（86–97）产出了新的数据、模型与结果 JSON，但**文件名带 _v2/_v9 后缀**，
而正文构建器 `64` / 附录构建器 `65` 是按固定的 V8 文件名读取的。
本脚本把 V9 真源翻译成 V8 schema，落到 output/v9_results/，使 64/65 只需换一个 OUT 目录
即可产出 V9 投稿包，而**不改动任何已交付的 V8 文件**（V8 包已快照到 output/_bak_round7/）。

唯一真源（V9）
--------------
  calibration_ci_and_severity_v2.json    (87)
  downstream_severity_gcs_v9.json        (90)
  leakage_sensitivity_v9.json            (91)
  v9_harmonisation.json                  (92)
  slope_epsilon_audit_v9.json            (93)
  model_performance_v9.csv               (94)  ← 仅用于与 98 自算的 5 学习器对账
  data/*_v2|v3|v4.csv                    (86)

沿用（V8，无 V9 对应物，已在本脚本内显式标注)
--------------------------------------------
  official_score_provenance.json   结构分支计数，与治疗窗/标签无关
  revision_external.json           M3_aetiology_strata 的 n（病例计数，未变）

在 V9 上**重算**的
------------------
  internal_learners.json   5 学习器（AUC/CI/Brier/nonzero）+ split
  revision_internal.json   M4_optimism（Harrell bootstrap B=200/60，与 V8 同 B）
  cohort_descriptives.json 各库 n/events/event_rate/female_pct
  external_current.json    discrimination + local_huaian（Brier 技能分要用）
  revision2.json           N3_gcs_sensitivity（改取 V9 的按通气分层）
  severity_headtohead.handwritten_vs_official  （移植 62 的 oasis()/saps2()）
  nwicu_vasopressor_recovery.json 的 prevalence（MIMIC/eICU/INSPIRE 重新计数）
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

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"
V9R = OUT / "v9_results"
V9R.mkdir(exist_ok=True)

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def j(name):
    return json.load(open(OUT / name, encoding="utf-8"))


def w(name, obj):
    p = V9R / name
    json.dump(obj, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("WROTE %s" % p.name)


AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
FEATS = ['age', 'female'] + AET + ['charlson_comorbidity_index'] + \
        [c for c in CONT if c not in ('age', 'charlson_comorbidity_index')] + BINV
assert len(FEATS) == 28

SEED = 42

MASTER = j("v9_number_master.json")
V2CAL = j("calibration_ci_and_severity_v2.json")
DS = j("downstream_severity_gcs_v9.json")
LK9 = j("leakage_sensitivity_v9.json")
HS9 = j("v9_harmonisation.json")
SE9 = j("slope_epsilon_audit_v9.json")

# V8 原件（只读，用于承载"未变字段"与逐项对账）
V8CAL = j("calibration_ci_and_severity.json")
V8AD = j("audit_decisionE.json")
V8RI = j("revision_internal.json")
V8RE = j("revision_external.json")
V8COH = j("cohort_descriptives.json")
V8VP = j("nwicu_vasopressor_recovery.json")
V8EC = j("external_current.json")
V8HS = j("v8_harmonisation.json")

CARRY = []          # 记下所有"沿用 V8"的字段，便于审计


def carry(what, why):
    CARRY.append({"field": what, "reason": why})


# ===================================================================== 1 校准 + 严重度
def build_calibration():
    log("\n" + "=" * 78)
    log("[1] calibration_ci_and_severity.json  ← 87(v2) + 90 + 62(手写对照在 V9 上重算)")
    s9 = DS["severity_headtohead"]
    sev = {
        "n": s9["n"], "events": s9["events"],
        "score_descriptives": s9["scores"],          # ★ V9 叫 scores，V8 叫 score_descriptives
        "comparison": s9["comparison"],
        "score_source": V8CAL["severity_headtohead"]["score_source"],
        "handwritten_vs_official": handwritten_vs_official(),
    }
    carry("severity_headtohead.score_source",
          "官方派生表来源说明，与治疗窗/标签无关")
    obj = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ece_definition": V8CAL["ece_definition"],
        "bootstrap": V8CAL["bootstrap"],
        "calibration_ci": MASTER["cohorts"],
        "severity_headtohead": sev,
        "external_scores_note": V8CAL.get("external_scores_note", ""),
    }
    w("calibration_ci_and_severity.json", obj)
    return obj


def handwritten_vs_official():
    """移植 scripts/62 的手写 oasis()/saps2()，在 V9 队列上重算对照（唯一真源=V9 数据）。"""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    sv = pd.read_csv(DATA / "mimic_severity_inputs.csv",
                     usecols=['stay_id', 'gcs_min', 'urineoutput', 'pao2fio2ratio_min',
                              'heart_rate_min', 'heart_rate_max', 'sbp_min', 'temperature_max',
                              'bilirubin_total_max', 'aids', 'mets', 'hem_malign', 'n_proc',
                              'pre_icu_los_hours'])
    mim = mim.merge(sv, on='stay_id', how='left')
    off = pd.read_csv(DATA / "mimic_severity_official.csv")
    mim = mim.merge(off, on='stay_id', how='left', validate='one_to_one')
    idx = np.arange(len(mim))
    _, te_idx = train_test_split(idx, test_size=0.30, stratify=mim['death_365'].values,
                                 random_state=SEED)
    te = mim.iloc[te_idx].copy()
    te['oasis'] = te['oasis_official'].values
    te['saps2'] = te['sapsii_official'].values
    te['oasis_hand'] = _oasis(te)
    te['saps2_hand'] = _saps2(te)
    yte = te['death_365'].values.astype(int)
    out = {}
    for k in ['oasis', 'saps2']:
        out[k] = {
            'pearson_r': float(np.corrcoef(te[k + '_hand'].values, te[k].values)[0, 1]),
            'mean_abs_diff': float(np.abs(te[k + '_hand'].values - te[k].values).mean()),
            'n_differ': int((te[k + '_hand'].values != te[k].values).sum()),
            'auc_alone_hand': float(roc_auc_score(yte, te[k + '_hand'].values)),
            'auc_alone_official': float(roc_auc_score(yte, te[k].values)),
        }
        log("  hand-vs-official %-6s hand %.4f official %.4f n_differ %d"
            % (k, out[k]['auc_alone_hand'], out[k]['auc_alone_official'], out[k]['n_differ']))
    return out


def _oasis(df):
    s = np.zeros(len(df))
    a = df['age'].values
    s += np.select([a < 24, a <= 53, a <= 77, a <= 89], [0, 3, 6, 9], default=7)
    hr = df['heart_rate_min'].fillna(70).values
    s += np.select([hr < 33, hr <= 89, hr <= 106, hr <= 125, hr <= 161],
                   [4, 0, 1, 3, 6], default=0)
    mbp = df['mbp_mean'].fillna(80).values
    s += np.select([mbp < 20.65, mbp <= 50.99, mbp <= 83.99, mbp <= 107.99, mbp <= 143.99],
                   [3, 2, 0, 1, 2], default=4)
    rr = df['resp_rate_mean'].values
    s += np.select([rr < 6, rr <= 12, rr <= 22, rr <= 30, rr <= 44], [10, 1, 0, 1, 6], default=9)
    t = df['temperature_mean'].values
    s += np.select([t < 33.22, t <= 35.93, t <= 36.39, t <= 36.88, t <= 39.88],
                   [3, 4, 2, 0, 2], default=6)
    uo = df['urineoutput'].fillna(df['urineoutput'].median()).values
    s += np.select([uo < 671, uo <= 1426.99, uo <= 2543.99, uo <= 6896], [10, 5, 1, 0], default=8)
    los = df['pre_icu_los_hours'].values
    s += np.select([los < 0.17, los <= 4.94, los <= 24.0, los <= 311.80], [5, 3, 0, 2], default=1)
    s += np.where(df['mech_vent'].values.astype(float) > 0, 9, 0)
    elective = df['admission_type'].astype(str).str.upper().eq('ELECTIVE').values
    s += np.where(elective, 0, 6)
    return s


def _saps2(df):
    s = np.zeros(len(df))
    a = df['age'].values
    s += np.select([a < 40, a < 60, a < 70, a < 75, a < 80], [0, 7, 12, 15, 16], default=18)
    hr_min = df['heart_rate_min'].fillna(80).values
    hr_max = df['heart_rate_max'].fillna(80).values

    def hr_pts(h):
        return np.select([h < 40, h < 70, h < 120, h < 160], [11, 2, 0, 4], default=7)
    s += np.maximum(hr_pts(hr_min), hr_pts(hr_max))
    sbp = df['sbp_min'].fillna(120).values
    s += np.select([sbp < 70, sbp < 100, sbp < 200], [13, 5, 0], default=2)
    t = df['temperature_max'].fillna(37).values
    s += np.where(t >= 39.0, 3, 0)
    g = df['gcs_min'].fillna(15).values
    s += np.select([g < 6, g < 9, g < 11, g < 14], [26, 13, 7, 5], default=0)
    pf = df['pao2fio2ratio_min'].values
    vent = df['mech_vent'].values.astype(float) > 0
    pf_pts = np.select([pf < 100, pf < 200], [11, 9], default=6)
    s += np.where(vent & ~np.isnan(pf), pf_pts, 0)
    uo = df['urineoutput'].fillna(df['urineoutput'].median()).values
    s += np.select([uo < 500, uo < 1000], [11, 4], default=0)
    bun = df['bun_max'].fillna(15).values
    s += np.select([bun < 28, bun < 84], [0, 6], default=10)
    wbc = df['wbc_max'].fillna(10).values
    s += np.select([wbc < 1.0, wbc < 20.0], [12, 0], default=3)
    k = df['potassium_max'].fillna(4).values
    s += np.select([k < 3.0, k < 5.0], [3, 0], default=3)
    na = df['sodium_min'].fillna(140).values
    s += np.select([na < 125, na < 145], [5, 0], default=1)
    hco3 = df['bicarbonate_min'].fillna(24).values
    s += np.select([hco3 < 15, hco3 < 20], [6, 3], default=0)
    bili = df['bilirubin_total_max'].fillna(0.5).values
    s += np.select([bili < 4.0, bili < 6.0], [0, 4], default=9)
    s += df['aids'].values * 17 + df['mets'].values * 9 + df['hem_malign'].values * 10
    elective = df['admission_type'].astype(str).str.upper().eq('ELECTIVE').values
    surgical = df['n_proc'].values > 0
    s += np.select([elective & surgical, surgical], [0, 8], default=6)
    return s


# ===================================================================== 2 audit_decisionE
def build_audit():
    log("\n" + "=" * 78)
    log("[2] audit_decisionE.json  ← V8 框架 + 90 的 gcs_increment / single_site_huaian")
    obj = json.loads(json.dumps(V8AD))          # 深拷贝
    obj["gcs_increment_decisionE"] = DS["gcs_increment"]
    ss = DS["single_site_huaian"]
    sc = obj.get("single_centre", {})
    sc = dict(sc)
    sc["tbi_pct"] = ss["aetiology_mix_pct"]["tbi"]
    sc["anoxic_pct"] = ss["aetiology_mix_pct"]["anoxic"]
    sc["aetiology_counts"] = ss["aetiology_mix"]
    sc["single_predictor_AUC"] = {k: {"auc": v} for k, v in ss["single_predictor_auc"].items()}
    obj["single_centre"] = sc
    obj["_v9_note"] = ("gcs_increment_decisionE 与 single_centre 的病因计数/AUC 来自 "
                       "downstream_severity_gcs_v9.json（90）；其余键沿用 V8（与治疗窗/标签无关）。")
    carry("audit_decisionE.json 除 gcs_increment_decisionE / single_centre 外全部键",
          "V8 原件中不含受 V9 数据重建影响的量")
    w("audit_decisionE.json", obj)


# ===================================================================== 3 斜率 ε
def build_slope_eps():
    log("\n" + "=" * 78)
    log("[3] slope_epsilon_audit.json  ← 93（把 slope_range_over_eps 改回 V8 的 slope_range）")
    res = {}
    for coh, d in SE9["results"].items():
        res[coh] = {"n": d["n"], "events": d["events"]}
        for m in ["Logistic", "XGBoost"]:
            mm = dict(d[m])
            mm["slope_range"] = mm.pop("slope_range_over_eps")
            res[coh][m] = mm
    obj = {"generated_at": SE9["generated_at"], "eps_list": SE9["eps_list"],
           "scope_note": SE9.get("scope_note", ""), "results": res,
           "fragile": SE9.get("fragile", [])}
    w("slope_epsilon_audit.json", obj)


# ===================================================================== 4 泄漏
def build_leakage():
    log("\n" + "=" * 78)
    log("[4] leakage_sensitivity.json  ← 91")
    obj = {"generated_at": LK9["generated_at"],
           "interpretation_bound": LK9.get("interpretation_bound", ""),
           "summary": LK9["summary"], "external": LK9["external_delta"]}
    w("leakage_sensitivity.json", obj)


# ===================================================================== 5 协调
def build_harmonisation():
    log("\n" + "=" * 78)
    log("[5] v8_harmonisation.json  ← 92")
    obj = dict(HS9)
    if "unit_conversion" not in obj:
        obj["unit_conversion"] = V8HS["unit_conversion"]
        carry("v8_harmonisation.unit_conversion",
              "换算因子（÷88.4 / ÷0.357 / ×18 / ÷10）是常量，与 V9 重建无关")
    w("v8_harmonisation.json", obj)


# ===================================================================== 6 内部学习器 + 乐观校正
def build_learners():
    log("\n" + "=" * 78)
    log("[6] internal_learners.json（V9 重算）+ revision_internal.M4_optimism（V9 重算）")
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    X = mim[FEATS].copy()
    y = mim['death_365'].values.astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
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
    log("  split: train=%d/%d  test=%d/%d" % (len(ytr), ytr.sum(), len(yte), yte.sum()))

    def auc_ci(yy, p, n_boot=1000, seed=SEED):
        rng = np.random.default_rng(seed)
        n = len(yy)
        vals = []
        for _ in range(n_boot):
            k = rng.choice(np.arange(n), size=n, replace=True)
            if len(np.unique(yy[k])) < 2:
                continue
            vals.append(roc_auc_score(yy[k], p[k]))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    # ★ 2026-10-19（118 号脚本）：树学习器超参 = 当前训练划分内 5 折 CV 的选中值，
    #   唯一真源 output/tuned_hyperparameters_v1.json（与 94 的 model_performance_v9.csv
    #   同源，下面的逐条对账才能成立）。缺失即报错，不允许静默回退旧开发期常量。
    HP = json.load(open(OUT / "tuned_hyperparameters_v1.json", encoding="utf-8"))["one_year"]

    def _hp(name):
        return dict(HP[name]["best_params"])

    specs = [
        ("Logistic", True, lambda: LogisticRegression(max_iter=2000, C=1e6)),
        ("LASSO", True, lambda: LogisticRegression(max_iter=5000, penalty='l1',
                                                   solver='liblinear', C=0.1, random_state=SEED)),
        ("RandomForest", False, lambda: RandomForestClassifier(
            random_state=SEED, n_jobs=-1, **_hp("RandomForest"))),
        ("GradientBoosting", False, lambda: GradientBoostingClassifier(
            random_state=SEED, **_hp("GradientBoosting"))),
        ("XGBoost", False, lambda: xgb.XGBClassifier(
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=SEED,
            tree_method='hist', n_jobs=-1, **_hp("XGBoost"))),
    ]
    res = {}
    for name, scaled, mk in specs:
        clf = mk()
        if scaled:
            clf.fit(Xtr_s[FEATS], ytr)
            p = clf.predict_proba(Xte_s[FEATS])[:, 1]
        else:
            clf.fit(Xtr[FEATS], ytr)
            p = clf.predict_proba(Xte[FEATS])[:, 1]
        res[name] = dict(auc=float(roc_auc_score(yte, p)), ci95=auc_ci(yte, p),
                         brier=float(brier_score_loss(yte, p)),
                         n_nonzero_coef=(int(np.sum(clf.coef_ != 0))
                                         if hasattr(clf, "coef_") else None),
                         standardised_input=bool(scaled))
        log("  %-18s AUC %.4f  Brier %.4f  nonzero=%s"
            % (name, res[name]['auc'], res[name]['brier'], res[name]['n_nonzero_coef']))

    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    payload = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        outcome="1-year death (MIMIC-IV internal held-out test set)",
        split=dict(test_size=0.30, stratify=True, seed=SEED,
                   n_train=int(len(ytr)), n_test=int(len(yte)),
                   events_train=int(ytr.sum()), events_test=int(yte.sum())),
        features=dict(n=len(FEATS), names=FEATS, continuous=CONT,
                      binary=AET + BINV + ['female']),
        preprocessing=dict(imputation="training-set median for continuous; 0 for indicators",
                           scaling="StandardScaler fitted on the training split; applied to the "
                                   "logistic/LASSO inputs only",
                           frozen_artefacts="preproc_v2.joblib / models_v2.joblib (1-year)"),
        bootstrap=dict(B=1000, method="percentile", resampled="patient-level, with replacement",
                       seed=SEED),
        learners=res,
    )
    w("internal_learners.json", payload)

    # ---- 与 94 的 model_performance_v9.csv 对账（不得有实质分歧） ----
    mp = pd.read_csv(OUT / "model_performance_v9.csv", index_col=0)
    _alias = {"GradientBoosting": "GBM"}
    for name in res:
        s = mp.loc[_alias.get(name, name), "AUC (95% CI)"]
        a = float(s.split(" ")[0])
        assert abs(a - res[name]["auc"]) < 5e-4, \
            "98 复算的 %s AUC %.4f 与 94 的 %s 不一致" % (name, res[name]["auc"], s)
    log("  [OK] 5 学习器 AUC 与 94 的 model_performance_v9.csv 逐条一致")

    # ---- Harrell 乐观校正（B 与 V8 一致：LR 200 / XGB 60） ----
    def harrell(name, mk, scaled, B):
        from sklearn.base import clone
        Xf = Xtr_s[FEATS] if scaled else Xtr[FEATS]
        Xall = pd.concat([Xtr, Xte], axis=0)
        yall = np.concatenate([ytr, yte])
        Xall_s = Xall.copy()
        Xall_s[CONT] = sc.transform(Xall[CONT])
        Xf_all = Xall_s[FEATS] if scaled else Xall[FEATS]
        base = clone(mk()).fit(Xf_all, yall)
        apparent = float(roc_auc_score(yall, base.predict_proba(Xf_all)[:, 1]))
        rng = np.random.default_rng(SEED)
        n = len(yall)
        ops = []
        for _ in range(B):
            k = rng.choice(np.arange(n), size=n, replace=True)
            if len(np.unique(yall[k])) < 2:
                continue
            m = clone(mk()).fit(Xf_all.iloc[k], yall[k])
            app = roc_auc_score(yall[k], m.predict_proba(Xf_all.iloc[k])[:, 1])
            tst = roc_auc_score(yall, m.predict_proba(Xf_all)[:, 1])
            ops.append(app - tst)
        op = float(np.mean(ops))
        return dict(apparent=round(apparent, 4), optimism=round(op, 4),
                    corrected=round(apparent - op, 4), B=B)

    lr_opt = harrell("Logistic", lambda: LogisticRegression(max_iter=2000, C=1e6), True, 200)
    log("  optimism LR : apparent %.4f  optimism %.4f  corrected %.4f (B=%d)"
        % (lr_opt['apparent'], lr_opt['optimism'], lr_opt['corrected'], lr_opt['B']))
    xg_opt = harrell("XGBoost", lambda: xgb.XGBClassifier(
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=SEED,
        tree_method='hist', n_jobs=-1, **_hp("XGBoost")), False, 60)
    log("  optimism XGB: apparent %.4f  optimism %.4f  corrected %.4f (B=%d)"
        % (xg_opt['apparent'], xg_opt['optimism'], xg_opt['corrected'], xg_opt['B']))

    ri = json.loads(json.dumps(V8RI))
    ri["M4_optimism"] = {"Logistic": lr_opt, "XGBoost": xg_opt,
                         "hyperparameters": HP,
                         "_v9_note": ("V10 重算（Harrell bootstrap，B 与 V8 一致：LR 200 / XGB 60）；"
                                      "XGBoost 超参取 118 在当前训练划分内 5 折 CV 的选中值"
                                      "（tuned_hyperparameters_v1.json），不再沿用 V8 开发期网格。")}
    carry("revision_internal.M4_optimism.hyperparameters",
          "V10：已由 V8 开发期网格换为 118 的当前划分选中值（tuned_hyperparameters_v1.json）")
    w("revision_internal.json", ri)
    return res


# ===================================================================== 7 队列描述
def build_cohorts():
    log("\n" + "=" * 78)
    log("[7] cohort_descriptives.json（V9 重新计数 n/events/rate/female_pct）")
    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    nw = pd.read_csv(DATA / "nwicu_abi_final_v6.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v3.csv")
    e = pd.read_csv(DATA / "eicu_abi_final_v3.csv")
    old = {c["database"]: c for c in V8COH["cohorts"]}
    new = []
    spec = [
        ("MIMIC-IV", mim, "death_365", "gender", "F"),
        ("eICU", e, "death_hosp", "gender", None),
        ("NWICU", nw, "death_365", "gender", None),
        ("INSPIRE", ins, "death_365", "sex", None),
    ]
    for db, df, ycol, scol, fval in spec:
        c = dict(old[db])
        c["n"] = int(len(df))
        c["events"] = int(df[ycol].sum())
        c["event_rate_pct"] = round(100.0 * c["events"] / c["n"], 1)
        if scol in df.columns:
            if fval:
                c["female_pct"] = round(100.0 * (df[scol] == fval).mean(), 1)
            else:
                c["female_pct"] = round(100.0 * df[scol].astype(str).str.upper()
                                        .isin(["F", "FEMALE"]).mean(), 1)
        # ★★ P2（2026-10-07 第二轮评审 §3.1）：治疗强度三项此前是**从 V8 条目整份拷过来**的
        #   （c = dict(old[db]) 里带着 mech_vent_pct / vasopressor_pct / rrt_pct），
        #   于是本文件在 V9 里仍旧写 MIMIC vasopressor 23.3 / eICU 11.8 —— 而 V9 的
        #   时间窗重建后真值是 18.8 / 8.8，与正文表 1 打架，且本文件属公开归档结果集。
        #   现按 V9 数据文件重算，与 Table 1 / 表 S3 同源。
        for _col, _key in (("mech_vent", "mech_vent_pct"),
                           ("vasopressor", "vasopressor_pct"),
                           ("rrt", "rrt_pct")):
            if _col in df.columns:
                c[_key] = round(100.0 * (df[_col] == 1).mean(), 1)
        new.append(c)
        log("  %-9s n=%d events=%d rate=%.1f%% female=%.1f%% vent=%.1f%% vaso=%.1f%% rrt=%.1f%%"
            % (db, c["n"], c["events"], c["event_rate_pct"], c["female_pct"],
               c.get("mech_vent_pct", float("nan")), c.get("vasopressor_pct", float("nan")),
               c.get("rrt_pct", float("nan"))))
    hua = dict(old["Huaian"])
    new.append(hua)
    carry("cohort_descriptives 的 Huaian 条目", "淮安队列未被 V9 数据重建触及")
    carry("cohort_descriptives 的 age_median/IQR（Table 1 用字面量，不读该键）",
          "Table 1 的年龄列为稿内字面量")
    obj = dict(V8COH)
    obj["cohorts"] = new
    obj["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    obj["sources"] = {"MIMIC_IV": "data/mimic_abi_cohort_v2.csv", "eICU": "data/eicu_abi_final_v3.csv",
                      "NWICU": "data/nwicu_abi_final_v6.csv", "INSPIRE": "data/inspire_abi_final_v3.csv",
                      "Huaian": V8COH["sources"].get("Huaian", "")}
    w("cohort_descriptives.json", obj)


# ===================================================================== 8 外验 + 淮安
def build_external_current():
    log("\n" + "=" * 78)
    log("[8] external_current.json（V9 重建 discrimination + local_huaian）")
    disc = {}
    tagmap = {"NWICU_1yr": ("NWICU", "1-year death"),
              "INSPIRE_1yr": ("INSPIRE", "1-year death"),
              "eICU_inhosp": ("eICU", "in-hospital death"),
              "MIMIC_internal_inhosp": ("MIMIC-IV", "in-hospital death")}
    for tag, (db, out) in tagmap.items():
        d = V2CAL[tag]
        disc[tag] = {"database": db, "outcome": out,
                     "n": d["n"], "events": d["events"],
                     "prevalence_pct": round(100.0 * d["events"] / d["n"], 1)}
        for m in ["Logistic", "XGBoost"]:
            pt, ci = d[m]["point"], d[m]["ci95"]
            disc[tag][m] = {"auc": pt["auc"], "ci": ci["auc"], "brier": pt["brier"],
                            "slope": pt["slope"], "intercept": pt["intercept"],
                            "oe": pt["oe"], "ece": pt["ece"]}
    h = V2CAL["Huaian_inhosp"]
    loc = {}
    for m in ["Logistic", "XGBoost"]:
        pt, ci = h[m]["point"], h[m]["ci95"]
        loc[m] = {"AUC": pt["auc"], "CI": ci["auc"], "obs": h["events"] / h["n"],
                  "pred": (h["events"] / h["n"]) / pt["oe"], "OE": pt["oe"],
                  "brier": pt["brier"], "slope": pt["slope"], "intercept": pt["intercept"]}
    obj = dict(V8EC)
    obj["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    obj["rule"] = "eICU / Huaian = same-outcome in-hospital parallel model; NWICU / INSPIRE = frozen 1-year model"
    obj["discrimination"] = disc
    obj["local_huaian"] = loc
    obj["_v9_note"] = "全部数字取自 calibration_ci_and_severity_v2.json（87）"
    carry("external_current.recalibration / inhosp_parallel_mimic_internal",
          "64 未读取；保留 V8 占位")
    w("external_current.json", obj)


# ===================================================================== 9 revision2（GCS 分层）
def build_revision2():
    log("\n" + "=" * 78)
    log("[9] revision2.json  ← N3_gcs_sensitivity 取 V9 的按通气分层")
    g = DS["gcs_increment"]["gcs_alone_by_ventilation"]
    mv = DS["gcs_increment"]["missingness_pct"]
    r2 = json.loads(json.dumps(j("revision2.json")))
    r2["N3_gcs_sensitivity"] = {
        "operationalization": {
            "source": "mimiciv_derived.first_day_gcs (ICU 首日最差/最低 GCS 总分)",
            "measure": "首日最差值(worst/min)", "missing_rate_pct": mv["gcs_min_pct"],
            "imputation": "训练集中位数插补", "train_median": 14.0,
            "vent_pct": round(100.0 * g["ventilated"]["n"] / g["full_test"]["n"], 1)},
        "gcs_alone_auc": {
            "full_test": {"auc": g["full_test"]["auc"], "n": g["full_test"]["n"],
                          "events": g["full_test"]["events"]},
            "ventilated": {"auc": g["ventilated"]["auc"], "n": g["ventilated"]["n"],
                           "events": g["ventilated"]["events"]},
            "nonventilated": {"auc": g["nonventilated"]["auc"], "n": g["nonventilated"]["n"],
                              "events": g["nonventilated"]["events"]},
            "motor_gcs_full": {"auc": DS["gcs_increment"]["alone_AUC_logit"]["gcs_motor"],
                               "n": g["full_test"]["n"], "events": g["full_test"]["events"]},
        },
        "_v9_note": ("通气分层改用 V9 的口径（mech_vent 首日窗）；方向约定取 "
                     "1−raw AUC（GCS 越低风险越高），与 90 一致。")}
    w("revision2.json", r2)


# ===================================================================== 10 revision_external
def build_revision_external():
    log("\n" + "=" * 78)
    log("[10] revision_external.json（保留病因分层计数；portable-core 由 V9 敏感性替代）")
    re = json.loads(json.dumps(V8RE))
    # ★ 2026-10-07 修：原先这里取 V8VP（74 的 V8 产物），于是 portable_core_slope_LR=0.035、
    #   portable_core_auc={0.6686, 0.7708} 全是 V8 数字 —— 而 full_model_slope_LR 已换成 V9 的
    #   0.8778，正文就会出现「斜率从 0.878 降到 0.035」这种跨版本拼接。portable-core 即
    #   「剔除全部三个治疗强度指标」的 25 变量敏感性，须与 scripts/100 的 V9 重算同源。
    _p100 = V9R / "nwicu_vasopressor_recovery.json"
    assert _p100.exists(), ("缺 %s —— 请先运行 scripts/100_v9_vasopressor_recovery.py"
                            % _p100.name)
    _o100 = json.load(open(_p100, encoding="utf-8"))
    assert _o100.get("generated_by") == "100_v9_vasopressor_recovery.py", \
        "%s 不是 100 的 V9 产物（generated_by=%r）" % (_p100.name, _o100.get("generated_by"))
    vpd = _o100["drop_sensitivity"]["drop_all_organ_support"]["recovered"]
    re["M3_nwicu_slope"] = dict(V8RE["M3_nwicu_slope"])
    re["M3_nwicu_slope"]["full_model_slope_LR"] = V2CAL["NWICU_1yr"]["Logistic"]["point"]["slope"]
    re["M3_nwicu_slope"]["portable_core_slope_LR"] = vpd["Logistic"]["slope"]
    re["M3_nwicu_slope"]["portable_core_auc"] = {"Logistic": vpd["Logistic"]["auc"],
                                                 "XGBoost": vpd["XGBoost"]["auc"]}
    # 患病率用 V9 的首日窗口径（V8 的 23.3/11.8 是全住院医嘱口径，两者不可混）
    assert set(_o100["prevalence"]) == {"MIMIC(dev)", "eICU", "NWICU", "INSPIRE"}, \
        "100 的 prevalence 键不齐：%s" % sorted(_o100["prevalence"])
    re["M3_nwicu_slope"]["vasopressor_prevalence"] = dict(_o100["prevalence"])
    carry("revision_external.M3_aetiology_strata", "分层的 n 是病例计数，未受 V9 治疗窗/标签影响")
    carry("revision_external.M2_inhosp_parallel / M3_inspire_noaet",
          "64 未读取该两键")
    re["_v9_note"] = ("V9：NWICU logistic 斜率改为 v2 真源；portable-core 分支改为"
                      "「剔除全部三个治疗强度指标」的单中心敏感性（数字取自 scripts/100 的 V9 重算，"
                      "不再沿用 74 的 V8 值）。vasopressor_prevalence 亦改为首日窗口径。")
    w("revision_external.json", re)


# ===================================================================== 11 血管活性药
def build_vasopressor():
    log("\n" + "=" * 78)
    log("[11] nwicu_vasopressor_recovery.json（必须由 scripts/100 先行产出）")
    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    e = pd.read_csv(DATA / "eicu_abi_final_v3.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v3.csv")
    # ★ 2026-10-07：V8 版本里这个文件整体搬运自 74，导致 NWICU 的 AUC/slope 仍是 V8 数字
    #   （0.6958→0.6971 等）。V9 改由 scripts/100_v9_vasopressor_recovery.py 用 V9 重训模型
    #   与 V9 标签重算，schema 与 74 完全一致。这里只做校验，不覆盖 —— 覆盖会让正文数字回退。
    p = V9R / "nwicu_vasopressor_recovery.json"
    assert p.exists(), ("缺 %s —— 请先运行 scripts/100_v9_vasopressor_recovery.py"
                        % p.name)
    obj = json.load(open(p, encoding="utf-8"))
    assert obj.get("generated_by") == "100_v9_vasopressor_recovery.py", \
        ("%s 不是 100 的 V9 产物（generated_by=%r）；请先运行 scripts/100"
         % (p.name, obj.get("generated_by")))
    prev = {
        "MIMIC(dev)": round(100.0 * mim["vasopressor"].sum() / len(mim), 1),
        "eICU": round(100.0 * e["vasopressor"].sum() / len(e), 1),
        "NWICU": float(obj["prevalence"]["NWICU"]),
        "INSPIRE": round(100.0 * ins["vasopressor"].sum() / len(ins), 1),
    }
    assert abs(prev["NWICU"] - 11.1) < 0.2, \
        "100 的 NWICU 患病率异常：%.1f%%" % prev["NWICU"]
    for k, v in prev.items():
        assert abs(float(obj["prevalence"][k]) - v) < 1e-9, \
            "100 与 98 的患病率不一致 %s：%s vs %s" % (k, obj["prevalence"][k], v)
    log("  prevalence V9: " + ", ".join("%s=%.1f%%" % kv for kv in prev.items()))
    log("  [OK] 100 的产物已就位，未覆盖（64 直接从 v9_results 读取）")


# ===================================================================== 12 直接搬运（与 V9 无关）
def copy_carried():
    log("\n" + "=" * 78)
    log("[12] 直接搬运的 V8 结果文件（64 会从 OUT 读取）")
    for f in ["official_score_provenance.json", "train_medians.json"]:
        src = OUT / f
        if src.exists():
            shutil.copy2(src, V9R / f)
            log("COPIED %s" % f)
            carry(f, "不含受 V9 数据重建影响的量")


def main():
    log("#" * 78)
    log("# 98_build_v9_results.py —— 生成 output/v9_results/（V8 schema / V9 数字）")
    log("#" * 78)
    build_calibration()
    build_audit()
    build_slope_eps()
    build_leakage()
    build_harmonisation()
    build_learners()
    build_cohorts()
    build_external_current()
    build_revision2()
    build_revision_external()
    build_vasopressor()
    copy_carried()
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sources": {k: str(OUT / k) for k in
                        ["calibration_ci_and_severity_v2.json", "downstream_severity_gcs_v9.json",
                         "leakage_sensitivity_v9.json", "v9_harmonisation.json",
                         "slope_epsilon_audit_v9.json"]},
            "carried_from_v8": CARRY}
    w("_v9_results_provenance.json", meta)
    (OUT / "_98_build_v9_results.log").write_text("\n".join(LOG), encoding="utf-8")
    log("\nALL DONE → %s" % V9R)
    return 0


if __name__ == "__main__":
    sys.exit(main())
