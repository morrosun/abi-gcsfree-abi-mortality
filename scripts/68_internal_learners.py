# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
68_internal_learners.py  ——  给 V8 Table 2 的 5 个候选学习器补上"可溯源"的内部测试集数字

背景（2026-09-30 实证核查）
--------------------------
V7 稿件 Table 2 列出 5 个候选学习器的内部 AUC：
    Logistic 0.816 / LASSO 0.816 / Random forest 0.832 / Gradient boosting 0.841 / XGBoost 0.846
但全项目检索 output/*.json，**只有 Logistic 与 XGBoost 两条有结果文件出处**
（revision_internal.json 的 M1_gcs_increment / M4_optimism、calibration_ci_and_severity.json）。
随机森林与梯度提升的 0.832 / 0.841 只存在于 HTML 正文字符串里 —— 与 66 脚本查出的
"+0.076 / −0.288" 属同一类"手敲无出处"数字，正是 Decision E 第 A 条的根因类型。
Figure 1（V8 的 Figure1.png）面板 A 画的是 Logistic / RandomForest / XGBoost 三条 ROC，
因此随机森林的 AUC 必须能落到结果文件上，否则图内数字无人可校验。

本脚本做的事（两段式）
----------------------
  STAGE 1  用与 03/10/60/62 完全一致的口径重建 70/30 分层划分与预处理，
           拟合 5 个学习器，在 4,980 人内部测试集上算 AUC / 95% CI / Brier，
           落盘 output/internal_learners.json（唯一真源）
  STAGE 2  从该 JSON **读回**，与 V7 Table 2 的手敲值逐条断言
           （容差 0.002：CI 算法不同会带来末位差异）；不一致即停机。

不覆盖：不改写任何图件，不碰 output/submission_v7/ 与 _v8/ 下任何已交付文件。
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

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

SEED = 42
N_BOOT = 1000
OUTJSON = OUT / "internal_learners.json"

# V7 Table 2 的手敲值（用于 STAGE 2 断言；None = 当时未报告）
V7_TABLE2 = {
    "Logistic": 0.816,
    "LASSO": 0.816,
    "RandomForest": 0.832,
    "GradientBoosting": 0.841,
    "XGBoost": 0.846,
}

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression                              # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score                      # noqa: E402
from sklearn.model_selection import train_test_split                             # noqa: E402
from sklearn.preprocessing import StandardScaler                                 # noqa: E402
import xgboost as xgb                                                            # noqa: E402


def auc_ci(y, p, n_boot=N_BOOT, seed=SEED):
    """患者级 bootstrap 百分位 95% CI（与 62 脚本同口径）。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_all = np.arange(n)
    vals = []
    for _ in range(n_boot):
        k = rng.choice(idx_all, size=n, replace=True)
        if len(np.unique(y[k])) < 2:
            continue
        vals.append(roc_auc_score(y[k], p[k]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    mim = pd.read_csv(DATA / "mimic_abi_cohort.csv")
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
    print("split: train=%d/%d  test=%d/%d (seed=%d)"
          % (len(ytr), int(ytr.sum()), len(yte), int(yte.sum()), SEED))

    # 冻结产物口径复核：1 年模型的"训练集中位数 + scaler"须与本划分一致
    pp = joblib.load(OUT / "preproc.joblib")
    assert list(pp['feats']) == FEATS, "preproc feats mismatch"
    assert list(pp['cont']) == CONT, "preproc cont mismatch"

    specs = [
        # (名字, 是否吃标准化特征, 构造器)
        ("Logistic", True, lambda: LogisticRegression(max_iter=2000, C=1e6)),
        ("LASSO", True, lambda: LogisticRegression(max_iter=5000, penalty='l1', solver='liblinear',
                                                   C=0.1, random_state=SEED)),
        ("RandomForest", False, lambda: RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=20,
            random_state=SEED, n_jobs=-1)),
        ("GradientBoosting", False, lambda: GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED)),
        ("XGBoost", False, lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric='logloss', random_state=SEED,
            tree_method='hist', n_jobs=-1)),
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
        auc = float(roc_auc_score(yte, p))
        ci = auc_ci(yte, p)
        br = float(brier_score_loss(yte, p))
        nz = int(np.sum(clf.coef_ != 0)) if hasattr(clf, "coef_") else None
        res[name] = dict(auc=auc, ci95=ci, brier=br, n_nonzero_coef=nz,
                         standardised_input=bool(scaled))
        print("  %-18s AUC %.4f [%.4f, %.4f]  Brier %.4f  nonzero=%s"
              % (name, auc, ci[0], ci[1], br, nz))

    payload = dict(
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        outcome="1-year death (MIMIC-IV internal held-out test set)",
        split=dict(test_size=0.30, stratify=True, seed=SEED,
                   n_train=int(len(ytr)), n_test=int(len(yte)),
                   events_train=int(ytr.sum()), events_test=int(yte.sum())),
        features=dict(n=len(FEATS), names=FEATS, continuous=CONT, binary=AET + BINV + ['female']),
        preprocessing=dict(imputation="training-set median for continuous; 0 for indicators",
                           scaling="StandardScaler fitted on the training split; applied to the "
                                   "logistic/LASSO inputs only",
                           frozen_artefacts="preproc.joblib / models.joblib (1-year)"),
        bootstrap=dict(B=N_BOOT, method="percentile", resampled="patient-level, with replacement",
                       seed=SEED),
        learners=res,
    )
    json.dump(payload, open(OUTJSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", OUTJSON)

    # ---------------- STAGE 2：读回 + 与 V7 Table 2 手敲值对账 ----------------
    back = json.load(open(OUTJSON, encoding="utf-8"))["learners"]
    print("\n=== STAGE 2  读回并与 V7 Table 2 手敲值对账（容差 0.002） ===")
    bad = []
    for k, v in V7_TABLE2.items():
        got = back[k]["auc"]
        d = abs(got - v)
        flag = "OK " if d <= 0.002 else "!! "
        print("  [%s] %-18s recomputed %.4f vs V7 %.3f  Δ=%.4f" % (flag.strip(), k, got, v, d))
        if d > 0.002:
            bad.append((k, got, v))
    assert not bad, "V7 Table 2 手敲值与重算不符：%s" % bad
    assert abs(back['XGBoost']['auc'] - 0.8465) < 1e-3, "XGBoost 与 62 脚本口径不一致"
    assert abs(back['Logistic']['auc'] - 0.8161) < 1e-3, "Logistic 与 62 脚本口径不一致"
    print("  [OK ] RandomForest / GradientBoosting 的 AUC 现已落到 %s（不再是手敲值）"
          % OUTJSON.name)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
