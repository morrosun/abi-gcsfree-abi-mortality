# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
94_v9_main_figures.py —— V9 主结果图（MIMIC-IV 内部）+ 表格真源

数据层（86 重建 + 89 清洗）与模型层（87 重训）都变了，因此 Figure 1 / Figure 2
所依赖的一切都必须重算：
  · 五个候选学习器（Logistic / LASSO / RandomForest / XGBoost / GBM）的 AUC、Brier
  · 主 Logistic 的调整 OR（statsmodels Logit，每 1-SD）
  · XGBoost 的 gain importance
  · ROC 曲线与 10 分位校准曲线

★ 口径与 03_model_dev.py 完全一致（7:3 分层、seed=42、训练集插补与标准化），
  只换数据文件；并对账：本脚本的 Logistic/XGBoost AUC 必须等于 87 的主结果。
★ 2026-10-19（118 号脚本起）：三个树学习器的超参不再用旧开发期常量，改为读
  output/tuned_hyperparameters_v1.json（当前训练划分内 5 折 CV 的选中值），
  与 118 的冻结模型、98 的复算三者同源。
★ 输出一律写到 output/v9_panels/，不覆盖任何 V8 已交付文件。

产出：output/v9_panels/{fig1_roc_v9,fig2_calibration_v9,fig4_forest_v9,
      fig5_xgb_importance_v9}.png
      output/model_performance_v9.csv、logit_OR_v9.csv、dca_logistic_v9.csv
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
import statsmodels.api as sm                             # noqa: E402
import xgboost as xgb                                    # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV         # noqa: E402
from sklearn.metrics import (brier_score_loss, roc_auc_score,                     # noqa: E402
                             roc_curve)
from sklearn.calibration import calibration_curve                                 # noqa: E402
from sklearn.model_selection import train_test_split                              # noqa: E402
from sklearn.preprocessing import StandardScaler                                  # noqa: E402

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"
PAN = OUT / "v9_panels"
PAN.mkdir(parents=True, exist_ok=True)

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
FEATS = ['age', 'female'] + AET + ['charlson_comorbidity_index'] + \
        [c for c in CONT if c not in ('age', 'charlson_comorbidity_index')] + BINV
SEED = 42
C = {'lr': '#185FA5', 'xgb': '#A32D2D', 'rf': '#0F6E56'}

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def boot_auc(y, p, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return np.percentile(a, [2.5, 97.5])


def main():
    log("=" * 78)
    log("94 —— V9 主结果图（MIMIC-IV 内部，5 个候选学习器重训）")
    log("=" * 78)
    df = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    df['female'] = (df['gender'] == 'F').astype(int)
    y = df['death_365'].values.astype(int)
    X = df[FEATS].copy()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                          random_state=SEED)
    medians = Xtr[CONT].median()
    for c in CONT:
        Xtr[c] = Xtr[c].fillna(medians[c])
        Xte[c] = Xte[c].fillna(medians[c])
    scaler = StandardScaler().fit(Xtr[CONT])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[CONT] = scaler.transform(Xtr[CONT])
    Xte_s[CONT] = scaler.transform(Xte[CONT])
    log(f"  train n={len(ytr):,} (events={int(ytr.sum()):,})  "
        f"test n={len(yte):,} (events={int(yte.sum()):,})")

    # ★ 2026-10-19（118 号脚本）：三个树学习器的超参改为**当前训练划分内 5 折 CV 重选**，
    #   选中值唯一真源 = output/tuned_hyperparameters_v1.json（94 / 98 都只读它，
    #   杜绝"选参口径"在两处漂移）。缺失该文件直接报错，不允许静默回退旧值。
    HP = json.load(open(OUT / "tuned_hyperparameters_v1.json", encoding="utf-8"))
    HP1 = HP["one_year"]

    def _hp(name):
        return dict(HP1[name]["best_params"])

    res, roc_store, calib_store = {}, {}, {}

    # 1. Logistic（statsmodels，供 OR）
    Xtr_sm = sm.add_constant(Xtr_s[FEATS])
    logit = sm.Logit(ytr, Xtr_sm).fit(disp=0, maxiter=200)
    or_tab = pd.DataFrame({'coef': logit.params, 'OR': np.exp(logit.params),
                           'CI_low': np.exp(logit.conf_int()[0]),
                           'CI_high': np.exp(logit.conf_int()[1]),
                           'p': logit.pvalues}).round(4)
    or_tab.to_csv(OUT / "logit_OR_v9.csv")
    p_lr = logit.predict(sm.add_constant(Xte_s[FEATS])).values
    res['Logistic'] = {'auc': roc_auc_score(yte, p_lr), 'ci': boot_auc(yte, p_lr),
                       'brier': brier_score_loss(yte, p_lr)}
    roc_store['Logistic'] = roc_curve(yte, p_lr)[:2]
    calib_store['Logistic'] = calibration_curve(yte, p_lr, n_bins=10, strategy='quantile')

    # 2. LASSO
    lasso = LogisticRegressionCV(Cs=20, cv=5, penalty='l1', solver='saga',
                                 scoring='roc_auc', max_iter=5000,
                                 random_state=SEED).fit(Xtr_s[FEATS], ytr)
    p_la = lasso.predict_proba(Xte_s[FEATS])[:, 1]
    sel = pd.Series(lasso.coef_[0], index=FEATS)
    sel[sel != 0].round(4).to_csv(OUT / "lasso_selected_v9.csv", header=['coef'])
    res['LASSO'] = {'auc': roc_auc_score(yte, p_la), 'ci': boot_auc(yte, p_la),
                    'brier': brier_score_loss(yte, p_la),
                    'n_selected': int((sel != 0).sum())}

    # 3. RandomForest（超参 = 118 在当前训练划分内重选，见 tuned_hyperparameters_v1.json）
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                **_hp("RandomForest")).fit(Xtr[FEATS], ytr)
    p_rf = rf.predict_proba(Xte[FEATS])[:, 1]
    res['RandomForest'] = {'auc': roc_auc_score(yte, p_rf), 'ci': boot_auc(yte, p_rf),
                           'brier': brier_score_loss(yte, p_rf)}
    roc_store['RandomForest'] = roc_curve(yte, p_rf)[:2]

    # 4. XGBoost（超参 = 118 在当前训练划分内重选）
    xgbm = xgb.XGBClassifier(subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                             random_state=SEED, scale_pos_weight=1, tree_method='hist',
                             n_jobs=-1, **_hp("XGBoost")).fit(Xtr[FEATS], ytr)
    p_xg = xgbm.predict_proba(Xte[FEATS])[:, 1]
    res['XGBoost'] = {'auc': roc_auc_score(yte, p_xg), 'ci': boot_auc(yte, p_xg),
                      'brier': brier_score_loss(yte, p_xg)}
    roc_store['XGBoost'] = roc_curve(yte, p_xg)[:2]
    calib_store['XGBoost'] = calibration_curve(yte, p_xg, n_bins=10, strategy='quantile')

    # 5. GBM（超参 = 118 在当前训练划分内重选）
    gbm = GradientBoostingClassifier(random_state=SEED,
                                     **_hp("GradientBoosting")).fit(Xtr[FEATS], ytr)
    p_gb = gbm.predict_proba(Xte[FEATS])[:, 1]
    res['GBM'] = {'auc': roc_auc_score(yte, p_gb), 'ci': boot_auc(yte, p_gb),
                  'brier': brier_score_loss(yte, p_gb)}

    # ★ 对账：与 87 的主结果必须一致（statsmodels Logit ≈ sklearn C=1e6）
    ref = __import__('json').load(
        open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))
    for nm, got in [('Logistic', res['Logistic']['auc']), ('XGBoost', res['XGBoost']['auc'])]:
        want = ref['MIMIC_internal_1yr'][nm]['point']['auc']
        assert abs(got - want) < 5e-4, \
            f"RECONCILE FAIL: {nm} AUC {got!r} vs 87 的 {want!r}（差 {abs(got-want):.2e}）"
        log(f"  [reconcile OK] {nm:<8} AUC {got:.6f} ≈ 87 的 {want:.6f} "
            f"(Δ{abs(got-want):.2e}，statsmodels vs sklearn C=1e6)")

    perf = pd.DataFrame(res).T
    perf['AUC (95% CI)'] = ["%.3f (%.3f-%.3f)" % (res[m]['auc'], res[m]['ci'][0],
                                                  res[m]['ci'][1]) for m in perf.index]
    perf['Brier'] = ["%.3f" % res[m]['brier'] for m in perf.index]
    perf[['AUC (95% CI)', 'Brier']].to_csv(OUT / "model_performance_v9.csv")
    log("")
    log(perf[['AUC (95% CI)', 'Brier']].to_string())

    # DCA
    def net_benefit(y_true, y_prob, thresholds):
        y_true = np.asarray(y_true)
        N = len(y_true)
        return np.array([np.sum((y_prob >= pt) & (y_true == 1)) / N -
                         np.sum((y_prob >= pt) & (y_true == 0)) / N * (pt / (1 - pt))
                         for pt in thresholds])
    th = np.arange(0.01, 0.80, 0.01)
    prev = yte.mean()
    pd.DataFrame({'threshold': th, 'model': net_benefit(yte, p_lr, th),
                  'treat_all': prev - (1 - prev) * (th / (1 - th)),
                  'treat_none': 0}).to_csv(OUT / "dca_logistic_v9.csv", index=False)

    # ---------------- Fig1: ROC ----------------
    plt.rcParams.update({'font.size': 8.5})
    # 印刷宽 = 6.3/2 ≈ 3.05 in（图 1 左右并排）
    fig, ax = plt.subplots(figsize=(3.05, 2.95))
    for m, col in [('Logistic', C['lr']), ('RandomForest', C['rf']), ('XGBoost', C['xgb'])]:
        fpr, tpr = roc_store[m]
        ax.plot(fpr, tpr, color=col, lw=2,
                label="%s: %.3f (%.3f-%.3f)" % (m, res[m]['auc'], res[m]['ci'][0],
                                                res[m]['ci'][1]))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC: 1-year mortality', fontsize=8.5)
    ax.legend(loc='lower right', fontsize=6.5, frameon=False)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(PAN / "fig1_roc_v9.png", dpi=300)
    plt.close(fig)
    log("\n  panel -> fig1_roc_v9.png")

    # ---------------- Fig2: Calibration ----------------
    fig, ax = plt.subplots(figsize=(3.05, 2.95))
    for m, col in [('Logistic', C['lr']), ('XGBoost', C['xgb'])]:
        t, p = calib_store[m]
        ax.plot(p, t, 'o-', color=col, label=m, ms=5)
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Ideal')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Calibration (10 quantile bins)', fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(PAN / "fig2_calibration_v9.png", dpi=300)
    plt.close(fig)
    log("  panel -> fig2_calibration_v9.png")

    # ---------------- Fig4: OR forest ----------------
    o = or_tab.drop(index=['const'], errors='ignore').copy()
    o = o[o['p'] < 0.05].sort_values('OR')
    # 印刷宽 ≈ 3.05 in（图 2 左右并排）
    fig, ax = plt.subplots(figsize=(3.05, max(1.7, 0.185 * len(o) + 0.7)))
    yy = np.arange(len(o))
    ax.errorbar(o['OR'], yy,
                xerr=[o['OR'] - o['CI_low'], o['CI_high'] - o['OR']],
                fmt='o', color=C['lr'], ecolor=C['lr'], capsize=3, ms=4)
    ax.axvline(1.0, color='k', ls=':', lw=1)
    ax.set_yticks(yy)
    # 3.05 in 宽：长标签必须换行，否则 tight_layout 也救不了（文字直接溢出画布）
    ax.set_yticklabels(o.index, fontsize=7)
    ax.set_xlabel('Adjusted odds ratio\n(per 1-SD for\ncontinuous predictors)', fontsize=7.5)
    ax.set_title('Predictors with P<.05\n(logistic model)', fontsize=8)
    fig.tight_layout()
    fig.savefig(PAN / "fig4_forest_v9.png", dpi=300)
    plt.close(fig)
    log(f"  panel -> fig4_forest_v9.png  ({len(o)} predictors with P<.05)")

    # ---------------- Fig5: XGBoost gain importance ----------------
    imp = xgbm.get_booster().get_score(importance_type='gain')
    s = pd.Series(imp).reindex(FEATS).fillna(0.0).sort_values(ascending=False)
    top = s.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(3.05, 2.4))
    ax.barh(range(len(top)), top.values, color=C['xgb'])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=7)
    ax.set_xlabel('Gain', fontsize=7.5)
    ax.set_title('XGBoost feature importance\n(gain)', fontsize=8)
    fig.tight_layout()
    fig.savefig(PAN / "fig5_xgb_importance_v9.png", dpi=300)
    plt.close(fig)
    log("  panel -> fig5_xgb_importance_v9.png")

    # importance 明细落盘（供 Table/附录引用）
    pd.DataFrame({'feature': s.index, 'gain': s.values}).to_csv(
        OUT / "xgb_importance_v9.csv", index=False)

    (OUT / "_94_v9_main_figures.md").write_text(
        "# 94 —— V9 主结果图（内部）\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    log("\n→ output/v9_panels/*.png, model_performance_v9.csv, logit_OR_v9.csv, "
        "xgb_importance_v9.csv")


if __name__ == "__main__":
    main()
