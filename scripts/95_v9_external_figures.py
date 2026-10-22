# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
95_v9_external_figures.py —— V9 外部验证图 + 根因面板 + 表格真源

★ 本轮最重要的构图变化：NWICU 院内死亡从 16 例修正为 378 例（84 脚本：
  hospital_expire_flag 在该库漏报约 23 倍），因此 NWICU 的**院内结局**首次可纳入
  外部验证图；V8 那句 "NWICU excluded (only 16 in-hospital deaths)" 必须删除。

图：
  fig6_ext_roc_v9           外部 1 年 ROC（NWICU / INSPIRE，双模型）
  fig7_ext_roc_bymodel_v9   每个验证库的双模型 ROC（含 NWICU 院内、淮安）
  fig8_ext_calibration_v9   再校准前的校准曲线
  fig9_recal_calibration_v9 logistic 再校准之后的校准曲线（在各自外部库内拟合）
  figR3_rootcause_v9        血管活性药记录跨库差异 + 病因构成（NWICU vs INSPIRE）

产出：output/external_validation_v9.csv、recalibration_v9.csv
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                           # noqa: E402

from sklearn.calibration import calibration_curve         # noqa: E402
from sklearn.linear_model import LogisticRegression        # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve  # noqa: E402

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
EPS = 1e-6
C = {'lr': '#185FA5', 'xgb': '#A32D2D'}

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def prep(art, df):
    Z = df[FEATS].copy()
    for c in CONT:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(art['medians'][c])
    for c in FEATS:
        if c not in CONT:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
    Zs = Z.copy()
    Zs[CONT] = art['scaler'].transform(Z[CONT])
    return Z, Zs


def predict(art, df):
    Z, Zs = prep(art, df)
    if 'logit_params' in art:
        lp = art['logit_params']
        a = 1 / (1 + np.exp(-(lp['const'] + Zs[FEATS].values @
                              np.array([lp[f] for f in FEATS]))))
    else:
        a = art['lr'].predict_proba(Zs[FEATS])[:, 1]
    xg_ = art.get('xg', art.get('xgb'))
    return a, xg_.predict_proba(Z[FEATS])[:, 1]


def cal_slope_intercept(y, p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs',
                           max_iter=2000).fit(lp, np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def ece(y, p, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(e)


def huaian_frame():
    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')      # noqa: E731
    d = pd.DataFrame(index=raw.index)
    d['y'] = (num('hospital_discharge_location') == 5).astype(int)
    for c in AET:
        d[c] = num(c).fillna(0)
    keep = (d[AET].sum(axis=1) >= 1) & (num('age') >= 18) & \
        num('age').notna() & num('hospital_discharge_location').notna()
    dx = d[keep].copy()
    Xl = pd.DataFrame(index=dx.index)
    for c in FEATS:
        Xl[c] = num(c) if c in raw.columns else np.nan
    for c, f in {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
                 'glucose_max': lambda s: s * 18.0,
                 'hemoglobin_min': lambda s: s / 10.0}.items():
        Xl[c] = f(Xl[c])
    return Xl, dx['y'].values.astype(int)


def main():
    log("=" * 78)
    log("95 —— V9 外部验证图（NWICU 院内结局首次可纳入）")
    log("=" * 78)
    pp = joblib.load(OUT / "preproc_v2.joblib")
    md = joblib.load(OUT / "models_v2.joblib")
    a1 = dict(medians=pp['medians'], scaler=pp['scaler'],
              logit_params=md['logit_params'], xgb=md['xgb'])
    ih = joblib.load(OUT / "model_inhosp_v2.joblib")
    a2 = dict(medians=ih['medians'], scaler=ih['scaler'], lr=ih['lr'], xg=ih['xg'])

    nw = pd.read_csv(DATA / "nwicu_abi_final_v6.csv")
    ins = pd.read_csv(DATA / "inspire_abi_final_v3.csv")
    e = pd.read_csv(DATA / "eicu_abi_final_v3.csv")
    Xl, yl = huaian_frame()

    COH = [('NWICU_1yr', nw, 'death_365', a1, '1-year'),
           ('INSPIRE_1yr', ins, 'death_365', a1, '1-year'),
           ('eICU_inhosp', e, 'death_hosp', a2, 'in-hospital'),
           ('NWICU_inhosp', nw, 'death_hosp', a2, 'in-hospital'),
           ('INSPIRE_inhosp', ins, 'death_hosp', a2, 'in-hospital'),
           ('Huaian_inhosp', Xl, None, a2, 'in-hospital')]

    P = {}
    rows = []
    ref = json.load(open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))
    for tag, df, yc, art, kind in COH:
        y = yl if yc is None else df[yc].values.astype(int)
        plr, pxg = predict(art, df)
        P[tag] = {'y': y, 'Logistic': plr, 'XGBoost': pxg, 'kind': kind}
        for nm, p in [('Logistic', plr), ('XGBoost', pxg)]:
            s, i = cal_slope_intercept(y, p)
            r = {'cohort': tag, 'outcome': kind, 'model': nm, 'n': int(len(y)),
                 'events': int(y.sum()), 'auc': float(roc_auc_score(y, p)),
                 'brier': float(brier_score_loss(y, p)), 'slope': s, 'intercept': i,
                 'oe': float(y.mean() / p.mean()), 'ece': ece(y, p)}
            rows.append(r)
            if tag in ref and nm in ref[tag]:
                w = ref[tag][nm]['point']['auc']
                assert abs(r['auc'] - w) < 1e-9, \
                    f"RECONCILE FAIL: {tag}/{nm} {r['auc']!r} != {w!r}"
                log(f"  [reconcile OK] {tag:<16} {nm:<8} AUC {r['auc']:.6f} == 87")
        log("  %-16s n=%6d events=%5d  LR %.4f  XGB %.4f" %
            (tag, len(y), int(np.sum(y)), roc_auc_score(y, plr), roc_auc_score(y, pxg)))
    ext = pd.DataFrame(rows)
    ext.to_csv(OUT / "external_validation_v9.csv", index=False)

    # ---------------- 再校准（在各自外部库内拟合 logistic recalibration） -------------
    rec = {}
    rec_rows = []
    for tag in P:
        y = P[tag]['y']
        rec[tag] = {}
        for nm in ['Logistic', 'XGBoost']:
            p = P[tag][nm]
            lp = np.log(np.clip(p, EPS, 1 - EPS)).reshape(-1, 1)
            m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs',
                                   max_iter=2000).fit(lp, y)
            p2 = m.predict_proba(lp)[:, 1]
            rec[tag][nm] = p2
            rec_rows.append({'cohort': tag, 'outcome': P[tag]['kind'], 'model': nm,
                             'n': int(len(y)), 'events': int(y.sum()),
                             'auc_before': float(roc_auc_score(y, p)),
                             'auc_after': float(roc_auc_score(y, p2)),
                             'brier_after': float(brier_score_loss(y, p2)),
                             'slope_after': cal_slope_intercept(y, p2)[0],
                             'oe_after': float(y.mean() / p2.mean()),
                             'ece_after': ece(y, p2)})
    pd.DataFrame(rec_rows).to_csv(OUT / "recalibration_v9.csv", index=False)
    log("")
    log("  再校准后 O:E / ECE 落盘 recalibration_v9.csv")

    # ---------------- Fig6: 外部 1 年 ROC ----------------
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for tag in ['NWICU_1yr', 'INSPIRE_1yr']:
        y = P[tag]['y']
        for nm, ls in [('Logistic', '-'), ('XGBoost', '--')]:
            p = P[tag][nm]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, ls, lw=1.8, color=C['lr'] if nm == 'Logistic' else C['xgb'],
                    label="%s %s: %.3f" % (tag.split('_')[0], nm, roc_auc_score(y, p)))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('External validation: 1-year outcome')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(PAN / "fig6_ext_roc_v9.png", dpi=200)
    plt.close(fig)
    log("  panel -> fig6_ext_roc_v9.png")

    # ---------------- Fig7: 各库双模型 ROC ----------------
    order = ['NWICU_1yr', 'INSPIRE_1yr', 'eICU_inhosp', 'NWICU_inhosp',
             'INSPIRE_inhosp', 'Huaian_inhosp']
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    for k, tag in enumerate(order):
        ax = axes[k // 3][k % 3]
        y = P[tag]['y']
        for nm in ['Logistic', 'XGBoost']:
            p = P[tag][nm]
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, lw=1.8, color=C['lr'] if nm == 'Logistic' else C['xgb'],
                    label="%s %.3f" % (nm, roc_auc_score(y, p)))
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=0.9)
        ax.set_title("%s\n(%s, n=%d, %d events)" %
                     (tag.replace('_', ' '), P[tag]['kind'], len(y), int(np.sum(y))),
                     fontsize=9)
        ax.set_xlabel('1 - Specificity', fontsize=8.5)
        ax.set_ylabel('Sensitivity', fontsize=8.5)
        ax.legend(fontsize=8, frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(PAN / "fig7_ext_roc_bymodel_v9.png", dpi=200)
    plt.close(fig)
    log("  panel -> fig7_ext_roc_bymodel_v9.png")

    # ---------------- Fig8: 再校准前校准曲线 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    for k, tag in enumerate(order):
        ax = axes[k // 3][k % 3]
        y = P[tag]['y']
        for nm in ['Logistic', 'XGBoost']:
            p = P[tag][nm]
            t, pp_ = calibration_curve(y, p, n_bins=10, strategy='quantile')
            ax.plot(pp_, t, 'o-', ms=4, lw=1.4,
                    color=C['lr'] if nm == 'Logistic' else C['xgb'],
                    label="%s (slope %.2f)" % (nm, cal_slope_intercept(y, p)[0]))
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=0.9)
        ax.set_title(tag.replace('_', ' '), fontsize=9)
        ax.set_xlabel('Predicted probability', fontsize=8.5)
        ax.set_ylabel('Observed proportion', fontsize=8.5)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Calibration before recalibration', fontsize=11)
    fig.tight_layout()
    fig.savefig(PAN / "fig8_ext_calibration_v9.png", dpi=200)
    plt.close(fig)
    log("  panel -> fig8_ext_calibration_v9.png")

    # ---------------- Fig9: 再校准后 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0))
    for k, tag in enumerate(order):
        ax = axes[k // 3][k % 3]
        y = P[tag]['y']
        for nm in ['Logistic', 'XGBoost']:
            p2 = rec[tag][nm]
            t, pp_ = calibration_curve(y, p2, n_bins=10, strategy='quantile')
            ax.plot(pp_, t, 'o-', ms=4, lw=1.4,
                    color=C['lr'] if nm == 'Logistic' else C['xgb'], label=nm)
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=0.9)
        ax.set_title(tag.replace('_', ' '), fontsize=9)
        ax.set_xlabel('Predicted probability', fontsize=8.5)
        ax.set_ylabel('Observed proportion', fontsize=8.5)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Calibration after logistic recalibration (fitted within each database)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PAN / "fig9_recal_calibration_v9.png", dpi=200)
    plt.close(fig)
    log("  panel -> fig9_recal_calibration_v9.png")

    # ---------------- FigR3: 根因面板 ----------------
    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    vp = {'MIMIC-IV': 100 * float(np.nanmean(mim['vasopressor'])),
          'eICU': 100 * float(np.nanmean(e['vasopressor'])),
          'NWICU': 100 * float(np.nanmean(nw['vasopressor'])),
          'INSPIRE': 100 * float(np.nanmean(ins['vasopressor']))}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    dbs = list(vp)
    axes[0].bar(range(len(dbs)), [vp[k] for k in dbs],
                color=['#4C72B0', '#55A868', '#C44E52', '#DD8452'])
    for i, k in enumerate(dbs):
        axes[0].text(i, vp[k] + 0.4, '%.1f%%' % vp[k], ha='center', fontsize=9)
    axes[0].set_xticks(range(len(dbs)))
    axes[0].set_xticklabels(dbs, rotation=15, fontsize=9)
    axes[0].set_ylabel('Vasopressor use on ICU day 1 (%)')
    axes[0].set_ylim(0, max(vp.values()) * 1.25)
    axes[0].set_title('(A) Vasopressor recording across databases')

    x = np.arange(len(AET))
    nwv = [int(nw[a].sum()) for a in AET]
    inv = [int(ins[a].sum()) for a in AET]
    axes[1].bar(x - 0.2, nwv, 0.4, label='NWICU', color='#4C72B0')
    axes[1].bar(x + 0.2, inv, 0.4, label='INSPIRE', color='#DD8452')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([a.upper() for a in AET], rotation=40, fontsize=8)
    axes[1].set_ylabel('Patients (n)')
    axes[1].legend(fontsize=9, frameon=False)
    axes[1].set_title('(B) Aetiology mix: NWICU vs INSPIRE')
    fig.tight_layout()
    fig.savefig(PAN / "figR3_rootcause_v9.png", dpi=200)
    plt.close(fig)
    log("  panel -> figR3_rootcause_v9.png  " +
        ", ".join("%s %.1f%%" % (k, v) for k, v in vp.items()))
    pd.DataFrame({'database': dbs, 'vasopressor_pct': [round(vp[k], 2) for k in dbs],
                  **{a: [int(nw[a].sum()), int(ins[a].sum()), None, None] for a in []}}
                 ).to_csv(OUT / "vasopressor_by_db_v9.csv", index=False)

    (OUT / "_95_v9_external_figures.md").write_text(
        "# 95 —— V9 外部验证图\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    log("\n→ output/v9_panels/*.png, external_validation_v9.csv, recalibration_v9.csv")


if __name__ == "__main__":
    main()
