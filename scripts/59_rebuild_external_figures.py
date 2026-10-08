# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
59_rebuild_external_figures.py  ——  修 Decision E 第 A 条（图 vs 表数字不一致）

背景（2026-09-30 实证核查）
--------------------------
output/recalibration.csv / recal_preds.joblib / external_validation.csv 是 07-29 的旧分析，
其中 eICU 用的是「1 年模型套到 eICU 院内结局」的旧口径：
    eICU Logistic AUC 0.763 / XGBoost 0.840，slope 0.55/0.79，O:E 0.42/0.40，ECE 0.179/0.188
而 V7 稿件 Table 4 与正文已经改成「同结局院内平行模型」口径：
    eICU Logistic AUC 0.811 / XGBoost 0.850，slope 0.675/0.703，O:E 0.81/0.59
Figure 4A / Figure S6 / Figure 5 下面板 / Figure 6 / Figure 7A 全部由旧产物绘制 → 图内仍是
0.763/0.840/0.179 等旧值。编辑看到的 "figures report values for eICU that differ from those in
the tables" 根因即此。

本脚本做两件事（两段式，强制"画图只能读落盘结果"）：
  STAGE 1  用冻结产物重算全部外验 + 重校准，落盘 external_current.json（唯一真源）
           —— eICU 用「院内平行模型」(model_inhosp.joblib)
           —— NWICU / INSPIRE 用「1 年模型」(preproc.joblib + models.joblib)
  STAGE 2  从 external_current.json **重新读回**再画图，落盘图内数字登记清单
           （图内每一个被打印出来的数字都登记在册），并与 Table 4 / 正文数字断言对齐。
           ⚠ 2026-10-07 起该清单写入 output/_archive_20261007/
             figure_number_manifest_v8legacy.json —— 它只反映 **V8 冻结口径**
             （本脚本读的是 output/external_current.json，含标签修正前的数字），
             V9 的数字基准是 output/v9_number_master.json。详见该目录 README_ARCHIVE.md。

不覆盖：本脚本只重写 output/fig*.png（旧版已备份到 output/_fig_backup_20260729/），
         不碰 output/submission_v7/ 下任何已交付文件。
"""
import json
import os
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

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
FEATS = ['age', 'female'] + AET + ['charlson_comorbidity_index'] + \
        [c for c in CONT if c not in ('age', 'charlson_comorbidity_index')] + BINV

EPS = 1e-6
JSON_PATH = OUT / "external_current.json"

# ★★ 2026-10-07 归档：本清单原写为 output/figure_number_manifest.json，
#   现已移入 output/_archive_20261007/figure_number_manifest_v8legacy.json。
#   归档原因：它**不带版本标记**，却承载两处结果标签修正之前的外验数字
#   （NWICU 1 年 Logistic AUC 0.6971 / 事件 635 / 原始斜率 0.039 —— V9 权威值是
#   0.7652 / 701 / 0.878）；且没有任何校验覆盖它（78 号的 G5 只管
#   v8_figure_number_manifest.json 与 v9_number_master.json），也不属于任何构建链，
#   于是永远停在旧值却最像"当前那一份"。实测代价：111 号一致性扫描初版以它为基准，
#   36 项数字全部被误报为"无出处"。
#   → V9 的数字基准是 output/v9_number_master.json；V8 的是
#     output/v8_figure_number_manifest.json。详见
#     output/_archive_20261007/README_ARCHIVE.md。
#   本脚本读的仍是 V8 冻结口径的 external_current.json，故其产物只能是 V8 回溯件。
ARCHIVE_DIR = OUT / "_archive_20261007"
MANIFEST_PATH = ARCHIVE_DIR / "figure_number_manifest_v8legacy.json"

# ============================================================================ #
#  通用统计函数
# ============================================================================ #
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold          # noqa: E402


def logit_f(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cal_slope_intercept(y, p):
    lp = logit_f(p)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=2000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


def oe_ratio(y, p):
    return float(np.mean(y) / np.mean(p))


def ece(y, p, bins=10):
    """等宽 10 分箱 ECE（与 07_recalibration.py 完全同口径）。"""
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


def boot_auc_ci(y, p, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    idx = np.arange(len(y))
    a = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        a.append(roc_auc_score(y[b], p[b]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def fit_intercept_only(y, lp):
    a = 0.0
    for _ in range(100):
        p = sigmoid(a + lp)
        g = np.sum(y - p)
        h = -np.sum(p * (1 - p))
        if abs(h) < 1e-12:
            break
        step = g / h
        a -= step
        if abs(step) < 1e-10:
            break
    return a


def fit_logistic_recal(y, lp):
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=2000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.intercept_[0]), float(m.coef_[0][0])


def cv_recalibrate(y, p, method, seed=7):
    lp = logit_f(p)
    oof = np.zeros_like(lp, dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(lp, y):
        if method == 'intercept':
            a = fit_intercept_only(np.asarray(y)[tr], lp[tr])
            oof[te] = sigmoid(a + lp[te])
        else:
            a, b = fit_logistic_recal(np.asarray(y)[tr], lp[tr])
            oof[te] = sigmoid(a + b * lp[te])
    return oof


def metrics(y, p):
    s, i = cal_slope_intercept(y, p)
    return {'brier': brier_score_loss(y, p), 'slope': s, 'intercept': i,
            'oe': oe_ratio(y, p), 'ece': ece(y, p)}


# ============================================================================ #
#  STAGE 1  重算
# ============================================================================ #
def make_predictor_1yr():
    pp = joblib.load(OUT / "preproc.joblib")
    models = joblib.load(OUT / "models.joblib")
    feats, cont = pp['feats'], pp['cont']
    scaler, medians = pp['scaler'], pd.Series(pp['medians'])
    logit = pd.Series(models['logit_params'])
    xgbm = models['xgb']

    def f(df):
        X = df[feats].copy()
        for c in cont:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(medians[c])
        for c in AET + BINV:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0).astype(int)
        Xs = X.copy()
        Xs[cont] = scaler.transform(X[cont])
        lp = logit['const'] + Xs[feats].values @ logit[feats].values
        return sigmoid(lp), xgbm.predict_proba(X[feats].values)[:, 1]
    return f


def make_predictor_inhosp():
    m = joblib.load(OUT / "model_inhosp.joblib")
    uf, c_in = m['feats'], m['cont']

    def f(df):
        X = df[uf].copy()
        for c in c_in:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(m['medians'][c])
        for c in [x for x in uf if x not in c_in]:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        Xs = X.copy()
        Xs[c_in] = m['scaler'].transform(X[c_in])
        return (m['lr'].predict_proba(Xs[uf])[:, 1],
                m['xg'].predict_proba(X[uf])[:, 1])
    return f


def main_stage1():
    log("=" * 74)
    log("STAGE 1  重算外验与重校准（eICU 用院内平行模型；NWICU/INSPIRE 用 1 年模型）")
    p1 = make_predictor_1yr()
    p2 = make_predictor_inhosp()

    nw = pd.read_csv(DATA / "nwicu_abi_final_vp2.csv")   # 74：恢复血管活性药
    ins = pd.read_csv(DATA / "inspire_abi_final.csv")
    dh = pd.read_csv(DATA / "inspire_deathhosp.csv")
    ins = ins.merge(dh, on='op_id', how='left')
    ins['death_hosp'] = ins['death_hosp'].fillna(0).astype(int)
    e = pd.read_csv(DATA / "eicu_abi_final.csv")

    spec = [
        # key,                db_label,                 outcome_label,     df,  ycol,          predictor
        ('NWICU_1yr', 'NWICU', '1-year death', nw, 'death_365', p1),
        ('INSPIRE_1yr', 'INSPIRE', '1-year death', ins, 'death_365', p1),
        ('eICU_inhosp', 'eICU', 'In-hospital death', e, 'death_hosp', p2),
        ('NWICU_inhosp', 'NWICU', 'In-hospital death', nw, 'death_hosp', p2),
        ('INSPIRE_inhosp', 'INSPIRE', 'In-hospital death', ins, 'death_hosp', p2),
    ]

    disc, recal, store = {}, {}, {}
    for key, db, olab, df, ycol, pred in spec:
        y = df[ycol].values.astype(int)
        plr, pxg = pred(df)
        disc[key] = {'database': db, 'outcome': olab, 'n': int(len(df)),
                     'events': int(y.sum()), 'prevalence_pct': round(100 * y.mean(), 1),
                     'Logistic': {}, 'XGBoost': {}}
        for mname, p in [('Logistic', plr), ('XGBoost', pxg)]:
            mt = metrics(y, p)
            disc[key][mname] = {
                'auc': round(float(roc_auc_score(y, p)), 4),
                'ci': [round(v, 4) for v in boot_auc_ci(y, p)],
                'brier': round(mt['brier'], 4), 'slope': round(mt['slope'], 3),
                'intercept': round(mt['intercept'], 3), 'oe': round(mt['oe'], 3),
                'ece': round(mt['ece'], 4)}
            log(f"  {key:<16} {mname:<8} AUC={disc[key][mname]['auc']:.3f} "
                f"slope={disc[key][mname]['slope']:.3f} O:E={disc[key][mname]['oe']:.3f} "
                f"ECE={disc[key][mname]['ece']:.3f}")

        # 只对进入正文重校准叙事的三个库做重校准
        if key in ('NWICU_1yr', 'INSPIRE_1yr', 'eICU_inhosp'):
            dbl = {'NWICU_1yr': 'NWICU (1-year)', 'INSPIRE_1yr': 'INSPIRE (1-year)',
                   'eICU_inhosp': 'eICU (in-hospital)'}[key]
            for mname, p in [('Logistic', plr), ('XGBoost', pxg)]:
                base = metrics(y, p)
                p_int = cv_recalibrate(y, p, 'intercept')
                p_log = cv_recalibrate(y, p, 'logistic')
                m_int, m_log = metrics(y, p_int), metrics(y, p_log)
                a_log, b_log = fit_logistic_recal(y, logit_f(p))
                recal[f"{dbl}|{mname}"] = {
                    'database': dbl, 'model': mname, 'n': int(len(y)), 'events': int(y.sum()),
                    'auc': round(float(roc_auc_score(y, p)), 4),
                    'oe_orig': round(base['oe'], 3), 'oe_log': round(m_log['oe'], 3),
                    'slope_orig': round(base['slope'], 3), 'slope_log': round(m_log['slope'], 3),
                    'intcpt_orig': round(base['intercept'], 3), 'intcpt_log': round(m_log['intercept'], 3),
                    'brier_orig': round(base['brier'], 4), 'brier_log': round(m_log['brier'], 4),
                    'ece_orig': round(base['ece'], 4), 'ece_int': round(m_int['ece'], 4),
                    'ece_log': round(m_log['ece'], 4),
                    'a_log': round(a_log, 3), 'b_log': round(b_log, 3)}
                store[(dbl, mname)] = {'y': y, 'orig': p, 'int': p_int, 'log': p_log}
            r = recal[f"{dbl}|Logistic"]
            log(f"    recal {dbl:<20} LR  O:E {r['oe_orig']:.2f}->{r['oe_log']:.2f} "
                f"slope {r['slope_orig']:.2f}->{r['slope_log']:.2f} ECE {r['ece_orig']:.3f}->{r['ece_log']:.3f}")

    # 院内平行模型在 MIMIC 的内部表现（来自冻结产物）
    mi = joblib.load(OUT / "model_inhosp.joblib")
    inhosp_internal = mi['internal']

    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'rule': ('eICU / Huaian = same-outcome in-hospital parallel model; '
                 'NWICU / INSPIRE = frozen 1-year model'),
        'source_artifacts': {
            'preproc.joblib': time.strftime('%Y-%m-%d %H:%M:%S',
                                            time.localtime(os.path.getmtime(OUT / 'preproc.joblib'))),
            'models.joblib': time.strftime('%Y-%m-%d %H:%M:%S',
                                           time.localtime(os.path.getmtime(OUT / 'models.joblib'))),
            'model_inhosp.joblib': time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(os.path.getmtime(OUT / 'model_inhosp.joblib')))},
        'discrimination': disc,
        'recalibration': recal,
        'inhosp_parallel_mimic_internal': inhosp_internal,
    }
    # 本地单中心（已由 24_local_inhosp_validation.py 产出，只做登记、不重算）
    lj = BASE / "ABI3" / "output" / "local_inhosp_validation.json"
    if lj.exists():
        payload['local_huaian'] = json.load(open(lj, encoding='utf-8'))

    json.dump(payload, open(JSON_PATH, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    joblib.dump(store, OUT / "recal_store_v2.joblib")
    rows = []

    def _flat(d):
        for k, v in d.items():
            r = {'cell': k}
            r.update(v)
            rows.append(r)
    _flat(recal)
    pd.DataFrame(rows).to_csv(OUT / "recalibration_v2.csv", index=False)
    log(f"\n  wrote {JSON_PATH.name} / recalibration_v2.csv / recal_store_v2.joblib")
    return payload


def log(*a):
    print(*a)
    sys.stdout.flush()


# ============================================================================ #
#  STAGE 2  从 JSON 读回再画图
# ============================================================================ #
def main_stage2():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    plt.rcParams.update({'font.size': 11, 'axes.linewidth': 0.8, 'font.family': 'DejaVu Sans'})
    D = json.load(open(JSON_PATH, encoding='utf-8'))
    disc = D['discrimination']
    rec = D['recalibration']
    store = joblib.load(OUT / "recal_store_v2.joblib")
    MAN = {}   # 图内数字清单

    def note(fig, item, value, src):
        MAN.setdefault(fig, []).append({'item': item, 'value': value, 'source': src})

    # ---------------- fig6_ext_roc  (Figure 4A: 1-year ROC in NWICU + INSPIRE) ----------
    # 图题写的是 "in NWICU and INSPIRE"，旧图却多画了一条 eICU（旧口径 0.840）。
    # 这里按图题只画两个 1 年库；eICU 归位到 Figure 3B（院内平行）。
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for key, col in [('NWICU_1yr', '#1D9E75'), ('INSPIRE_1yr', '#7F77DD')]:
        d = disc[key]
        yv = np.array([0] * (d['n'] - d['events']) + [1] * d['events'])
        p = store[({'NWICU_1yr': 'NWICU (1-year)', 'INSPIRE_1yr': 'INSPIRE (1-year)'}[key],
                   'XGBoost')]['orig']
        fpr, tpr, _ = roc_curve(yv, p)
        a = d['XGBoost']['auc']
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{d['database']} ({d['outcome']}): AUC {a:.3f}")
        note('fig6_ext_roc.png', f"{d['database']} XGBoost AUC", round(a, 3), f"discrimination.{key}.XGBoost.auc")
    ax.plot([0, 1], [0, 1], '--', color='#888780', lw=1)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('External validation ROC for 1-year mortality (XGBoost)')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(OUT / "fig6_ext_roc.png", dpi=200)
    plt.close()

    # ---------------- fig7_ext_roc_bymodel (Figure S6: 三库 x 两模型) -------------------
    dbmap = [('eICU_inhosp', 'eICU (in-hospital)'), ('NWICU_1yr', 'NWICU (1-year)'),
             ('INSPIRE_1yr', 'INSPIRE (1-year)')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (key, dbl) in zip(axes, dbmap):
        d = disc[key]
        yv = np.array([0] * (d['n'] - d['events']) + [1] * d['events'])
        for mname, c in [('Logistic', '#A32D2D'), ('XGBoost', '#185FA5')]:
            p = store[(dbl, mname)]['orig']
            fpr, tpr, _ = roc_curve(yv, p)
            a = d[mname]['auc']
            ax.plot(fpr, tpr, color=c, lw=2, label=f"{mname} {a:.3f}")
            note('fig7_ext_roc_bymodel.png', f"{dbl} {mname} AUC", round(a, 3),
                 f"discrimination.{key}.{mname}.auc")
        ax.plot([0, 1], [0, 1], '--', color='#888780', lw=1)
        ax.set_title(f"{d['database']} · {d['outcome']} (n={d['n']:,}, {d['events']:,} events)", fontsize=10)
        ax.set_xlabel('1 - Specificity')
        ax.legend(loc='lower right', fontsize=9, frameon=False)
        ax.set_aspect('equal')
    axes[0].set_ylabel('Sensitivity')
    plt.tight_layout()
    plt.savefig(OUT / "fig7_ext_roc_bymodel.png", dpi=200)
    plt.close()

    # ---------------- fig8_ext_calibration (Figure 4B: 重校准前) -----------------------
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for key, dbl, col in [('NWICU_1yr', 'NWICU (1-year)', '#1D9E75'),
                          ('INSPIRE_1yr', 'INSPIRE (1-year)', '#7F77DD')]:
        d = store[(dbl, 'XGBoost')]
        ct, cp = calibration_curve(d['y'], d['orig'], n_bins=8, strategy='quantile')
        ax.plot(cp, ct, 'o-', color=col, lw=1.8, ms=5, label=dbl)
    ax.plot([0, 1], [0, 1], '--', color='#888780', lw=1)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Initial calibration for 1-year mortality (XGBoost)')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "fig8_ext_calibration.png", dpi=200)
    plt.close()

    # ---------------- fig9_recal_calibration (Figure 6: 2x3 重校准前后) -----------------
    dbs = ['eICU (in-hospital)', 'NWICU (1-year)', 'INSPIRE (1-year)']
    mods = ['Logistic', 'XGBoost']
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9))
    for i, m in enumerate(mods):
        for j, db in enumerate(dbs):
            ax = axes[i, j]
            d = store[(db, m)]
            ax.plot([0, 1], [0, 1], '--', color='#888', lw=1, zorder=1)
            for p, lab, col in [(d['orig'], 'Original', '#c0392b'), (d['log'], 'Recalibrated', '#1f77b4')]:
                frac, mean = calibration_curve(d['y'], p, n_bins=8, strategy='quantile')
                ax.plot(mean, frac, 'o-', color=col, lw=1.8, ms=5, label=lab, zorder=3)
            r = rec[f"{db}|{m}"]
            txt = (f"Brier {r['brier_orig']:.3f}\u2192{r['brier_log']:.3f}\n"
                   f"ECE {r['ece_orig']:.3f}\u2192{r['ece_log']:.3f}")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
            note('fig9_recal_calibration.png', f"{db} {m} Brier orig\u2192log",
                 [r['brier_orig'], r['brier_log']], f"recalibration.{db}|{m}")
            note('fig9_recal_calibration.png', f"{db} {m} ECE orig\u2192log",
                 [r['ece_orig'], r['ece_log']], f"recalibration.{db}|{m}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{db} · {m}", fontsize=10, fontweight='bold')
            if j == 0:
                ax.set_ylabel('Observed frequency')
            if i == 1:
                ax.set_xlabel('Predicted probability')
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=8.5, frameon=True)
            ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle('External calibration before vs after logistic recalibration (5-fold CV)',
                 fontsize=12.5, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "fig9_recal_calibration.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---------------- fig10_recal_ece (Figure 5 下面板) --------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.2))
    labels = [f"{db.split(' (')[0]}\n{m}" for db in dbs for m in mods]
    eo = [rec[f"{db}|{m}"]['ece_orig'] for db in dbs for m in mods]
    ei = [rec[f"{db}|{m}"]['ece_int'] for db in dbs for m in mods]
    el = [rec[f"{db}|{m}"]['ece_log'] for db in dbs for m in mods]
    for db in dbs:
        for m in mods:
            r = rec[f"{db}|{m}"]
            note('fig10_recal_ece.png', f"{db} {m} ECE orig/int/log",
                 [r['ece_orig'], r['ece_int'], r['ece_log']], f"recalibration.{db}|{m}")
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, eo, w, label='Original', color='#c0392b')
    ax.bar(x, ei, w, label='Intercept-only', color='#e6a817')
    ax.bar(x + w, el, w, label='Logistic recal.', color='#1f77b4')
    for xi, (a, b, c) in enumerate(zip(eo, ei, el)):
        for off, v in [(-w, a), (0, b), (w, c)]:
            ax.text(xi + off, v + 0.003, f"{v:.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Expected Calibration Error (ECE)')
    ax.set_title('Calibration error before vs after recalibration', fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.set_ylim(0, max(eo + ei + el) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_recal_ece.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---------------- figR2_inhosp_parallel (Figure 3B) --------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ii = D['inhosp_parallel_mimic_internal']
    order = [('MIMIC_internal', 'MIMIC-IV\n(internal)'), ('eICU_inhosp', 'eICU'),
             ('INSPIRE_inhosp', 'INSPIRE')]
    lr, xg, labs = [], [], []
    for key, lab in order:
        if key == 'MIMIC_internal':
            a, b = ii['Logistic']['auc'], ii['XGBoost']['auc']
            src = 'inhosp_parallel_mimic_internal'
        else:
            a, b = disc[key]['Logistic']['auc'], disc[key]['XGBoost']['auc']
            src = f"discrimination.{key}"
        lr.append(a)
        xg.append(b)
        labs.append(lab)
        note('figR2_inhosp_parallel.png', f"{lab.replace(chr(10),' ')} Logistic AUC", round(a, 3), src)
        note('figR2_inhosp_parallel.png', f"{lab.replace(chr(10),' ')} XGBoost AUC", round(b, 3), src)
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w / 2, lr, w, label='Logistic', color='#4C72B0')
    ax.bar(x + w / 2, xg, w, label='XGBoost', color='#55A868')
    for xi, (a, b) in enumerate(zip(lr, xg)):
        ax.text(xi - w / 2, a + 0.005, f'{a:.3f}', ha='center', fontsize=9)
        ax.text(xi + w / 2, b + 0.005, f'{b:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel('AUROC (in-hospital mortality)')
    ax.set_ylim(0.5, 0.95)
    ax.set_title('Parallel validation on a common outcome (in-hospital death)')
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    nwd = disc['NWICU_inhosp']
    ax.text(0.5, 0.52,
            f"NWICU excluded (only {nwd['events']} in-hospital deaths of {nwd['n']:,}; "
            f"retained for 1-year validation)", fontsize=8, color='#777')
    note('figR2_inhosp_parallel.png', 'NWICU in-hospital events excluded',
         nwd['events'], 'discrimination.NWICU_inhosp.events')
    plt.tight_layout()
    plt.savefig(OUT / "figR2_inhosp_parallel.png", bbox_inches='tight')
    plt.close()

    # ---------------- figR4_external_dca (Figure 7A) -----------------------------------
    def net_benefit(y, p, th):
        y = np.asarray(y)
        N = len(y)
        nb = []
        for pt in th:
            pred = p >= pt
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            nb.append(tp / N - (fp / N) * (pt / (1 - pt)))
        return np.array(nb)

    th = np.arange(0.01, 0.60, 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, db in zip(axes, dbs):
        rec_ = store[(db, 'XGBoost')]
        y = np.asarray(rec_['y'])
        p = np.asarray(rec_['log'])
        prev = float(y.mean())
        nb_model = net_benefit(y, p, th)
        nb_all = prev - (1 - prev) * (th / (1 - th))
        ax.plot(th, nb_model, color='#1f4e79', lw=2.2, label='XGBoost (recalibrated)')
        ax.plot(th, nb_all, color='#b0891f', lw=1.4, ls='--', label='Treat all')
        ax.axhline(0, color='#888', lw=1.2, ls=':', label='Treat none')
        ax.set_xlim(0, 0.6)
        ax.set_ylim(min(-0.03, nb_model.min() - 0.01), max(nb_model.max(), prev) + 0.02)
        ax.set_title(f"{db.split(' (')[0]}  (event rate {prev*100:.1f}%)", fontsize=12, color='#1a1a1a')
        ax.set_xlabel('Threshold probability', fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('Net benefit', fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        note('figR4_external_dca.png', f"{db} event rate %", round(prev * 100, 1),
             f"recalibration.{db}|XGBoost (n/events)")
    fig.suptitle('External decision-curve analysis (recalibrated XGBoost)', fontsize=13.5,
                 y=1.02, color='#12324f')
    fig.tight_layout()
    fig.savefig(OUT / "figR4_external_dca.png", dpi=145, bbox_inches='tight', facecolor='white')
    plt.close()

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(MAN, open(MANIFEST_PATH, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    log(f"  wrote {MANIFEST_PATH.name}  ({sum(len(v) for v in MAN.values())} numbers registered)")
    log(f"  ⚠ 该清单是 **V8 冻结口径**（读 {JSON_PATH.name}），仅供 V8 回溯；"
        f"V9 的数字基准请用 output/v9_number_master.json")
    return MAN


# ============================================================================ #
#  STAGE 3  断言：图内数字 == 稿件表格数字
# ============================================================================ #
def main_stage3(MAN):
    log("=" * 74)
    log("STAGE 3  图内数字 vs V7 稿件 Table 4 / 正文 对齐断言")
    D = json.load(open(JSON_PATH, encoding='utf-8'))
    disc = D['discrimination']
    rec = D['recalibration']

    # Table 4 硬断言（编辑 A 条的核心）
    # ★ NWICU 两数已于 2026-10-01 更新：脚本 74 把 NWICU 的 vasopressor 从写死的 0
    #   恢复为真值（11.1%），故 0.696/0.796 → 0.697/0.801。改这个字面值前必须先跑 74。
    TABLE4 = {
        'NWICU_1yr': ('NWICU', 0.697, 0.801),
        'INSPIRE_1yr': ('INSPIRE', 0.729, 0.759),
        'eICU_inhosp': ('eICU', 0.811, 0.850),
    }
    ok = True
    for key, (lab, a_lr, a_xg) in TABLE4.items():
        g_lr = round(disc[key]['Logistic']['auc'], 3)
        g_xg = round(disc[key]['XGBoost']['auc'], 3)
        for mname, got, exp in [('Logistic', g_lr, a_lr), ('XGBoost', g_xg, a_xg)]:
            good = abs(got - exp) <= 0.001
            ok &= good
            log(f"  [{'OK ' if good else 'FAIL'}] Table4 {lab:<8} {mname:<8} recomputed={got:.3f} "
                f"manuscript={exp:.3f}")

    # 正文重校准段落（eICU: O:E 0.81/0.59, slope 0.68–0.70）
    for m, exp_oe, exp_slope in [('Logistic', 0.81, 0.675), ('XGBoost', 0.59, 0.703)]:
        r = rec[f"eICU (in-hospital)|{m}"]
        good_oe = abs(r['oe_orig'] - exp_oe) <= 0.03
        good_sl = abs(r['slope_orig'] - exp_slope) <= 0.03
        ok &= (good_oe and good_sl)
        log(f"  [{'OK ' if good_oe and good_sl else 'WARN'}] eICU {m:<8} O:E {r['oe_orig']:.3f} "
            f"(正文 {exp_oe})  slope {r['slope_orig']:.3f} (正文 {exp_slope})  "
            f"ECE {r['ece_orig']:.3f}->{r['ece_log']:.3f}")

    # NWICU 正文：Brier 0.245→0.150, slope 0.04→0.98
    r = rec["NWICU (1-year)|Logistic"]
    good = abs(r['brier_orig'] - 0.245) <= 0.01 and abs(r['slope_orig'] - 0.04) <= 0.02
    ok &= good
    log(f"  [{'OK ' if good else 'WARN'}] NWICU Logistic Brier {r['brier_orig']:.3f}->"
        f"{r['brier_log']:.3f} (正文 0.245->0.150)  slope {r['slope_orig']:.3f}->"
        f"{r['slope_log']:.3f} (正文 0.04->0.98)")

    # 图文件 mtime 必须晚于结果 JSON
    jt = os.path.getmtime(JSON_PATH)
    for f in ['fig6_ext_roc.png', 'fig7_ext_roc_bymodel.png', 'fig8_ext_calibration.png',
              'fig9_recal_calibration.png', 'fig10_recal_ece.png',
              'figR2_inhosp_parallel.png', 'figR4_external_dca.png']:
        ft = os.path.getmtime(OUT / f)
        good = ft >= jt
        ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] mtime {f:<30} "
            f"{time.strftime('%H:%M:%S', time.localtime(ft))} >= JSON "
            f"{time.strftime('%H:%M:%S', time.localtime(jt))}")

    # 图内不得再出现旧口径的 eICU 值
    bad_tokens = []
    for fig, items in MAN.items():
        for it in items:
            v = it['value']
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if isinstance(x, (int, float)) and abs(float(x) - 0.763) < 1e-9:
                    bad_tokens.append((fig, it['item'], x))
    if bad_tokens:
        ok = False
        log(f"  [FAIL] 旧口径 eICU 0.763 仍出现在图中: {bad_tokens}")
    else:
        log("  [OK ] 全部图内数字清单中已无旧口径 eICU 0.763")

    log("=" * 74)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED — see FAIL lines above")
    return 0 if ok else 1


if __name__ == '__main__':
    log("#" * 74)
    log("# 59_rebuild_external_figures.py  |  fix Decision-E item A (figure vs table)")
    log("#" * 74)
    main_stage1()
    MAN = main_stage2()
    sys.exit(main_stage3(MAN))
