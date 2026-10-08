# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
66_v8_audit_figures.py —— V8（跨库可移植性审计）的分析补算 + 正文图重排

背景：Decision E 的 A 条（图 vs 表不一致）根因之一是「手敲进正文、无结果文件出处」的数字。
本脚本即为消除这类数字而写：所有进图/进表的数字**必须**从 JSON 读回，图文件 mtime 必须
晚于其数据源 JSON 的 mtime（STAGE 4 断言）。

===============================================================================
PART 1  重新计算「两个静默陷阱」的 ΔAUC（此前 0.076 / −0.288 只存在于稿件正文里，
        全项目检索无任何结果文件出处 —— 必须重算，不能沿用）
        Threat U（测量单位）：淮安化验值不按 SI→传统单位换算，直接喂冻结模型
        Threat S（特征标度）：给 Logistic 喂未标准化特征 / 给 XGBoost 喂标准化特征
        每种扰动 × 2 个模型，报告 AUC 与 ΔAUC（配 DeLong）。
        同时把淮安的校准指标统一到 EPS=1e-6（与 62 一致；24 原用 1e-9，导致
        稿件 Table 5 slope 0.80 与 62 的 0.90 冲突）。
PART 2  6 张新面板（全部从 JSON 读数后绘制）
        gcs_components_v8.png   GCS 总分 vs 运动项增量 + GCS 单独按通气分层
        leakage_dauc_v8.png     剔除出院诊断病因码后的 ΔAUC（3 场景 × 内部/外部）
        slope_eps_v8.png        校准斜率对裁剪常数 ε 的敏感性
        calib_ci_v8.png         校准斜率 / 截距的 bootstrap 95% CI（森林式）
        huaian_panel_v8.png     淮安校准曲线 + ROC（ε=1e-6，替换 24 的旧面板）
PART 3  组合 8 张 V8 正文图（PNG / ≤1200×1200 / ≤5 MB / 无 alpha）
PART 4  自检：数字 vs JSON、像素/体积/通道、mtime、旧值残留（0.763 / 0.80 slope 陷阱）

产出（不覆盖任何已交付文件）：
  output/v8_harmonisation.json
  output/v8_panels/{5 张新面板}.png
  output/submission_v8/figures/Figure1..8.png
  output/submission_v8/figure_inventory_V8.json
  output/v8_figure_number_manifest.json
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.ticker import FuncFormatter                     # noqa: E402
from PIL import Image, ImageDraw, ImageFont                     # noqa: E402
from scipy import stats                                         # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score     # noqa: E402
from sklearn.preprocessing import StandardScaler                # noqa: E402

BASE = Path(ABI_BASE)
OUT = BASE / "output"
PAN = OUT / "v8_panels"
V8DIR = OUT / "submission_v8"
FIGDIR = V8DIR / "figures"
# ★ 版本化文件名（2026-10-07）：66 只产 V8，清单落 figure_inventory_V8.json；
#   V9 的清单由脚本 99 落 figure_inventory_V9.json。
MANIFEST = V8DIR / "figure_inventory_V8.json"

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
EPS = 1e-6          # ★ V8 全稿统一：校准指标的裁剪常数一律 1e-6
SEED = 42

# 正文图硬约束
MAX_PX = 1200
MAX_MB = 5.0

FAILS = []


def log(*a):
    print(*a)
    sys.stdout.flush()


def chk(cond, msg):
    tag = "OK " if cond else "FAIL"
    log("  [%s] %s" % (tag, msg))
    if not cond:
        FAILS.append(msg)


# --------------------------------------------------------------------------- #
#  通用指标（定义与 62 完全一致）
# --------------------------------------------------------------------------- #
def logit_f(p, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def cal_slope_intercept(y, p, eps=EPS):
    lp = logit_f(p, eps)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=2000).fit(
        lp.reshape(-1, 1), np.asarray(y, dtype=int))
    return float(m.coef_[0][0]), float(m.intercept_[0])


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


def all_metrics(y, p, eps=EPS):
    s, i = cal_slope_intercept(y, p, eps)
    return {'auc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
            'slope': s, 'intercept': i,
            'oe': float(np.mean(y) / np.mean(p)), 'ece': ece(y, p)}


def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, float)
    i = 0
    while i < n:
        j2 = i
        while j2 < n and z[j2] == z[i]:
            j2 += 1
        t[i:j2] = 0.5 * (i + j2 - 1) + 1
        i = j2
    a = np.zeros(n, float)
    a[j] = t
    return a


def delong_test(y, p1, p2):
    def dv(y, p):
        y, p = np.asarray(y), np.asarray(p)
        pos, neg = p[y == 1], p[y == 0]
        m, n = len(pos), len(neg)
        tx, ty = _midrank(pos), _midrank(neg)
        tz = _midrank(np.concatenate([pos, neg]))
        auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        return auc, v01, v10, m, n
    y = np.asarray(y)
    a1, v01a, v10a, m, n = dv(y, np.asarray(p1))
    a2, v01b, v10b, _, _ = dv(y, np.asarray(p2))
    S = np.cov(np.vstack([v01a, v01b])) / m + np.cov(np.vstack([v10a, v10b])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1), float(a2), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


# =========================================================================== #
#  PART 1  淮安：单位 / 标度 ΔAUC 重算 + ε 统一
# =========================================================================== #
def part1():
    log("=" * 96)
    log("PART 1 | 淮安单中心：单位 / 标度扰动的 ΔAUC 重算（旧值 +0.076 / −0.288 无出处，重算）")
    m = joblib.load(OUT / "model_inhosp.joblib")
    uf, uc = m['feats'], m['cont']
    medians, scaler, lr, xg = m['medians'], m['scaler'], m['lr'], m['xg']

    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')       # noqa: E731
    d = pd.DataFrame(index=raw.index)
    d['age'] = num('age')
    d['y'] = (num('hospital_discharge_location') == 5).astype(int)
    for c in AET:
        d[c] = num(c).fillna(0)
    keep = (d[AET].sum(axis=1) >= 1) & (d['age'] >= 18) & d['age'].notna() & \
        num('hospital_discharge_location').notna()
    dx = d[keep].copy()
    y = dx['y'].values.astype(int)

    def build():
        X = pd.DataFrame(index=dx.index)
        for c in uf:
            X[c] = num(c) if c in raw.columns else np.nan
        return X

    Xsi = build()                                   # SI 单位（医院原始上报）
    CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
            'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
    Xtr_ = build()                                  # 传统单位（换算后）
    for c, f in CONV.items():
        Xtr_[c] = f(Xtr_[c])

    def finish(X):
        """中位数插补（连续）+ 0 插补（其余），返回 (raw, standardized)。"""
        Z = X.copy()
        for c in uc:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(medians[c])
        for c in uf:
            if c not in uc:
                Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
        Zs = Z.copy()
        Zs[uc] = scaler.transform(Z[uc])
        return Z, Zs

    ref_raw, ref_s = finish(Xtr_)
    si_raw, si_s = finish(Xsi)

    # ---- 三种口径 ----
    VARIANTS = {
        # 名称: (给 Logistic 的矩阵, 给 XGBoost 的矩阵, 说明)
        'harmonised': (ref_s, ref_raw,
                       'Reference: SI→conventional unit conversion applied; '
                       'logistic fed standardised, XGBoost fed native-scale features'),
        'unit_discordant': (si_s, si_raw,
                            'Threat U: laboratory values fed in SI units without conversion'),
        'scale_discordant': (ref_raw, ref_s,
                             'Threat S: logistic fed unstandardised and XGBoost fed '
                             'standardised features (roles swapped)'),
    }
    res = {'n': int(len(y)), 'events': int(y.sum()),
           'eps': EPS, 'variants': {}, 'delta': {},
           'unit_conversion': {k: '÷88.4 (µmol/L→mg/dL)' if 'creat' in k else
                                  '÷0.357 (mmol/L→mg/dL)' if 'bun' in k else
                                  '×18 (mmol/L→mg/dL)' if 'glu' in k else
                                  '÷10 (g/L→g/dL)' for k in CONV},
           'medians_si_vs_conventional': {}}
    for c in CONV:
        res['medians_si_vs_conventional'][c] = {
            'SI_median': round(float(Xsi[c].median()), 2),
            'conventional_median': round(float(Xtr_[c].median()), 2)}

    preds = {}
    for tag, (Mlr, Mxg, desc) in VARIANTS.items():
        pl = lr.predict_proba(Mlr[uf])[:, 1]
        px = xg.predict_proba(Mxg[uf])[:, 1]
        preds[tag] = {'Logistic': pl, 'XGBoost': px}
        res['variants'][tag] = {'desc': desc,
                                # ★ 第八轮：全精度导出（呈现层只舍一次）
                                'Logistic': {k: float(v) for k, v in all_metrics(y, pl).items()},
                                'XGBoost': {k: float(v) for k, v in all_metrics(y, px).items()}}
        log("  %-18s LR AUC %.3f | XGB AUC %.3f" % (tag, roc_auc_score(y, pl), roc_auc_score(y, px)))

    for tag in ('unit_discordant', 'scale_discordant'):
        res['delta'][tag] = {}
        for mn in ('Logistic', 'XGBoost'):
            a_ref, a_alt, z, p = delong_test(y, preds['harmonised'][mn], preds[tag][mn])
            # ★ 第八轮：全精度导出（呈现层只舍一次）
            res['delta'][tag][mn] = {'auc_ref': float(a_ref), 'auc_alt': float(a_alt),
                                     'dAUC': float(a_alt - a_ref), 'z': float(z), 'p': p}
            log("  Δ %-18s %-8s %+.3f (%.3f→%.3f, DeLong p=%.3g)"
                % (tag, mn, a_alt - a_ref, a_ref, a_alt, p))

    # ---- 淮安校准指标（ε=1e-6 统一口径）----
    huaian = {'n': int(len(y)), 'events': int(y.sum()), 'eps': EPS}
    for mn, p in preds['harmonised'].items():
        # ★ 第八轮：全精度导出（呈现层只舍一次）
        huaian[mn] = {k: float(v) for k, v in all_metrics(y, p, EPS).items()}
    res['huaian_eps1e6'] = huaian
    log("  Huaian (ε=1e-6) LR slope %.3f int %.3f O:E %.3f | XGB slope %.3f int %.3f O:E %.3f"
        % (huaian['Logistic']['slope'], huaian['Logistic']['intercept'], huaian['Logistic']['oe'],
           huaian['XGBoost']['slope'], huaian['XGBoost']['intercept'], huaian['XGBoost']['oe']))

    json.dump(res, open(OUT / "v8_harmonisation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("  wrote v8_harmonisation.json")
    return res, y, preds


# =========================================================================== #
#  PART 2  5 张新面板
# =========================================================================== #
FS = 10
plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS + 1,
                     'xtick.labelsize': FS - 1, 'ytick.labelsize': FS - 1,
                     'legend.fontsize': FS - 1, 'figure.dpi': 200})
C_LR, C_XG, C_TOT, C_MOT = '#1f77b4', '#d62728', '#7f7f7f', '#2ca02c'


def _save(fig, name):
    PAN.mkdir(parents=True, exist_ok=True)
    fp = PAN / name
    fig.tight_layout()
    fig.savefig(fp, dpi=200)
    plt.close(fig)
    log("  panel -> %s" % name)
    return fp


def panel_gcs(aud):
    """GCS 总分 vs 运动项增量（审计发现：'省掉 GCS 几乎无代价'只在被插管污染的总分上成立）。"""
    g = aud['gcs_increment_decisionE']
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    # A：增量 ΔAUC
    models = ['Logistic', 'XGBoost']
    tot = [g[m_]['plus_total_gcs']['dAUC'] for m_ in models]
    mot = [g[m_]['plus_motor_gcs']['dAUC'] for m_ in models]
    x = np.arange(2)
    w = 0.36
    b1 = ax[0].bar(x - w / 2, tot, w, label='Add total GCS', color=C_TOT)
    b2 = ax[0].bar(x + w / 2, mot, w, label='Add motor GCS only', color=C_MOT)
    for b in list(b1) + list(b2):
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                   '%+.3f' % b.get_height(), ha='center', fontsize=FS - 1)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(models)
    ax[0].set_ylabel('Change in AUC')
    ax[0].set_title('A  Incremental value of the GCS', loc='left')
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].legend(frameon=False, loc='upper left')
    ax[0].set_ylim(0, max(max(tot), max(mot)) * 1.55)
    # B：GCS 单独（按通气分层）
    # ★ 铁律：图内数字只能从结果文件读回，不得硬编码。
    #   分层值取自 output/revision2.json（N3_gcs_sensitivity，MIMIC-IV 内部测试集 n=4,980）。
    rev = json.load(open(OUT / "revision2.json", encoding="utf-8"))
    ga = rev['N3_gcs_sensitivity']['gcs_alone_auc']
    assert abs(ga['ventilated']['auc'] - 0.488) < 1e-9, "revision2 ventilated AUC changed"
    assert abs(ga['nonventilated']['auc'] - 0.653) < 1e-9, "revision2 non-vent AUC changed"
    assert ga['ventilated']['n'] + ga['nonventilated']['n'] == ga['full_test']['n'], \
        "revision2 ventilation split does not sum to the test set"
    lab = ['Total GCS\nalone', 'Motor GCS\nalone', 'Total GCS alone\nventilated',
           'Total GCS alone\nnon-ventilated']
    val = [g['alone_AUC_logit']['gcs_total'], g['alone_AUC_logit']['gcs_motor'],
           ga['ventilated']['auc'], ga['nonventilated']['auc']]
    col = [C_TOT, C_MOT, '#bcbd22', '#9467bd']
    b = ax[1].bar(range(4), val, color=col)
    for r, v in zip(b, val):
        ax[1].text(r.get_x() + r.get_width() / 2, v + 0.012, '%.3f' % v, ha='center', fontsize=FS - 1)
    ax[1].axhline(0.5, color='k', ls='--', lw=1)
    ax[1].text(3.45, 0.505, 'chance', fontsize=FS - 2, ha='right')
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(lab, fontsize=FS - 2)
    ax[1].set_ylabel('AUC')
    ax[1].set_ylim(0.45, 0.72)
    ax[1].set_title('B  Discrimination of the GCS by itself', loc='left')
    return _save(fig, "gcs_components_v8.png")


def panel_leakage(lk):
    """剔除出院诊断病因码后的 ΔAUC（3 场景；MIMIC 无 POA/时间戳 → 只能报量级）。"""
    summ = lk['summary']
    order = ['A_1yr_full', 'B_inhosp', 'C_survivors']
    lab = ['A  One-year death\n(all patients)', 'B  In-hospital death\n(maximum leakage window)',
           'C  One-year death\n(hospital survivors)']
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    x = np.arange(3)
    w = 0.36
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        v = [summ[k][mn]['delta_no_aet'] for k in order]
        b = ax[0].bar(x + (j - 0.5) * w, v, w, label=mn, color=C_LR if j == 0 else C_XG)
        for r in b:
            ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() - 0.0022,
                       '%+.3f' % r.get_height(), ha='center', fontsize=FS - 2, color='w')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, fontsize=FS - 2)
    ax[0].set_ylabel('Change in AUC after removing\nall 7 discharge-coded aetiology indicators')
    ax[0].set_title('A  MIMIC-IV (internal, n=4,980)', loc='left')
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].legend(frameon=False)
    ax[0].set_ylim(-0.032, 0.004)

    ext = [('A_1yr_full|NWICU', 'NWICU\n(1-year)'), ('A_1yr_full|INSPIRE', 'INSPIRE\n(1-year)'),
           ('B_inhosp|eICU', 'eICU\n(in-hospital)'), ('B_inhosp|NWICU', 'NWICU\n(in-hospital)'),
           ('B_inhosp|INSPIRE', 'INSPIRE\n(in-hospital)'),
           ('C_survivors|NWICU', 'NWICU\n(survivors)'), ('C_survivors|INSPIRE', 'INSPIRE\n(survivors)')]
    v1 = [lk['external'][k]['Logistic']['delta'] for k, _ in ext]
    v2 = [lk['external'][k]['XGBoost']['delta'] for k, _ in ext]
    x2 = np.arange(len(ext))
    b1 = ax[1].bar(x2 - w / 2, v1, w, label='Logistic', color=C_LR)
    b2 = ax[1].bar(x2 + w / 2, v2, w, label='XGBoost', color=C_XG)
    for b in list(b1) + list(b2):
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + (0.0015 if b.get_height() >= 0 else -0.0045),
                   '%+.3f' % b.get_height(), ha='center', fontsize=FS - 3)
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels([t for _, t in ext], fontsize=FS - 3)
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_ylabel('Change in AUC')
    ax[1].set_ylim(-0.045, 0.05)
    ax[1].set_title('B  External databases', loc='left')
    return _save(fig, "leakage_dauc_v8.png")


def panel_slope_eps(se):
    """校准斜率对裁剪常数 ε 的敏感性（本轮新发现：小队列/触界预测下跨度可达 0.35）。"""
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    eps = [float(e) for e in se['eps_list']]
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        for tag, lab, ls in [('Huaian_inhosp', 'Huaian (n=225)', '-'),
                             ('NWICU_1yr', 'NWICU (n=3,420)', '--'),
                             ('INSPIRE_1yr', 'INSPIRE (n=1,543)', '-.'),
                             ('eICU_inhosp', 'eICU (n=14,852)', ':')]:
            ys = [se['results'][tag][mn]['by_eps']['%g' % e]['slope'] for e in eps]
            ax[j].plot(eps, ys, ls, marker='o', ms=4,
                       color=C_LR if mn == 'Logistic' else C_XG,
                       alpha=1.0 if 'Huaian' in lab else 0.75, label=lab)
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        ax[j].set_xscale('log')
        ax[j].set_xlabel('Clipping constant ε (predictions clipped to [ε, 1−ε] before logit)')
        ax[j].set_ylabel('Calibration slope')
        ax[j].axhline(1.0, color='k', ls=':', lw=0.8)
        ax[j].set_title('%s  %s' % ('A' if j == 0 else 'B', mn), loc='left')
        ax[j].legend(frameon=False, fontsize=FS - 2, loc='center left')
    return _save(fig, "slope_eps_v8.png")


def panel_calib_ci(cc):
    """校准斜率 / 截距的 bootstrap 95% CI（B=1000，百分位法）。"""
    tags = [('MIMIC_internal_1yr', 'MIMIC-IV\n(internal, n=4,980)'),
            ('NWICU_1yr', 'NWICU\n(n=3,420)'),
            ('INSPIRE_1yr', 'INSPIRE\n(n=1,543)'),
            ('eICU_inhosp', 'eICU\n(n=14,852)'),
            ('Huaian_inhosp', 'Huaian\n(n=225)')]
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        yy = np.arange(len(tags))[::-1]
        for i, (t, lab) in enumerate(tags):
            pt = cc['calibration_ci'][t][mn]['point']['slope']
            lo, hi = cc['calibration_ci'][t][mn]['ci95']['slope']
            ax[j].plot([lo, hi], [yy[i], yy[i]], color=C_LR if mn == 'Logistic' else C_XG, lw=2.2)
            ax[j].plot(pt, yy[i], 'o', color=C_LR if mn == 'Logistic' else C_XG, ms=6)
            ax[j].text(hi + 0.05, yy[i], '%.2f [%.2f–%.2f]' % (pt, lo, hi),
                       va='center', fontsize=FS - 2)
        ax[j].axvline(1.0, color='k', ls=':', lw=0.9)
        ax[j].axvline(0.0, color='gray', ls=':', lw=0.6)
        ax[j].set_yticks(yy)
        ax[j].set_yticklabels([t for _, t in tags], fontsize=FS - 2)
        ax[j].set_xlabel('Calibration slope (95%% CI, %d bootstrap resamples)'
                         % cc['bootstrap']['B'])
        ax[j].set_title('%s  %s' % ('A' if j == 0 else 'B', mn), loc='left')
        ax[j].set_xlim(-0.15, 2.15)
    return _save(fig, "calib_ci_v8.png")


def panel_harmonisation(res):
    """审计发现 5：单位 / 标度不协调的**作用靶点不同** ——
    线性模型判别力几乎不变但校准被打崩；树模型判别力被直接打击。
    ⚠ 旧稿把 +0.076 与 −0.288 分别记在"单位"与"标度"名下；重算显示二者**同属标度扰动**
      （XGBoost +0.0755 / Logistic −0.2881），单位扰动对 Logistic 判别力只有 +0.0001。"""
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.8))
    tags = ['unit_discordant', 'scale_discordant']
    lab = ['Units not converted\n(SI fed as if conventional)',
           'Scaling convention\nnot respected']
    x = np.arange(2)
    w = 0.36
    # A：ΔAUC
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        v = [res['delta'][t][mn]['dAUC'] for t in tags]
        b = ax[0].bar(x + (j - 0.5) * w, v, w, label=mn, color=C_LR if j == 0 else C_XG)
        for r in b:
            off = 0.006 if r.get_height() >= 0 else -0.022
            ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + off,
                       '%+.3f' % r.get_height(), ha='center', fontsize=FS - 1)
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, fontsize=FS - 2)
    ax[0].set_ylabel('Change in AUC')
    ax[0].set_ylim(-0.36, 0.13)
    ax[0].legend(frameon=False, loc='lower left')
    ax[0].set_title('A  Discrimination: the linear model is barely moved by units,\n'
                    'but collapses when the scaling convention is broken', loc='left',
                    fontsize=FS)
    # B：O:E（校准）
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        v = [res['variants'][t][mn]['oe'] for t in tags]
        b = ax[1].bar(x + (j - 0.5) * w, v, w, label=mn, color=C_LR if j == 0 else C_XG)
        for r in b:
            ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03,
                       '%.2f' % r.get_height(), ha='center', fontsize=FS - 1)
    ax[1].axhline(1.0, color='k', ls='--', lw=1)
    ax[1].text(1.45, 1.02, 'perfect calibration', fontsize=FS - 2, ha='right')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab, fontsize=FS - 2)
    ax[1].set_ylabel('Observed:expected ratio')
    ax[1].set_ylim(0, 2.05)
    ax[1].set_title('B  Calibration: unconverted units inflate the O:E ratio\n'
                    'of the linear model from 1.09 to 1.80', loc='left', fontsize=FS)
    return _save(fig, "harmonisation_v8.png")


def panel_huaian(res, y, preds):
    """淮安校准曲线 + ROC（ε=1e-6；替换 24 的旧面板，旧面板用 ε=1e-9 → slope 0.80）。"""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    # A：校准（10 等分位分箱）
    for mn, col in [('Logistic', C_LR), ('XGBoost', C_XG)]:
        p = preds['harmonised'][mn]
        q = np.quantile(p, np.linspace(0, 1, 11))
        q[0], q[-1] = 0.0, 1.0
        idx = np.clip(np.digitize(p, q[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() == 0:
                continue
            xs.append(p[m].mean())
            ys.append(y[m].mean())
        ax[0].plot(xs, ys, 'o-', color=col, label='%s (slope %.2f, intercept %+.2f)'
                   % (mn, res['huaian_eps1e6'][mn]['slope'], res['huaian_eps1e6'][mn]['intercept']))
    ax[0].plot([0, 1], [0, 1], 'k:', lw=1)
    ax[0].set_xlabel('Predicted probability of in-hospital death')
    ax[0].set_ylabel('Observed proportion')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=FS - 2, loc='upper left')
    ax[0].set_title('A  Calibration (Huaian, n=%d, %d events; ε=%g)'
                    % (res['n'], res['events'], EPS), loc='left')
    # B：ROC
    from sklearn.metrics import roc_curve
    for mn, col in [('Logistic', C_LR), ('XGBoost', C_XG)]:
        fpr, tpr, _ = roc_curve(y, preds['harmonised'][mn])
        ax[1].plot(fpr, tpr, color=col, lw=1.8,
                   label='%s AUC %.3f' % (mn, res['huaian_eps1e6'][mn]['auc']))
    ax[1].plot([0, 1], [0, 1], 'k:', lw=1)
    ax[1].set_xlabel('1 − specificity')
    ax[1].set_ylabel('Sensitivity')
    ax[1].legend(frameon=False, fontsize=FS - 2, loc='lower right')
    ax[1].set_title('B  Discrimination', loc='left')
    return _save(fig, "huaian_panel_v8.png")


# =========================================================================== #
#  PART 3  组合 8 张 V8 正文图
# =========================================================================== #
# (新图号, [(面板文件, 说明)], 版面, 面板标签, 图题)
PANEL_DIR = {"old": OUT, "new": PAN}
GROUPS = [
    dict(no=1, layout="h", labels=["A", "B"],
         panels=[("old", "fig1_roc.png"), ("old", "fig2_calibration.png")],
         cap="Development-cohort (MIMIC-IV) internal validation. (A) Receiver operating "
             "characteristic curves of the logistic, random forest and XGBoost models (all five "
             "candidate learners are compared in Table 2). (B) Calibration curves of the logistic "
             "and XGBoost models across ten quantile bins.",
         short="Internal validation (MIMIC-IV): discrimination and calibration"),
    dict(no=2, layout="h", labels=["A", "B"],
         panels=[("old", "fig4_forest.png"), ("old", "fig5_xgb_importance.png")],
         cap="Predictors of one-year mortality. (A) Adjusted odds ratios of the primary logistic "
             "model (per 1-SD increase for continuous variables; variables with P<.05 are shown). "
             "(B) XGBoost feature importance (gain) for the highest-ranked predictors.",
         short="Predictors: adjusted odds ratios and XGBoost importance"),
    dict(no=3, layout="h", labels=["A", "B"],
         panels=[("new", "gcs_components_v8.png")],
         cap="Audit finding 1: the cost of omitting the Glasgow Coma Scale (GCS). (A) Change in "
             "AUC when the total GCS score or the motor component alone is added to the GCS-free "
             "model (MIMIC-IV internal test set, n=4,980). (B) Discrimination of the GCS by "
             "itself, overall and stratified by mechanical ventilation; the total score is near "
             "chance in ventilated patients (AUC 0.488), which is where the GCS is most often "
             "recorded as intubation-adjusted.",
         short="Audit finding 1: what omitting the GCS actually costs"),
    dict(no=4, layout="v", labels=["A", "B"],
         panels=[("old", "fig6_ext_roc.png"), ("old", "fig8_ext_calibration.png")],
         cap="External validation for the one-year outcome. (A) Receiver operating characteristic "
             "curves in NWICU and INSPIRE. (B) Calibration curves before recalibration, showing "
             "the near-flat logistic slope in NWICU.",
         short="External validation: discrimination and initial calibration"),
    dict(no=5, layout="v", labels=["A", "B"],
         panels=[("old", "fig7_ext_roc_bymodel.png"), ("old", "fig9_recal_calibration.png")],
         cap="Discrimination by model and database, and recalibration. (A) Receiver operating "
             "characteristic curves of both models in every validation database. (B) Calibration "
             "curves after logistic recalibration, fitted in each external database.",
         short="Discrimination by model and post-recalibration calibration"),
    dict(no=6, layout="v", labels=["A", "B"],
         panels=[("old", "figR3_rootcause.png"), ("new", "leakage_dauc_v8.png")],
         cap="Audit findings 2 and 3: structural differences between databases, and how much the "
             "model depends on discharge-coded aetiology. Upper panel: vasopressor recording "
             "across databases (left; in NWICU the value shown is the prevalence recovered from "
             "the source prescription records during the audit, the harmonised file having "
             "recorded it as zero) and the aetiology-coding gap in INSPIRE (right). Lower panel: "
             "change in AUC after removing all seven "
             "discharge-coded aetiology indicators, internally across three label-timing "
             "scenarios (left) and in the external databases (right). MIMIC-IV carries no "
             "present-on-admission flag and no diagnosis timestamp, so the timing of these codes "
             "cannot be verified directly.",
         short="Audit findings 2-3: structural database differences and code-timing dependence"),
    # 原 no=7（外部决策曲线分析，figR4_external_dca.png）已于 2026-10-01 移入
    # Multimedia Appendix 1（Figure S8）：正文图数 9 > 建议上限 8。
    dict(no=7, layout="v", labels=["A", "B"],
         panels=[("new", "slope_eps_v8.png"), ("new", "calib_ci_v8.png")],
         cap="Audit finding 4: how stable are the calibration estimates? (A) The calibration "
             "slope depends on the constant ε used to clip predictions before the logit "
             "transformation; the dependence is negligible in the large databases but spans 0.35 "
             "in the 225-patient cohort. (B) Calibration slopes with 95% bootstrap confidence "
             "intervals (1,000 resamples); the intervals in the smallest cohort are too wide to "
             "support any calibration claim.",
         short="Audit finding 4: stability of the calibration estimates"),
    dict(no=8, layout="v", labels=["A", "B"],
         panels=[("new", "harmonisation_v8.png"), ("new", "huaian_panel_v8.png")],
         cap="Audit finding 5: measurement harmonisation at the independent site, and what the "
             "frozen model achieved there. (A) Change in AUC when laboratory units are not "
             "converted and when the feature-scaling convention is not respected (Huaian, n=225); "
             "the linear model's ranking survives unconverted units (ΔAUC +0.000) whereas the "
             "tree model does not. (B) Observed:expected ratio under the same two perturbations; "
             "unconverted units inflate the linear model's O:E ratio from 1.09 to 1.80 while "
             "leaving its AUC unchanged. Lower row: calibration and discrimination of the frozen "
             "in-hospital model after harmonisation.",
         short="Audit finding 5: harmonisation effects and independent-site performance"),
]

MARGIN, GAP, LABEL_H = 14, 16, 30


def _open(src):
    d, name = src
    return Image.open((PANEL_DIR[d] / name)).convert("RGB")


def compose():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    # 防陈图残留：图号减少（如 DCA 移附录）后，旧编号 PNG 必须先清掉，
    # 否则 64 的 zip 打包按 Figure1..N 通配会打包进过期图。
    for _stale in FIGDIR.glob("Figure*.png"):
        _stale.unlink()
    inv = []
    for g in GROUPS:
        ims = [_open(s) for s in g['panels']]
        labels = g['labels'] if len(g['labels']) == len(ims) else [""] * len(ims)
        if g['layout'] == "h":
            target_w = MAX_PX - 2 * MARGIN - GAP * (len(ims) - 1)
            scale = min(target_w / sum(i.width for i in ims),
                        (MAX_PX - 2 * MARGIN - LABEL_H) / max(i.height for i in ims))
            ims = [i.resize((max(1, int(i.width * scale)), max(1, int(i.height * scale))),
                            Image.LANCZOS) for i in ims]
            W = sum(i.width for i in ims) + GAP * (len(ims) - 1) + 2 * MARGIN
            H = max(i.height for i in ims) + LABEL_H + 2 * MARGIN
            canvas = Image.new("RGB", (W, H), "white")
            x = MARGIN
            for k, im in enumerate(ims):
                canvas.paste(im, (x, MARGIN + LABEL_H))
                if labels[k]:
                    dr = ImageDraw.Draw(canvas)
                    try:
                        ft = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
                    except Exception:
                        ft = ImageFont.load_default()
                    dr.text((x + 2, MARGIN + 4), labels[k], fill=(0, 0, 0), font=ft)
                x += im.width + GAP
        else:
            target_h = MAX_PX - 2 * MARGIN - LABEL_H * len(ims) - GAP * (len(ims) - 1)
            scale = min((MAX_PX - 2 * MARGIN) / max(i.width for i in ims),
                        target_h / sum(i.height for i in ims))
            ims = [i.resize((max(1, int(i.width * scale)), max(1, int(i.height * scale))),
                            Image.LANCZOS) for i in ims]
            W = max(i.width for i in ims) + 2 * MARGIN
            H = sum(i.height for i in ims) + LABEL_H * len(ims) + GAP * (len(ims) - 1) + 2 * MARGIN
            canvas = Image.new("RGB", (W, H), "white")
            y = MARGIN
            for k, im in enumerate(ims):
                if labels[k]:
                    dr = ImageDraw.Draw(canvas)
                    try:
                        ft = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
                    except Exception:
                        ft = ImageFont.load_default()
                    dr.text((MARGIN + 2, y + 4), labels[k], fill=(0, 0, 0), font=ft)
                canvas.paste(im, (MARGIN, y + LABEL_H))
                y += im.height + LABEL_H + GAP
        # 保险：仍在 1200 以内
        if W > MAX_PX or H > MAX_PX:
            s = MAX_PX / max(W, H)
            canvas = canvas.resize((int(W * s), int(H * s)), Image.LANCZOS)
        fp = FIGDIR / ("Figure%d.png" % g['no'])
        canvas.save(fp, "PNG", optimize=True)
        mb = os.path.getsize(fp) / 1e6
        log("  Figure %d -> %dx%d, %.2f MB (%d panels)"
            % (g['no'], canvas.width, canvas.height, mb, len(ims)))
        inv.append(dict(figure=g['no'], file=fp.name, px=[canvas.width, canvas.height],
                        size_mb=round(mb, 3), layout=g['layout'],
                        panels=[s[1] for s in g['panels']], panel_labels=labels,
                        caption=g['cap'], short_caption=g['short']))
    json.dump(inv, open(MANIFEST, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return inv


# =========================================================================== #
#  STAGE 4  自检
# =========================================================================== #
def verify(res, inv, srcs, lk, se, cc, aud):
    log("=" * 96)
    log("STAGE 4 | 自检")
    # A. 图源文件存在且 mtime 晚于其数据 JSON
    now = time.time()
    for name, dep in srcs:
        fp = PAN / name if (PAN / name).exists() else OUT / name
        chk(fp.exists(), "面板存在 %s" % name)
        if fp.exists() and dep:
            chk(os.path.getmtime(fp) >= os.path.getmtime(dep) - 1,
                "mtime %s (%s) >= %s (%s)"
                % (name, time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(fp))),
                   os.path.basename(dep), time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(dep)))))
    # B. 8 张正文图的硬约束
    for it in inv:
        fp = FIGDIR / it['file']
        im = Image.open(fp)
        chk(max(im.size) <= MAX_PX, "%s 最长边 %d <= %d" % (it['file'], max(im.size), MAX_PX))
        chk(it['size_mb'] <= MAX_MB, "%s %.2f MB <= %.1f" % (it['file'], it['size_mb'], MAX_MB))
        chk(im.mode in ("RGB", "L"), "%s 无 alpha 通道 (mode=%s)" % (it['file'], im.mode))
    # C. 两个威胁的**作用靶点不同**（本轮关键发现，写进正文）：
    #    单位不换算 → 线性模型**判别力几乎不变**（ΔAUC +0.0001，DeLong P=.996）而**校准被打崩**
    #    （O:E 1.09→1.80）；标度约定不遵守 → 线性模型判别力崩塌（−0.288）、树模型反而升高（+0.075）。
    du = res['delta']['unit_discordant']
    ds = res['delta']['scale_discordant']
    chk(abs(du['Logistic']['dAUC']) <= 0.005,
        "单位扰动：Logistic 判别力不变（ΔAUC %+.4f）" % du['Logistic']['dAUC'])
    oe0 = res['variants']['harmonised']['Logistic']['oe']
    oe1 = res['variants']['unit_discordant']['Logistic']['oe']
    chk(oe1 - oe0 > 0.5, "单位扰动：Logistic 校准崩塌（O:E %.2f → %.2f）" % (oe0, oe1))
    chk(abs(ds['Logistic']['dAUC']) > 0.2,
        "标度扰动：Logistic 判别力崩塌（ΔAUC %+.3f）" % ds['Logistic']['dAUC'])
    chk(abs(ds['XGBoost']['dAUC']) > 0.05,
        "标度扰动：XGBoost 判别力改变（ΔAUC %+.3f）" % ds['XGBoost']['dAUC'])
    chk(abs(du['XGBoost']['dAUC']) < 0.05,
        "单位扰动：XGBoost 判别力小幅下降（ΔAUC %+.3f）" % du['XGBoost']['dAUC'])
    # D. 淮安 ε=1e-6 校准值须与 62 一致
    h62 = cc['calibration_ci']['Huaian_inhosp']
    for mn in ('Logistic', 'XGBoost'):
        a = res['huaian_eps1e6'][mn]['slope']
        b = h62[mn]['point']['slope']
        chk(abs(a - b) <= 0.002, "淮安 %s slope %.3f == 62 的 %.3f" % (mn, a, b))
    # E. 旧的手敲值（+0.076 / −0.288）现已可追溯。重算结论：二者**同属标度扰动**，
    #    旧稿把 +0.076 归给"测量单位"是**错误归因**，V8 必须改写。此处只登记、不作 FAIL。
    log("  [NOTE] 旧稿手敲值溯源：+0.076 → 实为标度扰动 XGBoost ΔAUC %+.4f；"
        "−0.288 → 标度扰动 Logistic ΔAUC %+.4f；单位扰动 Logistic ΔAUC 仅 %+.4f"
        % (res['delta']['scale_discordant']['XGBoost']['dAUC'],
           res['delta']['scale_discordant']['Logistic']['dAUC'],
           res['delta']['unit_discordant']['Logistic']['dAUC']))
    # F. 审计数字与源 JSON 一致（进图的数字必须有出处）
    # ★ 第六轮（2026-10-01）：60/62 已改为**全精度导出**，源值不再等于 4 位舍入值，
    #   故此处按「显示精度」比较 —— 原先 `abs(x - (-0.0192)) < 1e-6` 只在源值
    #   本身就是舍入值时成立，是旧口径下的巧合，会对新成品误报。
    _lk_b = lk['summary']['B_inhosp']['Logistic']['delta_no_aet']
    chk("%.4f" % _lk_b == "-0.0192",
        "leakage 院内 ΔAUC(Logistic) = -0.0192（源 %.6f）" % _lk_b)
    chk(abs(se['results']['Huaian_inhosp']['Logistic']['slope_range'] - 0.3473) < 1e-3,
        "淮安斜率 ε 跨度 = 0.347")
    # ★ 第八轮（2026-10-02）：OASIS/SAPS-II 改用 mimiciv_derived 官方表后重算
    chk(abs(cc['severity_headtohead']['score_descriptives']['saps2']['auc_alone'] - 0.7800) < 1e-4,
        "SAPS-II AUC = 0.780（官方 mimiciv_derived.sapsii；手写版为 0.7845）")
    chk(abs(cc['severity_headtohead']['score_descriptives']['oasis']['auc_alone'] - 0.7202) < 1e-4,
        "OASIS AUC = 0.720（官方 mimiciv_derived.oasis；手写版为 0.7176）")
    _src = cc['severity_headtohead'].get('score_source', {})
    chk('mimiciv_derived.oasis' in _src.get('oasis', '')
        and 'mimiciv_derived.sapsii' in _src.get('saps2', ''),
        "严重度评分来源已标记为 mimiciv_derived 官方表（不再使用手写实现）")
    chk('handwritten_vs_official' in cc['severity_headtohead'],
        "JSON 保留 handwritten_vs_official 对照（可复核替换的实质影响）")
    chk(abs(aud['gcs_increment_decisionE']['Logistic']['plus_motor_gcs']['dAUC'] - 0.0154) < 1e-4,
        "运动项 GCS ΔAUC = +0.0154")


def main():
    t0 = time.strftime('%Y-%m-%d %H:%M:%S')
    log("66_v8_audit_figures.py  %s" % t0)
    V8DIR.mkdir(parents=True, exist_ok=True)
    res, y, preds = part1()

    log("=" * 96)
    log("PART 2 | 6 张新面板（全部从 JSON 读数后绘制）")
    aud = json.load(open(OUT / "gcs_increment_audit.json", encoding="utf-8"))
    lk = json.load(open(OUT / "leakage_sensitivity.json", encoding="utf-8"))
    se = json.load(open(OUT / "slope_epsilon_audit.json", encoding="utf-8"))
    cc = json.load(open(OUT / "calibration_ci_and_severity.json", encoding="utf-8"))
    panel_gcs(aud)
    panel_leakage(lk)
    panel_slope_eps(se)
    panel_calib_ci(cc)
    panel_harmonisation(res)
    panel_huaian(res, y, preds)

    log("=" * 96)
    log("PART 3 | 组合 %d 张 V8 正文图" % len(GROUPS))
    inv = compose()

    srcs = [("gcs_components_v8.png", OUT / "gcs_increment_audit.json"),
            ("leakage_dauc_v8.png", OUT / "leakage_sensitivity.json"),
            ("slope_eps_v8.png", OUT / "slope_epsilon_audit.json"),
            ("calib_ci_v8.png", OUT / "calibration_ci_and_severity.json"),
            ("harmonisation_v8.png", OUT / "v8_harmonisation.json"),
            ("huaian_panel_v8.png", OUT / "v8_harmonisation.json")]
    verify(res, inv, srcs, lk, se, cc, aud)

    # 图内数字清单（供 the archived-result verifier 与表对账）
    # ★ 第六轮（2026-10-01）：清单的值**一律由源 JSON 解析得到**，不再手敲字面量。
    #   手敲值在第四→五轮的重算中已静默失同步（NWICU ε 跨度 0.1291 → 实为 0.1312；
    #   NWICU 斜率 CI [0.0271, 0.0511] → 实为 [0.0276, 0.0516]）—— 正是本文所批评的
    #   "转录层静默失败"。改为派生后：源变则清单必然同步；source 路径打错则立即 KeyError。
    _rev2 = json.load(open(OUT / "revision2.json", encoding="utf-8"))
    _JS = {"audit_decisionE": aud, "leakage_sensitivity": lk, "slope_epsilon_audit": se,
           "calibration_ci_and_severity": cc, "v8_harmonisation": res, "revision2": _rev2}

    def _res(path):
        head, _, tail = path.partition(".")
        cur = _JS[head]
        for k in (tail.split(".") if tail else []):
            cur = cur[int(k)] if (isinstance(cur, list) and k.isdigit()) else cur[k]
        return cur

    def _q(path, nd=4):
        return float(("%%.%df" % nd) % float(_res(path)))

    def _ci(path, nd=4):
        return [_q(path + ".point.slope", nd),
                _q(path + ".ci95.slope.0", nd),
                _q(path + ".ci95.slope.1", nd)]

    _OE_PAIR = [_q("v8_harmonisation.variants.harmonised.Logistic.oe"),
                _q("v8_harmonisation.variants.unit_discordant.Logistic.oe")]
    _PREV = {("slope_eps_v8.png", "NWICU Logistic slope range over ε"): 0.1291,
             ("calib_ci_v8.png", "NWICU Logistic slope [CI]"): [0.039, 0.0271, 0.0511]}

    man = {
        "gcs_components_v8.png": [
            {"item": "Logistic ΔAUC + total GCS", "value": _q("audit_decisionE.gcs_increment_decisionE.Logistic.plus_total_gcs.dAUC"),
             "source": "audit_decisionE.gcs_increment_decisionE.Logistic.plus_total_gcs.dAUC"},
            {"item": "Logistic ΔAUC + motor GCS", "value": _q("audit_decisionE.gcs_increment_decisionE.Logistic.plus_motor_gcs.dAUC"),
             "source": "audit_decisionE.gcs_increment_decisionE.Logistic.plus_motor_gcs.dAUC"},
            {"item": "XGBoost ΔAUC + total GCS", "value": _q("audit_decisionE.gcs_increment_decisionE.XGBoost.plus_total_gcs.dAUC"),
             "source": "audit_decisionE.gcs_increment_decisionE.XGBoost.plus_total_gcs.dAUC"},
            {"item": "XGBoost ΔAUC + motor GCS", "value": _q("audit_decisionE.gcs_increment_decisionE.XGBoost.plus_motor_gcs.dAUC"),
             "source": "audit_decisionE.gcs_increment_decisionE.XGBoost.plus_motor_gcs.dAUC"},
            {"item": "Total GCS alone AUC", "value": _q("audit_decisionE.gcs_increment_decisionE.alone_AUC_logit.gcs_total"),
             "source": "audit_decisionE.gcs_increment_decisionE.alone_AUC_logit.gcs_total"},
            {"item": "Motor GCS alone AUC", "value": _q("audit_decisionE.gcs_increment_decisionE.alone_AUC_logit.gcs_motor"),
             "source": "audit_decisionE.gcs_increment_decisionE.alone_AUC_logit.gcs_motor"},
            {"item": "Total GCS alone, ventilated", "value": _q("revision2.N3_gcs_sensitivity.gcs_alone_auc.ventilated.auc", 3),
             "source": "revision2.N3_gcs_sensitivity.gcs_alone_auc.ventilated.auc"},
            {"item": "Total GCS alone, non-ventilated", "value": _q("revision2.N3_gcs_sensitivity.gcs_alone_auc.nonventilated.auc", 3),
             "source": "revision2.N3_gcs_sensitivity.gcs_alone_auc.nonventilated.auc"}],
        "leakage_dauc_v8.png": [
            {"item": "A 1-year ΔAUC Logistic", "value": _q("leakage_sensitivity.summary.A_1yr_full.Logistic.delta_no_aet"),
             "source": "leakage_sensitivity.summary.A_1yr_full.Logistic.delta_no_aet"},
            {"item": "B in-hospital ΔAUC Logistic", "value": _q("leakage_sensitivity.summary.B_inhosp.Logistic.delta_no_aet"),
             "source": "leakage_sensitivity.summary.B_inhosp.Logistic.delta_no_aet"},
            {"item": "C survivors ΔAUC Logistic", "value": _q("leakage_sensitivity.summary.C_survivors.Logistic.delta_no_aet"),
             "source": "leakage_sensitivity.summary.C_survivors.Logistic.delta_no_aet"},
            {"item": "eICU in-hospital ΔAUC Logistic", "value": _q("leakage_sensitivity.external.B_inhosp|eICU.Logistic.delta"),
             "source": "leakage_sensitivity.external.B_inhosp|eICU.Logistic.delta"}],
        "slope_eps_v8.png": [
            {"item": "Huaian Logistic slope range over ε", "value": _q("slope_epsilon_audit.results.Huaian_inhosp.Logistic.slope_range"),
             "source": "slope_epsilon_audit.results.Huaian_inhosp.Logistic.slope_range"},
            {"item": "NWICU Logistic slope range over ε", "value": _q("slope_epsilon_audit.results.NWICU_1yr.Logistic.slope_range"),
             "source": "slope_epsilon_audit.results.NWICU_1yr.Logistic.slope_range"}],
        "calib_ci_v8.png": [
            {"item": "MIMIC Logistic slope [CI]",
             "value": _ci("calibration_ci_and_severity.calibration_ci.MIMIC_internal_1yr.Logistic"),
             "source": "calibration_ci_and_severity.calibration_ci.MIMIC_internal_1yr.Logistic"},
            {"item": "NWICU Logistic slope [CI]",
             "value": _ci("calibration_ci_and_severity.calibration_ci.NWICU_1yr.Logistic"),
             "source": "calibration_ci_and_severity.calibration_ci.NWICU_1yr.Logistic"},
            {"item": "Huaian Logistic slope [CI]",
             "value": _ci("calibration_ci_and_severity.calibration_ci.Huaian_inhosp.Logistic"),
             "source": "calibration_ci_and_severity.calibration_ci.Huaian_inhosp.Logistic"}],
        "harmonisation_v8.png": [
            {"item": "Unit discordance ΔAUC Logistic", "value": _q("v8_harmonisation.delta.unit_discordant.Logistic.dAUC"),
             "source": "v8_harmonisation.delta.unit_discordant.Logistic.dAUC"},
            {"item": "Unit discordance O:E Logistic", "value": _OE_PAIR,
             "source": "v8_harmonisation.variants.{harmonised,unit_discordant}.Logistic.oe"},
            {"item": "Unit discordance ΔAUC XGBoost", "value": _q("v8_harmonisation.delta.unit_discordant.XGBoost.dAUC"),
             "source": "v8_harmonisation.delta.unit_discordant.XGBoost.dAUC"},
            {"item": "Scale discordance ΔAUC Logistic", "value": _q("v8_harmonisation.delta.scale_discordant.Logistic.dAUC"),
             "source": "v8_harmonisation.delta.scale_discordant.Logistic.dAUC"},
            {"item": "Scale discordance ΔAUC XGBoost", "value": _q("v8_harmonisation.delta.scale_discordant.XGBoost.dAUC"),
             "source": "v8_harmonisation.delta.scale_discordant.XGBoost.dAUC"}],
        "huaian_panel_v8.png": [
            {"item": "Huaian Logistic AUC", "value": _q("v8_harmonisation.huaian_eps1e6.Logistic.auc"),
             "source": "v8_harmonisation.huaian_eps1e6.Logistic.auc"},
            {"item": "Huaian Logistic slope (ε=1e-6)", "value": _q("v8_harmonisation.huaian_eps1e6.Logistic.slope"),
             "source": "v8_harmonisation.huaian_eps1e6.Logistic.slope"},
            {"item": "Huaian XGBoost AUC", "value": _q("v8_harmonisation.huaian_eps1e6.XGBoost.auc"),
             "source": "v8_harmonisation.huaian_eps1e6.XGBoost.auc"},
            {"item": "Huaian XGBoost slope (ε=1e-6)", "value": _q("v8_harmonisation.huaian_eps1e6.XGBoost.slope"),
             "source": "v8_harmonisation.huaian_eps1e6.XGBoost.slope"}],
    }
    # 自查 A：每个 source 都必须在源 JSON 里解析得到（含展开式 O:E 对）
    _n_src = 0
    for _f, _items in man.items():
        for _it in _items:
            _src = _it["source"]
            if "{" in _src:
                _n_src += 1
                continue
            _res(_src)
            _n_src += 1
    # 自查 B：本轮修正的 2 处失同步必须真的被改掉（否则就是没修）
    _fixed = []
    for _f, _items in man.items():
        for _it in _items:
            _k = (_f, _it["item"])
            if _k in _PREV and _it["value"] != _PREV[_k]:
                _fixed.append(("%s / %s" % (_f, _it["item"]), _PREV[_k], _it["value"]))
    for _nm, _ov, _nv in _fixed:
        log("  [FIXED] 图内数字清单失同步修正：%s   %s -> %s" % (_nm, _ov, _nv))
    assert len(_fixed) == 2, "应修正 2 处手敲失同步条目，实得 %d" % len(_fixed)
    log("  [OK] 图内数字清单 %d 项 / %d 条 source 全部由源 JSON 派生（无手敲值）"
        % (sum(len(v) for v in man.values()), _n_src))
    json.dump(man, open(OUT / "v8_figure_number_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log("  wrote v8_figure_number_manifest.json")

    log("=" * 96)
    if FAILS:
        log("FAILED CHECKS (%d):" % len(FAILS))
        for f in FAILS:
            log("  - %s" % f)
        log("RESULT: FAILED")
        return 1
    log("ALL CHECKS PASSED  (%d figures, %d new panels)" % (len(inv), len(srcs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
