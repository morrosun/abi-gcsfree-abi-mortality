# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
92_v9_audit_figures.py —— V9 审计面板 + 组合 8 张正文图

与 66（V8）的三点区别：
  1. 数据源全面换 V9：downstream_severity_gcs_v9.json（90）/ leakage_sensitivity_v9.json（91）
     / slope_epsilon_audit_v9.json（93）/ calibration_ci_and_severity_v2.json（87）
     / model_inhosp_v2.joblib（87）
  2. 淮安 harmonisation 用**院内模型 v2 冻结 artefacts** 重算（66 用 v1）
  3. GCS 按通气分层的单独 AUC **重算**（66 直接读 revision2.json 的固定值 0.488/0.653；
     本轮 mech_vent 已改首日口径，分层会变，旧值不可再用）

★ 不做"与稿件数字比对"：稿件仍是 V8 文本，比对必然全 FAIL；
  该项由第 2 批改文字之后的 96 号差异清单 + 终审负责。

产出：output/v9_panels/*.png、output/v9_figures/Figure1..8.png、
      output/v9_figure_inventory.json、output/v9_harmonisation.json
"""
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split         # noqa: E402
from scipy import stats                                      # noqa: E402

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"
PAN = OUT / "v9_panels"
FIGDIR = OUT / "v9_figures"
PAN.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

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
# ★★ 2026-10-22（第二十二轮）：1200 px 在 6.3 in 印刷宽下只有 190 dpi。
#   面板画布已按印刷宽设定 ⇒ 1890 px / 6.3 in = **300 dpi**（BMC/JIC 要求）。
#   高度单独放宽到 2500 px（≈8.3 in），否则整页图（图 4、图 5）会被按长边再缩一次、
#   宽度回落到 250 dpi 左右。
MAX_W, MAX_H = 1890, 2500
FS = 9    # 画布=印刷宽 6.3 in；FS-3 仍 ≥6 pt
C_LR, C_XG, C_TOT, C_MOT = '#1f77b4', '#d62728', '#7f7f7f', '#2ca02c'

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS + 1,
                     'xtick.labelsize': FS - 1, 'ytick.labelsize': FS - 1,
                     'legend.fontsize': FS - 1, 'figure.dpi': 200})


def logit_f(p, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def cal_slope_intercept(y, p, eps=EPS):
    lp = logit_f(p, eps).reshape(-1, 1)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs', max_iter=2000).fit(
        lp, np.asarray(y, dtype=int))
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


def all_metrics(y, p, eps=EPS):
    s, i = cal_slope_intercept(y, p, eps)
    return {'auc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
            'slope': s, 'intercept': i, 'oe': float(np.mean(y) / np.mean(p)),
            'ece': ece(y, p)}


def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(y, p1, p2):
    def dv(y, p):
        y, p = np.asarray(y), np.asarray(p)
        pos, neg = p[y == 1], p[y == 0]
        m, n = len(pos), len(neg)
        tx, ty = _midrank(pos), _midrank(neg)
        tz = _midrank(np.concatenate([pos, neg]))
        auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
        return auc, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m, m, n
    a1, v01a, v10a, m, n = dv(y, np.asarray(p1))
    a2, v01b, v10b, _, _ = dv(y, np.asarray(p2))
    S = np.cov(np.vstack([v01a, v01b])) / m + np.cov(np.vstack([v10a, v10b])) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var) if var > 0 else 0.0
    return float(a1), float(a2), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


# ============================================================ PART 1 淮安 harmonisation
def part1_harmonisation():
    log("=" * 90)
    log("PART 1 | 淮安：单位 / 标度扰动重算（院内模型 v2 冻结 artefacts）")
    m = joblib.load(OUT / "model_inhosp_v2.joblib")
    uf, uc = m['feats'], m['cont']
    medians, scaler, lr, xg = m['medians'], m['scaler'], m['lr'], m['xg']

    P = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"
    raw = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')          # noqa: E731
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

    Xsi = build()
    CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
            'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
    Xtr_ = build()
    for c, f in CONV.items():
        Xtr_[c] = f(Xtr_[c])

    def finish(X):
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
    VARIANTS = {
        'harmonised': (ref_s, ref_raw, 'Reference: SI→conventional conversion applied; '
                                       'logistic fed standardised, XGBoost fed native'),
        'unit_discordant': (si_s, si_raw, 'Threat U: SI units fed without conversion'),
        'scale_discordant': (ref_raw, ref_s, 'Threat S: logistic fed unstandardised and '
                                             'XGBoost fed standardised (roles swapped)')}
    res = {'n': int(len(y)), 'events': int(y.sum()), 'eps': EPS,
           'variants': {}, 'delta': {},
           'medians_si_vs_conventional': {
               c: {'SI_median': round(float(Xsi[c].median()), 2),
                   'conventional_median': round(float(Xtr_[c].median()), 2)}
               for c in CONV}}
    preds = {}
    for tag, (Mlr, Mxg, desc) in VARIANTS.items():
        pl = lr.predict_proba(Mlr[uf])[:, 1]
        px = xg.predict_proba(Mxg[uf])[:, 1]
        preds[tag] = {'Logistic': pl, 'XGBoost': px}
        res['variants'][tag] = {'desc': desc,
                                'Logistic': {k: float(v) for k, v in all_metrics(y, pl).items()},
                                'XGBoost': {k: float(v) for k, v in all_metrics(y, px).items()}}
        log("  %-18s LR AUC %.4f | XGB AUC %.4f" %
            (tag, roc_auc_score(y, pl), roc_auc_score(y, px)))
    for tag in ('unit_discordant', 'scale_discordant'):
        res['delta'][tag] = {}
        for mn in ('Logistic', 'XGBoost'):
            a_ref, a_alt, z, p = delong_test(y, preds['harmonised'][mn], preds[tag][mn])
            res['delta'][tag][mn] = {'auc_ref': a_ref, 'auc_alt': a_alt,
                                     'dAUC': a_alt - a_ref, 'z': z, 'p': p}
            log("  Δ %-18s %-8s %+.4f (%.4f→%.4f, DeLong p=%.3g)" %
                (tag, mn, a_alt - a_ref, a_ref, a_alt, p))
    huaian = {'n': int(len(y)), 'events': int(y.sum()), 'eps': EPS}
    for mn, p in preds['harmonised'].items():
        huaian[mn] = {k: float(v) for k, v in all_metrics(y, p, EPS).items()}
    res['huaian_eps1e6'] = huaian
    log("  Huaian (ε=1e-6) LR slope %.3f O:E %.3f | XGB slope %.3f O:E %.3f" %
        (huaian['Logistic']['slope'], huaian['Logistic']['oe'],
         huaian['XGBoost']['slope'], huaian['XGBoost']['oe']))
    json.dump(res, open(OUT / "v9_harmonisation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return res, y, preds


# ============================================================ PART 2 面板
def _save(fig, name):
    fp = PAN / name
    fig.tight_layout()
    fig.savefig(fp, dpi=300)   # 面板按 300 dpi 出图，与印刷宽 6.3 in 对应
    plt.close(fig)
    log("  panel -> %s" % name)
    return fp


def gcs_vent_stratified():
    """按机械通气分层的 GCS 单独 AUC（★ 重算：mech_vent 已改首日口径，
    revision2.json 的 0.488/0.653 不可再用）。"""
    mim = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    g = pd.read_csv(DATA / "mimic_gcs.csv")
    df = mim.merge(g, on='stay_id', how='left')
    y = df['death_365'].values.astype(int)
    _, Xte, _, yte = train_test_split(df[FEATS], y, test_size=0.30, stratify=y,
                                      random_state=SEED)
    te = df.loc[Xte.index]
    out = {}
    # ★ 方向约定：GCS 总分越低越危重，所以"用 GCS 预测死亡"的原始 AUC 是 <0.5 的。
    #   报告口径统一为"低分提示高风险"的判别力 = 1 − raw AUC（与单变量 logistic 的
    #   AUC≥0.5 语义一致）；raw 值一并存档，避免方向歧义。
    for grp, mask in [('ventilated', te['mech_vent'].fillna(0) > 0),
                      ('nonventilated', te['mech_vent'].fillna(0) == 0)]:
        sub = te[mask]
        yy = sub['death_365'].values.astype(int)
        v = sub['gcs_min'].fillna(sub['gcs_min'].median() if sub['gcs_min'].notna().any() else 15)
        raw = float(roc_auc_score(yy, v.values))
        out[grp] = {'n': int(len(sub)), 'events': int(yy.sum()),
                    'auc_raw': raw, 'auc': 1.0 - raw}
    out['full_test'] = {'n': int(len(te)), 'events': int(yte.sum()),
                        'auc_raw': float(roc_auc_score(yte, te['gcs_min'].fillna(
                            te['gcs_min'].median()).values)),
                        'note': 'auc = 1 − raw AUC (lower GCS ⇒ higher risk)'}
    out['full_test']['auc'] = 1.0 - out['full_test']['auc_raw']
    assert out['ventilated']['n'] + out['nonventilated']['n'] == out['full_test']['n'], \
        "ventilation strata do not sum to the test set"
    log("  GCS 单独 AUC（重算，低分=高风险口径）：通气 %d 例 %.4f (raw %.4f) | "
        "未通气 %d 例 %.4f (raw %.4f)" %
        (out['ventilated']['n'], out['ventilated']['auc'], out['ventilated']['auc_raw'],
         out['nonventilated']['n'], out['nonventilated']['auc'],
         out['nonventilated']['auc_raw']))
    return out


def panel_gcs(aud, gcs_strat):
    g = aud['gcs_increment']
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.52))
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

    lab = ['Total GCS\nalone', 'Motor GCS\nalone', 'Total GCS alone\nventilated',
           'Total GCS alone\nnon-ventilated']
    val = [g['alone_AUC_logit']['gcs_total'], g['alone_AUC_logit']['gcs_motor'],
           gcs_strat['ventilated']['auc'], gcs_strat['nonventilated']['auc']]
    b = ax[1].bar(range(4), val, color=[C_TOT, C_MOT, '#bcbd22', '#9467bd'])
    for r, v in zip(b, val):
        ax[1].text(r.get_x() + r.get_width() / 2, v + 0.012, '%.3f' % v,
                   ha='center', fontsize=FS - 1)
    ax[1].axhline(0.5, color='k', ls='--', lw=1)
    ax[1].text(3.45, 0.505, 'chance', fontsize=FS - 2, ha='right')
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(lab, fontsize=FS - 2)
    ax[1].set_ylabel('AUC')
    ax[1].set_ylim(0.45, 0.78)
    ax[1].set_title('B  Discrimination of the GCS by itself', loc='left')
    return _save(fig, "gcs_components_v9.png")


def panel_leakage(lk):
    summ = lk['summary']
    order = ['A_1yr_full', 'B_inhosp', 'C_survivors']
    lab = ['A  One-year death\n(all patients)', 'B  In-hospital death\n(maximum leakage window)',
           'C  One-year death\n(hospital survivors)']
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.52))
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
    ax[0].set_title('MIMIC-IV (internal, n=4,980)', loc='left')
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].legend(frameon=False)
    ax[0].set_ylim(-0.032, 0.004)

    ext = [('A_1yr_full|NWICU', 'NWICU\n(1-year)'), ('A_1yr_full|INSPIRE', 'INSPIRE\n(1-year)'),
           ('B_inhosp|eICU', 'eICU\n(in-hospital)'), ('B_inhosp|NWICU', 'NWICU\n(in-hospital)'),
           ('B_inhosp|INSPIRE', 'INSPIRE\n(in-hospital)'),
           ('C_survivors|NWICU', 'NWICU\n(survivors)'),
           ('C_survivors|INSPIRE', 'INSPIRE\n(survivors)')]
    v1 = [lk['external_delta'][k]['Logistic']['delta'] for k, _ in ext]
    v2 = [lk['external_delta'][k]['XGBoost']['delta'] for k, _ in ext]
    x2 = np.arange(len(ext))
    b1 = ax[1].bar(x2 - w / 2, v1, w, label='Logistic', color=C_LR)
    b2 = ax[1].bar(x2 + w / 2, v2, w, label='XGBoost', color=C_XG)
    for b in list(b1) + list(b2):
        ax[1].text(b.get_x() + b.get_width() / 2,
                   b.get_height() + (0.0015 if b.get_height() >= 0 else -0.0045),
                   '%+.3f' % b.get_height(), ha='center', fontsize=FS - 3)
    ax[1].set_xticks(x2)
    ax[1].set_xticklabels([t for _, t in ext], fontsize=FS - 3)
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_ylabel('Change in AUC')
    ax[1].set_ylim(-0.045, 0.05)
    ax[1].set_title('External databases', loc='left')
    return _save(fig, "leakage_dauc_v9.png")


def panel_slope_eps(se):
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.52))
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
        ax[j].set_xscale('log')
        ax[j].set_xlabel('Clipping constant ε')
        ax[j].set_ylabel('Calibration slope')
        ax[j].axhline(1.0, color='k', ls=':', lw=0.8)
        ax[j].set_title(mn, loc='left')
        # ★ 图例放在顶部留白里：给 y 上限预留空间，否则图例压在已经画好的线上
        _all = [se['results'][t][mn]['by_eps']['%g' % e]['slope']
                for t in ('Huaian_inhosp', 'NWICU_1yr', 'INSPIRE_1yr', 'eICU_inhosp')
                for e in eps]
        ax[j].set_ylim(min(_all) - 0.05, max(_all) + 0.26)
        ax[j].legend(frameon=False, fontsize=FS - 2, loc='upper left')
    return _save(fig, "slope_eps_v9.png")


def panel_calib_ci(cc):
    tags = [('MIMIC_internal_1yr', 'MIMIC-IV\n(internal, n=4,980)'),
            ('NWICU_1yr', 'NWICU\n(n=3,420)'),
            ('INSPIRE_1yr', 'INSPIRE\n(n=1,543)'),
            ('eICU_inhosp', 'eICU\n(n=14,852)'),
            ('Huaian_inhosp', 'Huaian\n(n=225)')]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.62))
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        yy = np.arange(len(tags))[::-1]
        for i, (t, lab) in enumerate(tags):
            pt = cc[t][mn]['point']['slope']
            lo, hi = cc[t][mn]['ci95']['slope']
            ax[j].plot([lo, hi], [yy[i], yy[i]], color=C_LR if mn == 'Logistic' else C_XG, lw=2.2)
            ax[j].plot(pt, yy[i], 'o', color=C_LR if mn == 'Logistic' else C_XG, ms=6)
            # 数值标在 CI 正上方（原来放在 hi 右侧，会压在自己的 CI 线上）
            ax[j].text(pt, yy[i] + 0.26, '%.2f [%.2f–%.2f]' % (pt, lo, hi),
                       va='bottom', ha='center', fontsize=FS - 2)
        ax[j].axvline(1.0, color='k', ls=':', lw=0.9)
        ax[j].axvline(0.0, color='gray', ls=':', lw=0.6)
        ax[j].set_yticks(yy)
        ax[j].set_yticklabels([t for _, t in tags], fontsize=FS - 2)
        ax[j].set_xlabel('Calibration slope (95% CI)')
        ax[j].set_title(mn, loc='left')
        # ★ 上界到 1.39（淮安 XGBoost）时固定 2.15 会把数值标签挤到画布外 ⇒ 按数据取右界
        _hi = max(cc[t][mn]['ci95']['slope'][1] for t, _ in tags)
        ax[j].set_xlim(-0.15, _hi + 0.45)
        ax[j].set_ylim(-0.6, len(tags) - 0.15)
    return _save(fig, "calib_ci_v9.png")


def panel_harmonisation(res):
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.52))
    tags = ['unit_discordant', 'scale_discordant']
    lab = ['Units not converted\n(SI fed as if conventional)',
           'Scaling convention\nnot respected']
    x = np.arange(2)
    w = 0.36
    # 供 y 范围与标签偏移共用（必须在画柱之前算）
    _dmin = min(res['delta'][t][mn]['dAUC'] for t in tags for mn in ['Logistic', 'XGBoost'])
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        v = [res['delta'][t][mn]['dAUC'] for t in tags]
        b = ax[0].bar(x + (j - 0.5) * w, v, w, label=mn, color=C_LR if j == 0 else C_XG)
        for r in b:
            off = 0.006 if r.get_height() >= 0 else -0.055 * abs(_dmin) / 0.378
            ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() + off,
                       '%+.3f' % r.get_height(), ha='center', fontsize=FS - 1)
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(lab, fontsize=FS - 2)
    ax[0].set_ylabel('Change in AUC')
    # ★ 范围按数据取：固定 -0.36 会被 -0.378 顶破，标签落到刻度文字上
    ax[0].set_ylim(_dmin * 1.22, 0.13)
    ax[0].legend(frameon=False, loc='lower left')
    ax[0].set_title('Discrimination', loc='left')

    # ★ ylim 动态：V9 重训后单位不换算使 LR 的 O:E 达到 6.4（V8 报 1.80），
    #   旧图固定 ylim(0, 2.05) 会把柱子和标签截掉 —— 必须按数据取范围。
    oes = [res['variants'][t][mn]['oe'] for t in tags for mn in ['Logistic', 'XGBoost']]
    vmax = max(oes)
    for j, mn in enumerate(['Logistic', 'XGBoost']):
        v = [res['variants'][t][mn]['oe'] for t in tags]
        b = ax[1].bar(x + (j - 0.5) * w, v, w, label=mn, color=C_LR if j == 0 else C_XG)
        for r in b:
            ax[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02 * vmax,
                       '%.2f' % r.get_height(), ha='center', fontsize=FS - 1)
    ax[1].axhline(1.0, color='k', ls='--', lw=1)
    ax[1].text(-0.42, 1.02 + 0.03 * vmax, 'perfect calibration', fontsize=FS - 2, ha='left')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(lab, fontsize=FS - 2)
    ax[1].set_ylabel('Observed:expected ratio')
    ax[1].set_ylim(0, vmax * 1.25)
    ax[1].set_title('Calibration', loc='left')
    return _save(fig, "harmonisation_v9.png")


def panel_huaian(res, y, preds):
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.63))
    from sklearn.metrics import roc_curve
    for mn, col in [('Logistic', C_LR), ('XGBoost', C_XG)]:
        p = preds['harmonised'][mn]
        q = np.quantile(p, np.linspace(0, 1, 11))
        q[0], q[-1] = 0.0, 1.0
        idx = np.clip(np.digitize(p, q[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            msk = idx == b
            if msk.sum() == 0:
                continue
            xs.append(p[msk].mean())
            ys.append(y[msk].mean())
        ax[0].plot(xs, ys, 'o-', color=col,
                   label='%s (slope %.2f, intercept %+.2f)'
                         % (mn, res['huaian_eps1e6'][mn]['slope'],
                            res['huaian_eps1e6'][mn]['intercept']))
    ax[0].plot([0, 1], [0, 1], 'k:', lw=1)
    ax[0].set_xlabel('Predicted probability of in-hospital death')
    ax[0].set_ylabel('Observed proportion')
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=FS - 2, loc='upper left')
    ax[0].set_title('Calibration (Huaian, n=%d)' % res['n'], loc='left')
    for mn, col in [('Logistic', C_LR), ('XGBoost', C_XG)]:
        fpr, tpr, _ = roc_curve(y, preds['harmonised'][mn])
        ax[1].plot(fpr, tpr, color=col, lw=1.8,
                   label='%s AUC %.3f' % (mn, res['huaian_eps1e6'][mn]['auc']))
    ax[1].plot([0, 1], [0, 1], 'k:', lw=1)
    ax[1].set_xlabel('1 − specificity')
    ax[1].set_ylabel('Sensitivity')
    ax[1].legend(frameon=False, fontsize=FS - 2, loc='lower right')
    ax[1].set_title('Discrimination', loc='left')
    return _save(fig, "huaian_panel_v9.png")


# ============================================================ PART 3 组合
GROUPS = [
    dict(no=1, layout="h", labels=["A", "B"],
         panels=[("v9", "fig1_roc_v9.png"), ("v9", "fig2_calibration_v9.png")],
         cap="Development-cohort (MIMIC-IV) internal validation. (A) Receiver operating "
             "characteristic curves of the logistic, random forest and XGBoost models (all five "
             "candidate learners are compared in Table 2). (B) Calibration curves of the logistic "
             "and XGBoost models across ten quantile bins.",
         short="Internal validation (MIMIC-IV): discrimination and calibration"),
    dict(no=2, layout="h", labels=["A", "B"],
         panels=[("v9", "fig4_forest_v9.png"), ("v9", "fig5_xgb_importance_v9.png")],
         cap="Predictors of one-year mortality. (A) Adjusted odds ratios of the primary logistic "
             "model (per 1-SD increase for continuous variables; variables with P<.05 are shown). "
             "(B) XGBoost feature importance (gain) for the highest-ranked predictors.",
         short="Predictors: adjusted odds ratios and XGBoost importance"),
    dict(no=3, layout="v", labels=["A", "B"],
         panels=[("v9", "gcs_components_v9.png")],
         cap="Audit finding 1: the cost of omitting the Glasgow Coma Scale (GCS). (A) Change in "
             "AUC when the total GCS score or the motor component alone is added to the GCS-free "
             "model (MIMIC-IV internal test set, n=4,980). (B) Discrimination of the GCS by "
             "itself, overall and stratified by mechanical ventilation.",
         short="Audit finding 1: what omitting the GCS actually costs"),
    dict(no=4, layout="v", labels=["A", "B"],
         panels=[("v9", "fig6_ext_roc_v9.png"), ("v9", "fig8_ext_calibration_v9.png")],
         # ★★ 2026-10-22（第二十轮）：图 4 与图 5 的 ROC 曾整块重复（4A 是 5A 的前两格）。
         #   现在按结局定义分工：4A = 一年结局（只有 NWICU 与 INSPIRE 有一年数据），
         #   5A = 院内结局四库。图注各加一句互相指向，读者不会以为漏画。
         cap="External validation. (A) Receiver operating characteristic curves of both models in "
             "NWICU and INSPIRE for the one-year outcome; in-hospital curves are in Figure 5A. "
             "(B) Calibration curves before recalibration, in every validation database.",
         short="External validation: one-year discrimination and initial calibration"),
    dict(no=5, layout="v", labels=["A", "B"],
         panels=[("v9", "fig7_ext_roc_bymodel_v9.png"), ("v9", "fig9_recal_calibration_v9.png")],
         # ⚠ 正文禁字面 em dash（64 的断言）⇒ 用冒号，不用 "— … —"
         # ★★ 2026-10-22（第十九轮模拟评审 §六）："the four in-hospital cohorts" 不点名四个队列，
         #   读者无法判断是否含开发队列。改为直接列出四个队列名（开发队列 MIMIC-IV 不在其内，
         #   它只提供院内结局的开发模型）。
         cap="Discrimination by model and database, and recalibration. (A) Receiver operating "
             "characteristic curves of both models in eICU, NWICU, INSPIRE and the Huaian "
             "cohort; no panel is repeated from Figure 4A. (B) Calibration curves after logistic "
             "recalibration, fitted in each external database.",
         short="In-hospital discrimination by database, and post-recalibration calibration"),
    dict(no=6, layout="v", labels=["A", "B"],
         panels=[("v9", "figR3_rootcause_v9.png"), ("v9", "leakage_dauc_v9.png")],
         cap="Audit findings 2 and 3: structural differences between databases, and how much the "
             "model depends on discharge-coded aetiology. Upper panel: first-day vasopressor "
             "recording across databases (left) and the aetiology mix in NWICU versus INSPIRE "
             "(right). Lower panel: change in AUC after removing all seven discharge-coded "
             "aetiology indicators, internally across three label-timing scenarios (left) and in "
             "the external databases (right). MIMIC-IV carries no present-on-admission flag and "
             "no diagnosis timestamp, so the timing of these codes cannot be verified directly.",
         short="Audit findings 2-3: structural database differences and code-timing dependence"),
    dict(no=7, layout="v", labels=["A", "B"],
         panels=[("v9", "slope_eps_v9.png"), ("v9", "calib_ci_v9.png")],
         # ★ P0（2026-10-07 第二轮评审 §2.5）：旧句写 "The dependence is small in the three large
         #   validation databases but pronounced in the 225-patient cohort, so what it reflects is
         #   the size of the cohort rather than a defect of the model." —— 把 ε 敏感**归因于样本量**，
         #   而本分析并未隔离任何决定因素（eICU 的 LR 跨度 0.0094 并不小于淮安的某些口径，且
         #   树模型在同一个 225 例队列上跨度只有 0.0001，说明决定因素是学习器/线性预测值分布，
         #   不是单纯的 n）。改为陈述现象而不认领因由。
         cap="Audit finding 4: how stable are the calibration estimates? (A) The calibration "
             "slope depends on the constant ε used to clip predictions before the logit "
             "transformation. The largest sensitivity was observed in the smallest cohort; its "
             "determinants were not isolated by this analysis. (B) Calibration slopes with 95% "
             # ★ 第十五轮评审 §五.2："too wide to support any calibration **claim**" 过绝对 ——
             #   估计值、区间与数值敏感性仍可报，不能作出的是"校准充分/精确"的结论。
             #   与正文 Audit Finding 4 及 External Calibration 一节统一为 "adequate calibration"。
             #   ⚠ 只有 92（V9）改；66（V8 链）逐字保留。
             "bootstrap confidence intervals (1,000 resamples); the intervals in the smallest "
             "cohort are too wide to establish adequate calibration.",
         short="Audit finding 4: stability of the calibration estimates"),
    dict(no=8, layout="v", labels=["A", "B"],
         panels=[("v9", "harmonisation_v9.png"), ("v9", "huaian_panel_v9.png")],
         cap="Audit finding 5: measurement harmonisation at the independent site, and what the "
             "frozen model achieved there. Upper row: change in AUC (left) and observed:expected "
             "ratio when laboratory units are not converted and when the feature-scaling "
             "convention is not respected (Huaian, n=225). Lower row: calibration and "
             "discrimination of the frozen in-hospital model after harmonisation.",
         short="Audit finding 5: harmonisation effects and independent-site performance"),
]
MARGIN, GAP, LABEL_H = 14, 16, 30


def compose():
    for stale in FIGDIR.glob("Figure*.png"):
        stale.unlink()
    inv = []
    for g in GROUPS:
        ims = [Image.open(PAN / n).convert("RGB") for _, n in g['panels']]
        labels = g['labels'] if len(g['labels']) == len(ims) else [""] * len(ims)
        if g['layout'] == "h":
            # 缩放后总宽恰为 MAX_W（= 6.3 in @300 dpi），行标与页边距已计入
            target_w = MAX_W - 2 * MARGIN - GAP * (len(ims) - 1)
            scale = min(target_w / sum(i.width for i in ims),
                        (MAX_H - 2 * MARGIN - LABEL_H) / max(i.height for i in ims))
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
            target_h = MAX_H - 2 * MARGIN - LABEL_H * len(ims) - GAP * (len(ims) - 1)
            scale = min((MAX_W - 2 * MARGIN) / max(i.width for i in ims),
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
        if W > MAX_W or H > MAX_H:
            s = min(MAX_W / W, MAX_H / H)
            canvas = canvas.resize((int(W * s), int(H * s)), Image.LANCZOS)
        fp = FIGDIR / ("Figure%d.png" % g['no'])
        canvas.save(fp, "PNG", optimize=True)
        mb = os.path.getsize(fp) / 1e6
        log("  Figure %d -> %dx%d, %.2f MB (%d panels)"
            % (g['no'], canvas.width, canvas.height, mb, len(ims)))
        inv.append(dict(figure=g['no'], file=fp.name, px=[canvas.width, canvas.height],
                        size_mb=round(mb, 3), layout=g['layout'],
                        panels=[n for _, n in g['panels']], panel_labels=labels,
                        caption=g['cap'], short_caption=g['short']))
    json.dump(inv, open(OUT / "v9_figure_inventory.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main():
    log("#" * 90)
    log("# 92_v9_audit_figures.py | V9 审计面板 + 组合 8 张正文图")
    log("#" * 90)
    res, y, preds = part1_harmonisation()
    aud = json.load(open(OUT / "downstream_severity_gcs_v9.json", encoding="utf-8"))
    lk = json.load(open(OUT / "leakage_sensitivity_v9.json", encoding="utf-8"))
    se = json.load(open(OUT / "slope_epsilon_audit_v9.json", encoding="utf-8"))
    cc = json.load(open(OUT / "calibration_ci_and_severity_v2.json", encoding="utf-8"))

    log("")
    log("=== PART 2 面板 ===")
    gcs_strat = gcs_vent_stratified()
    aud['gcs_increment']['gcs_alone_by_ventilation'] = gcs_strat
    json.dump(aud, open(OUT / "downstream_severity_gcs_v9.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    panel_gcs(aud, gcs_strat)
    panel_leakage(lk)
    panel_slope_eps(se)
    panel_calib_ci(cc)
    panel_harmonisation(res)
    panel_huaian(res, y, preds)

    log("")
    log("=== PART 3 组合 ===")
    compose()
    (OUT / "_92_v9_audit_figures.md").write_text(
        "# 92 —— V9 审计面板与正文图\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    log("\n→ output/v9_figures/Figure1..8.png, v9_figure_inventory.json")


if __name__ == "__main__":
    main()
