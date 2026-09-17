# -*- coding: utf-8 -*-
"""Part 1 | 附加图表生成
   正文图: 图1 流程图 / 图3 各队列AUC森林图 / 图4 决策曲线(DCA) / 图5 预测风险分布
   补充图: S1 缺失模式 / S2 敏感性森林图 / S3 变量重要性 / S4 重校准前后
   产出: ABI3/figs/*.png + ABI3/output/part1_figs_stats.json（供 HTML 组装）
"""
import os, json
# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = ABI_BASE
OUT = os.path.join(BASE, 'ABI3', 'output')
FIG = os.path.join(BASE, 'ABI3', 'figs')
os.makedirs(FIG, exist_ok=True)
P = os.path.join(OUT, 'ABI本地数据采集模板_已录入检验结果.xlsx')

try:
    font_manager.fontManager.addfont(r"C:/Windows/Fonts/simhei.ttf")
    plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception as e:
    print('font warn:', e)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 170

BLUE, RED, GREY, GREEN, ORANGE = '#2a7ab0', '#c0392b', '#9aa7b4', '#2e8b57', '#e08a1e'

# 变量中文标签（供缺失图与变量重要性图共用；避免使用 SimHei 缺失的下标字形）
FEAT_LAB = {'age': '年龄', 'female': '女性', 'charlson_comorbidity_index': 'Charlson合并症指数',
            'tbi': '创伤性脑损伤', 'sah': '蛛网膜下腔出血', 'ich': '脑出血',
            'ais': '急性缺血性卒中', 'cns_inf': '中枢神经系统感染', 'seizure': '癫痫持续状态',
            'anoxic': '缺氧性脑损伤', 'heart_rate_mean': '心率', 'mbp_mean': '平均动脉压',
            'resp_rate_mean': '呼吸频率', 'temperature_mean': '体温', 'spo2_mean': '血氧饱和度',
            'wbc_max': '白细胞', 'hemoglobin_min': '血红蛋白', 'platelets_min': '血小板',
            'sodium_min': '血钠', 'potassium_max': '血钾', 'creatinine_max': '肌酐',
            'bun_max': '尿素氮', 'glucose_max': '血糖', 'inr_max': 'INR',
            'mech_vent': '机械通气', 'vasopressor': '血管活性药', 'rrt': '肾脏替代治疗',
            'bicarbonate_min': '碳酸氢盐'}
FL = lambda c: FEAT_LAB.get(c, c)


def sens_short(tag):
    """敏感性分析标签在图中的短名：去掉括号内的补充说明"""
    return tag.split('（')[0].strip()

# ---------------- 载入模型与数据 ----------------
m = joblib.load(os.path.join(BASE, 'output', 'model_inhosp.joblib'))
feats, cont, medians, scaler = m['feats'], m['cont'], m['medians'], m['scaler']
binv = [c for c in feats if c not in cont]
print('模型 keys:', list(m.keys()), '| 特征数:', len(feats))

df = pd.read_excel(P, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
df = df.dropna(how='all').reset_index(drop=True)
num = lambda c: pd.to_numeric(df[c], errors='coerce')

d = pd.DataFrame(index=df.index)
for c in feats:
    d[c] = num(c)
d['age'] = num('age')
d['gcs'] = num('gcs')
d['dl'] = num('hospital_discharge_location')
d['y'] = (d['dl'] == 5).astype(int)
d['y_known'] = d['dl'].notna()

PRIO = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
nflag = d[PRIO].fillna(0).sum(axis=1)
keep = (nflag >= 1) & (d['age'] >= 18) & d['age'].notna() & d['y_known']
dfx = d[keep].copy()

CONV = {'creatinine_max': lambda s: s / 88.4, 'bun_max': lambda s: s / 0.357,
        'glucose_max': lambda s: s * 18.0, 'hemoglobin_min': lambda s: s / 10.0}
for c, f in CONV.items():
    dfx[c] = f(dfx[c])

X = dfx[feats].copy()
for c in cont:
    X[c] = X[c].fillna(medians[c])
for c in binv:
    X[c] = X[c].fillna(0)
Xs = X.copy(); Xs[cont] = scaler.transform(X[cont])
y = dfx['y'].values
p_lr = m['lr'].predict_proba(Xs)[:, 1]
p_xg = m['xg'].predict_proba(X)[:, 1]
n = len(dfx); ev = int(y.sum())
print('纳入 n =', n, '| 事件 =', ev, '| 死亡率 = %.1f%%' % (100 * y.mean()))

STATS = {}

# =====================================================================
# 图 1　研究流程图（STROBE）
# =====================================================================
def fig1_flow():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    def box(x, y, w, h, txt, fc, ec='#34506b', tc='#12283c', fs=10.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12,rounding_size=0.14',
                                    linewidth=1.3, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=tc, linespacing=1.5)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                                     linewidth=1.25, color='#5b6b7b'))
    box(2.4, 8.4, 5.2, 1.05, 'ICU 收治的急性脑损伤（ABI）患者\n初始筛选 n = 239', '#eaf1f7')
    arrow(5.0, 8.4, 5.0, 7.65)
    box(2.4, 6.35, 5.2, 1.3,
        '排除 n = 14\n· 七类病因标志全阴 n = 4\n· 年龄 < 18 岁 n = 10\n· 院内结局缺失 n = 0', '#fdf1e6')
    arrow(5.0, 6.35, 5.0, 5.6)
    box(2.4, 4.3, 5.2, 1.3, '纳入分析队列 n = 225\n院内死亡 55 例（24.4%）', '#e6f2ea')
    arrow(5.0, 4.3, 5.0, 3.55)
    box(0.55, 2.15, 3.9, 1.4, '存活出院\nn = 170（75.6%）', '#eaf1f7')
    box(5.55, 2.15, 3.9, 1.4, '院内死亡\nn = 55（24.4%）', '#fbeaea')
    ax.plot([5.0, 5.0], [3.55, 2.85], color='#5b6b7b', lw=1.25)
    ax.plot([2.5, 7.5], [2.85, 2.85], color='#5b6b7b', lw=1.25)
    ax.add_patch(FancyArrowPatch((2.5, 2.85), (2.5, 3.55), arrowstyle='-|>', mutation_scale=13, lw=1.25, color='#5b6b7b'))
    ax.add_patch(FancyArrowPatch((7.5, 2.85), (7.5, 3.55), arrowstyle='-|>', mutation_scale=13, lw=1.25, color='#5b6b7b'))
    ax.text(5.0, 0.85, '结局定义：出院去向 = 死亡（院内死亡）；验证采用冻结模型，不重新拟合',
            ha='center', fontsize=9.2, color='#5b6b7b')
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_fig1_flow.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    print('WROTE', f)
    return f

# =====================================================================
# 图 3　各队列判别力（AUC）森林图
# =====================================================================
def fig3_auc_forest():
    tr = {'mimic': (16597, 0.845, 0.8754), 'eICU': (14852, 0.811, 0.850),
          'INSPIRE': (1543, 0.644, 0.720), 'NWICU': (3420, 0.524, 0.683)}
    labels = ['MIMIC-IV\n（开发库·内部）', 'eICU-CRD\n（既往外部）', 'INSPIRE\n（既往外部）',
              'NWICU*\n（既往外部）', '本研究\n（本地队列）']
    lr = [0.845, 0.811, 0.644, 0.524, 0.865]
    xg = [0.8754, 0.850, 0.720, 0.683, 0.800]
    ci_lo = [None, None, None, None, 0.798]; ci_hi = [None, None, None, None, 0.922]
    ci_lo_x = [None, None, None, None, 0.733]; ci_hi_x = [None, None, None, None, 0.860]
    yv = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for i, yv_ in enumerate(yv):
        if ci_lo[i] is not None:
            ax.plot([ci_lo[i], ci_hi[i]], [yv_ + 0.16, yv_ + 0.16], color=BLUE, lw=1.8, alpha=.85)
            ax.plot([ci_lo_x[i], ci_hi_x[i]], [yv_ - 0.16, yv_ - 0.16], color=RED, lw=1.8, alpha=.85)
    ax.scatter(lr, yv + 0.16, s=64, color=BLUE, zorder=5, label='Logistic')
    ax.scatter(xg, yv - 0.16, s=64, color=RED, marker='s', zorder=5, label='XGBoost')
    for i, yv_ in enumerate(yv):
        bx = dict(fc='white', ec='none', alpha=.88, pad=0.6)
        ax.text(lr[i] + 0.012, yv_ + 0.16, '%.3f' % lr[i], va='center', fontsize=8.6, color=BLUE, bbox=bx)
        ax.text(xg[i] + 0.012, yv_ - 0.16, '%.3f' % xg[i], va='center', fontsize=8.6, color=RED, bbox=bx)
    ax.axvline(0.5, ls='--', color=GREY, lw=1)
    ax.text(0.505, len(labels) - 0.62, '随机猜测 AUC=0.5', fontsize=8.2, color='#5b6b7b')
    ax.set_yticks(yv); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.45, 0.99); ax.set_xlabel('AUC（院内死亡）', fontsize=10)
    ax.set_title('各队列院内死亡判别力对照', fontsize=11)
    ax.grid(axis='x', alpha=.25); ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_fig3_auc_forest.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    print('WROTE', f); return f

# =====================================================================
# 图 4　决策曲线分析（DCA）
# =====================================================================
def dca_curve(y, p, ths):
    nb = []
    for pt in ths:
        pred = p >= pt
        tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
        nb.append(tp / len(y) - fp / len(y) * (pt / (1 - pt)))
    return np.array(nb)

def fig4_dca():
    ths = np.arange(0.05, 0.71, 0.01)
    prev = y.mean()
    nb_lr = dca_curve(y, p_lr, ths); nb_xg = dca_curve(y, p_xg, ths)
    nb_all = prev - (1 - prev) * (ths / (1 - ths)); nb_none = np.zeros_like(ths)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(ths, nb_all, color=GREY, lw=1.5, ls='--', label='全部治疗（treat all）')
    ax.plot(ths, nb_none, color='#444', lw=1.5, ls=':', label='不治疗（treat none）')
    ax.plot(ths, nb_lr, color=BLUE, lw=2.2, label='Logistic 模型')
    ax.plot(ths, nb_xg, color=RED, lw=2.0, label='XGBoost 模型')
    ax.axhline(0, color='#000', lw=.6, alpha=.3)
    ax.set_xlim(0.05, 0.70); ax.set_ylim(-0.06, max(0.26, prev * 1.08))
    ax.set_xlabel('风险阈值（阈值概率）', fontsize=10)
    ax.set_ylabel('净获益（net benefit）', fontsize=10)
    ax.set_title('决策曲线分析（院内死亡）', fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_fig4_dca.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    # 记录关键阈值净获益
    kt = [0.10, 0.20, 0.30, 0.40, 0.50]
    dca_tab = []
    for t in kt:
        i = int(np.argmin(np.abs(ths - t)))
        dca_tab.append({'pt': t, 'model_lr': round(float(nb_lr[i]), 4), 'model_xg': round(float(nb_xg[i]), 4),
                        'treat_all': round(float(nb_all[i]), 4), 'treat_none': 0.0})
    STATS['dca'] = dca_tab
    # 临床有用区间：模型净获益 > treat all 且 > 0
    useful = [float(ths[i]) for i in range(len(ths)) if nb_lr[i] > max(nb_all[i], 0)]
    STATS['dca_useful_range_lr'] = [round(min(useful), 2), round(max(useful), 2)] if useful else None
    useful_x = [float(ths[i]) for i in range(len(ths)) if nb_xg[i] > max(nb_all[i], 0)]
    STATS['dca_useful_range_xg'] = [round(min(useful_x), 2), round(max(useful_x), 2)] if useful_x else None
    print('WROTE', f, '| LR 临床有用区间:', STATS['dca_useful_range_lr'],
          '| XGB:', STATS['dca_useful_range_xg'])
    return f

# =====================================================================
# 图 5　预测风险分布（按结局分层）
# =====================================================================
def fig5_riskdist():
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(p_lr[y == 0], bins=bins, color=BLUE, alpha=.55, label='存活出院（n=%d）' % int((y == 0).sum()))
    ax.hist(p_lr[y == 1], bins=bins, color=RED, alpha=.55, label='院内死亡（n=%d）' % int((y == 1).sum()))
    ax.axvline(np.median(p_lr[y == 0]), color=BLUE, ls='--', lw=1.4)
    ax.axvline(np.median(p_lr[y == 1]), color=RED, ls='--', lw=1.4)
    ax.set_xlabel('Logistic 模型预测的院内死亡概率', fontsize=10)
    ax.set_ylabel('例数', fontsize=10)
    ax.set_title('预测风险在存活与死亡组间的分布（虚线为中位数）', fontsize=11)
    ax.grid(alpha=.22); ax.legend(fontsize=9)
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_fig5_riskdist.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    STATS['riskdist'] = {'median_pred_surv': round(float(np.median(p_lr[y == 0])), 3),
                         'median_pred_died': round(float(np.median(p_lr[y == 1])), 3)}
    print('WROTE', f, '|', STATS['riskdist']); return f

# =====================================================================
# 补充图 S1　缺失模式
# =====================================================================
def figS1_missing():
    miss = {}
    for c in feats:
        miss[c] = 100.0 * dfx[c].isna().mean()
    ms = pd.Series(miss).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    cols = [RED if v > 40 else (ORANGE if v > 10 else BLUE) for v in ms.values]
    ax.barh(range(len(ms)), ms.values, color=cols, height=.72)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels([FL(i) for i in ms.index], fontsize=8.2)
    ax.axvline(40, ls='--', color=RED, lw=1.3)
    ax.text(41, 0.5, '40% 剔除阈值', color=RED, fontsize=8.6)
    ax.set_xlabel('缺失率（%）', fontsize=10); ax.set_xlim(0, 100)
    ax.set_title('模型变量在本地队列中的缺失情况（n=%d）' % n, fontsize=11)
    ax.grid(axis='x', alpha=.22)
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_figs1_missing.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    STATS['missing'] = {k: round(v, 1) for k, v in sorted(miss.items(), key=lambda x: -x[1])}
    STATS['missing_labeled'] = [{'var': k, 'label': FL(k), 'pct': round(v, 1)}
                                for k, v in sorted(miss.items(), key=lambda x: -x[1])]
    print('WROTE', f); return f

# =====================================================================
# 补充图 S2　敏感性分析森林图
# =====================================================================
def figS2_sens_forest():
    J = json.load(open(os.path.join(OUT, 'local_part1_stats.json'), encoding='utf-8'))
    items = [('主分析（n=225）', J['main'])] + [(sens_short(s['tag']), s) for s in J['sensitivity']]
    rows = []
    for tag, s in items:
        if 'Logistic' in s:
            rows.append((tag, s['Logistic']['AUC'], s['Logistic']['CI'],
                         s['XGBoost']['AUC'], s['XGBoost']['CI']))
        else:
            rows.append((tag, s['AUC'], None, None, None))
    yv = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, (tag, la, lc, xa, xc) in enumerate(rows):
        if lc:
            ax.plot(lc, [yv[i] + 0.15] * 2, color=BLUE, lw=1.8, alpha=.85)
        ax.scatter([la], [yv[i] + 0.15], color=BLUE, s=58, zorder=5)
        if xc:
            ax.plot(xc, [yv[i] - 0.15] * 2, color=RED, lw=1.8, alpha=.85)
            ax.scatter([xa], [yv[i] - 0.15], color=RED, s=58, marker='s', zorder=5)
    ax.axvline(0.845, ls='--', color=GREY, lw=1.2)
    ax.text(0.848, -0.44, 'MIMIC 内部 AUC 0.845', fontsize=8.2, color='#5b6b7b')
    ax.axvline(0.5, ls=':', color='#888', lw=1)
    ax.set_yticks(yv)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
    ax.set_xlim(0.45, 1.0); ax.set_ylim(-0.75, len(rows) - 0.4)
    ax.set_xlabel('AUC（95%CI）', fontsize=10)
    ax.set_title('敏感性分析：Logistic（圆）与 XGBoost（方）判别力', fontsize=11)
    ax.grid(axis='x', alpha=.25)
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_figs2_sens_forest.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    print('WROTE', f); return f

# =====================================================================
# 补充图 S3　变量重要性（LR 标准化系数 + XGB 增益）
# =====================================================================
def figS3_varimp():
    lab = FL
    # LR 系数（标准化空间，训练于 scaled）
    lr_coef = pd.Series(m['lr'].coef_[0], index=feats)
    imp_xg = pd.Series(getattr(m['xg'], 'feature_importances_', np.zeros(len(feats))), index=feats)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    for ax, s, ttl, col, note in [
            (axes[0], lr_coef, 'Logistic 标准化系数（每 1 SD 的对数优势比）', None, '正向=风险升高'),
            (axes[1], imp_xg, 'XGBoost 变量重要性（增益）', '#8e6bbf', '')]:
        s = s.reindex(s.abs().sort_values(ascending=False).index)[:15][::-1]
        ax.barh(range(len(s)), s.values,
                color=[('#c0392b' if v > 0 else '#2a7ab0') for v in s.values] if col is None else col,
                height=.72)
        ax.set_yticks(range(len(s))); ax.set_yticklabels([lab(i) for i in s.index], fontsize=8.4)
        ax.axvline(0, color='#333', lw=.8)
        ax.set_title(ttl, fontsize=10.2); ax.grid(axis='x', alpha=.22)
        ax.set_xlabel(note or '重要性', fontsize=9)
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_figs3_varimp.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    STATS['lr_coef_top'] = [{'var': i, 'label': lab(i), 'coef': round(float(v), 3)}
                            for i, v in lr_coef.abs().sort_values(ascending=False)[:10].items()]
    STATS['xgb_imp_top'] = [{'var': i, 'label': lab(i), 'imp': round(float(imp_xg[i]), 4)}
                            for i in imp_xg.sort_values(ascending=False)[:10].index]
    print('WROTE', f); return f

# =====================================================================
# 补充图 S4　XGBoost 重校准前后
# =====================================================================
def figS4_recal():
    lp = np.log(np.clip(p_xg, 1e-9, 1 - 1e-9) / (1 - np.clip(p_xg, 1e-9, 1 - 1e-9)))
    rc = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(lp.reshape(-1, 1), y)
    p_rc = rc.predict_proba(lp.reshape(-1, 1))[:, 1]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], '--', color=GREY, lw=1.2, label='理想校准')
    for p, lab_, col, mk in [(p_xg, '重校准前', RED, 'o'), (p_rc, '重校准后', GREEN, 's')]:
        q = pd.qcut(pd.Series(p), 5, duplicates='drop')
        g = pd.DataFrame({'p': p, 'y': y}).groupby(q, observed=True).mean()
        ax.plot(g['p'], g['y'], mk + '-', color=col, lw=1.9, ms=6.5, label='%s（O:E %.2f）' % (lab_, y.mean() / p.mean()))
    ax.set_xlabel('平均预测概率（五分位）', fontsize=10); ax.set_ylabel('观察死亡率', fontsize=10)
    ax.set_title('XGBoost 本地重校准前后（院内死亡）', fontsize=11)
    ax.set_xlim(0, .9); ax.set_ylim(0, .9); ax.grid(alpha=.25); ax.legend(fontsize=9)
    plt.tight_layout()
    f = os.path.join(FIG, 'part1_figs4_recal.png'); plt.savefig(f, bbox_inches='tight'); plt.close()
    STATS['recal'] = {'oe_before': round(float(y.mean() / p_xg.mean()), 2),
                      'oe_after': round(float(y.mean() / p_rc.mean()), 2),
                      'slope': round(float(rc.coef_[0][0]), 2),
                      'intercept': round(float(rc.intercept_[0]), 2)}
    print('WROTE', f, '|', STATS['recal']); return f

# ---------------- 运行 ----------------
FIG1 = fig1_flow(); FIG3 = fig3_auc_forest(); FIG4 = fig4_dca()
FIG5 = fig5_riskdist(); FS1 = figS1_missing(); FS2 = figS2_sens_forest()
FS3 = figS3_varimp(); FS4 = figS4_recal()

STATS['figs'] = {'fig1': os.path.basename(FIG1), 'fig3': os.path.basename(FIG3),
                 'fig4': os.path.basename(FIG4), 'fig5': os.path.basename(FIG5),
                 'S1': os.path.basename(FS1), 'S2': os.path.basename(FS2),
                 'S3': os.path.basename(FS3), 'S4': os.path.basename(FS4)}
STATS['n'] = n; STATS['events'] = ev; STATS['prev'] = round(100 * y.mean(), 1)

# ---------- 补充表 S1：模型输入变量清单 ----------
CAT = {'tbi': '病因（7 类）', 'sah': '病因（7 类）', 'ich': '病因（7 类）', 'ais': '病因（7 类）',
       'cns_inf': '病因（7 类）', 'seizure': '病因（7 类）', 'anoxic': '病因（7 类）',
       'female': '人口学', 'age': '人口学', 'charlson_comorbidity_index': '合并症',
       'heart_rate_mean': '首日生命体征', 'mbp_mean': '首日生命体征', 'resp_rate_mean': '首日生命体征',
       'temperature_mean': '首日生命体征', 'spo2_mean': '首日生命体征',
       'wbc_max': '首日实验室', 'hemoglobin_min': '首日实验室', 'platelets_min': '首日实验室',
       'sodium_min': '首日实验室', 'potassium_max': '首日实验室', 'creatinine_max': '首日实验室',
       'bun_max': '首日实验室', 'glucose_max': '首日实验室', 'inr_max': '首日实验室',
       'bicarbonate_min': '首日实验室',
       'mech_vent': '治疗强度', 'vasopressor': '治疗强度', 'rrt': '治疗强度'}
unit = {'creatinine_max': 'mg/dL', 'bun_max': 'mg/dL', 'glucose_max': 'mg/dL',
        'hemoglobin_min': 'g/dL', 'wbc_max': '10⁹/L', 'platelets_min': '10⁹/L',
        'sodium_min': 'mmol/L', 'potassium_max': 'mmol/L', 'bicarbonate_min': 'mmol/L',
        'inr_max': '—', 'heart_rate_mean': '次/分', 'mbp_mean': 'mmHg', 'resp_rate_mean': '次/分',
        'temperature_mean': '℃', 'spo2_mean': '%', 'age': '岁', 'charlson_comorbidity_index': '分'}
STATS['var_table'] = []
for c in feats:
    iscont = c in cont
    tr_med = medians.get(c)
    lc = pd.to_numeric(dfx[c], errors='coerce')
    STATS['var_table'].append({
        'var': c, 'label': FL(c), 'cat': CAT.get(c, '—'),
        'type': '连续' if iscont else '二分类',
        'train_median': (round(float(tr_med), 2) if (iscont and tr_med is not None) else '—'),
        'local_median': (round(float(lc.median()), 2) if (iscont and lc.notna().any()) else '—'),
        'unit': unit.get(c, '—'),
        'missing': round(100 * dfx[c].isna().mean(), 1)})

json.dump(STATS, open(os.path.join(OUT, 'part1_figs_stats.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('\nWROTE', os.path.join(OUT, 'part1_figs_stats.json'))
print('全部图件完成。')
