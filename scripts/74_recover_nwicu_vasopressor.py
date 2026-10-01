# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""74_recover_nwicu_vasopressor.py — 恢复 NWICU 被写死的血管活性药变量，并把这次
"协调层错误"量化成可报告的审计证据。

背景（2026-10-01 取证）
  sql/11_nwicu_cohort.sql 第 76 行写了 `0 as vasopressor,  -- NWICU has no vasopressor/
  infusion source`。这句话一半对、一半错：
    · 对的一半：NWICU 的 icu schema 只有 {abi_a1_ext, chartevents, d_items, icustays,
      procedureevents}，**确实没有 inputevents**，而 MIMIC 的
      mimiciv_derived.vasoactive_agent 正是基于 ICU 输注记录（带剂量速率）。
    · 错的一半：NWICU 的 hosp.prescriptions 有 1,852,983 行，五种升压药命中 84,398 行，
      **它是有血管活性药来源的**，只是没有输注（剂量）那一层。
  因此 NWICU 的 vasopressor = 0 不是数据性质，而是我们自己在抽取层构造的常数。

本脚本做什么
  STAGE 1  按四个口径从 hosp.prescriptions 抽取，选定口径写入 data/nwicu_abi_final_vp2.csv
           （**不覆盖** nwicu_abi_final.csv）。
  STAGE 2  用冻结模型重算 NWICU 外部验证：写死 0（旧） vs 恢复值（新），
           并按 -vasopressor-only 与 -全部三个治疗强度指标两种剔除方式重跑。
  STAGE 3  断言：行数/主键对齐、阳性率落在 plausible 区间、与既有 published 数字可复现、
           抽取可重复（同一查询跑两遍结果一致）。

铁律
  ★ 按 hadm_id merge 后再取值，绝不按位置赋值（首版就在这里翻车：得出错得整整
    0.007 AUC 的结果，因为有 ORDER BY 而 CSV 行序 ≠ hadm_id 序）。
  ★ 不覆盖已交付文件：只写 data/nwicu_abi_final_vp2.csv 与本脚本的 output JSON。
  ★ eps 全稿统一 1e-6；校准斜率 = 结局对线性预测值做逻辑回归（C=1e12）。

产出
  data/nwicu_abi_final_vp2.csv
  output/nwicu_vasopressor_recovery.json
  output/_74_nwicu_vaso.log（由调用方重定向）
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(ABI_BASE)
DATA, OUT = BASE / "data", BASE / "output"
SRC = DATA / "nwicu_abi_final.csv"
DST = DATA / "nwicu_abi_final_vp2.csv"
JSN = OUT / "nwicu_vasopressor_recovery.json"
DB = dict(host=_os.environ.get("ABI_DB_HOST", "localhost"),
          port=int(_os.environ.get("ABI_DB_PORT", "5432")),
          user=_os.environ.get("ABI_DB_USER", "postgres"),
          password=_os.environ.get("ABI_DB_PASSWORD", ""), dbname="nwicu")

EPS = 1e-6
SEED = 42
np.random.seed(SEED)

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
CONT = ['age', 'charlson_comorbidity_index', 'heart_rate_mean', 'mbp_mean', 'resp_rate_mean',
        'temperature_mean', 'spo2_mean', 'wbc_max', 'hemoglobin_min', 'platelets_min',
        'sodium_min', 'potassium_max', 'creatinine_max', 'bun_max', 'glucose_max',
        'bicarbonate_min', 'inr_max']
FEATS = (['age', 'female'] + AET + ['charlson_comorbidity_index']
         + [c for c in CONT if c not in ('age', 'charlson_comorbidity_index')] + BINV)
assert len(FEATS) == 28

LOGL = []


def log(m):
    LOGL.append(str(m))
    print(m)


def cal_slope(y, p):
    """校准斜率 = 结局对线性预测值做逻辑回归（V8 全稿统一，C=1e12，eps=1e-6）。"""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(fit_intercept=True, C=1e12, solver='lbfgs',
                           max_iter=2000).fit(lp, y)
    return dict(slope=round(float(m.coef_[0][0]), 4),
                intercept=round(float(m.intercept_[0]), 4),
                oe=round(float(np.mean(y) / max(np.mean(p), 1e-12)), 4),
                brier=round(float(np.mean((y - p) ** 2)), 4))


def metrics(y, p):
    d = dict(auc=round(float(roc_auc_score(y, p)), 4))
    d.update(cal_slope(y, p))
    return d


# ============================================================ STAGE 1  抽取
DRUG_SQL = (
    "lower(p.drug) LIKE '%%norepinephrine%%' OR lower(p.drug) LIKE '%%epinephrine%%' "
    "OR lower(p.drug) LIKE '%%dopamine%%' OR lower(p.drug) LIKE '%%phenylephrine%%' "
    "OR lower(p.drug) LIKE '%%vasopressin%%'"
)
WINDOW = "p.starttime BETWEEN b.intime - interval '6 hours' AND b.intime + interval '1 day'"

VARIANTS = {
    "A_any_route": "TRUE",
    "B_iv_or_injection": "lower(p.route) IN ('intravenous','injection')",
    "C_iv_only": "lower(p.route) = 'intravenous'",
    "D_iv_or_inj_no_local": (
        "lower(p.route) IN ('intravenous','injection') "
        "AND lower(p.drug) NOT LIKE '%%lidocaine%%' "
        "AND lower(p.drug) NOT LIKE '%%cocoa butter%%'"),
}
CHOSEN = "D_iv_or_inj_no_local"


def _query(cu, extra, ordered=True):
    sql = (
        "WITH v AS (SELECT DISTINCT b.hadm_id FROM icu.abi_a1_ext b "
        "JOIN hosp.prescriptions p ON p.hadm_id = b.hadm_id "
        "WHERE ( " + DRUG_SQL + " ) AND ( " + extra + " ) AND " + WINDOW + ") "
        "SELECT b.hadm_id, CASE WHEN v.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS vp "
        "FROM icu.abi_a1_ext b LEFT JOIN v ON v.hadm_id = b.hadm_id"
    )
    if ordered:
        sql += " ORDER BY b.hadm_id"
    cu.execute(sql)
    return pd.DataFrame(cu.fetchall(), columns=['hadm_id', 'vp'])


def align(df, vp):
    """★ 按 hadm_id merge 后再取值 —— 禁止按位置赋值。"""
    m = pd.DataFrame({'hadm_id': df['hadm_id'].values}).merge(vp, on='hadm_id', how='left')
    assert len(m) == len(df), "merge 后行数不符：%d vs %d" % (len(m), len(df))
    assert m['vp'].notna().all(), "有 hadm_id 未能匹配到抽取结果"
    return m['vp'].astype(int).values


def stage1():
    log("=" * 88)
    log("STAGE 1  恢复抽取（→ data/nwicu_abi_final_vp2.csv）")
    src = pd.read_csv(SRC)
    log("  源文件 %s  n=%d  列=%s" % (SRC.name, len(src), list(src.columns)[:6]))

    cn = psycopg2.connect(**DB)
    cn.set_session(autocommit=True)
    cu = cn.cursor()
    cu.execute("SET statement_timeout = '240s'")

    # --- 源端规模（写进论文要用的证据）
    cu.execute("SELECT count(*) FROM hosp.prescriptions")
    rx_rows = int(cu.fetchone()[0])
    cu.execute("SELECT count(*) FROM hosp.prescriptions AS p WHERE ( " + DRUG_SQL + " )")
    rx_vaso = int(cu.fetchone()[0])
    assert rx_vaso > 0, "源端血管活性药记录为 0，取证前提不成立"
    cu.execute("SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
               "WHERE n.nspname='icu' AND c.relkind IN ('r','v','m') ORDER BY 1")
    icu_tbl = [r[0] for r in cu.fetchall()]
    log("  hosp.prescriptions = %s 行；五种升压药原始命中 = %s 行"
        % (f"{rx_rows:,}", f"{rx_vaso:,}"))
    log("  NWICU icu schema 对象 = %s" % icu_tbl)
    assert rx_rows > 0 and rx_vaso > 0, "源端血管活性药记录为 0，取证前提不成立"
    assert 'inputevents' not in icu_tbl, "NWICU icu schema 竟有 inputevents，取证结论需重审"

    # --- 四口径
    varn = {}
    for k, extra in VARIANTS.items():
        v = align(src, _query(cu, extra))
        varn[k] = v
        log("    %-24s 阳性 %4d / %d = %5.2f%%"
            % (k, int(v.sum()), len(v), 100.0 * v.mean()))

    # --- 可重复性：换一种 SQL 排序再抽一次
    again = align(src, _query(cu, VARIANTS[CHOSEN], ordered=False))
    assert bool((again == varn[CHOSEN]).all()), "%s 两次抽取结果不一致" % CHOSEN
    log("  [OK] %s 重复抽取一致（换排序后逐行相同）" % CHOSEN)

    cn.close()

    # --- 写文件
    dst = src.copy()
    dst['vasopressor'] = varn[CHOSEN]
    dst.to_csv(DST, index=False)
    log("  [OK] 已写 %s  n=%d  阳性=%d (%.2f%%)"
        % (DST.name, len(dst), int(dst['vasopressor'].sum()),
           100.0 * dst['vasopressor'].mean()))

    return src, dst, varn, dict(rx_rows=rx_rows, rx_vaso_rows=rx_vaso, nw_icu_objects=icu_tbl)


# ============================================================ STAGE 2  重算
def build_frozen(ytr_target, use_feats, mim):
    """与 11_external_revision.build_frozen_model 完全同口径。"""
    X = mim[use_feats].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, ytr_target, test_size=0.30,
                                          stratify=ytr_target, random_state=SEED)
    c_in = [c for c in CONT if c in use_feats]
    med = Xtr[c_in].median()
    for c in c_in:
        Xtr[c] = Xtr[c].fillna(med[c])
        Xte[c] = Xte[c].fillna(med[c])
    sc = StandardScaler().fit(Xtr[c_in])
    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
    Xtr_s[c_in] = sc.transform(Xtr[c_in])
    Xte_s[c_in] = sc.transform(Xte[c_in])
    lr = LogisticRegression(max_iter=2000, C=1e6).fit(Xtr_s[use_feats], ytr)
    xg = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, eval_metric='loglog' if False else 'logloss',
                           random_state=SEED, tree_method='hist', n_jobs=-1).fit(
        Xtr[use_feats], ytr)
    internal = dict(Logistic=dict(auc=round(float(
        roc_auc_score(yte, lr.predict_proba(Xte_s[use_feats])[:, 1])), 4)),
        XGBoost=dict(auc=round(float(
            roc_auc_score(yte, xg.predict_proba(Xte[use_feats])[:, 1])), 4)))
    return dict(lr=lr, xg=xg, scaler=sc, cont=c_in, medians=med.to_dict(),
                feats=use_feats, internal=internal)


def apply_md(md, X):
    uf, c_in = md['feats'], md['cont']
    Z = X[uf].copy()
    for c in c_in:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(md['medians'][c])
    for c in [f for f in uf if f not in c_in]:
        Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
    Zs = Z.copy()
    Zs[c_in] = md['scaler'].transform(Z[c_in])
    return (md['lr'].predict_proba(Zs[uf])[:, 1], md['xg'].predict_proba(Z[uf])[:, 1])


def stage2(src, dst):
    log("")
    log("=" * 88)
    log("STAGE 2  重算 NWICU 外部验证（写死 0 vs 恢复值）")
    mim = pd.read_csv(DATA / "mimic_abi_cohort.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)

    pp = joblib.load(OUT / "preproc.joblib")
    md = joblib.load(OUT / "models.joblib")
    feats, cont = pp['feats'], pp['cont']
    scaler, med1 = pp['scaler'], pd.Series(pp['medians'])
    logitp, xgbm = pd.Series(md['logit_params']), md['xgb']
    assert list(feats) == FEATS, "冻结 1 年模型特征顺序与 FEATS 不一致"

    def pred_full(X):
        Z = X[feats].copy()
        for c in cont:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(med1[c])
        for c in AET + BINV:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
        Zs = Z.copy()
        Zs[cont] = scaler.transform(Z[cont])
        return (1 / (1 + np.exp(-(logitp['const'] + Zs[feats].values @ logitp[feats].values))),
                xgbm.predict_proba(Z[feats].values)[:, 1])

    # ---------- 2.1 全模型（28 项）
    res = {}
    for tag, X in (("hardcoded_zero", src), ("recovered", dst)):
        y = X['death_365'].values.astype(int)
        a, b = pred_full(X)
        r = dict(n=int(len(y)), events=int(y.sum()),
                 positives=int(pd.to_numeric(X['vasopressor']).fillna(0).sum()),
                 prevalence_pct=round(100.0 * pd.to_numeric(X['vasopressor']).fillna(0).mean(), 2),
                 Logistic=metrics(y, a), XGBoost=metrics(y, b))
        # 院内平行结局
        mh = joblib.load(OUT / "model_inhosp.joblib")
        uf, uc = mh['feats'], mh['cont']
        Z = X[uf].copy()
        for c in uc:
            Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(mh['medians'][c])
        for c in uf:
            if c not in uc:
                Z[c] = pd.to_numeric(Z[c], errors='coerce').fillna(0)
        Zs = Z.copy()
        Zs[uc] = mh['scaler'].transform(Z[uc])
        y2 = X['death_hosp'].values.astype(int)
        r2 = dict(events=int(y2.sum()),
                  Logistic=metrics(y2, mh['lr'].predict_proba(Zs[uf])[:, 1]),
                  XGBoost=metrics(y2, mh['xg'].predict_proba(Z[uf])[:, 1]))
        r['inhosp'] = r2
        res[tag] = r
        log("  [全模型 1 年] %-16s pos=%4d (%5.2f%%) | LR AUC %.4f slope %.3f B %.4f | "
            "XGB AUC %.4f slope %.3f B %.4f"
            % (tag, r['positives'], r['prevalence_pct'], r['Logistic']['auc'],
               r['Logistic']['slope'], r['Logistic']['brier'], r['XGBoost']['auc'],
               r['XGBoost']['slope'], r['XGBoost']['brier']))
        log("  [院内平行]     %-16s events=%d   | LR AUC %.4f slope %.3f | XGB AUC %.4f "
            "slope %.3f"
            % (tag, r2['events'], r2['Logistic']['auc'], r2['Logistic']['slope'],
               r2['XGBoost']['auc'], r2['XGBoost']['slope']))

    delta = {}
    for mn in ("Logistic", "XGBoost"):
        delta[mn] = dict(
            d_auc=round(res['recovered'][mn]['auc'] - res['hardcoded_zero'][mn]['auc'], 4),
            d_auc_inhosp=round(res['recovered']['inhosp'][mn]['auc']
                               - res['hardcoded_zero']['inhosp'][mn]['auc'], 4),
            d_brier=round(res['recovered'][mn]['brier'] - res['hardcoded_zero'][mn]['brier'], 4),
            d_slope=round(res['recovered'][mn]['slope'] - res['hardcoded_zero'][mn]['slope'], 4))
    log("  Δ（recovered - hardcoded_zero）: " + " | ".join(
        "%s ΔAUC %+.4f (院内 %+.4f) ΔBrier %+.4f Δslope %+.4f"
        % (m, delta[m]['d_auc'], delta[m]['d_auc_inhosp'], delta[m]['d_brier'],
           delta[m]['d_slope']) for m in ("Logistic", "XGBoost")))

    # ---------- 2.2 剔除敏感性
    #   注意：旧稿把「砍掉全部三个治疗强度指标」错写成了「去掉血管活性药」。这里两种都算。
    drop = {}
    ym = mim['death_365'].values
    for label, drop_cols in (("drop_vasopressor_only", ['vasopressor']),
                             ("drop_all_organ_support", list(BINV))):
        use = [f for f in FEATS if f not in drop_cols]
        mdsub = build_frozen(ym, use, mim)
        for tag, X in (("hardcoded_zero", src), ("recovered", dst)):
            a, b = apply_md(mdsub, X)
            y = X['death_365'].values.astype(int)
            drop.setdefault(label, {})[tag] = dict(
                n_predictors=len(use), Logistic=metrics(y, a), XGBoost=metrics(y, b))
        d = drop[label]
        log("  [剔除] %-24s n=%d | 旧 LR AUC %.4f slope %.3f → 新 LR AUC %.4f slope %.3f "
            "| 旧 XGB %.4f → 新 XGB %.4f"
            % (label, len(use), d['hardcoded_zero']['Logistic']['auc'],
               d['hardcoded_zero']['Logistic']['slope'], d['recovered']['Logistic']['auc'],
               d['recovered']['Logistic']['slope'], d['hardcoded_zero']['XGBoost']['auc'],
               d['recovered']['XGBoost']['auc']))
        drop[label]['internal'] = mdsub['internal']

    # ---------- 2.3 跨库患病率（用修正后的数据）
    prev = {}
    for nm, f in (("MIMIC(dev)", "mimic_abi_cohort.csv"), ("eICU", "eicu_abi_final.csv"),
                  ("INSPIRE", "inspire_abi_final.csv")):
        d_ = pd.read_csv(DATA / f)
        s = pd.to_numeric(d_['vasopressor'], errors='coerce').fillna(0)
        prev[nm] = round(100.0 * float(s.mean()), 1)
    prev["NWICU"] = round(100.0 * float(dst['vasopressor'].mean()), 1)
    log("  患病率（修正后）= %s" % prev)

    return dict(full_model=res, delta=delta, drop=drop, prevalence=prev), mim


# ============================================================ STAGE 3  断言
def stage3(src, dst, varn, evidence, R2, mim):
    log("")
    log("=" * 88)
    log("STAGE 3  断言")
    a = 0

    assert len(src) == len(dst) == 3420, "行数不符"
    a += 1
    assert list(src['hadm_id']) == list(dst['hadm_id']), "★ 主键顺序不一致（禁止按位置赋值）"
    a += 1
    pos = int(dst['vasopressor'].sum())
    assert 100 <= pos <= 1200, "恢复后的阳性数不合常理：%d" % pos
    pct_ = 100.0 * pos / len(dst)
    assert 5.0 <= pct_ <= 20.0, "恢复后的患病率 %.2f%% 落在 MIMIC(23.3)/INSPIRE(8.7) 之外" % pct_
    a += 1
    # 除 vasopressor 外，其余列逐列与原文件相同
    for c in src.columns:
        if c == 'vasopressor':
            continue
        same = (src[c].astype(str).values == dst[c].astype(str).values)
        assert bool(same.all()), "列 %s 被意外改动" % c
    a += 1
    log("  [OK] 除 vasopressor 外其余 %d 列逐列未变" % (len(src.columns) - 1))

    # A ⊇ B ⊇ D 且 A ⊇ C（阳性集合包含关系，符合收回的严格程度）
    sA, sB, sC, sD = (set(np.where(varn[k] > 0)[0]) for k in
                      ("A_any_route", "B_iv_or_injection", "C_iv_only", "D_iv_or_inj_no_local"))
    assert sD <= sB <= sA, "口径包含关系不成立"
    assert sC <= sA, "口径包含关系不成立(C)"
    a += 1
    log("  [OK] 四口径阳性集合满足包含关系 D⊆B⊆A、C⊆A")

    # 与既有 published 数字可复现（防止误用模型）
    R = R2['full_model']
    assert abs(R['hardcoded_zero']['Logistic']['auc'] - 0.6958) < 2e-3, \
        "写死 0 的 LR AUC 未能复现发表值：%.4f" % R['hardcoded_zero']['Logistic']['auc']
    assert abs(R['hardcoded_zero']['XGBoost']['auc'] - 0.7959) < 2e-3, \
        "写死 0 的 XGB AUC 未能复现发表值：%.4f" % R['hardcoded_zero']['XGBoost']['auc']
    assert abs(R['hardcoded_zero']['Logistic']['slope'] - 0.039) < 5e-3, \
        "写死 0 的 LR slope 未能复现发表值：%.4f" % R['hardcoded_zero']['Logistic']['slope']
    a += 1
    log("  [OK] 写死 0 的三个发表数字（0.6958 / 0.7959 / 0.039）全部复现")

    # 恢复后结论方向不变：AUC 变动必须很小
    for mn in ("Logistic", "XGBoost"):
        assert abs(R2['delta'][mn]['d_auc']) < 0.02, \
            "%s ΔAUC %.4f 超出预期量级，需人工审查" % (mn, R2['delta'][mn]['d_auc'])
    assert R['recovered']['Logistic']['slope'] < 0.20, \
        "恢复后 LR 斜率不再病态（%.3f），论文的 NWICU 论述需重写" \
        % R['recovered']['Logistic']['slope']
    a += 1
    log("  [OK] 恢复后 |ΔAUC| < 0.02 且 LR 斜率仍病态 → 全部结论方向不变")

    # drop_vasopressor_only 模型不含 vasopressor，故新旧必须完全一致（算法自检）
    dvo = R2['drop']['drop_vasopressor_only']
    for mn in ("Logistic", "XGBoost"):
        assert abs(dvo['recovered'][mn]['auc'] - dvo['hardcoded_zero'][mn]['auc']) < 1e-9, \
            "剔除模型不应受 NWICU 侧取值影响"
    a += 1
    log("  [OK] 剔除模型对 NWICU 侧取值不变（算法自检通过）")

    log("  [OK] 共 %d 组断言全部通过" % a)


def main():
    src, dst, varn, evidence = stage1()
    R2, mim = stage2(src, dst)
    stage3(src, dst, varn, evidence, R2, mim)

    payload = dict(
        generated_by="74_recover_nwicu_vasopressor.py",
        eps=EPS, seed=SEED,
        chosen_variant=CHOSEN,
        variants={k: dict(positives=int(v.sum()), prevalence_pct=round(100.0 * v.mean(), 2))
                  for k, v in varn.items()},
        source_evidence=evidence,
        harmonisation_error=dict(
            description="sql/11_nwicu_cohort.sql hard-coded `0 as vasopressor`; the source "
                        "does have hosp.prescriptions with vasopressor records.",
            published_prevalence_pct=0.0,
            recovered_prevalence_pct=R2['full_model']['recovered']['prevalence_pct'],
            delta_auc=R2['delta'],
        ),
        nwicu=R2['full_model'],
        prevalence=R2['prevalence'],
        drop_sensitivity=R2['drop'],
        output_data_file=str(DST.name),
        note="D 口径 = intravenous/injection route 且排除 lidocaine–epinephrine 局麻复方与 "
             "直肠栓剂；窗口 = intime-6h ~ intime+1day，与其余预测因子的首日口径一致。")
    json.dump(payload, open(JSN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("")
    log("[written] %s" % JSN.name)
    log("[written] %s" % DST.name)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
