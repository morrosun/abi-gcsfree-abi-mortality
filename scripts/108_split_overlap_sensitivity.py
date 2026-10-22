# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
108_split_overlap_sensitivity.py
================================
第六轮评审 §二（**本轮唯一的 P0**）：正文/S2/投稿信三处都写了
"the overlap is quantified by the sensitivity analysis on the 2,704 admissions
held out in both versions"，但**从未真的算过** —— S1–S10 里没有任何 2,704
子集的事件数、AUC、Brier 或校准。

披露重叠人数 ≠ 量化重叠对性能估计的影响。PROBAST 的 Analysis 域要求评价
过拟合、乐观偏倚与内部验证是否恰当，故必须给出**真实结果**，或撤回"已量化"
的声称。本脚本走前者：**实际算出来**。

★ 交集可以精确重建（106 号当时只给了规模）
------------------------------------------------
划分只依赖 (y, seed)：`train_test_split(idx, test_size=0.30, stratify=y, random_state=42)`。
两版标签的差集**逐行可定位**（`data/mimic_abi_labels_v2.csv` 里 `death_365`
与 `death_365_v2` 恰差 2 行：hadm_id 20469285 与 26870369，均为 0→1）。
于是：
  - y_old = death_365 （5,301 事件）
  - y_new = death_365_v2（5,303 事件）
两版切分各自可**精确重建**，交集 = test_new ∩ test_old 的行号**直接算出**，
不是估计、不是上界。（106 号的注释"旧标签逐行不可还原"已被本脚本纠正。）

口径（严格按评审要求）
----------------------
1. 使用**最终冻结** artefact（`preproc_v2.joblib` / `models_v2.joblib`），
   **不重拟合任何模型**；插补与标准化用训练集内已存的 medians/scaler。
2. 2,704 子集 = 两版都留出 ⇒ **未参与**所述早期树模型超参 CV。
   但仍如实标注：它不是"从未使用过的新测试集"，只是重叠部分的**无重叠子集**。
3. 与完整 4,980 测试集**并排**给：事件数、AUC(95% CI)、Brier、校准斜率、O:E，
   两个模型都算。差值即"重叠对内部估计的影响量"。
4. 只读：不修改任何既有数据或 artefact。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "output"
DATA = BASE / "data"

SEED = 42
TEST = 0.30
EPS = 1e-6
B_BOOT = 1000

# 与 87 号脚本逐字一致的常量（用 AST 抽取赋值节点，避免多行列表被截断）
import ast as _ast

_src = (BASE / "scripts" / "87_retrain_and_reevaluate.py").read_text(encoding="utf-8")
_tree = _ast.parse(_src)
_ns: dict = {}
WANT = {"CONT", "AET", "FEATS", "EPS", "SEED", "B_BOOT", "BINV"}
for _node in _tree.body:
    if isinstance(_node, _ast.Assign) and _node.targets:
        _t = _node.targets[0]
        if isinstance(_t, _ast.Name) and _t.id in WANT:
            try:
                _ns[_t.id] = _ast.literal_eval(_node.value)
            except ValueError:
                # FEATS = CONT + [...] 之类含表达式的赋值 → 逐条 eval，依赖按源码序累积
                _ns[_t.id] = eval(compile(_ast.Expression(_node.value), "f", "eval"),
                                  {"np": np, "pd": pd}, dict(_ns))
missing = WANT - set(_ns)
assert not missing, "未能从 87 号脚本抽取常量：%s" % sorted(missing)
CONT = _ns["CONT"]
AET = _ns["AET"]
FEATS = _ns["FEATS"]

LOG: list[str] = []


def log(m: str = "") -> None:
    print(m)
    LOG.append(m)


def metrics(y, p, eps=EPS) -> dict:
    """五项指标，与 87 号脚本逐字同口径。

    ⚠ 校准斜率的方向：87 号的 `cal_slope_intercept` 是把 **logit(p) 对 y 做
      LogisticRegression**（Cox 传统定义，斜率 1 = 完美校准），**不是**
      logit(y) 对 logit(p) 的 OLS。后者在本数据上会给 ~4.3（方向相反），
      两者不可混用。此处直接调用 87 号的同一函数以杜绝口径漂移。
    """
    from sklearn.linear_model import LogisticRegression as _LR

    y = np.asarray(y, float)
    p = np.asarray(p, float)
    pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc))
    lr = _LR(fit_intercept=True, C=1e12, solver="lbfgs",
             max_iter=2000).fit(lp.reshape(-1, 1), y.astype(int))
    return {"n": int(len(y)), "events": int(y.sum()),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "slope": float(lr.coef_[0][0]),
            "intercept": float(lr.intercept_[0]),
            "oe": float(y.mean() / p.mean())}


def auc_ci(y, p, B=B_BOOT, seed=SEED) -> list:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    acc = []
    n = len(y)
    for _ in range(B):
        sel = rng.integers(0, n, n)
        yy = y[sel]
        if yy.min() == yy.max():
            continue
        try:
            acc.append(roc_auc_score(yy, p[sel]))
        except Exception:  # noqa: BLE001
            continue
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def main() -> int:
    log("[108] 划分重叠敏感性分析 —— 2,704 例无重叠子集")
    log("=" * 72)

    # ---------------------------------------------------------- 1. 重建两版切分
    cur = pd.read_csv(DATA / "mimic_abi_cohort_v2.csv")
    lab = pd.read_csv(DATA / "mimic_abi_labels_v2.csv")
    # ⚠ cohort 里已有 death_365（86 已把 v2 写回），merge 会产生 _lab 后缀。
    #   两版标签必须**都取自 labels 文件**，故先从 cur 里剔除同名列。
    cur = cur.drop(columns=[c for c in ("death_365", "death_365_v2")
                            if c in cur.columns])
    m = cur.merge(lab[["hadm_id", "death_365", "death_365_v2"]],
                  on="hadm_id", how="left")
    assert len(m) == 16597, "合并后行数不符：%d" % len(m)

    y_new = m["death_365_v2"].astype(int).to_numpy()
    y_old = m["death_365"].astype(int).to_numpy()
    d_rows = m.loc[m["death_365"] != m["death_365_v2"], "hadm_id"].tolist()
    log("[OK] 两版标签差异行（逐行定位）：hadm_id=%s" % d_rows)
    assert len(d_rows) == 2, "标签差异应恰为 2 行，实为 %d" % len(d_rows)
    assert int(y_old.sum()) == 5301 and int(y_new.sum()) == 5303

    idx = np.arange(len(m))
    _, ite_new = train_test_split(idx, test_size=TEST, stratify=y_new, random_state=SEED)
    _, ite_old = train_test_split(idx, test_size=TEST, stratify=y_old, random_state=SEED)
    log("[OK] V9 test n=%d events=%d ｜ 早期 test n=%d events=%d"
        % (len(ite_new), int(y_new[ite_new].sum()), len(ite_old), int(y_old[ite_old].sum())))

    common = np.intersect1d(ite_new, ite_old)
    only_new = np.setdiff1d(ite_new, ite_old)
    log("[OK] ★交集（两版都留出）n=%d events=%d"
        % (len(common), int(y_new[common].sum())))
    log("    本版独有（曾在旧训练集）n=%d events=%d"
        % (len(only_new), int(y_new[only_new].sum())))

    # 与 106 号落盘计数对账
    sj = json.loads((OUT / "split_independence_v1.json").read_text(encoding="utf-8"))
    assert len(common) == int(sj["test_common"]) == 2704, \
        "交集应为 2,704，实为 %d" % len(common)
    assert len(only_new) == int(sj["test_only_v9"]) == 2276, \
        "本版独有应为 2,276，实为 %d" % len(only_new)
    log("[OK] 与 106 号脚本落盘计数完全一致（2,704 / 2,276）——交集已精确重建")

    # ---------------------------------------------------------- 2. 冻结 artefact 预测
    pre = joblib.load(OUT / "preproc_v2.joblib")
    models = joblib.load(OUT / "models_v2.joblib")

    # ★ 两个模型吃**不同的矩阵**（87 号 build() 原文）：
    #     LR  → Xtr_s（CONT 经 StandardScaler 标准化）
    #     XGB → Xtr   （**原始尺度**，只做插补/填 0，不标准化）
    #   我先前把标准化后的矩阵喂给 XGBoost，导致 AUC=0.5（完全无信息）——
    #   这是本脚本自己的坑，已按 87 号口径拆成两个矩阵。
    X_raw = pd.DataFrame(index=m.index)
    X_raw["female"] = (m["gender"] == "F").astype(int)
    for c in FEATS:
        if c != "female":
            X_raw[c] = pd.to_numeric(m[c], errors="coerce")
    X_raw = X_raw[FEATS]
    for c in CONT:
        X_raw[c] = X_raw[c].fillna(pre["medians"][c])
    for c in FEATS:
        if c not in CONT:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0)

    X_sc = X_raw.copy()
    X_sc[CONT] = pre["scaler"].transform(X_sc[CONT])

    res = {
        "n_test": int(len(ite_new)), "events_test": int(y_new[ite_new].sum()),
        "n_common": int(len(common)), "events_common": int(y_new[common].sum()),
        "n_only_v9": int(len(only_new)), "events_only_v9": int(y_new[only_new].sum()),
        "label_diff_hadm_ids": [int(x) for x in d_rows],
        "eps": EPS,
        "provenance": ("划分仅依赖 (y, seed)；两版标签差集逐行可定位"
                       "（data/mimic_abi_labels_v2.csv），故交集为精确重建而非估计。"
                       "预测用最终冻结 artefact，未重拟合。"),
    }

    log("")
    log("%-9s %-6s %-7s %-21s %-8s %-8s %-7s"
        % ("model", "subset", "events", "AUC (95% CI)", "Brier", "slope", "O:E"))
    log("-" * 72)
    for mname in ("logistic", "xgboost"):
        if mname == "logistic":
            # 冻结 artefact 里 logistic 以**系数向量**保存（logit_params），
            # 不是 sklearn 估计器 —— 直接用系数算预测，避免重拟合。
            lp = models["logit_params"]
            beta = np.array([lp[f] for f in FEATS], float)
            b0 = float(lp["const"])

            def _plr(frame, _b=beta, _b0=b0):
                return 1.0 / (1.0 + np.exp(-(_b0 + frame[FEATS].to_numpy() @ _b)))

            p_te = _plr(X_sc.iloc[ite_new])
            p_cm = _plr(X_sc.iloc[common])
        else:
            # ★ XGBoost 吃**原始尺度**矩阵（87 号：只插补/填 0，不标准化）
            # ★ 与 87 号 apply_frozen() 逐字同口径：predict_proba(Z[FEATS].values)[:, 1]
            _xgb = models["xgb"]
            p_te = _xgb.predict_proba(X_raw[FEATS].values[ite_new])[:, 1]
            p_cm = _xgb.predict_proba(X_raw[FEATS].values[common])[:, 1]
        m_te = metrics(y_new[ite_new], p_te)
        m_cm = metrics(y_new[common], p_cm)
        m_te["auc_ci"] = auc_ci(y_new[ite_new], p_te)
        m_cm["auc_ci"] = auc_ci(y_new[common], p_cm)
        res[mname] = {"full_test": m_te, "common_subset": m_cm,
                      "delta_auc": m_cm["auc"] - m_te["auc"],
                      "delta_auc_ci_width": (m_cm["auc_ci"][1] - m_cm["auc_ci"][0]
                                             - (m_te["auc_ci"][1] - m_te["auc_ci"][0]))}
        for tag, mm in (("4,980", m_te), ("2,704", m_cm)):
            log("%-9s %-6s %-7d %.3f (%.3f-%.3f)   %-8.4f %-8.3f %-7.3f"
                % (mname if tag == "4,980" else "", tag, mm["events"],
                   mm["auc"], mm["auc_ci"][0], mm["auc_ci"][1],
                   mm["brier"], mm["slope"], mm["oe"]))
        log("   → ΔAUC(%s−%s) = %+.4f"
            % (mname, mname, res[mname]["delta_auc"]))
    log("-" * 72)

    # ---------------------------------------------------------- 3. 写盘
    op = OUT / "split_overlap_sensitivity_v1.json"
    op.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[OK] 写出 %s" % op.name)
    (OUT / "_108_sensitivity.log").write_text("\n".join(LOG), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())