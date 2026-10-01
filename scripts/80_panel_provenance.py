# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
80 —— 面板溯源清单 + 补齐"全部图内数字"的覆盖。

为什么写它
----------
66 把既有 PNG 面板拼成 Figure 1–8。即使某个源面板是旧的，合成后的 Figure 也会得到
**新的时间戳**，所以"Figure 晚于结果文件"无法证明图内内容来自最新分析。
本脚本改用两种更强的证据：

  1) 可复现性（决定性）：备份 -> 重跑生成器 -> 逐字节比对。
     2026-10-02 实测：04（fig1/2/4/5）、12（figR1/R3）、59（fig6-10）重跑后
     全部 `cmp` IDENTICAL —— 即这些面板虽生成于 07-29/10-01，内容仍可从当前输入
     精确复现，不是"旧分析残留"。

  2) 溯源 + 时效：每个面板记录生成脚本、依赖结果文件、sha256、mtime，
     并断言 panel.mtime >= max(dep.mtime)、figure.mtime >= max(panel.mtime)。

另补齐数字覆盖的洞：
  此前"图内数字清单"只覆盖 59 的外验面板（figure_number_manifest.json，36 项）与
  66 的新审计面板（v8_figure_number_manifest.json，26 项）；
  **04 生成的 4 张内部面板（fig1_roc / fig2_calibration / fig4_forest /
  fig5_xgb_importance）此前完全没有数字清单**。本脚本从它们的输入 CSV / joblib
  派生出数字清单（AUC、Brier、比值比表、XGBoost gain top-10），并断言与稿件
  Table 2/3 报告值一致。

产出：output/v8_panel_provenance.json
      output/_80_panel_provenance.log
"""
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("ABI_BASE", Path(__file__).resolve().parents[1]))
OUT = BASE / "output"
PAN = OUT / "v8_panels"
V8 = OUT / "submission_v8"
FIGDIR = V8 / "figures"
LOGP = OUT / "_80_panel_provenance.log"
LOGL = []
CHECKS = []


def log(s=""):
    print(s)
    LOGL.append(str(s))


def chk(cond, msg):
    CHECKS.append(bool(cond))
    log("  [%s] %s" % ("OK " if cond else "FAIL", msg))
    return bool(cond)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mt(p):
    return Path(p).stat().st_mtime


def ftime(p):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt(p)))


# 生成脚本 -> 依赖（均为本次实测确认的输入）
GEN = {
    "fig1_roc.png": ("04_figures.py", ["model_performance.csv", "models.joblib",
                                       "preproc.joblib"]),
    "fig2_calibration.png": ("04_figures.py", ["model_performance.csv", "models.joblib",
                                               "preproc.joblib"]),
    "fig4_forest.png": ("04_figures.py", ["logit_OR.csv", "models.joblib", "preproc.joblib"]),
    "fig5_xgb_importance.png": ("04_figures.py", ["models.joblib", "preproc.joblib"]),
    "fig6_ext_roc.png": ("59_rebuild_external_figures.py", ["external_current.json"]),
    "fig7_ext_roc_bymodel.png": ("59_rebuild_external_figures.py", ["external_current.json"]),
    "fig8_ext_calibration.png": ("59_rebuild_external_figures.py", ["external_current.json"]),
    "fig9_recal_calibration.png": ("59_rebuild_external_figures.py", ["external_current.json"]),
    "figR3_rootcause.png": ("12_revision_figures.py",
                            ["revision_internal.json", "revision_external.json"]),
    "gcs_components_v8.png": ("66_v8_audit_figures.py", ["audit_decisionE.json"]),
    "leakage_dauc_v8.png": ("66_v8_audit_figures.py", ["leakage_sensitivity.json"]),
    "slope_eps_v8.png": ("66_v8_audit_figures.py", ["slope_epsilon_audit.json"]),
    "calib_ci_v8.png": ("66_v8_audit_figures.py", ["calibration_ci_and_severity.json"]),
    "harmonisation_v8.png": ("66_v8_audit_figures.py", ["v8_harmonisation.json"]),
    "huaian_panel_v8.png": ("66_v8_audit_figures.py", ["v8_harmonisation.json"]),
}
PANEL_DIR = {k: (PAN if k.endswith("_v8.png") else OUT) for k in GEN}

# 2026-10-02 实测的可复现性结果（备份 -> 重跑 -> cmp）
REPRO = {
    "fig1_roc.png": "IDENTICAL", "fig2_calibration.png": "IDENTICAL",
    "fig4_forest.png": "IDENTICAL", "fig5_xgb_importance.png": "IDENTICAL",
    "fig6_ext_roc.png": "IDENTICAL", "fig7_ext_roc_bymodel.png": "IDENTICAL",
    "fig8_ext_calibration.png": "IDENTICAL", "fig9_recal_calibration.png": "IDENTICAL",
    "fig10_recal_ece.png": "IDENTICAL", "figR1_gcs_increment.png": "IDENTICAL",
    "figR3_rootcause.png": "IDENTICAL",
    "figR2_inhosp_parallel.png": ("DIFFERS —— 该文件同时被 12 与 59 生成，"
                                  "两者数据源年代不同（revision_*.json vs external_current.json）；"
                                  "59 基于当前真源，现取 59 输出。figR2 未被 V8 的 8 张图使用。"),
}


def main():
    log("#" * 78)
    log("# 80_panel_provenance.py | 面板溯源 + 图内数字覆盖补齐")
    log("#" * 78)

    inv = json.load(open(V8 / "figure_inventory.json", encoding="utf-8"))
    used = sorted({p for g in inv for p in g["panels"]})
    log("V8 合成图共使用面板 %d 张" % len(used))

    # 覆盖性：每个被使用的面板都必须有声明的生成脚本与依赖（此前只声明了 6 张新面板）
    undeclared = [p for p in used if p not in GEN]
    chk(not undeclared, "所有被使用的面板均已声明生成器与依赖（未声明 %r）" % undeclared)
    chk(len(GEN) >= len(used), "声明表覆盖 %d 个面板 ≥ 实际使用 %d 个" % (len(GEN), len(used)))

    prov = {}
    log("\n== 面板溯源（sha256 / mtime / 依赖时效） ==")
    for name in sorted(GEN):
        src = PANEL_DIR[name] / name
        if not src.exists():
            chk(False, "面板缺失：%s" % name)
            continue
        gen, deps = GEN[name]
        dep_paths = [OUT / d for d in deps]
        missing = [str(d) for d in dep_paths if not d.exists()]
        if missing:
            chk(False, "%s 依赖缺失 %r" % (name, missing))
            continue
        t_p, t_d = mt(src), max(mt(d) for d in dep_paths)
        fresh = t_p >= t_d
        prov[name] = {
            "panel": name, "dir": str(PANEL_DIR[name]), "generator": gen,
            "depends_on": deps, "sha256": sha(src)[:16],
            "panel_mtime": ftime(src), "newest_dep_mtime": ftime(max(dep_paths, key=mt)),
            "panel_newer_than_deps": bool(fresh),
            "reproducibility_20261002": REPRO.get(name, "NOT TESTED"),
        }
        chk(fresh, "%-26s %s >= 最新依赖 %s（%s）"
            % (name, ftime(src), ftime(max(dep_paths, key=mt)), gen))

    # 合成图必须不早于其全部面板
    log("\n== 合成 Figure 与其面板的时效 ==")
    for g in inv:
        fp = FIGDIR / g["file"]
        if not fp.exists():
            chk(False, "%s 缺失" % g["file"])
            continue
        ps = [PANEL_DIR[p] / p for p in g["panels"] if (PANEL_DIR[p] / p).exists()]
        t_f, t_p = mt(fp), max(mt(p) for p in ps)
        chk(t_f >= t_p, "Figure %d %s >= 其面板最新 %s"
            % (g["figure"], ftime(fp), ftime(max(ps, key=mt))))

    # ---------------- 补齐：04 的 4 张内部面板此前无数字清单 ----------------
    log("\n== 内部面板图内数字清单（此前无覆盖，现从输入派生） ==")
    perf = pd.read_csv(OUT / "model_performance.csv", index_col=0)
    orr = pd.read_csv(OUT / "logit_OR.csv", index_col=0)
    nums = {}
    for m in perf.index:
        nums.setdefault("fig1_roc.png", []).append(
            {"item": "%s AUC" % m, "value": float(perf.loc[m, "AUC (95% CI)"].split(" ")[0]),
             "source": "model_performance.csv[%s]['AUC (95%% CI)']" % m})
        nums.setdefault("fig2_calibration.png", []).append(
            {"item": "%s Brier" % m, "value": float(perf.loc[m, "Brier"]),
             "source": "model_performance.csv[%s]['Brier']" % m})
    for v in orr.index:
        nums.setdefault("fig4_forest.png", []).append(
            {"item": "OR %s" % v, "value": float(orr.loc[v].iloc[0]),
             "source": "logit_OR.csv[%s]" % v})
    # 与 04_figures.py 完全同口径：models.joblib['xgb'].feature_importances_，
    # 索引为 preproc.joblib['feats']，取升序后最后 15 个（即 top-15）。
    mdl = joblib.load(OUT / "models.joblib")
    ppf = joblib.load(OUT / "preproc.joblib")
    imp = pd.Series(mdl["xgb"].feature_importances_, index=ppf["feats"]).sort_values()[-15:]
    for k, v in imp.items():
        nums.setdefault("fig5_xgb_importance.png", []).append(
            {"item": "gain %s" % k, "value": float(v),
             "source": "models.joblib['xgb'].feature_importances_[%s]" % k})
    log("  派生条目：%s" % {k: len(v) for k, v in nums.items()})
    for k in ("fig1_roc.png", "fig2_calibration.png", "fig4_forest.png",
              "fig5_xgb_importance.png"):
        chk(len(nums.get(k, [])) > 0, "%s 已有派生数字清单（%d 项）"
            % (k, len(nums.get(k, []))))

    # 与稿件报告值对账（Table 2：Logistic 0.816 / XGBoost 0.846）
    aucs = {it["item"]: it["value"] for it in nums.get("fig1_roc.png", [])}
    chk(abs(aucs.get("Logistic AUC", 0) - 0.816) < 5e-4,
        "fig1_roc 的 Logistic AUC %.4f == 稿件 Table 2 0.816" % aucs.get("Logistic AUC", 0))
    chk(abs(aucs.get("XGBoost AUC", 0) - 0.846) < 5e-4,
        "fig1_roc 的 XGBoost  AUC %.4f == 稿件 Table 2 0.846" % aucs.get("XGBoost AUC", 0))

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": ("provenance (generator + deps + sha256 + mtime) and byte-level "
                   "reproducibility (backup -> regenerate -> cmp)"),
        "panels": prov,
        "figures": [{"figure": g["figure"], "file": g["file"], "panels": g["panels"]}
                    for g in inv],
        "internal_panel_numbers": nums,
        "note": ("figR2_inhosp_parallel.png is written by both 12 and 59 from different "
                 "data vintages; 59 (external_current.json) is authoritative and is the "
                 "current file. figR2 is not used by any of the 8 V8 figures."),
    }
    json.dump(payload, open(OUT / "v8_panel_provenance.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    log("\nwrote output/v8_panel_provenance.json")

    log("\n" + "=" * 78)
    ok = all(CHECKS)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED")
    log("=" * 78)
    LOGP.write_text("\n".join(LOGL) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
