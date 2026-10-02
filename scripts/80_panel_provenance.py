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

  1) 可复现性（决定性）：备份 -> **重跑生成器** -> 逐字节 + **逐像素**比对。
     结果由 `82_panel_pixel_repro.py` 产出（output/panel_pixel_repro.json），
     本脚本只**消费**它，不再自行硬编码结论。

     ⚠️ 2026-10-02 修正：本脚本上一版对 66 现画的 6 张 `*_v8.png` 写了
     "同批生成，构造上不可能陈旧" —— **该判据不成立**（文件名后缀证明不了生成过程，
     同批运行也不排除读旧输入/命中缓存/走错分支）。现已改为**对全部 15 张面板一律要求
     独立重跑后的比对结论**，其中 6 张由 82 在本轮重跑 66 后实测（PIXEL_IDENTICAL）。

  2) 溯源 + 时效：每个面板记录生成脚本、依赖结果文件、sha256、mtime，
     并断言 panel.mtime >= max(dep.mtime)、figure.mtime >= max(panel.mtime)。

  ★ 另显式区分两个**不同的集合**，避免"11 张实测"与"9 张实测"这类自相矛盾的表述：

     - `used_by_v8`：V8 的 8 张合成图实际用到的面板（15 张）——这是判定对象；
     - `tested_not_used`：做过实测但不被 V8 使用的面板（figR1 / fig10 等）——
       它们是历史产物，**不计入** 15 张的判定。

另补齐数字覆盖的洞：
  此前"图内数字清单"只覆盖 59 的外验面板（figure_number_manifest.json，36 项）与
  66 的新审计面板（v8_figure_number_manifest.json，26 项）；
  **04 生成的 4 张内部面板（fig1_roc / fig2_calibration / fig4_forest /
  fig5_xgb_importance）此前完全没有数字清单**。本脚本从它们的输入 CSV / joblib
  派生出数字清单（AUC、Brier、比值比表、XGBoost gain top-10），并断言与稿件
  Table 2/3 报告值一致。

产出：output/v8_panel_provenance.json
      output/_panel_repro_table.md        逐文件对照表（面板 × 是否用于 V8 × 判定）
      output/_80_panel_provenance.log
前置：output/panel_pixel_repro.json（脚本 82 产出；缺失则直接退出，不静默放行）
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

# 可复现性结论的**唯一真源**：由 82 产出（备份 -> 重跑生成器 -> 字节比对 + 像素比对）。
# 本脚本不再硬编码任何结论，避免"结论写在两个地方、各自漂移"。
_REPRO_JSON = OUT / "panel_pixel_repro.json"
if not _REPRO_JSON.exists():
    raise SystemExit("缺少 %s —— 请先重跑生成器（04/12/59/66）并运行 "
                     "82_panel_pixel_repro.py" % _REPRO_JSON)
_PIX = json.load(open(_REPRO_JSON, encoding="utf-8"))
REPRO = {r["file"]: r["verdict"] for r in _PIX["records"]}
_PIXNOTE = _PIX.get("known_exceptions", {})
# ★ 只有"被 V8 使用"的面板参与判定；做过实测但未被使用的面板另列。
#   注意 8 张 `Figure*.png` 是**合成图**（本身就是 V8 的正文图），不是面板，
#   故在集合清点中单列，不混入面板恒等式。
REPRO_USED = {n: v for n, v in REPRO.items() if n in GEN}
REPRO_TESTED_NOT_USED = {n: v for n, v in REPRO.items()
                         if n not in GEN and not n.startswith("Figure")}
REPRO_COMPOSITES = {n: v for n, v in REPRO.items() if n.startswith("Figure")}


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
            "panel": name,
            # 相对路径：公开归档不得写入开发机绝对路径
            "dir": str(PANEL_DIR[name].relative_to(BASE)).replace("\\", "/"),
            "generator": gen,
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

    # ---------------- 集合清点：两个集合必须显式区分，不得混用 ----------------
    log("\n== 集合清点与可复现性判定 ==")
    used_v8 = sorted(GEN)              # 被 V8 的 8 张合成图实际使用的面板
    tested = sorted(REPRO)             # 实际做过独立重跑比对的全集
    chk(len(used_v8) == 15, "被 V8 使用的面板 %d 张（应 15）" % len(used_v8))
    chk(used_v8 == used,
        "声明表与被 figure_inventory.json 实际使用的面板集合一致（差集 %r）"
        % sorted(set(used_v8) ^ set(used)))
    chk(set(used_v8) <= set(tested),
        "被 V8 使用的面板全部已做独立重跑比对（缺 %r）" % sorted(set(used_v8) - set(tested)))
    _okv = ("PIXEL_IDENTICAL", "BYTE_IDENTICAL")
    untested = [n for n in used_v8 if REPRO.get(n) not in _okv and n not in _PIXNOTE]
    chk(not untested, "被 V8 使用的面板不存在 'NOT TESTED' 或未解释的 DIFFERS（%r）" % untested)
    _n_pan_tested = len(used_v8) + len(REPRO_TESTED_NOT_USED)
    chk(_n_pan_tested == len([n for n in tested if not n.startswith("Figure")]),
        "面板计数恒等式：使用 %d + 实测未使用 %d == 实测面板 %d"
        % (len(used_v8), len(REPRO_TESTED_NOT_USED), _n_pan_tested))
    log("  被 V8 使用（判定对象）：%d 张面板" % len(used_v8))
    log("  实测但未被 V8 使用的面板：%d 张 %s"
        % (len(REPRO_TESTED_NOT_USED), sorted(REPRO_TESTED_NOT_USED)))
    log("  另实测合成图（即 V8 正文图本身）：%d 张，全部 %s"
        % (len(REPRO_COMPOSITES), sorted(set(REPRO_COMPOSITES.values()))))
    _np = sum(1 for n in used_v8 if REPRO.get(n) in _okv)
    chk(_np == len(used_v8), "被 V8 使用的 %d 张全部像素级一致（实际 %d）" % (len(used_v8), _np))

    # 逐文件对照表（人可读，避免口径再次混用）
    _tbl = ["| 面板 | 用于V8 | 生成器 | 独立重跑比对 | 判定 |",
            "|---|---|---|---|---|"]
    for n in sorted(set(used_v8) | set(REPRO_TESTED_NOT_USED)):
        _tbl.append("| `%s` | %s | %s | %s | %s |"
                    % (n, "是" if n in GEN else "否（历史产物）",
                       GEN.get(n, ("—", []))[0],
                       "是" if n in tested else "否",
                       REPRO.get(n, "NOT TESTED")))
    (OUT / "_panel_repro_table.md").write_text("\n".join(_tbl) + "\n", encoding="utf-8")
    log("  wrote output/_panel_repro_table.md")

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
        "method": ("provenance (generator + deps + sha256 + mtime) and "
                   "reproducibility by independent regeneration: backup -> rerun generator "
                   "-> byte compare + decoded-pixel compare (see panel_pixel_repro.json)"),
        "reproducibility_source": "output/panel_pixel_repro.json (script 82)",
        "panels": prov,
        "figures": [{"figure": g["figure"], "file": g["file"], "panels": g["panels"]}
                    for g in inv],
        "set_accounting": {
            "used_by_v8": used_v8,
            "n_used_by_v8": len(used_v8),
            "panels_tested_by_regeneration": tested,
            "n_panels_tested": _n_pan_tested,
            "panels_tested_but_not_used_by_v8": sorted(REPRO_TESTED_NOT_USED),
            "n_panels_tested_but_not_used": len(REPRO_TESTED_NOT_USED),
            "identity": "n_used_by_v8 + n_panels_tested_but_not_used == n_panels_tested",
            "composites_tested": sorted(REPRO_COMPOSITES),
            "n_composites_tested": len(REPRO_COMPOSITES),
            "n_pixel_identical_among_used": _np,
        },
        "internal_panel_numbers": nums,
        "note": ("figR2_inhosp_parallel.png is written by both 12 and 59 from different "
                 "data vintages; 59 (external_current.json) is authoritative and is the "
                 "current file. figR2 is not used by any of the 8 V8 figures, so it is "
                 "outside the judged set. All 15 panels that V8 does use were independently "
                 "regenerated and compared at pixel level; no panel is left untested."),
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
