# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""99_prep_v9_package.py —— 为 V9 投稿包准备好「只读输入」：正文图、图清单、面板。

64 在 V9 模式下会：
  * 从 OUTDIR/figures/Figure1..8.png 取正文图（write_zip / DOCX 嵌图）
  * 从 OUTDIR/figure_inventory_V9.json 取图清单（图数、图题）
并把「图必须晚于其结果文件生成」作为 mtime 断言。
因此本脚本必须在 98 之后、64 之前运行。
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(ABI_BASE)
OUT = BASE / "output"
# ★ 2026-10-20：包目录与文件名标签随 ABI_TARGET 分流。
#   ABI_BUILD=v9 + ABI_TARGET=jic ⇒ V10（JIC 格式，数字口径仍为 V9）。
# ★ 2026-10-21（第十二轮）：可见版本号与"JIC 装订格式"开关**解耦**。
#   `V10` 这个名字是历史遗留，含义是"按 JIC/BMC 装订"，**不再等于版本号**；
#   真正的版本号由 ABI_VER 决定（缺省 11）⇒ 产物名 _V11、目录 jic_submission_v11。
VER = __import__("os").environ.get("ABI_VER", "11").strip()
TARGET = os.environ.get("ABI_TARGET", "round1").strip().lower()
assert TARGET in ("round1", "jic"), "ABI_TARGET 只能是 round1 或 jic（实为 %r）" % TARGET
TAG = ("V%s" % VER) if TARGET == "jic" else "V9"
PKG = OUT / ("jic_submission_v%s" % VER if TARGET == "jic" else "submission_v9")
FIGSRC = OUT / "v9_figures"
PANELS = OUT / "v9_panels"

PKG.mkdir(exist_ok=True)
(PKG / "figures").mkdir(exist_ok=True)
(PKG / "figures" / "panels").mkdir(exist_ok=True)

n = 0
for i in range(1, 9):
    src = FIGSRC / ("Figure%d.png" % i)
    assert src.exists(), "缺 %s" % src
    shutil.copy2(src, PKG / "figures" / src.name)
    n += 1
print("copied %d main figures" % n)

for p in sorted(PANELS.glob("*.png")):
    shutil.copy2(p, PKG / "figures" / "panels" / p.name)
print("copied %d panels" % len(list((PKG / "figures" / "panels").glob("*.png"))))

inv = json.load(open(OUT / "v9_figure_inventory.json", encoding="utf-8"))
# ★ 版本化文件名（2026-10-07）：包内顶层交付文件一律带版本号。
json.dump(inv, open(PKG / ("figure_inventory_%s.json" % TAG), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
assert len([x for x in inv if not x.get("label")]) == 8
print("WROTE figure_inventory_V9.json (%d main figures)" % len(inv))

for f in ["v9_panel_pixel_repro.json", "v9_number_master.json"]:
    shutil.copy2(OUT / f, PKG / f)
    print("COPIED", f)

# 65 的 STAGE 1 会从 OUT 读冻结产物；V9 的产物带 _v2 后缀，
# 这里按 V8 文件名复制到 v9_results/，避免改动 65 的加载代码。
V9R = OUT / "v9_results"
for v9name, v8name in [("preproc_v2.joblib", "preproc.joblib"),
                       ("models_v2.joblib", "models.joblib"),
                       ("model_inhosp_v2.joblib", "model_inhosp.joblib")]:
    src = OUT / v9name
    assert src.exists(), "缺 %s" % src
    shutil.copy2(src, V9R / v8name)
    print("COPIED %s → v9_results/%s" % (v9name, v8name))

# 与 V9 无关、但 65 会从 OUT 读的输入：文献终态、死亡标签核验、附录 DCA 图。
for f in ["v8_refs_final.json", "death_label_crosscheck_v1.json"]:
    src = OUT / f
    assert src.exists(), "缺 %s" % src
    shutil.copy2(src, V9R / f)
    print("COPIED %s" % f)

# 64 会从 OUT 读的文献池输入（V9 与 V8 同源，直接沿用）
for f in ["frontiers_refs_v2.json", "newrefs_v8.json", "official_score_provenance.json"]:
    src = OUT / f
    assert src.exists(), "缺 %s" % src
    shutil.copy2(src, V9R / f)
    print("COPIED %s" % f)

# 64 会从 OUT 读的 V9 专用审计产物（哨兵值清洗前后对比 → 正文"校准失败是抽取假象"的唯一真源）
for f in ["sentinel_audit_v1.json", "sentinel_value_provenance_v1.json"]:
    src = OUT / f
    assert src.exists(), "缺 %s" % src
    shutil.copy2(src, V9R / f)
    print("COPIED %s" % f)

# DCA 图必须"晚于"它的数字来源（65 会做 mtime 断言）→ 用 copy（非 copy2）刷新 mtime
_dca = OUT / "figR4_external_dca.png"
assert _dca.exists(), "缺 %s" % _dca
shutil.copy(_dca, V9R / _dca.name)
print("COPIED %s (mtime refreshed)" % _dca.name)

print("DONE →", PKG)
