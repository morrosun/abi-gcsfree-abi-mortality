# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
82 —— 面板/合成图的**像素级**可复现性验证（替代"同批生成，构造上不可能陈旧"这一不合格判据）。

为什么写它
----------
80 记录面板溯源时，对 66 现画的 6 张 `*_v8.png` 用了这样的理由：

    "均为 66 同批现画的 *_v8.png（构造上不可能陈旧）"

这个判据**不成立**：文件名后缀证明不了生成过程；同批运行也不能排除脚本读取旧输入、
命中缓存或走错分支。**唯一站得住的证据是"独立重跑后内容相同"**。

本脚本把口径统一为两级、内容等价性优先：

  1) 字节级：`backup/<f>` 与 `current/<f>` 逐字节比对（最严，但会被 PNG 元数据干扰）；
  2) **像素级（决定性）**：PIL 解码为 ndarray，比对形状与逐元素相等，
     并另存解码后原始字节的 sha256（该哈希与 PNG 元数据无关，可跨工具复核）。

方法：备份 → 重跑生成器 → 本脚本比对。**注意 `06`/`08` 不可重跑**（会回退到旧口径，
外验图的权威生成器是 `59`）；`figR2_inhosp_parallel.png` 被 `12` 与 `59` 同时生成且
数据源年代不同，取 `59` 输出，故该文件预期为 DIFFERS（不计入失败）。

输入（必须已存在，否则判 FAIL —— 不允许"没备份也算通过"）：
  output/_panel_backup_20261002/          04/12/59 面板在重跑前的状态
  output/_fig_backup_20261002/v8_panels/  66 的 6 张新面板在重跑前的状态
  output/_fig_backup_20261002/figures/    66 的 8 张合成图在重跑前的状态

产出：output/panel_pixel_repro.json
      output/_82_run.log
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(os.environ.get("ABI_BASE", Path(__file__).resolve().parents[1]))
OUT = BASE / "output"
V8 = OUT / "submission_v8"
PAN = OUT / "v8_panels"

BK_PANELS = OUT / "_panel_backup_20261002"          # 04/12/59 重跑前
BK_V8 = OUT / "_fig_backup_20261002" / "v8_panels"  # 66 重跑前（新面板）
BK_FIG = OUT / "_fig_backup_20261002" / "figures"   # 66 重跑前（合成图）

LOGP = OUT / "_82_run.log"
LOG = []
CHECKS = []


def log(s=""):
    print(s)
    LOG.append(str(s))


def chk(cond, msg):
    CHECKS.append(bool(cond))
    log("  [%s] %s" % ("OK " if cond else "FAIL", msg))
    return bool(cond)


def sha_bytes(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pixel_sha(p):
    """解码后像素数组的 sha256 —— 与 PNG 压缩/元数据无关。"""
    a = np.asarray(Image.open(p).convert("RGB"))
    return hashlib.sha256(a.tobytes()).hexdigest(), a.shape


def rel(p):
    """相对 BASE 的路径 —— 归档产物不得写入开发机绝对路径。"""
    try:
        return str(Path(p).resolve().relative_to(BASE.resolve())).replace("\\", "/")
    except ValueError:
        return "<outside ABI_BASE>/" + Path(p).name


def compare(tag, name, bak, cur):
    """返回一个 dict；bak/cur 任一缺失即 FAIL（不允许静默跳过）。"""
    rec = {"tag": tag, "file": name, "backup": rel(bak), "current": rel(cur)}
    if not bak.exists():
        chk(False, "%s %s：备份缺失（%s）—— 未做备份不算通过" % (tag, name, bak))
        rec["verdict"] = "NO BACKUP"
        return rec
    if not cur.exists():
        chk(False, "%s %s：当前文件缺失（%s）" % (tag, name, cur))
        rec["verdict"] = "MISSING CURRENT"
        return rec

    rec["bytes_identical"] = Path(bak).read_bytes() == Path(cur).read_bytes()
    rec["sha256_backup"] = sha_bytes(bak)[:16]
    rec["sha256_current"] = sha_bytes(cur)[:16]

    ph_b, shp_b = pixel_sha(bak)
    ph_c, shp_c = pixel_sha(cur)
    rec["pixel_sha256_backup"] = ph_b[:16]
    rec["pixel_sha256_current"] = ph_c[:16]
    rec["pixel_shape"] = list(shp_c)
    rec["pixel_identical"] = (ph_b == ph_c) and (shp_b == shp_c)
    if rec["pixel_identical"]:
        rec["verdict"] = "PIXEL_IDENTICAL"
    elif rec["bytes_identical"]:
        rec["verdict"] = "BYTE_IDENTICAL"      # 逻辑上不可达，留作哨兵
    else:
        # 量化差异规模，便于判断是"内容变了"还是"仅元数据/压缩变了"
        a = np.asarray(Image.open(bak).convert("RGB")).astype(np.int16)
        b = np.asarray(Image.open(cur).convert("RGB")).astype(np.int16)
        if a.shape == b.shape:
            d = np.abs(a - b)
            rec["n_pixels_differing"] = int((d.max(axis=2) > 0).sum())
            rec["max_abs_channel_diff"] = int(d.max())
        rec["verdict"] = "DIFFERS"
    return rec


def main():
    log("#" * 78)
    log("# 82_panel_pixel_repro.py | 面板与合成图的像素级可复现性验证")
    log("#" * 78)

    # ---- 前置：三份备份必须都在，否则整套验证没有意义 ----
    log("\n== 0. 前置：备份目录必须存在 ==")
    for p, why in ((BK_PANELS, "04/12/59 面板重跑前"),
                   (BK_V8, "66 新面板重跑前"),
                   (BK_FIG, "66 合成图重跑前")):
        chk(p.exists(), "备份目录存在：%s（%s）" % (p.name, why))

    records = []

    # ---- 1. 04 / 12 / 59 的面板（上一轮重跑的产物）----
    log("\n== 1. 生成器 04 / 12 / 59 的面板 ==")
    PRIOR = ["fig1_roc.png", "fig2_calibration.png", "fig3_dca.png",
             "fig4_forest.png", "fig5_xgb_importance.png",
             "fig6_ext_roc.png", "fig7_ext_roc_bymodel.png", "fig8_ext_calibration.png",
             "fig9_recal_calibration.png", "fig10_recal_ece.png",
             "figR1_gcs_increment.png", "figR2_inhosp_parallel.png", "figR3_rootcause.png"]
    for n in PRIOR:
        rec = compare("04/12/59", n, BK_PANELS / n, OUT / n)
        records.append(rec)
        log("  %-30s %s" % (n, rec["verdict"]))

    # ---- 2. 66 的 6 张新面板（本轮重跑）----
    log("\n== 2. 生成器 66 的新面板（本轮独立重跑后比对） ==")
    V8PANELS = ["gcs_components_v8.png", "leakage_dauc_v8.png", "slope_eps_v8.png",
                "calib_ci_v8.png", "harmonisation_v8.png", "huaian_panel_v8.png"]
    for n in V8PANELS:
        rec = compare("66", n, BK_V8 / n, PAN / n)
        records.append(rec)
        log("  %-30s %s" % (n, rec["verdict"]))

    # ---- 3. 66 的 8 张合成图（本轮重跑）----
    log("\n== 3. 生成器 66 的 8 张合成图（本轮独立重跑后比对） ==")
    figs = sorted(BK_FIG.glob("Figure*.png"))
    chk(len(figs) == 8, "备份中含 8 张合成图（实际 %d）" % len(figs))
    for f in figs:
        rec = compare("66-composite", f.name, f, V8 / "figures" / f.name)
        records.append(rec)
        log("  %-30s %s" % (f.name, rec["verdict"]))

    # ---- 判定 ----
    log("\n== 4. 判定 ==")
    KNOWN = {"figR2_inhosp_parallel.png": "被 12 与 59 同时生成，数据源年代不同；取 59（当前真源）。未被 V8 的 8 张图使用。"}
    bad = [r["file"] for r in records
           if r["verdict"] not in ("PIXEL_IDENTICAL",) and r["file"] not in KNOWN]
    chk(not bad, "除已知例外外，全部文件像素级一致（例外 %r）" % bad)
    for r in records:
        if r["file"] in KNOWN:
            log("  [NOTE] %s 判定 %s —— %s" % (r["file"], r["verdict"], KNOWN[r["file"]]))
    n_pix = sum(1 for r in records if r["verdict"] == "PIXEL_IDENTICAL")
    chk(n_pix >= len(records) - len(KNOWN),
        "像素级一致 %d / %d（已知例外 %d）" % (n_pix, len(records), len(KNOWN)))
    log("  字节级亦一致：%d / %d"
        % (sum(1 for r in records if r.get("bytes_identical")), len(records)))

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "backup -> regenerate (04/12/59 already rerun; 66 rerun this round) "
                  "-> byte compare + decoded-pixel compare (PIL RGB ndarray, sha256)",
        "note": ("pixel_sha256 is computed on the decoded RGB ndarray and is therefore "
                 "independent of PNG compression and metadata; bytes_identical is reported "
                 "for transparency but is the weaker criterion"),
        "known_exceptions": KNOWN,
        "n_compared": len(records),
        "n_pixel_identical": n_pix,
        "records": records,
    }
    (OUT / "panel_pixel_repro.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log("\n wrote panel_pixel_repro.json")

    log("\n" + "=" * 78)
    if all(CHECKS):
        log("ALL CHECKS PASSED  (%d files compared)" % len(records))
    else:
        log("FAILED  (%d/%d)" % (sum(CHECKS), len(CHECKS)))
    log("=" * 78)
    LOGP.write_text("\n".join(LOG), encoding="utf-8")
    return 0 if all(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
