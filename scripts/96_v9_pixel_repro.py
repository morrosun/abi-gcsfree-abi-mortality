# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
96_v9_pixel_repro.py —— V9 图的像素级可复现验证

铁律：图内数字只能有一个归档出处；"同一批时间戳"不算可复现。
本脚本做真正的复现验证：
  1. 备份当前 v9_panels/ 与 v9_figures/
  2. **独立重跑** 94（主结果图）→ 95（外部图）→ 92（审计面板 + 组合）
  3. 逐文件：PIL 解码为 ndarray 逐像素比对 + 解码后字节 sha256 比对
  4. 全部一致才算 PIXEL_IDENTICAL；列出任何不一致的文件并保留两侧副本供人工看

产出：output/v9_panel_pixel_repro.json + _96_v9_pixel_repro.md
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(ABI_BASE)
OUT = BASE / "output"
PAN = OUT / "v9_panels"
FIG = OUT / "v9_figures"
BAK = OUT / "v9_repro_backup"
PY = sys.executable   # 用当前解释器，避免写死本机路径

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def sha_bytes(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cmp_png(p1, p2):
    """返回 (ndarray 是否全等, 解码后字节 sha 是否相同, 尺寸)"""
    a = np.asarray(Image.open(p1).convert("RGB"))
    b = np.asarray(Image.open(p2).convert("RGB"))
    if a.shape != b.shape:
        return False, False, list(a.shape)
    same_arr = bool(np.array_equal(a, b))
    sa = hashlib.sha256(a.tobytes()).hexdigest()
    sb = hashlib.sha256(b.tobytes()).hexdigest()
    return same_arr, sa == sb, list(a.shape)


def main():
    log("=" * 88)
    log("96 —— V9 图像素级可复现验证")
    log("=" * 88)

    # ---------- 1. 备份 ----------
    if BAK.exists():
        shutil.rmtree(BAK)
    shutil.copytree(PAN, BAK / "v9_panels")
    shutil.copytree(FIG, BAK / "v9_figures")
    names_p = sorted(x.name for x in PAN.glob("*.png"))
    names_f = sorted(x.name for x in FIG.glob("*.png"))
    log("  备份 %d 个面板 + %d 张正文图 → %s" % (len(names_p), len(names_f), BAK.name))

    # ---------- 2. 独立重跑 ----------
    for script in ["94_v9_main_figures.py", "95_v9_external_figures.py",
                   "92_v9_audit_figures.py"]:
        log("")
        log("  ── 重跑 scripts/%s" % script)
        r = subprocess.run([PY, str(BASE / "scripts" / script)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(BASE))
        tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-3:]
        for l in tail:
            log("     | " + l[:150])
        if r.returncode != 0:
            log("  !!! %s 退出码 %d" % (script, r.returncode))
            log((r.stderr or "")[-1500:])
            return 1

    # ---------- 3. 比对 ----------
    log("")
    log("=" * 88)
    log("比对结果")
    log("=" * 88)
    res = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
           'panels': {}, 'figures': {}, 'n_ok': 0, 'n_fail': 0}
    for kind, names, cur, old in [('panels', names_p, PAN, BAK / "v9_panels"),
                                  ('figures', names_f, FIG, BAK / "v9_figures")]:
        for n in names:
            f_new, f_old = cur / n, old / n
            if not f_new.exists() or not f_old.exists():
                res[kind][n] = {'status': 'MISSING'}
                res['n_fail'] += 1
                log("  [MISSING] %s/%s" % (kind, n))
                continue
            same_arr, same_sha, shape = cmp_png(f_old, f_new)
            both = same_arr and same_sha
            res[kind][n] = {'status': 'PIXEL_IDENTICAL' if both else 'DIFFERS',
                            'ndarray_equal': same_arr, 'decoded_sha_equal': same_sha,
                            'shape': shape,
                            'file_sha_before': sha_bytes(f_old),
                            'file_sha_after': sha_bytes(f_new)}
            if both:
                res['n_ok'] += 1
                log("  [OK    ] %-9s %-28s %sx%s" % (kind, n, shape[1], shape[0]))
            else:
                res['n_fail'] += 1
                log("  [DIFFER] %-9s %-28s ndarray_equal=%s decoded_sha_equal=%s"
                    % (kind, n, same_arr, same_sha))
                shutil.copy(f_old, OUT / ("_reprodiff_before_%s" % n))
                shutil.copy(f_new, OUT / ("_reprodiff_after_%s" % n))

    total = res['n_ok'] + res['n_fail']
    res['total'] = total
    log("")
    log("  合计 %d 个文件：PIXEL_IDENTICAL %d / DIFFERS %d" %
        (total, res['n_ok'], res['n_fail']))
    log("  %s" % ("★ 全部像素级一致" if res['n_fail'] == 0 else "!!! 存在不一致，见 _reprodiff_* 文件"))

    json.dump(res, open(OUT / "v9_panel_pixel_repro.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    (OUT / "_96_v9_pixel_repro.md").write_text(
        "# 96 —— V9 图像素级可复现\n\n```\n" + "\n".join(LOG) + "\n```\n", encoding="utf-8")
    log("\n→ output/v9_panel_pixel_repro.json / _96_v9_pixel_repro.md")
    return 0 if res['n_fail'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
