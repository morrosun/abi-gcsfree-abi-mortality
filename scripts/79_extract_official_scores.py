# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
79 —— 抽取 MIMIC-IV 官方严重度评分（OASIS / SAPS-II / SOFA）供头对头重算。

为什么写它
----------
62_calib_ci_and_severity.py 的 PART B 里，OASIS 与 SAPS-II 是**手写的** oasis() / saps2()：
  · OASIS 用首日**均值**（heart_rate_mean / mbp_mean / resp_rate_mean / temperature_mean），
    而 MIT-LCP 官方实现用首日极值；且 HR<33 手写版给 **0 分**，官方给 **4 分**。
  · SAPS-II 只用 sbp_min / wbc_max / potassium_max / sodium_min，漏掉高收缩压、
    低白细胞、低血钾、高血钠的分支。
本脚本改为直接读取 mimiciv_derived 的官方表（MIT-LCP mimic-code 实现），
它们是**实体表**（relkind='r'），对本队列 stay_id 覆盖率 100%、无重复。

产出：data/mimic_severity_official.csv（新文件，不覆盖 mimic_severity_inputs.csv）
      output/_79_extract_official_scores.log

★ 交叉校验：同时抽出 first_day_sofa，与队列 CSV 里已有的 sofa_day1 逐行比对。
  两者同源（sql/01_mimic_cohort.sql:72 就是 LEFT JOIN first_day_sofa），
  若不一致说明 stay_id 映射或抽取口径有问题 —— 这是**映射自洽性**的硬证据。
"""
import csv
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

BASE = Path(os.environ.get("ABI_BASE", Path(__file__).resolve().parents[1]))
DATA = BASE / "data"
OUT = BASE / "output"
LOGP = OUT / "_79_extract_official_scores.log"
LOGL = []


def log(s=""):
    print(s)
    LOGL.append(str(s))


def conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        dbname=os.environ.get("PGDATABASE", "mimiciv"),
        connect_timeout=30,
    )


CHECKS = []


def chk(cond, msg):
    CHECKS.append(bool(cond))
    log("  [%s] %s" % ("OK " if cond else "FAIL", msg))
    return bool(cond)


def main():
    log("#" * 78)
    log("# 79_extract_official_scores.py | 抽取官方 OASIS / SAPS-II / SOFA")
    log("#" * 78)

    coh = pd.read_csv(DATA / "mimic_abi_cohort.csv",
                      usecols=["stay_id", "sofa_day1", "death_365"])
    log("队列 n=%d | 唯一 stay_id=%d" % (len(coh), coh["stay_id"].nunique()))
    chk(len(coh) == coh["stay_id"].nunique(), "队列 stay_id 唯一（无重复入组）")

    c = conn()
    cur = c.cursor()
    cur.execute("create temp table _t (stay_id integer primary key);")
    cur.execute("insert into _t(stay_id) select unnest(%s::int[])",
                (coh["stay_id"].astype(int).tolist(),))

    frames = {}
    for tbl, col in (("oasis", "oasis"), ("sapsii", "sapsii"),
                     ("first_day_sofa", "sofa")):
        cur.execute("""
            select d.stay_id, d.%s
            from mimiciv_derived.%s d join _t t using (stay_id)
        """ % (col, tbl))
        rows = cur.fetchall()
        d = pd.DataFrame(rows, columns=["stay_id", col + "_official"])
        frames[col] = d
        log("  官方 %-6s 取回 %d 行" % (col, len(d)))
        chk(len(d) == len(coh), "%s 抽取行数 == 队列行数（%d）" % (col, len(d)))
        chk(d["stay_id"].nunique() == len(d), "%s 无重复 stay_id" % col)
    c.close()

    m = coh.merge(frames["oasis"], on="stay_id", how="left") \
           .merge(frames["sapsii"], on="stay_id", how="left") \
           .merge(frames["sofa"], on="stay_id", how="left")

    for col in ("oasis_official", "sapsii_official", "sofa_official"):
        nna = int(m[col].isna().sum())
        chk(nna == 0, "%s 无缺失（缺失 %d）" % (col, nna))

    # ★ 映射自洽性：官方 SOFA 必须复现队列里已有的 sofa_day1
    both = m.dropna(subset=["sofa_official", "sofa_day1"])
    eq = float((both["sofa_official"].values == both["sofa_day1"].values).mean())
    mad = float(np.abs(both["sofa_official"].values - both["sofa_day1"].values).max())
    log("\n  映射自洽性：官方 first_day_sofa vs 队列 sofa_day1")
    log("     完全一致比例 = %.4f | 最大绝对差 = %.1f" % (eq, mad))
    chk(eq >= 0.999, "官方 SOFA 与队列 sofa_day1 一致率 %.4f ≥ 0.999（stay_id 映射自洽）" % eq)

    # 分布合理性
    log("\n  官方评分分布：")
    for col, lo, hi in (("oasis_official", 0, 90), ("sapsii_official", 0, 120),
                        ("sofa_official", 0, 24)):
        v = m[col].values.astype(float)
        log("     %-16s median %5.1f  IQR [%d,%d]  range [%d,%d]"
            % (col, np.median(v), int(np.percentile(v, 25)), int(np.percentile(v, 75)),
               int(v.min()), int(v.max())))
        chk(v.min() >= lo and v.max() <= hi, "%s 取值落在 [%d,%d]" % (col, lo, hi))

    # ★ 与手写版的差异（留痕，说明为什么必须换）
    #   直接查官方表的构成字段：oasis.heartrate / sapsii.sysbp_score
    c2 = conn()
    cur2 = c2.cursor()
    cur2.execute("create temp table _t2 (stay_id integer primary key);")
    cur2.execute("insert into _t2(stay_id) select unnest(%s::int[])",
                 (coh["stay_id"].astype(int).tolist(),))
    cur2.execute("""
        select count(*) filter (where d.heartrate < 33),
               count(*) filter (where d.heart_rate_score = 4 and d.heartrate < 33)
        from mimiciv_derived.oasis d join _t2 t using (stay_id)
    """)
    n_hr33, n_hr33_sc4 = cur2.fetchone()
    cur2.execute("""
        select count(*) filter (where d.sysbp_score = 2)
        from mimiciv_derived.sapsii d join _t2 t using (stay_id)
    """)
    n_sys_hi = cur2.fetchone()[0]
    c2.close()
    log("\n  与手写版的已知差异（留痕，取自官方构成字段）：")
    log("     官方 oasis.heartrate < 33            = %d 例（手写版给 0 分，官方 heart_rate_score=4）"
        % n_hr33)
    log("     其中官方确实赋 4 分者               = %d 例" % n_hr33_sc4)
    log("     官方 sapsii.sysbp_score = 2（高收缩压）= %d 例（手写版只用 sbp_min，必然漏掉）"
        % n_sys_hi)
    chk(n_hr33 > 0 and n_hr33 == n_hr33_sc4,
        "HR<33 的 %d 例在官方表中一律赋 4 分（手写版赋 0 分 → 确为实质差异）" % n_hr33)
    chk(n_sys_hi > 0,
        "高收缩压分支涉及 %d 例（手写版只看 sbp_min 无法覆盖）" % n_sys_hi)

    out = DATA / "mimic_severity_official.csv"
    m[["stay_id", "oasis_official", "sapsii_official", "sofa_official"]].to_csv(
        out, index=False)
    log("\nwrote %s  (%d rows)" % (out, len(m)))

    log("\n" + "=" * 78)
    ok = all(CHECKS)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED")
    log("=" * 78)
    LOGP.write_text("\n".join(LOGL) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
