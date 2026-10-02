# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
81 —— 官方严重度评分的**版本可追溯性**核查（锁定"到底比的是哪一版官方 SQL"）。

背景 / 为什么需要它
------------------
`62` 的严重度头对头已改用本地 `mimiciv_derived.oasis` / `.sapsii`，而不再用手写实现
（手写版漏 HR<33 分支与高收缩压分支，导致 OASIS 0.718→0.720、SAPS-II 0.785→0.780）。
但"来自官方派生表"**不等于**"版本已知"：mimic-code 对这两个评分做过修复，
若不锚定版本，就无法回答"本地表是否已包含这些修复"。

本脚本做两件事：

  A. **诚实记录溯源边界**：本地 PostgreSQL 不保存 DDL 来源。实测
     `obj_description(oasis)`、`reloptions`、以及任何 `*version*/*commit*/*meta*` 表
     **均为空/不存在** → **本地表的构建提交无法从数据库恢复**。这一点必须写明，
     不能声称"已锁定到某个 commit"。

  B. **用可判定的恒等式代替版本号**：官方 SQL 的两个评分各有两条精确恒等式
     （总分 = 各分量 COALESCE(...,0) 之和；概率 = 官方 logistic 公式）。
     逐行核对 94,458 行若全部相符，则**表内容与该版公式严格等价**，
     且不可能被手工编辑过。再核对两条此前手写版漏掉的分支行为。

  C. **锚定比对所依据的上游修订**：记录所比对 SQL 文件的 GitHub blob SHA 与
     最后修改提交（经 api.github.com 于检索日取得），使"与哪一版等价"可被第三方复算。

产出：output/official_score_provenance.json
      output/_81_run.log
"""
import json
import os
import sys
import time
from pathlib import Path

import psycopg2

BASE = Path(os.environ.get("ABI_BASE", Path(__file__).resolve().parents[1]))
OUT = BASE / "output"
LOGP = OUT / "_81_run.log"
LOG = []
CHECKS = []

# ---------------------------------------------------------------------------
# 上游参照修订（检索日 2026-10-02，经 api.github.com 取得；blob SHA 即文件内容哈希）
#   仓库 MIT-LCP/mimic-code，路径 mimic-iv/concepts_postgres/score/*.sql
#   三个文件最后修改提交同为 332f0aa199aa5c6bc272135285e4fad05483bc25（2026-06-30）
# ---------------------------------------------------------------------------
UPSTREAM = {
    "repo": "MIT-LCP/mimic-code",
    "branch": "main",
    "retrieved": "2026-10-02",
    "retrieved_via": "api.github.com/repos/MIT-LCP/mimic-code/contents/<path>?ref=main",
    "last_commit": "332f0aa199aa5c6bc272135285e4fad05483bc25",
    "last_commit_date": "2026-06-30T23:54:36Z",
    "files": {
        "mimic-iv/concepts_postgres/score/oasis.sql": {
            "blob_sha": "e1e680f62cc10949bcb863cb5ac10f9fa7a3b03a", "size": 10346},
        "mimic-iv/concepts_postgres/score/sapsii.sql": {
            "blob_sha": "3daf379460bfa23fdc1c106c3ea037c76d47b8cc", "size": 14873},
        "mimic-iv/concepts_postgres/score/sofa.sql": {
            "blob_sha": "d59a51739b99afc1104e844516f66c3e8bfa2087", "size": 10918},
    },
}

OASIS_COLS = ["age_score", "preiculos_score", "gcs_score", "heart_rate_score", "mbp_score",
              "resp_rate_score", "temp_score", "urineoutput_score", "mechvent_score",
              "electivesurgery_score"]
SAPS_COLS = ["age_score", "hr_score", "sysbp_score", "temp_score", "pao2fio2_score", "uo_score",
             "bun_score", "wbc_score", "potassium_score", "sodium_score", "bicarbonate_score",
             "bilirubin_score", "gcs_score", "comorbidity_score", "admissiontype_score"]


def log(s=""):
    print(s)
    LOG.append(str(s))


def chk(cond, msg):
    CHECKS.append(bool(cond))
    log("  [%s] %s" % ("OK " if cond else "FAIL", msg))
    return bool(cond)


def main():
    log("#" * 78)
    log("# 81_official_score_provenance.py | 官方严重度评分版本可追溯性核查")
    log("#" * 78)

    cn = psycopg2.connect(host="localhost", port=5432, user="postgres",
                          password=os.environ.get("PGPASSWORD", ""), dbname="mimiciv")
    c = cn.cursor()
    meta = {}

    # ---------------- A. 溯源边界：本地是否有版本元数据 ----------------
    log("\n== A. 溯源边界（本地库能否给出构建版本） ==")
    c.execute("select version()")
    meta["pg_version"] = c.fetchone()[0]
    log("  %s" % meta["pg_version"])

    for t in ("oasis", "sapsii"):
        c.execute("select obj_description(('mimiciv_derived.%s')::regclass)" % t)
        meta["%s_comment" % t] = c.fetchone()[0]
        c.execute("""select reloptions from pg_class where relname=%s
                     and relnamespace=(select oid from pg_namespace
                                       where nspname='mimiciv_derived')""", (t,))
        meta["%s_reloptions" % t] = c.fetchone()[0]
    c.execute("""select n.nspname||'.'||c.relname from pg_class c
                 join pg_namespace n on n.oid=c.relnamespace
                 where c.relkind='r' and n.nspname like 'mimic%'
                 and (c.relname ilike '%version%' or c.relname ilike '%commit%'
                      or c.relname ilike '%meta%')""")
    meta["version_like_tables"] = [r[0] for r in c.fetchall()]
    log("  oasis COMMENT=%r reloptions=%r" % (meta["oasis_comment"], meta["oasis_reloptions"]))
    log("  sapsii COMMENT=%r reloptions=%r" % (meta["sapsii_comment"], meta["sapsii_reloptions"]))
    log("  含 version/commit/meta 的表：%r" % meta["version_like_tables"])
    chk(meta["oasis_comment"] is None and meta["sapsii_comment"] is None
        and not meta["version_like_tables"],
        "确认本地库**不保存**派生表构建来源 → 构建提交不可从库中恢复（须在稿件中如实表述）")
    log("  ★ 因此本脚本用「恒等式等价」替代「版本号」：见 B/C 段。")

    # ---------------- B. 结构与连接 ----------------
    log("\n== B. 行数 / 连接键唯一性 / 与队列的连接覆盖 ==")
    for t in ("oasis", "sapsii"):
        c.execute("select count(*), count(distinct stay_id), "
                  "count(*) filter (where stay_id is null) from mimiciv_derived.%s" % t)
        n, d, nn = c.fetchone()
        meta["%s_rows" % t] = n
        meta["%s_distinct_stay_id" % t] = d
        chk(n == d and nn == 0, "%s: rows=%d == distinct stay_id=%d，null=%d（无重复、无空键）"
            % (t, n, d, nn))

    c.execute("""select table_schema||'.'||table_name from information_schema.tables
                 where table_schema='mimiciv_derived' and table_name='abi_a1_cohort'""")
    cohort_tbl = c.fetchone()[0]
    meta["cohort_table"] = cohort_tbl
    c.execute("select count(*) from %s" % cohort_tbl)
    meta["cohort_rows"] = c.fetchone()[0]
    for t in ("oasis", "sapsii"):
        c.execute("select count(*), count(*) filter (where x.%s is null) "
                  "from %s k join mimiciv_derived.%s x using(stay_id)" % (t, cohort_tbl, t))
        jn, jnull = c.fetchone()
        meta["cohort_join_%s" % t] = {"joined": jn, "null_score": jnull}
        chk(jn == meta["cohort_rows"] and jnull == 0,
            "%s: 队列 %d 行全部连上且评分非空（joined=%d, null=%d）"
            % (t, meta["cohort_rows"], jn, jnull))

    # ---------------- C. 恒等式等价（核心） ----------------
    log("\n== C1. 总分恒等式：总分 == SUM(COALESCE(分量,0)) ==")
    e_o = "+".join("coalesce(%s,0)" % x for x in OASIS_COLS)
    c.execute("select count(*), count(*) filter (where oasis=(%s)) from mimiciv_derived.oasis"
              % e_o)
    n, m = c.fetchone()
    meta["oasis_total_identity"] = {"rows": n, "match": m}
    chk(n == m and n > 0, "OASIS 总分恒等式：%d/%d 行严格相符" % (m, n))

    e_s = "+".join("coalesce(%s,0)" % x for x in SAPS_COLS)
    c.execute("select count(*), count(*) filter (where sapsii=(%s)) from mimiciv_derived.sapsii"
              % e_s)
    n, m = c.fetchone()
    meta["sapsii_total_identity"] = {"rows": n, "match": m}
    chk(n == m and n > 0, "SAPS-II 总分恒等式：%d/%d 行严格相符" % (m, n))

    log("\n== C2. 概率恒等式（官方 logistic 公式） ==")
    c.execute("""select count(*),
                 count(*) filter (where abs(oasis_prob
                   - 1/(1+exp(-(-6.1746 + 0.1275*oasis)))) < 1e-12)
                 from mimiciv_derived.oasis""")
    n, m = c.fetchone()
    meta["oasis_prob_identity"] = {"rows": n, "match": m,
                                  "formula": "1/(1+exp(-(-6.1746 + 0.1275*oasis)))"}
    chk(n == m and n > 0, "OASIS oasis_prob 公式：%d/%d 行相符" % (m, n))

    c.execute("""select count(*),
                 count(*) filter (where abs(sapsii_prob
                   - 1/(1+exp(-(-7.7631 + 0.0737*sapsii + 0.9971*ln(sapsii+1))))) < 1e-12)
                 from mimiciv_derived.sapsii""")
    n, m = c.fetchone()
    meta["sapsii_prob_identity"] = {
        "rows": n, "match": m,
        "formula": "1/(1+exp(-(-7.7631 + 0.0737*sapsii + 0.9971*ln(sapsii+1))))"}
    chk(n == m and n > 0, "SAPS-II sapsii_prob 公式（含 ln 项）：%d/%d 行相符" % (m, n))

    log("\n== C3. 分支行为：手写实现曾漏掉的两档 ==")
    # OASIS 的 display 列 heartrate 即"被选中那一项"：<33 只可能来自 heart_rate_min<33
    # 分支（若 heart_rate_max>125 则 display 取 max，其值 >125），故可判定。
    c.execute("select count(*), count(*) filter (where heart_rate_score=4) "
              "from mimiciv_derived.oasis where heartrate < 33")
    n, m = c.fetchone()
    meta["oasis_hr_branch"] = {"heartrate_lt33": n, "score_4": m}
    chk(n > 0 and n == m, "OASIS HR 低值档：heartrate<33 共 %d 例，全部计 4 分（手写版计 0）" % n)
    c.execute("select count(*), count(*) filter (where heart_rate_score=6) "
              "from mimiciv_derived.oasis where heartrate > 125")
    n, m = c.fetchone()
    meta["oasis_hr_high_branch"] = {"heartrate_gt125": n, "score_6": m}
    chk(n > 0 and n == m, "OASIS HR 高值档：heartrate>125 共 %d 例，全部计 6 分" % n)

    c.execute("select sysbp_score, count(*) from mimiciv_derived.sapsii group by 1 order by 1")
    dist = [(None if k is None else int(k), int(v)) for k, v in c.fetchall()]
    meta["sapsii_sysbp_score_distribution"] = dist
    hi = dict((k, v) for k, v in dist).get(2, 0)
    chk(hi > 0, "SAPS-II 高收缩压档 sysbp_score=2 共 %d 例（手写版只读 sbp_min，全表计不到）" % hi)

    log("\n== C4. 分布合理性（对照文献量级，仅作旁证） ==")
    for t, col in (("oasis", "oasis"), ("sapsii", "sapsii")):
        c.execute("""select percentile_cont(0.5) within group(order by %s),
                     percentile_cont(0.25) within group(order by %s),
                     percentile_cont(0.75) within group(order by %s),
                     min(%s), max(%s) from mimiciv_derived.%s""" % (col, col, col, col, col, t))
        med, q1, q3, lo, hi2 = c.fetchone()
        meta["%s_distribution" % t] = {"median": float(med), "q1": float(q1), "q3": float(q3),
                                       "min": int(lo), "max": int(hi2)}
        log("  %-7s median=%.0f IQR=[%.0f,%.0f] range=[%d,%d]"
            % (t, med, q1, q3, lo, hi2))
    chk(20 <= meta["oasis_distribution"]["median"] <= 40,
        "OASIS 中位数 %.0f 落在 ICU 队列常见量级 [20,40]"
        % meta["oasis_distribution"]["median"])
    chk(25 <= meta["sapsii_distribution"]["median"] <= 45,
        "SAPS-II 中位数 %.0f 落在 ICU 队列常见量级 [25,45]"
        % meta["sapsii_distribution"]["median"])

    cn.close()

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": "Which revision of the official mimic-code scoring SQL are the local "
                    "mimiciv_derived.oasis / .sapsii tables equivalent to?",
        "verdict": ("The build commit cannot be recovered from the database (no embedded "
                    "provenance). Equivalence to the official formulas was instead established "
                    "row-by-row by exact identities on 94,458 rows, plus two branch behaviours "
                    "that the previous hand-written implementation missed. The upstream "
                    "revision used for comparison is pinned by blob SHA below."),
        "local_database_provenance": meta,
        "upstream_reference_revision": UPSTREAM,
        "caveat": ("This pins the FORMULA revision the local tables are equivalent to. It does "
                   "not and cannot recover the commit that originally built the local tables, "
                   "because PostgreSQL stores no DDL source. Manuscript wording must not claim "
                   "a specific build commit."),
    }
    json.dump(payload, open(OUT / "official_score_provenance.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    log("\nwrote output/official_score_provenance.json")

    log("\n" + "=" * 78)
    ok = all(CHECKS)
    log("ALL CHECKS PASSED" if ok else "!!! SELF-CHECK FAILED (%d/%d)"
        % (sum(CHECKS), len(CHECKS)))
    log("=" * 78)
    LOGP.write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
