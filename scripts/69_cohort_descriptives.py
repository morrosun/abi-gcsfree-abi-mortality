# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
69_cohort_descriptives.py  ——  给 V8 Table 1 的队列描述量补"可溯源"出处

背景（2026-09-30 实证核查）
--------------------------
V7 稿件 Table 1 的 cohort 描述量（N / 年龄中位数 / 女性比例 / 事件数 / 事件率）
只存在于 HTML 正文字符串里，output/*.json 中没有任何出处 —— 与 66 脚本查出的
"+0.076 / −0.288"、68 脚本查出的"随机森林 0.832 / 梯度提升 0.841"属同一类问题：
**手敲、无结果文件可校验**。Table 1 是审稿人第一眼核对的表，必须先补齐。

本脚本做的事（两段式）
----------------------
  STAGE 1  直接读 5 个队列的原始分析文件，算：
             n、年龄中位数(IQR)、女性 %、结局事件数/率、机械通气 %、血管活性药 %、
             7 个病因指示器的可用个数（结构性可得性）
           落盘 output/cohort_descriptives.json
  STAGE 2  从该 JSON 读回，与 V7 Table 1 的手敲值逐条断言（年龄 ±1 岁、比例 ±0.5 pp、
           事件率 ±0.2 pp）；不一致即停机。

不覆盖：不改写任何图件与已交付投稿包。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(ABI_BASE)
OUT = BASE / "output"
DATA = BASE / "data"
HUAIAN = BASE / "ABI3" / "output" / "ABI本地数据采集模板_已录入检验结果.xlsx"

AET = ['tbi', 'sah', 'ich', 'ais', 'cns_inf', 'seizure', 'anoxic']
BINV = ['mech_vent', 'vasopressor', 'rrt']
OUTJSON = OUT / "cohort_descriptives.json"

# V7 Table 1 的手敲值：(n, age_median, female_pct, events, event_rate_pct)
V7_TABLE1 = {
    "MIMIC-IV": (16597, 67, 45.0, 5301, 31.9),
    "eICU": (14852, 63, 45.4, 1894, 12.8),
    "NWICU": (3420, 69, 46.5, 635, 18.6),
    "INSPIRE": (1543, 65, 44.4, 262, 17.0),
    "Huaian": (225, 61, 31.1, 55, 24.4),
}


def _fem(df):
    for c in ('female', 'gender', 'sex'):
        if c in df.columns:
            s = df[c]
            if s.dtype == object:
                return (s.astype(str).str.upper().str[:1] == 'F').astype(int)
            return (pd.to_numeric(s, errors='coerce').fillna(0) > 0.5).astype(int)
    raise KeyError("no sex column in %s" % list(df.columns)[:12])


def _desc(df, name, outcome_col, outcome_label, role):
    n = len(df)
    age = pd.to_numeric(df['age'], errors='coerce')
    fem = _fem(df)
    y = pd.to_numeric(df[outcome_col], errors='coerce').fillna(0).astype(int)
    rec = dict(
        database=name, role=role, outcome=outcome_label, outcome_column=outcome_col,
        n=int(n), events=int(y.sum()),
        event_rate_pct=round(float(y.mean() * 100), 1),
        age_median=float(age.median()),
        age_iqr=[float(age.quantile(0.25)), float(age.quantile(0.75))],
        female_pct=round(float(fem.mean() * 100), 1),
        mech_vent_pct=(round(float(pd.to_numeric(df['mech_vent'], errors='coerce')
                                   .fillna(0).mean() * 100), 1)
                       if 'mech_vent' in df.columns else None),
        vasopressor_pct=(round(float(pd.to_numeric(df['vasopressor'], errors='coerce')
                                     .fillna(0).mean() * 100), 1)
                         if 'vasopressor' in df.columns else None),
        rrt_pct=(round(float(pd.to_numeric(df['rrt'], errors='coerce')
                             .fillna(0).mean() * 100), 1)
                 if 'rrt' in df.columns else None),
        aetiology_available={a: int(pd.to_numeric(df[a], errors='coerce').fillna(0).sum())
                             for a in AET if a in df.columns},
        aetiology_structurally_empty=[a for a in AET
                                      if a in df.columns
                                      and pd.to_numeric(df[a], errors='coerce').fillna(0).sum() == 0],
        binary_available=[b for b in BINV if b in df.columns],
    )
    print("  %-9s n=%-6d events=%-5d (%.1f%%) age=%.0f (%.0f-%.0f) female=%.1f%% vaso=%s"
          % (name, rec['n'], rec['events'], rec['event_rate_pct'], rec['age_median'],
             rec['age_iqr'][0], rec['age_iqr'][1], rec['female_pct'], rec['vasopressor_pct']))
    return rec


def main():
    print("=== STAGE 1  从原始分析文件重算队列描述量 ===")
    mim = pd.read_csv(DATA / "mimic_abi_cohort.csv")
    mim['female'] = (mim['gender'] == 'F').astype(int)
    e = pd.read_csv(DATA / "eicu_abi_final.csv")
    nw = pd.read_csv(DATA / "nwicu_abi_final_vp2.csv")   # 74：恢复血管活性药
    ins = pd.read_csv(DATA / "inspire_abi_final.csv")
    dh = pd.read_csv(DATA / "inspire_deathhosp.csv")
    ins = ins.merge(dh, on='op_id', how='left')
    ins['death_hosp'] = ins['death_hosp'].fillna(0).astype(int)

    recs = [
        _desc(mim, "MIMIC-IV", "death_365", "1-year death", "derivation (70/30 split; 4,980 held out)"),
        _desc(e, "eICU", "death_hosp", "in-hospital death", "external (parallel in-hospital model)"),
        _desc(nw, "NWICU", "death_365", "1-year death", "external (1-year model)"),
        _desc(ins, "INSPIRE", "death_365", "1-year death", "external (1-year model)"),
    ]

    # 淮安单中心：从录入表重建（与 62/66 的口径完全一致）
    raw = pd.read_excel(HUAIAN, sheet_name='数据录入', header=0, skiprows=[1, 2, 3, 4, 5])
    raw = raw.dropna(how='all').reset_index(drop=True)
    num = lambda c: pd.to_numeric(raw[c], errors='coerce')      # noqa: E731
    d = pd.DataFrame(index=raw.index)
    d['age'] = num('age')
    d['female'] = num('female')
    d['y'] = (num('hospital_discharge_location') == 5).astype(int)
    for c in AET:
        d[c] = num(c).fillna(0)
    for c in BINV:
        d[c] = num(c).fillna(0) if c in raw.columns else np.nan
    keep = (d[AET].sum(axis=1) >= 1) & (d['age'] >= 18) & d['age'].notna() & \
        num('hospital_discharge_location').notna()
    dx = d[keep].copy()
    dx = dx.rename(columns={'y': 'death_hosp'})
    recs.append(_desc(dx, "Huaian", "death_hosp", "in-hospital death",
                      "external, independent healthcare system"))

    payload = dict(
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        sources=dict(
            MIMIC_IV="data/mimic_abi_cohort.csv",
            eICU="data/eicu_abi_final.csv",
            NWICU="data/nwicu_abi_final_vp2.csv",
            INSPIRE="data/inspire_abi_final.csv + data/inspire_deathhosp.csv",
            Huaian="ABI3/output/ABI本地数据采集模板_已录入检验结果.xlsx (sheet 数据录入)"),
        aetiology_indicators=AET,
        cohorts=recs,
    )
    json.dump(payload, open(OUTJSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("WROTE", OUTJSON)

    # ---------------- STAGE 2：读回 + 与 V7 Table 1 手敲值对账 ----------------
    print("\n=== STAGE 2  读回并与 V7 Table 1 手敲值对账 ===")
    back = {c['database']: c for c in json.load(open(OUTJSON, encoding="utf-8"))['cohorts']}
    bad = []
    for k, (n0, age0, fem0, ev0, rate0) in V7_TABLE1.items():
        c = back[k]
        checks = [("n", c['n'], n0, 0), ("age", c['age_median'], age0, 1.0),
                  ("female%", c['female_pct'], fem0, 0.5), ("events", c['events'], ev0, 0),
                  ("rate%", c['event_rate_pct'], rate0, 0.2)]
        for label, got, exp, tol in checks:
            ok = abs(got - exp) <= tol
            print("  [%s] %-9s %-8s got %-8s vs V7 %-8s" %
                  ("OK " if ok else "!! ", k, label, got, exp))
            if not ok:
                bad.append((k, label, got, exp, tol))
    assert not bad, "V7 Table 1 手敲值与重算不符：%s" % bad
    print("  [OK ] Table 1 全部描述量现已落到 %s（不再是手敲值）" % OUTJSON.name)
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
