# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
97_v9_vs_v8_diff.py —— V8 → V9 关键数字差异总表

目的：第 2 批改文字时，凡是"正文/表/图注里出现过的数字"都必须换成 V9 值。
本脚本把 V8 真源与 V9 真源逐项对齐列出，并标注是否变化，避免靠记忆改稿。

V8 真源：calibration_ci_and_severity.json（62）/ leakage_sensitivity.json（60）
         audit_decisionE.json（58）/ slope_epsilon_audit.json（63）
         v8_harmonisation.json（66）/ model_performance.csv（03）
V9 真源：calibration_ci_and_severity_v2.json（87）/ leakage_sensitivity_v9.json（91）
         downstream_severity_gcs_v9.json（90）/ slope_epsilon_audit_v9.json（93）
         v9_harmonisation.json（92）/ model_performance_v9.csv（94）

产出：output/_97_v9_vs_v8_diff.md + v9_number_master.json（V9 数字总表）
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(ABI_BASE) / "output"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def j(name):
    return json.load(open(OUT / name, encoding="utf-8"))


COHORTS = ['MIMIC_internal_1yr', 'NWICU_1yr', 'INSPIRE_1yr',
           'MIMIC_internal_inhosp', 'eICU_inhosp', 'Huaian_inhosp']


def main():
    v8 = j("calibration_ci_and_severity.json")['calibration_ci']
    v9 = j("calibration_ci_and_severity_v2.json")
    master = {'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'cohorts': {},
              'changes': []}

    def rec(section, item, a, b, fmt="%.4f"):
        changed = (a is None or b is None) or abs(float(a) - float(b)) > 5e-4
        if changed:
            master['changes'].append({'section': section, 'item': item,
                                      'v8': a, 'v9': b})
        return changed

    log("=" * 96)
    log("97 —— V8 → V9 关键数字差异总表")
    log("=" * 96)

    # ---------------- 1. 主结果 ----------------
    log("")
    log("【1】主结果：AUC / slope / O:E / ECE（括号内为 V8 值，变化以 * 标记）")
    log("%-22s %-9s %8s %8s %8s %8s" % ("cohort", "model", "AUC", "slope", "O:E", "ECE"))
    for c in COHORTS:
        master['cohorts'][c] = {}
        for m in ['Logistic', 'XGBoost']:
            if c not in v9 or m not in v9[c]:
                continue
            p = v9[c][m]['point']
            master['cohorts'][c][m] = {'n': v9[c]['n'], 'events': v9[c]['events'],
                                       'point': p, 'ci95': v9[c][m]['ci95']}
            old = None
            if c in v8 and m in v8[c]:
                old = v8[c][m]['point']
            def f(k, nd=4):
                return ("%.*f" % (nd, p[k])) + \
                       ("*" if (old and abs(old[k] - p[k]) > 5e-4) else "")
            def g(k, nd=4):
                return ("%.*f" % (nd, old[k])) if old else "—"
            log("%-22s %-9s %8s %8s %8s %8s   | V8: %s %s %s %s"
                % (c, m, f('auc'), f('slope', 3), f('oe', 3), f('ece'),
                   g('auc'), g('slope', 3), g('oe', 3), g('ece')))
            for k in ['auc', 'slope', 'intercept', 'oe', 'ece', 'brier']:
                rec('main/%s/%s' % (c, m), k, old[k] if old else None, p[k])
        log("     n=%d events=%d" % (v9[c]['n'], v9[c]['events']))

    # ---------------- 2. 五个候选学习器 ----------------
    log("")
    log("【2】内部测试集：五个候选学习器（Table 2）")
    try:
        p8 = pd.read_csv(OUT / "model_performance.csv", index_col=0)
        p9 = pd.read_csv(OUT / "model_performance_v9.csv", index_col=0)
        master['learners_v9'] = p9['AUC (95% CI)'].to_dict()
        master['learners_v8'] = p8['AUC (95% CI)'].to_dict()
        for m in p9.index:
            a = p9.loc[m, 'AUC (95% CI)']
            b = p8.loc[m, 'AUC (95% CI)'] if m in p8.index else '—'
            log("   %-14s V9 %s   | V8 %s %s" % (m, a, b, "*" if a != b else ""))
    except Exception as e:                                    # noqa: BLE001
        log("   (跳过：%s)" % e)

    # ---------------- 3. 严重度评分头对头 ----------------
    log("")
    log("【3】严重度评分头对头（MIMIC-IV 内部 30% 测试集，1 年死亡）")
    s8 = j("calibration_ci_and_severity.json")['severity_headtohead']
    s9 = j("downstream_severity_gcs_v9.json")['severity_headtohead']
    master['severity_v9'] = s9
    for k in ['sofa', 'oasis', 'saps2']:
        a9 = s9['scores'][k]['auc_alone']
        a8 = s8['scores'][k].get('auc_alone') if 'scores' in s8 else None
        log("   %-6s AUC-alone  V9 %.4f  | V8 %s" %
            (k, a9, ("%.4f" % a8) if a8 else "—"))
        for m in ['Logistic', 'XGBoost']:
            c9 = s9['comparison'][k][m]
            log("      vs %-8s Δ%s (p=%.3g)" % (m, "%+.4f" % c9['dAUC'], c9['p']))

    # ---------------- 4. GCS 增量 ----------------
    log("")
    log("【4】GCS 增量（+总分 / +运动项）")
    g8 = j("audit_decisionE.json")['gcs_increment_decisionE']
    g9 = j("downstream_severity_gcs_v9.json")['gcs_increment']
    master['gcs_v9'] = g9
    for m in ['Logistic', 'XGBoost']:
        for tag in ['plus_total_gcs', 'plus_motor_gcs']:
            a9 = g9[m][tag]['dAUC']
            a8 = g8[m][tag]['dAUC'] if m in g8 and tag in g8[m] else None
            log("   %-8s %-16s ΔAUC V9 %+.4f  | V8 %s" %
                (m, tag, a9, ("%+.4f" % a8) if a8 is not None else "—"))
    log("   运动项 / 总分 倍数：LR %.2f× | XGB %.2f×" %
        (g9['Logistic']['motor_over_total_ratio'],
         g9['XGBoost']['motor_over_total_ratio']))
    log("   单变量判别力：总分 %.4f | 运动项 %.4f" %
        (g9['alone_AUC_logit']['gcs_total'], g9['alone_AUC_logit']['gcs_motor']))
    st = g9.get('gcs_alone_by_ventilation', {})
    if st:
        log("   按通气分层（★ 重算）：通气 n=%d %.4f | 未通气 n=%d %.4f" %
            (st['ventilated']['n'], st['ventilated']['auc'],
             st['nonventilated']['n'], st['nonventilated']['auc']))
        log("   V8 旧值（revision2.json，mech_vent 全程口径）：通气 0.488 | 未通气 0.653")

    # ---------------- 5. 泄漏敏感性 ----------------
    log("")
    log("【5】对出院编码预测变量的依赖度（ΔAUC：去掉 7 个病因哑变量）")
    l8 = j("leakage_sensitivity.json")['summary']
    l9 = j("leakage_sensitivity_v9.json")['summary']
    master['leakage_v9'] = {'summary': l9,
                            'external_delta': j("leakage_sensitivity_v9.json")['external_delta']}
    for sc in ['A_1yr_full', 'B_inhosp', 'C_survivors']:
        for m in ['Logistic', 'XGBoost']:
            a9 = l9[sc][m]['delta_no_aet']
            a8 = l8.get(sc, {}).get(m, {}).get('delta_no_aet')
            log("   %-12s %-8s V9 %+.4f  | V8 %s" %
                (sc, m, a9, ("%+.4f" % a8) if a8 is not None else "—"))
    log("")
    log("   外部库：")
    ed9 = j("leakage_sensitivity_v9.json")['external_delta']
    ed8 = j("leakage_sensitivity.json").get('external', {})
    for k, r in ed9.items():
        a8 = ed8.get(k, {}).get('Logistic', {}).get('delta')
        log("   %-24s LR %+.4f  XGB %+.4f  | V8 LR %s" %
            (k, r['Logistic']['delta'], r['XGBoost']['delta'],
             ("%+.4f" % a8) if a8 is not None else "—"))

    # ---------------- 6. ε 敏感性 ----------------
    log("")
    log("【6】校准斜率对裁剪常数 ε 的敏感性")
    e8 = j("slope_epsilon_audit.json")['results']
    e9 = j("slope_epsilon_audit_v9.json")['results']
    master['slope_eps_v9'] = e9
    for t in e9:
        for m in ['Logistic', 'XGBoost']:
            r9 = e9[t][m]
            r8 = e8.get(t, {}).get(m, {})
            log("   %-20s %-8s V9 range %.4f (%s) η法 %.4f  | V8 range %s" %
                (t, m, r9['slope_range_over_eps'], r9['grade'], r9['slope_eta'],
                 ("%.4f" % r8.get('slope_range', -1)) if r8 else "—"))

    # ---------------- 7. 淮安 harmonisation ----------------
    log("")
    log("【7】淮安单位 / 标度扰动（n=225）")
    h8 = j("v8_harmonisation.json")
    h9 = j("v9_harmonisation.json")
    master['harmonisation_v9'] = h9
    for t in ['harmonised', 'unit_discordant', 'scale_discordant']:
        for m in ['Logistic', 'XGBoost']:
            v = h9['variants'][t][m]
            o = h8['variants'].get(t, {}).get(m, {})
            log("   %-18s %-8s AUC %.4f O:E %.3f slope %.3f  | V8 AUC %s O:E %s" %
                (t, m, v['auc'], v['oe'], v['slope'],
                 ("%.4f" % o['auc']) if o else "—", ("%.3f" % o['oe']) if o else "—"))
    for t in ['unit_discordant', 'scale_discordant']:
        for m in ['Logistic', 'XGBoost']:
            d9 = h9['delta'][t][m]['dAUC']
            d8 = h8['delta'].get(t, {}).get(m, {}).get('dAUC')
            log("   Δ %-16s %-8s V9 %+.4f  | V8 %s" %
                (t, m, d9, ("%+.4f" % d8) if d8 is not None else "—"))

    json.dump(master, open(OUT / "v9_number_master.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    (OUT / "_97_v9_vs_v8_diff.md").write_text(
        "# 97 —— V8 → V9 关键数字差异总表\n\n```\n" + "\n".join(LOG) + "\n```\n",
        encoding="utf-8")
    log("")
    log("  变更条目合计 %d 项" % len(master['changes']))
    log("\n→ output/_97_v9_vs_v8_diff.md / v9_number_master.json")


if __name__ == "__main__":
    main()
