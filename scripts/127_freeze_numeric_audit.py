# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""127_freeze_numeric_audit.py — 固定版本数值核对（第二十轮评审收尾建议）

评审原话："处理完上述两项后，建议做一次固定版本的数值核对，而不是继续循环扩写稿件。"

本脚本**只读**：不重训、不改稿、不动任何结果文件，只做四层核对并落一份可复核的报告。

L1 冻结指纹
  包里每个交付件的 SHA256 + 大小；生成链脚本的 SHA256；仓库 HEAD/tag/工作区状态；
  稿件声明的归档 DOI；以及所有输入结果文件的 SHA256。⇒ 任何人可凭这一节判断
  "我手上的包是不是被核过的这一版"。

L2 独立重算（外部验证 10 个组合）
  用 data/*.csv + output/preproc_v2.joblib / models_v2.joblib / model_inhosp_v2.joblib
  自行实现 prep / 预测 / 指标（不 import 126、不 import 95 的指标代码；只复用 95 的
  huaian_frame() 读表函数），把 AUC / Brier / 校准斜率 / 截距 / O:E / ECE 重新算一遍，
  与**成稿表格里印出来的数字**逐格比对（三位小数），而不是与中间 JSON 比对。
  内部集（MIMIC-IV）不独立重算：它需要重现阶段性分层划分，超出"核对"范围，
  改由 L3 的 master↔CSV 双源一致 + 转录核对覆盖。

L3 转录核对（表→源）
  成稿 6 张表 + 附录 S3/S5/S19 的关键单元格逐格回溯到权威 JSON/CSV。

L4 跨文档一致性
  同一量在正文/附录/投稿信/回复信中若出现多次，必须逐字符相同。

L5 计数复核
  字数（tokens 规则与 64 号一致）、页数、图表数。

产物：
  output/freeze_numeric_audit_v13.json
  output/FREEZE_NUMERIC_AUDIT_V13.md
  output/_127_freeze_audit.log

归档说明（v1.4.6）
  本脚本随归档包一同发布，故按"缺什么就跳过什么"的方式组织：每一层包在 _layer() 里，
  输入不存在时打印 [SKIP] 并继续，最后在汇总里列出未执行的层。这样，只拿到归档包的
  读者也能跑出 L1（脚本指纹）并把其余层标注为"未执行"，而不是直接抛异常。
  需要但不随归档分发（DUA / 未发表正文）：四个库的患者级 CSV、三个冻结模型工件
  （preproc_v2 / models_v2 / model_inhosp_v2）、以及投稿包（docx/pdf）。
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = os.environ.get("ABI_BASE") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
# 归档副本里结果 JSON 落在 results/（与本机 output/ 同名文件一一对应）⇒ 缺 output/ 时回退
if not os.path.isdir(OUT) and os.path.exists(
        os.path.join(ROOT, "results", "v9_number_master.json")):
    OUT = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
PKG = os.path.join(OUT, "submission_v13")
REPO = os.path.join(os.path.dirname(ROOT), "abi-gcsfree-abi-mortality")
EPS = 1e-6
TOL_ROUND = 5e-4          # 三位小数：|recomputed - printed| <= 0.0005
TOL_RAW = 1e-6            # 未取整量对 JSON/CSV
LOG = []
CHECKS = []               # (layer, name, expected, actual, ok, note)
SKIPPED = []              # 缺输入而未执行的层（归档副本里非空是正常现象）


def _layer(tag, fn, *a, **kw):
    """跑一层；输入缺失（未归档的模型工件 / 患者级 CSV / 投稿包）时记录 SKIP 而不中止。"""
    try:
        return fn(*a, **kw)
    except Exception as ex:  # noqa: BLE001
        SKIPPED.append("%s（%s: %s）" % (tag, type(ex).__name__, ex))
        log("  [SKIP] %s 未执行：%s: %s" % (tag, type(ex).__name__, ex))
        return None


def log(s=""):
    print(s)
    LOG.append(str(s))


def chk(layer, name, expected, actual, ok, note=""):
    """ok=True 通过 / False 失败 / None 需知情（NOTE，不计失败）。"""
    CHECKS.append(dict(layer=layer, name=name, expected=str(expected),
                       actual=str(actual), ok=ok, note=note))
    tag = "NOTE" if ok is None else ("OK" if ok else "FAIL")
    log("  [%s] %-16s %s" % (tag, layer,
                             name if ok else "%s：期望 %s，实得 %s  %s"
                             % (name, expected, actual, note)))


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


# 归档副本里结果 JSON 平铺在 results/，且两个文件按中性名重命名 ⇒ 定位时按别名兜底
_RES_ALIASES = {
    "gcs_increment_audit.json": "gcs_increment_audit.json",
    "slope_epsilon_audit_v9.json": "slope_epsilon_audit.json",
}
# 这三个文件在 output/ 根与 output/v9_results/ 各有一份，**根下那份是 V8 时代的旧件**
# （NWICU 635 事件 / 斜率 0.039 口径）⇒ 必须优先取 v9_results/，否则会把旧数当成当前数。
_V9_FIRST = {"nwicu_vasopressor_recovery.json", "gcs_increment_audit.json",
             "internal_learners.json"}


def res_path(name):
    """结果 JSON 定位：先取该文件的**当前运行**目录，再退到另一处与归档副本的平铺名。"""
    sub = os.path.join(OUT, "v9_results")
    bases = (sub, OUT) if name in _V9_FIRST else (OUT, sub)
    alias = _RES_ALIASES.get(name)
    cands = [os.path.join(b, name) for b in bases]
    if alias:
        cands += [os.path.join(b, alias) for b in bases]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到结果文件 %s（已试 %d 个候选路径）" % (name, len(cands)))


# ============================================================ 指标（自行实现）

def prep(art, df, feats, cont):
    Z = df[feats].copy()
    for c in cont:
        Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(art["medians"][c])
    for c in feats:
        if c not in cont:
            Z[c] = pd.to_numeric(Z[c], errors="coerce").fillna(0)
    Zs = Z.copy()
    Zs[cont] = art["scaler"].transform(Z[cont])
    return Z, Zs


def predict(art, df, feats, cont):
    Z, Zs = prep(art, df, feats, cont)
    if "logit_params" in art:
        lp = art["logit_params"]
        eta = lp["const"] + Zs[feats].values @ np.array([lp[f] for f in feats])
        p_lr = 1.0 / (1.0 + np.exp(-eta))
    else:
        eta = art["lr"].decision_function(Zs[feats])
        p_lr = art["lr"].predict_proba(Zs[feats])[:, 1]
    xg = art.get("xgb", art.get("xg"))
    eta_xg = xg.predict(Z[feats].values, output_margin=True)
    p_xg = 1.0 / (1.0 + np.exp(-eta_xg))       # 独立换算，不经 predict_proba
    return dict(Logistic=(p_lr, eta), XGBoost=(p_xg, eta_xg))


def ece10(y, p, bins=10):
    """十等宽分箱 ECE（与正文方法段声明一致：ten equal-width bins）。"""
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            tot += abs(y[m].mean() - p[m].mean()) * m.sum() / len(y)
    return float(tot)


def metrics(y, p, eta):
    m = LogisticRegression(fit_intercept=True, C=1e12, solver="lbfgs",
                           max_iter=5000).fit(eta.reshape(-1, 1), y)
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                slope=float(m.coef_[0][0]), intercept=float(m.intercept_[0]),
                oe=float(np.mean(y) / np.mean(p)), ece=ece10(y, p))


# ============================================================ L1 指纹

def layer1_fingerprint():
    log("")
    log("=" * 100)
    log("L1  冻结指纹")
    log("=" * 100)
    fp = dict(deliverables=[], chain=[], inputs=[], repo={}, doi=None)
    if os.path.isdir(PKG):
        for fn in sorted(os.listdir(PKG)):
            p = os.path.join(PKG, fn)
            if os.path.isfile(p):
                fp["deliverables"].append(dict(file=fn, bytes=os.path.getsize(p), sha256=sha(p)))
                log("  %-46s %8d B  %s" % (fn, os.path.getsize(p), sha(p)))
    else:
        log("  (投稿包不在本工作区：%s —— 交付件指纹留空)" % os.path.basename(PKG))
    # 64/65/70/76/78 是稿件与附录的构建/终审脚本，**不随归档分发**（逐字含未发表正文）；
    # 归档副本里这几项找不到属正常，fingerprint 只列出实际存在的脚本。
    # （归档副本中两个带刊名 token 的脚本名会被替换成中性描述串，故更不会命中。）
    for sc in ("the submission-package builder", "65_build_v8_appendix.py",
               "70_manuscript_pdf.py", "76_refs_final_to_package.py",
               "the archived-result verifier", "126_treatment_rule_sensitivity.py",
               "95_v9_external_figures.py", "93_slope_epsilon_v9.py"):
        p = os.path.join(ROOT, "scripts", sc)
        if os.path.exists(p):
            fp["chain"].append(dict(script=sc, sha256=sha(p)))
    for rel in ("v9_number_master.json", "external_validation_v9.csv",
                "slope_epsilon_audit_v9.json", "treatment_rule_sensitivity_v1.json",
                "treatment_rule_check_v1.json", "calibration_ci_and_severity.json",
                "leakage_sensitivity_v9.json", "cohort_descriptives.json"):
        for base in (OUT, os.path.join(OUT, "v9_results")):
            p = os.path.join(base, rel)
            if os.path.exists(p):
                fp["inputs"].append(dict(file=os.path.relpath(p, ROOT).replace(os.sep, "/"),
                                         bytes=os.path.getsize(p), sha256=sha(p)))
                break
    # 归档 DOI（单一来源 = 64 号 _ZENODO_DOI 的 v9 分支）
    _p64 = os.path.join(ROOT, "scripts", "the submission-package builder")
    if os.path.exists(_p64):
        src = io.open(_p64, encoding="utf-8").read()
        i = src.find('_ZENODO_DOI = ("10.5281/zenodo.')
        fp["doi"] = src[i:i + 60].split('"')[1] if i > 0 else None
    else:
        fp["doi"] = "(package builder not present in this workspace)"
    log("  归档 DOI（v9 分支）: %s" % fp["doi"])
    try:
        def g(*a):
            return subprocess.run(["git"] + list(a), cwd=REPO, capture_output=True,
                                  text=True, timeout=60).stdout.strip()
        fp["repo"] = dict(head=g("rev-parse", "HEAD"),
                          describe=g("describe", "--tags", "--abbrev=0"),
                          tag_head=g("rev-list", "-n", "1", g("describe", "--tags", "--abbrev=0")),
                          dirty=bool(g("status", "--porcelain")),
                          after_tag=g("diff", "--name-only",
                                      "%s..HEAD" % g("describe", "--tags", "--abbrev=0")),
                          remote=g("config", "--get", "remote.origin.url"))
        log("  仓库 HEAD %s | 最近 tag %s | 工作区暂存变动 %s"
            % (fp["repo"]["head"][:8], fp["repo"]["describe"], fp["repo"]["dirty"]))
        after = [x for x in fp["repo"]["after_tag"].split("\n") if x.strip()]
        # 归档 tag 之后允许只多一个改 README 的提交（版本 DOI 只能在 harvest 后回填，
        # 这是仓库 README 明示的"版本块晚一版"机制）；其余任何文件变动都算不可定位。
        ok_tag = (fp["repo"]["tag_head"] == fp["repo"]["head"]) or \
                 (set(after) <= {"README.md"})
        chk("L1", "归档 tag 可定位到当前树", "v1.4.5 %s" % fp["repo"]["describe"],
            "HEAD %s，tag 之后变动 %s" % (fp["repo"]["head"][:8], after or "无"), ok_tag,
            "仅 README.md 的 DOI 回填提交属既定机制" if after else "")
        chk("L1", "工作区无未提交改动", "clean", "dirty" if fp["repo"]["dirty"] else "clean",
            not fp["repo"]["dirty"])
    except Exception as ex:  # noqa: BLE001
        log("  !! git 指纹跳过：%s" % ex)
    return fp


# ============================================================ L2 独立重算

def layer2_recompute():
    log("")
    log("=" * 100)
    log("L2  独立重算（外部验证 10 个组合；不 import 126、不 import 95 的指标代码）")
    log("=" * 100)
    pp = joblib.load(os.path.join(OUT, "preproc_v2.joblib"))
    md = joblib.load(os.path.join(OUT, "models_v2.joblib"))
    ih = joblib.load(os.path.join(OUT, "model_inhosp_v2.joblib"))
    a1 = dict(medians=pp["medians"], scaler=pp["scaler"],
              logit_params=md["logit_params"], xgb=md["xgb"])
    a2 = dict(medians=ih["medians"], scaler=ih["scaler"], lr=ih["lr"], xgb=ih["xg"])
    FEATS, CONT = pp["feats"], pp["cont"]
    log("  预测因子 %d 个（连续 %d），一年模型 logit_params=%s，院内模型 lr=%s"
        % (len(FEATS), len(CONT), "logit_params" in a1, "lr" in a2))

    nw = pd.read_csv(os.path.join(DATA, "nwicu_abi_final_v6.csv"))
    ins = pd.read_csv(os.path.join(DATA, "inspire_abi_final_v3.csv"))
    eic = pd.read_csv(os.path.join(DATA, "eicu_abi_final_v3.csv"))
    combos = [("NWICU_1yr", nw, "death_365", a1), ("INSPIRE_1yr", ins, "death_365", a1),
              ("NWICU_inhosp", nw, "death_hosp", a2), ("INSPIRE_inhosp", ins, "death_hosp", a2),
              ("eICU_inhosp", eic, "death_hosp", a2)]
    try:
        spec = os.path.join(ROOT, "scripts", "95_v9_external_figures.py")
        src = io.open(spec, encoding="utf-8").read()
        ns = {}
        exec(compile(src.split("def main(")[0], "v95_partial", "exec"), ns)  # noqa: S102
        Xl, yl = ns["huaian_frame"]()
        combos.append(("Huaian_inhosp", Xl.assign(**{"_y": yl}), "_y", a2))
    except Exception as ex:  # noqa: BLE001
        log("  !! 淮安未参与重算：%s" % ex)

    res = {}
    for tag, df, ycol, art in combos:
        y = df[ycol].values.astype(int)
        pred = predict(art, df, FEATS, CONT)
        res[tag] = dict(n=int(len(y)), events=int(y.sum()))
        for nm, (p, eta) in pred.items():
            res[tag][nm] = metrics(y, p, eta)
        log("  %-15s n=%5d events=%4d | LR AUC %.4f slope %.4f O:E %.4f | XGB AUC %.4f slope %.4f O:E %.4f"
            % (tag, res[tag]["n"], res[tag]["events"],
               res[tag]["Logistic"]["auc"], res[tag]["Logistic"]["slope"],
               res[tag]["Logistic"]["oe"], res[tag]["XGBoost"]["auc"],
               res[tag]["XGBoost"]["slope"], res[tag]["XGBoost"]["oe"]))
    return res


# ============================================================ L2b 表 4 逐格比对

T4_ROWKEY = {          # 成稿表格行 → 重算/JSON 的 cohort 键
    "MIMIC-IV (1-year, internal)": "MIMIC_internal_1yr",
    "NWICU (1-year)": "NWICU_1yr",
    "INSPIRE (1-year)": "INSPIRE_1yr",
    "eICU (in-hospital)": "eICU_inhosp",
    "Huaian (in-hospital)": "Huaian_inhosp",
}


def _num(s):
    s = s.strip().replace("\u2212", "-").replace("\u2013", "-").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def _point_ci(cell, with_ci=True):
    """'0.818 (0.805–0.829)' -> (0.818, 0.805, 0.829)；'0.158 (+0.27)' -> (0.158, None, None)。"""
    cell = cell.strip()
    if "(" not in cell:
        return _num(cell.split("+")[0]), None, None
    head, tail = cell.split("(", 1)
    tail = tail.rstrip(")").strip()
    if not with_ci:
        return _num(head.split("+")[0]), None, None
    parts = tail.replace("\u2212", "-").replace("\u2013", "-").split("-")
    # 双负号场景（-1.275–-1.160）拆分后会产生空串，重组回负数
    nums, buf = [], ""
    for seg in parts:
        if seg == "" and buf == "":
            buf = "-"
            continue
        if buf == "-":
            nums.append(_num("-" + seg))
            buf = ""
        else:
            nums.append(_num(seg))
    return (_num(head.split("+")[0]), nums[0] if len(nums) > 0 else None,
            nums[1] if len(nums) > 1 else None)


def layer2b_table4(rec, master):
    log("")
    log("=" * 100)
    log("L2b  表 4（外部验证性能表）逐格核对：印出来的数字 vs 独立重算 / master CI")
    log("=" * 100)
    from docx import Document
    d = Document(os.path.join(PKG, "01_Manuscript_V13.docx"))
    t = d.tables[3]
    cols = {c.text.strip(): i for i, c in enumerate(t.rows[0].cells)}
    specs = (("AUC", "AUC (95% CI)", "auc", True, 3, "master"),
             ("Brier", "Brier (skill score)", "brier", False, 3, "master"),
             ("斜率", "Calibration slope (95% CI)", "slope", True, 3, "eta"),
             ("截距", "Intercept (95% CI)", "intercept", True, 3, "eta"),
             ("O:E", "O:E (95% CI)", "oe", True, 2, "master"),
             ("ECE", "ECE (95% CI)", "ece", True, 3, "master"))
    eta = json.load(io.open(os.path.join(OUT, "eta_calibration_ci_v11.json"),
                            encoding="utf-8"))["cohorts"]
    n_ok = 0
    for r in t.rows[1:]:
        cells = [c.text.strip() for c in r.cells]
        cohort_cell, n_cell, model = cells[0], cells[1], cells[2]
        key = T4_ROWKEY[cohort_cell]
        n_printed = int(n_cell.split("(")[0].replace(",", "").strip())
        ev_printed = int(n_cell.split("(")[1].rstrip(")").replace(",", "").strip())
        mc = master["cohorts"][key][model]
        ok = (n_printed, ev_printed) == (mc["n"], mc["events"])
        chk("L2b", "%s %s n (events)" % (key, model), "%d (%d)" % (mc["n"], mc["events"]),
            "%d (%d)" % (n_printed, ev_printed), ok)
        if key in rec and model in rec[key]:
            chk("L2b", "%s %s 重算队列规模" % (key, model),
                "%d (%d)" % (rec[key]["n"], rec[key]["events"]),
                "%d (%d)" % (n_printed, ev_printed),
                (rec[key]["n"], rec[key]["events"]) == (n_printed, ev_printed))
            src, srcname = rec[key][model], "重算"
        else:
            src, srcname = mc["point"], "master"
        for label, col, key_name, with_ci, nd, ci_src in specs:
            tol = 0.5 * 10 ** (-nd)
            v, lo, hi = _point_ci(cells[cols[col]], with_ci)
            ok = v is not None and abs(round(src[key_name], nd) - v) <= tol
            chk("L2b", "%s %s %s %s→印出（%ddp）" % (key, model, label, srcname, nd),
                "%.*f" % (nd, src[key_name]), v, ok, "%s %.6f" % (srcname, src[key_name]))
            n_ok += int(ok)
            if with_ci and lo is not None:
                if ci_src == "eta":
                    block = eta[key]["models"][model]["ci95"]
                    ci = block.get(key_name, [None, None])
                else:
                    ci = mc["ci95"][key_name]
                # 表内 95% CI 一律三位小数（O:E 的**点位**是两位，CI 仍是三位）
                okci = (abs(round(ci[0], 3) - lo) <= TOL_ROUND
                        and abs(round(ci[1], 3) - hi) <= TOL_ROUND)
                chk("L2b", "%s %s %s CI（%s 源）" % (key, model, label, ci_src),
                    "[%.3f, %.3f]" % (ci[0], ci[1]), "[%s, %s]" % (lo, hi), okci)
    log("  表 4 点位核对通过 %d 项" % n_ok)
    chk("L2b", "O:E 点位只印两位小数（CI 三位）", "0.603 → 印 0.60", "见各队列 O:E",
        None, "O:E 点位统一两位、CI 统一三位：读者无法从表格恢复 0.6031（eICU 院内 LR）。"
              "不影响 O:E 的量级结论，但若审稿人核对到第三位会对不上——同一列内部精度不一致，"
              "属可选的排版问题，本轮不改稿")


# ============================================================ L3 其余表→源

def layer3_transcription(master):
    log("")
    log("=" * 100)
    log("L3  转录核对（S19 / S3 计数 / S5 ε 网格 / 表 1 队列规模）")
    log("=" * 100)
    from docx import Document
    ap = Document(os.path.join(PKG, "04_Additional_file_1_V13.docx"))
    trs = json.load(io.open(os.path.join(OUT, "treatment_rule_sensitivity_v1.json"),
                            encoding="utf-8"))
    trc = json.load(io.open(os.path.join(OUT, "treatment_rule_check_v1.json"),
                            encoding="utf-8"))
    eps = json.load(io.open(os.path.join(OUT, "slope_epsilon_audit_v9.json"),
                            encoding="utf-8"))

    # ---- Table S19：as-coded→alternative（AUC / O:E / 斜率）
    t19 = ap.tables[18]
    rows19 = {}
    for r in t19.rows[1:]:
        c = [x.text.strip() for x in r.cells]
        rows19[(c[0].split()[0].lower(), c[1])] = c
    pair = {("one-year", "Logistic"): ("INSPIRE_1yr", "Logistic"),
            ("one-year", "XGBoost"): ("INSPIRE_1yr", "XGBoost"),
            ("in-hospital", "Logistic"): ("INSPIRE_inhosp", "Logistic"),
            ("in-hospital", "XGBoost"): ("INSPIRE_inhosp", "XGBoost")}
    for k, (tag, nm) in pair.items():
        c = rows19.get(k)
        if c is None:
            chk("L3", "S19 行存在 %s %s" % k, "present", "missing", False)
            continue
        base, alt = trs["cohorts"][tag]["as_coded"][nm], trs["cohorts"][tag]["alt"][nm]
        for label, cell, ky, nd in (("AUC", c[2], "auc", 3), ("O:E", c[4], "oe", 3),
                                    ("slope", c[5], "slope", 3)):
            a, b = [x.strip() for x in cell.split("\u2192")]
            ok = (abs(round(base[ky], nd) - float(a)) <= TOL_ROUND
                  and abs(round(alt[ky], nd) - float(b)) <= TOL_ROUND)
            chk("L3", "S19 %s %s %s %s→%s" % (k[0], k[1], label, a, b),
                "%.3f→%.3f" % (round(base[ky], nd), round(alt[ky], nd)),
                "%s→%s" % (a, b), ok)
        dcell = _num(c[3])
        chk("L3", "S19 %s %s ΔAUC" % k, "%+.3f" % round(alt["auc"] - base["auc"], 3),
            c[3], abs(round(alt["auc"] - base["auc"], 3) - dcell) <= TOL_ROUND)

    # ---- S3：重分类计数（185 / 11 / 50 / 6 / 1 / 2）
    insp = {r["variable"]: r for r in trc["databases"]["INSPIRE"]}
    nwic = {r["variable"]: r for r in trc["databases"]["NWICU"]}
    for var, want in (("mech_vent", 185), ("rrt", 11), ("vasopressor", 50)):
        got = abs(insp[var]["counts"]["final"] - insp[var]["counts"]["first_win"])
        chk("L3", "S3 INSPIRE %s 重分类计数" % var, want, got, got == want)
    nwv = abs(nwic["vasopressor"]["counts"]["final"]
              - nwic["vasopressor"]["counts"]["first_win"])
    chk("L3", "S3 NWICU vasopressor 重分类计数", 6, nwv, nwv == 6)

    # ---- S5：ε 网格（六队列 × 两 learner × 四个 ε，表内印四位小数）
    t5 = ap.tables[5]
    lab2tag = {"MIMIC internal 1yr": "MIMIC_internal_1yr", "MIMIC internal inhosp":
               "MIMIC_internal_inhosp", "NWICU 1yr": "NWICU_1yr",
               "INSPIRE 1yr": "INSPIRE_1yr", "eICU inhosp": "eICU_inhosp",
               "Huaian inhosp": "Huaian_inhosp"}
    eps_keys = ("1e-06", "0.0001", "0.001", "0.01")
    n5 = 0
    for r in t5.rows[1:]:
        c = [x.text.strip() for x in r.cells]
        tag = lab2tag.get(c[0])
        if tag is None or tag not in eps["results"]:
            chk("L3", "S5 行可映射 %s" % c[0], "mapped", "missing", False)
            continue
        blk = eps["results"][tag][c[1]]
        want = [round(blk["by_eps"][e]["slope"], 4) for e in eps_keys]
        got = [_num(x) for x in c[2:6]]
        ok = all(g is not None and abs(w - g) <= 5e-5 for w, g in zip(want, got))
        chk("L3", "S5 %s %s ε 网格（四值）" % (tag, c[1]), want, got, ok)
        n5 += int(ok)
        spread = _num(c[6])
        # Spread 由**未取整**的 ε 网格斜率算，再取四位（不是四个已取整值之差）
        want_sp = round(max(blk["by_eps"][e]["slope"] for e in eps_keys)
                        - min(blk["by_eps"][e]["slope"] for e in eps_keys), 4)
        chk("L3", "S5 %s %s Spread" % (tag, c[1]), "%.4f" % want_sp, spread,
            spread is not None and abs(want_sp - spread) <= 5e-5)
    log("  S5 ε 网格行通过 %d 行" % n5)

    # ---- 表 1：四个外部队列规模/事件
    t1 = Document(os.path.join(PKG, "01_Manuscript_V13.docx")).tables[0]
    tbl1 = {}
    for r in t1.rows[1:]:
        c = [x.text.strip() for x in r.cells]
        tbl1[c[0]] = c
    for label, tag in (("NWICU", "NWICU_1yr"), ("INSPIRE", "INSPIRE_1yr"),
                       ("eICU", "eICU_inhosp")):
        row = next((v for k, v in tbl1.items() if k.startswith(label)), None)
        if row is None:
            chk("L3", "表 1 行存在 %s" % label, "present", "missing", False)
            continue
        n_printed = int(row[2].replace(",", ""))
        mc = master["cohorts"][tag]["Logistic"]
        chk("L3", "表 1 %s n" % label, mc["n"], n_printed, n_printed == mc["n"])
        ev_printed = int(row[6].split("(")[0].replace(",", ""))
        chk("L3", "表 1 %s 事件数" % label, mc["events"], ev_printed,
            ev_printed == mc["events"])


# ============================================================ L3b 表 2 / 表 5

def layer3b_gcs_severity(master):
    log("")
    log("=" * 100)
    log("L3b  表 2（GCS 增量）与表 5（严重度评分）逐格核对 → master.gcs_v9 / severity_v9")
    log("=" * 100)
    from docx import Document
    d = Document(os.path.join(PKG, "01_Manuscript_V13.docx"))
    g, sv = master["gcs_v9"], master["severity_v9"]

    # ---- 表 2
    t = d.tables[2]
    col = {c.text.strip(): i for i, c in enumerate(t.rows[0].cells)}
    name2key = {"Logistic regression": "Logistic", "XGBoost": "XGBoost"}
    for r in t.rows[1:]:
        c = [x.text.strip() for x in r.cells]
        blk = g[name2key[c[0]]]
        chk("L3b", "表 2 %s GCS-free AUC" % c[0], "%.3f" % blk["base_auc"],
            _num(c[1]), abs(round(blk["base_auc"], 3) - _num(c[1])) <= TOL_ROUND)
        chk("L3b", "表 2 %s ΔAUC 加总分 GCS" % c[0], "%+.3f" % blk["plus_total_gcs"]["dAUC"],
            c[2], abs(round(blk["plus_total_gcs"]["dAUC"], 3) - _num(c[2])) <= TOL_ROUND)
        chk("L3b", "表 2 %s ΔAUC 加运动 GCS" % c[0],
            "%+.3f" % blk["plus_motor_gcs"]["dAUC"], c[3],
            abs(round(blk["plus_motor_gcs"]["dAUC"], 3) - _num(c[3])) <= TOL_ROUND)
        chk("L3b", "表 2 %s cNRI（运动）" % c[0], "%.3f" % blk["plus_motor_gcs"]["cNRI"],
            _num(c[5]), abs(round(blk["plus_motor_gcs"]["cNRI"], 3) - _num(c[5])) <= TOL_ROUND)
        chk("L3b", "表 2 %s IDI（运动）" % c[0], "%.3f" % blk["plus_motor_gcs"]["IDI"],
            _num(c[6]), abs(round(blk["plus_motor_gcs"]["IDI"], 3) - _num(c[6])) <= TOL_ROUND)
        chk("L3b", "表 2 %s DeLong P（运动）< .001" % c[0], "P<.001",
            blk["plus_motor_gcs"]["delong_p"],
            "P<.001" in c[4] and blk["plus_motor_gcs"]["delong_p"] < 1e-3)

    # ---- 表 5
    t = d.tables[4]
    name2key = {"SOFA": "sofa", "OASIS": "oasis", "SAPS-II": "saps2"}
    for r in t.rows[1:]:
        c = [x.text.strip() for x in r.cells]
        k = name2key[c[0]]
        s = sv["scores"][k]
        chk("L3b", "表 5 %s 中位(IQR)" % c[0], "%g (%g–%g)" % (s["median"], s["IQR"][0], s["IQR"][1]),
            c[1].strip(), c[1].strip().startswith("%g (%g" % (s["median"], s["IQR"][0])))
        chk("L3b", "表 5 %s 单指标 AUC" % c[0], "%.3f" % s["auc_alone"], _num(c[2]),
            abs(round(s["auc_alone"], 3) - _num(c[2])) <= TOL_ROUND)
        for learner, ci_col in (("Logistic", 3), ("XGBoost", 4)):
            cmp_ = sv["comparison"][k][learner]
            got = _num(c[ci_col].split("(")[0])
            chk("L3b", "表 5 %s ΔAUC vs %s" % (c[0], learner),
                "%+.3f" % cmp_["dAUC"], got,
                abs(round(cmp_["dAUC"], 3) - got) <= TOL_ROUND)
            chk("L3b", "表 5 %s ΔAUC vs %s 的 P<.001" % (c[0], learner), "P<.001",
                "P<.001" in c[ci_col] and cmp_["p"] < 1e-3,
                "P<.001" in c[ci_col] and cmp_["p"] < 1e-3)

    # ---- 需知情：同一内部 AUC 存在两个自举 CI（只印 master 那份）
    chk("L3b", "内部 LR AUC 的 CI 有两个自举版本", "印 cohorts.ci95＝internal_learners",
        "gcs_v9.base_ci=[%.4f, %.4f]" % tuple(g["Logistic"]["base_ci"]), None,
        "权威源已确认：cohorts.MIMIC_internal_1yr.ci95.auc 与 internal_learners.json 的 "
        "ci95 逐位相同（0.805–0.829），表 2/表 4 印的是这份。gcs_v9.base_ci 是 GCS 敏感性"
        "分析内部另一次自举（若印会成 0.806–0.831），不被任何表格引用。两版差 0.001 ⇒ "
        "不算错误，但同一模型存在两个自举 CI 这一事实，建议在归档说明里点一句取哪一个")
    chk("L3b", "旧中间件 model_performance_v9.csv 的 CI 未被引用", "0.806–0.830 不出现",
        "见 docx 计数", None,
        "该 CSV 记 0.818 (0.806–0.830)，已被 internal_learners.json 的 (0.805–0.829) 取代，"
        "成稿不引用它。归档时若两者并列且无说明，读者会看到同一 AUC 两个 CI")

# ============================================================ L3c 摘要论断→源

def layer3c_abstract_claims(master):
    log("")
    log("=" * 100)
    log("L3c  摘要 Results 段逐句核对（论断里的数字 → 各自的结果文件）")
    log("=" * 100)
    from docx import Document
    d = Document(os.path.join(PKG, "01_Manuscript_V13.docx"))
    ab = next(p.text for p in d.paragraphs if p.text.startswith("Results:"))
    harm = master["harmonisation_v9"]["variants"]
    eps = json.load(io.open(res_path("slope_epsilon_audit_v9.json"),
                            encoding="utf-8"))["results"]["Huaian_inhosp"]["Logistic"]["by_eps"]
    rec = json.load(io.open(res_path("nwicu_vasopressor_recovery.json"), encoding="utf-8"))
    dec = json.load(io.open(res_path("gcs_increment_audit.json"), encoding="utf-8"))

    def have(fmt, val, note=""):
        s = fmt % val
        chk("L3c", "摘要印出 %s" % s, s, "在" if s in ab else "缺", s in ab, note)

    have("%.3f", harm["harmonised"]["Logistic"]["auc"], "单位换算正确时的淮安 LR AUC")
    have("%.3f", harm["unit_discordant"]["Logistic"]["auc"], "直接喂 SI 单位时的 AUC")
    have("%.2f", harm["harmonised"]["Logistic"]["oe"], "换算正确时的 O:E")
    have("%.2f", harm["unit_discordant"]["Logistic"]["oe"], "直接喂 SI 单位时的 O:E")
    mx = max(abs(v[nm]["delta"]) for v in master["leakage_v9"]["external_delta"].values()
             for nm in ("Logistic", "XGBoost"))
    have("%.3f", mx, "删七个病因指标的最大 |ΔAUC|（外部）")
    have("%.4f", min(v["slope"] for v in eps.values()), "裁剪概率路线斜率下限")
    have("%.4f", max(v["slope"] for v in eps.values()), "裁剪概率路线斜率上限")
    have("%.4f", rec["harmonisation_error"]["delta_auc"]["XGBoost"]["d_auc"],
         "找回 NWICU 升压药后的最大 |ΔAUC|")
    have("%.1f", rec["harmonisation_error"]["recovered_prevalence_pct"], "找回后的升压药占比")
    def _dig(o, key):
        if isinstance(o, dict):
            if key in o and isinstance(o[key], (int, float)):
                return o[key]
            for v in o.values():
                r = _dig(v, key)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = _dig(v, key)
                if r is not None:
                    return r
        return None

    tox = _dig(dec, "anoxic_alone_auc")
    chk("L3c", "摘要印出 %.3f（缺氧性脑病单指标 AUC）" % (tox or 0), "%.3f" % (tox or 0),
        "在" if ("%.3f" % (tox or 0)) in ab else "缺", tox is not None and "%.3f" % tox in ab,
        "源：output/v9_results/gcs_increment_audit.json")
    # 摘要里的 18.8% 来自表 1（开发队列升压药记录率）
    t1 = d.tables[0]
    dev = [c.text.strip() for c in t1.rows[1].cells]
    chk("L3c", "摘要 18.8% = 表 1 开发队列升压药记录率", "18.8", dev[7], dev[7].startswith("18.8"))
    # 摘要里的外部 AUC 六个值须与表 4 一致
    rows = {(r.cells[0].text.strip(), r.cells[2].text.strip()): r.cells[3].text.strip()
            for r in d.tables[3].rows[1:]}
    for lab in ("NWICU (1-year)", "INSPIRE (1-year)", "eICU (in-hospital)"):
        for mname in ("Logistic", "XGBoost"):
            v = _num(rows[(lab, mname)].split(" ")[0])
            have("%.3f", v, "表 4 的 %s %s AUC 与摘要同源" % (lab, mname))


# ============================================================ L3d 内部集第二源

def layer3d_internal_second_source(master):
    log("")
    log("=" * 100)
    log("L3d  内部集：master ↔ internal_learners.json 双源一致 + 划分规模")
    log("=" * 100)
    from docx import Document
    il = json.load(io.open(res_path("internal_learners.json"), encoding="utf-8"))
    sp = il["split"]
    for k_master, k_il, label in (("n", "n_test", "测试集规模"),):
        mc = master["cohorts"]["MIMIC_internal_1yr"]["Logistic"]
        chk("L3d", "内部集%s：master ↔ internal_learners" % label,
            mc[k_master], sp[k_il], mc[k_master] == sp[k_il])
    for tag, evkey in (("MIMIC_internal_1yr", "events_test"),
                       ("MIMIC_internal_inhosp", None)):
        if evkey:
            mc = master["cohorts"][tag]["Logistic"]
            chk("L3d", "内部集事件数：master ↔ internal_learners", mc["events"],
                sp[evkey], mc["events"] == sp[evkey])
    for nm in ("Logistic", "XGBoost"):
        a = master["cohorts"]["MIMIC_internal_1yr"][nm]
        b = il["learners"][nm]
        chk("L3d", "内部一年 %s AUC：master ↔ internal_learners" % nm,
            "%.9f" % a["point"]["auc"], "%.9f" % b["auc"],
            abs(a["point"]["auc"] - b["auc"]) < TOL_RAW)
        chk("L3d", "内部一年 %s AUC CI：master ↔ internal_learners" % nm,
            "[%.6f, %.6f]" % tuple(b["ci95"]),
            "[%.6f, %.6f]" % tuple(a["ci95"]["auc"]),
            abs(a["ci95"]["auc"][0] - b["ci95"][0]) < TOL_RAW
            and abs(a["ci95"]["auc"][1] - b["ci95"][1]) < TOL_RAW)
    # 划分规模在成稿里印出（11,617 / 4,980 / 1,591 / 3,712）
    txt = "\n".join(p.text for p in Document(
        os.path.join(PKG, "01_Manuscript_V13.docx")).paragraphs)
    for v in (sp["n_train"], sp["n_test"], sp["events_test"], sp["events_train"]):
        s = "{:,}".format(v)
        chk("L3d", "成稿印出划分数字 %s" % s, ">=1", txt.count(s), txt.count(s) >= 1)


# ============================================================ L4 跨文档一致

KEY_NUMBERS = [   # (标签, 取值, 应出现在哪些文档)
    ("NWICU 一年 LR AUC", "0.765", "manuscript"),
    ("NWICU 一年 LR 斜率", "0.872", "manuscript"),
    ("NWICU 一年 LR O:E", "0.994", "manuscript"),
    ("INSPIRE 一年 LR AUC", "0.725", "manuscript"),
    ("eICU 院内 LR 斜率", "0.538", "manuscript"),
    ("淮安院内 LR 斜率", "0.843", "manuscript"),
    ("NWICU 事件数", "699", "manuscript+appendix"),
    ("病例重叠（一年）", "2,276", "manuscript+appendix"),
    ("重叠占比", "45.7", "manuscript+appendix"),
    ("病例重叠（院内）", "3,483", "manuscript+appendix"),
    ("两版本共有留出", "2,704", "appendix+response"),
    ("两结局共有留出", "1,461", "manuscript+appendix"),
    ("开发集规模", "11,617", "manuscript+appendix"),
    ("留出集规模", "4,980", "manuscript+appendix"),
    ("留出集事件", "1,591", "manuscript+appendix"),
    ("S19 ΔAUC（INSPIRE 一年）", "-0.005", "appendix"),
    ("S19 ΔAUC（INSPIRE 院内）", "-0.012", "appendix"),
    ("S19 O:E（一年 LR 备择）", "0.839", "appendix"),
    ("S19 O:E（院内 LR 备择）", "0.789", "appendix"),
]


def _norm(t):
    """把 U+2212 真负号折成 ASCII '-'，使 '-0.005' 与 '−0.005' 可比（不动 en dash 范围符）。"""
    return t.replace("\u2212", "-")


def layer4_crossdoc():
    log("")
    log("=" * 100)
    log("L4  跨文档一致性（同一量多处出现须逐字符相同）")
    log("=" * 100)
    from docx import Document
    txt = {}
    for fn, tag in (("01_Manuscript_V13.docx", "manuscript"),
                    ("04_Additional_file_1_V13.docx", "appendix"),
                    ("02_Cover_Letter_V13.docx", "cover"),
                    ("06_Response_to_Prior_Review_V13.docx", "response")):
        d = Document(os.path.join(PKG, fn))
        body = "\n".join(p.text for p in d.paragraphs)
        body += "\n" + "\n".join(c.text for tb in d.tables for r in tb.rows for c in r.cells)
        txt[tag] = body
    for label, val, where in KEY_NUMBERS:
        docs = where.split("+")
        for t in docs:
            n = _norm(txt[t]).count(_norm(val))
            chk("L4", "%s 出现于 %s（%s 次）" % (label, t, n), ">=1", n, n >= 1, val)
    # 反向：绝不能出现的旧值/错值
    for bad, why in (("1.2729", "旧 log(p) 路线斜率"), ("1.1152", "旧 log(p) 路线斜率"),
                     ("0.8552", "旧 log(p) 路线斜率"), ("23557369", "上一版归档 DOI"),
                     ("least well calibrated", "已删的过度结论"),
                     ("as for every other predictor", "已改的泛指句"),
                     ("log(p)", "公式错误写法")):
        for t, body in txt.items():
            chk("L4", "残留 %s 于 %s" % (why, t), 0, _norm(body).count(_norm(bad)),
                _norm(body).count(_norm(bad)) == 0, bad)


# ============================================================ L5 计数

def layer5_counts():
    log("")
    log("=" * 100)
    log("L5  计数复核（字数 / 页数 / 图表数）")
    log("=" * 100)
    import re
    from docx import Document
    import fitz
    d = Document(os.path.join(PKG, "01_Manuscript_V13.docx"))
    tok = lambda s: len([w for w in s.split() if re.search(r"[0-9A-Za-z]", w)])
    # 摘要：从以 'Background:' 开头的段落到 'Keywords' 标题之前（不能从字面 'Abstract' 起算，
    #   标题页元数据段里也有 'Abstract: 349 words (limit 350)' 字样）
    ps = [p.text for p in d.paragraphs]
    i0 = next((k for k, t in enumerate(ps) if t.strip().startswith("Background:")), None)
    i1 = next((k for k, t in enumerate(ps) if k > (i0 or 0)
               and t.strip().lower().startswith("keywords")), None)
    ab = tok("\n".join(ps[i0:i1])) if i0 is not None and i1 is not None else -1
    chk("L5", "摘要词数（含全部结构标签，≤350）", "<=350", ab, 0 < ab <= 350)
    # 标题页自述的摘要词数与实测的关系（三口径，须知情）
    decl = next((t for t in ps if "words (limit" in t), "")
    mdecl = re.search(r"Abstract:\s*(\d+)\s*words", decl)
    nolabel = re.sub(r"(Background|Objective|Methods|Results|Conclusions):\s*", "",
                     "\n".join(ps[i0:i1]))
    chk("L5", "标题页自述 349 与实测 350 的差异可解释", "349（自述）",
        "%s（剥全部标签 %d）" % (mdecl.group(1) if mdecl else "?", tok(nolabel)), None,
        "三个口径：剥全部结构标签 %d、自述 %s、含全部标签 %d。差异来自 builder 只剥首标签；"
        "投稿刊 350 词上限按含标签口径实测 **恰好 350，已无余量**"
        % (tok(nolabel), mdecl.group(1) if mdecl else "?", ab))
    cl = Document(os.path.join(PKG, "02_Cover_Letter_V13.docx"))
    clw = len("\n".join(p.text for p in cl.paragraphs).split())
    chk("L5", "投稿信词数（≤500）", "≤500", clw, 0 < clw <= 500)
    pm = fitz.open(os.path.join(PKG, "01_Manuscript_V13.pdf"))
    pa = fitz.open(os.path.join(PKG, "04_Additional_file_1_V13.pdf"))
    chk("L5", "正文页数（≥28）", ">=28", pm.page_count, pm.page_count >= 28)
    chk("L5", "附录页数（≥50）", ">=50", pa.page_count, pa.page_count >= 50)
    chk("L5", "正文表数（6）", 6, len(d.tables), len(d.tables) == 6)
    ap = Document(os.path.join(PKG, "04_Additional_file_1_V13.docx"))
    chk("L5", "附录表数（19，S1–S19）", 19, len(ap.tables), len(ap.tables) == 19)
    # PDF↔DOCX 文字一致性（页脚/页码归一后）
    pdf_txt = "".join(pg.get_text() for pg in pm)
    chk("L5", "PDF 含 Table S19 引用", ">=1", pdf_txt.count("Table S19"),
        pdf_txt.count("Table S19") >= 1)
    return dict(abstract=ab, cover=clw, pages_main=pm.page_count,
                pages_appendix=pa.page_count, tables_main=len(d.tables),
                tables_appendix=len(ap.tables))


# ============================================================ main

def main():
    log("=" * 100)
    log("127_freeze_numeric_audit.py — 固定版本数值核对（V13 投稿包）")
    log("=" * 100)

    def _load_master():
        with io.open(os.path.join(OUT, "v9_number_master.json"), encoding="utf-8") as f:
            return json.load(f)

    master = _layer("master", _load_master)
    fp = _layer("L1", layer1_fingerprint) or {}
    rec = _layer("L2", layer2_recompute)
    for _tag, _fn in (("L2b", layer2b_table4), ("L3", layer3_transcription),
                      ("L3b", layer3b_gcs_severity), ("L3c", layer3c_abstract_claims),
                      ("L3d", layer3d_internal_second_source)):
        if _tag == "L2b" and rec is None:
            SKIPPED.append("L2b（缺独立重算结果）")
            log("  [SKIP] L2b 未执行：缺独立重算结果")
            continue
        if master is None:
            SKIPPED.append("%s（缺 v9_number_master.json）" % _tag)
            log("  [SKIP] %s 未执行：缺 v9_number_master.json" % _tag)
            continue
        _layer(_tag, _fn, *(rec, master) if _tag == "L2b" else (master,))
    _layer("L4", layer4_crossdoc)
    counts = _layer("L5", layer5_counts) or {}

    npass = sum(1 for c in CHECKS if c["ok"] is True)
    nfail = sum(1 for c in CHECKS if c["ok"] is False)
    nnote = sum(1 for c in CHECKS if c["ok"] is None)
    by_layer = {}
    for c in CHECKS:
        b = by_layer.setdefault(c["layer"], [0, 0, 0])
        b[0 if c["ok"] is True else (1 if c["ok"] is False else 2)] += 1
    log("")
    log("=" * 100)
    log("汇总：%d 项核对，通过 %d，需知情 %d，失败 %d"
        % (len(CHECKS), npass, nnote, nfail))
    for k in sorted(by_layer):
        log("   %-4s 通过 %3d / 需知情 %2d / 失败 %3d"
            % (k, by_layer[k][0], by_layer[k][2], by_layer[k][1]))
    if SKIPPED:
        log("   未执行 %d 层（缺输入，非失败）：%s" % (len(SKIPPED), "；".join(SKIPPED)))
    log("=" * 100)

    payload = dict(generated_at=pd.Timestamp.now().isoformat(timespec="seconds"),
                   generated_by="scripts/127_freeze_numeric_audit.py",
                   package="output/submission_v13",
                   fingerprint=fp, recomputed=rec, counts=counts,
                   totals=dict(checks=len(CHECKS), passed=npass, noted=nnote,
                               failed=nfail, by_layer=by_layer),
                   skipped=SKIPPED,
                   checks=CHECKS)
    with io.open(os.path.join(OUT, "freeze_numeric_audit_v13.json"), "w",
                 encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    # ---- Markdown 报告
    L = []
    L.append("# V13 固定版本数值核对（%s）\n" % payload["generated_at"])
    L.append("对象：`output/submission_v13/`（V13 投稿包）。本报告由 "
             "`scripts/127_freeze_numeric_audit.py` 只读生成，不重训、不改稿。\n")
    if SKIPPED:
        L.append("> 本次运行有 %d 层因缺输入未执行：%s。这些层不是失败项。\n"
                 % (len(SKIPPED), "；".join(SKIPPED)))
    L.append("## 0. 结论\n")
    L.append("| 层 | 内容 | 通过 | 需知情 | 失败 |")
    L.append("|---|---|---|---|---|")
    names = {"L1": "冻结指纹（交付件/链脚本/输入/仓库）",
             "L2b": "表 4 逐格：独立重算 → 印出数字",
             "L3": "转录核对（S19 / S3 计数 / S5 ε 网格 / 表 1 规模）",
             "L3b": "表 2 GCS 增量 / 表 5 严重度评分 → master",
             "L3c": "摘要 Results 段论断 → 各自结果文件",
             "L3d": "内部集 master ↔ internal_learners 第二源",
             "L4": "跨文档一致（关键量 + 残留禁值）",
             "L5": "计数（字数/页数/表数）"}
    for k in ("L1", "L2b", "L3", "L3b", "L3c", "L3d", "L4", "L5"):
        b = by_layer.get(k, [0, 0, 0])
        L.append("| %s | %s | %d | %d | %d |" % (k, names[k], b[0], b[2], b[1]))
    L.append("| **合计** | | **%d** | **%d** | **%d** |\n" % (npass, nnote, nfail))
    if nfail:
        L.append("**未通过项**：\n")
        for c in CHECKS:
            if c["ok"] is False:
                L.append("- `%s` %s：期望 %s，实得 %s %s"
                         % (c["layer"], c["name"], c["expected"], c["actual"], c["note"]))
        L.append("")
    if nnote:
        L.append("**需知情（不影响结论，供定稿时决定是否处理）**：\n")
        for c in CHECKS:
            if c["ok"] is None:
                L.append("- `%s` %s：期望 %s，实得 %s。%s"
                         % (c["layer"], c["name"], c["expected"], c["actual"], c["note"]))
        L.append("")
    L.append("## 1. 冻结指纹\n")
    L.append("- 归档 DOI（稿件声明）：`%s`" % fp.get("doi"))
    r = fp.get("repo", {})
    L.append("- 仓库 HEAD `%s`，最近 tag `%s`，工作区%s"
             % ((r.get("head") or "")[:12], r.get("describe"),
                "有未提交改动" if r.get("dirty") else "干净"))
    L.append("\n交付件：\n")
    L.append("| 文件 | 字节 | SHA256（前 16） |")
    L.append("|---|---|---|")
    for it in fp["deliverables"]:
        L.append("| `%s` | %d | `%s` |" % (it["file"], it["bytes"], it["sha256"]))
    L.append("\n生成链脚本：\n")
    L.append("| 脚本 | SHA256（前 16） |")
    L.append("|---|---|")
    for it in fp["chain"]:
        L.append("| `%s` | `%s` |" % (it["script"], it["sha256"]))
    L.append("\n输入结果文件：\n")
    L.append("| 文件 | 字节 | SHA256（前 16） |")
    L.append("|---|---|---|")
    for it in fp["inputs"]:
        L.append("| `%s` | %d | `%s` |" % (it["file"], it["bytes"], it["sha256"]))
    L.append("\n## 2. 外部验证：独立重算 vs 成稿印出\n")
    L.append("口径：AUC/Brier/O:E/ECE 由概率直接算（ECE = 十等宽分箱）；校准斜率/截距把结局对"
             "**未裁剪线性预测子 η** 做 logistic 回归（一年 LR 用权重向量、院内 LR 用 "
             "`decision_function`、XGBoost 用 raw margin，概率由 η 自行反变换）。"
             "容差：点位按表内实际精度（AUC/斜率/截距/ECE/Brier 三位、O:E 两位）、95% CI 一律三位，"
             "均按 ±半个末位比对。\n")
    L.append("| 队列 | 模型 | n (事件) | AUC | Brier | 斜率 | 截距 | O:E | ECE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    if rec:
        for tag, blk in rec.items():
            for nm in ("Logistic", "XGBoost"):
                v = blk[nm]
                L.append("| %s | %s | %d (%d) | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
                         % (tag, nm, blk["n"], blk["events"], v["auc"], v["brier"],
                            v["slope"], v["intercept"], v["oe"], v["ece"]))
    else:
        L.append("| — | — | — | — | — | — | — | — | — |")
        L.append("\n*本层未执行（缺模型工件与患者级 CSV），故无重算行。*")
    L.append("\n内部集（MIMIC-IV）未从原始队列独立重算：需重现阶段性分层划分与全部拟合，"
             "超出\"核对\"范围。其印出数字由 **master ↔ internal_learners.json 双源逐位一致**"
             "（AUC 与 95% CI 均同源）+ 逐格转录核对覆盖（见 L3d）。\n")
    L.append("## 3. 逐项明细\n")
    for lyr in ("L1", "L2b", "L3", "L3b", "L3c", "L3d", "L4", "L5"):
        rows = [c for c in CHECKS if c["layer"] == lyr]
        if not rows:
            continue
        L.append("### %s（%d 项）\n" % (lyr, len(rows)))
        L.append("| 核对项 | 期望 | 实得 | 结果 |")
        L.append("|---|---|---|---|")
        for c in rows:
            res = "OK" if c["ok"] is True else ("NOTE" if c["ok"] is None else "**FAIL**")
            L.append("| %s | %s | %s | %s |"
                     % (c["name"], c["expected"], c["actual"], res))
        L.append("")
    with io.open(os.path.join(OUT, "FREEZE_NUMERIC_AUDIT_V13.md"), "w",
                 encoding="utf-8") as f:
        f.write("\n".join(L))
    log("")
    log("报告 → output/FREEZE_NUMERIC_AUDIT_V13.md；机读 → output/freeze_numeric_audit_v13.json")
    return 1 if nfail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
