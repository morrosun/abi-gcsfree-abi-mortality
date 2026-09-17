# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
# -*- coding: utf-8 -*-
"""figR4: external decision-curve analysis (net benefit) using recalibrated XGBoost
probabilities on the three external databases. Answers Minor Comment #1
(external DCA + calibration CIs). Light theme, English labels."""
import joblib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ABI_BASE + "/output"
d = joblib.load(OUT + "/recal_preds.joblib")

def net_benefit(y, p, th):
    y = np.asarray(y); N = len(y); nb = []
    for pt in th:
        pred = p >= pt
        tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
        nb.append(tp / N - (fp / N) * (pt / (1 - pt)))
    return np.array(nb)

dbs = [("eICU (in-hospital)", "eICU"),
       ("NWICU (1-year)", "NWICU"),
       ("INSPIRE (1-year)", "INSPIRE")]
th = np.arange(0.01, 0.60, 0.01)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (key, short) in zip(axes, dbs):
    rec = d[(key, "XGBoost")]
    y = np.asarray(rec["y"]); p = np.asarray(rec["log"])  # full logistic recalibrated
    prev = y.mean()
    nb_model = net_benefit(y, p, th)
    nb_all = prev - (1 - prev) * (th / (1 - th))
    ax.plot(th, nb_model, color="#1f4e79", lw=2.2, label="XGBoost (recalibrated)")
    ax.plot(th, nb_all, color="#b0891f", lw=1.4, ls="--", label="Treat all")
    ax.axhline(0, color="#888", lw=1.2, ls=":", label="Treat none")
    ax.set_xlim(0, 0.6); ax.set_ylim(min(-0.03, nb_model.min() - 0.01), max(nb_model.max(), prev) + 0.02)
    ax.set_title(f"{short}  (event rate {prev*100:.1f}%)", fontsize=12, color="#1a1a1a")
    ax.set_xlabel("Threshold probability", fontsize=11)
    if ax is axes[0]:
        ax.set_ylabel("Net benefit", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    ax.tick_params(labelsize=9)
fig.suptitle("External decision-curve analysis (recalibrated XGBoost)", fontsize=13.5, y=1.02, color="#12324f")
fig.tight_layout()
fig.savefig(OUT + "/figR4_external_dca.png", dpi=145, bbox_inches="tight", facecolor="white")
print("WROTE figR4_external_dca.png")
