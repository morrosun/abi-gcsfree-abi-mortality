# -*- coding: utf-8 -*-
"""详查 model_inhosp.joblib 结构，判断能否直接用于本地验证"""
import joblib, numpy as np, json, os
# --- Portability -----------------------------------------------------
# Set the environment variable ABI_BASE to the root of your analysis workspace
# (the directory that contains sql/, data/, output/ and ABI3/).  If unset, the
# original development path is used.
import os as _os
ABI_BASE = _os.environ.get("ABI_BASE", r"D:/BaiduSyncdisk/MIMIC/ABI/ABI1")
# ---------------------------------------------------------------------
BASE = ABI_BASE
m = joblib.load(os.path.join(BASE, 'output', 'model_inhosp.joblib'))
print('keys:', list(m.keys()))
for k in ['feats', 'cont', 'medians']:
    print(f'\n{k}:')
    print(' ', m[k])
for k in ['lr', 'xg']:
    o = m[k]
    print(f'\n{k}: {type(o).__name__}')
    print('  n_features_in_:', getattr(o, 'n_features_in_', None))
    print('  feature_names_in_:', getattr(o, 'feature_names_in_', None))
    c = getattr(o, 'coef_', None)
    if c is not None:
        print('  coef_ shape:', np.shape(c), '| intercept:', np.ravel(o.intercept_))
    print('  classes_:', getattr(o, 'classes_', None))
print('\nscaler:')
sc = m['scaler']
print('  mean_ len:', len(sc.mean_), '| scale_ len:', len(sc.scale_))
print('  feature_names_in_:', getattr(sc, 'feature_names_in_', None))
print('\ninternal:')
print(' ', json.dumps(m['internal'], ensure_ascii=False, default=str)[:1500])
