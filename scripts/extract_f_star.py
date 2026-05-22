#!/usr/bin/env python3
"""Extract f* (EC degradation onset) for all 23 conditions. Writes results/f_star.csv."""
import sys
import os
import csv
import numpy as np
sys.path.insert(0, '.')
from src.networks import ER_PARAMS, BA_PARAMS, WS_PARAMS

EPS = 0.01   # relative variance threshold — adjust after calibrating with calibrate_f_star.py
os.makedirs('results', exist_ok=True)

rows = []
for model, pvs in [('er', ER_PARAMS['mean_k']), ('ba', BA_PARAMS['m']), ('ws', WS_PARAMS['beta'])]:
    for pv in pvs:
        try:
            d = np.load(f'data/{model}_param{pv}_N1000.npz')
        except FileNotFoundError:
            print(f'  Missing: data/{model}_param{pv}_N1000.npz — skipping')
            continue
        f       = d['f_values']
        var_ec  = d['var_ec']          # (n_real, n_rec)
        v0      = var_ec[:, 0:1] + 1e-12
        rel_var = var_ec / v0

        f_star = []
        for r in range(rel_var.shape[0]):
            below = np.where(rel_var[r] < EPS)[0]
            f_star.append(float(f[below[0]]) if len(below) else 1.0)
        f_star = np.array(f_star)
        rows.append({'model': model, 'param': pv,
                     'f_star_mean': f_star.mean(), 'f_star_std': f_star.std()})
        print(f'{model} param={pv}: f*={f_star.mean():.3f} ± {f_star.std():.3f}')

with open('results/f_star.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['model', 'param', 'f_star_mean', 'f_star_std'])
    w.writeheader()
    w.writerows(rows)
print('Saved results/f_star.csv')
