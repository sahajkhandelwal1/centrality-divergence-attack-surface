#!/usr/bin/env python3
"""Fit Δf ~ structural parameter per model. Writes results/regression.csv."""
import sys
import os
import csv
import numpy as np
from scipy.stats import linregress
sys.path.insert(0, '.')
from src.networks import ER_PARAMS, BA_PARAMS, WS_PARAMS
from src.metrics import find_threshold

os.makedirs('results', exist_ok=True)


def load_delta_f(model, pv):
    d = np.load(f'data/{model}_param{pv}_N1000.npz')
    f = d['f_values']
    fc_bc = float(f[find_threshold(d['chi_bc'].mean(axis=0))])
    fc_ec = float(f[find_threshold(d['chi_ec'].mean(axis=0))])
    return fc_ec - fc_bc


rows = []
for model, pvs in [('er', ER_PARAMS['mean_k']), ('ba', BA_PARAMS['m']), ('ws', WS_PARAMS['beta'])]:
    params, deltas = [], []
    for pv in pvs:
        try:
            deltas.append(load_delta_f(model, pv))
            params.append(float(pv))
        except FileNotFoundError:
            print(f'  Missing: data/{model}_param{pv}_N1000.npz — skipping')
            continue
    if len(params) < 2:
        print(f'{model.upper()}: insufficient data for regression')
        continue
    params, deltas = np.array(params), np.array(deltas)
    slope, intercept, r, p, _ = linregress(params, deltas)
    print(f'{model.upper()}: slope={slope:.4f}, intercept={intercept:.4f}, R²={r**2:.3f}, p={p:.4f}')
    for pv, df in zip(params, deltas):
        rows.append({'model': model, 'param': pv, 'delta_f': df,
                     'predicted': intercept + slope * pv,
                     'r2': r**2, 'slope': slope, 'intercept': intercept, 'p_value': p})

with open('results/regression.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['model', 'param', 'delta_f', 'predicted',
                                        'r2', 'slope', 'intercept', 'p_value'])
    w.writeheader()
    w.writerows(rows)
print('Saved results/regression.csv')
