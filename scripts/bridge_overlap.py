#!/usr/bin/env python3
"""
Compute top-k BC/EC node overlap for intact networks across all conditions.
Low overlap = BC and EC target structurally different nodes.
Hypothesis: low overlap → high Δf.
Writes results/overlap.csv.
"""
import sys
import os
import csv
import numpy as np
import igraph as ig
sys.path.insert(0, '.')
from src.networks import make_er, make_ba, make_ws, ER_PARAMS, BA_PARAMS, WS_PARAMS
from src.metrics import find_threshold

K = 50        # top-k nodes to compare (5% of N=1000)
N_SAMPLE = 20 # realizations per condition
N = 1000

os.makedirs('results', exist_ok=True)


def top_k_overlap(model, pv, seed, k):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    ig.set_random_number_generator(_random.Random(seed))

    if model == 'er':
        g = make_er(N, pv, seed)
    elif model == 'ba':
        g = make_ba(N, int(pv), seed)
    elif model == 'ws':
        g = make_ws(N, pv, seed=seed)

    bc = np.array(g.betweenness(directed=False))
    try:
        ec = np.array(g.eigenvector_centrality(scale=True))
    except ig.InternalError:
        ec = np.ones(N)

    top_bc = set(np.argsort(bc)[-k:])
    top_ec = set(np.argsort(ec)[-k:])
    return len(top_bc & top_ec) / k


rows = []
for model, pvs in [('er', ER_PARAMS['mean_k']), ('ba', BA_PARAMS['m']), ('ws', WS_PARAMS['beta'])]:
    for pv in pvs:
        d = np.load(f'data/{model}_param{pv}_N{N}.npz')
        f = d['f_values']
        fc_bc = float(f[find_threshold(d['chi_bc'].mean(0))])
        fc_ec = float(f[find_threshold(d['chi_ec'].mean(0))])
        delta_f = fc_ec - fc_bc
        tau0 = float(d['tau_initial'].mean())

        overlaps = []
        for idx in range(N_SAMPLE):
            seed = abs(hash((model, str(pv), N, idx))) % (2 ** 31)
            overlaps.append(top_k_overlap(model, pv, seed, K))

        overlaps = np.array(overlaps)
        rows.append({
            'model': model, 'param': pv,
            'overlap_mean': overlaps.mean(), 'overlap_std': overlaps.std(),
            'delta_f': delta_f, 'tau_initial': tau0,
        })
        print(f'{model} param={pv}: overlap={overlaps.mean():.3f}±{overlaps.std():.3f}  Δf={delta_f:.3f}  τ={tau0:.3f}')

with open('results/overlap.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['model', 'param', 'overlap_mean', 'overlap_std',
                                        'delta_f', 'tau_initial'])
    w.writeheader()
    w.writerows(rows)
print('\nSaved results/overlap.csv')
