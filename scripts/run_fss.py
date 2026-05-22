#!/usr/bin/env python3
"""
FSS sweep: N in {500, 1000, 2000} on one mid-tau condition per model.
Mid-tau conditions chosen to avoid degenerate high/low-tau limits:
  ER  mean_k=4   (moderate degree heterogeneity)
  BA  m=3        (mid-range attachment parameter)
  WS  beta=0.2   (partial rewiring, non-trivial clustering)
N=1000 files are skipped if already produced by run_main.py.
"""
import sys
sys.path.insert(0, '.')
from sweep import run_sweep

FSS_CONDITIONS = [
    ('er', 4.0),
    ('ba', 3),
    ('ws', 0.2),
]
FSS_SIZES = [500, 1000, 2000]
N_REAL = 100
BATCH_SIZE = 10

if __name__ == '__main__':
    for model, pv in FSS_CONDITIONS:
        for n in FSS_SIZES:
            print(f'=== FSS: {model} param={pv} N={n} ===')
            run_sweep(model, [pv], n=n, n_real=N_REAL, batch_size=BATCH_SIZE)
    print('\nDone. FSS .npz files written to data/')
