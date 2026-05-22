#!/usr/bin/env python3
"""
Main sweep: N=1000, 100 realizations, all three models.
Estimated runtime: 4–8 hours on 8 cores.
Output: data/{model}_param{val}_N1000.npz  (23 files total)
"""
import sys
sys.path.insert(0, '.')

from src.networks import ER_PARAMS, BA_PARAMS, WS_PARAMS
from sweep import run_sweep

N = 1000
N_REAL = 100
BATCH_SIZE = 10

if __name__ == '__main__':
    print('=== ER sweep ===')
    run_sweep('er', ER_PARAMS['mean_k'], n=N, n_real=N_REAL, batch_size=BATCH_SIZE)

    print('\n=== BA sweep ===')
    run_sweep('ba', BA_PARAMS['m'], n=N, n_real=N_REAL, batch_size=BATCH_SIZE)

    print('\n=== WS sweep ===')
    run_sweep('ws', WS_PARAMS['beta'], n=N, n_real=N_REAL, batch_size=BATCH_SIZE)

    print('\nDone. All .npz files written to data/')
