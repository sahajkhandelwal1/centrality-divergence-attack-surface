"""
Sweep orchestration: run BC and EC attacks across all (model, param, N)
conditions in parallel with joblib and save one .npz per condition.

.npz schema per condition (n_real realizations, n_rec = N//batch_size + 1 steps):
  s_bc        (n_real, n_rec)  giant fraction under BC attack
  chi_bc      (n_real, n_rec)  susceptibility under BC attack
  tau_bc      (n_real, n_rec)  Kendall tau under BC attack
  s_ec        (n_real, n_rec)
  chi_ec      (n_real, n_rec)
  tau_ec      (n_real, n_rec)
  f_values    (n_rec,)         removal fractions (shared; index 0 = intact network)
  tau_initial (n_real,)        tau on intact graph (= tau_bc[:, 0])
"""
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.networks import make_er, make_ba, make_ws, WS_PARAMS
from src.attack import attack

DATA_DIR = 'data'


def _one_realization(model: str, param_val, n: int, idx: int,
                     batch_size: int) -> dict:
    """Generate one graph and run both BC and EC attacks on it."""
    import random as _random

    seed = abs(hash((model, str(param_val), n, idx))) % (2 ** 31)
    _random.seed(seed)
    np.random.seed(seed)

    if model == 'er':
        g = make_er(n=n, mean_k=param_val, seed=seed)
    elif model == 'ba':
        g = make_ba(n=n, m=param_val, seed=seed)
    elif model == 'ws':
        g = make_ws(n=n, beta=param_val, k_ws=WS_PARAMS['k_ws'], seed=seed)
    else:
        raise ValueError(f"Unknown model: {model!r}")

    r_bc = attack(g.copy(), attacker='bc', batch_size=batch_size)
    r_ec = attack(g.copy(), attacker='ec', batch_size=batch_size)

    return {
        's_bc':   r_bc['s'],   'chi_bc': r_bc['chi'], 'tau_bc': r_bc['tau'],
        's_ec':   r_ec['s'],   'chi_ec': r_ec['chi'], 'tau_ec': r_ec['tau'],
        'f_values': r_bc['f_values'],
    }


def run_sweep(model: str, param_vals: list, n: int, n_real: int,
              batch_size: int = 10, n_jobs: int = -1) -> None:
    """
    Run the sweep over param_vals for one model at one network size.
    Skips conditions whose .npz already exists (safe to re-run after interruption).
    Progress bars update per completed realization within each condition.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    pending = [pv for pv in param_vals
               if not os.path.exists(os.path.join(DATA_DIR, f'{model}_param{pv}_N{n}.npz'))]

    for pv in param_vals:
        out = os.path.join(DATA_DIR, f'{model}_param{pv}_N{n}.npz')
        if os.path.exists(out):
            tqdm.write(f'  Skip (exists): {out}')
            continue

        desc = f'{model} param={pv} N={n}'
        results = list(tqdm(
            Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
                delayed(_one_realization)(model, pv, n, r, batch_size)
                for r in range(n_real)
            ),
            total=n_real,
            desc=desc,
            unit='real',
            ncols=88,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ))

        arrays = {k: np.stack([res[k] for res in results]) for k in results[0]}
        arrays['f_values'] = results[0]['f_values']
        arrays['tau_initial'] = arrays['tau_bc'][:, 0]

        np.savez_compressed(out, **arrays)
        tqdm.write(f'  Saved: {out}')
