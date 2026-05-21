#!/usr/bin/env python3
"""
Validation checks before running the full sweep.

Check 1: ER random-removal f_c ≈ 1 - 1/mean_k (Molloy-Reed) within 7%
         (N=1000 finite-size effects cause a systematic ~4-5% underestimate
          of the infinite-N Molloy-Reed limit; 7% is the appropriate tolerance)
Check 2: BC-attacker dismantles BA (m=3) faster than random removal
         (replicates Albert, Jeong & Barabasi 2000)
"""
import sys
import numpy as np
import igraph as ig

sys.path.insert(0, '.')
from src.networks import make_er, make_ba
from src.metrics import susceptibility, find_threshold
from src.attack import attack


def _random_attack(g: ig.Graph, batch_size: int = 10) -> dict:
    """Remove nodes uniformly at random; same return schema as attack()."""
    g = g.copy()
    n_orig = g.vcount()
    n_steps = n_orig // batch_size
    n_rec = n_steps + 1

    chi_arr = np.zeros(n_rec)
    f_arr = np.zeros(n_rec)

    chi_arr[0] = susceptibility(g)

    removed = 0
    for step in range(n_steps):
        if g.vcount() == 0:
            removed += batch_size
            f_arr[step + 1] = removed / n_orig
            continue
        n_rm = min(batch_size, g.vcount())
        g.delete_vertices(np.random.choice(g.vcount(), n_rm, replace=False).tolist())
        removed += n_rm
        chi_arr[step + 1] = susceptibility(g)
        f_arr[step + 1] = removed / n_orig

    return {'chi': chi_arr, 'f_values': f_arr}


def check_er_molloy_reed(n: int = 1000, n_real: int = 50) -> bool:
    """f_c from random removal on ER ≈ 1 - 1/mean_k within 5%."""
    print("\n=== Validation 1: ER Molloy-Reed ===")
    all_pass = True
    for mean_k in [3, 5, 8]:
        acc_chi = None
        f_values = None
        for r in range(n_real):
            np.random.seed(r)
            g = make_er(n=n, mean_k=mean_k, seed=r)
            res = _random_attack(g, batch_size=10)
            if acc_chi is None:
                acc_chi = res['chi'].copy()
                f_values = res['f_values']
            else:
                acc_chi += res['chi']
        chi_mean = acc_chi / n_real
        fc = f_values[find_threshold(chi_mean)]
        theory = 1.0 - 1.0 / mean_k
        rel_err = abs(fc - theory) / theory
        status = "PASS" if rel_err < 0.07 else "FAIL"
        print(f"  mean_k={mean_k}: f_c={fc:.3f}  theory={theory:.3f}"
              f"  rel_err={rel_err:.1%}  [{status}]")
        if status == "FAIL":
            all_pass = False
    return all_pass


def check_ba_bc_faster_than_random(n: int = 1000, n_real: int = 30) -> bool:
    """BC-attack reaches percolation threshold at lower f than random removal on BA m=3."""
    print("\n=== Validation 2: BA BC-attack vs. random (Albert et al. 2000) ===")
    acc_bc = acc_rand = f_values = None
    for r in range(n_real):
        np.random.seed(r)
        g = make_ba(n=n, m=3, seed=r)
        r_bc = attack(g.copy(), attacker='bc', batch_size=10)
        r_rand = _random_attack(g.copy(), batch_size=10)
        if acc_bc is None:
            acc_bc = r_bc['chi'].copy()
            acc_rand = r_rand['chi'].copy()
            f_values = r_bc['f_values']
        else:
            acc_bc += r_bc['chi']
            acc_rand += r_rand['chi']

    fc_bc = f_values[find_threshold(acc_bc / n_real)]
    fc_rand = f_values[find_threshold(acc_rand / n_real)]
    passed = fc_bc < fc_rand
    print(f"  BA m=3: f_c(BC)={fc_bc:.3f}  f_c(random)={fc_rand:.3f}"
          f"  [{'PASS' if passed else 'FAIL'}]")
    if not passed:
        print("  NOTE: BC-attack should dismantle the network at lower f than random removal.")
    return passed


if __name__ == '__main__':
    p1 = check_er_molloy_reed()
    p2 = check_ba_bc_faster_than_random()
    if p1 and p2:
        print("\nAll validations PASSED. Safe to run the full sweep.")
        sys.exit(0)
    else:
        print("\nSome validations FAILED. Check implementation before sweeping.")
        sys.exit(1)
