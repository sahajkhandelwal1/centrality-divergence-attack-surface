import numpy as np
import igraph as ig
from src.metrics import susceptibility, kendall_tau_centralities


def _centrality_scores(g: ig.Graph, attacker: str) -> np.ndarray:
    if attacker == 'bc':
        return np.array(g.betweenness(directed=False))
    elif attacker == 'ec':
        if g.ecount() == 0:
            return np.zeros(g.vcount())
        try:
            return np.array(g.eigenvector_centrality(directed=False, scale=True))
        except ig.InternalError:
            return np.zeros(g.vcount())
    else:
        raise ValueError(f"Unknown attacker {attacker!r}. Use 'bc' or 'ec'.")


def attack(g: ig.Graph, attacker: str = 'bc', batch_size: int = 10) -> dict:
    """
    Batched adaptive targeted attack. Centralities are recomputed every
    batch_size removals.

    Returns dict with arrays of shape (n_steps + 1,):
      's'        : giant-component fraction S(f)
      'chi'      : susceptibility χ(f)
      'tau'      : Kendall τ between BC and EC on the residual graph
      'f_values' : removal fractions (index 0 = intact network, f=0)
    """
    g = g.copy()
    n_orig = g.vcount()
    n_steps = n_orig // batch_size
    n_rec = n_steps + 1

    s_arr = np.zeros(n_rec)
    chi_arr = np.zeros(n_rec)
    tau_arr = np.zeros(n_rec)
    f_arr = np.zeros(n_rec)

    # Record intact network at f=0
    sizes0 = g.connected_components().sizes()
    s_arr[0] = max(sizes0) / n_orig if sizes0 else 0.0
    chi_arr[0] = susceptibility(g)
    tau_arr[0] = kendall_tau_centralities(g)

    removed = 0
    for step in range(n_steps):
        if g.vcount() == 0:
            removed += batch_size
            f_arr[step + 1] = removed / n_orig
            continue

        scores = _centrality_scores(g, attacker)
        n_rm = min(batch_size, g.vcount())
        g.delete_vertices(np.argsort(scores)[::-1][:n_rm].tolist())
        removed += n_rm

        sizes = g.connected_components().sizes()
        s_arr[step + 1] = max(sizes) / n_orig if sizes else 0.0
        chi_arr[step + 1] = susceptibility(g)
        tau_arr[step + 1] = kendall_tau_centralities(g)
        f_arr[step + 1] = removed / n_orig

    return {'s': s_arr, 'chi': chi_arr, 'tau': tau_arr, 'f_values': f_arr}
