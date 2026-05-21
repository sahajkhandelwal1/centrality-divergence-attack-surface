import numpy as np
import igraph as ig
from scipy.stats import kendalltau


def susceptibility(g: ig.Graph) -> float:
    """χ = (1/N) Σ_{C ≠ C_max} |C|²"""
    if g.vcount() == 0:
        return 0.0
    sizes = np.array(g.connected_components().sizes())
    non_giant = np.delete(sizes, int(np.argmax(sizes)))
    return float(np.sum(non_giant ** 2) / g.vcount())


def kendall_tau_centralities(g: ig.Graph) -> float:
    """Kendall τ between betweenness and eigenvector centrality rankings."""
    if g.vcount() < 2:
        return 1.0
    bc = np.array(g.betweenness(directed=False))
    try:
        ec = np.array(g.eigenvector_centrality(directed=False, scale=True))
    except ig.InternalError:
        ec = np.ones(g.vcount()) / g.vcount()
    tau, _ = kendalltau(bc, ec)
    return float(tau) if not np.isnan(tau) else 1.0


def find_threshold(chi_mean: np.ndarray) -> int:
    """Index of the susceptibility peak in the averaged χ̄(f) array."""
    return int(np.argmax(chi_mean))
