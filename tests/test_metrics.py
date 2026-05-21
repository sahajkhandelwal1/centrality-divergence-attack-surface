import pytest
import numpy as np
import igraph as ig
from src.metrics import susceptibility, kendall_tau_centralities, find_threshold


def test_susceptibility_complete_graph():
    # All nodes in one component → no non-giant clusters → χ = 0
    assert susceptibility(ig.Graph.Full(10)) == pytest.approx(0.0)


def test_susceptibility_isolated_nodes():
    g = ig.Graph(10)  # 10 isolated nodes; each is a component of size 1
    # giant = 1, non-giant = 9 components of size 1 → χ = 9*1/10
    assert susceptibility(g) == pytest.approx(9 / 10)


def test_susceptibility_two_components():
    g = ig.Graph(7)
    g.add_edges([(0, 1), (1, 2), (2, 3)])  # giant size 4, plus 3 isolated nodes
    # non-giant = 3 components of size 1 → χ = 3*1/7
    assert susceptibility(g) == pytest.approx(3 / 7)


def test_susceptibility_empty_graph():
    assert susceptibility(ig.Graph(0)) == pytest.approx(0.0)


def test_kendall_tau_star_graph():
    # Star: center dominates both BC and EC → τ near 1
    g = ig.Graph.Star(11)
    assert kendall_tau_centralities(g) > 0.8


def test_kendall_tau_single_node():
    assert kendall_tau_centralities(ig.Graph(1)) == pytest.approx(1.0)


def test_kendall_tau_in_range():
    g = ig.Graph.Erdos_Renyi(n=50, p=0.1)
    tau = kendall_tau_centralities(g)
    assert -1.0 <= tau <= 1.0


def test_find_threshold_peak_middle():
    assert find_threshold(np.array([0.1, 0.3, 0.8, 0.4, 0.1])) == 2


def test_find_threshold_peak_start():
    assert find_threshold(np.array([1.0, 0.5, 0.2])) == 0
