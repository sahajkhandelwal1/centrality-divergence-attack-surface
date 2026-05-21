import pytest
import igraph as ig
from src.networks import make_er, make_ba, make_ws, ER_PARAMS, BA_PARAMS, WS_PARAMS


def test_er_vcount():
    g = make_er(n=100, mean_k=4.0, seed=42)
    assert g.vcount() == 100


def test_er_mean_degree_approx():
    g = make_er(n=1000, mean_k=5.0, seed=42)
    assert abs(2 * g.ecount() / g.vcount() - 5.0) < 0.5


def test_ba_vcount():
    assert make_ba(n=100, m=2, seed=42).vcount() == 100


def test_ba_connected():
    assert make_ba(n=200, m=2, seed=42).is_connected()


def test_ws_vcount():
    assert make_ws(n=100, beta=0.1, k_ws=6, seed=42).vcount() == 100


def test_er_params_has_mean_k():
    assert 'mean_k' in ER_PARAMS and 10 in ER_PARAMS['mean_k']


def test_ba_params_has_m():
    assert 'm' in BA_PARAMS and 2 in BA_PARAMS['m']


def test_ws_params_has_beta_and_k_ws():
    assert 'beta' in WS_PARAMS and 'k_ws' in WS_PARAMS
