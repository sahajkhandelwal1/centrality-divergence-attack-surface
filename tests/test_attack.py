import pytest
import numpy as np
import igraph as ig
from src.attack import attack


def test_attack_output_shape():
    g = ig.Graph.Erdos_Renyi(n=100, p=0.1)
    r = attack(g, attacker='bc', batch_size=10)
    n_records = 100 // 10 + 1  # steps + initial f=0 record
    assert r['s'].shape == (n_records,)
    assert r['chi'].shape == (n_records,)
    assert r['tau'].shape == (n_records,)
    assert r['f_values'].shape == (n_records,)


def test_attack_s_starts_near_one():
    g = ig.Graph.Barabasi(n=200, m=3)
    r = attack(g, attacker='bc', batch_size=10)
    assert r['s'][0] == pytest.approx(1.0, abs=0.05)


def test_attack_s_decreases():
    g = ig.Graph.Erdos_Renyi(n=200, p=0.15)
    r = attack(g, attacker='bc', batch_size=10)
    assert r['s'][-1] < r['s'][0]


def test_attack_f_values_correct():
    g = ig.Graph.Erdos_Renyi(n=100, p=0.1)
    r = attack(g, attacker='bc', batch_size=10)
    assert r['f_values'][0] == pytest.approx(0.0)
    assert r['f_values'][1] == pytest.approx(0.1)


def test_attack_tau_initial_same_for_bc_and_ec():
    g = ig.Graph.Barabasi(n=100, m=3)
    r_bc = attack(g.copy(), attacker='bc', batch_size=10)
    r_ec = attack(g.copy(), attacker='ec', batch_size=10)
    # Both start from the same intact graph so tau[0] must be identical
    assert r_bc['tau'][0] == pytest.approx(r_ec['tau'][0], abs=1e-10)


def test_attack_unknown_attacker_raises():
    g = ig.Graph.Erdos_Renyi(n=50, p=0.2)
    with pytest.raises(ValueError):
        attack(g, attacker='random')


def test_attack_rand_output_shape():
    g = ig.Graph.Erdos_Renyi(n=100, p=0.1)
    r = attack(g, attacker='rand', batch_size=10)
    assert r['s'].shape == (11,)
    assert r['f_values'].shape == (11,)


def test_attack_ec_has_var_ec():
    g = ig.Graph.Barabasi(n=100, m=3)
    r = attack(g, attacker='ec', batch_size=10)
    assert 'var_ec' in r
    assert r['var_ec'].shape == r['s'].shape


def test_attack_bc_has_no_var_ec():
    g = ig.Graph.Erdos_Renyi(n=100, p=0.1)
    r = attack(g, attacker='bc', batch_size=10)
    assert 'var_ec' not in r


def test_attack_rand_does_not_raise():
    g = ig.Graph.Erdos_Renyi(n=50, p=0.2)
    r = attack(g, attacker='rand', batch_size=10)
    assert r['s'][0] > 0
