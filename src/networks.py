import random as _random
import igraph as ig

ER_PARAMS = {'mean_k': [2, 2.5, 3, 4, 5, 6, 8, 10]}
BA_PARAMS = {'m': [1, 2, 3, 4, 5, 7, 10]}
WS_PARAMS = {'beta': [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 'k_ws': 6}


def _seed_igraph(seed: int) -> None:
    rng = _random.Random(seed)
    ig.set_random_number_generator(rng)


def make_er(n: int, mean_k: float, seed: int = None) -> ig.Graph:
    if seed is not None:
        _seed_igraph(seed)
    return ig.Graph.Erdos_Renyi(n=n, p=mean_k / (n - 1), directed=False, loops=False)


def make_ba(n: int, m: int, seed: int = None) -> ig.Graph:
    if seed is not None:
        _seed_igraph(seed)
    return ig.Graph.Barabasi(n=n, m=m, directed=False)


def make_ws(n: int, beta: float, k_ws: int = 6, seed: int = None) -> ig.Graph:
    if seed is not None:
        _seed_igraph(seed)
    return ig.Graph.Watts_Strogatz(dim=1, size=n, nei=k_ws // 2, p=beta)
