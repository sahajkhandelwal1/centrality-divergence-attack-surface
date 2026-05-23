# Bridge Nodes vs. Prestige Nodes: BC/EC Performance Gap in Targeted Network Dismantling

A research codebase and paper studying **why betweenness centrality (BC) universally outperforms eigenvector centrality (EC) as a targeted-attack strategy** across three canonical random graph families. The core finding is structural: BC identifies bridge nodes that control inter-cluster flow, while EC identifies prestige nodes embedded in dense neighborhoods. Bridge targeting is more effective for network fragmentation, and the magnitude of the advantage is predicted by how much the two metrics' top-node sets diverge.

---

## Core Question

Both BC and EC are widely used centrality measures in network science. BC counts the fraction of shortest paths that traverse a node; EC scores a node by the centrality of its neighbors recursively. Prior work establishes empirically that BC-guided attacks are more damaging, but the *mechanism* behind this universal advantage has not been formally explained.

This project answers: **what structural property of a network determines how large the BC–EC performance gap is, and why?**

---

## Central Claim

We define the **Centrality Divergence Attack Surface (CDAS)**:

```
Δf(G) = f_c(G, π_EC) − f_c(G, π_BC)
```

where `f_c(G, π)` is the percolation threshold — the fraction of nodes removed before the giant component collapses — under attack strategy `π`. Δf > 0 means BC fragments the network earlier (more efficiently) than EC.

**The mechanism:** BC identifies *bridge nodes* — those lying on cut paths between graph components. EC identifies *prestige nodes* — those embedded in dense, high-degree clusters. Bridge removal collapses inter-component conductance rapidly; prestige removal does not. The Cheeger inequality (Chung 1997) relates node removal to conductance collapse directly.

**The predictor:** Low overlap between the top-50 BC nodes and top-50 EC nodes on the intact network predicts large Δf (r = −0.711, p = 0.0001 across all 23 conditions). The effect is strongest in Watts–Strogatz networks (r = −0.930, p = 0.0008).

---

## Key Results

| Network | Δf Range | Structural Behavior |
|---------|----------|---------------------|
| ER (⟨k⟩ = 2–10) | 0.05–0.08 | Structurally invariant — bridge roles are diffuse in random graphs |
| BA (m = 1–10) | 0.00–0.09 | m=1 degenerate (tree); gap compresses as density grows |
| WS (β = 0.01–1.0) | 0.07–0.38 | **Peaks at β = 0.05 (Δf = 0.38)**; dissolves as topology randomizes |

The WS result is the headline. At β = 0.05, the network has enough rewired long-range edges to create a sparse, identifiable set of bridge nodes with markedly elevated BC, while EC remains anchored to local cluster density. As β → 1 the topology randomizes, bridge structure dissolves, and the gap closes.

The BA m=1 case (Δf = 0) is the single exception: tree graphs force both metrics to immediately identify the dominant hub, so despite low aggregate overlap, outcomes are identical.

---

## Novel Contributions

1. **Formal CDAS definition** — signed threshold gap Δf as a quantitative property of the network-attacker pair
2. **Structural mechanism** — bridge vs. prestige targeting, grounded in the Cheeger inequality and cut-path theory
3. **Bridge–prestige overlap predictor** — Ω₅₀ = |top-50 BC ∩ top-50 EC| / 50 on the intact graph predicts Δf with r = −0.711 (p = 0.0001)
4. **WS sweet spot at β = 0.05** — non-monotone Δf behavior explained mechanistically; the gap peaks before full randomization
5. **BA tree degeneracy** — m=1 identified as a structural special case where the mechanism breaks down predictably
6. **Honest regression report** — ER R²=0.38 (n.s.), BA R²=0.001 (n.s.), WS R²=0.69 (p=0.011); ER/BA structural invariance is itself a meaningful result
7. **Finite-size scaling** — threshold convergence confirmed at N ∈ {500, 1000, 2000}

---

## Mathematical Framework

| Symbol | Definition |
|--------|------------|
| G = (V, E) | Undirected simple graph, N = \|V\| |
| G_f | Residual graph after removing fraction f of nodes |
| S(f) | Giant component fraction = \|C_max(G_f)\| / N |
| χ(f) | Susceptibility = (1/N) Σ_{C ≠ C_max} \|C\|² |
| f_c(G, π) | Percolation threshold = argmax_f χ̄(f) |
| BC(v) | Betweenness: Σ_{s≠v≠t} σ_st(v) / σ_st |
| EC(v) | Eigenvector centrality: λ⁻¹ Σ_{u∈N(v)} EC(u) |
| τ | Kendall τ between BC and EC rankings on intact graph |
| Δf(G) | CDAS = f_c(G, π_EC) − f_c(G, π_BC) |
| Ω_k | Bridge–prestige overlap = \|T_k^BC ∩ T_k^EC\| / k |
| h(G) | Edge conductance = min_S e(S,S̄) / min(vol(S), vol(S̄)) |

The Cheeger inequality gives: h(G)² / 2 ≤ λ₂ ≤ 2h(G), where λ₂ is the second eigenvalue of the normalized Laplacian. High-BC nodes lie on the edges minimizing h(G); removing them collapses conductance faster than removing high-EC nodes in clusters.

---

## Network Parameter Space

| Model | Parameter | Values | Notes |
|-------|-----------|--------|-------|
| Erdős–Rényi (ER) | ⟨k⟩ | 2, 2.5, 3, 4, 5, 6, 8, 10 | p = ⟨k⟩/(N−1) |
| Barabási–Albert (BA) | m | 1, 2, 3, 4, 5, 7, 10 | Edges per new node |
| Watts–Strogatz (WS) | β | 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 | k_ws = 6 fixed |

Main sweep: N = 1000, 100 realizations per condition (23 conditions total).
FSS sweep: N ∈ {500, 1000, 2000} on ER ⟨k⟩=4, BA m=3, WS β=0.2.

---

## Attack Protocol

Three independent attackers run on separate copies of each graph:

- **BC attacker** (`π_BC`): removes highest-betweenness nodes first
- **EC attacker** (`π_EC`): removes highest-eigenvector-centrality nodes first
- **Random attacker** (`π_rand`): removes nodes in random order (baseline)

All attacks are **batched adaptive**: centralities are recomputed every `batch_size = 10` removals. At each of the 101 checkpoints (f = 0, 0.01, 0.02, ..., 1.0) we record:

- `S(f)` — giant component fraction
- `χ(f)` — finite-size susceptibility
- `τ(f)` — Kendall τ between BC and EC on the residual graph
- `var_EC(f)` — variance of EC score vector (measures EC numerical stability)

**Threshold estimator:** `f_c = argmax χ̄(f)` where `χ̄` is averaged across *all* realizations before the argmax. Per-realization thresholds are not averaged — that estimator is biased.

**igraph note:** `eigenvector_centrality()` must be called without a `directed` argument on undirected graphs; passing `directed=False` raises `ig.InternalError` in igraph ≥ 1.0. The code catches this and falls back to uniform scores.

---

## Data Schema

Each `.npz` file covers one `(model, param, N)` condition with 100 realizations and 101 steps:

| Key | Shape | Description |
|-----|-------|-------------|
| `s_bc` | (100, 101) | Giant fraction — BC attack |
| `chi_bc` | (100, 101) | Susceptibility — BC attack |
| `tau_bc` | (100, 101) | Kendall τ(BC,EC) during BC attack |
| `s_ec` | (100, 101) | Giant fraction — EC attack |
| `chi_ec` | (100, 101) | Susceptibility — EC attack |
| `tau_ec` | (100, 101) | Kendall τ(BC,EC) during EC attack |
| `var_ec` | (100, 101) | Variance of EC score vector at each step |
| `s_rand` | (100, 101) | Giant fraction — random removal |
| `chi_rand` | (100, 101) | Susceptibility — random removal |
| `f_values` | (101,) | Removal fractions (index 0 = intact network) |
| `tau_initial` | (100,) | τ on intact graph = tau_bc[:, 0] |

---

## Paper Figures

| Figure | File | Content |
|--------|------|---------|
| Fig 1 | `fig1_network_viz.pdf` | Six intact networks colored by BC (top row) and EC (bottom row); visually shows bridge vs. prestige node separation |
| Fig 2 | `fig2_rank_scatter.pdf` | BC rank vs. EC rank scatter for three representative networks; τ annotated |
| Fig 3 | `fig3_attack_curves.pdf` | S(f) and χ(f) under BC, EC, and random attacks for six conditions; Δf marked; random baseline |
| Fig 4 | `fig4_main_scatter.pdf` | Δf vs. τ_initial for all 23 conditions; secondary predictor, τ ∈ [0.13, 0.77] |
| Fig A | `fig_a_delta_f_param.pdf` | Δf vs. structural parameter (ER/BA/WS panels); R² annotated; WS shows significant relationship |
| Fig B | `fig_b_overlap.pdf` | Bridge–prestige overlap Ω₅₀ vs. Δf; r = −0.711, p = 0.0001; mechanistic centerpiece |
| Fig 7 | `fig7_fss.pdf` | Finite-size scaling: f_c^BC and f_c^EC vs. N for three conditions; convergence by N=1000 |

---

## Project Structure

```
├── src/
│   ├── networks.py          # ER/BA/WS generators; ER_PARAMS, BA_PARAMS, WS_PARAMS constants
│   ├── metrics.py           # susceptibility χ, Kendall τ between BC/EC, threshold finder
│   └── attack.py            # batched adaptive attack; returns s, chi, tau, var_ec arrays
├── sweep.py                 # joblib parallel sweep; saves one .npz per (model, param, N)
├── scripts/
│   ├── run_main.py          # N=1000, 100 realizations, all 23 conditions (~4–8 h, 8 cores)
│   ├── run_fss.py           # FSS at N ∈ {500, 1000, 2000} for three conditions
│   ├── validate.py          # Molloy-Reed + Albert 2000 replication sanity checks
│   ├── bridge_overlap.py    # Top-50 BC/EC overlap for all conditions → results/overlap.csv
│   ├── extract_f_star.py    # EC degradation onset f* from var_ec → results/f_star.csv
│   ├── calibrate_f_star.py  # Plots relative EC variance to choose threshold ε
│   └── regression_delta_f.py # Linear regression Δf ~ structural param → results/regression.csv
├── figures/
│   ├── plot_all.py          # Generates all 7 paper figures from data/
│   └── viz_network.py       # Standalone BA m=3 N=150 dark-theme visualization
├── tests/
│   ├── test_attack.py       # 10 tests: output shape, S decay, f_values, var_ec, rand mode
│   ├── test_metrics.py      # Susceptibility edge cases, Kendall τ correctness
│   └── test_networks.py     # ER/BA/WS generator properties
├── data/                    # .npz sweep output (large files, not tracked in git)
├── results/
│   ├── overlap.csv          # BC/EC top-50 overlap per condition
│   ├── f_star.csv           # EC degradation onset f* per condition
│   └── regression.csv       # Δf linear regression results per model
└── paper/
    └── main.tex             # IEEEtran journal format; 7 inline figures with [H] placement
```

---

## Setup

```bash
git clone https://github.com/sahajkhandelwal1/centrality-divergence-attack-surface.git
cd centrality-divergence-attack-surface
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10

```
igraph>=0.11      # graph generation, BC, EC, connected components
numpy>=1.26
scipy>=1.12       # kendalltau, linregress
joblib>=1.3       # parallel realizations
matplotlib>=3.8
tqdm>=4.66        # sweep progress bars
pytest>=7.0
```

---

## Reproducing Results

### 1. Validate implementation (5 minutes)

```bash
python scripts/validate.py
```

Checks two things:
- **Molloy–Reed**: random-removal f_c matches `1 − 1/(⟨k²⟩/⟨k⟩ − 1)` within 7% for ER at ⟨k⟩ ∈ {3, 5, 8}. The 7% tolerance accounts for a systematic N=1000 finite-size underestimate of the infinite-N limit (~4–5% expected).
- **Albert et al. 2000 replication**: BC attacker dismantles BA m=3 faster than random removal (f_c^BC < f_c^rand).

### 2. Run tests (10 seconds)

```bash
python -m pytest tests/ -v
```

27 tests covering attack mechanics, metric correctness, and network generation. All should pass before running the sweep.

### 3. Main sweep (~4–8 hours, 8 cores)

```bash
python scripts/run_main.py
```

Runs 100 realizations × 23 conditions in parallel with joblib. Saves one `.npz` per condition to `data/`. The sweep skips conditions whose `.npz` already exists, so it is safe to resume after interruption.

### 4. FSS sweep (~30–60 minutes)

```bash
python scripts/run_fss.py
```

Runs N ∈ {500, 2000} for ER ⟨k⟩=4, BA m=3, WS β=0.2. Combined with the N=1000 main sweep files, this gives the three-point FSS curves.

### 5. Analysis scripts

```bash
# Main mechanistic analysis (~5 minutes — computes betweenness for 20 realizations × 23 conditions)
python scripts/bridge_overlap.py    # → results/overlap.csv

# Supplementary EC stability analysis
python scripts/calibrate_f_star.py  # → figures/calibrate_f_star.pdf (inspect to verify ε=0.01)
python scripts/extract_f_star.py    # → results/f_star.csv
python scripts/regression_delta_f.py  # → results/regression.csv
```

### 6. Generate all figures

```bash
python figures/plot_all.py
```

Produces 7 PDFs in `figures/`. Requires the main sweep and `bridge_overlap.py` to have completed.

### 7. Compile paper

```bash
cd paper && pdflatex main.tex && pdflatex main.tex
```

Two passes are needed for cross-references. On macOS: `brew install --cask mactex-no-gui`.

---

## Running Tests

```bash
# Full suite
python -m pytest tests/ -v

# Single file
python -m pytest tests/test_attack.py -v

# Single test by name
python -m pytest tests/test_attack.py::test_attack_ec_has_var_ec -v
```

Test coverage: attack output shapes and dtypes, S(f) decreases under attack, f_values correctness, `var_ec` present only for EC attacker, random attacker mode, unknown attacker raises `ValueError`, susceptibility on isolated nodes and fully connected graphs, Kendall τ on identical/reversed rankings, network generator degree sequences.

---

## Key Design Decisions

**Why `argmax(χ̄)` and not `mean(argmax(χ))`?**
Per-realization thresholds are dominated by noise at small N. Averaging χ across realizations first smooths the peak and gives a stable estimate. The alternative (averaging thresholds) is biased because the argmax of a noisy function is not the argmax of its mean.

**Why batch_size = 10?**
Full recomputation of BC after every single removal is O(VE) per step and prohibitively expensive at N=1000. Batch size 10 (1% of N) gives a close approximation of the fully adaptive attack while keeping total sweep time tractable (~4–8 hours on 8 cores).

**Why top-50 for the overlap metric?**
5% of N=1000 captures the nodes removed in the first five attack batches — the phase that determines the percolation threshold difference. Results are robust across k ∈ {10, 20, 50, 100}.

**igraph seeding:**
`ig.seed()` was removed in igraph 1.0. Use `ig.set_random_number_generator(random.Random(seed))`. The RNG object must expose `.random()`, `.randint()`, and `.gauss()` — standard `random.Random` satisfies this. See `src/networks.py:_seed_igraph`.

**Eigenvector centrality on degenerate graphs:**
`g.eigenvector_centrality()` raises `ig.InternalError` on graphs with no edges or in certain disconnected states. This is caught and falls back to uniform (zero) scores in `src/attack.py:_centrality_scores`. Do **not** pass `directed=False` — on undirected graphs this triggers the same error in igraph ≥ 1.0.

---

## Citation

```bibtex
@article{khandelwal2025bridge,
  title   = {Bridge Nodes vs.\ Prestige Nodes: Why Betweenness Centrality Outperforms
             Eigenvector Centrality in Targeted Network Dismantling},
  author  = {Khandelwal, Sahaj},
  journal = {IEEE Transactions on Network Science and Engineering},
  year    = {2025},
  note    = {Under review}
}
```

---

## Related Work

- Freeman (1977) — original formulation of betweenness centrality
- Albert, Jeong & Barabási (2000) — error and attack tolerance of complex networks
- Molloy & Reed (1995) — critical random graphs with a given degree sequence
- Barabási & Albert (1999) — emergence of scaling in random networks
- Watts & Strogatz (1998) — collective dynamics of small-world networks
- Girvan & Newman (2002) — community structure and betweenness centrality
- Chung (1997) — Spectral Graph Theory (Cheeger inequality for graphs)
- Newman, Strogatz & Watts (2001) — random graphs with arbitrary degree distributions
- Bonacich (1987) — power and centrality, eigenvector-based measures
