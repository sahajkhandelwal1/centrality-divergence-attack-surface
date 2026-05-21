# Percolation Threshold via Centrality Divergence

A research codebase and paper studying how **disagreement between betweenness centrality (BC) and eigenvector centrality (EC)** creates an exploitable gap between two targeted-attack strategies on random networks.

## Core Question

Prior work asks *which* centrality measure guides better attacks. This project asks: **what happens structurally when the two measures disagree about who the critical nodes are?**

## Central Claim

Centrality divergence — quantified by Kendall τ between BC and EC node rankings — creates a topology-dependent gap

**Δf = f_c(EC) − f_c(BC)**

between the percolation thresholds of a BC-attacker and an EC-attacker. The sign and magnitude of this gap is determined by the network model.

## Novel Contributions

1. **Centrality Divergence Attack Surface (CDAS)** — formal definition with mathematical grounding
2. **Empirical mapping of Δf vs. τ** across Erdős–Rényi (ER), Barabási–Albert (BA), and Watts–Strogatz (WS) network families
3. **Asymmetric τ evolution during attack** — τ_BC(t) and τ_EC(t) diverge mid-collapse; the point of maximum divergence precedes the first attacker's threshold by a measurable lag; mechanistic explanation for why Δf exists
4. **τ divergence rate as early-warning signal** — d[τ_BC(t) − τ_EC(t)]/dt peaks before the first threshold
5. **Finite-size scaling** — threshold estimate validation at N ∈ {500, 1000, 2000}

## Mathematical Framework

| Symbol | Definition |
|--------|------------|
| G = (V, E) | Undirected simple graph, n = \|V\| |
| G_f | Residual graph after removing fraction f of nodes |
| S(G, f) | Giant component fraction = \|C_max(G_f)\| / n |
| χ(G, f) | Susceptibility = (1/n) Σ_{C ≠ C_max} \|C\|² |
| f_c(G, π) | Percolation threshold = argmax_f χ̄(f) |
| B(v) | Betweenness centrality = Σ σ_st(v) / σ_st |
| E(v) | Eigenvector centrality: Ax = λ₁x, E(v) = x_v |
| τ(π_B, π_E) | Kendall τ = (C − D) / [n(n−1)/2] |
| Δ(G) | CDAS gap = f_c(G, π_EC) − f_c(G, π_BC) |

## Network Parameter Sweeps

| Model | Parameter | Values |
|-------|-----------|--------|
| ER | ⟨k⟩ | 2, 2.5, 3, 4, 5, 6, 8, 10 |
| BA | m | 1, 2, 3, 4, 5, 7, 10 |
| WS | β (k_WS=6) | 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 |

Main sweep: N=1000, 100 realizations per condition. FSS: N ∈ {500, 1000, 2000} on one mid-τ condition per model.

## Attack Protocol

Two independent attackers (BC-ranker, EC-ranker) perform **batched adaptive attacks**: centralities are recomputed every k=10 node removals. At each step we record:

- Giant component fraction S(f)
- Susceptibility χ(f)
- Kendall τ between BC and EC on the residual graph, tracked separately per attacker

Threshold estimator: f_c = argmax χ̄(f), where χ̄ is averaged across realizations before taking the argmax.

## Paper Figures (7 total)

| Fig | Content | Purpose |
|-----|---------|---------|
| 1 | Network visualizations: nodes colored by BC (top) and EC (bottom), high-τ and low-τ examples per model | Opens the paper visually; shows divergence concept |
| 2 | BC rank vs. EC rank scatter, annotated with τ, one representative network per model | Shows what τ measures concretely |
| 3 | Attack curves S(f) and χ(f) for BC and EC attackers; high-τ and low-τ per model; Δf marked | Shows the gap empirically |
| 4 | **Main result**: Δf vs. τ_initial scatter, all conditions, colored by model | Central result |
| 5 | τ_BC(f) and τ_EC(f) trajectories per model; vertical lines at f_c^BC and f_c^EC | Mechanistic explanation of Δf |
| 6 | τ divergence rate δ(f) per model; peak and f_c(first attacker) marked | **Most novel**: early-warning signal |
| 7 | Finite-size scaling: f_c vs. N per model | Validates threshold estimates |

## Project Structure

```
├── src/
│   ├── networks.py     # ER/BA/WS generators; sweep configs as constants
│   ├── metrics.py      # susceptibility χ, Kendall τ, threshold finder
│   └── attack.py       # batched adaptive attack (k=10); returns S, χ, τ arrays
├── sweep.py            # joblib parallelization; saves .npz per (model, param, N)
├── scripts/
│   ├── validate.py     # Molloy-Reed + Albert 2000 replication checks
│   ├── run_main.py     # N=1000, 100 realizations, all three models
│   └── run_fss.py      # FSS: N ∈ {500, 1000, 2000} on one mid-τ condition per model
├── figures/
│   └── plot_all.py     # generates all 7 figures from data/
├── tests/              # pytest test suite
├── data/               # .npz output (gitignored)
└── paper/
    └── main.tex        # two-column PRE-format paper
```

## Dependencies

```
igraph>=0.11
numpy>=1.26
scipy>=1.12
joblib>=1.3
matplotlib>=3.8
```

Install:
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

## Quickstart

```bash
# 1. Validate implementation (before running full sweep)
python scripts/validate.py

# 2. Run full sweep (~4–8 hours on 8 cores)
python scripts/run_main.py

# 3. Run FSS sweep
python scripts/run_fss.py

# 4. Generate all figures
python figures/plot_all.py

# 5. Compile paper
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Validation Checks

1. **ER Molloy-Reed**: random-removal f_c ≈ 1 − 1/⟨k⟩ within 5% for ⟨k⟩ ∈ {3, 5, 8}
2. **Albert et al. 2000 replication**: BC-attacker dismantles BA (m=3) faster than random removal

## Paper

Target venue: Physical Review E (two-column, revtex4-2 format). See `paper/main.tex`.

---

*Research project — Percolation Threshold Across Betweenness and Eigenvector Centrality*
