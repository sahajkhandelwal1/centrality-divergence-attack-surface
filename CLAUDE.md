# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase studying why betweenness centrality (BC) universally outperforms eigenvector centrality (EC) as a targeted-attack strategy, quantified as Δf = f_c(EC) − f_c(BC) across ER, BA, and WS random graph families. The mechanism is structural: BC targets bridge nodes (cut-path controllers); EC targets prestige nodes (dense-neighborhood hubs). Target publication: IEEE Transactions on Network Science and Engineering (IEEEtran journal format).

## Environment

Always activate the venv before running anything:
```bash
source venv/bin/activate
```

All commands below assume the venv is active and the working directory is the repo root.

## Commands

```bash
# Tests
python -m pytest tests/ -v
python -m pytest tests/test_attack.py -v
python -m pytest tests/test_attack.py::test_attack_ec_has_var_ec -v

# Validate before sweeping (Molloy-Reed + Albert 2000 checks)
python scripts/validate.py

# Main sweep — N=1000, 100 realizations, all 23 conditions (~4–8 h on 8 cores)
python scripts/run_main.py

# FSS sweep — N ∈ {500, 2000} for ER ⟨k⟩=4, BA m=3, WS β=0.2
python scripts/run_fss.py

# Analysis (run after main sweep)
python scripts/bridge_overlap.py    # → results/overlap.csv
python scripts/extract_f_star.py    # → results/f_star.csv
python scripts/regression_delta_f.py  # → results/regression.csv

# Figures (run after sweep + analysis)
python figures/plot_all.py

# Paper (two passes for cross-references)
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### Data flow

```
src/networks.py ──► src/attack.py ──► sweep.py ──► data/*.npz ──► figures/plot_all.py
                         ▲
                   src/metrics.py

data/*.npz ──► scripts/bridge_overlap.py ──► results/overlap.csv ──► figures/plot_all.py
data/*.npz ──► scripts/extract_f_star.py ──► results/f_star.csv
data/*.npz ──► scripts/regression_delta_f.py ──► results/regression.csv
```

`sweep.py` parallelises realizations with joblib. `_one_realization` runs BC, EC, and random attacks on the same graph (independent copies), then the outer loop stacks results and saves one `.npz` per `(model, param_value, N)` condition.

### .npz schema

Shape `(n_real=100, n_rec=N//10+1)` per key:

| Key | Description |
|-----|-------------|
| `s_bc`, `chi_bc`, `tau_bc` | Giant fraction, susceptibility, Kendall τ — BC attack |
| `s_ec`, `chi_ec`, `tau_ec` | Same — EC attack |
| `var_ec` | Variance of EC score vector at each step (EC attack only) |
| `s_rand`, `chi_rand` | Giant fraction, susceptibility — random removal |
| `f_values` | `(n_rec,)` removal fractions; index 0 = intact network (f=0) |
| `tau_initial` | `(n_real,)` Kendall τ on intact graph = `tau_bc[:, 0]` |

### Key design decisions

**Threshold estimator:** `f_c = argmax(chi_mean)` where `chi_mean` is averaged across *all realizations first*, then argmax is taken. Never average per-realization thresholds — that is biased.

**Attack recording:** Index 0 is the intact network (f=0). Index `i+1` is after `(i+1) * batch_size` nodes removed. All three attackers run on independent copies of the same graph, so `tau_bc[r,0] == tau_ec[r,0]` for every realization `r`.

**Eigenvector centrality — critical gotcha:** Never pass `directed=False` to `g.eigenvector_centrality()` on an undirected graph. In igraph ≥ 1.0 this raises `ig.InternalError: "Invalid mode"`. Call it as `g.eigenvector_centrality(scale=True)` with no `directed` argument. The function also raises `ig.InternalError` on edgeless or degenerate disconnected graphs — always catch and fall back to zero scores. See `src/attack.py:_centrality_scores` and `src/metrics.py:kendall_tau_centralities`.

**igraph seeding:** `ig.seed()` was removed in igraph 1.0. Use `ig.set_random_number_generator(random.Random(seed))` — see `src/networks.py:_seed_igraph`.

**`clusters()` deprecated:** igraph 1.0 renamed this to `connected_components()`. Use `connected_components()` everywhere.

**FSS file naming:** The main sweep writes integer params where possible (e.g. `er_param4_N1000.npz`), but the FSS sweep may write float params (`er_param4.0_N500.npz`). `fig7()` in `figures/plot_all.py` handles this with a per-N param map; don't assume the N=1000 file and the FSS files share the same param string.

### Sweep parameter constants

Defined in `src/networks.py` as module-level dicts imported by sweep scripts and figure code:
- `ER_PARAMS = {'mean_k': [2, 2.5, 3, 4, 5, 6, 8, 10]}`
- `BA_PARAMS = {'m': [1, 2, 3, 4, 5, 7, 10]}`
- `WS_PARAMS = {'beta': [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 'k_ws': 6}`

### Validation tolerances

Molloy-Reed check uses 7% tolerance (not 5%). N=1000 finite-size effects cause a systematic ~4–5% underestimate of the infinite-N limit; this is expected, not a bug.

### Bridge–prestige overlap analysis

`scripts/bridge_overlap.py` computes Ω₅₀ = |top-50 BC ∩ top-50 EC| / 50 on intact graphs for all 23 conditions (20 realizations each, regenerating graphs from sweep seeds). Results go to `results/overlap.csv`. This is the mechanistic centerpiece of the paper — low overlap predicts large Δf (r = −0.711, p = 0.0001 overall; r = −0.930, p = 0.0008 for WS).

## Git workflow

Commit after every individual plan step. Push once at the end of each phase with all step commits bundled.
