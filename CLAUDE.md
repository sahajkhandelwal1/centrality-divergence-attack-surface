# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase studying how BC/EC centrality divergence creates a percolation threshold gap (Δf = f_c(EC) − f_c(BC)) in targeted-attack network dismantling across ER, BA, and WS random graph families. Target publication: IEEE Transactions on Network Science and Engineering (IEEEtran journal format).

## Environment

Always activate the venv before running anything:
```bash
source venv/bin/activate
```

All commands below assume the venv is active and the working directory is the repo root.

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_attack.py -v

# Run a single test by name
python -m pytest tests/test_metrics.py::test_susceptibility_isolated_nodes -v

# Validate implementation before sweeping (Molloy-Reed + Albert 2000 checks)
python scripts/validate.py

# Full sweep — N=1000, 100 realizations, all three models (~4–8 h on 8 cores)
python scripts/run_main.py

# FSS sweep — N ∈ {500, 1000, 2000} on one mid-τ condition per model
python scripts/run_fss.py

# Generate all 7 figures from data/
python figures/plot_all.py

# Compile paper (two passes for cross-references)
cd paper && pdflatex main.tex && pdflatex main.tex
```

## Architecture

### Data flow

```
src/networks.py  →  src/attack.py  →  sweep.py  →  data/*.npz
                         ↑
                  src/metrics.py
                                        data/*.npz  →  figures/plot_all.py
```

`sweep.py` parallelises realizations with joblib. Each `.npz` stores all realization arrays for one `(model, param_value, N)` condition: `s_bc`, `chi_bc`, `tau_bc`, `s_ec`, `chi_ec`, `tau_ec` with shape `(n_real, n_records)`, plus `f_values (n_records,)` and `tau_initial (n_real,)`.

### Key design decisions

**Threshold estimator:** `f_c = argmax(chi_mean)` where `chi_mean` is averaged across *all realizations first*, then argmax is taken. Never average per-realization thresholds — that is biased.

**Attack recording:** Index 0 in every output array is the intact network (f=0). Index `i+1` is after `(i+1) * batch_size` nodes have been removed. Both BC and EC attacks are run on an independent copy of the same graph so `tau[0]` is identical for both (same intact network).

**igraph seeding:** `ig.seed()` was removed in igraph 1.0. Use `ig.set_random_number_generator(random.Random(seed))` — see `src/networks.py:_seed_igraph`. The RNG object must expose `.random()`, `.randint()`, and `.gauss()` (standard `random.Random` satisfies this).

**`clusters()` deprecated:** igraph 1.0 renamed this to `connected_components()`. Use `connected_components()` throughout.

**Eigenvector centrality on degenerate graphs:** `g.eigenvector_centrality()` raises `ig.InternalError` on graphs with no edges or in certain disconnected states. Catch and fall back to uniform scores — see `src/attack.py:_centrality_scores`.

### Sweep parameter constants

Defined in `src/networks.py` as module-level dicts:
- `ER_PARAMS = {'mean_k': [2, 2.5, 3, 4, 5, 6, 8, 10]}`
- `BA_PARAMS = {'m': [1, 2, 3, 4, 5, 7, 10]}`
- `WS_PARAMS = {'beta': [...], 'k_ws': 6}`

### Validation tolerances

Molloy-Reed check uses 7% tolerance (not 5%). N=1000 finite-size effects cause a systematic ~4–5% underestimate of the infinite-N Molloy-Reed limit; this is expected behavior, not a bug.

## Git workflow

Commit after every individual plan step. Push once at the end of each phase with all step commits bundled.
