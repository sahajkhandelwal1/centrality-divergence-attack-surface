# Manim Animation — Rendering Guide

The high-quality video files are not stored in this repository (they are large binary files). This guide explains how to reproduce them locally from `figures/simulation.py`.

## What the animation contains

Two standalone scenes, each rendered as a separate MP4:

| Scene | Duration | Description |
|-------|----------|-------------|
| `CentralityComparison` | ~10 s | WS β=0.05 network shown side-by-side; nodes colour from gray → plasma colormap (pink = high BC, cyan = high EC); white circles highlight the top-7 bridge nodes (left) and prestige nodes (right) |
| `AttackSimulation` | ~13 s | Both attackers remove one node per step simultaneously; live "Network: X% intact" counter updates each step; freezes when BC side collapses (~43% intact), EC side still at ~84%; large percentage labels appear below each network |

## Prerequisites

### 1. Activate the project venv

```bash
cd "Percolation Threshold Across Betweenness and Eigenvector Centrality"
source venv/bin/activate
```

### 2. Install system dependencies (macOS)

Manim requires Cairo and pkg-config, which are not installed by pip:

```bash
brew install cairo pkg-config
```

### 3. Install Manim into the venv

```bash
pip install manim
```

Verify the install:

```bash
manim --version
# Manim Community v0.20.x
```

## Rendering

All commands should be run from the repo root with the venv active.

### High quality — 1080p 60 fps (YouTube upload)

```bash
manim -pqh figures/simulation.py CentralityComparison
manim -pqh figures/simulation.py AttackSimulation
```

Output:
```
media/videos/simulation/1080p60/CentralityComparison.mp4
media/videos/simulation/1080p60/AttackSimulation.mp4
```

### Low quality — 480p 15 fps (fast preview)

```bash
manim -pql figures/simulation.py CentralityComparison
manim -pql figures/simulation.py AttackSimulation
```

Output:
```
media/videos/simulation/480p15/CentralityComparison.mp4
media/videos/simulation/480p15/AttackSimulation.mp4
```

The `-p` flag opens the video automatically after rendering. Drop it to render without opening.

## How the simulation works

`simulation.py` is self-contained and uses the same network/attack helpers as the rest of the codebase:

- **Network:** WS β=0.05, N=70, k=6, seed=42 — built via `src/networks.make_ws`
- **Layout:** Fruchterman–Reingold (`igraph`), same seed for both panels so positions are identical
- **Attack sequence:** pre-computed before animation starts; BC and EC each remove one node per step from independent copies of the same graph; centralities are recomputed after every removal
- **Collapse threshold:** BC side "collapses" when the giant component drops below 50% of N (step ~11 out of 70)
- **Colormap:** `matplotlib` plasma, normalised per-panel to [min, max] of that centrality measure

## Gitignore note

`media/videos/simulation/1080p60/` and all `partial_movie_files/` directories are gitignored. Only the 480p15 preview renders are committed. The 1080p renders must be generated locally using the commands above.
