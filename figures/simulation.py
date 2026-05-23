#!/usr/bin/env python3
"""
YouTube simulation: BC vs EC targeted attack on WS β=0.05 network.

Scenes
------
  CentralityComparison  (~10 s)  side-by-side node coloring by BC / EC
  AttackSimulation      (~20 s)  simultaneous attack until BC side collapses

Render
------
    cd "Percolation Threshold Across Betweenness and Eigenvector Centrality"
    source venv/bin/activate
    manim -pqh figures/simulation.py CentralityComparison
    manim -pqh figures/simulation.py AttackSimulation
    # or low-quality preview (-ql) for speed
"""

import sys
import random as _random

import numpy as np
import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from manim import *

sys.path.insert(0, ".")
from src.networks import make_ws

# ── Global parameters ─────────────────────────────────────────────────────────
N             = 70       # network size (small enough to see individual nodes)
BETA          = 0.05     # WS rewiring probability — the Δf ≈ 0.38 sweet-spot
K_WS          = 6        # WS base degree (set inside make_ws via WS_PARAMS)
SEED          = 42
TOP_K         = 7        # number of top nodes to circle in Scene 1
COLLAPSE_FRAC = 0.50     # stop Scene 2 when BC giant < this fraction of N
DOT_R         = 0.11     # node dot radius (Manim units)
STEP_T        = 0.55     # seconds per removal step in Scene 2

_PLASMA = plt.get_cmap("plasma")
BC_COL  = ManimColor.from_hex("#e377c2")   # pink  — BC attacker / label
EC_COL  = ManimColor.from_hex("#17becf")   # cyan  — EC attacker / label
BG_COL  = ManimColor.from_hex("#0d1117")   # near-black background


# ── Network helpers ───────────────────────────────────────────────────────────

def _build():
    """Build the fixed WS graph deterministically."""
    _random.seed(SEED)
    np.random.seed(SEED)
    ig.set_random_number_generator(_random.Random(SEED))
    return make_ws(N, BETA, seed=SEED)


def _layout(g, x_off=0.0, scale=2.6):
    """FR layout normalised to [-scale, scale], shifted by x_off."""
    ig.set_random_number_generator(_random.Random(SEED))
    coords = np.array(g.layout("fr").coords)
    coords -= coords.mean(0)
    coords /= np.abs(coords).max() + 1e-12
    coords *= scale
    coords[:, 0] += x_off
    return coords


def _plasma(v):
    """Map scalar v ∈ [0,1] to a Manim colour via the plasma colormap."""
    r, g, b, _ = _PLASMA(float(np.clip(v, 0, 1)))
    return ManimColor.from_rgb((r, g, b))


def _centrality(g, mode):
    if mode == "bc":
        return np.array(g.betweenness(directed=False))
    # EC
    if g.ecount() == 0:
        return np.zeros(g.vcount())
    try:
        return np.array(g.eigenvector_centrality(scale=True))
    except ig.InternalError:
        return np.zeros(g.vcount())


def _precompute(mode):
    """
    Simulate targeted removal (batch = 1) and record:
      {'removed': [original_vertex_id], 'giant_frac': float}
    Index 0 = intact network.
    """
    g      = _build()
    n_orig = g.vcount()
    id_map = list(range(n_orig))

    sizes = g.connected_components().sizes()
    steps = [{"removed": [], "giant_frac": max(sizes) / n_orig}]

    while g.vcount() > 0:
        sc    = _centrality(g, mode)
        li    = int(np.argmax(sc))
        orig  = id_map[li]
        g.delete_vertices(li)
        id_map.pop(li)
        sizes = g.connected_components().sizes()
        gf    = max(sizes) / n_orig if sizes else 0.0
        steps.append({"removed": [orig], "giant_frac": gf})

    return steps


# ── Scene 1: Centrality Comparison ───────────────────────────────────────────

class CentralityComparison(Scene):
    def construct(self):
        self.camera.background_color = BG_COL

        g  = _build()
        bc = np.array(g.betweenness(directed=False))
        ec = np.array(g.eigenvector_centrality(scale=True))
        bc_n = (bc - bc.min()) / (bc.max() - bc.min() + 1e-12)
        ec_n = (ec - ec.min()) / (ec.max() - ec.min() + 1e-12)

        cL = _layout(g, x_off=-3.5)
        cR = _layout(g, x_off=+3.5)

        # ── Titles ────────────────────────────────────────────────────────────
        title   = Text("What Do BC and EC Target?", font_size=34, color=WHITE).to_edge(UP, buff=0.15)
        lbl_bc  = Text("Betweenness Centrality", font_size=21, color=BC_COL).move_to([-3.5, 3.3, 0])
        lbl_ec  = Text("Eigenvector Centrality",  font_size=21, color=EC_COL).move_to([ 3.5, 3.3, 0])
        divider = Line(UP * 3.8, DOWN * 3.8, stroke_color=GRAY, stroke_width=1)
        self.play(FadeIn(title, lbl_bc, lbl_ec, divider), run_time=0.7)

        # ── Static edges ──────────────────────────────────────────────────────
        eg_L = VGroup(*[
            Line([cL[e.source, 0], cL[e.source, 1], 0],
                 [cL[e.target, 0], cL[e.target, 1], 0],
                 stroke_color=GRAY, stroke_width=0.7, stroke_opacity=0.3)
            for e in g.es
        ])
        eg_R = VGroup(*[
            Line([cR[e.source, 0], cR[e.source, 1], 0],
                 [cR[e.target, 0], cR[e.target, 1], 0],
                 stroke_color=GRAY, stroke_width=0.7, stroke_opacity=0.3)
            for e in g.es
        ])

        # ── Nodes — start gray, will animate to plasma ─────────────────────
        nd_L = VGroup(*[Dot([cL[i, 0], cL[i, 1], 0], radius=DOT_R, color=GRAY) for i in range(N)])
        nd_R = VGroup(*[Dot([cR[i, 0], cR[i, 1], 0], radius=DOT_R, color=GRAY) for i in range(N)])

        self.play(FadeIn(eg_L, eg_R, nd_L, nd_R), run_time=0.5)

        # ── Animate nodes to plasma colours ───────────────────────────────────
        self.play(
            *[nd_L[i].animate.set_color(_plasma(bc_n[i])) for i in range(N)],
            *[nd_R[i].animate.set_color(_plasma(ec_n[i])) for i in range(N)],
            run_time=2.0,
        )

        # ── White circles around top-K nodes ─────────────────────────────────
        top_bc = np.argsort(bc_n)[-TOP_K:]
        top_ec = np.argsort(ec_n)[-TOP_K:]
        circles = VGroup(*[
            Circle(radius=DOT_R * 2.3, stroke_color=WHITE, stroke_width=1.8)
            .move_to([cL[i, 0], cL[i, 1], 0])
            for i in top_bc
        ] + [
            Circle(radius=DOT_R * 2.3, stroke_color=WHITE, stroke_width=1.8)
            .move_to([cR[i, 0], cR[i, 1], 0])
            for i in top_ec
        ])
        note_bc = Text("bridge\nnodes", font_size=14, color=LIGHT_GRAY).move_to([-5.8, -2.9, 0])
        note_ec = Text("dense-hub\nnodes",  font_size=14, color=LIGHT_GRAY).move_to([ 1.9, -2.9, 0])

        self.play(Create(circles), FadeIn(note_bc, note_ec), run_time=1.0)
        self.wait(3.0)
        self.play(FadeOut(*self.mobjects), run_time=0.8)


# ── Scene 2: Attack Simulation ────────────────────────────────────────────────

class AttackSimulation(Scene):
    def construct(self):
        self.camera.background_color = BG_COL

        g  = _build()
        cL = _layout(g, x_off=-3.5)
        cR = _layout(g, x_off=+3.5)

        print("Pre-computing BC attack …")
        bc_steps = _precompute("bc")
        print("Pre-computing EC attack …")
        ec_steps = _precompute("ec")

        # BC collapse step = first step where giant < COLLAPSE_FRAC
        collapse = next(
            (i for i, s in enumerate(bc_steps) if s["giant_frac"] < COLLAPSE_FRAC),
            len(bc_steps) - 1,
        )
        print(f"BC collapses at step {collapse}  "
              f"(giant={bc_steps[collapse]['giant_frac']:.2f})")

        # ── Titles ────────────────────────────────────────────────────────────
        title   = Text(r"Targeted Attack: WS β=0.05", font_size=30, color=WHITE).to_edge(UP, buff=0.18)
        lbl_bc  = Text("BC Attacker", font_size=24, color=BC_COL).move_to([-3.5, 3.2, 0])
        lbl_ec  = Text("EC Attacker", font_size=24, color=EC_COL).move_to([ 3.5, 3.2, 0])
        divider = Line(UP * 3.8, DOWN * 3.8, stroke_color=GRAY, stroke_width=1)
        self.play(FadeIn(title, lbl_bc, lbl_ec, divider), run_time=0.7)

        # ── Build node and edge objects ───────────────────────────────────────
        node_L: dict[int, Dot] = {}
        node_R: dict[int, Dot] = {}
        edge_L: dict[tuple, Line] = {}
        edge_R: dict[tuple, Line] = {}
        eg_L = VGroup(); eg_R = VGroup()

        adj: dict[int, set] = {i: set() for i in range(N)}
        for e in g.es:
            s, t = e.source, e.target
            adj[s].add(t); adj[t].add(s)
            key = (min(s, t), max(s, t))
            lobj = Line([cL[s,0], cL[s,1], 0], [cL[t,0], cL[t,1], 0],
                        stroke_color=GRAY, stroke_width=0.7, stroke_opacity=0.35)
            robj = Line([cR[s,0], cR[s,1], 0], [cR[t,0], cR[t,1], 0],
                        stroke_color=GRAY, stroke_width=0.7, stroke_opacity=0.35)
            edge_L[key] = lobj; edge_R[key] = robj
            eg_L.add(lobj);    eg_R.add(robj)

        for i in range(N):
            node_L[i] = Dot([cL[i,0], cL[i,1], 0], radius=DOT_R, color=BC_COL)
            node_R[i] = Dot([cR[i,0], cR[i,1], 0], radius=DOT_R, color=EC_COL)

        ng_L = VGroup(*node_L.values())
        ng_R = VGroup(*node_R.values())
        self.play(FadeIn(eg_L, eg_R, ng_L, ng_R), run_time=0.5)

        # ── Live counters ─────────────────────────────────────────────────────
        ctr_L = Text("Network: 100% intact", font_size=18, color=WHITE).move_to([-3.5, -3.6, 0])
        ctr_R = Text("Network: 100% intact", font_size=18, color=WHITE).move_to([ 3.5, -3.6, 0])
        self.play(FadeIn(ctr_L, ctr_R), run_time=0.3)

        faded_L: set[tuple] = set()
        faded_R: set[tuple] = set()

        # ── Attack loop ───────────────────────────────────────────────────────
        for step in range(1, collapse + 1):
            anims = []
            sbc   = bc_steps[step]
            sec   = ec_steps[min(step, len(ec_steps) - 1)]

            # BC side — shrink & fade removed node + its edges
            for vid in sbc["removed"]:
                anims.append(FadeOut(node_L[vid], scale=0.15))
                for nb in adj[vid]:
                    key = (min(vid, nb), max(vid, nb))
                    if key in edge_L and key not in faded_L:
                        anims.append(FadeOut(edge_L[key]))
                        faded_L.add(key)

            # EC side — same
            for vid in sec["removed"]:
                anims.append(FadeOut(node_R[vid], scale=0.15))
                for nb in adj[vid]:
                    key = (min(vid, nb), max(vid, nb))
                    if key in edge_R and key not in faded_R:
                        anims.append(FadeOut(edge_R[key]))
                        faded_R.add(key)

            new_cL = Text(
                f"Network: {sbc['giant_frac']*100:.0f}% intact",
                font_size=18, color=WHITE,
            ).move_to([-3.5, -3.6, 0])
            new_cR = Text(
                f"Network: {sec['giant_frac']*100:.0f}% intact",
                font_size=18, color=WHITE,
            ).move_to([ 3.5, -3.6, 0])

            self.play(
                *anims,
                Transform(ctr_L, new_cL),
                Transform(ctr_R, new_cR),
                run_time=STEP_T,
            )

        # ── Collapse reveal ───────────────────────────────────────────────────
        bc_final = bc_steps[collapse]["giant_frac"]
        ec_final = ec_steps[min(collapse, len(ec_steps) - 1)]["giant_frac"]

        bc_intact = Text(
            f"{bc_final*100:.0f}% intact",
            font_size=54, color=BC_COL, weight=BOLD,
        ).move_to([-3.5, -3.3, 0])
        ec_intact = Text(
            f"{ec_final*100:.0f}% intact",
            font_size=54, color=EC_COL, weight=BOLD,
        ).move_to([3.5, -3.3, 0])

        # Approximate Δf annotation centred between panels
        fc_bc_approx = collapse / N
        fc_ec_approx = next(
            (i for i, s in enumerate(ec_steps) if s["giant_frac"] < COLLAPSE_FRAC),
            len(ec_steps) - 1,
        ) / N
        delta_f = fc_ec_approx - fc_bc_approx
        delta_label = Text(
            rf"Δf ≈ {delta_f:.2f}",
            font_size=26, color=YELLOW,
        ).move_to([0, -3.3, 0])

        self.play(
            FadeOut(ctr_L, ctr_R),
            FadeIn(bc_intact, scale=1.3),
            FadeIn(ec_intact, scale=1.3),
            run_time=1.0,
        )
        self.play(FadeIn(delta_label), run_time=0.6)
        self.wait(3.0)
        self.play(FadeOut(*self.mobjects), run_time=1.0)


if __name__ == "__main__":
    print("Render with:")
    print("  manim -pqh figures/simulation.py CentralityComparison")
    print("  manim -pqh figures/simulation.py AttackSimulation")
    print("Low-quality preview (faster):")
    print("  manim -pql figures/simulation.py AttackSimulation")
