#!/usr/bin/env python3
"""Generate all paper figures from data/. Run after sweep completes."""
import os
import sys
import numpy as np
import igraph as ig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import kendalltau as kt

sys.path.insert(0, '.')
from src.networks import make_er, make_ba, make_ws, ER_PARAMS, BA_PARAMS, WS_PARAMS
from src.metrics import find_threshold

DATA_DIR = 'data'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

COL_W = 3.4  # single-column width (inches)
COLORS = {'er': '#1f77b4', 'ba': '#d62728', 'ws': '#2ca02c'}
BC_COLOR = '#e377c2'
EC_COLOR = '#17becf'


def load(model, pv, n):
    return dict(np.load(f'{DATA_DIR}/{model}_param{pv}_N{n}.npz'))


def chi_mean_and_fc(data, attacker):
    chi = data[f'chi_{attacker}'].mean(axis=0)
    f = data['f_values']
    return chi, f[find_threshold(chi)]


# ── Fig 1: Network visualizations ─────────────────────────────────────────────

def fig1():
    conditions = [
        ('er',  8.0,  100, r'ER $\langle k\rangle$=8'),
        ('er',  2.5,  100, r'ER $\langle k\rangle$=2.5'),
        ('ba',  5,    100, 'BA m=5'),
        ('ba',  1,    100, 'BA m=1'),
        ('ws',  0.8,  100, r'WS $\beta$=0.8'),
        ('ws',  0.01, 100, r'WS $\beta$=0.01'),
    ]
    fig, axes = plt.subplots(2, 6, figsize=(COL_W * 4, 3.5))
    for col, (model, pv, n, label) in enumerate(conditions):
        import random as _random
        _random.seed(42)
        ig.set_random_number_generator(_random.Random(42))
        g = (make_er(n, pv, 42) if model == 'er' else
             make_ba(n, int(pv), 42) if model == 'ba' else
             make_ws(n, pv, seed=42))
        bc = np.array(g.betweenness())
        try:
            ec = np.array(g.eigenvector_centrality(scale=True))
        except ig.InternalError:
            ec = np.ones(g.vcount())
        layout = np.array(g.layout('fr').coords)
        for row, scores in enumerate([bc, ec]):
            ax = axes[row, col]
            rng = scores.max() - scores.min()
            norm = (scores - scores.min()) / (rng + 1e-12)
            for e in g.es:
                s, t = e.tuple
                ax.plot([layout[s, 0], layout[t, 0]],
                        [layout[s, 1], layout[t, 1]],
                        'k-', lw=0.3, alpha=0.3, zorder=0)
            sc = ax.scatter(layout[:, 0], layout[:, 1], c=norm, cmap='plasma',
                            s=12, vmin=0, vmax=1, zorder=1)
            if row == 0:
                ax.set_title(label, fontsize=6)
            ax.set_ylabel('BC' if row == 0 else 'EC', fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig1_network_viz.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig1')


# ── Fig 2: Centrality rank scatter ────────────────────────────────────────────

def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.0))
    for ax, (model, pv, label) in zip(axes, [
        ('er', 4.0, r'ER $\langle k\rangle$=4'),
        ('ba', 3,   'BA m=3'),
        ('ws', 0.2, r'WS $\beta$=0.2'),
    ]):
        import random as _random
        _random.seed(0)
        ig.set_random_number_generator(_random.Random(0))
        g = (make_er(200, pv, 0) if model == 'er' else
             make_ba(200, int(pv), 0) if model == 'ba' else
             make_ws(200, pv, seed=0))
        bc = np.array(g.betweenness())
        try:
            ec = np.array(g.eigenvector_centrality(scale=True))
        except ig.InternalError:
            ec = np.ones(g.vcount())
        tau, _ = kt(bc, ec)
        bc_rank = np.argsort(np.argsort(-bc))
        ec_rank = np.argsort(np.argsort(-ec))
        ax.scatter(bc_rank, ec_rank, s=4, alpha=0.5, color=COLORS[model])
        ax.set_xlabel('BC rank', fontsize=6)
        ax.set_ylabel('EC rank', fontsize=6)
        ax.set_title(f'{label}\n' + r'$\tau$=' + f'{tau:.2f}', fontsize=6)
        ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig2_rank_scatter.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig2')


# ── Fig 3: Attack curves ──────────────────────────────────────────────────────

def fig3():
    pairs = {
        'er': [(8,   r'ER $\langle k\rangle$=8 (high $\tau$)'),
               (2.5, r'ER $\langle k\rangle$=2.5 (low $\tau$)')],
        'ba': [(5,   r'BA m=5 (high $\tau$)'),
               (1,   r'BA m=1 (low $\tau$)')],
        'ws': [(0.8, r'WS $\beta$=0.8 (high $\tau$)'),
               (0.01, r'WS $\beta$=0.01 (low $\tau$)')],
    }
    fig, axes = plt.subplots(3, 4, figsize=(COL_W * 3, 5))
    for row, (model, conds) in enumerate(pairs.items()):
        for ci, (pv, label) in enumerate(conds):
            data = load(model, pv, 1000)
            f = data['f_values']
            chi_bc, fc_bc = chi_mean_and_fc(data, 'bc')
            chi_ec, fc_ec = chi_mean_and_fc(data, 'ec')
            ax_s = axes[row, ci * 2]
            ax_c = axes[row, ci * 2 + 1]
            ax_s.plot(f, data['s_bc'].mean(0),   color=BC_COLOR,  lw=1,   label='BC')
            ax_s.plot(f, data['s_ec'].mean(0),   color=EC_COLOR,  lw=1,   label='EC')
            ax_s.plot(f, data['s_rand'].mean(0), color='#888888', lw=0.8, ls='--', label='rand')
            ax_s.set_title(label, fontsize=5)
            ax_s.set_xlabel('f', fontsize=5)
            ax_s.set_ylabel('S/N', fontsize=5)
            ax_s.legend(fontsize=4)
            ax_s.tick_params(labelsize=4)
            ax_c.plot(f, chi_bc, color=BC_COLOR, lw=1)
            ax_c.plot(f, chi_ec, color=EC_COLOR, lw=1)
            ax_c.axvline(fc_bc, color=BC_COLOR, ls='--', lw=0.7)
            ax_c.axvline(fc_ec, color=EC_COLOR, ls='--', lw=0.7)
            ax_c.set_title(r'$\Delta f$=' + f'{fc_ec - fc_bc:.3f}', fontsize=5)
            ax_c.set_xlabel('f', fontsize=5)
            ax_c.set_ylabel(r'$\chi$', fontsize=5)
            ax_c.tick_params(labelsize=4)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig3_attack_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig3')


# ── Fig 4: Main scatter Δf vs. τ ──────────────────────────────────────────────

def fig4():
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    model_params = {
        'er': ER_PARAMS['mean_k'],
        'ba': BA_PARAMS['m'],
        'ws': WS_PARAMS['beta'],
    }
    for model, pvs in model_params.items():
        taus, deltas = [], []
        for pv in pvs:
            try:
                data = load(model, pv, 1000)
            except FileNotFoundError:
                continue
            _, fc_bc = chi_mean_and_fc(data, 'bc')
            _, fc_ec = chi_mean_and_fc(data, 'ec')
            taus.append(float(data['tau_initial'].mean()))
            deltas.append(fc_ec - fc_bc)
        ax.scatter(taus, deltas, color=COLORS[model], label=model.upper(),
                   s=40, edgecolors='k', linewidths=0.4, zorder=3)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel(r'$\tau_{\rm initial}$', fontsize=9)
    ax.set_ylabel(r'$\Delta f = f_c^{\rm EC} - f_c^{\rm BC}$', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig4_main_scatter.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig4')


# ── Fig A: Δf vs structural parameter (3 panels, R² annotated) ────────────────

def fig_a():
    from scipy.stats import linregress
    import csv as _csv
    model_params = {
        'er': (ER_PARAMS['mean_k'], r'$\langle k \rangle$'),
        'ba': (BA_PARAMS['m'],      r'$m$'),
        'ws': (WS_PARAMS['beta'],   r'$\beta$'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.2))
    for ax, (model, (pvs, xlabel)) in zip(axes, model_params.items()):
        xs, ys = [], []
        for pv in pvs:
            try:
                data = load(model, pv, 1000)
            except FileNotFoundError:
                continue
            f = data['f_values']
            fc_bc = float(f[find_threshold(data['chi_bc'].mean(0))])
            fc_ec = float(f[find_threshold(data['chi_ec'].mean(0))])
            xs.append(float(pv))
            ys.append(fc_ec - fc_bc)
        xs, ys = np.array(xs), np.array(ys)
        slope, intercept, r, p, _ = linregress(xs, ys)
        ax.scatter(xs, ys, color=COLORS[model], s=30, edgecolors='k', linewidths=0.4, zorder=3)
        if p < 0.05:
            xfit = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(xfit, intercept + slope * xfit, color=COLORS[model], lw=1, ls='--')
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(r'$\Delta f$', fontsize=7)
        ax.set_title(f'{model.upper()}  $R^2$={r**2:.2f}'
                     + (f'  $p$={p:.3f}' if p < 0.05 else '  n.s.'), fontsize=6)
        ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_a_delta_f_param.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_a')


# ── Fig B: Bridge-prestige overlap vs Δf ──────────────────────────────────────

def fig_b():
    import csv as _csv
    from scipy.stats import linregress, pearsonr
    try:
        rows = list(_csv.DictReader(open('results/overlap.csv')))
    except FileNotFoundError:
        print('fig_b skipped: results/overlap.csv not found — run scripts/bridge_overlap.py first')
        return
    fig, ax = plt.subplots(figsize=(COL_W * 1.2, 2.8))
    all_ov, all_df = [], []
    for model in ['er', 'ba', 'ws']:
        pts = [(float(r['overlap_mean']), float(r['overlap_std']), float(r['delta_f']))
               for r in rows if r['model'] == model]
        if not pts:
            continue
        ovs, stds, dfs = zip(*pts)
        ax.errorbar(ovs, dfs, xerr=stds, fmt='o', color=COLORS[model],
                    label=model.upper(), ms=5, lw=0.7, capsize=2,
                    elinewidth=0.7, zorder=3)
        all_ov.extend(ovs)
        all_df.extend(dfs)
    # overall trend line
    all_ov, all_df = np.array(all_ov), np.array(all_df)
    slope, intercept, r, p, _ = linregress(all_ov, all_df)
    xfit = np.linspace(all_ov.min(), all_ov.max(), 100)
    ax.plot(xfit, intercept + slope * xfit, 'k--', lw=0.9,
            label=f'All: $r$={r:.2f}, $p$={p:.3f}')
    ax.set_xlabel('Top-50 BC/EC node overlap', fontsize=9)
    ax.set_ylabel(r'$\Delta f = f_c^{\rm EC} - f_c^{\rm BC}$', fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig_b_overlap.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig_b')


# ── Fig 7: FSS ────────────────────────────────────────────────────────────────

def fig7():
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 2, 2.2))
    for ax, (model, pv, label) in zip(axes, [
        ('er', 4.0, r'ER $\langle k\rangle$=4'),
        ('ba', 3,   'BA m=3'),
        ('ws', 0.2, r'WS $\beta$=0.2'),
    ]):
        sizes = [500, 1000, 2000]
        fc_bc_list, fc_ec_list = [], []
        for n in sizes:
            try:
                data = load(model, pv, n)
                _, fc_bc = chi_mean_and_fc(data, 'bc')
                _, fc_ec = chi_mean_and_fc(data, 'ec')
            except FileNotFoundError:
                fc_bc = fc_ec = np.nan
            fc_bc_list.append(fc_bc)
            fc_ec_list.append(fc_ec)
        ax.plot(sizes, fc_bc_list, 'o-', color=BC_COLOR, lw=1, label='BC')
        ax.plot(sizes, fc_ec_list, 's-', color=EC_COLOR, lw=1, label='EC')
        ax.set_xlabel('N', fontsize=7)
        ax.set_ylabel(r'$f_c$', fontsize=7)
        ax.set_title(label, fontsize=7)
        ax.legend(fontsize=5)
        ax.tick_params(labelsize=5)
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/fig7_fss.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved fig7')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig_a()
    fig_b()
    fig7()
    print(f'\nAll figures saved to {FIG_DIR}/')
