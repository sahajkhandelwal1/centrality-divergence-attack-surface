#!/usr/bin/env python3
"""Plot EC score variance vs. f for representative conditions — calibrate f* threshold."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

conditions = [
    ('ba', 1,    'BA m=1'),
    ('ba', 3,    'BA m=3'),
    ('ba', 10,   'BA m=10'),
    ('ws', 0.01, r'WS $\beta$=0.01'),
    ('ws', 0.2,  r'WS $\beta$=0.2'),
    ('ws', 1.0,  r'WS $\beta$=1.0'),
    ('er', 2,    r'ER $\langle k\rangle$=2'),
    ('er', 4,    r'ER $\langle k\rangle$=4'),
    ('er', 10,   r'ER $\langle k\rangle$=10'),
]

fig, ax = plt.subplots(figsize=(9, 4))
for model, pv, label in conditions:
    try:
        d = np.load(f'data/{model}_param{pv}_N1000.npz')
    except FileNotFoundError:
        continue
    f  = d['f_values']
    v  = d['var_ec'].mean(axis=0)
    v0 = v[0] + 1e-12
    ax.plot(f, v / v0, label=label, lw=1.2)

ax.axhline(0.01, color='k', ls='--', lw=0.8, label='ε=0.01')
ax.set_xlabel('f')
ax.set_ylabel('Relative EC variance')
ax.set_title('Calibrate ε: where do curves collapse?')
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig('figures/calibrate_f_star.pdf', bbox_inches='tight')
print('Saved figures/calibrate_f_star.pdf — inspect to confirm ε=0.01 is sensible')
