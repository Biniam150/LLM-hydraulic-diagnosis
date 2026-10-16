from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
R = Path(__file__).resolve().parents[2]
O = R / 'figures/main'
d = pd.read_csv(R / 'results/sensitivity/experiment3_verifier_sensitivity/full_factorial_sensitivity.csv')
params = [('improvement_floor', 'Minimum improvement, $\\delta$', 0.02), ('margin_floor', 'Separation margin, $\\eta$', 0.07), ('gof_alpha', 'GOF significance level, $\\alpha$', 0.05), ('noise_scale', 'Assumed noise multiplier, $k_\\sigma$', 1.0)]
metrics = [('exact_outcome_accuracy', 'Exact-outcome accuracy', '#627C9D'), ('accepted_diagnosis_precision', 'Accepted-diagnosis precision', '#47A2A9'), ('diagnostic_coverage', 'Diagnostic coverage', '#D69A4E')]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False})
f, axs = plt.subplots(4, 3, figsize=(16, 14), sharey=True)
for row, (key, label, ref) in enumerate(params):
    for col, net in enumerate(['Anytown', 'Net3', 'BWN-II']):
        ax = axs[row, col]
        s = d[d.network.eq(net)]
        assert len(s) == 1296
        levels = sorted(s[key].unique())
        idx = int(np.flatnonzero(np.isclose(levels, ref))[0])
        ax.axvspan(idx - 0.45, idx + 0.45, color='#eee8e2', zorder=0)
        for j, (metric, _, color) in enumerate(metrics):
            groups = [s.loc[np.isclose(s[key], x), metric].dropna().to_numpy() * 100 for x in levels]
            ax.boxplot(groups, positions=np.arange(len(levels)) + (j - 1) * 0.24, widths=0.2, patch_artist=True, manage_ticks=False, boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.65}, medianprops={'color': '#263238', 'linewidth': 1.1}, whiskerprops={'color': color, 'linewidth': 0.8}, capprops={'color': color, 'linewidth': 0.8}, flierprops={'marker': '.', 'markersize': 2, 'markerfacecolor': color, 'markeredgecolor': color, 'alpha': 0.45})
        ax.set_xticks(range(len(levels)), [f'{x:g}' for x in levels])
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_ylim(-4, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlabel(label)
        ax.set_ylabel('Cases (%)' if col == 0 else '')
        ax.grid(axis='y', color='#e4e7e9', lw=0.6)
        ax.set_axisbelow(True)
        if row == 0:
            ax.set_title(f'{net} (n = {(18 if col == 0 else 21)})', fontsize=14, pad=16)
        ax.text(-0.12, 1.03, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes)
handles = [Patch(facecolor=c, alpha=0.65, label=l) for _, l, c in metrics] + [Patch(facecolor='#eee8e2', label='Reference setting')]
f.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.995), ncol=2, frameon=False, fontsize=14)
f.subplots_adjust(left=0.065, right=0.99, top=0.91, bottom=0.055, wspace=0.14, hspace=0.43)
for ext in ['png', 'pdf', 'svg']:
    f.savefig(O / f'verifier_sensitivity_grouped_boxes.{ext}', dpi=300, bbox_inches='tight')
