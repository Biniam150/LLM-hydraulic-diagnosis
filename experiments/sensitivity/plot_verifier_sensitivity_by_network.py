from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[2]
O = R / 'figures/main'
d = pd.read_csv(R / 'results/sensitivity/experiment3_verifier_sensitivity/one_at_a_time_sensitivity.csv')
b = pd.read_csv(R / 'results/experiment_3/experiment3_five_models_three_networks/experiment3_verifier_summary.csv').set_index('network')
params = [('improvement_floor', 'Minimum improvement, $\\delta$', 0.02), ('margin_floor', 'Separation margin, $\\eta$', 0.07), ('gof_alpha', 'GOF significance level, $\\alpha$', 0.05), ('noise_scale', 'Assumed noise multiplier, $k_\\sigma$', 1.0)]
metrics = [('exact_outcome_accuracy', 'Exact-outcome accuracy', '#263D63', 'o'), ('accepted_diagnosis_precision', 'Accepted-diagnosis precision', '#278B96', 'D'), ('diagnostic_coverage', 'Diagnostic coverage', '#C48139', '^')]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
f, axs = plt.subplots(4, 3, figsize=(14, 13), sharey=True)
for row, (key, label, ref) in enumerate(params):
    for col, net in enumerate(['Anytown', 'Net3', 'BWN-II']):
        ax = axs[row, col]
        s = d[d.varied_parameter.eq(key) & d.network.eq(net)].sort_values('varied_value')
        assert len(s) == 6
        for metric, _, color, marker in metrics:
            actual = s.loc[np.isclose(s.varied_value, ref), metric].iloc[0]
            assert np.isclose(actual, b.loc[net, metric], equal_nan=True)
            ax.plot(s.varied_value, s[metric] * 100, color=color, marker=marker, lw=1.7, markersize=5, label=metric)
        ax.axvline(ref, color='#9A4848', ls='--', lw=1.1, zorder=0)
        ax.set_ylim(-4, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(s.varied_value, [f'{x:g}' for x in s.varied_value])
        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Cases (%)' if col == 0 else '')
        ax.grid(axis='y', color='#e4e7e9', lw=0.6)
        ax.set_axisbelow(True)
        if row == 0:
            ax.set_title(f'{net} (n = {(18 if col == 0 else 21)})', fontsize=13, pad=17)
        ax.text(-0.13, 1.03, f'({chr(97 + row * 3 + col)})', transform=ax.transAxes, fontsize=12)
handles = [Line2D([], [], color=c, marker=m, label=l, lw=1.7) for _, l, c, m in metrics]
handles.append(Line2D([], [], color='#9A4848', ls='--', label='Reference setting', lw=1.1))
f.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.997), ncol=2, frameon=False, fontsize=12)
f.subplots_adjust(left=0.075, right=0.99, top=0.91, bottom=0.055, wspace=0.15, hspace=0.43)
for ext in ['png', 'pdf', 'svg']:
    f.savefig(O / f'verifier_sensitivity_by_network.{ext}', dpi=300, bbox_inches='tight')
