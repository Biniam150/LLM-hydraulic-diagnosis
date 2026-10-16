from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[2]
O = R / 'figures/main'
d = pd.read_csv(O / 'experiment2_three_networks_final.csv')
v = pd.read_csv(R / 'results/experiment_3/experiment3_five_models_three_networks/experiment3_verifier_summary.csv')
v['method'] = 'Hydraulic verifier'
d = pd.concat([d, v], ignore_index=True)
r = pd.read_csv(O / 'experiment3_reliability_coverage.csv')
d = d.merge(r, on=['network', 'method'], validate='one_to_one')
methods = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B', 'Majority class', 'Multinomial logistic', 'Random forest', 'Hydraulic verifier']
colors = ['#7DABC9', '#BA82B1', '#6DB9B1', '#C9A569', '#6277B5', '#999999', '#B77B6F', '#CC963D', '#188267']
markers = ['o', 'v', '^', 'P', 'D', 'x', 's', 'h', '*']
nets = ['Anytown', 'Net3', 'BWN-II']
metrics = [('exact_outcome_accuracy', 'Exact-outcome accuracy (%)', 100), ('precision', 'Accepted-diagnosis precision (%)', 1), ('known_family_accuracy', 'Known-family accuracy (%)', 100), ('coverage', 'Diagnostic coverage (%)', 1), ('unknown_cause_recall', 'Unknown-cause recall (%)', 100)]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})
f, axes = plt.subplots(3, 2, figsize=(12, 12))
for j, (key, title, scale) in enumerate(metrics):
    ax = axes.flat[j]
    for i, name in enumerate(methods):
        vals = d[d.method.eq(name)].set_index('network').loc[nets, key] * scale
        ax.plot(range(3), vals, color=colors[i], marker=markers[i], ls='-' if i == 8 else '--' if i < 5 else ':', lw=2.6 if i == 8 else 1.1, markersize=12 if i == 8 else 6, alpha=1 if i == 8 else 0.85, zorder=10 if i == 8 else 2)
    ax.set_xticks(range(3), nets)
    ax.set_xlim(-0.16, 2.16)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(title)
    ax.grid(axis='y', color='#e5e7e9', lw=0.65)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color('#888888')
        sp.set_linewidth(0.8)
    ax.text(-0.12, 1.04, f'({chr(97 + j)})', transform=ax.transAxes, fontsize=12)
axes.flat[5].axis('off')
handles = [Line2D([], [], color=colors[i], marker=markers[i], ls='-' if i == 8 else '--' if i < 5 else ':', lw=2.6 if i == 8 else 1.1, markersize=11 if i == 8 else 7, label=name) for i, name in enumerate(methods)]
f.legend(handles=handles[:5], loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=5, frameon=False, fontsize=12, handlelength=2.2, columnspacing=1.2)
f.legend(handles=handles[5:], loc='upper center', bbox_to_anchor=(0.54, 0.963), ncol=4, frameon=False, fontsize=12, handlelength=2.2, columnspacing=1.8)
f.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.055, wspace=0.29, hspace=0.32)
for ext in ['png', 'pdf', 'svg']:
    f.savefig(O / f'experiment3_network_line_comparison.{ext}', dpi=300, bbox_inches='tight')
