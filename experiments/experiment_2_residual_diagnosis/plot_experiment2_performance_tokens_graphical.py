from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[2]
p = R / 'figures/main'
d = pd.read_csv(p / 'experiment2_three_networks_final.csv')
t = pd.read_csv(R / 'results/experiment_2/experiment2_three_network_comparison/experiment2_llm_token_summary.csv')
methods = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B', 'Majority class', 'Multinomial logistic', 'Random forest']
nets = ['Anytown', 'Net3', 'BWN-II']
colors = ['#278B96', '#8172B2', '#C48139', '#5476A5', '#263D63']
metrics = [('exact_outcome_accuracy', 'Exact-outcome accuracy', 'o', '#263D63', -0.19), ('known_family_accuracy', 'Known-family accuracy', 'D', '#278B96', 0), ('unknown_cause_recall', 'Unknown-cause recall', '^', '#C48139', 0.19)]
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
f, ax = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={'height_ratios': [1.4, 1]})
for k, net in enumerate(nets):
    a = ax[0, k]
    sub = d[d.network.eq(net)].set_index('method').loc[methods]
    for key, label, marker, color, off in metrics:
        a.scatter(sub[key] * 100, np.arange(8) + off, marker=marker, color=color, s=42, zorder=3)
    a.set_yticks(range(8), methods if k == 0 else [''] * 8)
    a.set_ylim(7.6, -0.6)
    a.set_xlim(-3, 103)
    a.set_xticks([0, 25, 50, 75, 100])
    a.set_xlabel('Cases (%)')
    a.axhline(4.5, color='#999999', lw=0.7)
    a.grid(axis='x', color='#e4e7e9', lw=0.7)
    a.set_axisbelow(True)
    a.set_title(f'{net} (n = {(18 if k == 0 else 21)})', pad=14, fontsize=13)
    a.text(0.5, -0.18, f'({chr(97 + k)})', transform=a.transAxes, ha='center', fontsize=12)
    b = ax[1, k]
    sub = t[t.network.eq(net)].set_index('model').loc[methods[:5]]
    offsets = [(-30, 16), (-12, -24), (-42, -26), (12, -8), (8, 12)]
    for i, (model, row) in enumerate(sub.iterrows()):
        x = row.mean_total_tokens / 1000
        y = row.exact_outcome_accuracy * 100
        b.scatter(x, y, s=65, color=colors[i], edgecolor='white', lw=0.6, zorder=3)
        label = model.replace('-7B', '').replace('-8B', '').replace('-9B', '')
        b.annotate(label, (x, y), xytext=offsets[i], textcoords='offset points', fontsize=10, color=colors[i], arrowprops={'arrowstyle': '-', 'color': colors[i], 'lw': 0.65})
    b.set_xlim(1, 4.1)
    b.set_ylim(0, 65)
    b.set_xticks([1, 2, 3, 4])
    b.set_yticks([0, 20, 40, 60])
    b.grid(color='#e4e7e9', lw=0.7)
    b.set_axisbelow(True)
    b.set_xlabel('Mean total tokens per case (thousands)')
    b.set_ylabel('Exact-outcome accuracy (%)' if k == 0 else '')
    b.text(0.5, -0.27, f'({chr(100 + k)})', transform=b.transAxes, ha='center', fontsize=12)
handles = [Line2D([], [], marker=m, color=c, ls='', label=l, markersize=7) for _, l, m, c, _ in metrics]
f.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.56, 1), ncol=3, frameon=False, fontsize=12)
f.subplots_adjust(left=0.16, right=0.98, top=0.91, bottom=0.09, hspace=0.52, wspace=0.24)
for ext in ['png', 'pdf', 'svg']:
    f.savefig(p / f'experiment2_performance_tokens_graphical.{ext}', dpi=300, bbox_inches='tight')
