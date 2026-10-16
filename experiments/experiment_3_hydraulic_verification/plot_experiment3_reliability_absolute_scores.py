from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[2]
B = R / 'results'
O = B / '../figures/main'
known = {'leak', 'demand_error', 'roughness_error', 'sensor_bias'}
rows = []

def add(net, name, d):
    a = d.predicted_label.isin(known)
    c = d.predicted_label.eq(d.expected_label)
    rows.append(dict(network=net, method=name, coverage=a.mean() * 100, precision=c[a].mean() * 100 if a.any() else np.nan))
a = pd.read_csv(B / 'experiment_2/experiment2_anytown_five_models_seed1009/common_seed_decisions.csv')
for name, d in a.groupby('model_label'):
    add('Anytown', name, d)
c = pd.read_csv(B / 'experiment_2/experiment2_anytown_classical_baselines/classical_baseline_decisions.csv')
for name, d in c.groupby('method'):
    add('Anytown', name, d)
s = pd.read_csv(B / 'experiment_2/net3_experiments_1_3_current_summary/experiment2_model_summary.csv')
for tag, name in [('qwen2.5:7b', 'Qwen2.5-7B'), ('llama3.1:8b', 'Llama3.1-8B')]:
    r = s[s.model.eq(tag)].iloc[0]
    rows.append(dict(network='Net3', method=name, coverage=r.accepted_known_diagnoses / r.runs * 100, precision=r.accepted_diagnosis_precision * 100))
for net, prefix in [('Net3', 'net3'), ('BWN-II', 'bwnii')]:
    models = [('qwen3', 'Qwen3-8B'), ('ministral3', 'Ministral3-8B'), ('qwen35', 'Qwen3.5-9B')]
    if net == 'BWN-II':
        models += [('qwen25', 'Qwen2.5-7B')]
    for slug, name in models:
        add(net, name, pd.read_csv(B / f'experiment_2/experiment2_{prefix}_llm_only_{slug}/llm_only_causal_results.csv'))
    folder = 'experiment_2/experiment2_net3_classical_baselines_current' if net == 'Net3' else 'experiment_2/experiment2_bwnii_classical_baselines'
    for _, r in pd.read_csv(B / folder / 'classical_baseline_summary.csv').iterrows():
        rows.append(dict(network=net, method=r.method, coverage=r.accepted_diagnosis_rate * 100, precision=r.accepted_diagnosis_precision * 100))
add('BWN-II', 'Llama3.1-8B', pd.read_csv(B / 'experiment_2/experiment2_three_network_comparison/bwnii_llama_semantically_scored_results.csv'))
v = pd.read_csv(B / 'experiment_3/experiment3_five_models_three_networks/experiment3_verifier_summary.csv').set_index('network')
for net, r in v.iterrows():
    rows.append(dict(network=net, method='Hydraulic verifier', coverage=r.diagnostic_coverage * 100, precision=r.accepted_diagnosis_precision * 100))
d = pd.DataFrame(rows)
assert len(d) == 27 and d[['coverage', 'precision']].notna().all().all()
d.to_csv(O / 'experiment3_reliability_coverage.csv', index=False)
methods = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B', 'Majority class', 'Multinomial logistic', 'Random forest', 'Hydraulic verifier']
colors = ['#477BB1', '#8568A2', '#27969C', '#B57D36', '#263D63', '#888888', '#B55B69', '#D29532', '#267B59']
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
f, axs = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={'height_ratios': [1.5, 1]})
for k, net in enumerate(['Anytown', 'Net3', 'BWN-II']):
    ax = axs[0, k]
    sub = d[d.network.eq(net)].set_index('method').loc[methods]
    groups = {}
    for i, (name, r) in enumerate(sub.iterrows()):
        ax.scatter(r.coverage, r.precision, s=135 if i == 8 else 75, marker='s' if i == 8 else 'o' if i < 5 else 'D', color=colors[i], edgecolors='white', lw=0.7, zorder=4 + i / 10)
        groups.setdefault((round(r.coverage, 6), round(r.precision, 6)), []).append(str(i + 1))
    for (x, y), ids in groups.items():
        ax.annotate(','.join(ids), (x, y), xytext=(-9 if x > 90 else 6, 8 if y < 95 else -17), textcoords='offset points', ha='right' if x > 90 else 'left', fontsize=11)
    ax.set_xlim(-5, 107)
    ax.set_ylim(-8, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Diagnostic coverage (%)')
    ax.set_ylabel('Accepted-diagnosis precision (%)' if k == 0 else '')
    ax.grid(color='#e4e7e9', lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(f'{net} (n = {(18 if k == 0 else 21)})', pad=12)
    ax.text(0.5, -0.23, f'({chr(97 + k)})', transform=ax.transAxes, ha='center')
    ax = axs[1, k]
    for i, (key, col) in enumerate(zip(['exact_outcome_accuracy', 'known_family_accuracy', 'unknown_cause_recall'], ['#263D63', '#278B96', '#C48139'])):
        val = v.loc[net, key] * 100
        ax.scatter(val, i, color=col, s=110, marker=['o', 'D', '^'][i])
        ax.annotate(f'{np.floor(val * 10 + 0.5000001) / 10:.1f}%', (val, i), xytext=(-10, 0), textcoords='offset points', ha='right', va='center')
    ax.set_yticks(range(3), ['Exact-outcome\naccuracy', 'Known-family\naccuracy', 'Unknown-cause\nrecall'] if k == 0 else [''] * 3)
    ax.set_ylim(2.6, -0.6)
    ax.set_xlim(-5, 107)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Cases (%)')
    ax.grid(axis='x', color='#e4e7e9', lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title('Hydraulic verifier', fontsize=12)
    ax.text(0.5, -0.28, f'({chr(100 + k)})', transform=ax.transAxes, ha='center')
handles = [Line2D([], [], ls='', marker='s' if i == 8 else 'o' if i < 5 else 'D', color=colors[i], label=f'{i + 1}  {m}', markersize=7) for i, m in enumerate(methods)]
f.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, 1), ncol=3, frameon=False, fontsize=12)
f.subplots_adjust(left=0.14, right=0.98, top=0.8, bottom=0.09, wspace=0.23, hspace=0.62)
for ext in ['png', 'pdf', 'svg']:
    f.savefig(O / f'experiment3_reliability_absolute_scores.{ext}', dpi=300, bbox_inches='tight')
