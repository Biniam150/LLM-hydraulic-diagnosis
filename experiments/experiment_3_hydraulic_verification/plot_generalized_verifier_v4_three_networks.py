from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'experiment_3/generalized_verifier_v4_three_networks'

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {'Anytown': ROOT / 'results/experiment_3/experiment3_anytown_generalized_verifier_v4/hybrid_v3_summary.csv', 'Net3': ROOT / 'results/experiment_3/experiment3_net3_generalized_verifier_v4/hybrid_v3_summary.csv', 'BWN-II': ROOT / 'results/experiment_3/experiment3_bwnii_generalized_verifier_v4/hybrid_v3_summary.csv'}
    rows = []
    for network, path in sources.items():
        row = pd.read_csv(path).iloc[0]
        rows.append({'network': network, 'runs': int(row['runs']), 'exact_outcome_accuracy': row['exact_outcome_accuracy'], 'known_family_accuracy': row['known_family_accuracy'], 'detectable_known_family_accuracy': row['detectable_known_family_accuracy'], 'accepted_diagnosis_precision': row['accepted_diagnosis_precision'], 'diagnostic_coverage': row['diagnostic_coverage'], 'unknown_cause_recall': row['unknown_cause_recall'], 'no_fault_recall': row['no_fault_recall'], 'false_attribution_rate': row['false_attribution_rate'], 'mean_logical_hydraulic_simulations': row['mean_logical_hydraulic_simulations']})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'generalized_verifier_three_networks.csv', index=False)
    metrics = ['exact_outcome_accuracy', 'known_family_accuracy', 'accepted_diagnosis_precision', 'diagnostic_coverage', 'unknown_cause_recall', 'no_fault_recall', 'false_attribution_rate']
    labels = ['Exact outcome accuracy', 'Known-family accuracy', 'Accepted-diagnosis precision', 'Diagnostic coverage', 'Unknown-cause recall', 'No-fault recall', 'False-attribution rate']
    colors = {'Anytown': '#4472c4', 'Net3': '#ed7d31', 'BWN-II': '#2e7d32'}
    markers = {'Anytown': 'o', 'Net3': 's', 'BWN-II': 'D'}
    fig, (ax, cost_ax) = plt.subplots(1, 2, figsize=(11.2, 5.1), gridspec_kw={'width_ratios': [2.35, 1.0]}, constrained_layout=True)
    y = np.arange(len(metrics))
    offsets = {'Anytown': -0.16, 'Net3': 0.0, 'BWN-II': 0.16}
    for network in sources:
        row = summary.loc[summary['network'] == network].iloc[0]
        values = 100 * row[metrics].to_numpy(dtype=float)
        ax.scatter(values, y + offsets[network], color=colors[network], marker=markers[network], s=60, label=network, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-4, 104)
    ax.set_xlabel('Cases (%)')
    ax.set_title('(a) Diagnostic performance', fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    costs = summary['mean_logical_hydraulic_simulations'].to_numpy(dtype=float)
    networks = summary['network'].tolist()
    bars = cost_ax.bar(networks, costs, color=[colors[n] for n in networks], width=0.62)
    cost_ax.bar_label(bars, labels=[f'{x:,.0f}' for x in costs], padding=3, fontsize=8)
    cost_ax.set_ylabel('Mean logical hydraulic simulations per case')
    cost_ax.set_title('(b) Search cost', fontweight='bold')
    cost_ax.grid(axis='y', alpha=0.25)
    cost_ax.set_axisbelow(True)
    cost_ax.tick_params(axis='x', rotation=20)
    cost_ax.set_ylim(0, max(costs) * 1.16)
    fig.savefig(OUT / 'generalized_verifier_three_networks.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'generalized_verifier_three_networks.pdf', bbox_inches='tight')
    plt.close(fig)
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
