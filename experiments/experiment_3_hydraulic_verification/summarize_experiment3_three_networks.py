"""Consolidate Experiment 3 verifier and search-order results and plot them."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'experiment_3/experiment3_five_models_three_networks'
OUT.mkdir(parents=True, exist_ok=True)
NETWORKS = {'Anytown': 'anytown', 'Net3': 'net3', 'BWN-II': 'bwnii'}
MODELS = {'Qwen2.5-7B': 'qwen25', 'Llama3.1-8B': 'llama31', 'Qwen3-8B': 'qwen3', 'Ministral3-8B': 'ministral3', 'Qwen3.5-9B': 'qwen35'}

def expected_label(true_cause: pd.Series) -> pd.Series:
    return true_cause.replace({'source_head': 'unknown_cause'})

def verifier_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    cause_rows = []
    for network, key in NETWORKS.items():
        path = ROOT / 'results' / f'experiment_3/experiment3_{key}_generalized_verifier_v4' / 'hybrid_v3_results.csv'
        data = pd.read_csv(path).drop_duplicates(['scenario_id', 'observation_seed'])
        expected = expected_label(data['true_cause'])
        accepted = data['predicted_label'].isin(['roughness_error', 'demand_error', 'leak', 'sensor_bias'])
        accepted_correct = data.loc[accepted, 'predicted_label'].eq(data.loc[accepted, 'true_cause'])
        known = data['true_cause'].isin(['roughness_error', 'demand_error', 'leak', 'sensor_bias'])
        unknown = data['true_cause'].eq('source_head')
        no_fault = data['true_cause'].eq('no_fault')
        summaries.append({'network': network, 'cases': len(data), 'exact_outcome_accuracy': data['predicted_label'].eq(expected).mean(), 'known_family_accuracy': data.loc[known, 'predicted_label'].eq(data.loc[known, 'true_cause']).mean(), 'unknown_cause_recall': data.loc[unknown, 'predicted_label'].eq('unknown_cause').mean(), 'no_fault_recall': data.loc[no_fault, 'predicted_label'].eq('no_fault').mean(), 'accepted_diagnosis_precision': accepted_correct.mean(), 'diagnostic_coverage': accepted.mean(), 'false_attribution_rate': (accepted & data['predicted_label'].ne(data['true_cause'])).mean(), 'mean_full_verdict_simulations': data['logical_hydraulic_simulations'].mean()})
        for cause, group in data.groupby('true_cause'):
            cause_rows.append({'network': network, 'true_cause': cause, 'cases': len(group), 'exact_outcome_accuracy': group['predicted_label'].eq(expected.loc[group.index]).mean()})
    return (pd.DataFrame(summaries), pd.DataFrame(cause_rows))

def ordering_summary() -> pd.DataFrame:
    rows = []
    for network, key in NETWORKS.items():
        for model, model_key in MODELS.items():
            base = ROOT / 'results' / f'experiment_3/experiment3_ordering_{key}_{model_key}'
            summary = pd.read_csv(base / 'search_order_summary.csv')
            ranks = pd.read_csv(base / 'llm_ranking_summary.csv').iloc[0]
            tokens = pd.read_csv(ROOT / 'results' / f'experiment_3/experiment3_rankings_{key}_{model_key}' / 'llm_hypothesis_rankings.csv')
            for row in summary.itertuples(index=False):
                policy_type = 'random' if row.policy == 'Random order' else 'fixed' if row.policy == 'Fixed order' else 'llm'
                rows.append({'network': network, 'model': model if policy_type == 'llm' else '—', 'policy': row.policy, 'policy_type': policy_type, 'provisional_exact_outcome_accuracy': row.exact_outcome_accuracy, 'provisional_accepted_precision': row.accepted_diagnosis_precision, 'provisional_false_attribution_rate': row.false_attribution_rate, 'agreement_with_exhaustive_verdict': row.agreement_with_frozen_final, 'mean_simulations_to_provisional': row.mean_simulations_to_provisional, 'mean_full_verdict_simulations': row.mean_full_verdict_simulations, 'mean_simulation_reduction': row.mean_simulation_reduction, 'mean_family_tests_to_provisional': row.mean_family_tests_to_provisional, 'top1_true_family_recall': ranks.top1_recall if policy_type == 'llm' else np.nan, 'top2_true_family_recall': ranks.top2_recall if policy_type == 'llm' else np.nan, 'mean_true_family_rank': ranks.mean_true_family_rank if policy_type == 'llm' else np.nan, 'mean_prompt_tokens': tokens['prompt_tokens'].mean() if policy_type == 'llm' else np.nan, 'mean_generated_tokens': tokens['generated_tokens'].mean() if policy_type == 'llm' else np.nan})
    result = pd.DataFrame(rows)
    return pd.concat([result[result.policy_type == 'llm'], result[result.policy_type != 'llm'].drop_duplicates(['network', 'policy_type'])], ignore_index=True)

def make_figure(verifier: pd.DataFrame, ordering: pd.DataFrame) -> None:
    exp2 = pd.read_csv(ROOT / 'results' / 'experiment_2/experiment2_three_network_comparison' / 'experiment2_three_network_summary.csv')
    exp2 = exp2[exp2['group'] == 'LLM only']
    fig = plt.figure(figsize=(11.2, 7.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.18])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])
    positions = np.arange(len(NETWORKS))
    jitter = np.linspace(-0.12, 0.12, len(MODELS))
    for j, model in enumerate(MODELS):
        model_data = exp2[exp2['method'] == model].set_index('network')
        vals = [100 * model_data.loc[n, 'exact_outcome_accuracy'] for n in NETWORKS]
        ax1.scatter(positions + jitter[j], vals, s=32, color='#8a8a8a', alpha=0.85, label='LLM only' if j == 0 else None, zorder=3)
    v = verifier.set_index('network')
    vvals = [100 * v.loc[n, 'exact_outcome_accuracy'] for n in NETWORKS]
    ax1.scatter(positions, vvals, s=75, marker='D', color='#2f7d32', label='Hydraulic verifier', zorder=4)
    ax1.set_xticks(positions, NETWORKS)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('Exact-outcome accuracy (%)')
    ax1.set_title('(a) Diagnostic outcome', loc='left', fontweight='bold')
    ax1.legend(frameon=False, loc='upper left')
    metrics = [('Known-family\naccuracy', 'known_family_accuracy'), ('Accepted-diagnosis\nprecision', 'accepted_diagnosis_precision'), ('Unknown-cause\nrecall', 'unknown_cause_recall'), ('No-fault\nrecall', 'no_fault_recall')]
    offsets = np.linspace(-0.18, 0.18, len(NETWORKS))
    markers = ['o', 's', 'D']
    for i, network in enumerate(NETWORKS):
        values = [100 * v.loc[network, col] for _, col in metrics]
        ax2.scatter(np.arange(len(metrics)) + offsets[i], values, s=50, marker=markers[i], label=network, zorder=3)
    ax2.set_xticks(np.arange(len(metrics)), [m[0] for m in metrics])
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Cases (%)')
    ax2.set_title('(b) Verifier reliability', loc='left', fontweight='bold')
    ax2.legend(frameon=False, loc='lower right')
    group_x = np.arange(len(NETWORKS))
    fixed = ordering[ordering.policy_type == 'fixed'].set_index('network')
    random = ordering[ordering.policy_type == 'random'].set_index('network')
    ax3.scatter(group_x - 0.23, [100 * fixed.loc[n, 'mean_simulation_reduction'] for n in NETWORKS], marker='s', s=60, color='#e69f00', label='Fixed order', zorder=4)
    ax3.scatter(group_x, [100 * random.loc[n, 'mean_simulation_reduction'] for n in NETWORKS], marker='_', s=260, linewidths=2.2, color='#5f5f5f', label='Random-order mean', zorder=3)
    model_offsets = np.linspace(-0.09, 0.09, len(MODELS)) + 0.23
    llm = ordering[ordering.policy_type == 'llm']
    for j, model in enumerate(MODELS):
        md = llm[llm.model == model].set_index('network')
        vals = [100 * md.loc[n, 'mean_simulation_reduction'] for n in NETWORKS]
        ax3.scatter(group_x + model_offsets[j], vals, s=38, marker='o', alpha=0.9, label=model, zorder=4)
    ax3.axhline(0, color='#555555', linewidth=0.8)
    ax3.set_xticks(group_x, NETWORKS)
    ax3.set_ylabel('Simulations avoided before provisional result (%)')
    ax3.set_title('(c) Hypothesis-ordering efficiency', loc='left', fontweight='bold')
    ax3.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.14))
    for ax in (ax1, ax2, ax3):
        ax.grid(axis='y', color='#d9d9d9', linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'experiment3_verifier_and_ordering.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'experiment3_verifier_and_ordering.pdf', bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    verifier, cause = verifier_summary()
    ordering = ordering_summary()
    verifier.to_csv(OUT / 'experiment3_verifier_summary.csv', index=False)
    cause.to_csv(OUT / 'experiment3_cause_specific_summary.csv', index=False)
    ordering.to_csv(OUT / 'experiment3_ordering_summary.csv', index=False)
    make_figure(verifier, ordering)
    print('VERIFIER')
    print(verifier.to_string(index=False))
    print('\nLLM ORDERING')
    print(ordering[ordering.policy_type == 'llm'].to_string(index=False))
if __name__ == '__main__':
    main()
