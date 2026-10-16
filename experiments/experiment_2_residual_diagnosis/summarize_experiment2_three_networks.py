from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'experiment_2/experiment2_three_network_comparison'
KNOWN = {'roughness_error', 'demand_error', 'leak', 'sensor_bias'}
ALLOWED = KNOWN | {'ambiguous_causes', 'unknown_cause', 'no_fault'}

def summarize(rows: pd.DataFrame, model: str) -> dict[str, object]:
    known = rows['true_cause'].isin(KNOWN)
    unknown = rows['expected_label'].eq('unknown_cause')
    no_fault = rows['expected_label'].eq('no_fault')
    accepted = rows['predicted_label'].isin(KNOWN)
    correct = rows['predicted_label'].eq(rows['expected_label'])
    return {'model': model, 'cases': len(rows), 'exact_outcome_accuracy': correct.mean(), 'known_family_accuracy': correct[known].mean(), 'accepted_diagnosis_precision': correct[accepted].mean() if accepted.any() else np.nan, 'unknown_cause_recall': correct[unknown].mean(), 'no_fault_recall': correct[no_fault].mean(), 'valid_output_rate': rows['valid_output'].mean()}

def rescore_bwn_llama() -> pd.DataFrame:
    run = ROOT / 'results' / 'experiment_2/experiment2_bwnii_llm_only_llama31'
    records: list[dict[str, object]] = []
    for case_dir in sorted((p for p in run.iterdir() if p.is_dir())):
        decision = json.loads((case_dir / 'decision.json').read_text(encoding='utf-8'))
        if decision['predicted_label'] == 'invalid_output':
            raw = json.loads((case_dir / 'raw_response.txt').read_text(encoding='utf-8'))
            semantic_outcome = str(raw.get('outcome', 'invalid_output'))
            if semantic_outcome in ALLOWED:
                decision['predicted_label'] = semantic_outcome
        records.append(decision)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'bwnii_llama_semantically_scored_results.csv', index=False)
    pd.DataFrame([summarize(frame, 'Llama3.1-8B')]).to_csv(OUT / 'bwnii_llama_semantically_scored_summary.csv', index=False)
    return frame

def standard_row(network: str, method: str, group: str, row: pd.Series) -> dict[str, object]:
    return {'network': network, 'method': method, 'group': group, 'exact_outcome_accuracy': float(row['exact_outcome_accuracy']), 'known_family_accuracy': float(row['known_family_accuracy']), 'unknown_cause_recall': float(row['unknown_cause_recall']), 'no_fault_recall': float(row['no_fault_recall']), 'accepted_diagnosis_precision': float(row['accepted_diagnosis_precision']), 'valid_output_rate': float(row.get('valid_output_rate', 1.0))}

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bwn_llama = rescore_bwn_llama()
    bwn_llama_summary = pd.Series(summarize(bwn_llama, 'Llama3.1-8B'))
    rows: list[dict[str, object]] = []
    any_llm = pd.read_csv(ROOT / 'results' / 'experiment_2/experiment2_anytown_five_models_seed1009' / 'five_model_summary.csv').set_index('model')
    llm_order = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B']
    for model in llm_order:
        rows.append(standard_row('Anytown', model, 'LLM only', any_llm.loc[model]))
    any_cls = pd.read_csv(ROOT / 'results' / 'experiment_2/experiment2_anytown_classical_baselines' / 'classical_baseline_summary.csv').set_index('method')
    for method in ['Multinomial logistic', 'Random forest']:
        rows.append(standard_row('Anytown', method, 'Classical baseline', any_cls.loc[method]))
    net_llm = pd.read_csv(ROOT / 'results' / 'experiment_2/net3_experiments_1_3_current_summary' / 'experiment2_model_summary.csv').set_index('model')
    for source, label in [('qwen2.5:7b', 'Qwen2.5-7B'), ('llama3.1:8b', 'Llama3.1-8B')]:
        rows.append(standard_row('Net3', label, 'LLM only', net_llm.loc[source]))
    for folder, label in [('experiment_2/experiment2_net3_llm_only_qwen3', 'Qwen3-8B'), ('experiment_2/experiment2_net3_llm_only_ministral3', 'Ministral3-8B'), ('experiment_2/experiment2_net3_llm_only_qwen35', 'Qwen3.5-9B')]:
        row = pd.read_csv(ROOT / 'results' / folder / 'llm_only_causal_summary.csv').iloc[0]
        rows.append(standard_row('Net3', label, 'LLM only', row))
    net_cls = pd.read_csv(ROOT / 'results' / 'experiment_2/net3_experiments_1_3_current_summary' / 'experiment2_classical_summary.csv').set_index('method')
    for method in ['Multinomial logistic', 'Random forest']:
        rows.append(standard_row('Net3', method, 'Classical baseline', net_cls.loc[method]))
    qwen = pd.read_csv(ROOT / 'results' / 'experiment_2/experiment2_bwnii_llm_only_qwen25' / 'llm_only_causal_summary.csv').iloc[0]
    rows.append(standard_row('BWN-II', 'Qwen2.5-7B', 'LLM only', qwen))
    rows.append(standard_row('BWN-II', 'Llama3.1-8B', 'LLM only', bwn_llama_summary))
    for folder, label in [('experiment_2/experiment2_bwnii_llm_only_qwen3', 'Qwen3-8B'), ('experiment_2/experiment2_bwnii_llm_only_ministral3', 'Ministral3-8B'), ('experiment_2/experiment2_bwnii_llm_only_qwen35', 'Qwen3.5-9B')]:
        row = pd.read_csv(ROOT / 'results' / folder / 'llm_only_causal_summary.csv').iloc[0]
        rows.append(standard_row('BWN-II', label, 'LLM only', row))
    bwn_cls = pd.read_csv(ROOT / 'results' / 'experiment_2/experiment2_bwnii_classical_baselines' / 'classical_baseline_summary.csv').set_index('method')
    for method in ['Multinomial logistic', 'Random forest']:
        rows.append(standard_row('BWN-II', method, 'Classical baseline', bwn_cls.loc[method]))
    combined = pd.DataFrame(rows)
    combined.to_csv(OUT / 'experiment2_three_network_summary.csv', index=False)
    metrics = [('exact_outcome_accuracy', 'Exact outcome'), ('known_family_accuracy', 'Known family'), ('unknown_cause_recall', 'Unknown-cause recall')]
    methods = llm_order + ['Multinomial logistic', 'Random forest']
    colors = {'LLM only': '#7f7f7f', 'Classical baseline': '#1676a9'}
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 6.6), sharey=True)
    for ax, network in zip(axes, ['Anytown', 'Net3', 'BWN-II']):
        sub = combined[combined.network.eq(network)].set_index('method').loc[methods]
        for y, method in enumerate(methods):
            for x, (metric, _) in enumerate(metrics):
                value = 100 * float(sub.loc[method, metric])
                group = str(sub.loc[method, 'group'])
                size = 55 if value == 0 else 190 + 6.0 * value
                ax.scatter(x, y, s=size, color=colors[group], alpha=0.9, edgecolor='white', linewidth=0.8)
                ax.text(x, y, f'{value:.1f}', ha='center', va='center', fontsize=7.5, color='white' if value > 0 else 'black')
        ax.set_title(network, fontweight='bold', fontsize=11)
        ax.set_xticks(range(len(metrics)), [label for _, label in metrics], rotation=25, ha='right')
        ax.set_xlim(-0.55, len(metrics) - 0.45)
        ax.set_ylim(len(methods) - 0.5, -0.5)
        ax.grid(axis='x', color='#dddddd', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    axes[0].set_yticks(range(len(methods)), methods)
    for ax in axes[1:]:
        ax.tick_params(axis='y', left=False)
    handles = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[g], markeredgecolor='none', markersize=8, label=g) for g in ['LLM only', 'Classical baseline']]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.text(0.5, 0.01, 'Circle area and label indicate cases correctly handled (%)', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93], w_pad=2.0)
    fig.savefig(OUT / 'experiment2_three_network_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / 'experiment2_three_network_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(combined.to_string(index=False))
if __name__ == '__main__':
    main()
