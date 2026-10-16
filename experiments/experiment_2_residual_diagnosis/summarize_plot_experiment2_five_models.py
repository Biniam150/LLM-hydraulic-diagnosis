from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / 'results' / 'experiment_2/experiment2_anytown_five_models_seed1009'
FIGURE = PROJECT / 'results' / '../figures/main' / 'anytown_experiment2_five_models'
SEED = 1009
KNOWN = {'leak', 'demand_error', 'roughness_error', 'sensor_bias'}
MODELS = {'Qwen2.5-7B': PROJECT / 'results' / 'experiment_2/anytown_causal_llm_only_qwen25', 'Llama3.1-8B': PROJECT / 'results' / 'experiment_2/experiment2_anytown_llama31_8b', 'Qwen3-8B': PROJECT / 'results' / 'experiment_2/experiment2_anytown_qwen3_8b_seed1009', 'Ministral3-8B': PROJECT / 'results' / 'experiment_2/experiment2_anytown_ministral3_8b_seed1009', 'Qwen3.5-9B': PROJECT / 'results' / 'experiment_2/experiment2_anytown_qwen35_9b_seed1009'}
CAUSES = ['roughness_error', 'demand_error', 'leak', 'sensor_bias', 'source_head', 'no_fault']
CAUSE_LABELS = ['Roughness', 'Demand', 'Leakage', 'Sensor bias', 'Unknown\nsource head', 'No fault']

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq('true')

def load_model(label: str, folder: Path) -> pd.DataFrame:
    frame = pd.read_csv(folder / 'llm_only_causal_results.csv')
    frame = frame[(frame['tier'] == 'matched_model') & (frame['observation_seed'].astype(int) == SEED)].copy()
    if len(frame) != 18:
        raise ValueError(f'{label}: expected 18 common-seed cases, found {len(frame)}')
    for column in ['exact_outcome_correct', 'true_family_covered', 'valid_output']:
        frame[column] = as_bool(frame[column])
    for index, row in frame.iterrows():
        metadata_path = folder / str(row['neutral_case_id']) / 'ollama_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        prompt = int(metadata.get('prompt_eval_count', 0))
        generated = int(metadata.get('eval_count', 0))
        frame.loc[index, 'prompt_tokens'] = prompt
        frame.loc[index, 'generated_tokens'] = generated
        frame.loc[index, 'total_tokens'] = prompt + generated
        frame.loc[index, 'inference_duration_s'] = float(metadata.get('total_duration_ns', 0)) / 1000000000.0
    frame['model_label'] = label
    return frame

def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cause_rows = []
    for model, group in frame.groupby('model_label', sort=False):
        known = group[group['true_cause'].isin(KNOWN)]
        accepted = group[group['predicted_label'].isin(KNOWN)]
        accepted_correct = accepted['predicted_label'] == accepted['true_cause']
        unknown = group[group['true_cause'] == 'source_head']
        no_fault = group[group['true_cause'] == 'no_fault']
        correct = group['exact_outcome_correct']
        rows.append({'model': model, 'cases': len(group), 'exact_outcome_accuracy': correct.mean(), 'known_family_accuracy': known['exact_outcome_correct'].mean(), 'known_family_coverage': known['true_family_covered'].mean(), 'accepted_diagnosis_precision': accepted_correct.mean() if len(accepted) else np.nan, 'unknown_cause_recall': unknown['exact_outcome_correct'].mean(), 'no_fault_recall': no_fault['exact_outcome_correct'].mean(), 'valid_output_rate': group['valid_output'].mean(), 'mean_confidence': group['confidence'].mean(), 'mean_confidence_correct': group.loc[correct, 'confidence'].mean(), 'mean_confidence_incorrect': group.loc[~correct, 'confidence'].mean(), 'mean_prompt_tokens': group['prompt_tokens'].mean(), 'mean_generated_tokens': group['generated_tokens'].mean(), 'mean_total_tokens': group['total_tokens'].mean(), 'mean_inference_duration_s': group['inference_duration_s'].mean()})
        for cause in CAUSES:
            subset = group[group['true_cause'] == cause]
            cause_rows.append({'model': model, 'true_cause': cause, 'cases': len(subset), 'correct': int(subset['exact_outcome_correct'].sum()), 'accuracy': subset['exact_outcome_correct'].mean()})
    return (pd.DataFrame(rows), pd.DataFrame(cause_rows))

def plot(summary: pd.DataFrame, by_cause: pd.DataFrame) -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.8, 'axes.labelsize': 9, 'xtick.labelsize': 8.2, 'ytick.labelsize': 8.4, 'legend.fontsize': 8})
    fig = plt.figure(figsize=(11.4, 7.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.02], width_ratios=[1.12, 0.88])
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_tokens = fig.add_subplot(grid[0, 1])
    ax_causes = fig.add_subplot(grid[1, :])
    models = summary['model'].tolist()
    y = np.arange(len(models))
    metric_specs = [('exact_outcome_accuracy', 'Exact outcome accuracy', 'o', '#4C78A8'), ('known_family_coverage', 'Known-family coverage', 's', '#59A14F'), ('accepted_diagnosis_precision', 'Accepted-diagnosis precision', '^', '#F28E2B')]
    for column, label, marker, color in metric_specs:
        values = 100 * summary[column].to_numpy()
        ax_metrics.scatter(values, y, s=58, marker=marker, color=color, label=label, zorder=3)
    ax_metrics.set_yticks(y, models)
    ax_metrics.invert_yaxis()
    ax_metrics.set_xlim(0, 100)
    ax_metrics.set_xlabel('Cases (%)')
    ax_metrics.grid(axis='x', color='#DDDDDD', linewidth=0.7)
    ax_metrics.set_axisbelow(True)
    ax_metrics.legend(frameon=False, ncol=1, loc='lower right')
    ax_metrics.text(0.0, 1.05, '(a) Diagnostic performance', transform=ax_metrics.transAxes, fontweight='bold')
    token_x = summary['mean_generated_tokens'].to_numpy()
    accuracy_y = 100 * summary['exact_outcome_accuracy'].to_numpy()
    colors = ['#4C78A8', '#F28E2B', '#59A14F', '#B279A2', '#E15759']
    ax_tokens.scatter(token_x, accuracy_y, s=72, c=colors)
    offsets = [(-56, 8), (5, -12), (8, 8), (8, -14), (8, 8)]
    for x_value, y_value, label, (dx, dy) in zip(token_x, accuracy_y, models, offsets):
        ax_tokens.annotate(label, (x_value, y_value), xytext=(dx, dy), textcoords='offset points')
    x_pad = max(5.0, 0.08 * (token_x.max() - token_x.min() or token_x.max()))
    ax_tokens.set_xlim(token_x.min() - x_pad, token_x.max() + 4 * x_pad)
    ax_tokens.set_ylim(0, max(35, accuracy_y.max() + 8))
    ax_tokens.set_xlabel('Mean generated tokens per case')
    ax_tokens.set_ylabel('Exact outcome accuracy (%)')
    ax_tokens.grid(color='#DDDDDD', linewidth=0.7)
    ax_tokens.set_axisbelow(True)
    ax_tokens.text(0.0, 1.05, '(b) Output length and accuracy', transform=ax_tokens.transAxes, fontweight='bold')
    for model_index, model in enumerate(models):
        for cause_index, cause in enumerate(CAUSES):
            row = by_cause[(by_cause['model'] == model) & (by_cause['true_cause'] == cause)].iloc[0]
            accuracy = 100 * float(row['accuracy'])
            size = 45 + 7.5 * accuracy
            color = '#2E7D32' if accuracy > 0 else '#D9D9D9'
            ax_causes.scatter(cause_index, model_index, s=size, color=color, edgecolors='white', linewidths=1.0)
            ax_causes.text(cause_index, model_index, f'{accuracy:.0f}%', ha='center', va='center', color='white' if accuracy > 0 else '#555555', fontsize=8)
    ax_causes.set_xticks(range(len(CAUSES)), CAUSE_LABELS)
    ax_causes.set_yticks(range(len(models)), models)
    ax_causes.invert_yaxis()
    ax_causes.set_xlim(-0.55, len(CAUSES) - 0.45)
    ax_causes.set_ylim(len(models) - 0.45, -0.55)
    ax_causes.set_xlabel('Hidden condition')
    ax_causes.set_ylabel('Local LLM')
    ax_causes.set_xticks(np.arange(-0.5, len(CAUSES), 1), minor=True)
    ax_causes.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax_causes.grid(which='minor', color='#DDDDDD', linewidth=0.7)
    ax_causes.tick_params(which='minor', bottom=False, left=False)
    ax_causes.text(0.0, 1.05, '(c) Cause-specific exact accuracy', transform=ax_causes.transAxes, fontweight='bold')
    for ax in [ax_metrics, ax_tokens, ax_causes]:
        ax.spines[['top', 'right']].set_visible(False)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = [load_model(label, folder) for label, folder in MODELS.items()]
    combined = pd.concat(frames, ignore_index=True)
    summary, by_cause = summarize(combined)
    combined.to_csv(OUTPUT / 'common_seed_decisions.csv', index=False)
    summary.to_csv(OUTPUT / 'five_model_summary.csv', index=False)
    by_cause.to_csv(OUTPUT / 'accuracy_by_model_and_cause.csv', index=False)
    plot(summary, by_cause)
    print(summary.to_string(index=False))
    print(f"Saved {FIGURE.with_suffix('.png')}")
if __name__ == '__main__':
    main()
