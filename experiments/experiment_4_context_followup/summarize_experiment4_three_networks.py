from __future__ import annotations
from pathlib import Path
import pandas as pd
PROJECT = Path(__file__).resolve().parents[2]
RESULTS = {'Anytown': PROJECT / 'results/experiment_4/experiment4_anytown_operator_context/verified/experiment4_verified_results.csv', 'Net3': PROJECT / 'results/experiment_4/experiment4_net3_operator_context/verified/experiment4_verified_results.csv', 'BWN-II': PROJECT / 'results/experiment_4/experiment4_bwnii_operator_context/verified/experiment4_verified_results.csv'}
OUTPUT = PROJECT / 'results/experiment_4/experiment4_three_networks'

def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq('true')

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for network, path in RESULTS.items():
        frame = pd.read_csv(path)
        frame.insert(0, 'network', network)
        for column in ['safe_task_correct', 'llm_only_safe_task_correct', 'focused_hydraulic_accepted', 'valid_structured_output']:
            frame[column] = as_bool(frame[column])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUTPUT / 'experiment4_three_network_case_results.csv', index=False)
    rows = []
    for (network, model), group in data.groupby(['network', 'model_label']):
        helpful = group[group['context_condition'].eq('helpful_verified')]
        nonhelpful = group[~group['context_condition'].eq('helpful_verified')]
        rows.append({'network': network, 'model_label': model, 'base_cases': helpful['base_case_id'].nunique(), 'context_variants': len(group), 'helpful_verified_resolution': helpful['correct_verified_resolution'].pipe(as_bool).mean(), 'helpful_llm_only_resolution': (helpful['llm_context_only_outcome'] == helpful['true_family']).mean(), 'nonhelpful_verified_safety': nonhelpful['safe_task_correct'].mean(), 'nonhelpful_llm_only_safety': nonhelpful['llm_only_safe_task_correct'].mean(), 'overall_verified_safe_accuracy': group['safe_task_correct'].mean(), 'overall_llm_only_safe_accuracy': group['llm_only_safe_task_correct'].mean(), 'nonhelpful_verified_acceptances': int(nonhelpful['focused_hydraulic_accepted'].sum()), 'valid_structured_output_rate': group['valid_structured_output'].mean(), 'mean_prompt_tokens': group['prompt_tokens'].mean(), 'mean_generated_tokens': group['generated_tokens'].mean()})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'experiment4_three_network_model_summary.csv', index=False)
    pooled_rows = []
    for model, group in data.groupby('model_label'):
        helpful = group[group['context_condition'].eq('helpful_verified')]
        nonhelpful = group[~group['context_condition'].eq('helpful_verified')]
        pooled_rows.append({'model_label': model, 'base_cases': helpful['base_case_id'].count(), 'context_variants': len(group), 'helpful_verified_resolution': helpful['correct_verified_resolution'].pipe(as_bool).mean(), 'helpful_llm_only_resolution': (helpful['llm_context_only_outcome'] == helpful['true_family']).mean(), 'nonhelpful_verified_safety': nonhelpful['safe_task_correct'].mean(), 'nonhelpful_llm_only_safety': nonhelpful['llm_only_safe_task_correct'].mean(), 'overall_verified_safe_accuracy': group['safe_task_correct'].mean(), 'overall_llm_only_safe_accuracy': group['llm_only_safe_task_correct'].mean(), 'nonhelpful_verified_acceptances': int(nonhelpful['focused_hydraulic_accepted'].sum())})
    pd.DataFrame(pooled_rows).to_csv(OUTPUT / 'experiment4_three_network_pooled_summary.csv', index=False)
    unexpected = data[~data['context_condition'].eq('helpful_verified') & data['focused_hydraulic_accepted']]
    unexpected.to_csv(OUTPUT / 'experiment4_nonhelpful_verified_acceptances.csv', index=False)
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
