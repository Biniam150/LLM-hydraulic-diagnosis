from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.experiment_2_residual_diagnosis.evaluate_anytown_causal_llm_only import validate_response
from wdn_diagnostics.ollama_client import OllamaClient
KNOWN_FAMILIES = {'roughness_error', 'demand_error', 'leak', 'sensor_bias'}

def build_compact_prompt(features: dict, topology: dict) -> tuple[str, str]:
    system_prompt = 'You are performing blind root-cause diagnosis of a water-distribution digital-twin mismatch. Use only the residual summary and compact network context provided. No candidate simulations, parameter fits, evidence scores, scenario labels, or ground truth are available. Return one JSON object.'
    user_prompt = json.dumps({'task': 'Select the most defensible diagnostic outcome. Choose a cause family only when the patterns support it; otherwise use ambiguous_causes, unknown_cause, or no_fault as appropriate.', 'modeled_families': {'leak': 'pressure-dependent water loss at one junction', 'demand_error': 'incorrect nodal demand magnitude', 'roughness_error': 'incorrect pipe resistance or roughness', 'sensor_bias': 'persistent offset affecting one pressure sensor'}, 'allowed_outcomes': ['ambiguous_causes', 'demand_error', 'leak', 'no_fault', 'roughness_error', 'sensor_bias', 'unknown_cause'], 'required_output': {'outcome': 'exactly one allowed outcome', 'candidate_families': 'zero or more modeled families', 'target': 'suspected component identifier or null', 'confidence': 'number from 0 to 1', 'reasoning': 'list containing at most three short statements'}, 'residual_summary': features, 'network_context': topology}, separators=(',', ':'))
    return (system_prompt, user_prompt)

def main() -> None:
    parser = argparse.ArgumentParser(description='Direct diagnosis from frozen blind inputs.')
    parser.add_argument('--blind-root', type=Path, required=True)
    parser.add_argument('--case-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--inference-seed', type=int, default=424242)
    parser.add_argument('--num-ctx', type=int, default=4096)
    parser.add_argument('--tier', default=None, help='Optional case-index tier to retain (for example, matched_model).')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(args.case_index)
    if args.tier is not None:
        if 'tier' not in cases.columns:
            raise ValueError('--tier was supplied but the case index has no tier column')
        cases = cases.loc[cases['tier'].eq(args.tier)].copy()
        if cases.empty:
            raise ValueError(f'No cases found for tier={args.tier!r}')
    client = OllamaClient(model=args.model)
    rows: list[dict[str, object]] = []
    for case in cases.itertuples():
        case_dir = args.output / case.neutral_case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        decision_path = case_dir / 'decision.json'
        if args.resume and decision_path.exists():
            rows.append(json.loads(decision_path.read_text(encoding='utf-8')))
            print(f'reused {case.neutral_case_id}')
            continue
        blind = json.loads((args.blind_root / case.neutral_case_id / 'blind_input.json').read_text(encoding='utf-8'))['user_prompt']
        system_prompt, user_prompt = build_compact_prompt(blind['residual_summary'], blind['network_context'])
        raw = client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, seed=args.inference_seed, num_predict=512, num_ctx=args.num_ctx, think=False)
        metadata = dict(client.last_response_metadata)
        (case_dir / 'raw_response.txt').write_text(raw, encoding='utf-8')
        (case_dir / 'ollama_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        parse_error = None
        try:
            decision = validate_response(raw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
            decision = {'outcome': 'invalid_output', 'candidate_families': [], 'target': None, 'confidence': 0.0, 'reasoning': []}
        true_cause = str(case.true_cause)
        expected = true_cause if true_cause in KNOWN_FAMILIES else 'no_fault' if true_cause == 'no_fault' else 'unknown_cause'
        outcome = str(decision['outcome'])
        accepted = outcome in KNOWN_FAMILIES
        record = {'neutral_case_id': case.neutral_case_id, 'tier': case.tier, 'scenario_id': case.scenario_id, 'observation_seed': int(case.observation_seed), 'model': args.model, 'true_cause': true_cause, 'true_target': case.true_target, 'expected_label': expected, 'predicted_label': outcome, 'predicted_families': ','.join(decision['candidate_families']), 'predicted_target': decision['target'], 'confidence': decision['confidence'], 'exact_outcome_correct': outcome == expected, 'accepted': accepted, 'false_attribution': accepted and outcome != true_cause, 'valid_output': parse_error is None, 'parse_error': parse_error, 'prompt_tokens': metadata.get('prompt_eval_count'), 'generated_tokens': metadata.get('eval_count')}
        decision_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        rows.append(record)
        print(f'{case.neutral_case_id}: {outcome}')
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'llm_only_causal_results.csv', index=False)
    known = results['true_cause'].isin(KNOWN_FAMILIES)
    unknown = results['expected_label'] == 'unknown_cause'
    no_fault = results['expected_label'] == 'no_fault'
    accepted = results['accepted']
    summary = pd.DataFrame([{'model': args.model, 'cases': len(results), 'exact_outcome_accuracy': float(results['exact_outcome_correct'].mean()), 'known_family_accuracy': float(results.loc[known, 'exact_outcome_correct'].mean()), 'known_family_coverage': float(results.loc[known, 'accepted'].mean()), 'accepted_diagnosis_precision': float(results.loc[accepted, 'exact_outcome_correct'].mean()) if accepted.any() else np.nan, 'unknown_cause_recall': float(results.loc[unknown, 'exact_outcome_correct'].mean()), 'no_fault_recall': float(results.loc[no_fault, 'exact_outcome_correct'].mean()), 'false_attribution_rate': float(results['false_attribution'].mean()), 'valid_output_rate': float(results['valid_output'].mean())}])
    summary.to_csv(args.output / 'llm_only_causal_summary.csv', index=False)
    (args.output / 'llm_only_causal_manifest.json').write_text(json.dumps({'model': args.model, 'cases': len(results), 'inference_seed': args.inference_seed, 'temperature': 0.0, 'num_ctx': args.num_ctx, 'hydraulic_verification_used': False, 'ground_truth_in_prompt': False}, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
