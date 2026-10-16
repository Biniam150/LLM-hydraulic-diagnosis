from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.shared.evaluate_anytown_causal_benchmark import KNOWN_FAMILIES, generate_perturbed_observations
from experiments.shared.run_hydraulic_diagnosis_demo import DEFAULT_PROFILE, PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, compute_residual_features, generate_pseudo_observations, nominal_prediction, topology_context
from wdn_diagnostics.ollama_client import OllamaClient, extract_json_object
ALLOWED_OUTCOMES = {'leak', 'demand_error', 'roughness_error', 'sensor_bias', 'ambiguous_causes', 'unknown_cause', 'no_fault'}

def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def build_prompt(features: dict, topology: dict) -> tuple[str, str]:
    system_prompt = 'You are performing blind root-cause diagnosis of a water-distribution digital-twin mismatch. Use only the residual summary and network context provided. You have no candidate hydraulic simulations, parameter fits, evidence scores, scenario labels, or ground truth. Do not assume that one of the modeled fault families must be present. Return only one JSON object.'
    prompt_topology = {'junction_count': topology['junction_count'], 'pipe_count': topology['pipe_count'], 'reservoir_count': topology['reservoir_count'], 'pump_count': topology['pump_count'], 'pressure_sensors': topology['pressure_sensors'], 'valid_junctions': topology['valid_junctions'], 'valid_pipes': topology['valid_pipes'], 'sensor_shortest_path_distances': topology['sensor_shortest_path_distances']}
    user_prompt = json.dumps({'task': 'Select the most defensible diagnostic outcome. Choose a cause family only when the supplied patterns support it. Use ambiguous_causes when multiple listed families remain plausible, unknown_cause when a meaningful mismatch is not explained by the listed families, and no_fault when deviations appear consistent with measurement noise.', 'modeled_families': {'leak': 'pressure-dependent water loss at one junction', 'demand_error': 'incorrect nodal demand magnitude', 'roughness_error': 'incorrect pipe resistance or roughness', 'sensor_bias': 'persistent offset affecting one pressure sensor'}, 'allowed_outcomes': sorted(ALLOWED_OUTCOMES), 'required_output': {'outcome': 'exactly one allowed outcome', 'candidate_families': 'list of zero or more modeled families; normally one for a single-family outcome', 'target': 'suspected node, pipe, or sensor identifier, or null', 'confidence': 'number from 0 to 1', 'reasoning': 'list containing at most three short statements'}, 'residual_summary': features, 'network_context': prompt_topology}, separators=(',', ':'))
    return (system_prompt, user_prompt)

def validate_response(raw: str) -> dict[str, object]:
    parsed = extract_json_object(raw)
    outcome = str(parsed.get('outcome', '')).strip().lower()
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(f'Unsupported outcome: {outcome!r}')
    families = parsed.get('candidate_families', [])
    if not isinstance(families, list):
        raise ValueError('candidate_families must be a list')
    families = [str(item).strip().lower() for item in families]
    if any((item not in KNOWN_FAMILIES for item in families)):
        raise ValueError(f'Unsupported candidate family: {families}')
    if outcome in KNOWN_FAMILIES:
        families = [outcome]
    confidence = float(parsed.get('confidence', 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError('confidence must be between 0 and 1')
    target = parsed.get('target')
    if target is not None:
        target = str(target).strip() or None
    reasoning = parsed.get('reasoning', [])
    if not isinstance(reasoning, list):
        raise ValueError('reasoning must be a list')
    return {'outcome': outcome, 'candidate_families': families, 'target': target, 'confidence': confidence, 'reasoning': [str(item) for item in reasoning[:3]]}

def main() -> None:
    parser = argparse.ArgumentParser(description='Run blind LLM-only diagnosis on the frozen AnyTown benchmark.')
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument('--inference-seed', type=int, default=424242)
    parser.add_argument('--tiers', nargs='+', choices=['matched_model', 'perturbed_truth_model'], help='Optional subset of design tiers to evaluate.')
    parser.add_argument('--observation-seeds', nargs='+', type=int, help='Optional predeclared observation-seed subset for a replication run.')
    parser.add_argument('--num-predict', type=int, default=512)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding='utf-8'))
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    args.output.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(model=args.model)
    topology = topology_context(inp_path, profile.pressure_sensors)
    rows: list[dict[str, object]] = []
    case_number = 0
    selected_tiers = args.tiers or design['tiers']
    for tier in selected_tiers:
        excluded = set(design.get('hydraulic_preflight', {}).get('excluded_scenarios', {}).get(tier, []))
        for scenario in design['scenarios']:
            if scenario['id'] in excluded:
                continue
            fault = FaultSpec(**scenario['fault'])
            if tier == 'perturbed_truth_model' and fault.fault_type == 'no_fault':
                continue
            expected_label = fault.fault_type if fault.fault_type in KNOWN_FAMILIES else 'no_fault' if fault.fault_type == 'no_fault' else 'unknown_cause'
            observation_seeds = args.observation_seeds or design['seeds']
            for observation_seed in observation_seeds:
                case_number += 1
                neutral_case_id = f'case_{case_number:03d}'
                case_dir = args.output / neutral_case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                result_path = case_dir / 'decision.json'
                if args.resume and result_path.exists():
                    rows.append(json.loads(result_path.read_text(encoding='utf-8')))
                    print(f'reused {neutral_case_id}')
                    continue
                config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(observation_seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)
                if tier == 'matched_model':
                    observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
                else:
                    observed, _, _ = generate_perturbed_observations(inp_path, fault, config, design['perturbed_truth_model'], scenario['id'])
                nominal = nominal_prediction(inp_path, config)
                _, features = compute_residual_features(observed, nominal, config)
                system_prompt, user_prompt = build_prompt(features, topology)
                write_json(case_dir / 'blind_input.json', {'neutral_case_id': neutral_case_id, 'system_prompt': system_prompt, 'user_prompt': json.loads(user_prompt)})
                raw = client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, seed=args.inference_seed, num_predict=args.num_predict, num_ctx=8192, think=False)
                metadata = dict(client.last_response_metadata)
                (case_dir / 'raw_response.txt').write_text(raw, encoding='utf-8')
                write_json(case_dir / 'ollama_metadata.json', metadata)
                parse_error = None
                try:
                    decision = validate_response(raw)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    parse_error = str(exc)
                    decision = {'outcome': 'invalid_output', 'candidate_families': [], 'target': None, 'confidence': 0.0, 'reasoning': []}
                outcome = str(decision['outcome'])
                families = list(decision['candidate_families'])
                correct = outcome == expected_label
                covered = bool(fault.fault_type in KNOWN_FAMILIES and (outcome == fault.fault_type or (outcome == 'ambiguous_causes' and fault.fault_type in families)))
                target_correct = bool(outcome == fault.fault_type and decision['target'] is not None and (str(decision['target']) == str(fault.target)))
                record = {'neutral_case_id': neutral_case_id, 'tier': tier, 'scenario_id': scenario['id'], 'observation_seed': int(observation_seed), 'inference_seed': args.inference_seed, 'model': args.model, 'true_cause': fault.fault_type, 'true_target': fault.target, 'expected_label': expected_label, 'predicted_label': outcome, 'predicted_families': ','.join(families), 'predicted_target': decision['target'], 'confidence': decision['confidence'], 'reasoning': json.dumps(decision['reasoning']), 'exact_outcome_correct': correct, 'true_family_covered': covered, 'target_correct': target_correct, 'false_roughness_attribution': bool(fault.fault_type != 'roughness_error' and outcome == 'roughness_error'), 'valid_output': parse_error is None, 'parse_error': parse_error, 'hydraulic_hypothesis_tests': 0, 'prompt_tokens': metadata.get('prompt_eval_count'), 'generated_tokens': metadata.get('eval_count'), 'total_tokens': int(metadata.get('prompt_eval_count', 0)) + int(metadata.get('eval_count', 0)), 'inference_duration_s': float(metadata.get('total_duration_ns', 0)) / 1000000000.0}
                rows.append(record)
                write_json(result_path, record)
                print(f"{neutral_case_id}: outcome={outcome} confidence={decision['confidence']:.2f}")
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'llm_only_causal_results.csv', index=False)
    summaries = []
    for tier, group in results.groupby('tier'):
        known = group[group['true_cause'].isin(KNOWN_FAMILIES)]
        nonrough = group[group['true_cause'] != 'roughness_error']
        true_roughness = group[group['true_cause'] == 'roughness_error']
        unknown = group[group['expected_label'] == 'unknown_cause']
        no_fault = group[group['expected_label'] == 'no_fault']
        valid = group[group['valid_output']]
        accepted = valid[valid['predicted_label'].isin(KNOWN_FAMILIES)]
        accepted_correct = accepted['predicted_label'] == accepted['true_cause']
        summaries.append({'tier': tier, 'method': 'llm_only', 'model': args.model, 'runs': len(group), 'exact_outcome_accuracy': float(group['exact_outcome_correct'].mean()), 'known_family_accuracy': float(known['exact_outcome_correct'].mean()), 'known_family_coverage': float(known['true_family_covered'].mean()), 'accepted_known_diagnoses': len(accepted), 'accepted_diagnosis_precision': float(accepted_correct.mean()) if len(accepted) else None, 'false_attribution_rate': float((accepted['predicted_label'] != accepted['true_cause']).sum() / len(group)), 'false_roughness_attribution_rate': float(nonrough['false_roughness_attribution'].mean()), 'true_roughness_sensitivity': float(true_roughness['exact_outcome_correct'].mean()), 'unknown_cause_recall': float(unknown['exact_outcome_correct'].mean()) if len(unknown) else None, 'no_fault_recall': float(no_fault['exact_outcome_correct'].mean()) if len(no_fault) else None, 'target_accuracy_when_family_correct': float(accepted.loc[accepted_correct, 'target_correct'].mean()) if accepted_correct.any() else None, 'valid_output_rate': float(group['valid_output'].mean()), 'mean_confidence': float(valid['confidence'].mean()), 'mean_confidence_correct': float(valid.loc[valid['exact_outcome_correct'], 'confidence'].mean()) if valid['exact_outcome_correct'].any() else None, 'mean_confidence_incorrect': float(valid.loc[~valid['exact_outcome_correct'], 'confidence'].mean()) if (~valid['exact_outcome_correct']).any() else None})
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output / 'llm_only_causal_summary.csv', index=False)
    write_json(args.output / 'llm_only_causal_manifest.json', {'experiment': 'blind_llm_only_causal_diagnosis', 'model': args.model, 'temperature': 0.0, 'inference_seed': args.inference_seed, 'case_runs': len(results), 'tiers': selected_tiers, 'hydraulic_hypothesis_tests': 0, 'deterministic_verification_used': False, 'design': str(args.design.resolve())})
    print(summary_frame.to_string(index=False))
if __name__ == '__main__':
    main()
