from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.shared.evaluate_greedy_multipipe_false_calibration import RoughnessPredictionCache, pressure_mae
from experiments.shared.run_hydraulic_diagnosis_demo import DEFAULT_PROFILE, PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import ALLOWED_HYPOTHESES, load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, apply_physical_fault, compute_residual_features, generate_pseudo_observations, load_hydraulic_network, nominal_prediction, results_to_sensor_frame, run_hydraulics, test_hypothesis_family, validate_observation_config
KNOWN_FAMILIES = set(ALLOWED_HYPOTHESES)

def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256('|'.join(map(str, parts)).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'little')

def add_observation_noise(clean: pd.DataFrame, config: ObservationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    observed = clean.copy()
    pressure_columns = [c for c in observed if c.startswith('pressure_')]
    observed[pressure_columns] += rng.normal(0.0, config.pressure_noise_std_m, size=(len(observed), len(pressure_columns)))
    flow_column = f'flow_{config.flow_link}'
    flow_scale = max(float(clean[flow_column].abs().mean()), 1e-06)
    observed[flow_column] += rng.normal(0.0, config.flow_noise_fraction * flow_scale, size=len(observed))
    return observed

def generate_perturbed_observations(inp_path: Path, fault: FaultSpec, config: ObservationConfig, uncertainty: dict[str, float], scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    network = load_hydraulic_network(inp_path, config.duration_hours, config.demand_model)
    validate_observation_config(network, config)
    rng = np.random.default_rng(stable_seed(uncertainty['seed'], scenario_id))
    demand_adjustments = {}
    for node_name in network.junction_name_list:
        multiplier = float(np.clip(rng.normal(1.0, uncertainty['junction_demand_relative_std']), 0.9, 1.1))
        demand_adjustments[node_name] = multiplier
        junction = network.get_node(node_name)
        for demand in junction.demand_timeseries_list:
            demand.base_value *= multiplier
    roughness_adjustments = {}
    for pipe_name in network.pipe_name_list:
        multiplier = float(np.clip(rng.normal(1.0, uncertainty['pipe_roughness_relative_std']), 0.9, 1.1))
        roughness_adjustments[pipe_name] = multiplier
        network.get_link(pipe_name).roughness *= multiplier
    head_adjustments = {}
    for reservoir_name in network.reservoir_name_list:
        offset = float(rng.normal(0.0, uncertainty['reservoir_head_std_m']))
        head_adjustments[reservoir_name] = offset
        network.get_node(reservoir_name).base_head += offset
    network = apply_physical_fault(network, fault, config.solver)
    clean = results_to_sensor_frame(run_hydraulics(network, config.solver), config)
    if len(clean) != config.duration_hours or not np.isfinite(clean.to_numpy()).all():
        raise RuntimeError(f'Incomplete perturbed truth simulation for {scenario_id}')
    if fault.fault_type == 'sensor_bias':
        column = f'pressure_{fault.target}'
        active = (clean.index >= fault.start_hour) & (clean.index < fault.end_hour)
        clean.loc[active, column] += float(fault.magnitude)
    observed = add_observation_noise(clean, config)
    manifest = {'fault': asdict(fault), 'tier': 'perturbed_truth_model', 'background_uncertainty': uncertainty, 'background_draws': {'demand_multipliers': demand_adjustments, 'roughness_multipliers': roughness_adjustments, 'reservoir_head_offsets_m': head_adjustments}}
    return (observed, clean, manifest)

def baseline_noise_ratios(observed: pd.DataFrame, nominal: pd.DataFrame, config: ObservationConfig, hours: list[int]) -> tuple[float, float]:
    pressure_columns = [c for c in observed if c.startswith('pressure_')]
    flow_column = f'flow_{config.flow_link}'
    pressure_error = observed.loc[hours, pressure_columns] - nominal.loc[hours, pressure_columns]
    pressure_rmse = float(np.sqrt(np.mean(pressure_error.to_numpy() ** 2)))
    flow_error = observed.loc[hours, flow_column] - nominal.loc[hours, flow_column]
    flow_rmse = float(np.sqrt(np.mean(flow_error.to_numpy() ** 2)))
    flow_scale = max(float(nominal.loc[hours, flow_column].abs().mean()), 1e-06)
    return (pressure_rmse / config.pressure_noise_std_m, flow_rmse / flow_scale / config.flow_noise_fraction)

def target_from_evidence(family: str, parameters: dict[str, object]) -> str | None:
    key = {'leak': 'node', 'demand_error': 'node', 'roughness_error': 'pipe', 'sensor_bias': 'sensor'}[family]
    value = parameters.get(key)
    return None if value is None else str(value)

def evidence_gated_decision(evidence_by_family: dict[str, object], baseline_pressure_ratio: float, baseline_flow_ratio: float, strong_noise_limit: float) -> tuple[str, list[str], str]:
    supported = [family for family, evidence in evidence_by_family.items() if evidence.result in {'moderate_support', 'strong_support'}]
    if len(supported) == 1:
        return (supported[0], supported, 'one_supported_family')
    if len(supported) >= 2:
        return ('ambiguous_causes', supported, 'multiple_supported_families')
    if baseline_pressure_ratio <= strong_noise_limit and baseline_flow_ratio <= strong_noise_limit:
        return ('no_fault', [], 'nominal_model_is_noise_consistent')
    return ('unknown_cause', [], 'no_candidate_family_is_supported')

def evaluate_method_row(*, method: str, tier: str, scenario_id: str, seed: int, true_cause: str, true_target: str, expected_label: str, predicted_label: str, predicted_families: list[str], predicted_target: str | None, validation_improvement: float | None, rationale: str) -> dict[str, object]:
    exact = predicted_label == expected_label
    coverage = bool(true_cause in KNOWN_FAMILIES and (predicted_label == true_cause or (predicted_label == 'ambiguous_causes' and true_cause in predicted_families)))
    target_correct = bool(predicted_label == true_cause and predicted_target is not None and (str(predicted_target) == str(true_target)))
    return {'method': method, 'tier': tier, 'scenario_id': scenario_id, 'seed': seed, 'true_cause': true_cause, 'true_target': true_target, 'expected_label': expected_label, 'predicted_label': predicted_label, 'predicted_families': ','.join(predicted_families), 'predicted_target': predicted_target, 'exact_outcome_correct': exact, 'true_family_covered': coverage, 'target_correct': target_correct, 'false_roughness_attribution': bool(true_cause != 'roughness_error' and predicted_label == 'roughness_error'), 'validation_improvement': validation_improvement, 'rationale': rationale}

def main() -> None:
    parser = argparse.ArgumentParser(description='Run the frozen AnyTown causal-diagnosis benchmark.')
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, default=PROJECT_DIR / 'results' / 'candidate_prediction_cache' / 'anytown')
    parser.add_argument('--tiers', nargs='+', choices=['matched_model', 'perturbed_truth_model'], help='Optional tier subset for an independent robustness run.')
    parser.add_argument('--seeds', nargs='+', type=int, help='Optional observation-noise seeds overriding the frozen design.')
    parser.add_argument('--perturbation-seed', type=int, help='Optional background-uncertainty seed for a new holdout draw.')
    parser.add_argument('--skip-incomplete-truth', action='store_true', help='Record and exclude non-convergent truth simulations without replacement.')
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding='utf-8'))
    run_tiers = args.tiers or design['tiers']
    run_seeds = args.seeds or design['seeds']
    perturbation_policy = dict(design['perturbed_truth_model'])
    if args.perturbation_seed is not None:
        perturbation_policy['seed'] = args.perturbation_seed
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    incomplete_truth_cases: list[dict[str, object]] = []
    roughness_cache: RoughnessPredictionCache | None = None
    for tier in run_tiers:
        excluded_scenarios = set(design.get('hydraulic_preflight', {}).get('excluded_scenarios', {}).get(tier, []))
        for scenario in design['scenarios']:
            if scenario['id'] in excluded_scenarios:
                continue
            fault = FaultSpec(**scenario['fault'])
            if tier == 'perturbed_truth_model' and fault.fault_type == 'no_fault':
                continue
            expected_label = fault.fault_type if fault.fault_type in KNOWN_FAMILIES else 'no_fault' if fault.fault_type == 'no_fault' else 'unknown_cause'
            for seed in run_seeds:
                config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=args.candidate_cache)
                if roughness_cache is None:
                    roughness_cache = RoughnessPredictionCache(inp_path, config)
                try:
                    if tier == 'matched_model':
                        observed, clean, manifest = generate_pseudo_observations(inp_path, fault, config)
                        manifest['tier'] = tier
                    else:
                        observed, clean, manifest = generate_perturbed_observations(inp_path, fault, config, perturbation_policy, scenario['id'])
                except RuntimeError as exc:
                    if not args.skip_incomplete_truth:
                        raise
                    incomplete_truth_cases.append({'tier': tier, 'scenario_id': scenario['id'], 'seed': int(seed), 'reason': str(exc)})
                    print(f"excluded incomplete truth {tier} {scenario['id']} seed={seed}")
                    continue
                nominal = nominal_prediction(inp_path, config)
                _, features = compute_residual_features(observed, nominal, config)
                train_hours = [hour for hour in observed.index if int(hour) % 2 == 0]
                validation_hours = [hour for hour in observed.index if int(hour) % 2 == 1]
                baseline_pressure_ratio, baseline_flow_ratio = baseline_noise_ratios(observed, nominal, config, validation_hours)
                evidence_by_family = {}
                prediction_by_family = {}
                for family in ALLOWED_HYPOTHESES:
                    evidence, prediction = test_hypothesis_family(family, inp_path, observed, nominal, config, onset_hour=int(features['estimated_onset_hour']))
                    evidence_by_family[family] = evidence
                    prediction_by_family[family] = prediction
                roughness_candidates = []
                for pipe in roughness_cache.pipe_names:
                    for multiplier in design.get('roughness_multipliers', profile.search_policy.roughness_multipliers):
                        prediction = roughness_cache.predict({pipe: float(multiplier)})
                        score = pressure_mae(observed, prediction, train_hours)
                        roughness_candidates.append((score, pipe, float(multiplier), prediction))
                _, rough_pipe, _, rough_prediction = min(roughness_candidates, key=lambda item: item[0])
                rough_before = pressure_mae(observed, nominal, validation_hours)
                rough_after = pressure_mae(observed, rough_prediction, validation_hours)
                rough_improvement = (rough_before - rough_after) / rough_before
                rough_accept = rough_improvement >= float(design['roughness_only_min_validation_improvement'])
                rows.append(evaluate_method_row(method='roughness_only', tier=tier, scenario_id=scenario['id'], seed=int(seed), true_cause=fault.fault_type, true_target=fault.target, expected_label=expected_label, predicted_label='roughness_error' if rough_accept else 'abstain', predicted_families=['roughness_error'] if rough_accept else [], predicted_target=rough_pipe if rough_accept else None, validation_improvement=rough_improvement, rationale='pressure_only_held_out_improvement_gate'))
                best_family = min(ALLOWED_HYPOTHESES, key=lambda family: evidence_by_family[family].validation_score_after)
                best_evidence = evidence_by_family[best_family]
                rows.append(evaluate_method_row(method='ungated_best_fit', tier=tier, scenario_id=scenario['id'], seed=int(seed), true_cause=fault.fault_type, true_target=fault.target, expected_label=expected_label, predicted_label=best_family, predicted_families=[best_family], predicted_target=target_from_evidence(best_family, best_evidence.best_parameters), validation_improvement=best_evidence.validation_improvement, rationale='lowest_validation_score_without_evidence_gate'))
                gated_label, gated_families, gated_reason = evidence_gated_decision(evidence_by_family, baseline_pressure_ratio, baseline_flow_ratio, profile.evidence_policy.strong_max_pressure_noise_ratio)
                gated_target = None
                gated_improvement = None
                if gated_label in KNOWN_FAMILIES:
                    gated_evidence = evidence_by_family[gated_label]
                    gated_target = target_from_evidence(gated_label, gated_evidence.best_parameters)
                    gated_improvement = gated_evidence.validation_improvement
                rows.append(evaluate_method_row(method='evidence_gated', tier=tier, scenario_id=scenario['id'], seed=int(seed), true_cause=fault.fault_type, true_target=fault.target, expected_label=expected_label, predicted_label=gated_label, predicted_families=gated_families, predicted_target=gated_target, validation_improvement=gated_improvement, rationale=gated_reason))
                if fault.fault_type in KNOWN_FAMILIES:
                    oracle_evidence = evidence_by_family[fault.fault_type]
                    oracle_supported = oracle_evidence.result in {'moderate_support', 'strong_support'}
                    oracle_label = fault.fault_type if oracle_supported else 'unknown_cause'
                    oracle_target = target_from_evidence(fault.fault_type, oracle_evidence.best_parameters) if oracle_supported else None
                    oracle_improvement = oracle_evidence.validation_improvement
                else:
                    oracle_label = expected_label
                    oracle_target = None
                    oracle_improvement = None
                rows.append(evaluate_method_row(method='oracle_family', tier=tier, scenario_id=scenario['id'], seed=int(seed), true_cause=fault.fault_type, true_target=fault.target, expected_label=expected_label, predicted_label=oracle_label, predicted_families=[fault.fault_type] if oracle_label in KNOWN_FAMILIES else [], predicted_target=oracle_target, validation_improvement=oracle_improvement, rationale='correct_family_given_to_family_tester'))
                case_dir = args.output / tier / f"{scenario['id']}_seed_{seed}"
                case_dir.mkdir(parents=True, exist_ok=True)
                write_json(case_dir / 'ground_truth_manifest.json', manifest)
                write_json(case_dir / 'family_evidence.json', {family: asdict(evidence) for family, evidence in evidence_by_family.items()})
                case_rows.append({'tier': tier, 'scenario_id': scenario['id'], 'seed': int(seed), 'true_cause': fault.fault_type, 'expected_label': expected_label, 'baseline_pressure_noise_ratio': baseline_pressure_ratio, 'baseline_flow_noise_ratio': baseline_flow_ratio})
                print(f"completed {tier} {scenario['id']} seed={seed}")
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'causal_benchmark_results.csv', index=False)
    pd.DataFrame(case_rows).to_csv(args.output / 'causal_benchmark_cases.csv', index=False)
    write_json(args.output / 'incomplete_truth_exclusions.json', incomplete_truth_cases)
    summaries = []
    for (tier, method), group in results.groupby(['tier', 'method']):
        known = group[group['true_cause'].isin(KNOWN_FAMILIES)]
        nonrough = group[group['true_cause'] != 'roughness_error']
        true_roughness = group[group['true_cause'] == 'roughness_error']
        unknown = group[group['expected_label'] == 'unknown_cause']
        no_fault = group[group['expected_label'] == 'no_fault']
        correct_family_rows = known[group.loc[known.index, 'predicted_label'] == known['true_cause']]
        summaries.append({'tier': tier, 'method': method, 'runs': len(group), 'exact_outcome_accuracy': float(group['exact_outcome_correct'].mean()), 'known_family_accuracy': float(known['exact_outcome_correct'].mean()), 'known_family_coverage': float(known['true_family_covered'].mean()), 'target_accuracy_when_family_correct': float(correct_family_rows['target_correct'].mean()) if len(correct_family_rows) else None, 'false_roughness_attribution_rate': float(nonrough['false_roughness_attribution'].mean()), 'true_roughness_sensitivity': float(true_roughness['exact_outcome_correct'].mean()), 'unknown_cause_recall': float(unknown['exact_outcome_correct'].mean()) if len(unknown) else None, 'no_fault_recall': float(no_fault['exact_outcome_correct'].mean()) if len(no_fault) else None, 'ambiguity_rate': float((group['predicted_label'] == 'ambiguous_causes').mean()), 'unknown_or_abstention_rate': float(group['predicted_label'].isin(['unknown_cause', 'abstain']).mean())})
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output / 'causal_benchmark_method_summary.csv', index=False)
    summary = {'case_runs': int(len(case_rows)), 'method_decisions': int(len(results)), 'design': design, 'important_interpretation': 'The matched tier tests a closed synthetic setting. The perturbed tier tests modest forward-model mismatch but remains synthetic. The oracle is an upper bound, not a deployable method.'}
    write_json(args.output / 'causal_benchmark_summary.json', summary)
    print(summary_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
