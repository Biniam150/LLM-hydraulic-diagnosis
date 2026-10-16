from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from experiments.shared.evaluate_anytown_causal_benchmark import KNOWN_FAMILIES, generate_perturbed_observations
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, _candidate_is_complete, _combined_score, _simulate_candidate, generate_pseudo_observations, load_hydraulic_network, nominal_prediction
SOURCE_HEAD_OFFSETS_M = (-3.2, -2.2, -1.2, -0.6, 0.6, 1.2, 2.2, 3.2)
IMPROVEMENT_FLOOR = 0.02
RELATIVE_MARGIN_FLOOR = 0.07
NO_FAULT_NOISE_LIMIT = 1.5

def improvement(before: float, after: float) -> float:
    return (before - after) / before if before > 0 else 0.0

def fit_source_head_probe(inp_path: Path, observed: pd.DataFrame, nominal: pd.DataFrame, config: ObservationConfig) -> dict[str, object]:
    """Fit an off-grid source-head discrepancy probe on even hours.

    The probe is a competing physical hypothesis, not knowledge of the hidden
    scenario. Its offset grid deliberately excludes the benchmark's true fault
    magnitudes to avoid an exact candidate/ground-truth match.
    """
    train_hours = list(range(0, config.duration_hours, 2))
    validation_hours = list(range(1, config.duration_hours, 2))
    baseline_train, _, _ = _combined_score(observed, nominal, nominal, config, train_hours)
    baseline_valid, _, _ = _combined_score(observed, nominal, nominal, config, validation_hours)
    network = load_hydraulic_network(inp_path, config.duration_hours, config.demand_model)
    candidates: list[tuple[float, str, float, pd.DataFrame]] = []
    for reservoir in network.reservoir_name_list:
        for offset in SOURCE_HEAD_OFFSETS_M:
            prediction = _simulate_candidate(inp_path, config, FaultSpec(fault_type='source_head', target=str(reservoir), start_hour=0, end_hour=config.duration_hours, magnitude=float(offset)))
            if not _candidate_is_complete(prediction, config):
                continue
            score, _, _ = _combined_score(observed, prediction, nominal, config, train_hours)
            candidates.append((score, str(reservoir), float(offset), prediction))
    if not candidates:
        raise RuntimeError('No complete source-head discrepancy candidates')
    train_after, reservoir, offset, prediction = min(candidates, key=lambda x: x[0])
    valid_after, pressure_rmse, flow_nrmse = _combined_score(observed, prediction, nominal, config, validation_hours)
    return {'hypothesis': 'source_head', 'best_parameters': {'reservoir': reservoir, 'offset_m': offset}, 'training_score_before': baseline_train, 'training_score_after': train_after, 'validation_score_before': baseline_valid, 'validation_score_after': valid_after, 'training_improvement': improvement(baseline_train, train_after), 'validation_improvement': improvement(baseline_valid, valid_after), 'pressure_rmse_m': pressure_rmse, 'flow_nrmse': flow_nrmse, 'pressure_noise_ratio': pressure_rmse / config.pressure_noise_std_m, 'flow_noise_ratio': flow_nrmse / config.flow_noise_fraction, 'simulations': len(network.reservoir_name_list) * len(SOURCE_HEAD_OFFSETS_M)}

def choose_relative_hypothesis(evidence: dict[str, dict[str, object]], baseline_pressure_noise_ratio: float, baseline_flow_noise_ratio: float) -> tuple[str, str, float, float]:
    """Compare all hypotheses on held-out score and abstain on weak/tied evidence."""
    ranked = sorted(evidence, key=lambda family: float(evidence[family]['validation_score_after']))
    winner, runner_up = ranked[:2]
    win = evidence[winner]
    runner = evidence[runner_up]
    balanced_improvement = min(float(win['training_improvement']), float(win['validation_improvement']))
    runner_score = float(runner['validation_score_after'])
    relative_margin = (runner_score - float(win['validation_score_after'])) / runner_score if runner_score > 0 else 0.0
    nominal_consistent = baseline_pressure_noise_ratio <= NO_FAULT_NOISE_LIMIT and baseline_flow_noise_ratio <= NO_FAULT_NOISE_LIMIT
    if nominal_consistent and balanced_improvement < IMPROVEMENT_FLOOR:
        return ('no_fault', 'nominal_noise_consistent', balanced_improvement, relative_margin)
    if balanced_improvement < IMPROVEMENT_FLOOR:
        return ('unknown_cause', 'weak_cross_split_evidence', balanced_improvement, relative_margin)
    if relative_margin < RELATIVE_MARGIN_FLOOR:
        return ('ambiguous_causes', 'held_out_hypotheses_tied', balanced_improvement, relative_margin)
    return (winner, 'relative_held_out_winner', balanced_improvement, relative_margin)

def target_from_evidence(family: str, item: dict[str, object]) -> str | None:
    key = {'leak': 'node', 'demand_error': 'node', 'roughness_error': 'pipe', 'sensor_bias': 'sensor', 'source_head': 'reservoir'}.get(family)
    if key is None:
        return None
    value = item['best_parameters'].get(key)
    return None if value is None else str(value)

def summarize(results: pd.DataFrame) -> pd.DataFrame:
    accepted_families = KNOWN_FAMILIES | {'source_head'}
    rows = []
    for tier, group in results.groupby('tier'):
        accepted = group[group['predicted_label'].isin(accepted_families)]
        known = group[group['true_cause'].isin(KNOWN_FAMILIES)]
        source = group[group['true_cause'] == 'source_head']
        no_fault = group[group['true_cause'] == 'no_fault']
        correct_accepted = accepted['predicted_label'] == accepted['true_cause']
        rows.append({'tier': tier, 'method': 'hybrid_relative_evidence_with_discrepancy_probe', 'runs': len(group), 'expanded_root_cause_accuracy': float(group['expanded_correct'].mean()), 'legacy_outcome_accuracy': float(group['legacy_correct'].mean()), 'known_family_accuracy': float(known['expanded_correct'].mean()), 'source_head_accuracy': float(source['expanded_correct'].mean()) if len(source) else None, 'no_fault_recall': float(no_fault['expanded_correct'].mean()) if len(no_fault) else None, 'accepted_diagnoses': len(accepted), 'accepted_diagnosis_precision': float(correct_accepted.mean()) if len(accepted) else None, 'false_attribution_rate': float(group['false_attribution'].mean()), 'abstention_or_ambiguity_rate': float(group['predicted_label'].isin({'unknown_cause', 'ambiguous_causes'}).mean()), 'target_accuracy_when_family_correct': float(accepted.loc[correct_accepted, 'target_correct'].mean()) if correct_accepted.any() else None, 'mean_logical_hydraulic_simulations': float(group['logical_hydraulic_simulations'].mean())})
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate uncertainty-aware AnyTown hybrid verifier v2.')
    parser.add_argument('--llm-results', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--case-index', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    llm = pd.read_csv(args.llm_results) if args.llm_results.exists() else pd.DataFrame()
    llm_lookup = {(r.tier, r.scenario_id, int(r.observation_seed)): r for r in llm.itertuples()}
    cases = pd.read_csv(args.case_index)
    case_lookup = {(r.tier, r.scenario_id, int(r.seed)): r for r in cases.itertuples()}
    design = json.loads(args.design.read_text(encoding='utf-8'))
    scenario_lookup = {item['id']: item for item in design['scenarios']}
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    rows: list[dict[str, object]] = []
    for case in cases.itertuples():
        record = llm_lookup.get((case.tier, case.scenario_id, int(case.seed)))
        observation_seed = int(case.seed)
        neutral_case_id = record.neutral_case_id if record is not None else f'{case.tier}:{case.scenario_id}:seed_{observation_seed}'
        llm_initial_label = record.predicted_label if record is not None else None
        fault = FaultSpec(**scenario_lookup[case.scenario_id]['fault'])
        config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=observation_seed, solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=args.candidate_cache)
        if case.tier == 'matched_model':
            observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
        else:
            observed, _, _ = generate_perturbed_observations(inp_path, fault, config, design['perturbed_truth_model'], case.scenario_id)
        nominal = nominal_prediction(inp_path, config)
        evidence_path = args.evidence_root / case.tier / f'{case.scenario_id}_seed_{observation_seed}' / 'family_evidence.json'
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        evidence['source_head'] = fit_source_head_probe(inp_path, observed, nominal, config)
        decision, reason, balanced, margin = choose_relative_hypothesis(evidence, float(case.baseline_pressure_noise_ratio), float(case.baseline_flow_noise_ratio))
        target = target_from_evidence(decision, evidence[decision]) if decision in evidence else None
        legacy_prediction = 'unknown_cause' if decision == 'source_head' else decision
        legacy_expected = 'unknown_cause' if fault.fault_type == 'source_head' else fault.fault_type
        accepted = decision in KNOWN_FAMILIES | {'source_head'}
        rows.append({'neutral_case_id': neutral_case_id, 'tier': case.tier, 'scenario_id': case.scenario_id, 'observation_seed': observation_seed, 'true_cause': fault.fault_type, 'true_target': fault.target, 'llm_initial_label': llm_initial_label, 'predicted_label': decision, 'legacy_predicted_label': legacy_prediction, 'predicted_target': target, 'decision_reason': reason, 'balanced_improvement': balanced, 'relative_winner_margin': margin, 'expanded_correct': decision == fault.fault_type, 'legacy_correct': legacy_prediction == legacy_expected, 'target_correct': decision == fault.fault_type and target == str(fault.target), 'false_attribution': accepted and decision != fault.fault_type, 'logical_hydraulic_simulations': sum((int(item['simulations']) for item in evidence.values())), 'source_head_probe_parameters': json.dumps(evidence['source_head']['best_parameters'], sort_keys=True), 'source_head_probe_training_improvement': evidence['source_head']['training_improvement'], 'source_head_probe_validation_improvement': evidence['source_head']['validation_improvement'], 'source_head_probe_validation_score_after': evidence['source_head']['validation_score_after'], 'source_head_probe_pressure_noise_ratio': evidence['source_head']['pressure_noise_ratio'], 'source_head_probe_flow_noise_ratio': evidence['source_head']['flow_noise_ratio']})
        print(f'completed {neutral_case_id} {decision}')
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'hybrid_v2_results.csv', index=False)
    summary = summarize(results)
    summary.to_csv(args.output / 'hybrid_v2_summary.csv', index=False)
    manifest = {'experiment': 'hybrid_relative_evidence_with_discrepancy_probe', 'case_runs': len(results), 'frozen_llm_source': str(args.llm_results.resolve()), 'known_family_evidence_source': str(args.evidence_root.resolve()), 'decision_policy': {'compare_all_families_on_held_out_score': True, 'cross_split_improvement_floor': IMPROVEMENT_FLOOR, 'relative_winner_margin_floor': RELATIVE_MARGIN_FLOOR, 'no_fault_noise_limit': NO_FAULT_NOISE_LIMIT, 'source_head_probe_offsets_m': list(SOURCE_HEAD_OFFSETS_M), 'true_source_head_magnitudes_excluded_from_probe_grid': True}, 'interpretation': 'The source-head family is used as a model-discrepancy probe. Results report both expanded root-cause accuracy and the legacy open-set metric that maps a detected source-head cause to unknown_cause.'}
    (args.output / 'hybrid_v2_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
