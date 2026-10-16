from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from experiments.shared.evaluate_anytown_causal_benchmark import KNOWN_FAMILIES
from experiments.shared.evaluate_anytown_hybrid_initial_boundaries import DEFAULT_ORDER, IMPROVEMENT_FLOOR, NO_FAULT_NOISE_LIMIT, WINNER_MARGIN_FLOOR, llm_order, target_from_evidence
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, _candidate_is_complete, _combined_score, _simulate_candidate, compute_residual_features, generate_pseudo_observations, nominal_prediction, test_hypothesis_family
FAMILIES = ['leak', 'demand_error', 'roughness_error', 'sensor_bias']
GOF_ALPHA = 0.05

def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def fault_from_parameters(family: str, parameters: dict[str, object], magnitude: float, duration_hours: int) -> FaultSpec:
    if family == 'leak':
        return FaultSpec('leak', str(parameters['node']), int(parameters['start_hour']), duration_hours, magnitude)
    if family == 'demand_error':
        return FaultSpec('demand_error', str(parameters['node']), 0, duration_hours, magnitude)
    if family == 'roughness_error':
        return FaultSpec('roughness_error', str(parameters['pipe']), 0, duration_hours, magnitude)
    raise ValueError(f'Continuous refinement is not defined for {family}')

def refinement_spec(family: str, parameters: dict[str, object], config: ObservationConfig) -> tuple[float, float, float, str]:
    if family == 'leak':
        values = list(config.search_policy.leak_areas_m2)
        return (min(values), max(values), float(parameters['area_m2']), 'area_m2')
    if family == 'demand_error':
        values = list(config.search_policy.demand_multipliers)
        return (min(values), max(values), float(parameters['multiplier']), 'multiplier')
    if family == 'roughness_error':
        values = list(config.search_policy.roughness_multipliers)
        return (min(values), max(values), float(parameters['multiplier']), 'multiplier')
    raise ValueError(f'Continuous refinement is not defined for {family}')

def validation_gof(observed: pd.DataFrame, prediction: pd.DataFrame, nominal: pd.DataFrame, config: ObservationConfig, hours: list[int]) -> tuple[float, float, int]:
    pressure_columns = [c for c in observed if c.startswith('pressure_')]
    flow_column = f'flow_{config.flow_link}'
    pressure_z = (observed.loc[hours, pressure_columns].to_numpy(dtype=float) - prediction.loc[hours, pressure_columns].to_numpy(dtype=float)) / config.pressure_noise_std_m
    flow_scale = config.flow_noise_fraction * max(float(nominal.loc[hours, flow_column].abs().mean()), 1e-06)
    flow_z = (observed.loc[hours, flow_column].to_numpy(dtype=float) - prediction.loc[hours, flow_column].to_numpy(dtype=float)) / flow_scale
    statistic = float(np.sum(pressure_z ** 2) + np.sum(flow_z ** 2))
    degrees_of_freedom = int(pressure_z.size + flow_z.size)
    return (float(chi2.sf(statistic, degrees_of_freedom)), statistic, degrees_of_freedom)

def refine_family(family: str, coarse: dict[str, object], coarse_prediction: pd.DataFrame, inp_path: Path, observed: pd.DataFrame, nominal: pd.DataFrame, config: ObservationConfig) -> tuple[dict[str, object], pd.DataFrame]:
    train_hours = list(range(0, config.duration_hours, 2))
    validation_hours = list(range(1, config.duration_hours, 2))
    baseline_train, _, _ = _combined_score(observed, nominal, nominal, config, train_hours)
    baseline_validation, _, _ = _combined_score(observed, nominal, nominal, config, validation_hours)
    parameters = dict(coarse['best_parameters'])
    extra_simulations = 0
    if family == 'sensor_bias':
        best_prediction = coarse_prediction
        best_train = float(coarse['training_score_after'])
        refined_parameters = parameters
    else:
        lower, upper, coarse_value, parameter_name = refinement_spec(family, parameters, config)
        predictions: dict[float, pd.DataFrame] = {}

        def objective(value: float) -> float:
            nonlocal extra_simulations
            numeric = float(value)
            prediction = _simulate_candidate(inp_path, config, fault_from_parameters(family, parameters, numeric, config.duration_hours))
            extra_simulations += 1
            if not _candidate_is_complete(prediction, config):
                return 1000000000000.0
            predictions[numeric] = prediction
            score, _, _ = _combined_score(observed, prediction, nominal, config, train_hours)
            return float(score)
        span = upper - lower
        optimum = minimize_scalar(objective, bounds=(lower, upper), method='bounded', options={'xatol': max(span / 2000.0, 1e-09), 'maxiter': 30})
        optimum_value = float(optimum.x)
        optimum_score = float(optimum.fun)
        coarse_score = float(coarse['training_score_after'])
        if not np.isfinite(optimum_score) or coarse_score <= optimum_score:
            best_prediction = coarse_prediction
            best_train = coarse_score
            optimum_value = coarse_value
        else:
            best_prediction = predictions.get(optimum_value)
            if best_prediction is None:
                best_prediction = _simulate_candidate(inp_path, config, fault_from_parameters(family, parameters, optimum_value, config.duration_hours))
                extra_simulations += 1
            best_train = optimum_score
        refined_parameters = dict(parameters)
        refined_parameters[parameter_name] = optimum_value
        refined_parameters['coarse_value'] = coarse_value
        refined_parameters['continuous_refinement'] = True
    validation_score, pressure_rmse, flow_nrmse = _combined_score(observed, best_prediction, nominal, config, validation_hours)
    training_improvement = (baseline_train - best_train) / max(baseline_train, 1e-12)
    validation_improvement = (baseline_validation - validation_score) / max(baseline_validation, 1e-12)
    gof_pvalue, gof_statistic, gof_df = validation_gof(observed, best_prediction, nominal, config, validation_hours)
    return ({'hypothesis': family, 'best_parameters': refined_parameters, 'training_score_before': baseline_train, 'training_score_after': best_train, 'validation_score_before': baseline_validation, 'validation_score_after': validation_score, 'training_improvement': training_improvement, 'validation_improvement': validation_improvement, 'pressure_rmse_m': pressure_rmse, 'flow_nrmse': flow_nrmse, 'pressure_noise_ratio': pressure_rmse / config.pressure_noise_std_m, 'flow_noise_ratio': flow_nrmse / config.flow_noise_fraction, 'validation_gof_pvalue': gof_pvalue, 'validation_gof_statistic': gof_statistic, 'validation_gof_df': gof_df, 'coarse_simulations': int(coarse['simulations']), 'refinement_simulations': extra_simulations, 'simulations': int(coarse['simulations']) + extra_simulations}, best_prediction)

def decide(evidence: dict[str, dict[str, object]], baseline_pressure_ratio: float, baseline_flow_ratio: float, nominal_gof_pvalue: float | None=None) -> dict[str, object]:
    ranked = sorted(FAMILIES, key=lambda family: float(evidence[family]['validation_score_after']))
    winner, runner_up = ranked[:2]
    win = evidence[winner]
    balanced = min(float(win['training_improvement']), float(win['validation_improvement']))
    winner_score = float(win['validation_score_after'])
    runner_score = float(evidence[runner_up]['validation_score_after'])
    margin = (runner_score - winner_score) / runner_score if runner_score > 0 else 0.0
    nominal_consistent = nominal_gof_pvalue >= GOF_ALPHA if nominal_gof_pvalue is not None else baseline_pressure_ratio <= NO_FAULT_NOISE_LIMIT and baseline_flow_ratio <= NO_FAULT_NOISE_LIMIT
    supported = [family for family in ranked if min(float(evidence[family]['training_improvement']), float(evidence[family]['validation_improvement'])) >= IMPROVEMENT_FLOOR and float(evidence[family]['validation_gof_pvalue']) >= GOF_ALPHA]
    if not supported:
        if nominal_consistent:
            outcome, reason = ('no_fault', 'nominal_gof_not_rejected')
        elif balanced < IMPROVEMENT_FLOOR:
            outcome, reason = ('unknown_cause', 'weak_cross_split_evidence')
        else:
            outcome, reason = ('unknown_cause', 'no_supported_family_passes_absolute_held_out_gof')
    elif len(supported) == 1:
        winner = supported[0]
        runner_up = ranked[1] if ranked[0] == winner else ranked[0]
        win = evidence[winner]
        balanced = min(float(win['training_improvement']), float(win['validation_improvement']))
        winner_score = float(win['validation_score_after'])
        runner_score = float(evidence[runner_up]['validation_score_after'])
        margin = (runner_score - winner_score) / runner_score if runner_score > 0 else 0.0
        outcome, reason = (winner, 'sole_cross_split_and_gof_supported_family')
    else:
        winner, runner_up = supported[:2]
        win = evidence[winner]
        balanced = min(float(win['training_improvement']), float(win['validation_improvement']))
        winner_score = float(win['validation_score_after'])
        runner_score = float(evidence[runner_up]['validation_score_after'])
        margin = (runner_score - winner_score) / runner_score if runner_score > 0 else 0.0
        if margin < WINNER_MARGIN_FLOOR:
            outcome, reason = ('ambiguous_causes', 'supported_hypotheses_tied')
        else:
            outcome, reason = (winner, 'supported_family_has_held_out_margin')
    return {'outcome': outcome, 'reason': reason, 'winner': winner, 'runner_up': runner_up, 'balanced_improvement': balanced, 'relative_winner_margin': margin, 'winner_validation_gof_pvalue': float(win['validation_gof_pvalue']), 'supported_families': supported}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-results', type=Path, required=True)
    parser.add_argument('--case-index', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    llm = pd.read_csv(args.llm_results)
    cases = pd.read_csv(args.case_index)
    cases = cases[cases['tier'] == 'matched_model']
    case_lookup = {(row.tier, row.scenario_id, int(row.seed)): row for row in cases.itertuples()}
    design = json.loads(args.design.read_text(encoding='utf-8'))
    scenarios = {item['id']: item for item in design['scenarios']}
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    unique_cases = llm[['tier', 'scenario_id', 'observation_seed', 'true_cause', 'true_target']].drop_duplicates()
    decisions: dict[tuple[str, str, int], dict[str, object]] = {}
    for case in unique_cases.itertuples():
        key = (case.tier, case.scenario_id, int(case.observation_seed))
        index_case = case_lookup[key]
        fault = FaultSpec(**scenarios[case.scenario_id]['fault'])
        config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(case.observation_seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=args.candidate_cache)
        observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
        nominal = nominal_prediction(inp_path, config)
        _, features = compute_residual_features(observed, nominal, config)
        nominal_gof_pvalue, _, _ = validation_gof(observed, nominal, nominal, config, list(range(1, config.duration_hours, 2)))
        refined_evidence: dict[str, dict[str, object]] = {}
        for family in FAMILIES:
            coarse, coarse_prediction = test_hypothesis_family(family, inp_path, observed, nominal, config, onset_hour=int(features['estimated_onset_hour']))
            refined, _ = refine_family(family, asdict(coarse), coarse_prediction, inp_path, observed, nominal, config)
            refined_evidence[family] = refined
        decision = decide(refined_evidence, float(index_case.baseline_pressure_noise_ratio), float(index_case.baseline_flow_noise_ratio), nominal_gof_pvalue)
        decisions[key] = {'decision': decision, 'evidence': refined_evidence}
        case_dir = args.output / 'case_evidence' / f'{case.scenario_id}_seed_{case.observation_seed}'
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(case_dir / 'refined_family_evidence.json', refined_evidence)
        write_json(case_dir / 'decision.json', decision)
        print(f"{case.scenario_id}: {decision['outcome']} winner={decision['winner']} p={decision['winner_validation_gof_pvalue']:.3g}")
    rows: list[dict[str, object]] = []
    for record in llm.itertuples():
        key = (record.tier, record.scenario_id, int(record.observation_seed))
        bundle = decisions[key]
        decision = dict(bundle['decision'])
        evidence = dict(bundle['evidence'])
        prediction = str(decision['outcome'])
        expected = 'unknown_cause' if record.true_cause == 'source_head' else record.true_cause
        target = target_from_evidence(prediction, evidence[prediction]) if prediction in KNOWN_FAMILIES else None
        order = llm_order(record)
        accepted = prediction in KNOWN_FAMILIES
        rows.append({'model': record.model_label, 'neutral_case_id': record.neutral_case_id, 'tier': record.tier, 'scenario_id': record.scenario_id, 'observation_seed': int(record.observation_seed), 'true_cause': record.true_cause, 'true_target': record.true_target, 'expected_label': expected, 'llm_initial_label': record.predicted_label, 'llm_test_order': ','.join(order), 'predicted_label': prediction, 'predicted_target': target, 'decision_reason': decision['reason'], 'hydraulic_winner': decision['winner'], 'hydraulic_runner_up': decision['runner_up'], 'balanced_improvement': decision['balanced_improvement'], 'relative_winner_margin': decision['relative_winner_margin'], 'winner_validation_gof_pvalue': decision['winner_validation_gof_pvalue'], 'exact_outcome_correct': prediction == expected, 'accepted': accepted, 'false_attribution': accepted and prediction != record.true_cause, 'target_correct': bool(prediction == record.true_cause and target is not None and (str(target) == str(record.true_target))), 'logical_hydraulic_simulations': sum((int(evidence[family]['simulations']) for family in FAMILIES))})
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'hybrid_v3_results.csv', index=False)
    summaries: list[dict[str, object]] = []
    for model, group in results.groupby('model', sort=False):
        known = group[group['true_cause'].isin(KNOWN_FAMILIES)]
        accepted = group[group['accepted']]
        accepted_correct = accepted['predicted_label'] == accepted['true_cause']
        unknown = group[group['true_cause'] == 'source_head']
        no_fault = group[group['true_cause'] == 'no_fault']
        detectable_known = known[known.apply(lambda row: float(case_lookup[row.tier, row.scenario_id, int(row.observation_seed)].baseline_pressure_noise_ratio) > NO_FAULT_NOISE_LIMIT or float(case_lookup[row.tier, row.scenario_id, int(row.observation_seed)].baseline_flow_noise_ratio) > NO_FAULT_NOISE_LIMIT, axis=1)]
        correct_known = known[known['predicted_label'] == known['true_cause']]
        summaries.append({'model': model, 'runs': len(group), 'exact_outcome_accuracy': group['exact_outcome_correct'].mean(), 'known_family_accuracy': known['exact_outcome_correct'].mean(), 'detectable_known_family_accuracy': detectable_known['exact_outcome_correct'].mean(), 'accepted_diagnosis_precision': accepted_correct.mean() if len(accepted) else None, 'diagnostic_coverage': len(accepted) / len(group), 'false_attribution_rate': group['false_attribution'].mean(), 'unknown_cause_recall': unknown['exact_outcome_correct'].mean(), 'no_fault_recall': no_fault['exact_outcome_correct'].mean(), 'target_accuracy_when_family_correct': correct_known['target_correct'].mean(), 'mean_logical_hydraulic_simulations': group['logical_hydraulic_simulations'].mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / 'hybrid_v3_summary.csv', index=False)
    write_json(args.output / 'hybrid_v3_manifest.json', {'experiment': 'refined_known_family_verification_with_absolute_adequacy', 'known_families': FAMILIES, 'withheld_unknown': 'source_head', 'source_head_probe_used': False, 'coarse_search': 'full configured location and magnitude grids', 'refinement': 'bounded continuous magnitude refinement at each coarse winning location using training hours', 'validation': 'odd-hour held-out chi-square goodness-of-fit', 'decision_order': ['require cross-split improvement for each family', 'require absolute held-out goodness-of-fit for each family', 'evaluate ambiguity only among the remaining supported families', 'return unknown when the nominal model is rejected and no family is supported', 'return no fault only when the nominal held-out goodness-of-fit is not rejected'], 'decision_boundaries': {'cross_split_improvement_floor': IMPROVEMENT_FLOOR, 'relative_winner_margin_floor': WINNER_MARGIN_FLOOR, 'no_fault_noise_ratio_limit': NO_FAULT_NOISE_LIMIT, 'absolute_validation_gof_alpha': GOF_ALPHA}})
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
