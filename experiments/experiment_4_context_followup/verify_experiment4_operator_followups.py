from __future__ import annotations
import argparse
import ast
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from experiments.experiment_3_hydraulic_verification.evaluate_anytown_hybrid_v3_adequacy import validation_gof
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, _candidate_is_complete, _combined_score, _simulate_candidate, generate_pseudo_observations, nominal_prediction
GOF_ALPHA = 0.05
FAMILIES = {'leak', 'demand_error', 'roughness_error', 'sensor_bias', 'source_head'}

def target_id(value: object) -> str:
    text = str(value).strip()
    if re.fullmatch('\\d+\\.0+', text):
        return text.split('.', 1)[0]
    match = re.findall('[A-Za-z]*\\d+[A-Za-z]*', text)
    if match:
        return match[-1]
    prefixed = re.fullmatch('(?:reservoir|source|junction|node|pipe|sensor)\\s*[:#-]?\\s+([A-Za-z][A-Za-z0-9_-]*)', text, flags=re.IGNORECASE)
    return prefixed.group(1) if prefixed else text

def bounds(family: str, config: ObservationConfig) -> tuple[float, float]:
    if family == 'leak':
        return (min(config.search_policy.leak_areas_m2), max(config.search_policy.leak_areas_m2))
    if family == 'demand_error':
        return (min(config.search_policy.demand_multipliers), max(config.search_policy.demand_multipliers))
    if family == 'roughness_error':
        return (min(config.search_policy.roughness_multipliers), max(config.search_policy.roughness_multipliers))
    if family == 'sensor_bias':
        return (-2.0, 2.0)
    if family == 'source_head':
        return (-3.2, 3.2)
    raise ValueError(family)

def candidate_fault(family: str, target: str, onset: int | None, magnitude: float, duration: int) -> FaultSpec:
    start = int(onset) if family in {'leak', 'sensor_bias'} and onset is not None else 0
    return FaultSpec(family, target, start, duration, float(magnitude))

def fit_focused_test(family: str, target: str, onset: int | None, inp_path: Path, observed: pd.DataFrame, nominal: pd.DataFrame, config: ObservationConfig) -> dict[str, object]:
    train = list(range(0, config.duration_hours, 2))
    validation = list(range(1, config.duration_hours, 2))
    before_train, _, _ = _combined_score(observed, nominal, nominal, config, train)
    before_validation, _, _ = _combined_score(observed, nominal, nominal, config, validation)
    baseline_p, baseline_stat, baseline_df = validation_gof(observed, nominal, nominal, config, validation)
    lower, upper = bounds(family, config)
    predictions: dict[float, pd.DataFrame] = {}
    simulations = 0

    def objective(value: float) -> float:
        nonlocal simulations
        try:
            prediction = _simulate_candidate(inp_path, config, candidate_fault(family, target, onset, float(value), config.duration_hours))
        except (ValueError, KeyError, RuntimeError):
            return 1000000000000.0
        simulations += 1
        if not _candidate_is_complete(prediction, config):
            return 1000000000000.0
        predictions[float(value)] = prediction
        score, _, _ = _combined_score(observed, prediction, nominal, config, train)
        return float(score)
    span = upper - lower
    grid = np.linspace(lower, upper, 21)
    grid_scores = np.array([objective(float(value)) for value in grid])
    best_index = int(np.argmin(grid_scores))
    best_value = float(grid[best_index])
    best_score = float(grid_scores[best_index])
    local_lower = float(grid[max(0, best_index - 1)])
    local_upper = float(grid[min(len(grid) - 1, best_index + 1)])
    if local_upper > local_lower:
        optimum = minimize_scalar(objective, bounds=(local_lower, local_upper), method='bounded', options={'xatol': max(span / 2500.0, 1e-09), 'maxiter': 36})
        if np.isfinite(float(optimum.fun)) and float(optimum.fun) < best_score:
            best_value = float(optimum.x)
            best_score = float(optimum.fun)
    value = best_value
    prediction = predictions.get(value)
    if prediction is None or not np.isfinite(best_score) or best_score >= 100000000000.0:
        return {'valid_candidate': False, 'family': family, 'target': target, 'onset_hour': onset, 'simulations': simulations, 'baseline_validation_gof_pvalue': baseline_p}
    after_train = best_score
    after_validation, pressure_rmse, flow_nrmse = _combined_score(observed, prediction, nominal, config, validation)
    candidate_p, candidate_stat, candidate_df = validation_gof(observed, prediction, nominal, config, validation)
    training_improvement = (before_train - after_train) / max(before_train, 1e-12)
    validation_improvement = (before_validation - after_validation) / max(before_validation, 1e-12)
    accepted = bool(baseline_p < GOF_ALPHA and candidate_p >= GOF_ALPHA and (training_improvement >= 0.0) and (validation_improvement >= 0.0))
    return {'valid_candidate': True, 'accepted': accepted, 'family': family, 'target': target, 'onset_hour': onset, 'fitted_magnitude': value, 'training_improvement': training_improvement, 'validation_improvement': validation_improvement, 'pressure_rmse_m': pressure_rmse, 'flow_nrmse': flow_nrmse, 'baseline_validation_gof_pvalue': baseline_p, 'baseline_validation_gof_statistic': baseline_stat, 'baseline_validation_gof_df': baseline_df, 'candidate_validation_gof_pvalue': candidate_p, 'candidate_validation_gof_statistic': candidate_stat, 'candidate_validation_gof_df': candidate_df, 'simulations': simulations}

def context_gate(decision: object, variant: dict[str, object]) -> tuple[bool, str]:
    if decision.action != 'propose_follow_up_test':
        return (False, 'llm_requested_more_information')
    if decision.proposed_family not in FAMILIES or not str(decision.proposed_target).strip():
        return (False, 'invalid_test_configuration')
    records = {record['record_id']: record for record in variant['operational_records']}
    raw_citations = decision.cited_record_ids
    if isinstance(raw_citations, str) and raw_citations.strip().startswith('['):
        try:
            parsed_citations = ast.literal_eval(raw_citations)
        except (ValueError, SyntaxError):
            parsed_citations = []
    elif isinstance(raw_citations, str):
        parsed_citations = [item for item in raw_citations.split('|') if item]
    else:
        parsed_citations = list(raw_citations)
    cited = [str(item) for item in parsed_citations]
    if not cited or any((item not in records for item in cited)):
        return (False, 'missing_or_invalid_citation')
    supporting = [records[item] for item in cited if records[item]['reliability'] == 'high' and records[item]['time_relation'] == 'within the diagnostic observation window']
    if not supporting:
        return (False, 'no_reliable_time_relevant_record')
    if decision.record_assessment != 'reliable_and_relevant':
        return (False, 'llm_did_not_assess_record_as_reliable_and_relevant')
    return (True, 'context_gate_passed')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--decisions', nargs='+', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
    variant_lookup = {item['variant_id']: item for item in benchmark['variants']}
    design = json.loads(args.design.read_text(encoding='utf-8'))
    scenario_lookup = {item['id']: item for item in design['scenarios']}
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    decisions = pd.concat([pd.read_csv(path) for path in args.decisions], ignore_index=True)
    decisions['cited_record_ids'] = decisions['cited_record_ids'].fillna('')
    hydraulic_inputs: dict[tuple[str, int], tuple[pd.DataFrame, pd.DataFrame, ObservationConfig]] = {}
    evidence_cache: dict[tuple[str, int, str, str, int | None], dict[str, object]] = {}
    rows = []
    for decision in decisions.itertuples():
        variant = variant_lookup[decision.variant_id]
        gate_passed, gate_reason = context_gate(decision, variant)
        evidence = None
        if gate_passed:
            base_key = (variant['scenario_id'], int(variant['observation_seed']))
            if base_key not in hydraulic_inputs:
                fault = FaultSpec(**scenario_lookup[variant['scenario_id']]['fault'])
                config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(variant['observation_seed']), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)
                observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
                nominal = nominal_prediction(inp_path, config)
                hydraulic_inputs[base_key] = (observed, nominal, config)
            observed, nominal, config = hydraulic_inputs[base_key]
            normalized_target = target_id(decision.proposed_target)
            onset = int(decision.proposed_onset_hour) if pd.notna(decision.proposed_onset_hour) else None
            cache_key = (variant['scenario_id'], int(variant['observation_seed']), decision.proposed_family, normalized_target, onset)
            if cache_key not in evidence_cache:
                evidence_cache[cache_key] = fit_focused_test(decision.proposed_family, normalized_target, onset, inp_path, observed, nominal, config)
            evidence = evidence_cache[cache_key]
        llm_only_outcome = decision.proposed_family if decision.action == 'propose_follow_up_test' else variant['experiment3_outcome']
        accepted = bool(gate_passed and evidence and evidence.get('accepted', False))
        final = decision.proposed_family if accepted else variant['experiment3_outcome']
        rows.append({**decision._asdict(), 'valid_structured_output': bool(pd.isna(decision.parse_error) or not str(decision.parse_error).strip()), 'context_gate_passed': gate_passed, 'context_gate_reason': gate_reason, 'llm_context_only_outcome': llm_only_outcome, 'focused_hydraulic_accepted': accepted, 'final_outcome': final, 'correct_verified_resolution': final == variant['true_family'], 'safe_task_correct': final == variant['safe_expected_outcome'], 'llm_only_safe_task_correct': llm_only_outcome == variant['safe_expected_outcome'], 'hydraulic_evidence': json.dumps(evidence, sort_keys=True) if evidence else None})
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'experiment4_verified_results.csv', index=False)
    summary = results.groupby(['model_label', 'context_condition'], as_index=False).agg(cases=('variant_id', 'count'), proposal_rate=('action', lambda x: (x == 'propose_follow_up_test').mean()), context_gate_rate=('context_gate_passed', 'mean'), verified_resolution_rate=('focused_hydraulic_accepted', 'mean'), correct_verified_resolution_rate=('correct_verified_resolution', 'mean'), hybrid_safe_task_accuracy=('safe_task_correct', 'mean'), llm_context_only_safe_task_accuracy=('llm_only_safe_task_correct', 'mean'), valid_structured_output_rate=('valid_structured_output', 'mean'))
    summary.to_csv(args.output / 'experiment4_verified_summary.csv', index=False)
    model_summary = results.groupby('model_label', as_index=False).agg(cases=('variant_id', 'count'), overall_hybrid_safe_task_accuracy=('safe_task_correct', 'mean'), overall_llm_context_only_safe_task_accuracy=('llm_only_safe_task_correct', 'mean'), valid_structured_output_rate=('valid_structured_output', 'mean'), false_resolution_rate_nonhelpful=('focused_hydraulic_accepted', lambda x: x[results.loc[x.index, 'context_condition'] != 'helpful_verified'].mean()))
    model_summary.to_csv(args.output / 'experiment4_model_summary.csv', index=False)
    (args.output / 'focused_hydraulic_evidence_cache.json').write_text(json.dumps({'|'.join(map(str, key)): value for key, value in evidence_cache.items()}, indent=2), encoding='utf-8')
    manifest = {'experiment': 'operator_context_guided_follow_up_verification', 'absolute_validation_gof_alpha': GOF_ALPHA, 'acceptance': 'baseline fails held-out GOF; focused candidate passes held-out GOF; training and validation scores do not worsen', 'context_gate': 'valid citation to a high-reliability record within the diagnostic window', 'ground_truth_exposed_to_llm': False}
    (args.output / 'experiment4_verification_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
