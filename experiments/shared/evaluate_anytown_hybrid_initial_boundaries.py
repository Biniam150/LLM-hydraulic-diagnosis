from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
KNOWN_FAMILIES = ['leak', 'demand_error', 'roughness_error', 'sensor_bias']
DEFAULT_ORDER = ['sensor_bias', 'leak', 'demand_error', 'roughness_error']
IMPROVEMENT_FLOOR = 0.02
WINNER_MARGIN_FLOOR = 0.07
NO_FAULT_NOISE_LIMIT = 1.5

def parse_families(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split(',') if item.strip()]

def llm_order(record: object) -> list[str]:
    proposed = parse_families(record.predicted_families)
    if record.predicted_label in KNOWN_FAMILIES:
        proposed = [record.predicted_label]
    order: list[str] = []
    for family in proposed + DEFAULT_ORDER:
        if family in KNOWN_FAMILIES and family not in order:
            order.append(family)
    return order

def decide(evidence: dict[str, dict[str, object]], baseline_pressure_ratio: float, baseline_flow_ratio: float) -> tuple[str, str, float, float, str, str]:
    ranked = sorted(KNOWN_FAMILIES, key=lambda family: float(evidence[family]['validation_score_after']))
    winner, runner_up = ranked[:2]
    win = evidence[winner]
    balanced_improvement = min(float(win['training_improvement']), float(win['validation_improvement']))
    winner_score = float(win['validation_score_after'])
    runner_score = float(evidence[runner_up]['validation_score_after'])
    relative_margin = (runner_score - winner_score) / runner_score if runner_score > 0 else 0.0
    nominal_consistent = baseline_pressure_ratio <= NO_FAULT_NOISE_LIMIT and baseline_flow_ratio <= NO_FAULT_NOISE_LIMIT
    if nominal_consistent and balanced_improvement < IMPROVEMENT_FLOOR:
        return ('no_fault', 'nominal_noise_consistent', balanced_improvement, relative_margin, winner, runner_up)
    if balanced_improvement < IMPROVEMENT_FLOOR:
        return ('unknown_cause', 'weak_cross_split_evidence', balanced_improvement, relative_margin, winner, runner_up)
    if relative_margin < WINNER_MARGIN_FLOOR:
        return ('ambiguous_causes', 'held_out_hypotheses_tied', balanced_improvement, relative_margin, winner, runner_up)
    return (winner, 'unique_cross_split_winner', balanced_improvement, relative_margin, winner, runner_up)

def target_from_evidence(family: str, item: dict[str, object]) -> str | None:
    key = {'leak': 'node', 'demand_error': 'node', 'roughness_error': 'pipe', 'sensor_bias': 'sensor'}.get(family)
    if key is None:
        return None
    value = item['best_parameters'].get(key)
    return None if value is None else str(value)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-results', type=Path, required=True)
    parser.add_argument('--case-index', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    llm = pd.read_csv(args.llm_results)
    cases = pd.read_csv(args.case_index)
    cases = cases[cases['tier'] == 'matched_model'].copy()
    case_lookup = {(row.tier, row.scenario_id, int(row.seed)): row for row in cases.itertuples()}
    rows: list[dict[str, object]] = []
    for record in llm.itertuples():
        key = (record.tier, record.scenario_id, int(record.observation_seed))
        case = case_lookup[key]
        evidence_path = args.evidence_root / record.tier / f'{record.scenario_id}_seed_{record.observation_seed}' / 'family_evidence.json'
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        evidence = {family: evidence[family] for family in KNOWN_FAMILIES}
        order = llm_order(record)
        prediction, reason, balanced, margin, winner, runner_up = decide(evidence, float(case.baseline_pressure_noise_ratio), float(case.baseline_flow_noise_ratio))
        expected = 'unknown_cause' if record.true_cause == 'source_head' else record.true_cause
        target = target_from_evidence(prediction, evidence[prediction]) if prediction in KNOWN_FAMILIES else None
        accepted = prediction in KNOWN_FAMILIES
        rows.append({'model': record.model_label, 'neutral_case_id': record.neutral_case_id, 'tier': record.tier, 'scenario_id': record.scenario_id, 'observation_seed': int(record.observation_seed), 'true_cause': record.true_cause, 'true_target': record.true_target, 'expected_label': expected, 'llm_initial_label': record.predicted_label, 'llm_test_order': ','.join(order), 'tested_families': ','.join(order), 'family_tests': len(order), 'logical_hydraulic_simulations': sum((int(evidence[family]['simulations']) for family in order)), 'hydraulic_winner': winner, 'hydraulic_runner_up': runner_up, 'balanced_improvement': balanced, 'relative_winner_margin': margin, 'predicted_label': prediction, 'predicted_target': target, 'decision_reason': reason, 'exact_outcome_correct': prediction == expected, 'target_correct': bool(prediction == record.true_cause and target is not None and (str(target) == str(record.true_target))), 'accepted': accepted, 'false_attribution': bool(accepted and prediction != record.true_cause), 'llm_first_family_is_hydraulic_winner': bool(order and order[0] == winner)})
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'hybrid_results.csv', index=False)
    summaries: list[dict[str, object]] = []
    for model, group in results.groupby('model', sort=False):
        known = group[group['true_cause'].isin(KNOWN_FAMILIES)]
        accepted = group[group['accepted']]
        accepted_correct = accepted['predicted_label'] == accepted['true_cause']
        correct_known_family = group[group['true_cause'].isin(KNOWN_FAMILIES) & (group['predicted_label'] == group['true_cause'])]
        unknown = group[group['true_cause'] == 'source_head']
        no_fault = group[group['true_cause'] == 'no_fault']
        summaries.append({'model': model, 'runs': len(group), 'exact_outcome_accuracy': group['exact_outcome_correct'].mean(), 'known_family_accuracy': known['exact_outcome_correct'].mean(), 'accepted_diagnosis_precision': accepted_correct.mean() if len(accepted) else None, 'false_attribution_rate': group['false_attribution'].mean(), 'ambiguity_rate': (group['predicted_label'] == 'ambiguous_causes').mean(), 'unknown_cause_recall': unknown['exact_outcome_correct'].mean(), 'no_fault_recall': no_fault['exact_outcome_correct'].mean(), 'target_accuracy_when_family_correct': correct_known_family['target_correct'].mean(), 'mean_family_tests': group['family_tests'].mean(), 'mean_logical_hydraulic_simulations': group['logical_hydraulic_simulations'].mean(), 'llm_first_family_matches_hydraulic_winner': group['llm_first_family_is_hydraulic_winner'].mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / 'hybrid_summary.csv', index=False)
    manifest = {'experiment': 'llm_guided_exhaustive_known_family_hydraulic_verification', 'network': 'Anytown', 'cases_per_model': 18, 'observation_seed': 1009, 'known_families': KNOWN_FAMILIES, 'withheld_unknown_mechanism': 'source_head', 'llm_role': 'prioritize the order of configured family tests', 'acceptance_control': 'deterministic comparison after all four configured families are tested', 'decision_boundaries': {'cross_split_improvement_floor': IMPROVEMENT_FLOOR, 'relative_winner_margin_floor': WINNER_MARGIN_FLOOR, 'no_fault_noise_ratio_limit': NO_FAULT_NOISE_LIMIT}, 'sensitivity_analysis_status': 'planned', 'source_head_probe_used': False}
    (args.output / 'hybrid_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
