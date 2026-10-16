from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
FAMILIES = ['sensor_bias', 'leak', 'demand_error', 'roughness_error']
FIXED_ORDER = list(FAMILIES)
MIN_IMPROVEMENT = 0.02
GOF_ALPHA = 0.05

def supported(item: dict[str, object]) -> bool:
    return min(float(item['training_improvement']), float(item['validation_improvement'])) >= MIN_IMPROVEMENT and float(item['validation_gof_pvalue']) >= GOF_ALPHA

def load_evidence(root: Path, scenario_id: str, seed: int) -> dict[str, dict[str, object]]:
    path = root / 'case_evidence' / f'{scenario_id}_seed_{seed}' / 'refined_family_evidence.json'
    return json.loads(path.read_text(encoding='utf-8'))

def evaluate_order(order: list[str], evidence: dict[str, dict[str, object]], final_label: str) -> dict[str, object]:
    full_cost = sum((int(evidence[f]['simulations']) for f in FAMILIES))
    spent = 0
    first_supported = None
    tested = 0
    for family in order:
        spent += int(evidence[family]['simulations'])
        tested += 1
        if supported(evidence[family]):
            first_supported = family
            break
    if first_supported is None:
        provisional = final_label
        spent = full_cost
        tested = len(FAMILIES)
    else:
        provisional = first_supported
    return {'order': ','.join(order), 'provisional_label': provisional, 'first_supported_family': first_supported, 'simulations_to_provisional': spent, 'full_verdict_simulations': full_cost, 'family_tests_to_provisional': tested, 'provisional_agrees_with_final': provisional == final_label, 'simulation_reduction': 1.0 - spent / full_cost if full_cost else 0.0}

def main() -> None:
    parser = argparse.ArgumentParser(description='Frozen-verifier search-order ablation.')
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--reference-results', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--random-replicates', type=int, default=500)
    parser.add_argument('--random-seed', type=int, default=280826)
    parser.add_argument('--llm-policy-label', default='LLM-guided')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rankings = pd.read_csv(args.rankings)
    reference = pd.read_csv(args.reference_results)
    stable_keys = ['scenario_id', 'observation_seed']
    if all((key in rankings.columns for key in stable_keys)):
        reference = reference.drop(columns=['neutral_case_id'], errors='ignore')
        reference = reference.drop_duplicates(subset=stable_keys)
        cases = rankings[['neutral_case_id', *stable_keys, 'ranking', 'valid_output']].merge(reference, on=stable_keys, validate='one_to_one')
    else:
        reference = reference.drop_duplicates(subset=['neutral_case_id'])
        cases = reference.merge(rankings[['neutral_case_id', 'ranking', 'valid_output']], on='neutral_case_id', validate='one_to_one')
    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, object]] = []
    for case in cases.itertuples():
        evidence = load_evidence(args.evidence_root, case.scenario_id, int(case.observation_seed))
        llm_order = [x for x in str(case.ranking).split(',') if x in FAMILIES]
        if len(llm_order) != len(FAMILIES):
            llm_order = list(FIXED_ORDER)
        policies: list[tuple[str, int, list[str]]] = [('Fixed order', 0, list(FIXED_ORDER)), (args.llm_policy_label, 0, llm_order)]
        for replicate in range(args.random_replicates):
            policies.append(('Random order', replicate, list(rng.permutation(FAMILIES))))
        for policy, replicate, order in policies:
            evaluated = evaluate_order(order, evidence, str(case.predicted_label))
            provisional = str(evaluated['provisional_label'])
            accepted = provisional in FAMILIES
            rows.append({'neutral_case_id': case.neutral_case_id, 'scenario_id': case.scenario_id, 'observation_seed': int(case.observation_seed), 'true_cause': case.true_cause, 'final_frozen_label': case.predicted_label, 'policy': policy, 'replicate': replicate, **evaluated, 'provisional_exact_correct': provisional == ('unknown_cause' if case.true_cause == 'source_head' else case.true_cause), 'provisional_accepted': accepted, 'provisional_false_attribution': accepted and provisional != case.true_cause})
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'search_order_case_results.csv', index=False)
    replicate_rows = []
    for (policy, replicate), group in results.groupby(['policy', 'replicate']):
        accepted = group[group['provisional_accepted']]
        replicate_rows.append({'policy': policy, 'replicate': int(replicate), 'cases': len(group), 'exact_outcome_accuracy': float(group['provisional_exact_correct'].mean()), 'accepted_diagnosis_precision': float((~accepted['provisional_false_attribution']).mean()) if len(accepted) else np.nan, 'false_attribution_rate': float(group['provisional_false_attribution'].mean()), 'agreement_with_frozen_final': float(group['provisional_agrees_with_final'].mean()), 'mean_simulations_to_provisional': float(group['simulations_to_provisional'].mean()), 'mean_full_verdict_simulations': float(group['full_verdict_simulations'].mean()), 'mean_simulation_reduction': float(group['simulation_reduction'].mean()), 'mean_family_tests_to_provisional': float(group['family_tests_to_provisional'].mean())})
    replicate_summary = pd.DataFrame(replicate_rows)
    replicate_summary.to_csv(args.output / 'search_order_replicate_summary.csv', index=False)
    summary = replicate_summary.groupby('policy', as_index=False).agg(exact_outcome_accuracy=('exact_outcome_accuracy', 'mean'), accepted_diagnosis_precision=('accepted_diagnosis_precision', 'mean'), false_attribution_rate=('false_attribution_rate', 'mean'), agreement_with_frozen_final=('agreement_with_frozen_final', 'mean'), mean_simulations_to_provisional=('mean_simulations_to_provisional', 'mean'), sd_simulations_to_provisional=('mean_simulations_to_provisional', 'std'), mean_full_verdict_simulations=('mean_full_verdict_simulations', 'mean'), mean_simulation_reduction=('mean_simulation_reduction', 'mean'), sd_simulation_reduction=('mean_simulation_reduction', 'std'), mean_family_tests_to_provisional=('mean_family_tests_to_provisional', 'mean'))
    summary.to_csv(args.output / 'search_order_summary.csv', index=False)
    known = cases[cases['true_cause'].isin(FAMILIES)].copy()
    rank_rows = []
    for case in known.itertuples():
        order = [x for x in str(case.ranking).split(',') if x in FAMILIES]
        rank_rows.append({'neutral_case_id': case.neutral_case_id, 'true_cause': case.true_cause, 'true_family_rank': order.index(case.true_cause) + 1})
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(args.output / 'llm_true_family_ranks.csv', index=False)
    ranking_summary = pd.DataFrame([{'known_cases': len(rank_df), 'top1_recall': float((rank_df['true_family_rank'] <= 1).mean()), 'top2_recall': float((rank_df['true_family_rank'] <= 2).mean()), 'top3_recall': float((rank_df['true_family_rank'] <= 3).mean()), 'mean_true_family_rank': float(rank_df['true_family_rank'].mean())}])
    ranking_summary.to_csv(args.output / 'llm_ranking_summary.csv', index=False)
    (args.output / 'manifest.json').write_text(json.dumps({'experiment': 'BWN-II frozen-verifier search-order ablation', 'configured_families': FAMILIES, 'fixed_order': FIXED_ORDER, 'minimum_split_improvement': MIN_IMPROVEMENT, 'held_out_gof_alpha': GOF_ALPHA, 'random_replicates': args.random_replicates, 'random_seed': args.random_seed, 'interpretation': 'A provisional result is the first physically supported family. The authoritative frozen verdict still requires exhaustive comparison.'}, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
    print(ranking_summary.to_string(index=False))
if __name__ == '__main__':
    main()
