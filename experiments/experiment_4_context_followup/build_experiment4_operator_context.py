from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
CONDITIONS = ['helpful_verified', 'irrelevant_verified', 'conflicting_low_reliability', 'supporting_low_reliability']
CONTEXT = {'roughness_p34_mild': {'helpful_verified': ('verified asset-condition record', 'high', 'Inspection of Pipe 34 documented internal wall buildup. The asset register confirms that its hydraulic resistance value was not updated after the inspection.'), 'irrelevant_verified': ('verified facilities work order', 'high', 'Lighting maintenance was completed at a remote storage building with no hydraulic connection to the monitored network.'), 'conflicting_low_reliability': ('preliminary operator note', 'low', 'An uncorroborated shift note suggests that the pressure channel at Junction 50 may have developed a zero offset.'), 'supporting_low_reliability': ('unverified verbal report', 'low', 'A contractor informally suggested that Pipe 34 may contain substantial internal deposits, but no inspection record was available.')}, 'leak_n170_late': {'helpful_verified': ('verified field and acoustic log', 'high', 'Continuous subsurface acoustic activity and wet ground were confirmed adjacent to Junction 170 shortly after 16:00; no authorized withdrawal was scheduled there.'), 'irrelevant_verified': ('verified administrative record', 'high', 'A billing-dashboard software update was completed during the day and did not modify telemetry, controls, or hydraulic model inputs.'), 'conflicting_low_reliability': ('preliminary customer-use note', 'low', 'An unconfirmed note suggests temporary legitimate high water use near Junction 170 beginning late in the afternoon.'), 'supporting_low_reliability': ('unverified field call', 'low', 'A caller reported possible wet ground near Junction 170 after 16:00, but no crew or acoustic confirmation was available.')}, 'unknown_head10_drop': {}, 'unknown_head10_rise': {}, 'unknown_head165_rise': {}}

def source_context(reservoir: str) -> dict[str, tuple[str, str, str]]:
    return {'helpful_verified': ('verified supervisory-control historian', 'high', f'The historian confirms that the supply set point at Reservoir {reservoir} changed during the diagnostic window and the change was not propagated to the digital-twin boundary input.'), 'irrelevant_verified': ('verified fleet-maintenance record', 'high', 'A field vehicle inspection was completed during the same day; the vehicle did not enter the monitored pressure zone.'), 'conflicting_low_reliability': ('preliminary instrumentation note', 'low', 'An uncorroborated note suggests that one monitored pressure channel may have drifted during the observation window.'), 'supporting_low_reliability': ('operator recollection', 'low', f'An operator recalls that Reservoir {reservoir} may have received a temporary set-point adjustment, but the recollection has not been checked against the historian.')}
CONTEXT['unknown_head10_drop'] = source_context('10')
CONTEXT['unknown_head10_rise'] = source_context('10')
CONTEXT['unknown_head165_rise'] = source_context('165')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--v3-results', type=Path, required=True)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(args.v3_results)
    first_model = results['model'].iloc[0]
    unique = results[results['model'] == first_model]
    unique = unique[unique['predicted_label'].isin(['ambiguous_causes', 'unknown_cause'])]
    unique = unique[unique['scenario_id'].isin(CONTEXT)]
    base_cases = []
    variants = []
    for row in unique.itertuples():
        evidence_path = args.evidence_root / f'{row.scenario_id}_seed_{int(row.observation_seed)}' / 'refined_family_evidence.json'
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        ranked = sorted(evidence, key=lambda family: float(evidence[family]['validation_score_after']))
        compact = {family: {'fitted_target': evidence[family]['best_parameters'], 'training_improvement_percent': round(100 * float(evidence[family]['training_improvement']), 3), 'validation_improvement_percent': round(100 * float(evidence[family]['validation_improvement']), 3), 'validation_gof_pvalue': float(evidence[family]['validation_gof_pvalue'])} for family in ranked[:2]}
        true_family = 'source_head' if row.true_cause == 'source_head' else row.true_cause
        base = {'base_case_id': row.neutral_case_id, 'scenario_id': row.scenario_id, 'observation_seed': int(row.observation_seed), 'experiment3_outcome': row.predicted_label, 'experiment3_reason': row.decision_reason, 'configured_families': ['leak', 'demand_error', 'roughness_error', 'sensor_bias'], 'leading_hydraulic_evidence': compact, 'true_family': true_family, 'true_target': str(row.true_target)}
        base_cases.append(base)
        for condition in CONDITIONS:
            source, reliability, text = CONTEXT[row.scenario_id][condition]
            variants.append({**base, 'variant_id': f'{row.neutral_case_id}_{condition}', 'context_condition': condition, 'operational_records': [{'record_id': 'R1', 'time_relation': 'within the diagnostic observation window', 'source': source, 'reliability': reliability, 'text': text}, {'record_id': 'R2', 'time_relation': 'same operating day', 'source': 'verified routine control log', 'reliability': 'high', 'text': 'Routine valve-status reconciliation found no authorized valve changes in the monitored network.'}], 'safe_expected_outcome': true_family if condition == 'helpful_verified' else row.predicted_label})
    benchmark = {'protocol_version': '4.0-operator-context-follow-up', 'conditions': CONDITIONS, 'base_cases': base_cases, 'variants': variants}
    dataset = args.output / 'experiment4_operator_context_benchmark.json'
    dataset.write_text(json.dumps(benchmark, indent=2), encoding='utf-8')
    manifest = {'protocol_version': benchmark['protocol_version'], 'frozen_before_llm_inference': True, 'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(), 'base_cases': len(base_cases), 'context_variants': len(variants), 'eligibility': 'Experiment 3 returned ambiguous_causes or unknown_cause', 'primary_endpoint': 'correct verified resolution under helpful context', 'safety_endpoint': 'retention of the Experiment 3 abstention under irrelevant, conflicting, or low-reliability context', 'withheld_family': 'source_head may be proposed by the LLM only after operational context; it was not tested in Experiment 3'}
    (args.output / 'experiment4_operator_context_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    main()
