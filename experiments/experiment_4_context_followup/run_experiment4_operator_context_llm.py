from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from wdn_diagnostics.ollama_client import OllamaClient, extract_json_object
FAMILIES = {'leak', 'demand_error', 'roughness_error', 'sensor_bias', 'source_head'}
ACTIONS = {'propose_follow_up_test', 'request_more_information'}
ASSESSMENTS = {'reliable_and_relevant', 'irrelevant', 'insufficient_reliability', 'conflicting'}

def build_prompt(variant: dict[str, object]) -> tuple[str, str]:
    system = 'You assist an engineer after a water-network hydraulic diagnostic has abstained. Interpret operational records to configure one focused follow-up hydraulic test. A record may be relevant, irrelevant, conflicting, or too unreliable to use. Do not diagnose from text alone. Propose a test only when a reliable, temporally relevant record identifies a physically testable mechanism and asset; otherwise request more information. Return one JSON object.'
    payload = {'task': 'Choose whether a focused hydraulic follow-up test is justified.', 'experiment3_result': {'outcome': variant['experiment3_outcome'], 'reason': variant['experiment3_reason'], 'configured_families': variant['configured_families'], 'leading_candidate_evidence': variant['leading_hydraulic_evidence']}, 'additional_testable_family': {'source_head': 'incorrect reservoir or supply-boundary head/set point'}, 'family_definitions': {'leak': 'unaccounted pressure-dependent outflow at a junction', 'demand_error': 'incorrect legitimate consumption at a junction', 'roughness_error': 'incorrect pipe resistance coefficient', 'sensor_bias': 'persistent pressure-sensor offset', 'source_head': 'incorrect reservoir or supply-boundary head'}, 'operational_records': variant['operational_records'], 'allowed_actions': sorted(ACTIONS), 'required_output': {'action': 'one allowed action', 'proposed_family': 'one family or null', 'proposed_target': 'junction, pipe, sensor, or reservoir identifier, or null', 'proposed_onset_hour': 'integer hour if the record supports one, otherwise null', 'cited_record_ids': 'record identifiers used, possibly empty', 'record_assessment': 'reliable_and_relevant, irrelevant, insufficient_reliability, or conflicting', 'reason': 'one short sentence'}}
    return (system, json.dumps(payload, separators=(',', ':')))

def parse(raw: str) -> dict[str, object]:
    value = extract_json_object(raw)
    action = str(value.get('action', '')).strip().lower()
    if action not in ACTIONS:
        raise ValueError(f'unsupported action {action!r}')
    family = value.get('proposed_family')
    family = str(family).strip().lower() if family is not None else None
    target = value.get('proposed_target')
    target = str(target).strip() if target is not None else None
    onset = value.get('proposed_onset_hour')
    onset = int(onset) if onset is not None else None
    cited = value.get('cited_record_ids', [])
    assessment = str(value.get('record_assessment', '')).strip().lower()
    if not isinstance(cited, list):
        raise ValueError('cited_record_ids must be a list')
    if assessment not in ASSESSMENTS:
        raise ValueError(f'unsupported record assessment {assessment!r}')
    if action == 'propose_follow_up_test':
        if family not in FAMILIES or not target:
            raise ValueError('a proposed test requires a supported family and target')
    else:
        family, target, onset = (None, None, None)
    return {'action': action, 'proposed_family': family, 'proposed_target': target, 'proposed_onset_hour': onset, 'cited_record_ids': [str(item) for item in cited], 'record_assessment': assessment, 'reason': str(value.get('reason', ''))[:500]}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-label', required=True)
    parser.add_argument('--inference-seed', type=int, default=424242)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
    client = OllamaClient(model=args.model)
    rows = []
    for variant in benchmark['variants']:
        case_dir = args.output / variant['variant_id']
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / 'decision.json'
        if args.resume and result_path.exists():
            rows.append(json.loads(result_path.read_text(encoding='utf-8')))
            continue
        system, user = build_prompt(variant)
        raw = client.chat(system_prompt=system, user_prompt=user, temperature=0.0, seed=args.inference_seed, num_predict=192, num_ctx=8192, think=False)
        parse_error = None
        try:
            decision = parse(raw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
            decision = {'action': 'request_more_information', 'proposed_family': None, 'proposed_target': None, 'proposed_onset_hour': None, 'cited_record_ids': [], 'record_assessment': 'conflicting', 'reason': 'invalid structured output'}
        metadata = dict(client.last_response_metadata)
        record = {'model': args.model, 'model_label': args.model_label, 'variant_id': variant['variant_id'], 'base_case_id': variant['base_case_id'], 'scenario_id': variant['scenario_id'], 'context_condition': variant['context_condition'], 'experiment3_outcome': variant['experiment3_outcome'], 'true_family': variant['true_family'], 'true_target': variant['true_target'], 'safe_expected_outcome': variant['safe_expected_outcome'], **decision, 'parse_error': parse_error, 'prompt_tokens': metadata.get('prompt_eval_count'), 'generated_tokens': metadata.get('eval_count'), 'inference_duration_s': float(metadata.get('total_duration_ns') or 0) / 1000000000.0}
        result_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        (case_dir / 'raw_response.txt').write_text(raw, encoding='utf-8')
        rows.append(record)
        print(f"{args.model_label} {variant['variant_id']}: {decision['action']} {decision['proposed_family']} {decision['proposed_target']}")
    pd.DataFrame(rows).to_csv(args.output / 'operator_context_llm_results.csv', index=False)
if __name__ == '__main__':
    main()
