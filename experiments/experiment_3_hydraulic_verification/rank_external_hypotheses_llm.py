from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from wdn_diagnostics.ollama_client import OllamaClient, extract_json_object
FAMILIES = ['leak', 'demand_error', 'roughness_error', 'sensor_bias']

def validate_ranking(raw: str) -> tuple[list[str], str]:
    parsed = extract_json_object(raw)
    ranking = [str(x).strip().lower() for x in parsed.get('ranking', [])]
    if len(ranking) != len(FAMILIES) or set(ranking) != set(FAMILIES):
        raise ValueError('ranking must contain every configured family exactly once')
    return (ranking, str(parsed.get('reason', '')))

def main() -> None:
    parser = argparse.ArgumentParser(description='Blind four-family LLM ranking.')
    parser.add_argument('--blind-root', type=Path, required=True)
    parser.add_argument('--case-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument('--inference-seed', type=int, default=424242)
    parser.add_argument('--num-ctx', type=int, default=4096)
    parser.add_argument('--tier', default=None, help='Optional case-index tier to retain (for example, matched_model).')
    parser.add_argument('--cost-aware', action='store_true')
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
        decision_path = case_dir / 'ranking.json'
        if args.resume and decision_path.exists():
            rows.append(json.loads(decision_path.read_text(encoding='utf-8')))
            print(f'reused {case.neutral_case_id}')
            continue
        blind = json.loads((args.blind_root / case.neutral_case_id / 'blind_input.json').read_text(encoding='utf-8'))['user_prompt']
        system_prompt = 'Rank configured water-network mismatch hypotheses using only the supplied blind residual summary and compact topology. The ranking controls test order only and is not a diagnosis. Return one JSON object.'
        network_context = blind['network_context']
        family_costs = {'sensor_bias': 0, 'leak': int(network_context['junction_count']) * 3, 'demand_error': int(network_context['junction_count']) * 5, 'roughness_error': int(network_context['pipe_count']) * 4}
        task = 'Rank all configured cause families from most to least plausible.'
        if args.cost_aware:
            task = 'Rank all configured cause families to minimize expected hydraulic search cost while retaining diagnostically plausible families early. A zero-cost analytic test should normally be performed first.'
        prompt_payload = {'task': task, 'configured_families': {'leak': 'pressure-dependent outflow at one junction', 'demand_error': 'incorrect demand magnitude at one junction', 'roughness_error': 'incorrect resistance at one pipe', 'sensor_bias': 'persistent offset at one monitored pressure sensor'}, 'required_output': {'ranking': FAMILIES, 'reason': 'one short sentence'}, 'residual_summary': blind['residual_summary'], 'network_context': network_context}
        if args.cost_aware:
            prompt_payload['estimated_test_costs'] = family_costs
        user_prompt = json.dumps(prompt_payload, separators=(',', ':'))
        raw = client.chat(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.0, seed=args.inference_seed, num_predict=256, num_ctx=args.num_ctx, think=False)
        metadata = dict(client.last_response_metadata)
        parse_error = None
        try:
            ranking, reason = validate_ranking(raw)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
            ranking, reason = (list(FAMILIES), 'invalid output; fixed fallback')
        record = {'neutral_case_id': case.neutral_case_id, 'tier': case.tier, 'scenario_id': case.scenario_id, 'observation_seed': int(case.observation_seed), 'model': args.model, 'ranking': ranking, 'reason': reason, 'valid_output': parse_error is None, 'parse_error': parse_error, 'prompt_tokens': metadata.get('prompt_eval_count'), 'generated_tokens': metadata.get('eval_count')}
        decision_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        (case_dir / 'raw_response.txt').write_text(raw, encoding='utf-8')
        (case_dir / 'ollama_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        rows.append(record)
        print(f"{case.neutral_case_id}: {','.join(ranking)}")
    pd.DataFrame([{**row, 'ranking': ','.join(row['ranking'])} for row in rows]).to_csv(args.output / 'llm_hypothesis_rankings.csv', index=False)
    (args.output / 'manifest.json').write_text(json.dumps({'model': args.model, 'temperature': 0.0, 'inference_seed': args.inference_seed, 'num_ctx': args.num_ctx, 'cost_aware': args.cost_aware, 'families': FAMILIES, 'cases': len(rows)}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
