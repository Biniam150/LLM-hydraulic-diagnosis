from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from wdn_diagnostics.diagnostic_profile import ALLOWED_HYPOTHESES, NetworkProfile, load_network_profile
from wdn_diagnostics.hydraulic_diagnostic_agent import HydraulicDiagnosticAgent
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, compute_residual_features, generate_pseudo_observations, nominal_prediction, test_hypothesis_family, topology_context
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INP = PROJECT_DIR / 'data/networks/anytown/anytown.inp'
DEFAULT_PROFILE = PROJECT_DIR / 'configs' / 'networks/anytown.json'
BWNII_PROFILE = PROJECT_DIR / 'configs' / 'networks/bwnii.json'
NET3_PROFILE = PROJECT_DIR / 'configs' / 'networks/net3.json'
PROFILE_ALIASES = {'anytown': DEFAULT_PROFILE, 'bwnii': BWNII_PROFILE, 'net3': NET3_PROFILE}
DEFAULT_OUTPUT = PROJECT_DIR / 'results' / 'hydraulic_demo'

def default_fault(fault_type: str, profile: NetworkProfile | None=None) -> FaultSpec:
    profile = profile or load_network_profile(DEFAULT_PROFILE)
    if fault_type not in profile.scenarios:
        raise ValueError(f'Unknown demonstration fault: {fault_type}')
    scenario = profile.scenarios[fault_type]
    return FaultSpec(scenario.fault_type, scenario.target, scenario.start_hour, scenario.end_hour, scenario.magnitude)

def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')

def _plot_residuals(residuals: pd.DataFrame, output_path: Path) -> None:
    pressure_columns = [c for c in residuals.columns if c.startswith('pressure_')]
    fig, ax = plt.subplots(figsize=(10, 5))
    for column in pressure_columns:
        ax.plot(residuals.index, residuals[column], marker='o', markersize=2.5, linewidth=1, label=column.removeprefix('pressure_'))
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Observed - nominal pressure (m)')
    ax.set_title('Hydraulic digital-twin pressure residuals')
    ax.legend(title='Sensor', ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def _plot_ranking(history: list[dict], output_path: Path) -> None:
    if not history:
        return
    last = history[-1]['decision']['ranking']
    names = [item['hypothesis'] for item in last]
    scores = [item['score'] for item in last]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, scores, color='#277da1')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Agent score')
    ax.set_title('Final diagnostic hypothesis ranking')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

def run_experiment(inp_path: Path, output_dir: Path, fault: FaultSpec, mode: str='ollama', model: str='qwen2.5:7b', seed: int=42, max_rounds: int=3, profile: NetworkProfile | None=None, candidate_cache_dir: Path | None=None) -> Dict[str, object]:
    profile = profile or load_network_profile(DEFAULT_PROFILE)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_patterns = ('agent_round_*.json', 'ollama_raw_round_*.txt', 'ollama_metadata_round_*.json', 'test_round_*.json', 'best_prediction_*.csv')
    generated_names = ('observations.csv', 'clean_ground_truth_signals.csv', 'nominal_prediction.csv', 'nominal_residuals.csv', 'ground_truth_manifest.json', 'residual_features.json', 'topology_context.json', 'pressure_residuals.png', 'final_hypothesis_ranking.png', 'diagnostic_summary.json', 'diagnostic_history.json')
    for pattern in generated_patterns:
        for path in output_dir.glob(pattern):
            path.unlink()
    for name in generated_names:
        path = output_dir / name
        if path.exists():
            path.unlink()
    config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=seed, solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=candidate_cache_dir)
    observed, clean_truth, manifest = generate_pseudo_observations(inp_path, fault, config)
    nominal = nominal_prediction(inp_path, config)
    residuals, features = compute_residual_features(observed, nominal, config)
    topology = topology_context(inp_path, config.pressure_sensors, config)
    observed.to_csv(output_dir / 'observations.csv', float_format='%.8f')
    clean_truth.to_csv(output_dir / 'clean_ground_truth_signals.csv', float_format='%.8f')
    nominal.to_csv(output_dir / 'nominal_prediction.csv', float_format='%.8f')
    residuals.to_csv(output_dir / 'nominal_residuals.csv', float_format='%.8f')
    _write_json(output_dir / 'ground_truth_manifest.json', manifest)
    _write_json(output_dir / 'residual_features.json', features)
    _write_json(output_dir / 'topology_context.json', topology)
    _plot_residuals(residuals, output_dir / 'pressure_residuals.png')
    agent = HydraulicDiagnosticAgent(mode=mode, model=model, max_rounds=max_rounds, deterministic_test_order=profile.deterministic_test_order)
    evidence = []
    history = []
    final_decision = None
    total_simulations = 2
    for round_number in range(1, max_rounds + 2):
        decision, raw = agent.decide(features, topology, evidence)
        if decision.executed_hypothesis is None and decision.requested_test is not None:
            decision.executed_hypothesis = decision.requested_test.hypothesis
        if decision.estimated_test_cost is None and decision.executed_hypothesis is not None:
            cost_item = topology.get('deterministic_test_costs', {}).get(decision.executed_hypothesis, {})
            if 'estimated_hydraulic_simulations' in cost_item:
                decision.estimated_test_cost = int(cost_item['estimated_hydraulic_simulations'])
        final_decision = decision
        history_item = {'round': round_number, 'decision': asdict(decision), 'evidence_available': [asdict(item) for item in evidence], 'action_provenance': {'llm_top_hypothesis': decision.llm_top_hypothesis, 'llm_requested_hypothesis': decision.llm_requested_hypothesis, 'executed_hypothesis': decision.executed_hypothesis, 'estimated_test_cost': decision.estimated_test_cost, 'used_fallback': decision.used_fallback, 'fallback_reason': decision.fallback_reason, 'repair_attempted': decision.repair_attempted, 'repair_succeeded': decision.repair_succeeded, 'llm_generation_attempts': list(agent.last_attempt_metadata)}}
        history.append(history_item)
        _write_json(output_dir / f'agent_round_{round_number}.json', history_item)
        if raw is not None:
            (output_dir / f'ollama_raw_round_{round_number}.txt').write_text(raw, encoding='utf-8')
        for attempt_number, attempt_raw in enumerate(agent.last_raw_attempts, start=1):
            (output_dir / f'ollama_raw_round_{round_number}_attempt_{attempt_number}.txt').write_text(attempt_raw, encoding='utf-8')
        for attempt_number, attempt_metadata in enumerate(agent.last_attempt_metadata, start=1):
            _write_json(output_dir / f'ollama_metadata_round_{round_number}_attempt_{attempt_number}.json', attempt_metadata)
        if decision.decision in {'diagnose', 'abstain'}:
            break
        if decision.requested_test is None or len(evidence) >= max_rounds:
            break
        request = decision.requested_test
        test_evidence, best_prediction = test_hypothesis_family(request.hypothesis, inp_path, observed, nominal, config, onset_hour=int(features['estimated_onset_hour']), requested_nodes=request.target_nodes, requested_links=request.target_links)
        evidence.append(test_evidence)
        total_simulations += test_evidence.simulations
        _write_json(output_dir / f'test_round_{len(evidence)}_{request.hypothesis}.json', asdict(test_evidence))
        best_prediction.to_csv(output_dir / f'best_prediction_{request.hypothesis}.csv', float_format='%.8f')
    if final_decision is not None and final_decision.decision == 'continue_testing':
        from wdn_diagnostics.hydraulic_diagnostic_agent import rule_based_decision
        final_decision = rule_based_decision(features, evidence, max_rounds, profile.deterministic_test_order)
        final_decision.rationale.append('Applied mandatory evidence adjudication at round limit')
        final_decision.used_fallback = False
        history.append({'round': 'final_evidence_gate', 'decision': asdict(final_decision), 'evidence_available': [asdict(item) for item in evidence]})
    _plot_ranking(history, output_dir / 'final_hypothesis_ranking.png')
    diagnosed = final_decision.diagnosis if final_decision is not None else None
    expected = fault.fault_type if fault.fault_type in ALLOWED_HYPOTHESES else None
    correct = diagnosed == expected if expected is not None else diagnosed is None
    agent_rounds = [item for item in history if isinstance(item.get('round'), int)]
    first_agent_decision = agent_rounds[0]['decision'] if agent_rounds else {}
    first_ranking = first_agent_decision.get('ranking', [])
    first_decision_correct_family_rank = None
    if expected is not None:
        for rank, item in enumerate(first_ranking, start=1):
            if item.get('hypothesis') == expected:
                first_decision_correct_family_rank = rank
                break
    raw_llm_correct_family_initial_rank = first_decision_correct_family_rank if first_agent_decision.get('llm_top_hypothesis') is not None else None
    correct_family_test_position = None
    if expected is not None:
        for position, item in enumerate(evidence, start=1):
            if item.hypothesis == expected:
                correct_family_test_position = position
                break
    fallback_reasons = [item['decision'].get('fallback_reason') for item in agent_rounds if item['decision'].get('fallback_reason')]
    summary = {'agent_version': 'v2_evidence_grounded', 'mode': mode, 'model': model if mode == 'ollama' else None, 'network_profile': profile.name, 'deterministic_test_order': profile.deterministic_test_order, 'feature_policy': asdict(profile.feature_policy), 'search_policy': asdict(profile.search_policy), 'evidence_policy': asdict(profile.evidence_policy), 'deterministic_test_costs': topology.get('deterministic_test_costs', {}), 'seed': seed, 'hidden_fault_type': fault.fault_type, 'hidden_fault_target': fault.target, 'final_decision': final_decision.decision if final_decision else 'no_decision', 'diagnosis': diagnosed, 'expected_diagnosis': expected, 'correct_family_or_abstention': correct, 'tested_hypotheses': [item.hypothesis for item in evidence], 'test_evidence': [asdict(item) for item in evidence], 'total_hydraulic_simulations': total_simulations, 'used_fallback_in_any_round': any((item['decision']['used_fallback'] for item in history)), 'llm_action_provenance': {'raw_first_ranked_hypothesis': first_agent_decision.get('llm_top_hypothesis'), 'raw_first_requested_hypothesis': first_agent_decision.get('llm_requested_hypothesis'), 'first_executed_hypothesis': first_agent_decision.get('executed_hypothesis'), 'correct_family_initial_rank': raw_llm_correct_family_initial_rank, 'first_executed_decision_correct_family_rank': first_decision_correct_family_rank, 'correct_family_test_position': correct_family_test_position, 'fallback_reasons': fallback_reasons, 'repair_attempted_in_any_round': any((item['decision'].get('repair_attempted', False) for item in agent_rounds)), 'repair_succeeded_in_any_round': any((item['decision'].get('repair_succeeded', False) for item in agent_rounds)), 'length_limit_hit_in_any_round': any((attempt.get('done_reason') == 'length' for item in agent_rounds for attempt in item.get('action_provenance', {}).get('llm_generation_attempts', []))), 'total_llm_generation_attempts': sum((len(item.get('action_provenance', {}).get('llm_generation_attempts', [])) for item in agent_rounds))}}
    _write_json(output_dir / 'diagnostic_summary.json', summary)
    _write_json(output_dir / 'diagnostic_history.json', history)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description='Run the end-to-end hidden hydraulic fault diagnosis experiment.')
    parser.add_argument('--network', choices=sorted(PROFILE_ALIASES), default=None)
    parser.add_argument('--profile', type=Path, default=None)
    parser.add_argument('--inp', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fault', default='leak')
    parser.add_argument('--mode', choices=['ollama', 'rule'], default='ollama')
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-rounds', type=int, default=3)
    args = parser.parse_args()
    if args.network is not None and args.profile is not None:
        parser.error('Use either --network or --profile, not both')
    profile_path = args.profile or PROFILE_ALIASES.get(args.network, DEFAULT_PROFILE)
    profile = load_network_profile(profile_path)
    inp_path = args.inp or PROJECT_DIR / profile.inp_file
    summary = run_experiment(inp_path=inp_path, output_dir=args.output, fault=default_fault(args.fault, profile), mode=args.mode, model=args.model, seed=args.seed, max_rounds=args.max_rounds, profile=profile)
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
