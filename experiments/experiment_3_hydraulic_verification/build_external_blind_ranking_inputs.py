from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, compute_residual_features, generate_pseudo_observations, nominal_prediction, topology_context

def main() -> None:
    parser = argparse.ArgumentParser(description='Build blind residual/topology inputs without invoking an LLM.')
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design = json.loads(args.design.read_text(encoding='utf-8'))
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    topology = topology_context(inp_path, profile.pressure_sensors)
    prompt_topology = {'junction_count': topology['junction_count'], 'pipe_count': topology['pipe_count'], 'reservoir_count': topology['reservoir_count'], 'pump_count': topology['pump_count'], 'tank_count': topology['tank_count'], 'pressure_sensors': topology['pressure_sensors'], 'sensor_shortest_path_distances': topology['sensor_shortest_path_distances']}
    rows: list[dict[str, object]] = []
    case_number = 0
    for scenario in design['scenarios']:
        fault = FaultSpec(**scenario['fault'])
        for seed in design['seeds']:
            case_number += 1
            neutral_case_id = f'case_{case_number:03d}'
            case_dir = args.output / neutral_case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)
            observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
            nominal = nominal_prediction(inp_path, config)
            _, features = compute_residual_features(observed, nominal, config)
            payload = {'neutral_case_id': neutral_case_id, 'user_prompt': {'residual_summary': features, 'network_context': prompt_topology}}
            (case_dir / 'blind_input.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
            rows.append({'neutral_case_id': neutral_case_id, 'tier': 'matched_model', 'scenario_id': scenario['id'], 'observation_seed': int(seed), 'true_cause': fault.fault_type, 'true_target': fault.target})
            print(f"prepared {neutral_case_id} ({scenario['id']})")
    pd.DataFrame(rows).to_csv(args.output / 'blind_case_index.csv', index=False)
    (args.output / 'manifest.json').write_text(json.dumps({'network': design['network'], 'cases': len(rows), 'ground_truth_excluded_from_prompt': True, 'candidate_hydraulic_evidence_excluded_from_prompt': True}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
