from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.shared.evaluate_anytown_causal_benchmark import baseline_noise_ratios
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, generate_pseudo_observations, nominal_prediction

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design = json.loads(args.design.read_text(encoding='utf-8'))
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    case_rows: list[dict[str, object]] = []
    verifier_rows: list[dict[str, object]] = []
    for scenario_number, scenario in enumerate(design['scenarios'], start=1):
        fault = FaultSpec(**scenario['fault'])
        for seed in design['seeds']:
            config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=args.candidate_cache)
            observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
            nominal = nominal_prediction(inp_path, config)
            validation_hours = [hour for hour in observed.index if int(hour) % 2 == 1]
            pressure_ratio, flow_ratio = baseline_noise_ratios(observed, nominal, config, validation_hours)
            expected = 'unknown_cause' if fault.fault_type == 'source_head' else fault.fault_type
            case_rows.append({'tier': 'matched_model', 'scenario_id': scenario['id'], 'seed': int(seed), 'true_cause': fault.fault_type, 'expected_label': expected, 'baseline_pressure_noise_ratio': pressure_ratio, 'baseline_flow_noise_ratio': flow_ratio})
            verifier_rows.append({'neutral_case_id': f'case_{scenario_number:03d}', 'tier': 'matched_model', 'scenario_id': scenario['id'], 'observation_seed': int(seed), 'model': 'Frozen deterministic order', 'model_label': 'Frozen deterministic order', 'true_cause': fault.fault_type, 'true_target': fault.target, 'expected_label': expected, 'predicted_label': 'ambiguous_causes', 'predicted_families': '', 'predicted_target': ''})
            print(f"prepared {scenario['id']} seed={seed}")
    pd.DataFrame(case_rows).to_csv(args.output / 'external_case_index.csv', index=False)
    pd.DataFrame(verifier_rows).to_csv(args.output / 'fixed_order_verifier_input.csv', index=False)
    (args.output / 'external_input_manifest.json').write_text(json.dumps({'network': design['network'], 'cases': len(case_rows), 'verifier_policy': design['verifier_policy'], 'policy_frozen_before_results': design['frozen_before_results'], 'llm_inference_used': False, 'purpose': 'External validation of the deterministic verifier component.'}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
