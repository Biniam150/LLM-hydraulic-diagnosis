from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from experiments.experiment_4_context_followup.verify_experiment4_operator_followups import fit_focused_test
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, generate_pseudo_observations, nominal_prediction

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding='utf-8'))
    design = json.loads(args.design.read_text(encoding='utf-8'))
    scenarios = {item['id']: item for item in design['scenarios']}
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    rows = []
    for base in benchmark['base_cases']:
        fault = FaultSpec(**scenarios[base['scenario_id']]['fault'])
        config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(base['observation_seed']), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)
        observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
        nominal = nominal_prediction(inp_path, config)
        evidence = fit_focused_test(base['true_family'], str(base['true_target']), int(fault.start_hour) if base['true_family'] in {'leak', 'sensor_bias'} else None, inp_path, observed, nominal, config)
        rows.append({'scenario_id': base['scenario_id'], **evidence})
        print(f"{base['scenario_id']}: accepted={evidence.get('accepted')} p={evidence.get('candidate_validation_gof_pvalue')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
if __name__ == '__main__':
    main()
