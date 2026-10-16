from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.shared.run_hydraulic_diagnosis_demo import DEFAULT_PROFILE, PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, compute_residual_features, generate_pseudo_observations, nominal_prediction, test_hypothesis_family

def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def pressure_mae(observed: pd.DataFrame, predicted: pd.DataFrame) -> float:
    columns = [column for column in observed if column.startswith('pressure_')]
    return float(np.abs(observed[columns] - predicted[columns]).to_numpy().mean())

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate false roughness calibration on AnyTown mismatches.')
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path, default=PROJECT_DIR / 'results' / 'candidate_prediction_cache' / 'anytown')
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding='utf-8'))
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for scenario in design['scenarios']:
        fault = FaultSpec(**scenario['fault'])
        for seed in design['seeds']:
            case_dir = args.output / f"{scenario['id']}_seed_{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=args.candidate_cache)
            observed, clean, manifest = generate_pseudo_observations(inp_path, fault, config)
            nominal = nominal_prediction(inp_path, config)
            _, features = compute_residual_features(observed, nominal, config)
            evidence, calibrated = test_hypothesis_family('roughness_error', inp_path, observed, nominal, config, onset_hour=int(features['estimated_onset_hour']))
            mae_before = pressure_mae(observed, nominal)
            mae_after = pressure_mae(observed, calibrated)
            clean_mae_before = pressure_mae(clean, nominal)
            clean_mae_after = pressure_mae(clean, calibrated)
            improvement = (mae_before - mae_after) / mae_before if mae_before > 0 else 0.0
            wrong_cause = fault.fault_type not in {'roughness_error', 'no_fault'}
            false_calibration = bool(wrong_cause and improvement >= 0.1)
            false_attribution = bool(wrong_cause and evidence.result in {'strong_support', 'moderate_support'})
            row = {'scenario_id': scenario['id'], 'seed': int(seed), 'true_cause': fault.fault_type, 'true_target': fault.target, 'pressure_mae_before_m': mae_before, 'pressure_mae_after_m': mae_after, 'pressure_mae_improvement': improvement, 'clean_pressure_mae_before_m': clean_mae_before, 'clean_pressure_mae_after_m': clean_mae_after, 'fitted_pipe': evidence.best_parameters.get('pipe'), 'fitted_roughness_multiplier': evidence.best_parameters.get('multiplier'), 'roughness_evidence': evidence.result, 'validation_improvement': evidence.validation_improvement, 'pressure_noise_ratio': evidence.pressure_noise_ratio, 'flow_noise_ratio': evidence.flow_noise_ratio, 'false_calibration': false_calibration, 'false_causal_attribution': false_attribution, 'simulations': evidence.simulations}
            rows.append(row)
            observed.to_csv(case_dir / 'observations.csv', float_format='%.8f')
            clean.to_csv(case_dir / 'clean_truth.csv', float_format='%.8f')
            calibrated.to_csv(case_dir / 'roughness_calibrated_prediction.csv', float_format='%.8f')
            write_json(case_dir / 'ground_truth_manifest.json', manifest)
            write_json(case_dir / 'roughness_evidence.json', asdict(evidence))
            write_json(case_dir / 'case_summary.json', row)
            print(f"{scenario['id']} seed={seed}: MAE {mae_before:.4f} -> {mae_after:.4f} ({improvement:.1%}), evidence={evidence.result}")
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'false_roughness_results.csv', index=False)
    wrong = results[~results['true_cause'].isin(['roughness_error', 'no_fault'])]
    by_scenario = results.groupby(['scenario_id', 'true_cause'], as_index=False).agg(runs=('seed', 'size'), mean_pressure_mae_improvement=('pressure_mae_improvement', 'mean'), false_calibration_rate=('false_calibration', 'mean'), false_causal_attribution_rate=('false_causal_attribution', 'mean'), mean_pressure_noise_ratio=('pressure_noise_ratio', 'mean'), mean_flow_noise_ratio=('flow_noise_ratio', 'mean'))
    by_scenario.to_csv(args.output / 'false_roughness_by_scenario.csv', index=False)
    summary = {'runs': int(len(results)), 'non_roughness_runs': int(len(wrong)), 'non_roughness_false_calibration_rate': float(wrong['false_calibration'].mean()), 'non_roughness_false_causal_attribution_rate': float(wrong['false_causal_attribution'].mean()), 'mean_non_roughness_pressure_mae_improvement': float(wrong['pressure_mae_improvement'].mean()), 'design': design, 'interpretation': 'Pressure-fit improvement is not causal identification. False causal attribution additionally requires the frozen evidence policy to report moderate or strong roughness support.'}
    write_json(args.output / 'false_roughness_summary.json', summary)
    print(by_scenario.to_string(index=False))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
