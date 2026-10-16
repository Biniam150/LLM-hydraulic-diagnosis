from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, compute_residual_features, generate_pseudo_observations, nominal_prediction
TRAIN_SEEDS = [1103, 2207, 3301, 4409, 5501]
VALIDATION_SEEDS = [6607, 7703, 8807, 9901, 11113]

def fault_scenarios(split: str, network_name: str) -> list[FaultSpec]:
    scenarios: list[FaultSpec] = []
    is_net3 = 'net3' in network_name.lower() or 'net 3' in network_name.lower()
    is_bwnii = 'bwnii' in network_name.lower() or 'bwn' in network_name.lower()
    if is_bwnii and split == 'train':
        roughness_targets = ['P10', 'P100', 'P103']
        demand_targets = ['J511', 'J411', 'J414']
        leak_targets = ['J415', 'J416', 'J417']
        bias_targets = ['J296', 'J145']
    elif is_bwnii:
        roughness_targets = ['P110']
        demand_targets = ['J418']
        leak_targets = ['J419']
        bias_targets = ['J360']
    elif is_net3 and split == 'train':
        roughness_targets = ['20', '101', '145']
        demand_targets = ['15', '35', '60']
        leak_targets = ['40', '105', '139']
        bias_targets = ['15', '203']
    elif is_net3:
        roughness_targets = ['199']
        demand_targets = ['167']
        leak_targets = ['181']
        bias_targets = ['167']
    elif split == 'train':
        roughness_targets = ['12', '24', '50']
        demand_targets = ['50', '90', '130']
        leak_targets = ['30', '60', '100']
        bias_targets = ['50', '100']
    else:
        roughness_targets = ['68']
        demand_targets = ['150']
        leak_targets = ['150']
        bias_targets = ['150']
    for target in roughness_targets:
        for magnitude in [0.65, 0.8, 1.2, 1.35]:
            scenarios.append(FaultSpec('roughness_error', target, 0, 24, magnitude))
    for target in demand_targets:
        for magnitude in [0.6, 0.8, 1.2, 1.5]:
            scenarios.append(FaultSpec('demand_error', target, 0, 24, magnitude))
    leak_onsets = [0, 0, 0] if is_net3 or is_bwnii else [4, 8, 12]
    for target in leak_targets:
        for magnitude, onset in zip([0.00055, 0.0009, 0.0013], leak_onsets):
            scenarios.append(FaultSpec('leak', target, onset, 24, magnitude))
    for target in bias_targets:
        for magnitude, onset in [(-0.4, 4), (0.5, 8), (-0.7, 12), (0.8, 16)]:
            scenarios.append(FaultSpec('sensor_bias', target, onset, 24, magnitude))
    return scenarios

def flatten_features(features: dict[str, object], sensors: list[str]) -> dict[str, float]:
    direction = str(features['pressure_direction'])
    row: dict[str, float] = {'estimated_onset_hour': float(features['estimated_onset_hour']), 'affected_sensor_count': float(features['affected_sensor_count']), 'flow_noise_ratio': float(features['flow_noise_ratio']), 'flow_changed': float(bool(features['flow_changed'])), 'direction_negative': float(direction == 'mostly_negative'), 'direction_positive': float(direction == 'mostly_positive'), 'direction_mixed': float(direction == 'mixed_or_small')}
    per_sensor = dict(features['pressure_residual_by_sensor'])
    for sensor in sensors:
        values = dict(per_sensor[sensor])
        for statistic in ['mean_m', 'rmse_m', 'minimum_m', 'maximum_m']:
            row[f'pressure_{sensor}_{statistic}'] = float(values[statistic])
    return row

def make_config(profile: object, seed: int) -> ObservationConfig:
    return ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=seed, solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)

def generate_split(split: str, seeds: list[int], inp_path: Path, profile: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    scenarios = fault_scenarios(split, str(profile.name))
    nominal = nominal_prediction(inp_path, make_config(profile, seeds[0]))
    for scenario_index, fault in enumerate(scenarios, start=1):
        for seed in seeds:
            config = make_config(profile, seed)
            try:
                observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
                _, features = compute_residual_features(observed, nominal, config)
            except (ValueError, RuntimeError) as exc:
                failures.append({'split': split, 'fault': fault.__dict__, 'seed': seed, 'error': str(exc)})
                continue
            rows.append({'split': split, 'scenario_id': f'{split}_{fault.fault_type}_{scenario_index:03d}', 'seed': seed, 'true_cause': fault.fault_type, 'target': fault.target, 'magnitude': fault.magnitude, 'start_hour': fault.start_hour, **flatten_features(features, list(profile.pressure_sensors))})
    for offset, seed in enumerate(seeds, start=1):
        for replicate in range(5 if split == 'train' else 2):
            noise_seed = seed + 100000 * (replicate + 1)
            config = make_config(profile, noise_seed)
            fault = FaultSpec('no_fault', 'none', 0, 24, 0.0)
            observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
            _, features = compute_residual_features(observed, nominal, config)
            rows.append({'split': split, 'scenario_id': f'{split}_no_fault_{offset:03d}_{replicate:02d}', 'seed': noise_seed, 'true_cause': 'no_fault', 'target': 'none', 'magnitude': 0.0, 'start_hour': 0, **flatten_features(features, list(profile.pressure_sensors))})
    return (rows, failures)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    train, train_failures = generate_split('train', TRAIN_SEEDS, inp_path, profile)
    validation, validation_failures = generate_split('validation', VALIDATION_SEEDS, inp_path, profile)
    dataset = pd.DataFrame(train + validation)
    dataset.to_csv(args.output / 'residual_classifier_development.csv', index=False)
    failures = train_failures + validation_failures
    manifest = {'network': profile.name, 'training_cases': len(train), 'validation_cases': len(validation), 'training_seeds': TRAIN_SEEDS, 'validation_seeds': VALIDATION_SEEDS, 'test_cases_generated_here': False, 'test_set': 'a separate frozen Experiment 2 seed-1009 benchmark', 'source_head_in_development_data': False, 'same_family_target_overlap_with_frozen_test': False, 'failures': failures}
    (args.output / 'development_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    print(dataset.groupby(['split', 'true_cause']).size().to_string())
if __name__ == '__main__':
    main()
