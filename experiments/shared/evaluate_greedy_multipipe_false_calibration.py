from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.shared.run_hydraulic_diagnosis_demo import DEFAULT_PROFILE, PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, generate_pseudo_observations, load_hydraulic_network, nominal_prediction, results_to_sensor_frame, run_hydraulics, validate_observation_config

def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

def pressure_mae(observed: pd.DataFrame, predicted: pd.DataFrame, hours: list[int]) -> float:
    columns = [column for column in observed if column.startswith('pressure_')]
    if not set(hours).issubset(predicted.index) or not set(columns).issubset(predicted.columns):
        return float('inf')
    difference = observed.loc[hours, columns] - predicted.loc[hours, columns]
    values = np.abs(difference).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return float('inf')
    return float(values.mean())

class RoughnessPredictionCache:

    def __init__(self, inp_path: Path, config: ObservationConfig):
        self.inp_path = inp_path
        self.config = config
        base = load_hydraulic_network(inp_path, config.duration_hours, config.demand_model)
        validate_observation_config(base, config)
        self.base_roughness = {pipe: float(base.get_link(pipe).roughness) for pipe in base.pipe_name_list}
        self.pipe_names = list(base.pipe_name_list)
        self.frames: dict[tuple[tuple[str, float], ...], pd.DataFrame] = {}
        self.simulations = 0

    def predict(self, settings: dict[str, float]) -> pd.DataFrame:
        key = tuple(sorted(((pipe, float(value)) for pipe, value in settings.items())))
        if key in self.frames:
            return self.frames[key]
        network = load_hydraulic_network(self.inp_path, self.config.duration_hours, self.config.demand_model)
        for pipe, multiplier in settings.items():
            network.get_link(pipe).roughness = self.base_roughness[pipe] * float(multiplier)
        frame = results_to_sensor_frame(run_hydraulics(network, self.config.solver), self.config)
        self.frames[key] = frame
        self.simulations += 1
        return frame

def greedy_calibrate(observed: pd.DataFrame, nominal: pd.DataFrame, cache: RoughnessPredictionCache, multipliers: list[float], maximum_adjusted_pipes: int, shortlist_size: int) -> tuple[dict[str, float], pd.DataFrame, list[dict[str, object]]]:
    train_hours = [hour for hour in observed.index if int(hour) % 2 == 0]
    validation_hours = [hour for hour in observed.index if int(hour) % 2 == 1]
    baseline_train = pressure_mae(observed, nominal, train_hours)
    baseline_validation = pressure_mae(observed, nominal, validation_hours)
    best_by_pipe: list[tuple[float, str, float, pd.DataFrame]] = []
    for pipe in cache.pipe_names:
        pipe_candidates = []
        for multiplier in multipliers:
            prediction = cache.predict({pipe: multiplier})
            score = pressure_mae(observed, prediction, train_hours)
            pipe_candidates.append((score, pipe, multiplier, prediction))
        best_by_pipe.append(min(pipe_candidates, key=lambda item: item[0]))
    best_by_pipe.sort(key=lambda item: item[0])
    shortlist = [item[1] for item in best_by_pipe[:shortlist_size]]
    settings: dict[str, float] = {}
    current_prediction = nominal
    current_train = baseline_train
    trajectory: list[dict[str, object]] = []
    for round_number in range(1, maximum_adjusted_pipes + 1):
        candidates = []
        for pipe in shortlist:
            if pipe in settings:
                continue
            for multiplier in multipliers:
                proposed = dict(settings)
                proposed[pipe] = multiplier
                prediction = cache.predict(proposed)
                train_score = pressure_mae(observed, prediction, train_hours)
                candidates.append((train_score, pipe, multiplier, prediction))
        if not candidates:
            break
        train_score, pipe, multiplier, prediction = min(candidates, key=lambda item: item[0])
        if train_score >= current_train - 1e-09:
            break
        settings[pipe] = float(multiplier)
        current_prediction = prediction
        current_train = train_score
        validation_score = pressure_mae(observed, current_prediction, validation_hours)
        trajectory.append({'round': round_number, 'added_pipe': pipe, 'roughness_multiplier': float(multiplier), 'adjusted_pipes': dict(settings), 'training_pressure_mae_m': current_train, 'validation_pressure_mae_m': validation_score, 'training_improvement': (baseline_train - current_train) / baseline_train if baseline_train > 0 else 0.0, 'validation_improvement': (baseline_validation - validation_score) / baseline_validation if baseline_validation > 0 else 0.0})
    return (settings, current_prediction, trajectory)

def main() -> None:
    parser = argparse.ArgumentParser(description='Run greedy multi-pipe false roughness calibration on AnyTown.')
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding='utf-8'))
    profile = load_network_profile(args.profile)
    inp_path = PROJECT_DIR / profile.inp_file
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    shared_cache: RoughnessPredictionCache | None = None
    for scenario in design['scenarios']:
        fault = FaultSpec(**scenario['fault'])
        for seed in design['seeds']:
            config = ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=int(seed), solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy)
            if shared_cache is None:
                shared_cache = RoughnessPredictionCache(inp_path, config)
            observed, clean, manifest = generate_pseudo_observations(inp_path, fault, config)
            nominal = nominal_prediction(inp_path, config)
            simulations_before = shared_cache.simulations
            settings, calibrated, trajectory = greedy_calibrate(observed, nominal, shared_cache, [float(value) for value in design['roughness_multipliers']], int(design['maximum_adjusted_pipes']), int(design['shortlist_pipes']))
            train_hours = [hour for hour in observed.index if int(hour) % 2 == 0]
            validation_hours = [hour for hour in observed.index if int(hour) % 2 == 1]
            train_before = pressure_mae(observed, nominal, train_hours)
            train_after = pressure_mae(observed, calibrated, train_hours)
            validation_before = pressure_mae(observed, nominal, validation_hours)
            validation_after = pressure_mae(observed, calibrated, validation_hours)
            train_improvement = (train_before - train_after) / train_before
            validation_improvement = (validation_before - validation_after) / validation_before
            wrong_cause = fault.fault_type not in {'roughness_error', 'no_fault'}
            false_calibration = bool(wrong_cause and validation_improvement >= float(design['false_calibration_min_validation_improvement']))
            case_dir = args.output / f"{scenario['id']}_seed_{seed}"
            case_dir.mkdir(parents=True, exist_ok=True)
            row = {'scenario_id': scenario['id'], 'seed': int(seed), 'true_cause': fault.fault_type, 'adjusted_pipe_count': len(settings), 'adjusted_pipes': json.dumps(settings, sort_keys=True), 'training_pressure_mae_before_m': train_before, 'training_pressure_mae_after_m': train_after, 'training_improvement': train_improvement, 'validation_pressure_mae_before_m': validation_before, 'validation_pressure_mae_after_m': validation_after, 'validation_improvement': validation_improvement, 'false_calibration': false_calibration, 'new_simulations': shared_cache.simulations - simulations_before}
            rows.append(row)
            calibrated.to_csv(case_dir / 'multipipe_roughness_prediction.csv', float_format='%.8f')
            write_json(case_dir / 'ground_truth_manifest.json', manifest)
            write_json(case_dir / 'calibration_trajectory.json', trajectory)
            write_json(case_dir / 'case_summary.json', row)
            print(f"{scenario['id']} seed={seed}: validation MAE {validation_before:.4f} -> {validation_after:.4f} ({validation_improvement:.1%}); pipes={settings}")
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'multipipe_false_calibration_results.csv', index=False)
    wrong = results[~results['true_cause'].isin(['roughness_error', 'no_fault'])]
    by_scenario = results.groupby(['scenario_id', 'true_cause'], as_index=False).agg(runs=('seed', 'size'), mean_training_improvement=('training_improvement', 'mean'), mean_validation_improvement=('validation_improvement', 'mean'), false_calibration_rate=('false_calibration', 'mean'), mean_adjusted_pipe_count=('adjusted_pipe_count', 'mean'))
    by_scenario.to_csv(args.output / 'multipipe_false_calibration_by_scenario.csv', index=False)
    summary = {'runs': int(len(results)), 'non_roughness_runs': int(len(wrong)), 'non_roughness_false_calibration_rate': float(wrong['false_calibration'].mean()), 'mean_non_roughness_training_improvement': float(wrong['training_improvement'].mean()), 'mean_non_roughness_validation_improvement': float(wrong['validation_improvement'].mean()), 'unique_hydraulic_simulations': int(shared_cache.simulations), 'design': design, 'important_scope': "Greedy pressure-only calibration with at most four individually adjusted pipes; this is a controlled synthetic stress test, not a replication of the paper's agent implementation."}
    write_json(args.output / 'multipipe_false_calibration_summary.json', summary)
    print(by_scenario.to_string(index=False))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
