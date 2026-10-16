"""Train residual-only classical baselines and evaluate the frozen Net3 cases."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
DIAGNOSTIC_FAMILIES = {'roughness_error', 'demand_error', 'leak', 'sensor_bias'}

def flatten_summary(summary: dict) -> dict[str, float]:
    direction = str(summary['pressure_direction'])
    row = {'estimated_onset_hour': float(summary['estimated_onset_hour']), 'affected_sensor_count': float(summary['affected_sensor_count']), 'flow_noise_ratio': float(summary['flow_noise_ratio']), 'flow_changed': float(bool(summary['flow_changed'])), 'direction_negative': float(direction == 'mostly_negative'), 'direction_positive': float(direction == 'mostly_positive'), 'direction_mixed': float(direction in {'mixed', 'mixed_or_small'})}
    for sensor, values in summary['pressure_residual_by_sensor'].items():
        for source, suffix in [('mean_m', 'mean_m'), ('rmse_m', 'rmse_m'), ('minimum_m', 'minimum_m'), ('maximum_m', 'maximum_m')]:
            row[f'pressure_{sensor}_{suffix}'] = float(values[source])
    return row

def choose_rejection_threshold(probabilities: np.ndarray, coverage: float=0.95) -> float:
    confidence = probabilities.max(axis=1)
    return float(np.quantile(confidence, 1.0 - coverage, method='lower'))

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    exact = y_true == y_pred
    known = np.isin(y_true, list(DIAGNOSTIC_FAMILIES))
    unknown = y_true == 'unknown_cause'
    no_fault = y_true == 'no_fault'
    accepted = np.isin(y_pred, list(DIAGNOSTIC_FAMILIES))
    return {'cases': int(len(y_true)), 'exact_outcome_accuracy': float(exact.mean()), 'known_family_accuracy': float(exact[known].mean()), 'accepted_diagnosis_precision': float(exact[accepted].mean()) if accepted.any() else np.nan, 'unknown_cause_recall': float(exact[unknown].mean()), 'no_fault_recall': float(exact[no_fault].mean()), 'false_attribution_rate': float((accepted & ~exact).mean()), 'accepted_diagnosis_rate': float(accepted.mean())}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--network-name', default='EPANET Net3')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(args.development)
    truth = pd.read_csv(args.truth)
    truth = truth.loc[truth['tier'] == 'matched_model'].set_index('neutral_case_id')
    test_rows = []
    for path in sorted(args.test_dir.glob('case_*/blind_input.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        case_id = payload['neutral_case_id']
        if case_id not in truth.index:
            continue
        meta = truth.loc[case_id]
        row = flatten_summary(payload['user_prompt']['residual_summary'])
        row.update(neutral_case_id=case_id, scenario_id=meta['scenario_id'], true_cause=meta['true_cause'], expected_label=meta['expected_label'])
        test_rows.append(row)
    test = pd.DataFrame(test_rows)
    if len(test) != 21:
        raise RuntimeError(f'Expected 21 frozen Net3 test cases, found {len(test)}')
    metadata = {'split', 'scenario_id', 'seed', 'true_cause', 'target', 'magnitude', 'start_hour'}
    features = [column for column in dev.columns if column not in metadata]
    missing = sorted(set(features) - set(test.columns))
    if missing:
        raise RuntimeError(f'Test cases are missing features: {missing}')
    train = dev.loc[dev['split'] == 'train']
    validation = dev.loc[dev['split'] == 'validation']
    X_train, y_train = (train[features], train['true_cause'].to_numpy())
    X_validation = validation[features]
    X_test, y_test = (test[features], test['expected_label'].to_numpy())
    estimators = {'Majority class': DummyClassifier(strategy='most_frequent'), 'Multinomial logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced', random_state=2026)), 'Random forest': RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced_subsample', random_state=2026, n_jobs=1)}
    summary_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for name, estimator in estimators.items():
        model = clone(estimator).fit(X_train, y_train)
        validation_probability = model.predict_proba(X_validation)
        threshold = choose_rejection_threshold(validation_probability)
        probability = model.predict_proba(X_test)
        raw = model.classes_[probability.argmax(axis=1)]
        confidence = probability.max(axis=1)
        prediction = raw.copy()
        prediction[confidence < threshold] = 'unknown_cause'
        row = metrics(y_test, prediction)
        row.update(method=name, rejection_threshold=threshold, validation_known_coverage=float((validation_probability.max(axis=1) >= threshold).mean()))
        summary_rows.append(row)
        for index, case in test.reset_index(drop=True).iterrows():
            decision_rows.append({'method': name, 'neutral_case_id': case['neutral_case_id'], 'scenario_id': case['scenario_id'], 'true_cause': case['true_cause'], 'expected_label': case['expected_label'], 'raw_prediction': raw[index], 'confidence': confidence[index], 'predicted_label': prediction[index], 'exact_outcome_correct': bool(prediction[index] == y_test[index])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / 'classical_baseline_summary.csv', index=False)
    pd.DataFrame(decision_rows).to_csv(args.output / 'classical_baseline_decisions.csv', index=False)
    manifest = {'network': args.network_name, 'training_cases': int(len(train)), 'validation_cases': int(len(validation)), 'test_cases': int(len(test)), 'source_head_in_training': False, 'test_target_overlap': False}
    (args.output / 'classical_baseline_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))
if __name__ == '__main__':
    main()
