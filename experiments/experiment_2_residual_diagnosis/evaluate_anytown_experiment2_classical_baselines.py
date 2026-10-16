"""Evaluate classical residual-only baselines on the frozen Anytown Experiment 2 set.

Development simulations are disjoint from the frozen test targets. Source-head
error is absent from development data and is therefore an unknown mechanism.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[2]
DEV_CSV = ROOT / 'results/experiment_2/experiment2_anytown_classical_baselines/residual_classifier_development.csv'
TEST_DIR = ROOT / 'results/experiment_2/experiment2_anytown_qwen3_8b_seed1009'
TEST_RESULTS = TEST_DIR / 'llm_only_causal_results.csv'
LLM_SUMMARY = ROOT / 'results/experiment_2/experiment2_anytown_five_models_seed1009/five_model_summary.csv'
OUT_DIR = ROOT / 'results/experiment_2/experiment2_anytown_classical_baselines'
FIG_DIR = ROOT / 'figures/main'
KNOWN = ['roughness_error', 'demand_error', 'leak', 'sensor_bias', 'no_fault']
DIAGNOSTIC_FAMILIES = set(KNOWN[:-1])
PRETTY = {'roughness_error': 'Roughness', 'demand_error': 'Demand error', 'leak': 'Leakage', 'sensor_bias': 'Sensor bias', 'no_fault': 'No fault', 'unknown_cause': 'Unknown cause'}

def flatten_summary(summary: dict) -> dict[str, float]:
    direction = summary['pressure_direction']
    row = {'estimated_onset_hour': float(summary['estimated_onset_hour']), 'affected_sensor_count': float(summary['affected_sensor_count']), 'flow_noise_ratio': float(summary['flow_noise_ratio']), 'flow_changed': float(bool(summary['flow_changed'])), 'direction_negative': float(direction == 'mostly_negative'), 'direction_positive': float(direction == 'mostly_positive'), 'direction_mixed': float(direction == 'mixed')}
    for sensor, values in summary['pressure_residual_by_sensor'].items():
        for source, suffix in [('mean_m', 'mean_m'), ('rmse_m', 'rmse_m'), ('minimum_m', 'minimum_m'), ('maximum_m', 'maximum_m')]:
            row[f'pressure_{sensor}_{suffix}'] = float(values[source])
    return row

def load_frozen_test() -> pd.DataFrame:
    truth = pd.read_csv(TEST_RESULTS).set_index('neutral_case_id')
    rows = []
    for path in sorted(TEST_DIR.glob('case_*/blind_input.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        case_id = payload['neutral_case_id']
        meta = truth.loc[case_id]
        row = flatten_summary(payload['user_prompt']['residual_summary'])
        row.update(neutral_case_id=case_id, scenario_id=meta['scenario_id'], true_cause=meta['true_cause'], true_target=str(meta['true_target']), expected_label=meta['expected_label'])
        rows.append(row)
    test = pd.DataFrame(rows)
    if len(test) != 18:
        raise RuntimeError(f'Expected 18 frozen test cases, found {len(test)}')
    return test

def choose_rejection_threshold(probabilities: np.ndarray, coverage: float=0.95) -> float:
    """Set confidence cutoff using known validation data only.

    The lower (1-coverage) quantile retains approximately 95% of validation
    cases. No source-head or other unknown case is used to tune this value.
    """
    confidence = probabilities.max(axis=1)
    return float(np.quantile(confidence, 1.0 - coverage, method='lower'))

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    exact = y_true == y_pred
    known_mask = np.isin(y_true, list(DIAGNOSTIC_FAMILIES))
    unknown_mask = y_true == 'unknown_cause'
    no_fault_mask = y_true == 'no_fault'
    accepted = np.isin(y_pred, list(DIAGNOSTIC_FAMILIES))
    return {'cases': int(len(y_true)), 'exact_outcome_accuracy': float(exact.mean()), 'known_family_accuracy': float(exact[known_mask].mean()), 'known_family_coverage': float(accepted[known_mask].mean()), 'accepted_diagnosis_precision': float(exact[accepted].mean()) if accepted.any() else np.nan, 'unknown_cause_recall': float(exact[unknown_mask].mean()), 'no_fault_recall': float(exact[no_fault_mask].mean()), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0))}

def wilson_interval(successes: int, trials: int, z: float=1.96) -> tuple[float, float]:
    if trials == 0:
        return (np.nan, np.nan)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    center = (p + z ** 2 / (2 * trials)) / denom
    spread = z * np.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denom
    return (center - spread, center + spread)

def make_figure(summary: pd.DataFrame, path: Path) -> None:
    methods = summary['method'].tolist()
    y = np.arange(len(methods))[::-1]
    colors = ['#8a8a8a' if group == 'LLM only' else '#1676a9' for group in summary['group']]
    specs = [('exact_outcome_accuracy', 'Exact outcome accuracy', 18), ('known_family_accuracy', 'Known-family accuracy', 14), ('unknown_cause_recall', 'Unknown-cause recall', 3)]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.9), sharey=True)
    for ax, (column, title, trials) in zip(axes, specs):
        values = summary[column].to_numpy(float)
        lower, upper = ([], [])
        for value in values:
            lo, hi = wilson_interval(int(round(value * trials)), trials)
            lower.append(value - lo)
            upper.append(hi - value)
        ax.errorbar(values * 100, y, xerr=np.array([lower, upper]) * 100, fmt='none', ecolor='#9b9b9b', elinewidth=1.2, capsize=2.5, zorder=1)
        ax.scatter(values * 100, y, c=colors, s=43, edgecolor='white', linewidth=0.6, zorder=2)
        for value, yy in zip(values, y):
            ax.text(min(value * 100 + 2.4, 95), yy, f'{value * 100:.1f}', va='center', fontsize=8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(-3, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Cases (%)')
        ax.grid(axis='x', color='#dddddd', linewidth=0.7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y, methods)
    axes[0].scatter([], [], c='#8a8a8a', s=43, label='LLM only')
    axes[0].scatter([], [], c='#1676a9', s=43, label='Classical baseline')
    fig.legend(loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(DEV_CSV)
    test = load_frozen_test()
    metadata = {'split', 'scenario_id', 'seed', 'true_cause', 'target', 'magnitude', 'start_hour'}
    feature_columns = [c for c in dev.columns if c not in metadata]
    missing = sorted(set(feature_columns) - set(test.columns))
    if missing:
        raise RuntimeError(f'Frozen test is missing features: {missing}')
    train = dev[dev['split'] == 'train']
    validation = dev[dev['split'] == 'validation']
    X_train, y_train = (train[feature_columns], train['true_cause'].to_numpy())
    X_val, y_val = (validation[feature_columns], validation['true_cause'].to_numpy())
    X_test, y_test = (test[feature_columns], test['expected_label'].to_numpy())
    models = {'Majority class': DummyClassifier(strategy='most_frequent'), 'Multinomial logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced', random_state=2026)), 'Random forest': RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight='balanced_subsample', random_state=2026, n_jobs=1)}
    metrics_rows, decision_rows, validation_rows = ([], [], [])
    for name, estimator in models.items():
        model = clone(estimator).fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)
        threshold = choose_rejection_threshold(val_prob, coverage=0.95)
        val_classes = model.classes_
        val_argmax = val_prob.argmax(axis=1)
        val_pred = val_classes[val_argmax]
        val_conf = val_prob.max(axis=1)
        val_rejected = val_conf < threshold
        val_pred_with_rejection = val_pred.copy()
        val_pred_with_rejection[val_rejected] = 'unknown_cause'
        probability = model.predict_proba(X_test)
        classes = model.classes_
        argmax = probability.argmax(axis=1)
        raw_pred = classes[argmax]
        confidence = probability.max(axis=1)
        rejected = confidence < threshold
        prediction = raw_pred.copy()
        prediction[rejected] = 'unknown_cause'
        metric = calculate_metrics(y_test, prediction)
        metric.update(method=name, group='Classical baseline', rejection_threshold=threshold, validation_known_coverage=float((~val_rejected).mean()), validation_accuracy=float((val_pred_with_rejection == y_val).mean()))
        metrics_rows.append(metric)
        for i, case in test.reset_index(drop=True).iterrows():
            decision_rows.append({'method': name, 'neutral_case_id': case['neutral_case_id'], 'scenario_id': case['scenario_id'], 'true_cause': case['true_cause'], 'true_target': case['true_target'], 'expected_label': case['expected_label'], 'raw_prediction': raw_pred[i], 'confidence': confidence[i], 'rejection_threshold': threshold, 'rejected_as_unknown': bool(rejected[i]), 'predicted_label': prediction[i], 'exact_outcome_correct': bool(prediction[i] == y_test[i])})
        validation_rows.append({'method': name, 'threshold': threshold, 'known_validation_cases': len(validation), 'retained_cases': int((~val_rejected).sum()), 'retained_fraction': float((~val_rejected).mean()), 'accuracy_after_rejection': float((val_pred_with_rejection == y_val).mean())})
    classical = pd.DataFrame(metrics_rows)
    llm = pd.read_csv(LLM_SUMMARY).rename(columns={'model': 'method'})
    llm['group'] = 'LLM only'
    keep = ['method', 'group', 'cases', 'exact_outcome_accuracy', 'known_family_accuracy', 'known_family_coverage', 'accepted_diagnosis_precision', 'unknown_cause_recall', 'no_fault_recall']
    comparison = pd.concat([llm[keep], classical[keep]], ignore_index=True)
    order = ['Qwen2.5-7B', 'Llama3.1-8B', 'Qwen3-8B', 'Ministral3-8B', 'Qwen3.5-9B', 'Majority class', 'Multinomial logistic', 'Random forest']
    comparison['_order'] = comparison['method'].map({m: i for i, m in enumerate(order)})
    comparison = comparison.sort_values('_order').drop(columns='_order')
    classical.to_csv(OUT_DIR / 'classical_baseline_summary.csv', index=False)
    pd.DataFrame(decision_rows).to_csv(OUT_DIR / 'classical_baseline_decisions.csv', index=False)
    pd.DataFrame(validation_rows).to_csv(OUT_DIR / 'unknown_rejection_calibration.csv', index=False)
    comparison.to_csv(OUT_DIR / 'llm_classical_comparison.csv', index=False)
    test[['neutral_case_id', 'scenario_id', 'true_cause', 'true_target', 'expected_label'] + feature_columns].to_csv(OUT_DIR / 'frozen_test_features.csv', index=False)
    make_figure(comparison, FIG_DIR / 'anytown_experiment2_llm_classical_comparison.png')
    print('Development class counts:')
    print(dev.groupby(['split', 'true_cause']).size())
    print('\nClassical baseline results:')
    print(classical[['method', 'exact_outcome_accuracy', 'known_family_accuracy', 'accepted_diagnosis_precision', 'unknown_cause_recall', 'no_fault_recall', 'rejection_threshold', 'validation_known_coverage']].to_string(index=False))
if __name__ == '__main__':
    main()
