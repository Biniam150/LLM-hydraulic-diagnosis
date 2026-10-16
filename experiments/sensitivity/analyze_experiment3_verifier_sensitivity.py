"""Full-factorial sensitivity analysis for the generalized hydraulic verifier."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from experiments.experiment_3_hydraulic_verification.evaluate_anytown_hybrid_v3_adequacy import validation_gof
from experiments.shared.run_hydraulic_diagnosis_demo import PROJECT_DIR
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_experiment import FaultSpec, ObservationConfig, generate_pseudo_observations, nominal_prediction
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'sensitivity/experiment3_verifier_sensitivity'
FAMILIES = ['leak', 'demand_error', 'roughness_error', 'sensor_bias']
NETWORKS = {'Anytown': {'key': 'anytown', 'profile': ROOT / 'configs' / 'networks/anytown.json', 'design': ROOT / 'configs' / 'scenarios/anytown_causal_benchmark.json'}, 'Net3': {'key': 'net3', 'profile': ROOT / 'configs' / 'networks/net3.json', 'design': ROOT / 'configs' / 'scenarios/net3_causal_replication.json'}, 'BWN-II': {'key': 'bwnii', 'profile': ROOT / 'configs' / 'networks/bwnii.json', 'design': ROOT / 'configs' / 'scenarios/bwnii_generalized_external_test.json'}}
DEFAULTS = {'improvement_floor': 0.02, 'margin_floor': 0.07, 'gof_alpha': 0.05, 'noise_scale': 1.0}
GRIDS = {'improvement_floor': [0.0, 0.01, 0.02, 0.03, 0.05, 0.1], 'margin_floor': [0.0, 0.03, 0.07, 0.1, 0.15, 0.2], 'gof_alpha': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15], 'noise_scale': [0.75, 0.9, 1.0, 1.1, 1.25, 1.5]}

def make_config(profile, seed: int) -> ObservationConfig:
    return ObservationConfig(pressure_sensors=list(profile.pressure_sensors), flow_link=profile.flow_link, pressure_noise_std_m=profile.pressure_noise_std_m, flow_noise_fraction=profile.flow_noise_fraction, duration_hours=profile.duration_hours, seed=seed, solver=profile.solver, demand_model=profile.demand_model, feature_policy=profile.feature_policy, search_policy=profile.search_policy, evidence_policy=profile.evidence_policy, candidate_cache_dir=None)

def load_physical_evidence() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for network, settings in NETWORKS.items():
        key = settings['key']
        result_path = ROOT / 'results' / f'experiment_3/experiment3_{key}_generalized_verifier_v4' / 'hybrid_v3_results.csv'
        evidence_root = result_path.parent / 'case_evidence'
        result = pd.read_csv(result_path).drop_duplicates(['scenario_id', 'observation_seed'])
        design = json.loads(Path(settings['design']).read_text(encoding='utf-8'))
        scenarios = {item['id']: item['fault'] for item in design['scenarios']}
        profile = load_network_profile(Path(settings['profile']))
        inp_path = PROJECT_DIR / profile.inp_file
        for case in result.itertuples(index=False):
            seed = int(case.observation_seed)
            config = make_config(profile, seed)
            fault = FaultSpec(**scenarios[case.scenario_id])
            observed, _, _ = generate_pseudo_observations(inp_path, fault, config)
            nominal = nominal_prediction(inp_path, config)
            _, nominal_stat, nominal_df = validation_gof(observed, nominal, nominal, config, list(range(1, config.duration_hours, 2)))
            evidence_path = evidence_root / f'{case.scenario_id}_seed_{seed}' / 'refined_family_evidence.json'
            evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
            rows.append({'network': network, 'scenario_id': case.scenario_id, 'observation_seed': seed, 'true_cause': case.true_cause, 'expected_label': 'unknown_cause' if case.true_cause == 'source_head' else case.true_cause, 'nominal_gof_statistic': nominal_stat, 'nominal_gof_df': nominal_df, 'family_evidence': evidence})
    return pd.DataFrame(rows)

def adjusted_pvalue(statistic: float, degrees_of_freedom: int, noise_scale: float) -> float:
    return float(chi2.sf(statistic / noise_scale ** 2, degrees_of_freedom))

def decide(case: pd.Series, policy: dict[str, float]) -> str:
    evidence = case['family_evidence']
    ranked = sorted(FAMILIES, key=lambda f: float(evidence[f]['validation_score_after']))
    supported = []
    for family in ranked:
        item = evidence[family]
        balanced = min(float(item['training_improvement']), float(item['validation_improvement']))
        pvalue = adjusted_pvalue(float(item['validation_gof_statistic']), int(item['validation_gof_df']), policy['noise_scale'])
        if balanced >= policy['improvement_floor'] and pvalue >= policy['gof_alpha']:
            supported.append(family)
    if not supported:
        nominal_p = adjusted_pvalue(float(case['nominal_gof_statistic']), int(case['nominal_gof_df']), policy['noise_scale'])
        return 'no_fault' if nominal_p >= policy['gof_alpha'] else 'unknown_cause'
    if len(supported) == 1:
        return supported[0]
    winner, runner = supported[:2]
    winner_score = float(evidence[winner]['validation_score_after'])
    runner_score = float(evidence[runner]['validation_score_after'])
    margin = (runner_score - winner_score) / runner_score if runner_score > 0 else 0.0
    return 'ambiguous_causes' if margin < policy['margin_floor'] else winner

def metrics(data: pd.DataFrame, policy: dict[str, float]) -> list[dict[str, object]]:
    output = []
    for network, cases in data.groupby('network', sort=False):
        predicted = cases.apply(decide, axis=1, policy=policy)
        true = cases['true_cause'].reset_index(drop=True)
        expected = cases['expected_label'].reset_index(drop=True)
        predicted = predicted.reset_index(drop=True)
        accepted = predicted.isin(FAMILIES)
        known = true.isin(FAMILIES)
        unknown = true.eq('source_head')
        no_fault = true.eq('no_fault')
        accepted_correct = predicted[accepted].eq(true[accepted])
        output.append({'network': network, **policy, 'cases': len(cases), 'exact_outcome_accuracy': predicted.eq(expected).mean(), 'known_family_accuracy': predicted[known].eq(true[known]).mean(), 'accepted_diagnosis_precision': accepted_correct.mean() if accepted.any() else np.nan, 'diagnostic_coverage': accepted.mean(), 'false_attribution_rate': (accepted & predicted.ne(true)).mean(), 'unknown_cause_recall': predicted[unknown].eq('unknown_cause').mean(), 'no_fault_recall': predicted[no_fault].eq('no_fault').mean(), 'ambiguity_rate': predicted.eq('ambiguous_causes').mean(), 'unknown_output_rate': predicted.eq('unknown_cause').mean()})
    return output

def plot_boxplots(factorial: pd.DataFrame) -> None:
    parameter_labels = {'improvement_floor': 'Minimum split improvement', 'margin_floor': 'Minimum winner margin', 'gof_alpha': 'GOF significance level', 'noise_scale': 'Assumed noise multiplier'}
    metric_labels = {'exact_outcome_accuracy': 'Exact-outcome accuracy (%)', 'accepted_diagnosis_precision': 'Accepted-diagnosis precision (%)', 'diagnostic_coverage': 'Diagnostic coverage (%)'}
    fig, axes = plt.subplots(3, 4, figsize=(13.0, 8.5), constrained_layout=True)
    for col, (parameter, parameter_label) in enumerate(parameter_labels.items()):
        levels = GRIDS[parameter]
        for row, (metric, metric_label) in enumerate(metric_labels.items()):
            ax = axes[row, col]
            groups = [100 * factorial.loc[np.isclose(factorial[parameter], level), metric].dropna() for level in levels]
            bp = ax.boxplot(groups, positions=np.arange(len(levels)), widths=0.58, patch_artist=True, showfliers=True, medianprops={'color': '#222222', 'linewidth': 1.2}, boxprops={'facecolor': '#d9e8f5', 'edgecolor': '#496b88'}, whiskerprops={'color': '#666666'}, capprops={'color': '#666666'}, flierprops={'marker': '.', 'markersize': 2.5, 'markerfacecolor': '#777777', 'markeredgecolor': '#777777', 'alpha': 0.4})
            del bp
            default_index = levels.index(DEFAULTS[parameter])
            ax.scatter(default_index, 102, marker='v', s=28, color='#b22222', clip_on=False, zorder=5)
            ax.set_xticks(np.arange(len(levels)), [f'{x:g}' for x in levels], rotation=0)
            ax.set_ylim(-3, 105)
            ax.grid(axis='y', alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)
            if row == 0:
                ax.set_title(parameter_label, fontsize=10.5, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric_label)
            if row == 2:
                ax.set_xlabel('Parameter value')
    fig.text(0.5, 1.012, 'Red marker indicates the frozen verifier setting', ha='center', fontsize=9)
    fig.savefig(OUT / 'experiment3_verifier_sensitivity_boxplots.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'experiment3_verifier_sensitivity_boxplots.pdf', bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    physical = load_physical_evidence()
    serializable = physical.drop(columns=['family_evidence']).copy()
    serializable.to_csv(OUT / 'physical_case_gof_evidence.csv', index=False)
    factorial_rows: list[dict[str, object]] = []
    for values in itertools.product(*(GRIDS[name] for name in GRIDS)):
        policy = dict(zip(GRIDS, values))
        factorial_rows.extend(metrics(physical, policy))
    factorial = pd.DataFrame(factorial_rows)
    factorial.to_csv(OUT / 'full_factorial_sensitivity.csv', index=False)
    oat_rows: list[dict[str, object]] = []
    for parameter, levels in GRIDS.items():
        for level in levels:
            policy = dict(DEFAULTS)
            policy[parameter] = level
            for row in metrics(physical, policy):
                oat_rows.append({'varied_parameter': parameter, 'varied_value': level, **row})
    oat = pd.DataFrame(oat_rows)
    oat.to_csv(OUT / 'one_at_a_time_sensitivity.csv', index=False)
    baseline = pd.DataFrame(metrics(physical, DEFAULTS))
    baseline.to_csv(OUT / 'baseline_reproduction.csv', index=False)
    plot_boxplots(factorial)
    summary = {'scope': 'decision-policy sensitivity using fixed fitted hydraulic evidence', 'physical_cases': int(len(physical)), 'networks': list(NETWORKS), 'defaults': DEFAULTS, 'grids': GRIDS, 'factorial_policy_combinations': int(np.prod([len(v) for v in GRIDS.values()])), 'factorial_network_evaluations': int(len(factorial)), 'baseline_reproduction': baseline.to_dict('records')}
    (OUT / 'sensitivity_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(baseline.to_string(index=False))
    print(f'factorial rows={len(factorial)}')
if __name__ == '__main__':
    main()
