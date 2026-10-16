from experiments.experiment_3_hydraulic_verification.evaluate_anytown_hybrid_v3_adequacy import FAMILIES, decide

def evidence(winner_pvalue: float=0.1, tied: bool=False):
    scores = {'leak': 0.2, 'demand_error': 0.5, 'roughness_error': 0.7, 'sensor_bias': 0.9}
    if tied:
        scores['demand_error'] = 0.205
    return {family: {'validation_score_after': score, 'training_improvement': 0.5, 'validation_improvement': 0.5, 'validation_gof_pvalue': winner_pvalue if family == 'leak' else 0.1} for family, score in scores.items()}

def test_accepts_relative_winner_only_when_absolute_fit_is_adequate():
    result = decide(evidence(winner_pvalue=0.1), 3.0, 3.0, 1e-08)
    assert result['outcome'] == 'leak'

def test_rejects_best_available_family_when_absolute_fit_is_inadequate():
    candidate_evidence = evidence(winner_pvalue=1e-08)
    for family in FAMILIES:
        candidate_evidence[family]['validation_gof_pvalue'] = 1e-08
    result = decide(candidate_evidence, 3.0, 3.0, 1e-08)
    assert result['outcome'] == 'unknown_cause'

def test_preserves_ambiguity_before_absolute_acceptance():
    result = decide(evidence(winner_pvalue=0.1, tied=True), 3.0, 3.0, 1e-08)
    assert result['outcome'] == 'ambiguous_causes'

def test_inadequate_runner_up_cannot_create_ambiguity():
    candidate_evidence = evidence(winner_pvalue=0.1, tied=True)
    for family in FAMILIES:
        if family != 'leak':
            candidate_evidence[family]['validation_gof_pvalue'] = 1e-08
    result = decide(candidate_evidence, 3.0, 3.0, 1e-08)
    assert result['outcome'] == 'leak'
    assert result['reason'] == 'sole_cross_split_and_gof_supported_family'

def test_preserves_no_fault_when_nominal_residual_is_noise_consistent():
    weak = evidence()
    for family in FAMILIES:
        weak[family]['training_improvement'] = 0.01
        weak[family]['validation_improvement'] = 0.01
    result = decide(weak, 1.1, 1.2, 0.2)
    assert result['outcome'] == 'no_fault'

def test_significant_nominal_mismatch_cannot_be_declared_no_fault() -> None:
    weak = evidence(winner_pvalue=1e-08)
    for item in weak.values():
        item['training_improvement'] = 0.01
        item['validation_improvement'] = 0.01
    result = decide(weak, 1.1, 1.2, 1e-08)
    assert result['outcome'] == 'unknown_cause'
    assert result['reason'] == 'weak_cross_split_evidence'
