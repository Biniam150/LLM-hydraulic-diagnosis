from experiments.shared.evaluate_anytown_hybrid_initial_boundaries import decide

def item(score: float, train: float, validation: float) -> dict[str, object]:
    return {'validation_score_after': score, 'training_improvement': train, 'validation_improvement': validation}

def evidence(winner_score=0.5, runner_score=0.7, improvement=0.2):
    return {'leak': item(winner_score, improvement, improvement), 'demand_error': item(runner_score, 0.1, 0.1), 'roughness_error': item(0.9, 0.05, 0.05), 'sensor_bias': item(1.0, 0.04, 0.04)}

def test_unique_cross_split_winner_is_accepted():
    outcome, reason, *_ = decide(evidence(), 4.0, 2.0)
    assert outcome == 'leak'
    assert reason == 'unique_cross_split_winner'

def test_small_winner_margin_returns_ambiguity():
    outcome, reason, *_ = decide(evidence(0.66, 0.7), 4.0, 2.0)
    assert outcome == 'ambiguous_causes'
    assert reason == 'held_out_hypotheses_tied'

def test_weak_evidence_and_noise_consistency_returns_no_fault():
    outcome, reason, *_ = decide(evidence(improvement=0.01), 1.2, 1.1)
    assert outcome == 'no_fault'
    assert reason == 'nominal_noise_consistent'

def test_weak_evidence_and_nominal_mismatch_returns_unknown():
    outcome, reason, *_ = decide(evidence(improvement=0.01), 3.0, 2.0)
    assert outcome == 'unknown_cause'
    assert reason == 'weak_cross_split_evidence'
