from experiments.shared.evaluate_anytown_causal_hybrid_v2 import choose_relative_hypothesis

def item(score: float, train_improvement: float, valid_improvement: float):
    return {'validation_score_after': score, 'training_improvement': train_improvement, 'validation_improvement': valid_improvement}

def test_nominal_consistency_returns_no_fault_for_weak_evidence():
    evidence = {'leak': item(1.0, 0.01, 0.01), 'demand_error': item(1.01, 0.0, 0.0)}
    decision, reason, _, _ = choose_relative_hypothesis(evidence, 1.1, 1.2)
    assert decision == 'no_fault'
    assert reason == 'nominal_noise_consistent'

def test_strong_held_out_winner_is_accepted_despite_absolute_mismatch():
    evidence = {'leak': item(0.7, 0.2, 0.25), 'demand_error': item(1.0, 0.1, 0.1)}
    decision, reason, _, margin = choose_relative_hypothesis(evidence, 8.0, 2.0)
    assert decision == 'leak'
    assert reason == 'relative_held_out_winner'
    assert margin > 0.02

def test_tied_hypotheses_are_not_forced_to_single_cause():
    evidence = {'leak': item(0.99, 0.2, 0.2), 'roughness_error': item(1.0, 0.2, 0.2)}
    decision, reason, _, _ = choose_relative_hypothesis(evidence, 4.0, 2.0)
    assert decision == 'ambiguous_causes'
    assert reason == 'held_out_hypotheses_tied'

def test_weak_inconsistent_candidate_returns_unknown():
    evidence = {'leak': item(0.8, 0.01, 0.2), 'roughness_error': item(1.0, 0.1, 0.1)}
    decision, reason, _, _ = choose_relative_hypothesis(evidence, 4.0, 2.0)
    assert decision == 'unknown_cause'
    assert reason == 'weak_cross_split_evidence'
