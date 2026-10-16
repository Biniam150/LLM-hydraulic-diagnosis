from types import SimpleNamespace
from experiments.experiment_4_context_followup.verify_experiment4_operator_followups import context_gate, target_id
VARIANT = {'operational_records': [{'record_id': 'R1', 'reliability': 'high', 'time_relation': 'within the diagnostic observation window'}, {'record_id': 'R2', 'reliability': 'low', 'time_relation': 'within the diagnostic observation window'}]}

def proposal(**updates):
    value = {'action': 'propose_follow_up_test', 'proposed_family': 'source_head', 'proposed_target': 'Reservoir 10', 'cited_record_ids': 'R1', 'record_assessment': 'reliable_and_relevant'}
    value.update(updates)
    return SimpleNamespace(**value)

def test_normalizes_asset_descriptions():
    assert target_id('Pipe:34') == '34'
    assert target_id('Reservoir 165') == '165'
    assert target_id('34.0') == '34'

def test_accepts_reliable_time_relevant_context_configuration():
    accepted, _ = context_gate(proposal(), VARIANT)
    assert accepted

def test_accepts_csv_serialized_citation_list():
    accepted, _ = context_gate(proposal(cited_record_ids="['R1']"), VARIANT)
    assert accepted

def test_rejects_low_reliability_record_even_when_it_supports_a_test():
    accepted, reason = context_gate(proposal(cited_record_ids='R2'), VARIANT)
    assert not accepted
    assert reason == 'no_reliable_time_relevant_record'

def test_rejects_invalid_citation():
    accepted, reason = context_gate(proposal(cited_record_ids='R9'), VARIANT)
    assert not accepted
    assert reason == 'missing_or_invalid_citation'
