from __future__ import annotations
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from wdn_diagnostics.diagnostic_profile import load_network_profile
from wdn_diagnostics.hydraulic_diagnostic_agent import HydraulicDiagnosticAgent
from wdn_diagnostics.hydraulic_experiment import DEFAULT_PRESSURE_SENSORS, FaultSpec, ObservationConfig, _candidate_leak_start_hour, _candidate_nodes, _sensor_bias_bounds, apply_physical_fault, compute_residual_features, generate_pseudo_observations, load_hydraulic_network, nominal_prediction, test_hypothesis_family, topology_context
PROJECT_DIR = Path(__file__).resolve().parents[1]
INP_PATH = PROJECT_DIR / 'data/networks/anytown/anytown.inp'
PROFILE_PATH = PROJECT_DIR / 'configs' / 'networks/anytown.json'

class HydraulicDiagnosticExperimentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ObservationConfig(pressure_sensors=list(DEFAULT_PRESSURE_SENSORS), pressure_noise_std_m=0.03, flow_noise_fraction=0.01, seed=42)
        cls.nominal = nominal_prediction(INP_PATH, cls.config)

    def test_hidden_leak_has_detectable_onset(self) -> None:
        observed, _, _ = generate_pseudo_observations(INP_PATH, FaultSpec('leak', '90', 8, 24, 0.001), self.config)
        _, features = compute_residual_features(observed, self.nominal, self.config)
        self.assertEqual(features['estimated_onset_hour'], 8)
        self.assertGreaterEqual(features['affected_sensor_count'], 2)
        self.assertEqual(features['pressure_direction'], 'mostly_negative')

    def test_sensor_bias_is_local_and_does_not_change_pump_flow(self) -> None:
        observed, _, _ = generate_pseudo_observations(INP_PATH, FaultSpec('sensor_bias', '100', 8, 24, -0.8), self.config)
        _, features = compute_residual_features(observed, self.nominal, self.config)
        self.assertIn('100', features['affected_sensors'])
        self.assertFalse(features['pump_flow_changed'])

    def test_leak_family_recovers_hidden_parameters(self) -> None:
        observed, _, _ = generate_pseudo_observations(INP_PATH, FaultSpec('leak', '90', 8, 24, 0.001), self.config)
        evidence, _ = test_hypothesis_family('leak', INP_PATH, observed, self.nominal, self.config, onset_hour=8, requested_nodes=['70', '100', '150'])
        self.assertEqual(evidence.best_parameters['node'], '90')
        self.assertAlmostEqual(evidence.best_parameters['area_m2'], 0.001)
        self.assertEqual(evidence.result, 'strong_support')

    def test_roughness_search_cannot_be_pruned_by_wrong_llm_target(self) -> None:
        observed, _, _ = generate_pseudo_observations(INP_PATH, FaultSpec('roughness_error', '24', 0, 24, 0.6), self.config)
        evidence, _ = test_hypothesis_family('roughness_error', INP_PATH, observed, self.nominal, self.config, onset_hour=0, requested_nodes=['150'])
        self.assertEqual(evidence.best_parameters['pipe'], '24')
        self.assertAlmostEqual(evidence.best_parameters['multiplier'], 0.6)
        self.assertEqual(evidence.best_parameters['search_scope'], 'full_network')
        self.assertEqual(evidence.result, 'strong_support')

    def test_missing_llm_request_is_recorded_as_partial_fallback(self) -> None:
        agent = HydraulicDiagnosticAgent(mode='rule', max_rounds=3)
        parsed = {'ranking': [{'hypothesis': 'leak', 'score': 0.8, 'supporting_evidence': [], 'contradicting_evidence': []}], 'requested_test': None, 'decision': 'continue_testing', 'diagnosis': None, 'confidence': 0.5, 'rationale': ['More evidence is required']}
        features = {'affected_sensor_count': 3, 'pump_flow_changed': True, 'estimated_onset_hour': 8, 'pressure_direction': 'mostly_negative', 'largest_residual_sensors': ['100', '70', '150']}
        topology = {'valid_junctions': ['70', '100', '150'], 'valid_pipes': []}
        decision = agent._validate_decision(parsed, features, topology, evidence=[])
        self.assertTrue(decision.used_fallback)
        self.assertIsNotNone(decision.requested_test)
        self.assertIn('Rule-based fallback supplied', decision.rationale[-1])

    def test_network_profile_contains_all_anytown_specific_inputs(self) -> None:
        profile = load_network_profile(PROFILE_PATH)
        self.assertEqual(profile.flow_link, '82')
        self.assertEqual(profile.pressure_sensors, DEFAULT_PRESSURE_SENSORS)
        self.assertEqual(profile.search_policy.candidate_node_scope, 'full_network')
        self.assertIn('leak', profile.scenarios)
        self.assertEqual(set(profile.deterministic_test_order), {'leak', 'demand_error', 'roughness_error', 'sensor_bias'})

    def test_bwnii_profile_uses_epanet_without_replacing_anytown(self) -> None:
        bwnii = load_network_profile(PROJECT_DIR / 'configs' / 'networks/bwnii.json')
        anytown = load_network_profile(PROFILE_PATH)
        self.assertEqual(bwnii.solver, 'epanet')
        self.assertEqual(bwnii.demand_model, 'DDA')
        self.assertEqual(bwnii.flow_link, 'P316')
        self.assertEqual(len(bwnii.pressure_sensors), 12)
        self.assertEqual(bwnii.search_policy.sensor_bias_bound_mode, 'baseline_fraction')
        self.assertIsNone(bwnii.search_policy.sensor_bias_bounds_m)
        self.assertNotEqual(bwnii.inp_file, anytown.inp_file)
        self.assertTrue((PROJECT_DIR / bwnii.inp_file).exists())

    def test_bwnii_sensor_bias_bound_is_derived_from_baseline_pressure(self) -> None:
        bwnii = load_network_profile(PROJECT_DIR / 'configs' / 'networks/bwnii.json')
        config = ObservationConfig(pressure_sensors=['A', 'B'], pressure_noise_std_m=bwnii.pressure_noise_std_m, search_policy=bwnii.search_policy)
        nominal = pd.DataFrame({'pressure_A': [60.0, 62.0], 'pressure_B': [64.0, 66.0], 'flow_Q': [1.0, 1.0]})
        lower, upper, scale = _sensor_bias_bounds(nominal, config)
        self.assertAlmostEqual(scale, 63.0)
        self.assertAlmostEqual(lower, -3.15)
        self.assertAlmostEqual(upper, 3.15)

    def test_net3_profile_is_separate_and_uses_shared_evidence_policy(self) -> None:
        net3 = load_network_profile(PROJECT_DIR / 'configs' / 'networks/net3.json')
        bwnii = load_network_profile(PROJECT_DIR / 'configs' / 'networks/bwnii.json')
        self.assertEqual(net3.solver, 'epanet')
        self.assertEqual(net3.demand_model, 'DDA')
        self.assertEqual(net3.flow_link, '60')
        self.assertEqual(len(net3.pressure_sensors), 12)
        self.assertEqual(net3.pressure_sensors[0], '109')
        self.assertEqual(net3.evidence_policy, bwnii.evidence_policy)
        self.assertEqual(net3.search_policy, bwnii.search_policy)
        self.assertNotEqual(net3.inp_file, bwnii.inp_file)
        self.assertTrue((PROJECT_DIR / net3.inp_file).exists())

    def test_epanet_leak_adapter_uses_pressure_dependent_emitter(self) -> None:
        bwnii = load_network_profile(PROJECT_DIR / 'configs' / 'networks/bwnii.json')
        wn = load_hydraulic_network(PROJECT_DIR / bwnii.inp_file, bwnii.duration_hours, bwnii.demand_model)
        scenario = bwnii.scenarios['leak']
        apply_physical_fault(wn, FaultSpec(scenario.fault_type, scenario.target, scenario.start_hour, scenario.end_hour, scenario.magnitude), solver=bwnii.solver)
        self.assertGreater(wn.get_node(scenario.target).emitter_coefficient, 0.0)

    def test_epanet_leak_search_uses_persistent_supported_onset(self) -> None:
        self.assertEqual(_candidate_leak_start_hour(8, 'epanet'), 0)
        self.assertEqual(_candidate_leak_start_hour(8, 'wntr'), 8)

    def test_test_costs_are_derived_from_network_and_search_policy(self) -> None:
        context = topology_context(INP_PATH, self.config.pressure_sensors, self.config)
        costs = context['deterministic_test_costs']
        self.assertEqual(costs['sensor_bias']['estimated_hydraulic_simulations'], 0)
        self.assertEqual(costs['leak']['estimated_hydraulic_simulations'], context['junction_count'] * len(self.config.search_policy.leak_areas_m2))
        self.assertEqual(costs['roughness_error']['estimated_hydraulic_simulations'], context['pipe_count'] * len(self.config.search_policy.roughness_multipliers))

    def test_llm_gets_one_repair_attempt_for_missing_hypothesis(self) -> None:
        first_response = {'ranking': [{'hypothesis': 'sensor_bias', 'score': 0.8, 'supporting_evidence': ['localized residual'], 'contradicting_evidence': []}], 'requested_test': {'hypothesis': None, 'target_nodes': ['100'], 'target_links': [], 'reason': 'Test the localized sensor'}, 'decision': 'continue_testing', 'diagnosis': None, 'confidence': 0.8, 'rationale': ['A deterministic test is required']}
        repaired_response = {**first_response, 'requested_test': {'hypothesis': 'sensor_bias', 'target_nodes': ['100'], 'target_links': [], 'reason': 'Test the localized sensor'}}

        class StubClient:

            def __init__(self) -> None:
                self.responses = [json.dumps(first_response), json.dumps(repaired_response)]
                self.calls = []

            def chat(self, system_prompt, user_prompt, **kwargs):
                self.calls.append((system_prompt, user_prompt))
                return self.responses.pop(0)
        topology = {'valid_junctions': ['100'], 'valid_pipes': ['24'], 'deterministic_test_costs': {'sensor_bias': {'estimated_hydraulic_simulations': 0}}}
        agent = HydraulicDiagnosticAgent(mode='ollama', max_rounds=4)
        stub = StubClient()
        agent.client = stub
        decision, raw = agent.decide(features={'largest_residual_sensors': ['100']}, topology=topology, evidence=[])
        self.assertEqual(raw, json.dumps(first_response))
        self.assertEqual(len(agent.last_raw_attempts), 2)
        self.assertTrue(decision.repair_attempted)
        self.assertTrue(decision.repair_succeeded)
        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.llm_requested_hypothesis, 'sensor_bias')
        self.assertEqual(decision.executed_hypothesis, 'sensor_bias')
        self.assertEqual(decision.estimated_test_cost, 0)
        self.assertIn('ratios above 3', stub.calls[0][0])
        self.assertIn('evidence_semantics', stub.calls[0][1])
        self.assertNotIn('deterministic_test_costs', stub.calls[0][1])

    def test_length_limited_llm_output_gets_larger_regeneration_budget(self) -> None:
        repaired_response = {'ranking': [{'hypothesis': 'sensor_bias', 'score': 0.9}], 'requested_test': {'hypothesis': 'sensor_bias', 'reason': 'Test the localized residual'}, 'decision': 'continue_testing', 'diagnosis': None, 'confidence': 0.9, 'rationale': ['Localized residual requires deterministic testing']}

        class LengthLimitedStubClient:

            def __init__(self) -> None:
                self.responses = ['{"ranking": [', json.dumps(repaired_response)]
                self.calls = []
                self.last_response_metadata = {}

            def chat(self, system_prompt, user_prompt, **kwargs):
                self.calls.append(kwargs)
                response = self.responses.pop(0)
                if len(self.calls) == 1:
                    self.last_response_metadata = {'done_reason': 'length', 'eval_count': 1000}
                else:
                    self.last_response_metadata = {'done_reason': 'stop', 'eval_count': 120}
                return response
        agent = HydraulicDiagnosticAgent(mode='ollama', max_rounds=4)
        stub = LengthLimitedStubClient()
        agent.client = stub
        decision, _ = agent.decide(features={'largest_residual_sensors': ['100']}, topology={'valid_junctions': ['100'], 'valid_pipes': [], 'deterministic_test_costs': {'sensor_bias': {'estimated_hydraulic_simulations': 0}}}, evidence=[])
        self.assertEqual(stub.calls[0]['num_predict'], 1000)
        self.assertEqual(stub.calls[1]['num_predict'], 2000)
        self.assertEqual(agent.last_attempt_metadata[0]['done_reason'], 'length')
        self.assertEqual(agent.last_attempt_metadata[1]['done_reason'], 'stop')
        self.assertTrue(decision.repair_attempted)
        self.assertTrue(decision.repair_succeeded)
        self.assertFalse(decision.used_fallback)

    def test_full_network_scope_cannot_be_pruned_by_requested_nodes(self) -> None:
        wn = load_hydraulic_network(INP_PATH)
        candidates = _candidate_nodes(wn, ['150'], scope='full_network')
        self.assertEqual(candidates, list(wn.junction_name_list))
        self.assertIn('90', candidates)

    def test_onset_requires_consecutive_threshold_exceedance(self) -> None:
        index = np.arange(24)
        nominal = pd.DataFrame({'pressure_A': np.full(24, 50.0), 'pressure_B': np.full(24, 50.0), 'flow_Q': np.full(24, 1.0)}, index=index)
        observed = nominal.copy()
        observed.loc[5, 'pressure_A'] -= 0.2
        observed.loc[8:12, ['pressure_A', 'pressure_B']] -= 0.2
        config = ObservationConfig(pressure_sensors=['A', 'B'], flow_link='Q', pressure_noise_std_m=0.03, flow_noise_fraction=0.01)
        _, features = compute_residual_features(observed, nominal, config)
        self.assertEqual(features['estimated_onset_hour'], 8)

    def test_nullable_llm_target_lists_do_not_trigger_fallback(self) -> None:
        agent = HydraulicDiagnosticAgent(mode='rule', max_rounds=4)
        parsed = {'ranking': [{'hypothesis': 'leak', 'score': 0.8, 'supporting_evidence': None, 'contradicting_evidence': None}], 'requested_test': {'hypothesis': 'leak', 'target_nodes': None, 'target_links': None, 'reason': 'Test leak over the deterministic search domain'}, 'decision': 'continue_testing', 'diagnosis': None, 'confidence': 0.8, 'rationale': None}
        decision = agent._validate_decision(parsed, features={'largest_residual_sensors': []}, topology={'valid_junctions': ['90'], 'valid_pipes': ['24']}, evidence=[])
        self.assertFalse(decision.used_fallback)
        self.assertIsNotNone(decision.requested_test)
        self.assertEqual(decision.requested_test.target_nodes, [])
        self.assertEqual(decision.requested_test.target_links, [])

    def test_seed_101_exact_leak_is_accepted_at_the_noise_floor(self) -> None:
        config = ObservationConfig(pressure_sensors=list(DEFAULT_PRESSURE_SENSORS), pressure_noise_std_m=0.03, flow_noise_fraction=0.01, seed=101)
        nominal = nominal_prediction(INP_PATH, config)
        observed, _, _ = generate_pseudo_observations(INP_PATH, FaultSpec('leak', '90', 8, 24, 0.001), config)
        evidence, _ = test_hypothesis_family('leak', INP_PATH, observed, nominal, config, onset_hour=8)
        self.assertEqual(evidence.best_parameters['node'], '90')
        self.assertAlmostEqual(evidence.best_parameters['area_m2'], 0.001)
        self.assertTrue(evidence.noise_consistent)
        self.assertEqual(evidence.result, 'strong_support')
if __name__ == '__main__':
    unittest.main()
