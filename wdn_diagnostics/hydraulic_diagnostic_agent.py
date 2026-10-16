from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple

from wdn_diagnostics.agent_schema import (
    DiagnosticDecision,
    DiagnosticTestRequest,
    HypothesisScore,
    TestEvidence,
)
from wdn_diagnostics.hydraulic_experiment import ALLOWED_HYPOTHESES
from wdn_diagnostics.ollama_client import OllamaClient, extract_json_object


def _estimated_test_cost(
    topology: Dict[str, object],
    hypothesis: str | None,
) -> int | None:
    if hypothesis is None:
        return None
    costs = topology.get("deterministic_test_costs", {})
    if not isinstance(costs, dict):
        return None
    item = costs.get(hypothesis)
    if not isinstance(item, dict):
        return None
    try:
        return int(item["estimated_hydraulic_simulations"])
    except (KeyError, TypeError, ValueError):
        return None


def _ranking_from_scores(
    scores: Dict[str, float],
    supporting: Dict[str, List[str]] | None = None,
) -> List[HypothesisScore]:
    supporting = supporting or {}
    total = sum(max(score, 0.0) for score in scores.values()) or 1.0
    return [
        HypothesisScore(
            hypothesis=hypothesis,
            score=max(float(score), 0.0) / total,
            supporting_evidence=supporting.get(hypothesis, []),
        )
        for hypothesis, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def rule_based_decision(
    features: Dict[str, object],
    evidence: Sequence[TestEvidence],
    max_rounds: int,
    test_order: Sequence[str] = ALLOWED_HYPOTHESES,
) -> DiagnosticDecision:
    """Transparent evidence-first fallback and non-LLM sequential baseline.

    The core policy intentionally contains no network-specific mapping such as
    "negative residuals imply a leak." Test order is supplied by the network
    profile, while deterministic held-out evidence controls stopping.
    """
    tested = {item.hypothesis for item in evidence}
    if evidence:
        best = max(evidence, key=lambda item: item.validation_improvement)
        if best.result == "strong_support":
            scores = {
                name: (0.85 if name == best.hypothesis else 0.05)
                for name in ALLOWED_HYPOTHESES
            }
            return DiagnosticDecision(
                ranking=_ranking_from_scores(scores),
                requested_test=None,
                decision="diagnose",
                diagnosis=best.hypothesis,
                confidence=min(0.99, max(0.0, best.validation_improvement)),
                rationale=[
                    f"{best.hypothesis} produced strong held-out support",
                    f"validation improvement={best.validation_improvement:.3f}",
                ],
                used_fallback=True,
            )

    valid_order = [item for item in test_order if item in ALLOWED_HYPOTHESES]
    valid_order.extend(item for item in ALLOWED_HYPOTHESES if item not in valid_order)
    prior_scores = {
        hypothesis: float(len(valid_order) - index)
        for index, hypothesis in enumerate(valid_order)
    }
    scores = dict(prior_scores)
    support: Dict[str, List[str]] = {name: [] for name in ALLOWED_HYPOTHESES}

    for item in evidence:
        scores[item.hypothesis] = max(
            0.01,
            float(len(valid_order)) * max(0.0, item.validation_improvement),
        )
        support[item.hypothesis].append(
            f"deterministic test result={item.result}, validation improvement="
            f"{item.validation_improvement:.3f}, pressure noise ratio="
            f"{item.pressure_noise_ratio:.3f}, flow noise ratio="
            f"{item.flow_noise_ratio:.3f}"
        )
    ranking = _ranking_from_scores(scores, support)
    untested = [item.hypothesis for item in ranking if item.hypothesis not in tested]
    if not untested or len(evidence) >= max_rounds:
        best = max(evidence, key=lambda item: item.validation_improvement) if evidence else None
        if best is not None and best.result == "strong_support":
            return DiagnosticDecision(
                ranking=ranking,
                requested_test=None,
                decision="diagnose",
                diagnosis=best.hypothesis,
                confidence=max(0.0, min(0.8, best.validation_improvement)),
                rationale=["Selected a strongly supported tested hypothesis at the round limit"],
                used_fallback=True,
            )
        return DiagnosticDecision(
            ranking=ranking,
            requested_test=None,
            decision="abstain",
            diagnosis=None,
            confidence=0.0,
            rationale=["No tested hypothesis achieved adequate held-out support"],
            used_fallback=True,
        )

    selected = untested[0]
    targets = list(features.get("largest_residual_sensors", []))
    return DiagnosticDecision(
        ranking=ranking,
        requested_test=DiagnosticTestRequest(
            hypothesis=selected,
            target_nodes=targets,
            reason="Next untested hypothesis in the configured deterministic test order",
        ),
        decision="continue_testing",
        confidence=ranking[0].score,
        rationale=[f"Requesting deterministic evidence for {selected}"],
        used_fallback=True,
    )


class HydraulicDiagnosticAgent:
    """Constrained hypothesis manager; all numerical work remains deterministic."""

    def __init__(
        self,
        mode: str = "ollama",
        model: str = "qwen2.5:7b",
        max_rounds: int = 3,
        deterministic_test_order: Sequence[str] = ALLOWED_HYPOTHESES,
    ) -> None:
        if mode not in {"ollama", "rule"}:
            raise ValueError("mode must be 'ollama' or 'rule'")
        self.mode = mode
        self.model = model
        self.max_rounds = max_rounds
        self.deterministic_test_order = tuple(deterministic_test_order)
        self.client = OllamaClient(model=model) if mode == "ollama" else None
        self.last_raw_attempts: List[str] = []
        self.last_attempt_metadata: List[Dict[str, object]] = []

    def decide(
        self,
        features: Dict[str, object],
        topology: Dict[str, object],
        evidence: Sequence[TestEvidence],
    ) -> Tuple[DiagnosticDecision, str | None]:
        self.last_raw_attempts = []
        self.last_attempt_metadata = []
        if self.mode == "rule":
            decision = rule_based_decision(
                features,
                evidence,
                self.max_rounds,
                self.deterministic_test_order,
            )
            decision.used_fallback = False
            return decision, None

        system_prompt = (
            "You are a hydraulic digital-twin diagnostic triage agent. You do not perform "
            "hydraulic calculations and you do not estimate parameter values. Rank only the "
            "allowed hypothesis families using the residual summary and deterministic test "
            "evidence. Request exactly one untested hypothesis-family test at a time. Suggested "
            "nodes or links are advisory only; deterministic tools control parameter localization. A lower "
            "validation score is better. Strong held-out validation evidence is required before "
            "diagnosis. Interpret deterministic evidence literally: pressure and flow noise ratios "
            "at or below 1.5 are noise-consistent, ratios above 3 are incompatible with a good fit, "
            "and negative training or validation improvement contradicts a hypothesis. Graph "
            "distance is localization context only and is not evidence that a leak exists. A "
            "localized persistent residual at one sensor with no coherent physical flow response "
            "is compatible with sensor bias; coherent changes across multiple hydraulically related "
            "signals are more compatible with a physical cause. Select the most plausible untested "
            "cause family from the available physical and statistical evidence; do not optimize the "
            "choice for computational cost. If evidence is insufficient after the available rounds, "
            "abstain. Return only one concise JSON object and never include markdown or hidden-cause "
            "speculation. Use no more than two short supporting and two short contradicting evidence "
            "statements per hypothesis."
        )
        evidence_semantics = {
            "strong_noise_consistency": "pressure and flow noise ratios <= 1.5",
            "incompatible_fit": "either noise ratio > 3",
            "improvement_direction": (
                "positive improvement supports fit; negative improvement contradicts fit"
            ),
            "topology_warning": (
                "short graph distance localizes relationships but does not establish a leak"
            ),
            "planning_objective": (
                "select the most plausible untested family using the residual, topology, "
                "and deterministic validation evidence"
            ),
        }
        llm_topology = {
            key: value
            for key, value in topology.items()
            if key
            not in {
                "deterministic_test_costs",
                "valid_junctions",
                "valid_pipes",
            }
        }
        payload = {
            "residual_features": features,
            "topology": llm_topology,
            "evidence_semantics": evidence_semantics,
            "deterministic_test_evidence": [asdict(item) for item in evidence],
            "allowed_hypotheses": list(ALLOWED_HYPOTHESES),
            "tested_hypotheses": [item.hypothesis for item in evidence],
            "rounds_remaining": self.max_rounds - len(evidence),
            "required_schema": {
                "ranking": [
                    {
                        "hypothesis": "one allowed hypothesis",
                        "score": "number from 0 to 1",
                    }
                ],
                "requested_test": {
                    "hypothesis": "one untested allowed hypothesis or null",
                    "reason": "short reason",
                },
                "decision": "continue_testing, diagnose, or abstain",
                "diagnosis": "allowed tested hypothesis or null",
                "confidence": "number from 0 to 1",
                "rationale": ["at most four short evidence-grounded statements"],
            },
        }
        user_prompt = "/no_think\n" + json.dumps(payload, indent=2)
        try:
            raw = self.client.chat(
                system_prompt,
                user_prompt,
                temperature=0.0,
                seed=10,
                num_predict=1000,
                num_ctx=8192,
            )
            self.last_raw_attempts.append(raw)
            self.last_attempt_metadata.append(
                dict(getattr(self.client, "last_response_metadata", {}) or {})
            )
        except Exception as exc:
            fallback = rule_based_decision(
                features,
                evidence,
                self.max_rounds,
                self.deterministic_test_order,
            )
            fallback.rationale.append(f"Local LLM call failed: {exc}")
            fallback.used_fallback = True
            fallback.fallback_reason = f"initial_llm_call_failed: {exc}"
            if fallback.requested_test is not None:
                fallback.executed_hypothesis = fallback.requested_test.hypothesis
                fallback.estimated_test_cost = _estimated_test_cost(
                    topology,
                    fallback.executed_hypothesis,
                )
            return fallback, None

        initial_metadata = (
            self.last_attempt_metadata[0] if self.last_attempt_metadata else {}
        )
        initial_was_truncated = initial_metadata.get("done_reason") == "length"
        initial_issue = None
        if initial_was_truncated:
            initial_issue = (
                "initial_output_truncated: Ollama reported done_reason=length "
                f"after eval_count={initial_metadata.get('eval_count')}"
            )
        else:
            try:
                parsed = extract_json_object(raw)
                decision = self._validate_decision(parsed, features, topology, evidence)
                if not decision.used_fallback:
                    return decision, raw
                initial_issue = (
                    decision.fallback_reason
                    or "initial_output_did_not_supply_a_valid_action"
                )
            except Exception as exc:
                initial_issue = f"initial_output_rejected: {exc}"

        repair_payload = {
            "instruction": (
                "Repair the previous response. Return one complete JSON object matching "
                "the required schema. For continue_testing, requested_test.hypothesis must "
                "be exactly one untested allowed hypothesis. Be concise, use at most two "
                "short rationale statements, and close every JSON array, object, and string. "
                "Do not copy or continue the truncated response; regenerate a shorter object."
            ),
            "validation_issue": initial_issue,
            "previous_output": raw,
            "allowed_hypotheses": list(ALLOWED_HYPOTHESES),
            "tested_hypotheses": [item.hypothesis for item in evidence],
            "required_schema": payload["required_schema"],
        }
        try:
            repaired_raw = self.client.chat(
                system_prompt,
                "/no_think\n" + json.dumps(repair_payload, indent=2),
                temperature=0.0,
                seed=11,
                num_predict=2000 if initial_was_truncated else 1000,
                num_ctx=8192,
            )
            self.last_raw_attempts.append(repaired_raw)
            self.last_attempt_metadata.append(
                dict(getattr(self.client, "last_response_metadata", {}) or {})
            )
            repair_metadata = self.last_attempt_metadata[-1]
            if repair_metadata.get("done_reason") == "length":
                raise ValueError(
                    "repair output was truncated at the generation limit "
                    f"after eval_count={repair_metadata.get('eval_count')}"
                )
            repaired = self._validate_decision(
                extract_json_object(repaired_raw),
                features,
                topology,
                evidence,
            )
            repaired.repair_attempted = True
            repaired.repair_succeeded = not repaired.used_fallback
            if repaired.used_fallback:
                repaired.fallback_reason = (
                    repaired.fallback_reason
                    or "repair_did_not_supply_a_valid_action"
                )
            return repaired, raw
        except Exception as repair_exc:
            fallback = rule_based_decision(
                features,
                evidence,
                self.max_rounds,
                self.deterministic_test_order,
            )
            fallback.rationale.append(
                f"Local LLM output repair failed after {initial_issue}: {repair_exc}"
            )
            fallback.used_fallback = True
            fallback.fallback_reason = (
                f"{initial_issue}; repair_failed: {repair_exc}"
            )
            fallback.repair_attempted = True
            fallback.repair_succeeded = False
            if fallback.requested_test is not None:
                fallback.executed_hypothesis = fallback.requested_test.hypothesis
                fallback.estimated_test_cost = _estimated_test_cost(
                    topology,
                    fallback.executed_hypothesis,
                )
            return fallback, raw

    def _validate_decision(
        self,
        parsed: Dict[str, object],
        features: Dict[str, object],
        topology: Dict[str, object],
        evidence: Sequence[TestEvidence],
    ) -> DiagnosticDecision:
        used_fallback = False
        fallback_reason = None
        tested = {item.hypothesis for item in evidence}
        ranking_items = []
        seen = set()
        for item in parsed.get("ranking", []) or []:
            hypothesis = str(item.get("hypothesis", ""))
            if hypothesis not in ALLOWED_HYPOTHESES or hypothesis in seen:
                continue
            seen.add(hypothesis)
            ranking_items.append(
                HypothesisScore(
                    hypothesis=hypothesis,
                    score=float(np_clip(item.get("score", 0.0))),
                    supporting_evidence=[
                        str(x) for x in (item.get("supporting_evidence", []) or [])
                    ][:6],
                    contradicting_evidence=[
                        str(x) for x in (item.get("contradicting_evidence", []) or [])
                    ][:6],
                )
            )
        if not ranking_items:
            raise ValueError("ranking did not contain an allowed hypothesis")
        ranking_items.sort(key=lambda item: item.score, reverse=True)
        llm_top_hypothesis = ranking_items[0].hypothesis

        requested = parsed.get("requested_test")
        request = None
        llm_requested_hypothesis = None
        if isinstance(requested, dict):
            raw_hypothesis = requested.get("hypothesis")
            hypothesis = str(raw_hypothesis) if raw_hypothesis is not None else ""
            if hypothesis:
                llm_requested_hypothesis = hypothesis
            if hypothesis in ALLOWED_HYPOTHESES and hypothesis not in tested:
                valid_nodes = set(topology.get("valid_junctions", []))
                valid_links = set(topology.get("valid_pipes", []))
                request = DiagnosticTestRequest(
                    hypothesis=hypothesis,
                    target_nodes=[
                        str(x)
                        for x in (requested.get("target_nodes", []) or [])
                        if str(x) in valid_nodes
                    ][:5],
                    target_links=[
                        str(x)
                        for x in (requested.get("target_links", []) or [])
                        if str(x) in valid_links
                    ][:8],
                    reason=str(requested.get("reason", ""))[:500],
                )

        decision_text = str(parsed.get("decision", "continue_testing"))
        if decision_text not in {"continue_testing", "diagnose", "abstain"}:
            decision_text = "continue_testing"
        diagnosis = parsed.get("diagnosis")
        diagnosis = str(diagnosis) if diagnosis is not None else None
        strong = [item for item in evidence if item.result == "strong_support"]
        if strong:
            strongest = max(strong, key=lambda item: item.validation_improvement)
            # Deterministic evidence gate: once a family explains held-out data strongly,
            # do not allow the language model to spend more tests or drift away from it.
            decision_text = "diagnose"
            diagnosis = strongest.hypothesis
            request = None
        if decision_text == "diagnose":
            if diagnosis not in tested or not any(item.hypothesis == diagnosis for item in strong):
                decision_text = "continue_testing"
                diagnosis = None
        if decision_text == "abstain" and len(evidence) < self.max_rounds:
            decision_text = "continue_testing"
        if decision_text == "continue_testing" and request is None:
            fallback = rule_based_decision(
                features,
                evidence,
                self.max_rounds,
                self.deterministic_test_order,
            )
            request = fallback.requested_test
            used_fallback = True
            fallback_reason = "missing_or_invalid_requested_test"
            if request is None:
                decision_text = fallback.decision
                diagnosis = fallback.diagnosis

        rationale = [str(x) for x in (parsed.get("rationale", []) or [])][:8]
        if used_fallback:
            rationale.append(
                "Rule-based fallback supplied a missing test request or terminal decision"
            )

        return DiagnosticDecision(
            ranking=ranking_items,
            requested_test=request,
            decision=decision_text,
            diagnosis=diagnosis,
            confidence=float(np_clip(parsed.get("confidence", 0.0))),
            rationale=rationale,
            used_fallback=used_fallback,
            llm_top_hypothesis=llm_top_hypothesis,
            llm_requested_hypothesis=llm_requested_hypothesis,
            executed_hypothesis=(
                request.hypothesis if request is not None else None
            ),
            estimated_test_cost=_estimated_test_cost(
                topology,
                request.hypothesis if request is not None else None,
            ),
            fallback_reason=fallback_reason,
        )


def np_clip(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, numeric))
