from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentTask:
    """Structured form of a natural-language EPANET/WNTR request."""

    task_type: str
    target_date: str
    baseline_dates: List[str] = field(default_factory=list)
    quality_mode: str = "CHEMICAL"
    source_node: str = "10"
    source_strength: float = 2.0
    source_type: str = "CONCEN"
    duration_hours: int = 24
    timestep_hours: int = 1
    end_node_count: int = 5
    notes: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    demand_total_m3: float
    min_pressure_m: float
    min_quality: Optional[float]
    mean_end_node_quality: Optional[float]
    end_nodes: List[str]


@dataclass
class AgentAnswer:
    task: AgentTask
    holiday: ScenarioResult
    baseline: ScenarioResult
    conclusion: str


@dataclass
class HypothesisScore:
    """Agent ranking for one allowed hydraulic mismatch hypothesis."""

    hypothesis: str
    score: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)


@dataclass
class DiagnosticTestRequest:
    """A validated request for one deterministic hypothesis-family test."""

    hypothesis: str
    target_nodes: List[str] = field(default_factory=list)
    target_links: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class TestEvidence:
    """Numerical evidence returned by a deterministic WNTR test."""

    hypothesis: str
    best_parameters: Dict[str, object]
    training_score_before: float
    training_score_after: float
    validation_score_before: float
    validation_score_after: float
    training_improvement: float
    validation_improvement: float
    pressure_rmse_m: float
    flow_nrmse: float
    simulations: int
    result: str
    pressure_noise_ratio: float = 0.0
    flow_noise_ratio: float = 0.0
    noise_consistent: bool = False
    evidence_basis: List[str] = field(default_factory=list)


@dataclass
class DiagnosticDecision:
    """One evidence-grounded decision made by the diagnostic agent."""

    ranking: List[HypothesisScore]
    requested_test: Optional[DiagnosticTestRequest]
    decision: str
    diagnosis: Optional[str] = None
    confidence: float = 0.0
    rationale: List[str] = field(default_factory=list)
    used_fallback: bool = False
    llm_top_hypothesis: Optional[str] = None
    llm_requested_hypothesis: Optional[str] = None
    executed_hypothesis: Optional[str] = None
    estimated_test_cost: Optional[int] = None
    fallback_reason: Optional[str] = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
