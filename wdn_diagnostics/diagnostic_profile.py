from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List


ALLOWED_HYPOTHESES = ("leak", "demand_error", "roughness_error", "sensor_bias")


@dataclass(frozen=True)
class FeaturePolicy:
    """Network-independent feature definitions expressed in uncertainty units."""

    residual_z_threshold: float = 3.0
    onset_min_consecutive: int = 2
    direction_min_fraction: float = 0.10
    flow_z_threshold: float = 3.0


@dataclass(frozen=True)
class SearchPolicy:
    """Configurable engineering bounds for deterministic family tests."""

    candidate_node_scope: str = "full_network"
    leak_areas_m2: List[float] = field(
        default_factory=lambda: [0.0005, 0.0010, 0.0015]
    )
    demand_multipliers: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.25, 1.5, 2.0]
    )
    roughness_multipliers: List[float] = field(
        default_factory=lambda: [0.6, 0.8, 1.2, 1.4]
    )
    sensor_bias_bound_mode: str = "absolute"
    sensor_bias_bounds_m: List[float] | None = field(
        default_factory=lambda: [-2.0, 2.0]
    )
    sensor_bias_baseline_fraction: float = 0.05
    sensor_bias_min_noise_multiplier: float = 10.0


@dataclass(frozen=True)
class EvidencePolicy:
    """Noise-aware evidence thresholds shared by all hypothesis families."""

    strong_max_pressure_noise_ratio: float = 1.5
    strong_max_flow_noise_ratio: float = 1.5
    strong_min_training_improvement: float = 0.10
    strong_min_validation_improvement: float = 0.10
    moderate_max_pressure_noise_ratio: float = 3.0
    moderate_max_flow_noise_ratio: float = 3.0
    moderate_min_training_improvement: float = 0.05
    moderate_min_validation_improvement: float = 0.05


@dataclass(frozen=True)
class ScenarioSpec:
    fault_type: str
    target: str
    start_hour: int
    end_hour: int
    magnitude: float
    expected_diagnosis: str | None


@dataclass(frozen=True)
class NetworkProfile:
    """All network- and study-specific inputs kept outside the core engine."""

    name: str
    inp_file: str
    solver: str
    demand_model: str
    pressure_sensors: List[str]
    flow_link: str
    pressure_noise_std_m: float
    flow_noise_fraction: float
    duration_hours: int
    feature_policy: FeaturePolicy
    search_policy: SearchPolicy
    evidence_policy: EvidencePolicy
    deterministic_test_order: List[str]
    scenarios: Dict[str, ScenarioSpec]


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_profile(profile: NetworkProfile) -> NetworkProfile:
    if profile.solver not in {"wntr", "epanet"}:
        raise ValueError("solver must be 'wntr' or 'epanet'")
    if profile.demand_model not in {"DDA", "PDD", "PDA"}:
        raise ValueError("demand_model must be 'DDA' or 'PDD'")
    _require_positive("pressure_noise_std_m", profile.pressure_noise_std_m)
    _require_positive("flow_noise_fraction", profile.flow_noise_fraction)
    _require_positive("duration_hours", profile.duration_hours)
    _require_positive(
        "feature_policy.residual_z_threshold",
        profile.feature_policy.residual_z_threshold,
    )
    if profile.feature_policy.onset_min_consecutive < 1:
        raise ValueError("feature_policy.onset_min_consecutive must be at least 1")
    if not 0 < profile.feature_policy.direction_min_fraction <= 1:
        raise ValueError("feature_policy.direction_min_fraction must be in (0, 1]")
    if profile.search_policy.candidate_node_scope not in {
        "full_network",
        "requested_plus_neighbors",
    }:
        raise ValueError(
            "search_policy.candidate_node_scope must be "
            "'full_network' or 'requested_plus_neighbors'"
        )
    bias_mode = profile.search_policy.sensor_bias_bound_mode
    if bias_mode not in {"absolute", "baseline_fraction"}:
        raise ValueError(
            "search_policy.sensor_bias_bound_mode must be "
            "'absolute' or 'baseline_fraction'"
        )
    if bias_mode == "absolute":
        bounds = profile.search_policy.sensor_bias_bounds_m
        if bounds is None or len(bounds) != 2:
            raise ValueError(
                "search_policy.sensor_bias_bounds_m must contain [lower, upper] "
                "when sensor_bias_bound_mode is 'absolute'"
            )
        lower, upper = bounds
        if lower >= upper:
            raise ValueError("sensor bias lower bound must be below upper bound")
    else:
        _require_positive(
            "search_policy.sensor_bias_baseline_fraction",
            profile.search_policy.sensor_bias_baseline_fraction,
        )
        _require_positive(
            "search_policy.sensor_bias_min_noise_multiplier",
            profile.search_policy.sensor_bias_min_noise_multiplier,
        )
    invalid_order = [
        item for item in profile.deterministic_test_order if item not in ALLOWED_HYPOTHESES
    ]
    if invalid_order:
        raise ValueError(f"Invalid hypotheses in deterministic_test_order: {invalid_order}")
    if set(profile.deterministic_test_order) != set(ALLOWED_HYPOTHESES):
        raise ValueError(
            "deterministic_test_order must contain each allowed hypothesis exactly once"
        )
    return profile


def load_network_profile(path: str | Path) -> NetworkProfile:
    profile_path = Path(path)
    raw = json.loads(profile_path.read_text(encoding="utf-8"))
    scenarios = {
        name: ScenarioSpec(**scenario)
        for name, scenario in raw.get("scenarios", {}).items()
    }
    profile = NetworkProfile(
        name=str(raw["name"]),
        inp_file=str(raw["inp_file"]),
        solver=str(raw.get("solver", "wntr")).lower(),
        demand_model=str(raw.get("demand_model", "PDD")).upper(),
        pressure_sensors=[str(x) for x in raw["pressure_sensors"]],
        flow_link=str(raw["flow_link"]),
        pressure_noise_std_m=float(raw["pressure_noise_std_m"]),
        flow_noise_fraction=float(raw["flow_noise_fraction"]),
        duration_hours=int(raw.get("duration_hours", 24)),
        feature_policy=FeaturePolicy(**raw.get("feature_policy", {})),
        search_policy=SearchPolicy(**raw.get("search_policy", {})),
        evidence_policy=EvidencePolicy(**raw.get("evidence_policy", {})),
        deterministic_test_order=[
            str(x)
            for x in raw.get("deterministic_test_order", ALLOWED_HYPOTHESES)
        ],
        scenarios=scenarios,
    )
    return validate_profile(profile)
