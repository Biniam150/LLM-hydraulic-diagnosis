from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Sequence, Tuple
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import wntr

from wdn_diagnostics.agent_schema import TestEvidence
from wdn_diagnostics.diagnostic_profile import (
    ALLOWED_HYPOTHESES,
    EvidencePolicy,
    FeaturePolicy,
    SearchPolicy,
)


DEFAULT_PRESSURE_SENSORS = ["20", "50", "70", "100", "120", "150", "170"]
DEFAULT_FLOW_LINK = "82"
_IN_MEMORY_CANDIDATE_CACHE: Dict[str, pd.DataFrame] = {}
ALLOWED_FAULTS = {
    "leak",
    "demand_error",
    "roughness_error",
    "sensor_bias",
    "source_head",
    "no_fault",
}


@dataclass
class FaultSpec:
    """Reproducible hidden fault used to generate pseudo-observations."""

    fault_type: str = "leak"
    target: str = "90"
    start_hour: int = 8
    end_hour: int = 24
    magnitude: float = 0.001


@dataclass
class ObservationConfig:
    pressure_sensors: List[str]
    flow_link: str = DEFAULT_FLOW_LINK
    pressure_noise_std_m: float = 0.03
    flow_noise_fraction: float = 0.01
    duration_hours: int = 24
    seed: int = 42
    solver: str = "wntr"
    demand_model: str = "PDD"
    feature_policy: FeaturePolicy = field(default_factory=FeaturePolicy)
    search_policy: SearchPolicy = field(default_factory=SearchPolicy)
    evidence_policy: EvidencePolicy = field(default_factory=EvidencePolicy)
    candidate_cache_dir: Path | None = None


def configure_network(
    wn: wntr.network.WaterNetworkModel,
    duration_hours: int = 24,
    demand_model: str = "PDD",
) -> wntr.network.WaterNetworkModel:
    """Apply one consistent hydraulic configuration to every experiment run."""
    wn.options.time.duration = duration_hours * 3600
    wn.options.time.hydraulic_timestep = 3600
    wn.options.time.report_timestep = 3600
    wn.options.time.pattern_timestep = 3600
    wn.options.hydraulic.demand_model = demand_model
    return wn


def load_hydraulic_network(
    inp_path: str | Path,
    duration_hours: int = 24,
    demand_model: str = "PDD",
) -> wntr.network.WaterNetworkModel:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message='Not all curves were used.*')
        network = wntr.network.WaterNetworkModel(str(inp_path))
    return configure_network(
        network,
        duration_hours=duration_hours,
        demand_model=demand_model,
    )


def validate_observation_config(
    wn: wntr.network.WaterNetworkModel,
    config: ObservationConfig,
) -> None:
    missing_nodes = [n for n in config.pressure_sensors if n not in wn.junction_name_list]
    if missing_nodes:
        raise ValueError(f"Pressure sensors are not junctions in the network: {missing_nodes}")
    if config.flow_link not in wn.link_name_list:
        raise ValueError(f"Flow sensor link {config.flow_link!r} is not in the network")


def _multiply_node_demand(
    wn: wntr.network.WaterNetworkModel,
    node_name: str,
    multiplier: float,
) -> None:
    junction = wn.get_node(node_name)
    if not junction.demand_timeseries_list:
        raise ValueError(f"Junction {node_name} has no demand time series")
    for demand in junction.demand_timeseries_list:
        demand.base_value *= multiplier


def apply_physical_fault(
    wn: wntr.network.WaterNetworkModel,
    fault: FaultSpec,
    solver: str = "wntr",
) -> wntr.network.WaterNetworkModel:
    """Apply a physical/model fault. Sensor bias is applied to observations later."""
    if fault.fault_type not in ALLOWED_FAULTS:
        raise ValueError(f"Unknown fault type: {fault.fault_type}")

    if fault.fault_type == "no_fault":
        return wn
    if fault.fault_type == "leak":
        if fault.target not in wn.junction_name_list:
            raise ValueError(f"Leak target {fault.target!r} is not a junction")
        junction = wn.get_node(fault.target)
        if solver == "wntr":
            junction.add_leak(
                wn,
                area=float(fault.magnitude),
                discharge_coeff=0.75,
                start_time=int(fault.start_hour * 3600),
                end_time=int(fault.end_hour * 3600),
            )
        elif solver == "epanet":
            if fault.start_hour != 0:
                raise ValueError(
                    "EPANET emitter leaks currently require start_hour=0; "
                    "use the WNTR solver for time-controlled leaks"
                )
            junction.emitter_coefficient = (
                0.75 * float(fault.magnitude) * np.sqrt(2.0 * 9.81)
            )
        else:
            raise ValueError(f"Unsupported hydraulic solver: {solver}")
    elif fault.fault_type == "demand_error":
        if fault.target not in wn.junction_name_list:
            raise ValueError(f"Demand target {fault.target!r} is not a junction")
        _multiply_node_demand(wn, fault.target, float(fault.magnitude))
    elif fault.fault_type == "roughness_error":
        if fault.target not in wn.pipe_name_list:
            raise ValueError(f"Roughness target {fault.target!r} is not a pipe")
        pipe = wn.get_link(fault.target)
        pipe.roughness *= float(fault.magnitude)
    elif fault.fault_type == "source_head":
        if fault.target not in wn.reservoir_name_list:
            raise ValueError(f"Source-head target {fault.target!r} is not a reservoir")
        reservoir = wn.get_node(fault.target)
        reservoir.base_head += float(fault.magnitude)
    return wn


def run_hydraulics(
    wn: wntr.network.WaterNetworkModel,
    solver: str = "wntr",
):
    """WNTRSimulator is used because it represents pressure-dependent leaks directly."""
    # Some deliberately implausible fault candidates make the nonlinear solver diverge.
    # Bound their cost; incomplete results are rejected by the hypothesis tester.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Simulation did not converge.*")
        if solver == "wntr":
            return wntr.sim.WNTRSimulator(wn).run_sim(
                solver_options={"MAXITER": 250, "TIME_LIMIT": 3}
            )
        if solver == "epanet":
            with tempfile.TemporaryDirectory(prefix="wdn_epanet_") as temp_dir:
                prefix = str(Path(temp_dir) / "simulation")
                return wntr.sim.EpanetSimulator(wn).run_sim(
                    file_prefix=prefix,
                    version=2.2,
                    convergence_error=False,
                )
        raise ValueError(f"Unsupported hydraulic solver: {solver}")


def results_to_sensor_frame(
    results,
    config: ObservationConfig,
) -> pd.DataFrame:
    pressure = results.node["pressure"].iloc[: config.duration_hours]
    flow = results.link["flowrate"].iloc[: config.duration_hours]
    data = {
        f"pressure_{node}": pressure[node].astype(float).to_numpy()
        for node in config.pressure_sensors
    }
    data[f"flow_{config.flow_link}"] = flow[config.flow_link].astype(float).to_numpy()
    frame = pd.DataFrame(data, index=np.arange(len(pressure), dtype=int))
    frame.index.name = "hour"
    return frame


def generate_pseudo_observations(
    inp_path: str | Path,
    fault: FaultSpec,
    config: ObservationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Generate clean and noisy pseudo-observations without exposing the label to the agent."""
    truth_network = load_hydraulic_network(
        inp_path,
        config.duration_hours,
        config.demand_model,
    )
    validate_observation_config(truth_network, config)
    truth_network = apply_physical_fault(truth_network, fault, config.solver)
    clean = results_to_sensor_frame(
        run_hydraulics(truth_network, config.solver),
        config,
    )
    if not _candidate_is_complete(clean, config):
        raise ValueError(
            "Ground-truth simulation did not produce a complete, finite "
            "observation window; reduce the injected fault severity or inspect "
            "hydraulic feasibility."
        )

    if fault.fault_type == "sensor_bias":
        column = f"pressure_{fault.target}"
        if column not in clean.columns:
            raise ValueError(
                f"Sensor-bias target {fault.target!r} must be one of the pressure sensors"
            )
        active = (clean.index >= fault.start_hour) & (clean.index < fault.end_hour)
        clean.loc[active, column] += float(fault.magnitude)

    rng = np.random.default_rng(config.seed)
    noisy = clean.copy()
    pressure_columns = [c for c in noisy.columns if c.startswith("pressure_")]
    noisy[pressure_columns] += rng.normal(
        0.0,
        config.pressure_noise_std_m,
        size=(len(noisy), len(pressure_columns)),
    )
    flow_column = f"flow_{config.flow_link}"
    flow_scale = max(float(clean[flow_column].abs().mean()), 1e-6)
    noisy[flow_column] += rng.normal(
        0.0,
        config.flow_noise_fraction * flow_scale,
        size=len(noisy),
    )

    observation_config = asdict(config)
    if observation_config.get("candidate_cache_dir") is not None:
        observation_config["candidate_cache_dir"] = str(
            observation_config["candidate_cache_dir"]
        )
    manifest = {
        "network": str(Path(inp_path).resolve()),
        "fault": asdict(fault),
        "observation_config": observation_config,
        "ground_truth_is_hidden_from_agent": True,
    }
    return noisy, clean, manifest


def generate_compound_pseudo_observations(
    inp_path: str | Path,
    faults: Sequence[FaultSpec],
    config: ObservationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Generate observations containing multiple simultaneous hidden mechanisms.

    Physical/model faults are applied sequentially to one truth network. Sensor
    faults are then applied to the resulting sensor frame. The diagnostic model
    still receives only the observations, residual features, and topology.
    """
    if len(faults) < 2:
        raise ValueError("Compound observations require at least two faults")
    truth_network = load_hydraulic_network(
        inp_path,
        config.duration_hours,
        config.demand_model,
    )
    validate_observation_config(truth_network, config)
    for fault in faults:
        if fault.fault_type == "sensor_bias":
            continue
        truth_network = apply_physical_fault(truth_network, fault, config.solver)
    clean = results_to_sensor_frame(
        run_hydraulics(truth_network, config.solver),
        config,
    )
    if not _candidate_is_complete(clean, config):
        raise ValueError(
            "Compound ground-truth simulation did not produce a complete, finite "
            "observation window; reduce the injected fault severity or inspect "
            "hydraulic feasibility."
        )
    for fault in faults:
        if fault.fault_type != "sensor_bias":
            continue
        column = f"pressure_{fault.target}"
        if column not in clean.columns:
            raise ValueError(
                f"Sensor-bias target {fault.target!r} must be one of the pressure sensors"
            )
        active = (clean.index >= fault.start_hour) & (clean.index < fault.end_hour)
        clean.loc[active, column] += float(fault.magnitude)

    rng = np.random.default_rng(config.seed)
    noisy = clean.copy()
    pressure_columns = [c for c in noisy.columns if c.startswith("pressure_")]
    noisy[pressure_columns] += rng.normal(
        0.0,
        config.pressure_noise_std_m,
        size=(len(noisy), len(pressure_columns)),
    )
    flow_column = f"flow_{config.flow_link}"
    flow_scale = max(float(clean[flow_column].abs().mean()), 1e-6)
    noisy[flow_column] += rng.normal(
        0.0,
        config.flow_noise_fraction * flow_scale,
        size=len(noisy),
    )

    observation_config = asdict(config)
    if observation_config.get("candidate_cache_dir") is not None:
        observation_config["candidate_cache_dir"] = str(
            observation_config["candidate_cache_dir"]
        )
    manifest = {
        "network": str(Path(inp_path).resolve()),
        "faults": [asdict(fault) for fault in faults],
        "true_families": sorted({fault.fault_type for fault in faults}),
        "observation_config": observation_config,
        "ground_truth_is_hidden_from_agent": True,
        "compound_truth": True,
    }
    return noisy, clean, manifest


def nominal_prediction(
    inp_path: str | Path,
    config: ObservationConfig,
) -> pd.DataFrame:
    nominal = load_hydraulic_network(
        inp_path,
        config.duration_hours,
        config.demand_model,
    )
    validate_observation_config(nominal, config)
    return results_to_sensor_frame(run_hydraulics(nominal, config.solver), config)


def compute_residual_features(
    observed: pd.DataFrame,
    nominal: pd.DataFrame,
    config: ObservationConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    residuals = observed - nominal
    pressure_columns = [c for c in residuals.columns if c.startswith("pressure_")]
    flow_column = f"flow_{config.flow_link}"
    feature_policy = config.feature_policy
    pressure_threshold = (
        feature_policy.residual_z_threshold * config.pressure_noise_std_m
    )

    hourly_affected = (residuals[pressure_columns].abs() >= pressure_threshold).sum(axis=1)
    active_hours = (hourly_affected > 0).to_numpy(dtype=bool)
    onset_hour = 0
    required_run = feature_policy.onset_min_consecutive
    for start in range(0, len(active_hours) - required_run + 1):
        if bool(active_hours[start : start + required_run].all()):
            onset_hour = int(hourly_affected.index[start])
            break
    post = residuals.loc[residuals.index >= onset_hour]

    per_sensor = {}
    for column in pressure_columns:
        node = column.removeprefix("pressure_")
        values = post[column]
        per_sensor[node] = {
            "mean_m": float(values.mean()),
            "rmse_m": float(np.sqrt(np.mean(values.to_numpy() ** 2))),
            "minimum_m": float(values.min()),
            "maximum_m": float(values.max()),
        }

    ranked_nodes = sorted(
        per_sensor,
        key=lambda node: per_sensor[node]["rmse_m"],
        reverse=True,
    )
    affected_nodes = [
        node for node in ranked_nodes if per_sensor[node]["rmse_m"] >= pressure_threshold
    ]
    pressure_values = post[pressure_columns].to_numpy().ravel()
    negative_fraction = float(np.mean(pressure_values < -pressure_threshold))
    positive_fraction = float(np.mean(pressure_values > pressure_threshold))
    if (
        negative_fraction > positive_fraction
        and negative_fraction >= feature_policy.direction_min_fraction
    ):
        direction = "mostly_negative"
    elif (
        positive_fraction > negative_fraction
        and positive_fraction >= feature_policy.direction_min_fraction
    ):
        direction = "mostly_positive"
    else:
        direction = "mixed_or_small"

    flow_scale = max(float(nominal[flow_column].abs().mean()), 1e-6)
    flow_rmse = float(np.sqrt(np.mean(post[flow_column].to_numpy() ** 2)))
    flow_nrmse = flow_rmse / flow_scale
    features = {
        "observation_window_hours": int(len(observed)),
        "estimated_onset_hour": onset_hour,
        "onset_min_consecutive": feature_policy.onset_min_consecutive,
        "residual_z_threshold": feature_policy.residual_z_threshold,
        "pressure_threshold_m": pressure_threshold,
        "pressure_direction": direction,
        "affected_sensor_count": len(affected_nodes),
        "affected_sensors": affected_nodes,
        "largest_residual_sensors": ranked_nodes[:3],
        "pressure_residual_by_sensor": per_sensor,
        "monitored_flow_link": config.flow_link,
        "flow_nrmse": flow_nrmse,
        "flow_noise_ratio": flow_nrmse / config.flow_noise_fraction,
        "flow_changed": bool(
            flow_nrmse
            >= feature_policy.flow_z_threshold * config.flow_noise_fraction
        ),
        # Backward-compatible aliases retained for historical Anytown outputs.
        "pump_flow_nrmse": flow_nrmse,
        "pump_flow_noise_ratio": flow_nrmse / config.flow_noise_fraction,
        "pump_flow_changed": bool(
            flow_nrmse
            >= feature_policy.flow_z_threshold * config.flow_noise_fraction
        ),
        "available_hypotheses": list(ALLOWED_HYPOTHESES) + ["none"],
    }
    return residuals, features


def _combined_score(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
    nominal: pd.DataFrame,
    config: ObservationConfig,
    hours: Iterable[int],
) -> Tuple[float, float, float]:
    selected = list(hours)
    pressure_columns = [c for c in observed.columns if c.startswith("pressure_")]
    flow_column = f"flow_{config.flow_link}"
    p_error = observed.loc[selected, pressure_columns] - predicted.loc[selected, pressure_columns]
    q_error = observed.loc[selected, flow_column] - predicted.loc[selected, flow_column]
    pressure_rmse = float(np.sqrt(np.mean(p_error.to_numpy() ** 2)))
    pressure_scale = max(
        float(nominal.loc[selected, pressure_columns].stack().std()),
        1.0,
    )
    flow_rmse = float(np.sqrt(np.mean(q_error.to_numpy() ** 2)))
    flow_scale = max(float(nominal.loc[selected, flow_column].abs().mean()), 1e-6)
    pressure_nrmse = pressure_rmse / pressure_scale
    flow_nrmse = flow_rmse / flow_scale
    return pressure_nrmse + flow_nrmse, pressure_rmse, flow_nrmse


def _improvement(before: float, after: float) -> float:
    return float((before - after) / max(before, 1e-12))


def _candidate_nodes(
    wn: wntr.network.WaterNetworkModel,
    requested_nodes: Sequence[str] | None,
    scope: str,
) -> List[str]:
    if scope == "full_network":
        return list(wn.junction_name_list)
    if scope != "requested_plus_neighbors":
        raise ValueError(f"Unsupported candidate-node scope: {scope}")
    valid = [n for n in (requested_nodes or []) if n in wn.junction_name_list]
    if not valid:
        return list(wn.junction_name_list)
    graph = wn.to_graph().to_undirected()
    expanded = set(valid)
    for node in valid:
        if node in graph:
            expanded.update(n for n in graph.neighbors(node) if n in wn.junction_name_list)
    return [n for n in wn.junction_name_list if n in expanded]


def _candidate_pipes(
    wn: wntr.network.WaterNetworkModel,
    requested_nodes: Sequence[str] | None,
    requested_links: Sequence[str] | None,
) -> List[str]:
    valid_links = [p for p in (requested_links or []) if p in wn.pipe_name_list]
    valid_nodes = set(n for n in (requested_nodes or []) if n in wn.node_name_list)
    if valid_links:
        return valid_links
    if valid_nodes:
        adjacent = []
        for pipe_name in wn.pipe_name_list:
            pipe = wn.get_link(pipe_name)
            if pipe.start_node_name in valid_nodes or pipe.end_node_name in valid_nodes:
                adjacent.append(pipe_name)
        if adjacent:
            return adjacent
    return list(wn.pipe_name_list)


def _simulate_candidate(
    inp_path: str | Path,
    config: ObservationConfig,
    fault: FaultSpec,
) -> pd.DataFrame:
    cache_path = None
    if config.candidate_cache_dir is not None:
        inp = Path(inp_path).resolve()
        stat = inp.stat()
        cache_payload = {
            "inp_path": str(inp),
            "inp_size": stat.st_size,
            "inp_mtime_ns": stat.st_mtime_ns,
            "duration_hours": config.duration_hours,
            "solver": config.solver,
            "demand_model": config.demand_model,
            "pressure_sensors": list(config.pressure_sensors),
            "flow_link": config.flow_link,
            "fault": asdict(fault),
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in _IN_MEMORY_CANDIDATE_CACHE:
            return _IN_MEMORY_CANDIDATE_CACHE[cache_key]
        cache_path = Path(config.candidate_cache_dir) / f"{cache_key}.pkl"
        if cache_path.exists():
            prediction = pd.read_pickle(cache_path)
            _IN_MEMORY_CANDIDATE_CACHE[cache_key] = prediction
            return prediction
    wn = load_hydraulic_network(
        inp_path,
        config.duration_hours,
        config.demand_model,
    )
    apply_physical_fault(wn, fault, config.solver)
    prediction = results_to_sensor_frame(run_hydraulics(wn, config.solver), config)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_pickle(cache_path)
        _IN_MEMORY_CANDIDATE_CACHE[cache_key] = prediction
    return prediction


def _simulate_compound_candidate(
    inp_path: str | Path,
    config: ObservationConfig,
    faults: Sequence[FaultSpec],
) -> pd.DataFrame:
    """Simulate a deterministic multi-fault candidate with cache provenance."""
    cache_path = None
    cache_key = None
    if config.candidate_cache_dir is not None:
        inp = Path(inp_path).resolve()
        stat = inp.stat()
        cache_payload = {
            "inp_path": str(inp),
            "inp_size": stat.st_size,
            "inp_mtime_ns": stat.st_mtime_ns,
            "duration_hours": config.duration_hours,
            "solver": config.solver,
            "demand_model": config.demand_model,
            "pressure_sensors": list(config.pressure_sensors),
            "flow_link": config.flow_link,
            "compound_faults": [asdict(fault) for fault in faults],
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if cache_key in _IN_MEMORY_CANDIDATE_CACHE:
            return _IN_MEMORY_CANDIDATE_CACHE[cache_key]
        cache_path = Path(config.candidate_cache_dir) / f"{cache_key}.pkl"
        if cache_path.exists():
            prediction = pd.read_pickle(cache_path)
            _IN_MEMORY_CANDIDATE_CACHE[cache_key] = prediction
            return prediction

    wn = load_hydraulic_network(inp_path, config.duration_hours, config.demand_model)
    for fault in faults:
        apply_physical_fault(wn, fault, config.solver)
    prediction = results_to_sensor_frame(run_hydraulics(wn, config.solver), config)
    if cache_path is not None and cache_key is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_pickle(cache_path)
        _IN_MEMORY_CANDIDATE_CACHE[cache_key] = prediction
    return prediction


def _candidate_is_complete(prediction: pd.DataFrame, config: ObservationConfig) -> bool:
    required_hours = set(range(config.duration_hours))
    return (
        required_hours.issubset(set(int(x) for x in prediction.index))
        and len(prediction) >= config.duration_hours
        and bool(np.isfinite(prediction.to_numpy()).all())
    )


def _candidate_leak_start_hour(onset_hour: int, solver: str) -> int:
    """Return a leak onset supported by the selected hydraulic adapter.

    The current EPANET adapter represents leaks with a persistent emitter, so a
    residual-derived onset later than hour zero cannot be applied directly. WNTR
    supports time-controlled leaks and retains the estimated onset.
    """
    if solver.lower() == "epanet":
        return 0
    return int(onset_hour)


def _sensor_bias_bounds(
    nominal: pd.DataFrame,
    config: ObservationConfig,
) -> Tuple[float, float, float | None]:
    """Resolve absolute or baseline-pressure-scaled sensor-bias bounds."""
    policy = config.search_policy
    if policy.sensor_bias_bound_mode == "absolute":
        if policy.sensor_bias_bounds_m is None:
            raise ValueError("Absolute sensor-bias bounds are missing")
        lower, upper = policy.sensor_bias_bounds_m
        return float(lower), float(upper), None

    pressure_columns = [
        column for column in nominal.columns if column.startswith("pressure_")
    ]
    baseline_scale = float(
        np.median(np.abs(nominal[pressure_columns].to_numpy(dtype=float)))
    )
    half_width = max(
        policy.sensor_bias_baseline_fraction * baseline_scale,
        policy.sensor_bias_min_noise_multiplier * config.pressure_noise_std_m,
    )
    return -float(half_width), float(half_width), baseline_scale


def test_hypothesis_family(
    hypothesis: str,
    inp_path: str | Path,
    observed: pd.DataFrame,
    nominal: pd.DataFrame,
    config: ObservationConfig,
    onset_hour: int,
    requested_nodes: Sequence[str] | None = None,
    requested_links: Sequence[str] | None = None,
) -> Tuple[TestEvidence, pd.DataFrame]:
    """Fit one family on even hours and evaluate on interleaved held-out hours.

    Interleaving is important for diurnal hydraulic systems: a contiguous final-time
    split can contain only low-sensitivity operating states and fail to test the fault.
    """
    if hypothesis not in ALLOWED_HYPOTHESES:
        raise ValueError(f"Unsupported hypothesis: {hypothesis}")

    train_hours = [hour for hour in range(config.duration_hours) if hour % 2 == 0]
    validation_hours = [hour for hour in range(config.duration_hours) if hour % 2 == 1]

    baseline_train, _, _ = _combined_score(observed, nominal, nominal, config, train_hours)
    baseline_valid, _, _ = _combined_score(observed, nominal, nominal, config, validation_hours)
    wn_for_names = load_hydraulic_network(
        inp_path,
        config.duration_hours,
        config.demand_model,
    )
    candidate_predictions: List[Tuple[float, Dict[str, object], pd.DataFrame]] = []
    simulations = 0

    if hypothesis == "leak":
        nodes = _candidate_nodes(
            wn_for_names,
            requested_nodes,
            config.search_policy.candidate_node_scope,
        )
        leak_start_hour = _candidate_leak_start_hour(onset_hour, config.solver)
        for node in nodes:
            for area in config.search_policy.leak_areas_m2:
                prediction = _simulate_candidate(
                    inp_path,
                    config,
                    FaultSpec(
                        "leak",
                        node,
                        leak_start_hour,
                        config.duration_hours,
                        area,
                    ),
                )
                simulations += 1
                if not _candidate_is_complete(prediction, config):
                    continue
                score, _, _ = _combined_score(
                    observed, prediction, nominal, config, train_hours
                )
                candidate_predictions.append(
                    (
                        score,
                        {
                            "node": node,
                            "area_m2": area,
                            "start_hour": leak_start_hour,
                        },
                        prediction,
                    )
                )
    elif hypothesis == "demand_error":
        nodes = _candidate_nodes(
            wn_for_names,
            requested_nodes,
            config.search_policy.candidate_node_scope,
        )
        for node in nodes:
            for multiplier in config.search_policy.demand_multipliers:
                prediction = _simulate_candidate(
                    inp_path,
                    config,
                    FaultSpec(
                        "demand_error",
                        node,
                        0,
                        config.duration_hours,
                        multiplier,
                    ),
                )
                simulations += 1
                if not _candidate_is_complete(prediction, config):
                    continue
                score, _, _ = _combined_score(
                    observed, prediction, nominal, config, train_hours
                )
                candidate_predictions.append(
                    (score, {"node": node, "multiplier": multiplier}, prediction)
                )
    elif hypothesis == "roughness_error":
        # The LLM selects the cause family, not the numerical search domain. A prior
        # version restricted this search to pipes adjacent to an LLM-selected node and
        # missed the true pipe. Anytown is small enough for deterministic full-network
        # coverage; larger networks should replace this with a hydraulic-sensitivity
        # shortlist that is generated by code rather than by the language model.
        pipes = list(wn_for_names.pipe_name_list)
        for pipe in pipes:
            for multiplier in config.search_policy.roughness_multipliers:
                prediction = _simulate_candidate(
                    inp_path,
                    config,
                    FaultSpec(
                        "roughness_error",
                        pipe,
                        0,
                        config.duration_hours,
                        multiplier,
                    ),
                )
                simulations += 1
                if not _candidate_is_complete(prediction, config):
                    continue
                score, _, _ = _combined_score(
                    observed, prediction, nominal, config, train_hours
                )
                candidate_predictions.append(
                    (
                        score,
                        {
                            "pipe": pipe,
                            "multiplier": multiplier,
                            "search_scope": "full_network",
                            "candidate_pipe_count": len(pipes),
                        },
                        prediction,
                    )
                )
    else:
        pressure_columns = [c for c in observed.columns if c.startswith("pressure_")]
        lower_bias, upper_bias, baseline_pressure_scale = _sensor_bias_bounds(
            nominal,
            config,
        )
        for column in pressure_columns:
            prediction = nominal.copy()
            active_train = [h for h in train_hours if h >= onset_hour]
            if not active_train:
                active_train = train_hours
            bias = float((observed.loc[active_train, column] - nominal.loc[active_train, column]).mean())
            bias = float(np.clip(bias, lower_bias, upper_bias))
            prediction.loc[prediction.index >= onset_hour, column] += bias
            score, _, _ = _combined_score(observed, prediction, nominal, config, train_hours)
            candidate_predictions.append(
                (
                    score,
                    {
                        "sensor": column.removeprefix("pressure_"),
                        "bias_m": bias,
                        "bias_bound_mode": (
                            config.search_policy.sensor_bias_bound_mode
                        ),
                        "bias_bounds_m": [lower_bias, upper_bias],
                        "baseline_pressure_scale_m": baseline_pressure_scale,
                    },
                    prediction,
                )
            )

    if not candidate_predictions:
        raise RuntimeError(f"No candidates were generated for {hypothesis}")
    best_train_score, best_parameters, best_prediction = min(
        candidate_predictions,
        key=lambda item: item[0],
    )
    validation_score, pressure_rmse, flow_nrmse = _combined_score(
        observed,
        best_prediction,
        nominal,
        config,
        validation_hours,
    )
    train_improvement = _improvement(baseline_train, best_train_score)
    valid_improvement = _improvement(baseline_valid, validation_score)
    pressure_noise_ratio = pressure_rmse / config.pressure_noise_std_m
    flow_noise_ratio = flow_nrmse / config.flow_noise_fraction
    policy = config.evidence_policy
    noise_consistent = bool(
        pressure_noise_ratio <= policy.strong_max_pressure_noise_ratio
        and flow_noise_ratio <= policy.strong_max_flow_noise_ratio
    )
    if (
        noise_consistent
        and valid_improvement >= policy.strong_min_validation_improvement
        and train_improvement >= policy.strong_min_training_improvement
    ):
        result = "strong_support"
    elif (
        pressure_noise_ratio <= policy.moderate_max_pressure_noise_ratio
        and flow_noise_ratio <= policy.moderate_max_flow_noise_ratio
        and valid_improvement >= policy.moderate_min_validation_improvement
        and train_improvement >= policy.moderate_min_training_improvement
    ):
        result = "moderate_support"
    else:
        result = "weak_or_no_support"

    evidence = TestEvidence(
        hypothesis=hypothesis,
        best_parameters=best_parameters,
        training_score_before=baseline_train,
        training_score_after=best_train_score,
        validation_score_before=baseline_valid,
        validation_score_after=validation_score,
        training_improvement=train_improvement,
        validation_improvement=valid_improvement,
        pressure_rmse_m=pressure_rmse,
        flow_nrmse=flow_nrmse,
        simulations=simulations,
        result=result,
        pressure_noise_ratio=pressure_noise_ratio,
        flow_noise_ratio=flow_noise_ratio,
        noise_consistent=noise_consistent,
        evidence_basis=[
            (
                "post-fit pressure RMSE / configured pressure noise="
                f"{pressure_noise_ratio:.3f}"
            ),
            (
                "post-fit flow NRMSE / configured flow noise fraction="
                f"{flow_noise_ratio:.3f}"
            ),
            f"training improvement={train_improvement:.3f}",
            f"validation improvement={valid_improvement:.3f}",
        ],
    )
    return evidence, best_prediction


def test_leak_demand_compound(
    inp_path: str | Path,
    observed: pd.DataFrame,
    nominal: pd.DataFrame,
    config: ObservationConfig,
    leak_evidence: TestEvidence,
    demand_evidence: TestEvidence,
    onset_step_hours: int = 4,
    minimum_gain_over_single: float = 0.10,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Verify a leak+demand explanation against both fitted single-family models.

    The demand location and multiplier come from the independently selected
    demand-family fit. The conditional test then searches every junction for a
    leak, all configured leak magnitudes, and a network-independent onset grid.
    Selection uses even hours and all acceptance checks use held-out odd hours.
    """
    if onset_step_hours < 1:
        raise ValueError("onset_step_hours must be positive")
    demand_parameters = demand_evidence.best_parameters
    demand_node = str(demand_parameters["node"])
    demand_multiplier = float(demand_parameters["multiplier"])
    train_hours = list(range(0, config.duration_hours, 2))
    validation_hours = list(range(1, config.duration_hours, 2))
    baseline_train, _, _ = _combined_score(
        observed, nominal, nominal, config, train_hours
    )
    baseline_validation, _, _ = _combined_score(
        observed, nominal, nominal, config, validation_hours
    )
    wn_for_names = load_hydraulic_network(
        inp_path, config.duration_hours, config.demand_model
    )
    onset_candidates = list(range(0, config.duration_hours, onset_step_hours))
    candidates: List[Tuple[float, Dict[str, object], pd.DataFrame]] = []
    simulations = 0
    for leak_node in wn_for_names.junction_name_list:
        for leak_area in config.search_policy.leak_areas_m2:
            for leak_onset in onset_candidates:
                faults = [
                    FaultSpec(
                        "demand_error",
                        demand_node,
                        0,
                        config.duration_hours,
                        demand_multiplier,
                    ),
                    FaultSpec(
                        "leak",
                        leak_node,
                        _candidate_leak_start_hour(leak_onset, config.solver),
                        config.duration_hours,
                        float(leak_area),
                    ),
                ]
                prediction = _simulate_compound_candidate(inp_path, config, faults)
                simulations += 1
                if not _candidate_is_complete(prediction, config):
                    continue
                train_score, _, _ = _combined_score(
                    observed, prediction, nominal, config, train_hours
                )
                candidates.append(
                    (
                        train_score,
                        {
                            "demand_node": demand_node,
                            "demand_multiplier": demand_multiplier,
                            "leak_node": leak_node,
                            "leak_area_m2": float(leak_area),
                            "leak_start_hour": int(leak_onset),
                        },
                        prediction,
                    )
                )
    if not candidates:
        raise RuntimeError("No complete leak+demand compound candidates were generated")

    train_score, best_parameters, best_prediction = min(
        candidates, key=lambda item: item[0]
    )
    validation_score, pressure_rmse, flow_nrmse = _combined_score(
        observed, best_prediction, nominal, config, validation_hours
    )
    best_single_train = min(
        leak_evidence.training_score_after,
        demand_evidence.training_score_after,
    )
    best_single_validation = min(
        leak_evidence.validation_score_after,
        demand_evidence.validation_score_after,
    )
    train_gain_over_single = _improvement(best_single_train, train_score)
    validation_gain_over_single = _improvement(
        best_single_validation, validation_score
    )
    train_improvement = _improvement(baseline_train, train_score)
    validation_improvement = _improvement(
        baseline_validation, validation_score
    )
    pressure_noise_ratio = pressure_rmse / config.pressure_noise_std_m
    flow_noise_ratio = flow_nrmse / config.flow_noise_fraction
    policy = config.evidence_policy
    noise_consistent = bool(
        pressure_noise_ratio <= policy.strong_max_pressure_noise_ratio
        and flow_noise_ratio <= policy.strong_max_flow_noise_ratio
    )
    both_families_partially_supported = all(
        evidence.training_improvement >= policy.moderate_min_training_improvement
        and evidence.validation_improvement
        >= policy.moderate_min_validation_improvement
        for evidence in (leak_evidence, demand_evidence)
    )
    verified = bool(
        both_families_partially_supported
        and noise_consistent
        and train_improvement >= policy.strong_min_training_improvement
        and validation_improvement >= policy.strong_min_validation_improvement
        and train_gain_over_single >= minimum_gain_over_single
        and validation_gain_over_single >= minimum_gain_over_single
    )
    result = {
        "hypothesis": "leak+demand_error",
        "best_parameters": best_parameters,
        "training_score_before": baseline_train,
        "training_score_after": train_score,
        "validation_score_before": baseline_validation,
        "validation_score_after": validation_score,
        "training_improvement": train_improvement,
        "validation_improvement": validation_improvement,
        "training_gain_over_best_single": train_gain_over_single,
        "validation_gain_over_best_single": validation_gain_over_single,
        "pressure_rmse_m": pressure_rmse,
        "flow_nrmse": flow_nrmse,
        "pressure_noise_ratio": pressure_noise_ratio,
        "flow_noise_ratio": flow_noise_ratio,
        "noise_consistent": noise_consistent,
        "both_families_partially_supported": both_families_partially_supported,
        "minimum_gain_over_single": minimum_gain_over_single,
        "simulations": simulations,
        "result": "verified_compound_support" if verified else "not_verified",
    }
    return result, best_prediction


def test_roughness_demand_compound(
    inp_path: str | Path,
    observed: pd.DataFrame,
    nominal: pd.DataFrame,
    config: ObservationConfig,
    roughness_evidence: TestEvidence,
    demand_evidence: TestEvidence,
    minimum_gain_over_single: float = 0.10,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Verify a roughness+demand explanation on interleaved held-out hours."""
    demand_parameters = demand_evidence.best_parameters
    demand_node = str(demand_parameters["node"])
    demand_multiplier = float(demand_parameters["multiplier"])
    train_hours = list(range(0, config.duration_hours, 2))
    validation_hours = list(range(1, config.duration_hours, 2))
    baseline_train, _, _ = _combined_score(
        observed, nominal, nominal, config, train_hours
    )
    baseline_validation, _, _ = _combined_score(
        observed, nominal, nominal, config, validation_hours
    )
    wn_for_names = load_hydraulic_network(
        inp_path, config.duration_hours, config.demand_model
    )
    candidates: List[Tuple[float, Dict[str, object], pd.DataFrame]] = []
    simulations = 0
    for pipe_name in wn_for_names.pipe_name_list:
        for roughness_multiplier in config.search_policy.roughness_multipliers:
            faults = [
                FaultSpec(
                    "demand_error",
                    demand_node,
                    0,
                    config.duration_hours,
                    demand_multiplier,
                ),
                FaultSpec(
                    "roughness_error",
                    pipe_name,
                    0,
                    config.duration_hours,
                    float(roughness_multiplier),
                ),
            ]
            prediction = _simulate_compound_candidate(inp_path, config, faults)
            simulations += 1
            if not _candidate_is_complete(prediction, config):
                continue
            train_score, _, _ = _combined_score(
                observed, prediction, nominal, config, train_hours
            )
            candidates.append(
                (
                    train_score,
                    {
                        "demand_node": demand_node,
                        "demand_multiplier": demand_multiplier,
                        "pipe": pipe_name,
                        "roughness_multiplier": float(roughness_multiplier),
                    },
                    prediction,
                )
            )
    if not candidates:
        raise RuntimeError(
            "No complete roughness+demand compound candidates were generated"
        )

    train_score, best_parameters, best_prediction = min(
        candidates, key=lambda item: item[0]
    )
    validation_score, pressure_rmse, flow_nrmse = _combined_score(
        observed, best_prediction, nominal, config, validation_hours
    )
    best_single_train = min(
        roughness_evidence.training_score_after,
        demand_evidence.training_score_after,
    )
    best_single_validation = min(
        roughness_evidence.validation_score_after,
        demand_evidence.validation_score_after,
    )
    train_gain_over_single = _improvement(best_single_train, train_score)
    validation_gain_over_single = _improvement(
        best_single_validation, validation_score
    )
    train_improvement = _improvement(baseline_train, train_score)
    validation_improvement = _improvement(
        baseline_validation, validation_score
    )
    pressure_noise_ratio = pressure_rmse / config.pressure_noise_std_m
    flow_noise_ratio = flow_nrmse / config.flow_noise_fraction
    policy = config.evidence_policy
    noise_consistent = bool(
        pressure_noise_ratio <= policy.strong_max_pressure_noise_ratio
        and flow_noise_ratio <= policy.strong_max_flow_noise_ratio
    )
    both_families_partially_supported = all(
        evidence.training_improvement >= policy.moderate_min_training_improvement
        and evidence.validation_improvement
        >= policy.moderate_min_validation_improvement
        for evidence in (roughness_evidence, demand_evidence)
    )
    verified = bool(
        both_families_partially_supported
        and noise_consistent
        and train_improvement >= policy.strong_min_training_improvement
        and validation_improvement >= policy.strong_min_validation_improvement
        and train_gain_over_single >= minimum_gain_over_single
        and validation_gain_over_single >= minimum_gain_over_single
    )
    result = {
        "hypothesis": "roughness_error+demand_error",
        "best_parameters": best_parameters,
        "training_score_before": baseline_train,
        "training_score_after": train_score,
        "validation_score_before": baseline_validation,
        "validation_score_after": validation_score,
        "training_improvement": train_improvement,
        "validation_improvement": validation_improvement,
        "training_gain_over_best_single": train_gain_over_single,
        "validation_gain_over_best_single": validation_gain_over_single,
        "pressure_rmse_m": pressure_rmse,
        "flow_nrmse": flow_nrmse,
        "pressure_noise_ratio": pressure_noise_ratio,
        "flow_noise_ratio": flow_noise_ratio,
        "noise_consistent": noise_consistent,
        "both_families_partially_supported": both_families_partially_supported,
        "minimum_gain_over_single": minimum_gain_over_single,
        "simulations": simulations,
        "result": "verified_compound_support" if verified else "not_verified",
    }
    return result, best_prediction


def classify_four_outcome(
    evidence_by_family: Dict[str, TestEvidence],
    compound_evidence: Dict[str, object] | None = None,
    evidence_policy: EvidencePolicy | None = None,
) -> Dict[str, object]:
    """Return one of four operational outcomes from deterministic evidence."""
    if compound_evidence and compound_evidence.get("result") == "verified_compound_support":
        compound_families = str(compound_evidence["hypothesis"]).split("+")
        return {
            "outcome": "compound_cause",
            "diagnosis": compound_families,
            "rationale": "The joint model is noise-consistent and outperforms both single-family fits on held-out hours.",
        }
    strong = [
        family
        for family, evidence in evidence_by_family.items()
        if evidence.result == "strong_support"
    ]
    moderate_or_strong = [
        family
        for family, evidence in evidence_by_family.items()
        if evidence.result in {"moderate_support", "strong_support"}
    ]
    if len(strong) == 1 and len(moderate_or_strong) == 1:
        return {
            "outcome": "single_cause",
            "diagnosis": strong,
            "rationale": "One family has strong support without a competing noise-plausible fit.",
        }
    if len(moderate_or_strong) >= 2:
        return {
            "outcome": "ambiguous_causes",
            "diagnosis": moderate_or_strong,
            "rationale": "Multiple families are noise-plausible, but no unique joint explanation has been verified.",
        }
    return {
        "outcome": "unknown_cause",
        "diagnosis": [],
        "rationale": "No known single or compound explanation satisfies the deterministic evidence requirements.",
    }


def topology_context(
    inp_path: str | Path,
    sensor_nodes: Sequence[str],
    config: ObservationConfig | None = None,
) -> Dict[str, object]:
    """Compact topology context for the agent, without sending the whole INP file."""
    duration_hours = config.duration_hours if config is not None else 24
    demand_model = config.demand_model if config is not None else "PDD"
    wn = load_hydraulic_network(inp_path, duration_hours, demand_model)
    graph: nx.Graph = wn.to_graph().to_undirected()
    sensor_distances = {}
    for source in sensor_nodes:
        sensor_distances[source] = {
            target: int(nx.shortest_path_length(graph, source, target))
            for target in sensor_nodes
            if target != source and nx.has_path(graph, source, target)
        }
    context = {
        "junction_count": wn.num_junctions,
        "pipe_count": wn.num_pipes,
        "pump_count": wn.num_pumps,
        "reservoir_count": wn.num_reservoirs,
        "tank_count": wn.num_tanks,
        "pressure_sensors": list(sensor_nodes),
        "valid_junctions": list(wn.junction_name_list),
        "valid_pipes": list(wn.pipe_name_list),
        "sensor_shortest_path_distances": sensor_distances,
    }
    if config is not None:
        context["deterministic_test_costs"] = {
            "sensor_bias": {
                "candidate_count": len(sensor_nodes),
                "estimated_hydraulic_simulations": 0,
                "basis": "analytic correction at each monitored pressure sensor",
            },
            "leak": {
                "candidate_count": (
                    wn.num_junctions * len(config.search_policy.leak_areas_m2)
                ),
                "estimated_hydraulic_simulations": (
                    wn.num_junctions * len(config.search_policy.leak_areas_m2)
                ),
                "basis": "junctions multiplied by configured leak areas",
            },
            "demand_error": {
                "candidate_count": (
                    wn.num_junctions
                    * len(config.search_policy.demand_multipliers)
                ),
                "estimated_hydraulic_simulations": (
                    wn.num_junctions
                    * len(config.search_policy.demand_multipliers)
                ),
                "basis": "junctions multiplied by configured demand multipliers",
            },
            "roughness_error": {
                "candidate_count": (
                    wn.num_pipes
                    * len(config.search_policy.roughness_multipliers)
                ),
                "estimated_hydraulic_simulations": (
                    wn.num_pipes
                    * len(config.search_policy.roughness_multipliers)
                ),
                "basis": "pipes multiplied by configured roughness multipliers",
            },
        }
    return context
