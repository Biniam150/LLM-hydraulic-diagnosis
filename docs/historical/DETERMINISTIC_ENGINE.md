# Deterministic hydraulic diagnostic engine

## Design boundary

The implementation separates the transferable diagnostic workflow from
network-specific study inputs.

The core workflow is:

1. calculate observed-minus-nominal residuals;
2. standardize residuals using configured measurement uncertainty;
3. extract spatial and temporal features;
4. fit a requested hypothesis family with deterministic tools;
5. select parameters using training observations;
6. evaluate the selected parameters on held-out observations;
7. classify evidence using post-fit noise consistency and explanatory
   improvement;
8. diagnose only a tested, strongly supported family.

The core does not contain rules such as "negative residuals imply a leak" or
"one affected sensor implies sensor bias." Those Anytown-oriented weights were
removed.

## Network profile

`configs/networks/anytown.json` contains the Anytown-specific configuration:

- INP filename;
- pressure- and flow-sensor IDs;
- measurement-noise assumptions;
- duration;
- feature thresholds;
- candidate parameter ranges;
- deterministic test order;
- hidden demonstration scenarios.

A profile for another network must supply the same categories, but it should
use sensor uncertainty and engineering bounds appropriate to that network.

The current deterministic test order is an explicit baseline policy. It is not
claimed to be an optimal or transferable hydraulic inference rule.

## Transfer-oriented changes

### Standardized detection

Pressure detection uses:

`absolute residual >= residual_z_threshold * pressure_noise_std_m`

The profile currently uses a three-standard-deviation threshold.

### Robust onset

Onset requires the threshold to be exceeded for the configured number of
consecutive reporting times. The Anytown profile uses two consecutive hours,
which prevents one isolated noisy value from defining the event onset.

### Search independence

The default candidate-node scope is `full_network`. LLM-suggested nodes cannot
exclude the true leak or demand node. `requested_plus_neighbors` remains
available only as an explicit, profile-level approximation for large networks.

### Noise-aware evidence

The fitted validation errors are divided by configured measurement noise:

- `pressure_noise_ratio = pressure_rmse / pressure_noise_std_m`
- `flow_noise_ratio = flow_nrmse / flow_noise_fraction`

Strong support requires both post-fit ratios to be within the configured noise
limits and requires positive training and validation improvement. This replaces
the previous Anytown-only rule that required exactly 25% training improvement
and 35% validation improvement.

## Running the profiled Anytown experiment

```powershell
python run_hydraulic_diagnosis_demo.py `
  --profile configs/networks/anytown.json `
  --mode rule `
  --fault leak `
  --seed 101 `
  --max-rounds 4 `
  --output outputs\leak_seed101_profiled_rule
```

The result summary records the complete feature, search, evidence, and
test-order policies so every run can be audited.

## Evidence-grounded LLM interface (V2)

The experiment records deterministic test costs calculated from the loaded
network:

- sensor bias: monitored pressure-sensor count, with zero hydraulic candidates;
- leak: junction count multiplied by configured leak areas;
- demand error: junction count multiplied by demand multipliers;
- roughness error: pipe count multiplied by roughness multipliers.

These costs are descriptive secondary metrics and are not supplied to the LLM
or used as its planning objective. The prompt instead defines the numerical
meaning of noise ratios and score improvement and warns that graph distance is
not evidence of leakage. Invalid structured output receives one repair attempt
before deterministic fallback.

Every round separately records the raw LLM top hypothesis, raw requested
hypothesis, executed hypothesis, estimated test cost, repair status, fallback
status, and fallback reason. The final evaluation additionally records the
correct family's initial rank and test position. Deterministic parameter
fitting, held-out evidence classification, and the strong-support diagnosis
gate are unchanged.

The required LLM JSON is deliberately compact: hypothesis names and scores,
one requested family and reason, the decision, diagnosis, confidence, and a
short rationale. Large valid-node and valid-pipe lists remain in the
deterministic validator but are not placed in the prompt. A repair call receives
the exact parser error, uses a different generation seed, and must regenerate a
short complete object rather than continue a truncated response.

Normal generations use a 1,000-token limit. The Ollama client records
`done_reason`, prompt and generated token counts, and timing metadata for every
attempt. Only a response explicitly terminated with `done_reason="length"`
triggers a concise regeneration with a 2,000-token allowance. Metadata is saved
beside the raw response and summarized as length-limit and generation-attempt
provenance; other parse or schema errors retain the normal repair budget.

## Sensor-bias bounds

Profiles can use an explicit absolute bound or a baseline-scaled bound. BWNII
uses:

```text
half_width = max(
    0.05 * median(abs(nominal monitored pressure)),
    10 * pressure_noise_std_m
)
```

Its current nominal median monitored pressure is approximately 63.23 m, giving
a symmetric bound of approximately ±3.16 m. The resolved bounds and baseline
pressure scale are recorded with the fitted sensor-bias parameters. Anytown
retains its explicit ±2 m development-profile bound.

## Selecting a network

The bundled profiles can be selected without replacing either INP model:

```powershell
python run_hydraulic_diagnosis_demo.py --network anytown --mode rule --fault leak
python run_hydraulic_diagnosis_demo.py --network bwnii --mode rule --fault sensor_bias
```

An explicit profile remains supported:

```powershell
python run_hydraulic_diagnosis_demo.py --profile configs/networks/bwnii.json ...
```

The solver is part of the profile. Anytown uses `WNTRSimulator`; BWNII uses
EPANET 2.2 because its variable-speed pump controls are unsupported by the
installed WNTR hydraulic solver. Persistent BWNII leaks are represented as
pressure-dependent EPANET emitters using the same discharge equation as the
WNTR leak model.

## Adding another network

1. Copy `configs/networks/anytown.json`.
2. Change the INP path and sensor IDs.
3. Enter sensor uncertainty values supported by the monitoring system.
4. Define engineering search bounds without using hidden test labels.
5. Define scenarios that reference valid assets in the new network.
6. Run profile validation and the diagnostic regression suite.
7. Freeze the profile before evaluating hidden scenarios.

Changing the general method after inspecting a held-out network makes that
network part of development. A new network or leave-one-network-out evaluation
is then required for an unbiased transfer claim.

## Remaining work

This refactor establishes a clean boundary but does not by itself prove
cross-network transfer. The next methodological steps are:

- generate hydraulic sensitivity signatures for automatic candidate screening;
- replace fixed parameter grids with network-scaled engineering bounds or
  optimizers;
- add explicit provisional and ambiguous outcomes;
- compare exhaustive, fixed-order, rule-based, and LLM-guided selection;
- freeze the engine and test it on an unseen benchmark network.
