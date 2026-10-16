# Hydraulic Digital-Twin Diagnostic Experiment

## Purpose

This experiment tests a different problem from known-parameter calibration. The
diagnostic agent is not told which cause family generated the mismatch. It must
prioritize one of four bounded hypotheses, request deterministic WNTR tests, and
either diagnose a supported family or abstain.

The allowed diagnosis families are:

1. unreported leak;
2. local demand error;
3. pipe-roughness error; and
4. pressure-sensor bias.

A source-head error is included as an out-of-library negative control. The correct
response to that case is abstention because source-head error is deliberately not
available to the agent.

## Separation of responsibilities

- The local LLM ranks hypotheses, explains its evidence, and requests one bounded
  family test.
- WNTR performs all hydraulic calculations and searches candidate parameters.
- For roughness testing on Anytown, the deterministic tool searches all pipes. The
  LLM selects the family but cannot exclude pipes from the numerical search domain.
- A schema validator rejects unknown hypotheses, invalid network elements, and
  unsupported diagnoses.
- A deterministic evidence gate requires strong improvement on interleaved held-out
  hours before a diagnosis is accepted.
- The hidden ground-truth manifest is written to disk for evaluation but is never
  included in the agent prompt.

## Network and observations

The experiment uses `data/networks/anytown/anytown.inp` with a 24-hour, one-hour-step pressure-
dependent-demand simulation. Pseudo-observations contain pressure at junctions 20,
50, 70, 100, 120, 150, and 170, plus flow at pump/link 82.

The default leak demonstration injects a hidden leak at uninstrumented junction 90
from hour 8 through hour 24 with an area of 0.001 square metres. Gaussian pressure
noise of 0.03 m and flow noise of 1 percent are added with a fixed seed.

## Run the local-LLM demonstration

```powershell
.\venv\Scripts\python.exe run_hydraulic_diagnosis_demo.py `
  --mode ollama `
  --model qwen2.5:7b `
  --fault leak `
  --output outputs\hydraulic_demo_ollama
```

The transparent non-LLM baseline uses the identical evidence loop:

```powershell
.\venv\Scripts\python.exe run_hydraulic_diagnosis_demo.py `
  --mode rule `
  --fault leak `
  --output outputs\hydraulic_demo_rule
```

## Run the multi-fault matrix

```powershell
.\venv\Scripts\python.exe evaluate_hydraulic_diagnosis.py `
  --mode rule `
  --seeds 42 43 44 `
  --output outputs\hydraulic_evaluation_rule
```

After the fast baseline validates the scenario definitions, run the same matrix with
the local model:

```powershell
.\venv\Scripts\python.exe evaluate_hydraulic_diagnosis.py `
  --mode ollama `
  --model qwen2.5:7b `
  --seeds 42 43 44 `
  --output outputs\hydraulic_evaluation_ollama
```

Local-model runs are intentionally slower because every diagnostic round is logged
and evaluated separately.

## Important outputs

- `ground_truth_manifest.json`: hidden label and reproducible injection parameters;
- `observations.csv`: noisy pseudo-sensor data;
- `nominal_prediction.csv`: unfaulted digital-twin output;
- `nominal_residuals.csv`: observed minus nominal signals;
- `residual_features.json`: compact information shown to the agent;
- `agent_round_*.json`: validated decisions;
- `ollama_raw_round_*.txt`: unmodified local-model outputs;
- `test_round_*.json`: deterministic fitting and held-out evidence;
- `diagnostic_summary.json`: final diagnosis, correctness, and simulation count;
- `pressure_residuals.png` and `final_hypothesis_ranking.png`: visual summaries.

## Current claim and limitations

The leak demonstration establishes end-to-end feasibility. It does not establish
general diagnostic performance. A paper-level evaluation should randomize fault
locations and magnitudes, use multiple noise seeds, compare multiple networks, and
measure top-k family accuracy, correct abstention, ranking stability, and simulation
cost.

The present family tests change one element at a time. Multi-fault cases and spatially
distributed demand/roughness errors remain future extensions. The synthetic sensor
data are useful for controlled validation but do not replace evaluation on utility
data.
