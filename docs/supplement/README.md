# Supplementary study inputs

The three `*_scenarios.csv` files list the final retained benchmark scenarios and map neutral exported IDs to recorded run IDs. They were generated from the final verifier case selection and frozen scenario profiles.

- `net3_scenarios.csv` supplies the scenario details referenced as Table S1 in the current manuscript.
- `bwnii_scenarios.csv` supplies the scenario details referenced as Table S2.
- `anytown_exclusions.csv` lists the three matched-model exclusions from the frozen Anytown profile for Table S3. The source records the common feasibility-screening rationale; it does not supply a distinct detailed solver error for each row.
- `anytown_scenarios.csv` provides a machine-readable counterpart to the retained Anytown scenario table.

These CSVs are source tables for a supplement, not a completed journal supplementary document. Experiment 1's actual six-scenario definitions and seeds are in `configs/scenarios/anytown_false_roughness_calibration.json` and `anytown_false_demand_calibration.json`; these are different from the final diagnostic case set.

Complete context records are in `data/operational_records/`; their prompt generation and validation are implemented under `experiments/experiment_4_context_followup/`. The experimental condition names are evaluator metadata. Reliability and timing fields are supplied to the context gate, which checks them before fitting a focused candidate.
