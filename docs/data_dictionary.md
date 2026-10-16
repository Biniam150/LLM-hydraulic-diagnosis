# Data dictionary

| Field or artifact | Meaning |
|---|---|
| `hour` | Simulation hour; canonical exported observations have 24 samples, hours 0–23 |
| Pressure observation columns | Simulated noisy pressure in metres at the configured sensor; names follow the engine export |
| Flow observation column | Simulated noisy flow at the configured monitored link; use the engine/network unit conversion definitions |
| `case_id` | Neutral ID within a network's canonical exported dataset |
| `source_neutral_case_id` | ID used by the original saved diagnostic run |
| `scenario_id` | Evaluator scenario identifier; may reveal the injected cause and must be withheld from blind inference |
| `true_cause`, `true_target`, `magnitude` | Evaluator-only injected mechanism, component, and family-specific parameter |
| `observation_seed`, `seed` | Pseudorandom noise seed; not a physical parameter |
| `expected_label` | Expected initial diagnostic outcome; source head maps to `unknown_cause` |
| `predicted_label` | Returned family or unresolved/no-fault outcome |
| `logical_hydraulic_simulations` | Logical full-search simulation count as recorded by the experiment |
| `context_condition` | Helpful verified, irrelevant verified, conflicting low-reliability, or supporting low-reliability synthetic record condition |
| `focused_hydraulic_accepted` | Focused follow-up passes the separate acceptance rule |
| `safe_task_correct` | Context-condition-specific endpoint; not universally the same as family correctness |
| `hydraulic_evidence` | Serialized evidence JSON; the normalized tested target is recorded here |

Cause-family magnitudes use leak area in m², demand/roughness multipliers, pressure-sensor offset in metres, or source-head offset in metres. Ground-truth and operational-record files are evaluator inputs, not a ready-to-send LLM prompt.
