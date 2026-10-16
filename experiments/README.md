# Experiment modules

Run modules from the repository root with `python -m experiments.<group>.<module>`. Module names are listed in `docs/module_map.json`, together with their previous ungrouped names. Core imported helpers remain under `shared/` because later experiments depend on them; their presence does not mean their older verifier policy supplied the final results.

The final configured-family decision policy is implemented in `experiment_3_hydraulic_verification/evaluate_anytown_hybrid_v3_adequacy.py` and documented in `configs/verifier/generalized_verifier_v4.json`. The contextual follow-up policy is separate and lives in `experiment_4_context_followup/verify_experiment4_operator_followups.py`.

Use `scripts/reproduce_results.py` and `scripts/reproduce_figures.py` for the validated saved-output workflow. Original runners with arguments remain available for fresh studies. Supply paths from `docs/run_map.json` and use a separate `results/reruns/` destination to preserve the frozen result snapshot.
