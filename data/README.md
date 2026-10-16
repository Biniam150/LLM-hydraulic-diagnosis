# Study inputs

`networks/` contains exact copied `.inp` inputs. Record their provenance and redistribution terms in each network README before public release.

`benchmark/{anytown,net3,bwnii}/case_index.csv` links neutral case IDs to 24-hour noisy synthetic observations. `evaluation_only/ground_truth.csv` contains labels and injected parameters and must not be supplied to the LLM. These neutral IDs are local to the exported dataset; `source_neutral_case_id` links evaluator truth to the original recorded run IDs. The scenario selection follows the final Experiment 3 case results, not every preliminary scenario in the broader design files.

The observations were regenerated during repository assembly using the frozen final case definitions and seed 1009. Their generation checks and hashes are retained. The saved LLM decisions were not rerun during assembly.

`development/` contains classical baseline development tables, manifests, and frozen test features. It is distinct from the blind evaluation set. `operational_records/` contains synthetic Experiment 4 records with evaluator metadata; the runner's prompt builder selects the blind fields and excludes truth. Do not send the complete benchmark JSON directly to an LLM.

Historical input/evidence snapshots referenced by the original scripts are retained under `results/source_inputs/`; they can contain broader development tiers or seeds. The canonical final 60-case release is under `data/benchmark/`.
