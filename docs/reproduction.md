# Reproduction

Run commands from the repository root using an environment with the dependencies in `pyproject.toml`. Inputs are resolved relative to this checkout, not the original user's desktop. This is a source-checkout workflow; network/data files are not embedded into a wheel.

## Saved-output workflow

```sh
python scripts/validate_release.py --checksums
python scripts/reproduce_results.py
python scripts/reproduce_figures.py
python scripts/run_tests.py
```

The first command checks the staged snapshot. Summary reproduction reads saved case decisions and evidence and does not perform LLM inference or refit the full candidate library. It also regenerates some diagnostic plots. Figure reproduction generates numerical Figures 3–7; intermediate plotting tables/figures may also be emitted. Framework and topology artwork require separate authoring.

The test runner includes both unittest classes and plain zero-argument regression functions. It requires hydraulic dependencies but no live Ollama service; LLM tests use simulated responses.

## Synthetic observations

```sh
python scripts/export_benchmark.py
```

This regenerates the canonical final 60 observation files from the frozen final case IDs, scenario definitions, network models, and seed 1009. Evaluator truth is exported separately. It does not rerun the historical LLM outputs or guarantee byte-for-byte equivalence across arbitrary solver/library versions. The release inventory records the exported observation hashes.

## Fresh initial verification

The following example repeats the Anytown full search and can be substantially more expensive than summary reproduction:

```sh
python -m experiments.experiment_3_hydraulic_verification.evaluate_anytown_hybrid_v3_adequacy --llm-results results/experiment_2/experiment2_anytown_qwen3_8b_seed1009/llm_only_causal_results.csv --case-index results/source_inputs/anytown_causal_benchmark/causal_benchmark_cases.csv --design configs/scenarios/anytown_causal_benchmark.json --profile configs/networks/anytown.json --candidate-cache results/reruns/cache --output results/reruns/anytown_verifier
```

The final Anytown run's exact source LLM label/order can differ from this example, while exhaustive hydraulic verdicts are order-independent. Use `docs/run_map.json` and the retained run metadata to select the exact input run for a historical rerun.

## Contextual follow-up

Each network's complete synthetic records and evaluator mapping are in `data/operational_records/<network>/`. Fresh inference uses `run_experiment4_operator_context_llm`; focused verification uses `verify_experiment4_operator_followups`, both under `experiments.experiment_4_context_followup`. Inspect `--help` for required paths. The prompt builder excludes truth fields; never send the complete evaluator JSON directly to the model. Source head is available in this follow-up stage.

Five Ollama tags are listed in `configs/models/models.json`. Reusing a tag without recording the installed digest is not a guarantee of identical model weights. Fresh classical baselines additionally require the optional `classical` dependencies. Large candidate caches and new run outputs belong under ignored `results/reruns/` until curated.
