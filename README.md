# LLM hydraulic diagnosis

Code, synthetic benchmark inputs, saved results, and figures for **Diagnosing Model-Observation Discrepancies in Water Distribution Networks with Local Large Language Models**.

The framework combines locally hosted LLM interpretation with deterministic hydraulic fitting and verification. It evaluates leakage, demand error, pipe-roughness error, and pressure-sensor bias, with explicit ambiguous, unknown-cause, and no-fault outcomes. Source-head discrepancies are withheld from the initial hypothesis library and may be investigated during contextual follow-up.

## Repository contents

| Folder | Contents |
|---|---|
| `data/` | Network models, regenerated synthetic observations, evaluator-only truth, classical development inputs, and synthetic operational records |
| `configs/` | Network profiles, scenario designs, verifier settings, and model identities |
| `wdn_diagnostics/` | Reusable hydraulic engine and local LLM client |
| `experiments/` | Runners, summarizers, and plotting modules grouped by experiment |
| `results/` | Saved case-level evidence, decisions, summaries, sensitivity analysis, and provenance manifests |
| `figures/` | Numerical paper figures and retained framework artwork |
| `scripts/` | Benchmark export, summary/figure reproduction, regression tests, and release validation |
| `tests/` | Regression tests copied and adapted to the repository paths |
| `docs/` | Reproduction guide, methodology, data dictionary, and supplementary scenario tables |

## Study organization

1. **Assumed-family calibration:** roughness-only and demand-only fitting in six Anytown conditions; seeds 101, 202, and 404.
2. **Residual-only diagnosis:** five local LLMs and classical classifiers across Anytown, Net3, and BWN-II.
3. **Hydraulic verification:** independent fitting/held-out evaluation, abstention, and search-order comparisons.
4. **Contextual follow-up:** synthetic operational records guide focused tests of 20 unresolved hydraulic cases.

Decision-policy sensitivity is reported separately. The main diagnostic benchmark contains 18 Anytown cases, 21 Net3 cases, and 21 BWN-II cases, using observation seed 1009. Observations are synthetic; these are not field measurements.

## Quick start

Use a source checkout and Python 3.10 or later. The exact core package versions in `pyproject.toml` reflect the existing scientific environment used to check this staging copy. A clean installation has not yet been independently tested.

```sh
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -e .
python scripts/validate_release.py --checksums
python scripts/reproduce_results.py
python scripts/reproduce_figures.py
python scripts/run_tests.py
```

Summary and figure reproduction uses saved outputs and does not call an LLM. Regeneration overwrites the corresponding derived tables/figures; original-copy hashes and staging baseline hashes are retained in separate manifests. Run checksum validation before regeneration; afterward use validation without `--checksums` to check numerical content.

For fresh classical-model fitting install `python -m pip install -e ".[classical]"`. The exact original scikit-learn version was not recoverable from the currently inspected scientific environment; the saved classifier decisions are included. Fresh LLM inference requires a local Ollama service and the models listed in `configs/models/models.json`. See [reproduction instructions](docs/reproduction.md) before rerunning inference or the full hydraulic search.

## Reported verifier results

| Network | Cases | Exact-outcome accuracy | Accepted family diagnoses | Correct accepted diagnoses |
|---|---:|---:|---:|---:|
| Anytown | 18 | 72.2% | 9 | 9 |
| Net3 | 21 | 61.9% | 8 | 8 |
| BWN-II | 21 | 76.2% | 11 | 11 |

All tested source-head cases and no-fault controls were correctly classified by the initial verifier. These finite synthetic benchmark results do not establish guaranteed precision in operational networks. Noise assumptions materially affect acceptance. LLM ranking does not change the exhaustive verifier's final decision.

## Provenance and release status

`results/publication_manifest.csv` maps copied files to original project paths and records original and relocated hashes. `results/release_inventory.csv` inventories the complete staged deliverable. `data/provenance.csv` identifies observations regenerated from frozen scenarios, rather than implying they were recovered as original recorded files. Historical run-directory names are retained inside each experiment for traceability.

This is a local publication staging copy. No GitHub remote has been created or updated. Before public release, authors must select the code license, confirm redistribution terms for the three network files, complete model digest/quantization details, and reconcile the supplementary material with the final manuscript. No code license has been invented. See [release status](docs/release_status.md).
