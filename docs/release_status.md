# Release status

This staging copy includes the selected source code, three network files, scenario/configuration files, saved case results, LLM responses where available, figures, and reproduction utilities. Original project files remain unchanged.

Before public release:

- Authors must choose a code license. There is deliberately no fabricated LICENSE file.
- Confirm the original source and redistribution terms of each network model; copying a network into this local staging folder does not establish public distribution permission.
- Complete exact Ollama model digests, quantization, runtime version, and hardware details. Tags and saved request/response metadata are retained.
- Recover the original scikit-learn version if exact retraining reproduction is required. The current inspected environment lacks that package; saved classifier results remain reproducible at the summary level.
- Confirm correspondence of the selected figure candidates to the final manuscript. Figure 1 framework artwork is retained; the network topology artwork for Figure 2 requires final author selection and is not silently substituted.
- Convert the exported scenario CSVs and documented context protocol into the final journal supplement. Scenario CSVs do not by themselves constitute the completed supplementary manuscript.
- Resolve the manuscript's seed wording, duplicated text, and policy descriptions identified in the content review.

See `validation_report.json` for checks actually performed during assembly. Full hydraulic searches, LLM inference, a clean dependency installation, and visual manuscript layout are separate checks and are not implied by summary reproduction.
