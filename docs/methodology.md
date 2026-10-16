# Methodology

The initial hypothesis library contains leakage, demand error, pipe-roughness error, and pressure-sensor bias. The LLM prioritizes families; every family is evaluated before the authoritative verdict, so ranking affects provisional search order rather than the exhaustive final diagnosis.

Candidate magnitudes and locations are fitted on even hours and evaluated on odd hours. The final verifier requires training and validation improvements of at least 0.02 and a held-out noise-relative goodness-of-fit p-value of at least 0.05. A relative separation of 0.07 distinguishes supported alternatives. With no supported family, a nominal-model goodness-of-fit check distinguishes no fault from unknown cause. Read the implementation for exact boundary comparisons.

Anytown uses WNTR/PDD with time-controlled leaks; Net3 and BWN-II use EPANET/DDA and persistent emitter leaks. The profiles contain sensor IDs, numerical noise assumptions, and search bounds. Their solver differences are part of the benchmark design.

Experiment 4 additionally allows a source-head test. Its context gate checks cited record IDs, high-reliability metadata, observation-window timing, and the LLM's relevance assessment. A focused test is accepted when the nominal model fails held-out adequacy, the focused candidate passes at alpha 0.05, and fitting/validation improvements are nonnegative. This differs from the exhaustive initial verifier. The reported safety improvement measures the combined gate and hydraulic test.

The sensitivity analysis changes decision settings on fixed fitted evidence. It does not refit the hydraulic models or test robustness to arbitrary sensor layouts, correlations, or simultaneous faults. Historical method notes are retained under `historical/` and may describe earlier policies; use the final configuration and implementation as the release authority.
