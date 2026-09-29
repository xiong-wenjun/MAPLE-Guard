# Restart protocol for interrupted AppWorld experiments

This recovery preserves failed runs as diagnostics and uses distinct run identities. Partially executed memory streams are not spliced into new results.

## INFA-Guard

Generation selects outputs by exact recipe grid, uses isolated native output directories, and locks each grid before inspection/publication. Markerless outputs require held-out, dialogue, journal, cache-only native replay, and full-data equality validation. Migration verifies previous frozen source/stage hashes and preserves attempt history in an audit. Incompatible outputs are not silently counted or overwritten.

The released generation model, prompts, sampling, token budget and accepted finish reasons remain unchanged. Training requires all 800 audited dialogues, followed by 50 epochs and checkpoint validation before evaluation.

## AgentSafe

The old Qwen run exceeded the deployed 32,768-token context. Sidecars do not provide complete transactional task checkpoints; both backbones restart fresh 200-task streams.

The explicitly disclosed bounded_recent_v1 adaptation counts the full local chat template, reserves the original output budget plus 128 tokens of margin, and selects complete retained records by first-observation recency. Text already in the task prompt is omitted from the appended view. Full HierarCache storage is preserved. Traces report selected/deferred/duplicate record IDs and token counts. Reflection visits all prior junk in complete-record batches and treats a candidate as junk if any batch judges it so. These selection and batching choices are additional host adaptations, not recovered author settings.

Criteria, admission threshold, independent calibration and review interval remain frozen. The previous 0/64 validation detection for the cosine admission criterion remains a disclosed limitation; recovery does not retune it.

## A-MemGuard

The former urllib client hardcoded 120 seconds, ignored configured judge timeout and bypassed the requests API journal. The restart uses --full-judge-timeout 600 and AMEMGUARD_TRANSPORT_MAX_ATTEMPTS=3: three total attempts with one/two-second backoff. Only transient network errors, HTTP 429 and 5xx retry. Other HTTP errors, invalid outputs and truncation remain terminal. Model, prompts, sampling, token budgets and embedding timeout are unchanged.

Each HTTP attempt is recorded without credentials, prompts or response content. Logical call counts differ from HTTP attempts; timed-out usage can be unknown. Identical request bytes do not imply identical stochastic generations. These are disclosed transport choices, not recovered official numeric parameters.

The failed 12-task stream has no transactional task checkpoint, so its records remain diagnostics and the replacement starts fresh 200.

## Isolation

Existing live runs and model servers are not restarted. Completed results and older failures with an active replacement are preserved. Only merged main is published, and worker source is frozen before launch.
