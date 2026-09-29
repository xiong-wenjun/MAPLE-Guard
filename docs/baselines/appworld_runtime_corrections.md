# AppWorld runtime corrections — 2026-09-29

The 085c318 AppWorld pilot is incomplete and must not be used as a completed main-table result. Passing a two-task smoke test did not exercise the cross-agent feedback path encountered in the full 200-task stream.

## Feedback routing

The third task in MAPLE and No Defense failed while updating a peer-derived private memory. The memory was physically stored in the recipient's cube (agent 0), but origin_agent correctly recorded its author (agent 7). MemoryBackendBundle incorrectly used that author as the feedback destination.

The backend now attaches a physical store identifier when writing or reconstructing a candidate. Feedback uses this identifier; provenance metadata is unchanged. Strict execution rejects unknown stores or unregistered memories rather than silently dropping the update.

A regression reproduced the wrong-store failure before the fix. Validation includes recipient/author separation, reconstruction after clearing the entry cache, shared storage, and rejection of unregistered feedback. A real MemoryOS/Qdrant check wrote an agent-1-authored memory into agent 0's cube, cleared the entry cache, and persisted a negative-feedback update from 0.6 to 0.44 in the recipient cube.

## Judge transport

The shared Challenger/Inspector client did not inherit disabled thinking. Challenger exhausted its official 50-token answer budget in reasoning without a final answer. Inspector timed out; that failed response alone cannot establish its internal generation state. The client now inherits the configured switch while preserving original prompts and token budgets. Small real Qwen checks completed with 2 and 21 answer tokens, respectively; these are transport checks, not benchmark scores.

AgentXposed's official helper catches ordinary provider exceptions. Old Kick traces therefore continued after judge timeouts. Its urllib judge calls also bypassed the requests API journal, so zero journal errors did not prove zero judge errors. The affected old run was stopped and marked ineligible. The full-method judge now uses the instrumented requests transport and raises a fatal strict-run error on provider failure, truncation or missing final content. A regression invokes the released detector itself and verifies that its ordinary exception handler cannot swallow the failure. The old Guide and Kick two-task smokes had 14 and 11 swallowed detector errors, respectively; both were explicitly marked ineligible despite their old completed status.

## Rerun policy

Keep failed artifacts. Use a new code snapshot, output directory, memory database and baseline state for recovery runs. Do not resume a partially mutated memory stream at the failed task or combine rows from different versions. The old ACL run can continue as a diagnostic run, but a new paired campaign should use the same corrected code snapshot for every method.

The protocol remains AppWorld-derived action selection, not native AppWorld execution. A single star/seed42 run is still a pilot, not the original multi-topology repeated experiment.

Validation after all corrections: 243 tests passed. Logs: /mnt/public/data/wj/maple-strict-results/appworld-20260929-feedback-fix-tests.log.
