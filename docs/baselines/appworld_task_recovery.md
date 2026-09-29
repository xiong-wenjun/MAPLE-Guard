# AppWorld task recovery and revision comparisons

## Recovery is stateful

A trace line is an output record, not a complete task checkpoint. AppWorld currently carries private/shared memories, MemRL query-group assignments and cached embeddings/Q-values, baseline provenance labels, poison bookkeeping, and agent reputation between tasks.

The failed Qwen ACL run in campaign-r3 completed 75 tasks. It persisted SQLite memories and the ACL ledger, but did not save the original MemRL query-group mapping and embedding cache or a complete task-boundary snapshot. Its API journal contains usage metadata, not replayable responses. Re-embedding saved text or appending task 76 with fresh caches would be a different state trajectory. Preserve those 75 records for diagnostics; do not describe a fresh run as an exact continuation.

Task checkpoints are opt-in and initially restricted to provenance_acl. Normal runs of other methods are unchanged. A checkpoint is committed only after a complete task and trace flush. It must preserve both durable stores and the in-memory retrieval state; restoring must validate source, configuration, inputs, and checkpoint hashes before changing working state. Failed partial-task state must be retained separately and rolled back to the committed boundary.

The ACL restart retains the original generation budgets, seed, topology, scorer-free rules and task order. Saving state does not permit accepting truncated responses. Recovery tests and fixtures are not benchmark results.

## Comparison inventory

There are three new comparison families: provenance/taint + ACL, A-MemGuard, and PIGuard retrieval. Two additional placement controls are MAPLE retrieval-only versus full lifecycle and PIGuard lifecycle versus retrieval. PIGuard lifecycle is a combination constructed for this study.

AgentSafe, native INFA-Guard and AgentXposed are repairs of existing comparisons, not three newly introduced methods. AgentSafe must remain labeled a full-component adaptation because the published release does not specify the complete original criteria and threshold configuration. GUARDIAN and Inspector additionally need distinct identities; unverifiable historical results cannot be relabeled by assumption.

The current plan contains 11 main-table candidates, three mechanism configurations and one Inspector identity-audit configuration. This is 15 configurations, not 15 new baselines.

## Keep, replace, and extend

- Retain successful complete strict runs with matching protocol, full traces, summaries, and valid API responses. Gemma IT ACL star/42 completed 200 tasks with exit zero and no recorded invalid responses.
- Continue healthy strict runs. Adding seeds or topologies does not invalidate an existing matching star/42 cell.
- Replace incomplete/adapted historical AgentSafe, native INFA and AgentXposed comparisons with their corrected, explicitly documented versions.
- Keep failed feedback-routing runs, swallowed-detector-error runs, old stopped ACL runs and Gemma base checks as diagnostic evidence only.
- The failed Qwen ACL run needs a fresh checkpoint-enabled restart; AgentXposed Kick completed zero tasks and can restart with fresh memory under the same frozen protocol.
- Old paper values require a trace/config/source match before reuse in the revised table. Healthy execution alone does not establish scientific comparability.

The strict host protocol still includes reference-outcome feedback. No-feedback/oracle-removal experiments, false-positive/cost analysis, cross-attack evaluation and topology/seed statistics are additional evidence requirements, not new baseline names. A future change to feedback permissions or the graph definition requires new affected cells. The existing star implementation includes a leaf ring and must not be silently described as a pure star.

## Matrix status distinction

The 15-seed by four-topology queue has started. At 17:02 UTC+8 on 2026-09-29 it had three running expanded No Defense/chain cells, 882 queued, 59 held after failure, 360 blocked and 16 reserved legacy star/42 cells, totaling 1,320 planned main-table cells. No expanded cell was complete at that observation.

The repaired AgentSafe, INFA and GUARDIAN pipelines are separate from the old queue's blocked assets. Starting their star/42 evaluations does not automatically dispatch all 360 corresponding expanded cells. Report prepared, queued, running, blocked and completed counts separately.
