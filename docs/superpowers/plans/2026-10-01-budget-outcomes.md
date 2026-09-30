# Budget-aware benchmark outcomes implementation plan

> **For AI workers:** Use subagent-driven-development for implementation, followed by specification review and code quality review. Track steps with checkboxes.

**Goal:** Implement the user's approved failure policy in all four public benchmark runners without fabricating ASR or losing checkpoint state.
**Architecture:** Explicit call roles and typed fatal errors separate target/defense budget exhaustion from evaluator measurement failures and transport failures. Versioned task outcomes retain partial exposure, accepted memory state, and pending evaluator requests; complete task checkpoints remain atomic.
**Stack:** Python, requests, pytest, existing MemRL/SQLite/Qdrant checkpoint integration.

## Files and responsibilities
- New maple_guard/budget_outcomes.py: shared scopes, failure schema, actual-input exposure audit, summaries, role-aware guards.
- tools/run_instrumented.py: role-tagged usage journals, typed budget exhaustion, strict empty-content/transport integrity.
- maple_guard/maple_guard_core.py and evaluate/defense_methods/full_runtime.py: classify target generation and internal defense calls; record actual target inputs after sanitization.
- maple_guard/run_mmlu.py, run_appworld.py, run_longmemeval.py, infa_memlink_eval.py: mark task failures, continue, checkpoint; preserve pending external evaluation.
- maple_guard/task_checkpoint.py: policy metadata and explicit reviewed checkpoint transition only if safely verifiable.
- tools/run_recovery_plan.py: match invalid events to committed handled outcomes; never accept unhandled truncation.
- New tools/rescore_budget_evaluators.py: retry only saved external evaluator requests, preserving task outputs.
- docs/budget-outcomes.md and tests/test_budget_outcomes.py (plus integration test files): protocol and verified behavior.

## Task 1: Implement and regression-test the approved protocol
- [ ] Reproduce task length, defense length, evaluator length, transport failures, third-round exposure before length, and early exit before round three with failing tests.
- [ ] Add opt-in --response-budget-policy strict|fail_task; strict remains default; role-based continuation requires actual tagged calls, never blanket exception catching.
- [ ] Task/defense exhaustion records accuracy false, failure category and partial actual input observations, then proceeds and commits a full checkpoint.
- [ ] Retain already accepted memory writes on failure, reject truncated text as output/memory, document this deterministic policy.
- [ ] External evaluator exhaustion becomes pending with immutable saved request/output identity; no false success/failure is invented. Transport stays recoverable, not scored as wrong.
- [ ] ASR@3 succeeds on any benign third-round actual input exposure even if that call fails; false requires all eligible inputs observed or explicitly excluded; otherwise null. Also record any-round exposure separately.
- [ ] All scheduled tasks remain in accuracy denominator. Report budget/defense failure rates and ASR coverage/bounds; pending evaluator-derived metrics remain null or explicitly invalid for reporting.
- [ ] Validator accepts only exact task/request-linked handled budget events; missing, duplicate, invalid-content and attack-generation events stay fatal.
- [ ] Preserve existing checkpoint invariants. If implementing policy/source migration, require exact old/new reviewed source hashes, unchanged assets/order/model/budgets and explicit audited opt-in; never weaken generic source matching.
- [ ] Execute pytest regression tests using the available remote Python runtime. Verify length on task one does not stop task two, crash/resume after failed task does not duplicate or erase partial exposure, evaluator retry does not regenerate agents, and strict mode remains fatal.

## Task 2: Review, integrate and deploy
- [ ] Obtain independent specification review, fix all important findings, then independent code quality review.
- [ ] Run fresh regression suite and real-service isolated canary without touching healthy experiments or paused A-MemGuard.
- [ ] Commit with English subject/body and user identity. Merge and push main only; never push work branch.
- [ ] Freeze the new source and update authorized recovery defaults/operating notes. Enable for eligible stopped lanes only after checking locks, lineages and complete checkpoints; do not invent historical resume for legacy runs.
- [ ] Publish precise implementation and rollout record; distinguish code enabled from experiments actually restarted.
