# Strict baseline comparisons implementation plan

> AI workers: use subagent-driven-development for isolated tasks, then specification and quality review. Implementation remains on the server worktree; no local clones.

Goal: implement operational provenance/taint + ACL, a matched MAPLE retrieval-only arm, faithful A-MemGuard integration, and official PIGuard detector deployment pairs; recheck the existing full AgentSafe/AgentXposed/INFA implementations and artifact readiness.
Architecture: independent components use runtime observations only. The controller owns method dispatch, memory serialization and public runner wiring. External baselines have scoped state and strict configuration; failures do not silently turn into permission grants or substitute heuristic classifiers.
Stack: existing Python unittest runners, OpenAI-compatible model APIs, pinned official upstream source and model revisions.

Files and responsibilities:
- evaluate/defense_methods/amemguard_full.py, tests/test_amemguard_full.py, docs/baselines/amemguard_full.md: A-MemGuard implementation worker; official-source reasoning-path consensus, source reasoning-chain lessons/retrieval/preventive prompts, isolated persistent state.
- evaluate/defense_methods/piguard.py, tests/test_piguard.py, docs/baselines/piguard.md: PIGuard implementation worker after A-MemGuard; official checkpoint inference and bounded input handling, no keyword fallback.
- evaluate/defense_methods/rule_baseline.py, tests/test_rule_baseline.py: controller; explicit ownership/ACL, operational provenance union and taint propagation through writes, retrieval, promotion and transfers.
- maple_guard/maple_guard_core.py, memory_backend.py, run_mmlu.py, run_longmemeval.py, run_appworld.py, infa_memlink_eval.py, evaluate/defense_methods/full_runtime.py: controller; method registration, correct lifecycle placement, actual prompt/decision integration, schema persistence.
- tests/test_comparison_wiring.py: controller; behavior across public runners, clean feature view, phase selection, persistent lessons and metadata.
- tools/baseline_preflight.py, docs/baselines/README.md, configs/strict_baselines: controller; readiness, reproducible execution and fidelity matrix.

- [x] Inspect exact upstream commits and runtime artifacts; distinguish author code from paper reconstruction and unavailable artifacts.
- [x] Write failing rule tests: unauthorized reader denied, descendant taint persists, no implicit declassification, evaluator poison tags never establish taint, no MAPLE scoring call.
- [x] Implement rules and wire real lifecycle operations; verify red then green with python3 -m unittest discover -s tests -p test_rule_baseline.py -v.
- [x] Write failing matched-placement tests: retrieval-only shares MAPLE scoring/ranking, bypasses defense at write/promotion/transfer, still respects intrinsic storage constraints; unchanged metadata/prompt/feedback configuration.
- [x] Implement matched retrieval arm and protocol checks; run python3 -m unittest discover -s tests -p test_comparison_wiring.py -v.
- [x] Implement A-MemGuard from audited official code. Tests must distinguish current-query reasoning from historical answer-letter voting and exercise lesson use on a later task and restart.
- [x] Implement PIGuard from pinned official model. Tests verify malicious label mapping, long-text policy, strict errors, reusable detector and per-experiment counters.
- [x] Integrate both external methods into real runners; publish native-vs-adapted components explicitly.
- [x] Audit existing full baselines against official sources and run preflights; fix demonstrated defects only after failing regression tests.
- [x] Run deterministic integration tests and model-backed smoke runs when actual artifacts/services are available; do not label smoke checks as paper benchmark performance.
- [x] Review specification first then code quality; run full tests, compile/help/preflight checks and git diff --check; record exact source/model fingerprints.
- [x] Prepare verified code and runnable commands, explanation of custom controls and remaining artifact limitations for a feature-branch commit.

Acceptance tests are the named test_* files above plus test_strict_runner_paths.py.
The complete suite, pinned-source protocol differentials and official PIGuard
CPU parity are software/artifact validation, not benchmark performance.

Execution limitation: native INFA trained two-head checkpoint, configured datasets,
and reachable chat/embedding services are unavailable. No synthetic result will
be entered as an official baseline score. Full multi-seed benchmark execution
remains pending those inputs.
