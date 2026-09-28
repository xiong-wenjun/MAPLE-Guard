# Full baseline integration implementation plan

> AI workers: use subagent-driven-development for isolated modules; test behavior before implementation; review specification first, then code quality.

Goal: provide opt-in agentsafe_full, infa_guard_full, agentxposed_full_guide and agentxposed_full_kick with all declared paper components, preserving legacy methods.
Architecture: independent defense components consume text, operational identities and model callbacks only. A task-scoped shared runtime connects memory admission/retrieval, pre-summary routing, real agent regeneration and persistent agent replacement. No evaluator labels enter components. Persist AgentSafe metadata/state; reset communication state at task boundaries.
Stack: Python 3.10, unittest; existing OpenAI-compatible clients; native INFA MyGAT loaded from an explicitly configured upstream checkout and strict raw checkpoint.

Source audit complete:
- AgentSafe cc253ad48532fa6614a27557587086cfb87968ed: hierarchy/relationships/junk in code, full criteria/identity library not released. Criteria and threshold required; document adaptation.
- INFA 80b1156cb22d576d9149046c36c540c728a43dde: dual binary heads and native temporal GAT; no published pretrained checkpoint. Preserve all detector/postprocessing/remediation components; document full-protocol departures from release quirks.
- AgentXposed anonymous AgentXposed-F814: v2 baseline deviation and adaptive questioning; source final score accumulation bug. Disclose selected protocol.

Files/ownership:
- evaluate/defense_methods/agentsafe_full.py, tests/test_agentsafe_full.py, docs/baselines/agentsafe_full.md: AgentSafe worker.
- evaluate/defense_methods/agentxposed_full.py, tests/test_agentxposed_full.py, docs/baselines/agentxposed_full.md: AgentXposed worker, after AgentSafe implementation.
- evaluate/defense_methods/infa_full.py, tests/test_infa_full.py, docs/baselines/infa_full.md: INFA worker, after AgentXposed implementation.
- evaluate/defense_methods/full_runtime.py, tests/test_full_runtime.py: controller.
- maple_guard/maple_guard_core.py, run_mmlu.py, run_longmemeval.py, run_appworld.py, infa_memlink_eval.py, memory_backend.py: controller integration.
- tools/baseline_preflight.py, docs/baselines/README.md, README.md: controller readiness, reproducibility and running instructions.

- [x] AgentSafe behavior tests: asymmetric permission, cumulative levels, all cosine criteria strictly above threshold, identity mismatch, junk review and reload, no evaluation label access. Run python3 -m unittest discover -s tests -p test_agentsafe_full.py -v; capture red then green.
- [x] AgentXposed tests: baseline establishment, temporal deviation, multiple actual progressive replies, guidance changes a regenerated answer, kick suppresses later rounds, malformed detector fails. Run corresponding unittest file; capture red then green.
- [x] INFA tests: graph direction/features, dual labels, EMA/refinement, donor replacement persists, infected correction applied to context, missing/incompatible checkpoint explicitly fails. Run corresponding unittest file; capture red then green.
- [x] Shared runtime tests: nested task session cleanup, experiment isolation, no oracle fields passed, real callback retains target context, reset context persists, kicked agents do not generate, AgentSafe routing before receiver summary, reload metadata.
- [x] Integrate all benchmark paths. Existing methods unchanged. New methods fail fast if a required session/config/artifact is missing; no allow-all fallback.
- [x] Add acceptance tests using controlled model responses through actual public runners; verify old-method behavior still runs and full components are called. Do not mistake deterministic fixtures for benchmark performance.
- [x] Provide explicit policy/criteria schema, all CLI flags, native source/checkpoint requirements, source fingerprints and fidelity matrix. Native INFA weights absent is a readiness blocker, not reason to substitute a different detector.
- [x] Run full unittest suite, compile checks, --help for all entrypoints, preflight and git diff --check. Independent review in two passes. Address findings and rerun affected tests.
- [x] Commit validated source changes on feature branch; keep original checkout unchanged and report server location, tested coverage, and any live-model validation still unavailable.

Validation examples:
```python
self.assertNotIn(secret_id, [m.memory_id for m in retrieve_for_agent(...)])
self.assertNotIn(kicked_agent, next_round_outputs)
self.assertEqual(model_input.shape[-1], 384)
with self.assertRaises(FileNotFoundError):
    create_native_detector(missing_checkpoint)
```
The exact executable cases live in the owned test files; tests must assert observed routing/generation/store effects rather than only mock call counts.


Verification recorded 2026-09-29: 64 tests passed without skips in the isolated CPU environment, including real native-source MyGAT fixture loading/dual-head forward and both preserved user checkpoints. The fixture is software-only; no benchmark claims were produced. Existing bundled weights were verified as GSafeguardGAT single-head checkpoints; this supersedes the earlier unverified description of them as three-class. Native research checkpoint compatibility remains a clearly reported artifact requirement. Shared runtime review findings (oracle vote exclusions, revoked memories, state persistence, donor context/memory source, route rewrites, metadata aliasing, seed source identity, parser fallback, and method sidecar collision) were fixed and tested.
