# INFA resumable generation and train-to-test execution plan

> Execution: isolated remote worktree; main agent implements INFA, parallel scoped audit handles AgentSafe.

Goal: preserve every complete training reply and automatically run training and held-out tests after all 800 native dialogues validate.
Architecture: keep pinned upstream source untouched. A declared Qwen recovery profile wraps its async API, atomically journals accepted replies, replays by call ordinal and request hash, and expands only truncated requests through 1024/2048/4096/8192 tokens. A controller gates protocol, 20 generation grids, merge, embeddings, 50 epochs, checkpoint validation, two smoke runs, and two AppWorld-200 evaluations.
Stack: Python, OpenAI SDK, PyTorch, upstream INFA, remote inference1/inference2 and Gemma IT inference3.

- [x] Add regression tests in tests/test_infa_recovery.py for unchanged default profile, incomplete response rejection, bounded retries, atomic replay after restart, changed request rejection, concurrent completion order, and failed-stage test gating.
- [x] Implement tools/infa_generation_recovery.py with make_audited_create(original, audit_path, request_overrides, recovery_policy, cache_dir). Only complete nonempty stop responses can be returned or cached; corrupt caches fail closed.
- [x] Extend evaluate/defense_methods/reproduction.py and tools/official_reproduction.py with qwen_no_thinking_recover; original profiles retain their exact settings. Recovery changes are explicit recipe provenance.
- [x] Connect tools/run_infa_release_stage.py to durable journal, append-only retry logs and recovered complete output validation. Never overwrite upstream source or change infection labels.
- [x] Add tools/run_infa_training_pipeline.py: bounded generation workers, process locks, progress/exit records, validated stage reuse, then checkpoint strict load -> smoke Qwen/Gemma -> 200-task runs. A failed prerequisite cannot reach its downstream stage.
- [x] Run targeted tests with /mnt/public/data/wj/venvs/maple-experiments/bin/python -m unittest discover -s tests -p "test_*reproduction.py" -v and tests/test_infa_recovery.py, then repository tests.
- [ ] Merge reviewed changes into main using the configured user identity; push main only. Freeze that commit into a new server snapshot.
- [ ] Prepare a fresh recovery recipe/workspace, start background pipeline and verify actual API calls/cache records. Record that training is queued until 800/800 data, not already trained.

Validation: 322 repository unittest cases passed; native upstream interrupted/replayed fixture reproduced the uninterrupted graph, replies and labels. Real API launches and train/test results are recorded separately in pipeline state.
