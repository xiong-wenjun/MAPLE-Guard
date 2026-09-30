# Durable task checkpoints

The AppWorld, MMLU, LongMemEval and INFA-format CSQA/TA runners can commit a complete local state after each task. A crash restores the last published task boundary and repeats only the uncommitted task. It does not reconstruct memory from a trace or append results from incompatible attempts.

## Starting and resuming

Keep one method, seed and topology per output/checkpoint directory. Use an isolated, explicit MEMOS_BASE_PATH and the MemRL backend.

Append these options to the benchmark's normal command:

    --task-checkpoint-dir /absolute/run/task-checkpoints

On recovery, reuse that exact command, frozen source, services, assets and output paths, adding:

    --resume-task-checkpoint

tools/run_recovery_plan.py enables checkpoints on new attempts and automatically uses the same directory for bounded transport retries when a checkpoint exists. A new controller can also recover an original run's checkpoint. Existing lane and run locks prevent duplicate workers. The AppWorld matrix enables task checkpoints by default; --no-task-checkpoints is available for explicit diagnostics.

A failed initialization before the first checkpoint has no completed prefix. Historical runs without a complete checkpoint cannot be safely converted from traces alone. Existing processes and controllers executing an older frozen checkout do not acquire this feature when main changes; preserve healthy workers and use the new frozen source for subsequent launches. No timer or periodic automation is created by this feature.

## Saved state

- Ordered task IDs, completed records, poison schedule and target caches, trust, LongMemEval answer/haystack state, CSQA counters and local random generator.
- Private, shared and quarantine memory, entry aliases, backend IDs, SQLite/WAL content, MemOS registry, query/value caches and exact Qdrant RAM vectors, payloads and deletion masks.
- Comparison runtime caches, provenance ledger and references; AgentSafe HierarCache and review state; A-MemGuard catalog, lessons, pending prompts and counters; AgentXposed history/engine state; PIGuard counters; native INFA model parameters, mutable BatchNorm buffers and model mode.
- Official communication history and reconstructed immutable detector/module caches; Python, NumPy and PyTorch CPU/CUDA RNG state.
- Completed trace, baseline sidecar, API journal and CSQA progress, metrics, decision and memory journals.

Callable API clients, generation closures and immutable model implementations are reconstructed from pinned assets. AgentSafe criterion vectors are restored without repeating initialization embedding/canary calls. Task-scoped generation closures are recreated when the next task begins.

## Integrity and publication

Schema 2 uses typed JSON and explicit whitelisted state serializers, not executable pickle. Each generation is staged, fsynced and hashed, then published through an atomic latest.json replacement. Only the latest two committed generations are retained. Failed-attempt artifacts are archived separately; they are not silently merged with committed results.

Restore verifies source, configuration, dataset/prompt/model asset bytes, dependency versions, relevant environment, ordered task prefix and file hashes before replacing working files. Checkpoint files are private. Strict truncated/empty response checks remain enabled. An unknown state type or incompatible configuration fails closed.

These are local state guarantees. An external model service may remain nondeterministic when the interrupted task is repeated.

## Explicit token budget changes

Normal recovery rejects configuration drift. To resume with a changed task/judge token limit, update only the corresponding numeric token option and explicitly add:

    --checkpoint-allow-budget-change

The transition is recorded with the next task index, before/after values and protocol mixed_budget_resume in the checkpoint, recovery audit and final summary. Such a result has uniform_budget=false and must not be reported as a uniform-budget run. Changing the method, seed, topology, prompts, policies, thresholds, input bytes or source is not allowed by this flag.

Automatic transport retries never increase token budgets or disable strict response checks. Deterministic failures require diagnosis.

## Verification

The regression suite exercises interrupted-versus-uninterrupted task streams, completed-prefix reuse, SQLite WAL backup, Qdrant exact restoration, method state and tensor/RNG round trips, asset/config corruption rejection, atomic publication failure, two-generation retention, output rollback and recovery-controller directory reuse.

A remote canary also wrote real MemRL/Qdrant state using the configured embedding API, exited without cleanup after uncommitted writes, then restored the previous task boundary in a new process. Restoration itself made no embedding API requests; normal subsequent retrieval still uses the configured embedding service.
