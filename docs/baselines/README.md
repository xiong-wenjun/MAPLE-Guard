# Strict baselines and evaluation controls

Method names ending in `full` identify component coverage, not reproduced paper scores or identical experimental settings.

| Method | Implementation | Material boundary |
| --- | --- | --- |
| provenance_acl | Independent provenance propagation and ownership/ACL across lifecycle | Custom conservative rule control |
| maple_guard_retrieval_only | Same MAPLE score/read policy; non-retrieval gates bypassed | Compare with maple_guard --strict-comparison |
| agentsafe_full | Holder-local hierarchy, identity/criteria, cumulative generation history, permissions, reflection and junk | Paper components; unpublished criteria/threshold/prompts remain explicit configuration |
| infa_guard_full | Native MyGAT dual heads and released orchestration by default | Trained native checkpoint required; original single-output weights preserved |
| agentxposed_full_guide / _kick | Hash-pinned released source and explicit minimal bug repair | Original detection bug; MAPLE scheduling differs from offline replay |
| amemguard_full | Released EHRAgent reasoning paths, joint audit and lesson retrieval/warning | Official joint audit differs from paper Appendix pairwise description |
| piguard_retrieval / _lifecycle | Official pinned HF detector at different stages | Lifecycle deployment is our experiment; CPU parity passed, benchmark pending |

Details: [custom controls](strict_controls.md), [AgentSafe](agentsafe_full.md), [INFA](infa_full.md), [AgentXposed](agentxposed_full.md), [A-MemGuard](amemguard_full.md), [PIGuard](piguard.md).

## Comparison protocol

MMLU, LongMemEval, AppWorld through the stream wrapper, core episodes and INFA PI/TA transfer expose the new methods. Use `--strict-comparison --peer-communication` for every arm, including no-defense. Full communication methods exchange incoming-edge messages; old memory-only MAPLE results are not a matched comparison.

Freeze agents, rounds, topology, task/attack ordering, prompt, model, retrieval and feedback settings. Use fresh stores and a distinct `--baseline-state-path` per method/seed. Within-run tasks retain lessons/history; changed experiment configurations cannot silently reuse runtime state. Never share a sidecar across concurrent processes.

Oracle vote exclusions, extra heuristic/source-aware communication guards and causal replay without isolated state are rejected. Existing post-task reference-answer feedback remains a benchmark assumption. The launcher uses accepted-answer promotion instead of optional success-only promotion.

## Required configuration

Flags also work under YAML `defense.full` using underscore names. Full judge URL/model default to task chat settings. Credentials belong in environment variables. Prompt schemas and failure semantics follow the selected profile. Strict retrieval infrastructure never silently substitutes hash vectors or another retrieval strategy.

AgentSafe needs explicit policy, criterion library and threshold. Examples are illustrative MAPLE choices, not official defaults or validated hyperparameters. Include identity -1 for LongMemEval history. Component state v3 and runtime state v2 preserve holder-specific evidence. A private copy for another holder requires a new memory ID.

A-MemGuard needs `--amemguard-experiment-id`, real judge and real embeddings. Source prompts retain MIT attribution. The old pure_a_memguard_memrl is a heuristic proxy and must not be described as the full method.

INFA needs `--infa-code-dir` and `--infa-checkpoint`. Default `--infa-protocol released` preserves released first/later-turn behavior; reconstruction preserves the former paper approximation. Detector mode profile preserves the official loader's train/dropout behavior; eval is a recorded deviation. The functional sanitizer transport explicitly fixes upstream's synchronous-await bug; released_sync_bug reproduces its redaction fallback. Bundled communication_gnn weights declare single-output GSafeguardGAT without infection heads. Their bytes and legacy use are unchanged; native INFA never invents missing trained heads.

AgentXposed needs `--agentxposed-code-dir`. released_minimal_fix patches the absent final-score assignment; released_unmodified exposes the original no-detection behavior; reconstruction preserves the former live-inquiry approximation. Record source hashes, patch status and actual judge model.

PIGuard's verified server snapshot is `/mnt/public/data/wj/baseline-references/PIGuard-hf-dd78b24`. Fixed revision: dd78b24e330193a22d2293ac66922dd4f982f563; default CPU FP32, length512, threshold0.5. [Real parity evidence](evidence/piguard-cpu-parity.json) is not benchmark performance.

## Running

Use the isolated server environment:

```bash
cd /mnt/public/data/wj/MAPLE-Guard-baselines
PY=/mnt/public/data/wj/venvs/maple-baselines/bin/python
"$PY" -m unittest discover -s tests -v
"$PY" tools/baseline_preflight.py --method provenance_acl
"$PY" tools/baseline_preflight.py --method piguard_retrieval --piguard-smoke --piguard-model /mnt/public/data/wj/baseline-references/PIGuard-hf-dd78b24
"$PY" tools/baseline_preflight.py --inspect-checkpoint communication_gnn/longmemeval.pth
```

[Launch script](../../configs/strict_baselines/run_mmlu.sh) uses distinct method/seed stores and validates required configuration. Default dry-run prints commands; EXECUTE=1 performs real model calls. Use corresponding runner flags for other benchmarks.

Preflight and deterministic tests do not establish paper accuracy. Dataset/model services were unavailable during implementation; multi-seed benchmarks remain pending. PIGuard has real trained-model smoke/parity evidence. Native INFA fixture tests verify tensor wiring only. Record git/source/checkpoint hashes, model IDs, policy, thresholds, stream identity and environment with each experiment; do not relabel old table numbers as new results.
