# AppWorld fixed-subset pilot

This launcher evaluates **AppWorld-derived action selection**, not native AppWorld
completion. The 200-task input contains specs, not executable databases. Keep raw
datasets and credentials outside this repository.

## Matrix

Main-table candidates (11): No Defense, MAPLE, provenance/ACL, AgentSafe full,
INFA full, AgentXposed Guide, A-MemGuard, PIGuard retrieval, Challenger,
G-Safeguard, GUARDIAN.

Mechanism arms first on AppWorld/Qwen: MAPLE retrieval-only, PIGuard lifecycle,
AgentXposed Kick. Inspector is a separate identity-audit arm because the paper's
GUARDIAN description/citation conflicts with its graph implementation. See
baseline_identity_audit.md. Do not select between Guide/Kick from pilot scores.

Main-table methods should eventually share the same five frozen benchmark
subsets and both task backbones; one AppWorld/star/seed42 run is only the first
paired pilot, not a replacement for the original topology/repeat matrix.
Single-benchmark mechanism results support only that benchmark; extend key
lifecycle and feedback tests to LongMemEval before a cross-domain claim.

## Profiles and fidelity

- paper-code preserves current YAML logic for MAPLE/No Defense. It is a bridge
  diagnosis of the current code, not proof of reproducing published numbers.
- strict shares operational metadata, memory rendering, peer communication,
  all-active-agent voting and accepted-retrieved-private promotion. It uses
  brokered-shared memory so the broker can actually intervene. MAPLE top-k stays3
  for this AppWorld experiment. Reference-outcome feedback still exists.
- The strict placement pair shares MAPLE scoring and read filtering; only gate
  placement differs. PIGuard is another detector-placement pair.
- Qwen3.5 is the fixed guard/judge even for Gemma task generation. Native GNN
  encoders retain their required MiniLM model; MAS memory embeddings use the
  shared Qwen embedding service.
- Strict calls disable thinking consistently for agents and memory summaries.
  In the actual Qwen service, the original128-token/default-thinking probe
  truncated without final content. Such responses must not count as valid
  main-table results. The journal records finish reasons, usage and latency;
  strict truncation or missing final content fails the run.
- Bundled specs are used as provided, without importing ground-truth difficulty
  labels. This differs from the original native-directory adapter's question
  rendering. Dataset SHA and selected IDs are saved. The source manifest DB
  version0.1.0 and available native data0.2.0 must not be conflated.

## Run

Use a separate environment containing the actual MemOS backend; dependency
versions from the historical paper runs were not published. Runtime compatibility
must be checked before scientific interpretation.

```bash
/mnt/public/data/wj/venvs/maple-experiments/bin/python -B tools/run_appworld_matrix.py \
  --bundle /mnt/public/data/wj/maple-benchmark-inputs/20260929/appworld_200.json \
  --services /mnt/public/data/wj/maple-benchmark-inputs/20260929/service-credentials.json \
  --task-service inference1 --judge-service inference1 \
  --profile strict --phase pilot --tasks 200 --seeds 42 --workers 2 \
  --minilm-model /mnt/public/data/wj/baseline-references/MiniLM-L6-v2-hf-1110a243 \
  --run-root /mnt/public/data/wj/maple-strict-results/UNIQUE_RUN_NAME
```

The command prepares a reviewable matrix. Add --execute for actual model calls.
An existing run root is rejected; never reuse memory across arms. Smoke phase
changes warmup to0 and poisoning rate to0.25 for integration coverage, and must
not be reported as benchmark performance.

The service credential JSON must have mode0600, with keyed objects containing
base_url, model, api_key. It stays outside Git; keys never appear in commands or
run manifests. Prepared jobs are not declared asset-ready: actual strict detector
loading still must succeed. Missing AgentSafe settings/native INFA weights/
GUARDIAN assets appear as blocked rows rather than allow-all results.

Artifacts: matrix.json, source fingerprint/commit, YAML snapshot, per-run
resolved-args.json, run.json, api-calls.jsonl, trace.jsonl, summary, and isolated
memory/state directories. Completed output with response warnings is explicitly
ineligible for main-table reporting.
