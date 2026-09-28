# INFA-Guard native detector and complete remediation

`infa_guard_full` connects the native published dual-head detector to MAPLE's live agent contexts. It is a paper-component integration, with the released-control-flow differences disclosed below. It is separate from the legacy `infa_guard` adapter, which loads the repository's G-Safeguard-style model.

Reference checkout: [official INFA-Guard](https://github.com/yjzscode/INFA-Guard) commit `80b1156cb22d576d9149046c36c540c728a43dde`. The integration loads `train/models/defender/model.py` and `gat_with_attr_conv.py` from an explicitly configured external checkout; it records their SHA-256 hashes and the checkout commit. It neither vendors those source files nor rewrites them. Module aliases and a module-local import hook avoid replacing another project's top-level `train` package. This independent orchestration code does not reproduce upstream prompt text verbatim.

## Detector and feature contract

Native `MyGAT` is constructed with `in_channels=384`, `hidden_channels=1024`, `out_channels=2`, `heads=8`, `edge_dim=(3,384)`, `guard="ours"`, and the released default branches for 1, 2, 3, and 4–5 turns (last branch beyond that). It includes the shared custom edge-aware GAT, progressive branch layers, temporal base/residual/trend processing, self-reply features, input projection, and separate malicious/infected heads. Both logits receive **independent sigmoid**, not a three-class softmax.

The embedding model defaults to `sentence-transformers/all-MiniLM-L6-v2`; embeddings must be finite 384-dimensional vectors. Use the exact embedding model used for training the provided checkpoint, not merely an arbitrary model with the same dimension. History stays in raw pre-repair form for detection; corrected history is stored separately for rehabilitation.

Graph construction follows the released evaluation implementation: `edge_index=adjacency.nonzero()`, edge attributes use the destination agent's reply for every turn, and initial node features average first-turn edge attributes at destinations. Node self replies are `[nodes, turns, 384]`. The native model itself groups temporal edge statistics at sources. The training/evaluation source has direction differences; this adapter preserves the published **evaluation** convention rather than silently reversing it. Isolated nodes receive zero initial features and retain self-reply features.

The loader accepts either a raw state dictionary or a `model_state_dict` wrapper whose complete keys/shapes match native MyGAT. Signature checking is followed by `load_state_dict(strict=True)`, `eval()`, and `no_grad()`. No missing parameters are randomly filled; no different detector or allow-all fallback is selected after an error. Checkpoint SHA-256, source hashes/commit, architecture, embedding model, class ordering, thresholds, and selected branch are included in decision provenance.

## Existing checkpoint verification

The bundled `communication_gnn/longmemeval.pth` and `communication_gnn/appworld.pth` internally declare `model_class=GSafeguardGAT`, `out_channels=1`, and `method_scope=communication_only`. Their parameters are `layers.0.*`, `layers.1.*`, `out.weight=[1,1024]`, and `out.bias=[1]`. Actual strict loading and CPU forward under the declared model produce `[nodes,1]` finite outputs. They have no native `shared_convs`, `branch_convs`, `input_proj`, or `branch_heads_inf` parameters.

These existing weights are preserved and remain usable with the legacy detector. Their content does not supply the trained temporal/infection components required by native INFA, regardless of the checkpoint filename or its claimed origin. The new loader reports this incompatibility explicitly. Reproducing native INFA performance needs a compatible trained checkpoint; the inspected upstream source tree does not bundle one. This implementation does not claim that a synthetic software-test fixture is a trained research model.

## Control flow

For each task, reset raw history, repaired history, infection EMA, and isolation state. For each round:

1. Observe every active agent's response and run the native detector over cumulative raw dialogue.
2. Threshold malicious/infected probabilities (default 0.5). Maintain `EMA_t = 0.3*p_inf + 0.7*EMA_(t-1)`, initialized at zero, and its change.
3. Apply released topology refinement: local suspicious-neighbor consistency, distance-based pruning beyond two hops for changes below 0.05, and likely-source inference from neighboring malicious/infected probabilities. Preserve the release's outgoing-neighbor convention and deterministic tie behavior.
4. Select a donor outside both refined sets with the lowest malicious probability, only if that probability is at most the donor threshold (default 0.1).
5. Replace each detected attacker's output with the donor output and invoke the runner's replacement callback. This keeps the donor's operational system/task/tool context in future turns and redirects private-memory retrieval to the donor source. The donor is determined by predictions, not evaluator labels.
6. Correct each infected output using the public task, current output and previous repaired incoming-neighbor responses. The judge returns exactly `{"response": "..."}`. Update both the output and the actual target conversation through the repair callback.
7. If no eligible donor exists, explicitly isolate the detected attacker: suppress further generation, output and incident-edge communication for the remaining task. If all agents are isolated, future empty rounds avoid detector/judge calls and record that state.

All external correction responses are validated before callbacks mutate live agent context. Model/provider/schema failures stop the run. The observer never receives hidden system prompts or evaluator attacker/answer fields. The runner holds operational context and executes callbacks.

## Deliberate differences from released orchestration

- The released first round handles only attackers and has different donor eligibility. This integration applies the same disclosed dual-head refinement and donor cutoff on every round.
- Released replacement copies donor system/output but does not fully replace continuing private history. This integration replaces live operational context and memory source so an old attacker persona does not reappear next round. Agents share one model/tool implementation in MAPLE; executable heterogeneous tool cloning is not claimed.
- Released role-based sender filtering can keep corrected nodes muted. Here repaired/replaced nodes remain active; only explicit no-donor isolation removes participation.
- Released correction can fall back to textual redaction on failure. Here malformed or failed correction raises, making the failure visible.
- The release does not explicitly call `eval()` on the detector. This adapter uses deterministic inference mode.
- The no-donor policy is explicit isolation, not an unreported skipped repair.

These choices must be reported in an experiment description. Do not call this a bit-identical run of the released orchestration or mix its fresh results with old communication-only adapter results.

## API and dependencies

```python
guard = InfaGuardFull(args, judge)
guard.begin_task(task_id)
outputs, decisions = guard.defend(
    raw_outputs, respond, question, round_idx, adjacency,
    replace=runner.replace_agent, repair=runner.repair_agent,
)
```

The shared `respond` argument exists for API compatibility; INFA correction uses its judge and the repair callback. `round_idx` must be consecutive and zero-based. Output IDs must cover exactly the active agents. `begin_task` ends prior task isolation/replacement history.

Required configuration: `--infa-code-dir`, `--infa-checkpoint`, positive `--agents`; optional `--infa-device cpu`, `--infa-embedding-model`, `--infa-threshold`, and `--infa-donor-threshold`. Use a compatible installation of torch, torch-geometric, torch-scatter, einops and sentence-transformers. Source/version hashes are recorded rather than inferred from package names. The default endpoint/credentials for correction are the shared full-judge settings in [README](README.md).

Unit-test injection of a detector or embedder is explicit and marked `injected` in provenance. Production CLI construction does not use injection. The native test constructs a temporary synthetic state only to verify strict architecture loading, both output heads, graph tensors, and shape/error handling. It never exports that fixture for experiments.

```bash
python3 -m unittest discover -s tests -p test_infa_full.py -v
# Include the native source test in an environment with the model dependencies:
INFA_NATIVE_SOURCE=/PATH/TO/INFA-Guard OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p test_infa_full.py -v
```

Passing software tests does not establish detection accuracy, successful rehabilitation on real models, or reproduced paper numbers. The checkpoint, MiniLM assets, endpoints, and benchmark data must be configured before a model-backed evaluation.
