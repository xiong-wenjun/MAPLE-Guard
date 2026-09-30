# INFA-Guard native detector and released protocol

`infa_guard_full` defaults to `infa_protocol=released`. It follows the first-turn and subsequent-turn control flow of [INFA-Guard at commit 80b1156cb22d576d9149046c36c540c728a43dde](https://github.com/yjzscode/INFA-Guard/tree/80b1156cb22d576d9149046c36c540c728a43dde). The previous paper reconstruction remains available only as `infa_protocol=reconstruction`. These profiles are separate from the legacy `infa_guard` communication-only adapter. None of them establishes reproduction of paper accuracy without trained weights and fresh experiments.

The external native model is loaded from `train/models/defender/model.py` and `gat_with_attr_conv.py` without source rewriting or replacement of another project's top-level `train` package. Runtime provenance records the actual source commit, source hashes, checkpoint hash, protocol and detector mode. The orchestration is integrated into the MAPLE runner; it does not claim bitwise equivalence to the entire upstream experiment harness.

## Released control flow and source fidelity

Line references below refer to `evaluate/defense_methods/ours_defense.py` in the pinned source unless stated otherwise.

| Behavior | Released profile | Source |
| --- | --- | --- |
| Initial detection | Raw malicious/infection thresholds are traced; only attackers are acted upon. No infection refinement, correction or EMA update. | Lines 314–355 |
| Initial donor | Lowest malicious probability among non-attackers, including predicted infected nodes; no donor cutoff. | Lines 331–337 |
| Later detection | Float32 infection EMA with alpha 0.3, trend threshold 0.05, outgoing-neighbor refinement, distance cutoff 2, likely-source inference. | Lines 187–245 |
| Later donor | Excludes both refined sets, uses minimum malicious probability, rejects values above donor threshold (default 0.1). | Lines 247–256 |
| No donor | Retains the original attacker output and flags its sender. Does not stop generation or delete detector edges. | Lines 257–272 and 347–349 |
| Takeover | Host callback resets the target to donor system messages plus the replacement assistant response. No donor private-memory reassignment. | Lines 113–119 |
| Infected correction | Source security-filter prompt, raw rewritten-text response, preceding repaired incoming-neighbor history. Updates public output while preserving accumulated private conversation. | Lines 79–103 and 273–303; `MAS/agents.py:84–95` |
| Correction failure | Source case-sensitive replacement of six substrings and their capitalized forms with `[REDACTED]`; trace records `correction_fallback=true`. | Lines 104–110 |
| Sender filtering | Attackers and corrected infected nodes remain eligible to generate, but their current flagged roles suppress outgoing messages until cleared by a later detection. | Lines 267, 271, 303, 305; `MAS/agents.py:538–554` |
| Detector mode | The profile leaves the native model in training mode, as the released loader does; dropout can remain active. | `evaluate/main_defense_repair_test.py:413–421` |

`begin_task` resets raw/repaired history, EMA, inactive nodes and blocked senders. The first turn does not advance EMA. Every subsequent turn observes all agents, including flagged senders. The detector always receives cumulative raw responses and the original graph; corrected history is used only by the correction prompt. The flagged sender set is replaced each round, so a later normal classification permits outgoing communication again.

Correction uses the published prompt and raw text response contract; the host must not force JSON response format for the released profile. The literal fallback preserves the source's limited case handling, including leaving all-uppercase variants untouched. An empty successful text response is not converted into a fabricated answer. Detector errors, invalid probability shapes, and incompatible weights still fail explicitly; the source redaction fallback applies only to correction failures.

## Runtime integration contract

```python
guard = InfaGuardFull(args, judge)
guard.begin_task(task_id)
outputs, decisions = guard.defend(
    raw_outputs, respond, question, round_idx, adjacency,
    replace=runner.replace_agent, repair=runner.repair_agent,
)
```

- `guard.protocol` is `released` or `reconstruction`.
- `guard.blocked_senders` is distinct from `guard.inactive`. In released mode the runner must skip blocked outgoing messages, including memory handoff summarization, while continuing generation and retaining detector adjacency. It must not remove these agents from final output participation merely because their sender role is flagged.
- `guard.takeover_context` is `donor_system_only` in released mode and `donor_operational_context` in reconstruction mode. The replacement callback must honor it. The released reset comprises donor system context and one replacement assistant message; it must not redirect private-memory ownership.
- The released profile publishes infected corrections through returned outputs and does not call `repair`, matching the source parser's preservation of accumulated conversation. Reconstruction invokes `repair` to replace live operational context.
- The shared `respond` argument is unused. The guard never inspects evaluator labels or private system prompts; the runner alone handles the replacement callback.
- Rounds must be consecutive and zero-based; output IDs must cover exactly the active agents. Each trace records protocol, raw/refined sets, EMA update status, sender blocking, donor decisions and correction fallback when applicable.

## Detector, features and reproducibility

Native `MyGAT` uses `in_channels=384`, `hidden_channels=1024`, `out_channels=2`, `heads=8`, `edge_dim=(3,384)`, `guard="ours"`, with released default branches for 1, 2, 3 and 4–5 turns (last branch beyond that). Temporal base/residual/trend features, self replies, shared/branch convolutions and both independent heads remain native. Each output logit receives sigmoid independently.

Embeddings default to `sentence-transformers/all-MiniLM-L6-v2` and must have 384 dimensions. Use the checkpoint's training embedding model. The graph follows the released evaluation convention: `edge_index=adjacency.nonzero()`, destination-reply edge attributes, and destination-grouped first-turn averages for initial node features. Node self replies are `[nodes, turns, 384]`. The native model aggregates some temporal statistics at sources; this evaluation/training direction distinction is preserved.

Checkpoint loading accepts a raw state dictionary or a `model_state_dict` wrapper, validates native keys/shapes, and calls `load_state_dict(strict=True)`. It never fills missing heads or falls back to a different detector. Model calls run under `no_grad()`.

`infa_detector_mode=profile` resolves to `train` for released and `eval` for reconstruction. Explicit `train` and `eval` overrides are also supported. Selecting `eval` disables dropout and is a reproducibility choice that differs from the released loader. Provenance records the resolved `detector_mode`, `detector_mode_overrides_release` and `bitwise_release_parity_claimed=false`. Training-mode inference requires reporting the random seed, call ordering, device and package versions; matching the control flow alone cannot ensure identical outputs.

Remaining host adaptations include task/agent prompts, response parsing and serialization, the selected correction model/endpoint and generation limits, MAPLE persistent-memory retrieval, and deterministic ascending-ID traversal for tied neighbor probabilities. Explicit threshold overrides also change published defaults. The source correction prompt requires `<REASON/UPDATED REASON>` and `<ANSWER>` formatting even when the host task uses a different format. These choices must be reported, not described as an unchanged upstream experiment.

## Reconstruction profile

`infa_protocol=reconstruction` preserves the prior integration: both-head refinement and donor cutoff on every turn, infected-node exclusion from first-turn donation, no-donor isolation, repaired senders remaining active, transfer of donor operational context/private-memory source, strict JSON correction with no redaction fallback, and replacement of infected target conversation. Its detector defaults to evaluation mode. Do not combine results from the two profiles under one unlabeled INFA row.

## Existing checkpoints and readiness

The preserved `communication_gnn/longmemeval.pth` and `communication_gnn/appworld.pth` declare `GSafeguardGAT`, `out_channels=1`, `method_scope=communication_only`, and `out.weight=[1,1024]`. They contain neither native temporal branches nor infection heads. Both load and run under their declared legacy architecture; neither can supply native INFA's missing trained components. The full loader explicitly rejects them. A temporary random unit-test fixture is not a trained checkpoint and is never exported for experiments.

Required settings are `--infa-code-dir`, `--infa-checkpoint`, and positive `--agents`. Profile controls are `--infa-protocol released` and `--infa-detector-mode profile`; optional settings include `--infa-device cpu`, `--infa-embedding-model`, `--infa-threshold` and `--infa-donor-threshold`. Dependencies include NumPy, torch, torch-geometric, torch-scatter, einops and sentence-transformers. Correction uses the full-judge endpoint/model settings described in [README](README.md).

On the audited server, no compatible trained native checkpoint was found in the relevant MAPLE/official-INFA trees. A real run also requires MiniLM assets, reachable configured services and benchmark data. Passing construction or software tests does not establish detection accuracy or paper reproduction.

## Native InjecAgent evaluator dependency

Native TA evaluation imports the released `utils.evaluation_utils` and
`utils.text_utils`; repetition detection uses `nltk.ngrams`. The pinned INFA
source's `requirements.txt` specifies **`nltk==3.9.2`**. Install that exact package
in the evaluation runtime without changing the other baseline dependencies:

```bash
python -m pip install --no-deps nltk==3.9.2
```

The local n-gram check requires no NLTK corpus download. Verify the real official
TA evaluator import and a `succ`/`unsucc` verdict before launching native bundle
runs; mocked unit-test verdicts do not prove environment readiness.

## Verification

Tests exercise released first-turn donation and untouched EMA, later cutoff behavior, float32 cutoff/EMA semantics, continued generation with blocked senders, task reset, text correction and source fallback. Explicit reconstruction tests preserve the previous behavior. Native fixture tests check strict two-head loading, train/eval choices, graph tensors and incompatible signatures without downloading assets.

```bash
INFA_NATIVE_SOURCE=/PATH/TO/INFA-Guard OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 -m unittest discover -s tests -p test_infa_full.py -v
```

## Explicit sanitizer transport bug and repair

The pinned release obtains a synchronous OpenAI client in ours_defense.py:68-74
(factory.py:167-186), then awaits its synchronous completion in
sanitize_with_llm at line93. Even a successful API response therefore triggers
TypeError and enters the literal redaction fallback.

The default --infa-correction-transport functional repairs this transport bug
and permits successful rewriting. This is an intentional exception to source
execution, not an unchanged release. Every decision and run provenance records
correction_transport and sanitizer_sync_await_fixed.

Use --infa-correction-transport released_sync_bug to reproduce the released
synchronous-client behavior: one completion call is attempted, its successful
answer is discarded, and the original response receives source redaction.
Provider errors follow the same fallback. Keep this flag in result labels and
never pool functional and unmodified sanitizer results. The paper reconstruction
profile retains its own strict JSON contract instead.
