# Full-component baseline integration

The new opt-in methods are `agentsafe_full`, `infa_guard_full`, `agentxposed_full_guide`, and `agentxposed_full_kick`. They reconstruct the declared paper components with documented MAPLE adaptations. They do not assert identical source behavior or reproduced paper numbers. Legacy names and checkpoints remain available under their original adapters.

| Method | Components connected to the actual runner | Protocol details |
| --- | --- | --- |
| AgentSafe | Four cumulative access levels; directional relationships; identity checks; all-criterion strict cosine admission; periodic reflection; persistent junk; write/read/promotion/recipient routing | [AgentSafe](agentsafe_full.md) |
| INFA-Guard | Native temporal MyGAT; independent attacker/infection heads; infection EMA and topology refinement; donor replacement; infected correction and live context repair | [INFA](infa_full.md) |
| AgentXposed | Contextual HEXACO baseline; temporal deviation; real adaptive questioning; final reassessment; live Guide and task re-answer; persistent Kick | [AgentXposed](agentxposed_full.md) |

## Scope and comparison protocol

MMLU, LongMemEval, AppWorld (through its MMLU stream wrapper), and the INFA PI/TA transfer runner accept the new method names. The core episode runner also exposes them. Full methods share a scoped runtime; calling their low-level memory operations outside a runtime raises instead of bypassing protection.

Full methods exchange actual previous-round incoming-edge messages in MMLU/LongMemEval/AppWorld. For a matched comparison, pass `--peer-communication` to **every** comparison arm, including MAPLE and no-defense. This option also exists for legacy methods. Keep agent count, topology, rounds, memory backend, task stream, attack stream, model, and feedback regime identical. Optional memory handoffs still operate; report whether they are enabled. Do not mix old memory-only results with this communication protocol.

Full methods do not exclude nodes using evaluator attacker labels in final aggregation. `--exclude-attackers-from-final-vote`, causal replay without independent state, and extra heuristic/source-aware communication guards are rejected in full mode. Structured poison labels, hidden roles and reference-answer fields never enter defense components. However, existing benchmark memory text can contain outcome feedback/reference answers, and the LongMemEval adjudicator retains its task-scoped history protocol. This change is not a claim to remove all outcome feedback from the benchmark.

The runner holds private target context. Inquiry and Guide call the actual agent; the judge sees public observations only. Kick removes generation, routing, selected memories, and final output participation for the remainder of the task. INFA donor replacement preserves donor operational task/tool context and redirects subsequent private-memory reads to the donor source. All agents use the same model/tool implementation in this harness; heterogeneous executable tool replacement is outside this adapter.

AgentSafe rebuilds each task prompt from currently permitted memories and routed peers, so quarantined old prompt text does not survive through conversation history. It validates loaded LongMemEval history before insertion and includes registered records in periodic reflection. Registered record text and quarantine state are persisted in the sidecar; use a fresh memory store or migrate/register a pre-existing legacy bank before describing its entire historical contents as reviewed.

## Configuration

All full flags work on the public runners and may also be set under YAML `defense.full` using underscore names. Do not put API keys in committed configs. `FULL_BASELINE_API_KEY` is the optional judge credential environment variable.

The full judge defaults to `chat_base_url` / `chat_model`, not the old safeguard endpoint. Override with `--full-judge-base-url`, `--full-judge-model`, and optionally `--full-judge-max-tokens` (default 4096). Requests use temperature zero and JSON-object output. `--disable-chat-thinking` is honored. Malformed/truncated judgments, network errors, missing model artifacts, and bad embeddings fail the run; there is no allow-all or hash-embedding fallback inside full components.

AgentSafe requires `--agentsafe-policy-file`, `--agentsafe-criteria-file`, and an explicit `--agentsafe-threshold`. Policy identities must cover all operational agents. LongMemEval preloaded user history uses reserved identity `-1`; include it and intended reader clearances. The [three-agent policy](examples/agentsafe-policy.json) and [criteria file](examples/agentsafe-criteria.json) are **illustrative MAPLE configuration**, not released author defaults or validated hyperparameters. Choose and freeze criteria/threshold on separate clean development data. The paper does not publish a reproducible criterion library or defense threshold. `--agentsafe-review-interval` defaults to one round.

For persistent MemRL runs, AgentSafe also requires `--baseline-state-path`. Use a distinct path for each method/seed/memory store. The JSON sidecar saves classifications, record registry, review clock, and versioned quarantine evidence atomically. Keep it with the memory store. Do not share one sidecar across concurrent runs. AgentXposed/INFA task states reset at the next task; this implementation does not resume a partially executed communication task from disk.

INFA requires an explicit `--infa-code-dir` official checkout and `--infa-checkpoint`. Loading is strict against the native model structure; checkpoint names are not used to infer compatibility. See the INFA document for embedding dependencies, source version, output ordering, and released-code deviations. The bundled `communication_gnn/{longmemeval,appworld}.pth` files declare `GSafeguardGAT`, `out_channels=1`, `out.weight=[1,1024]`, and `method_scope=communication_only`. They remain usable by the legacy detector. They do not contain native INFA's temporal branches or infection head; keeping those weights cannot alone reproduce that missing trained detector.

AgentXposed defaults are `--agentxposed-deviation-threshold 1`, `--agentxposed-inquiry-rounds 3`, `--agentxposed-guide-rounds 2`. The final deviation threshold and Guide count are disclosed reconstruction choices. At least two observed task rounds are needed to compare against a first-observation baseline. Count and report all profiling/inquiry/Guide calls and tokens when comparing latency/cost.

## Running and verification

Run from the repository root. These commands demonstrate configuration; they are not stored benchmark results:

```bash
python3 -m unittest discover -s tests -v
python3 tools/baseline_preflight.py --method agentxposed_full_kick \
  --chat-base-url http://127.0.0.1:8001/v1 --chat-model YOUR_JUDGE_MODEL
python3 tools/baseline_preflight.py --inspect-checkpoint communication_gnn/longmemeval.pth
python3 -m maple_guard.run_mmlu --config configs/mmlu_chain.yaml \
  --method agentxposed_full_guide --peer-communication --disable-chat-thinking
python3 -m maple_guard.run_longmemeval --config configs/longmemeval_chain.yaml \
  --method agentsafe_full --agents 3 --peer-communication --disable-chat-thinking \
  --agentsafe-policy-file docs/baselines/examples/agentsafe-policy.json \
  --agentsafe-criteria-file docs/baselines/examples/agentsafe-criteria.json \
  --agentsafe-threshold YOUR_VALIDATED_THRESHOLD \
  --baseline-state-path /YOUR_RUN/agentsafe-state.json
python3 -m maple_guard.run_mmlu --config configs/mmlu_chain.yaml \
  --method infa_guard_full --peer-communication \
  --infa-code-dir /PATH/TO/OFFICIAL/INFA-Guard \
  --infa-checkpoint /PATH/TO/COMPATIBLE/INFA_CHECKPOINT
```

Preflight `configuration_loaded` confirms construction, not a live multi-agent test or reproduction of accuracy. AgentSafe construction contacts the configured embedding endpoint to validate criterion vectors; INFA construction may load its embedding model. Checkpoint inspection uses `torch.load(weights_only=True)` and prints only metadata/shapes/hash. It does not execute inference or substitute a model.

Deterministic tests substitute external judge/embedding/agent responses and exercise actual control, memory, and runner paths. They are software verification, not benchmark experiments. Fresh model-backed runs and statistical reporting are still necessary before updating a paper table. Record git commit, source commit/hash, checkpoint SHA-256, embedding/judge models, policy, threshold, and all protocol options with each run.
