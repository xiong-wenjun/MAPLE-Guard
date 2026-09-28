# AgentSafe full component reproduction

`evaluate/defense_methods/agentsafe_full.py` is an opt-in, standalone implementation of the AgentSafe components for MAPLE adapters. It does not replace the existing `agentsafe` communication-filter baseline. The host owns the experiment session, memory storage, topology, persistence, and lifecycle hooks; this module owns AgentSafe policy decisions and junk memory.

## Source and fidelity

The inspected official repository is [junyuanM/Agentsafe](https://github.com/junyuanM/Agentsafe/tree/cc253ad48532fa6614a27557587086cfb87968ed), commit `cc253ad48532fa6614a27557587086cfb87968ed`. No license file or licensing statement was present in that checkout. This module uses independently written implementation and prompts; it does not vendor the released source or prompts.

[AgentSafe v2, Section 3.2.2 and Algorithm 1](https://arxiv.org/html/2503.04392v2) specify permission checks, identity validation, hierarchical memory, instruction-library validity detection, junk memory, and periodic reflection. Equations 8–9 require every criterion to pass a strict similarity threshold. Equations 10–12 use reflection with criteria and prior junk; the algorithm removes rejected records from active memory. The paper does not supply the criterion library, threshold, embedding model, or complete identity-matching implementation. Its sender/receiver permission notation is inconsistent. This adapter chooses explicit directional pair permissions and cosine similarity, and discloses those choices rather than claiming bit-identical reproduction.

The released [Agent class](https://github.com/junyuanM/Agentsafe/blob/cc253ad48532fa6614a27557587086cfb87968ed/Code/agents.py) has cumulative level reads, categorized writes, a junk field, and prompt-based review. Its review removes unreasonable lines without adding them to junk. The separate `Code/attack.py` prototype and `Code/process.py` do not provide the complete paper workflow. In particular, the `0.2` cosine cutoff in `process.py` measures attack success, not memory admission.

## Required configuration

Constructor:

```python
AgentSafeFull(args, judge, embed)
```

- `judge(messages: list[dict[str, str]]) -> str` is supplied by the host.
- `embed(text: str) -> list[float]` is supplied by the host; no network client or model loading exists in this module.
- `args.agentsafe_policy_file` must name a JSON object containing `identities` and an optional `relations` mapping.
- `args.agentsafe_criteria_file` must name a nonempty JSON list of nonblank verification-criterion strings.
- `args.agentsafe_threshold` must explicitly supply a finite cosine threshold in `[-1, 1]`. There is no default or silent fallback.
- `args.agentsafe_review_interval` defaults to `1` and must be a positive integer. The default follows Algorithm 1; less frequent review is an explicit experiment setting.

Example relationship policy:

```json
{
  "identities": {"0": "Agent 0", "1": "Agent 1", "2": "Agent 2"},
  "relations": {"0": {"1": 1, "2": 3}, "1": {"0": 4}},
  "default_level": 1,
  "self_level": 4
}
```

`relations[owner][reader]` authorizes the reader to access all levels up to and including that integer. Relationships are directional and independent of binary topology edges. Unknown pairs default to level 1, and self access defaults to level 4; these are declared MAPLE adaptations, not released experimental defaults. Unknown senders absent from `identities` fail identity verification. Level names are Stranger 1, Colleague 2, Friend 3, Family 4. The classifier uses a generalized privacy rubric for these levels, not the released social-demo prompt.

Criterion vectors are normalized once. Admission computes cosine against every configured criterion and accepts validity only when every score is strictly greater than the configured threshold. Empty criteria, nonfinite/zero vectors, dimension mismatches, and missing configuration cannot silently grant access. No extra similarity-to-junk cutoff is introduced.

## Public APIs

```python
admit(memory_id, text, owner, recipient, metadata) -> (bool, details)
read(memory_id, text, owner, recipient, metadata) -> (bool, details)
route(text, sender, recipient) -> (str | None, details)
review(records, round_idx) -> list[dict]
state_dict() -> dict
load_state_dict(state) -> None
```

`owner`, `sender`, and each actual recipient are operational integer agent IDs. `text` is the content to be inspected. `metadata` is a mutable dictionary belonging to the host memory record, typically `entry.baseline_metadata`. The module changes only `metadata["agentsafe_full"]`, containing `level`, `owner`, `fingerprint`, `config_fingerprint`, `quarantined`, and `reason`.

- `admit`: validate a private or shared write. `recipient=None` is allowed only for shared admission; eligibility is subsequently checked for each actual reader. Private admission checks the owner-to-recipient permission. Failed content, identity, or private permission goes to junk.
- `read`: validate content and require cumulative reader clearance. A denied shared reader does not quarantine an otherwise valid record for authorized readers. Reads require an actual recipient ID.
- `route`: return the original text or `None` for a particular edge. Invoke before delivering content to a recipient or its summarizer. A reader-specific denial cannot suppress an authorized destination. The host enforces whether the topology edge exists.
- `review`: accept records shaped as `{"memory_id": str, "text": str, "owner": int, "metadata": dict}`. The round index is zero-based; interval `N` runs when `(round_idx + 1) % N == 0`. Each active unique record receives reflection using the instruction library and accumulated junk. Rejected records are marked quarantined and copied to the junk dictionary. Already quarantined records are excluded. Returned traces contain `memory_id`, `action`, `reason`, and `details`.
- `state_dict` and `load_state_dict`: round-trip JSON-compatible junk and assessment caches. Snapshots must match policy, criteria, threshold, and normalized criterion vectors. The host must also persist record metadata. Snapshot quarantine remains effective even if a later caller supplies an empty metadata dictionary. A changed record text or operational owner forces fresh classification, identity checking, and criterion scoring.

Classifier, identity, and reflection responses must be exact single-field JSON objects: `{"level": 2}`, `{"valid": true}`, and `{"junk": false}` respectively. Wrong types, duplicate keys, extra fields, nonfinite JSON, malformed output, and provider failures raise `AgentSafeRuntimeError`. Invalid startup policy raises `AgentSafeConfigError`. Hosts must surface those failures, not reinterpret them as allow decisions.

## Host responsibilities

Keep one instance per experiment memory session. Apply `admit` to all private writes and team promotions, `read` before prompt assembly on both in-memory and backend retrieval paths, and `route` before raw handoff summarization. Call `review` at the configured lifecycle boundaries. Synchronize quarantine metadata with the host's active/inactive backend state, and persist both the module state and record metadata. Junk is checker context only and must never enter task prompts.

The interface intentionally accepts content and operational identities rather than complete MAPLE entries. It does not receive structured evaluator poison flags, answer fields, provenance scores, taint, or hidden attacker labels. Existing MAPLE consolidation can write reference answers into the memory text itself; this adapter does not remove that text. Experiments using those memories therefore retain the benchmark's outcome-feedback assumption and must not be described as fully free of reference-answer feedback. Memory IDs select state but are not sent to the judge. Storage scope and permissions must come from the host's actual routing decisions, not claims embedded in text.

## Validation

The standard-library tests in `tests/test_agentsafe_full.py` use deterministic judge and embedding callables and require no live model or new dependencies. They exercise asymmetric cumulative permissions, strict/all-criterion admission, malformed judgments and vectors, identity rejection, changed-content/owner revalidation, private admission, routing, periodic quarantine, JSON persistence, configuration validation, and independence from evaluator labels.

```sh
python3 -m unittest discover -s tests -p test_agentsafe_full.py -v
```

For LongMemEval, configure the reserved source `"-1"` as user-history and give its intended readers explicit clearances. Those initial history records are admitted and reviewed just like other memories; their benchmark trusted designation is not an AgentSafe bypass. With MemRL, `--baseline-state-path` is mandatory. The sidecar persists all registered records, classifications, versioned quarantine history, and reflection state; use a distinct sidecar per experimental seed/memory store. The host also checks quarantine before handoff summarization and rebuilds AgentSafe task prompts each round so revoked memory cannot survive in an old conversation.
