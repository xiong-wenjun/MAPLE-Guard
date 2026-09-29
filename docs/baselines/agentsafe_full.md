# AgentSafe paper-component reproduction with holder-local HierarCache

`evaluate/defense_methods/agentsafe_full.py` is an opt-in implementation of the AgentSafe paper components for MAPLE adapters. It does not replace the existing `agentsafe` communication-filter baseline. The host owns experiment sessions, external memory storage, topology, persistence, and lifecycle hooks. The component owns per-holder hierarchical memory, assessment caches, policy decisions, and junk memory.

## Source and fidelity

The inspected official repository is [junyuanM/Agentsafe](https://github.com/junyuanM/Agentsafe/tree/cc253ad48532fa6614a27557587086cfb87968ed), commit `cc253ad48532fa6614a27557587086cfb87968ed`. No license file or licensing statement was present in that checkout. This module uses independently written implementation and prompts; it does not vendor the released source or prompts. `agentsafe_full` names the component coverage, not an exact reproduction of the released social-demo experiment.

[AgentSafe v2, Section 3.2.2 and Algorithm 1](https://arxiv.org/html/2503.04392v2) specify permission checks, identity validation, hierarchical memory, instruction-library validity detection, junk memory, and periodic reflection. Equations 8–9 require every criterion to pass a strict similarity threshold. Equations 10–12 use reflection with criteria and prior junk; the algorithm removes rejected records from active memory. The paper does not supply the criterion library, threshold, embedding model, or complete identity-matching implementation. Its sender/receiver permission notation is inconsistent. This adapter uses explicit directional pair permissions and cosine similarity; these choices are disclosed adaptations.

The released [Agent class](https://github.com/junyuanM/Agentsafe/blob/cc253ad48532fa6614a27557587086cfb87968ed/Code/agents.py) has these properties:

- Each `Agent` instance holds its own `memory[level]` and `junkMem`. `GenHistory` reads levels cumulatively, and `UpdateMem` writes to that instance. The implementation here preserves this holder isolation even when two holders receive records with identical IDs and contents.
- `GetRelation` searches both orders of a name pair, so the released relation lookup is symmetric. This adapter's directional `relations[owner][reader]` is an explicit MAPLE policy adaptation; it must not be reported as source-equivalent relationship semantics.
- `ReviewMemory` reviews individual lines and removes unreasonable lines without adding them to junk. This adapter reviews complete records and moves rejected records to holder-local junk according to the paper workflow. It includes the configured criteria and that holder's prior junk, which is a paper-component recreation rather than the released `ReviewMemory` prompt.
- The source stores question/answer conversation text. MAPLE admission stores memory-record text, and routed inputs are retained as receiver-local records. Those different record boundaries can affect classification and reflection.

The separate `Code/attack.py` prototype and `Code/process.py` do not provide the complete paper workflow. In particular, the `0.2` cosine cutoff in `process.py` measures attack success, not memory admission. This implementation does not substitute that number for the unpublished admission threshold.

## Required configuration

```python
AgentSafeFull(args, judge, embed)
```

- `judge(messages: list[dict[str, str]]) -> str` and `embed(text: str) -> list[float]` are supplied by the host. No network client or model loading exists in this module.
- `args.agentsafe_policy_file` names a JSON object containing `identities` and an optional `relations` mapping.
- `args.agentsafe_criteria_file` names a nonempty JSON list of nonblank verification-criterion strings.
- `args.agentsafe_threshold` explicitly supplies a finite cosine threshold in `[-1, 1]`; there is no default or silent fallback.
- `args.agentsafe_review_interval` defaults to `1` and must be a positive integer. The default follows Algorithm 1; less frequent review is an experiment setting.

Example relationship policy:

```json
{
  "identities": {"0": "Agent 0", "1": "Agent 1", "2": "Agent 2"},
  "relations": {"0": {"1": 1, "2": 3}, "1": {"0": 4}},
  "default_level": 1,
  "self_level": 4
}
```

`relations[owner][reader]` authorizes the reader to access every level up to and including that integer. Relationships are directional and independent of binary topology edges. Unknown pairs default to level 1, and self access defaults to level 4; these are declared MAPLE adaptations, not released experimental defaults. Unknown senders absent from `identities` fail identity verification. Level names are Stranger 1, Colleague 2, Friend 3, Family 4. The classifier uses a generalized privacy rubric, not the released social-demo prompt.

Criterion vectors are normalized once. Admission computes cosine against every configured criterion and accepts validity only when every score is strictly greater than the threshold. Empty criteria, nonfinite/zero vectors, dimension mismatches, and missing configuration cannot silently grant access. No extra similarity-to-junk cutoff is introduced.

## Public APIs and holder identity

```python
admit(memory_id, text, owner, recipient, metadata, *, holder=None) -> (bool, details)
read(memory_id, text, owner, recipient, metadata, *, holder=None) -> (bool, details)
route(text, sender, recipient) -> (str | None, details)
history(holder, clearance=4) -> list[dict]
review(records, round_idx) -> list[dict]
state_dict() -> dict
load_state_dict(state) -> None
```

`owner` identifies the operational source of the data, while `holder` identifies the agent whose cache contains this copy. `recipient` is the operational reader or receiver used for permission checks. They are separate integer IDs; a record's owner does not determine all of its holders. The component never infers these values from text claims or evaluator labels.

`metadata` is a mutable host record dictionary, typically `entry.baseline_metadata`. The only top-level key changed by this module is `agentsafe_full`, with this shape:

```python
{
    "version": 3,
    "holders": {
        "2": {
            "holder": 2, "owner": 0, "level": 2,
            "fingerprint": "...", "config_fingerprint": "...",
            "quarantined": False, "reason": "agentsafe_allowed",
        }
    },
}
```

There is deliberately no shared `quarantined` flag. A holder-specific decision is selected with `metadata["agentsafe_full"]["holders"].get(str(holder), {})`. A host must never convert one holder's refusal into global inactivation of a shared source record.

- `admit` validates one holder's write. The default holder is `recipient` for private admission and `owner` for `recipient=None` shared admission. Shared admission defers reader clearance checks; private admission checks owner-to-recipient permission. Failed content, identity, or private permission enters only that holder's junk.
- `read` defaults the holder to the actual recipient and accepts an explicit holder for an existing cache owned by a different operational agent. It checks that holder's assessment/quarantine and the owner-to-recipient clearance. An ordinary permission refusal does not quarantine an otherwise valid copy; an allowed read retains the record in the selected holder's hierarchy.
- `route` checks an edge before recipient delivery or summarization. It always uses the recipient as holder, retaining accepted input in the receiver's hierarchy and rejected input in the receiver's junk. The same sender text may pass for one receiver and fail for another. The host enforces topology edges.
- `history` returns deep copies of one holder's active records from levels `1..clearance`. `memory[str(holder)][str(level)]` contains the active records at that level. External prompt assembly remains the host's responsibility and must still apply operational access checks.
- `review` accepts records shaped as `{"memory_id": str, "text": str, "owner": int, "holder": int, "metadata": dict}`; omitted holder defaults to owner for compatibility. At the configured interval it merges those records with its own active per-holder cache, including accepted routes, and deduplicates by holder, record ID, and content/owner version. A judge call receives one candidate, the criteria, and only that holder's prior junk. Junk prompts contain content and operational source identity, not record IDs, fingerprints, or evaluator metadata. Rejected records are removed only from the corresponding holder's active hierarchy and copied to that holder's junk. Already quarantined versions are excluded. Returned traces include `memory_id`, `action`, `reason`, and `details.holder`.
- The round index is zero-based; interval `N` runs when `(round_idx + 1) % N == 0`.

The JSON state format is version 3. `junk`, `cache`, and `memory` are all partitioned by canonical string holder ID. Assessment and junk keys include record ID plus the content/operational-owner fingerprint; changed contents or source identities get fresh assessments, while prior rejected versions remain rejected for that holder. The active hierarchy contains the current version for each ID. Snapshots must match the policy, criteria, threshold, and normalized criterion vectors. Junk remains effective even if a caller later supplies empty metadata. Version 2 snapshots and flat legacy metadata are rejected because their global decisions cannot be reliably assigned to holders; start a fresh experiment store rather than silently migrate them.

Classifier, identity, and reflection responses must be exact single-field JSON objects: `{"level": 2}`, `{"valid": true}`, and `{"junk": false}`. Wrong types, duplicate keys, extra fields, nonfinite JSON, malformed output, and provider failures raise `AgentSafeRuntimeError`; invalid startup policy raises `AgentSafeConfigError`. Hosts must surface these failures rather than reinterpret them as allow decisions.

## Host responsibilities

Keep one instance per experiment memory session. Pass actual holders on all private writes, team admissions, retrievals, and review records. Register host records by holder and content version rather than a single session-global memory ID. Call `review` at lifecycle boundaries and persist both component state and scoped record metadata. A private physical record may have its active/inactive status synchronized with its own holder; a shared physical record must remain available to other holders and use scoped filtering instead. Junk is checker context only and must never enter task prompts.

The interface accepts content and operational identities rather than complete MAPLE entries. It does not receive structured evaluator poison flags, answer fields, provenance scores, taint, or hidden attacker labels. Existing MAPLE consolidation can write reference answers into the memory text itself; this adapter does not remove that text. Such experiments retain the benchmark's outcome-feedback assumption and must not be described as fully free of reference-answer feedback. Storage scope and permissions must come from actual host routing decisions, not claims embedded in text.

For LongMemEval, configure reserved source `"-1"` as user-history and give intended readers explicit clearances. Initial history records are admitted and reviewed like other memories; their benchmark trusted designation is not an AgentSafe bypass. With MemRL, `--baseline-state-path` is mandatory. Use a distinct sidecar per experimental seed/memory store. The host must check scoped quarantine before handoff summarization and rebuild AgentSafe task prompts each round so revoked memory cannot survive in an old conversation.

## Validation

The standard-library tests use deterministic judge and embedding callables and need no live model or new dependencies. They verify independent holder decisions for identical source records, isolated reflection context, receiver-only route junk, route-cache reflection, cumulative levels, content-version caches, restart persistence, strict/all-criterion validity, malformed judgments/vectors, identity checking, directional permissions, and independence from evaluator labels. These are behavioral tests of the adapter, not validation of unavailable original prompts or experimental accuracy.

```sh
python3 -m unittest discover -s tests -p test_agentsafe_full.py -v
```

## Verified host integration

The runtime now inserts retained cumulative holder history into actual generation
prompts, rechecking owner-to-reader permission for every record before insertion.
It rebuilds prompts so reviewed junk cannot persist as stale conversational
context. Private physical-holder bindings are separate from source ownership
and survive restart. Shared holder quarantine never changes global shared status.

The component supports holder-local IDs. The external MAPLE store uses globally
unique memory IDs: creating a private copy for a different physical holder must
allocate a new ID. Both admission and retrieval reject conflicting private-ID
reuse before changing the registry; this prevents reflection under the wrong
holder. Runtime state version2 stores these bindings and configuration identity.
