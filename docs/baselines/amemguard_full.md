# A-MemGuard released-source profile

This component implements the released **joint LLM audit** and EHRAgent lesson
retrieval workflow from [TangciuYueng/AMemGuard](https://github.com/TangciuYueng/AMemGuard),
pinned to commit **dd92f7ff21b9a904a703141be3d5b80170e57228**. The reference checkout is
`/mnt/public/data/wj/baseline-references/AMemGuard` on inference1.
The method is identified as `official_joint_llm` in diagnostics and state.

The reasoning, joint-audit, provider-system, and preventive-warning prompts are
copied verbatim from `ReAct/consistency.py` and
`EhrAgent/ehragent/medagent.py` at that revision (including upstream wording).
The upstream MIT license and copyright notice are preserved in the module.
No answer-letter consensus, lexical similarity, attack labels, evaluator outcomes,
private system prompts, or synthetic model judgments enter the algorithm.

## Executed algorithm

1. Register all supplied operational records in the agent-visible catalog. Rank
   candidate questions against the current query with real embedding cosine
   similarity; retain at most `amemguard_top_k` (default 4).
2. Retrieve **previously stored** lessons before the current audit. For each
   query-selected record, embed its action and rank **the full visible catalog's
   actions**, including actions without lessons. Take the top
   `amemguard_lesson_top_k` (default 4) and collect their existing annotations.
   Duplicate lessons reached by multiple searches remain duplicated, as upstream.
3. Independently call the model once for each selected memory, combining the
   current query and only that memory. The upstream prompt requests response
   inference, entity extraction, and an entity–relation reasoning chain. Complete
   generations are preserved; the component does not truncate their text.
4. Send all generated paths and the query to the upstream joint-audit prompt.
   Retain a memory iff its JSON result has both `consistent: true` and
   `safe: true`. No path is restored if all are rejected.
5. Store each rejected path itself as that memory's lesson. This is the released
   structured lesson distillation: there is no additional invented summarizer.
6. Before the host generates the target response/action, inject the exact
   preventive warning and the lessons retrieved in step 2. The warning explicitly
   identifies counterexamples and instructs the model to avoid repeating them.
   The exact warning header is unconditional, even at cold start or when no
   action neighbor has a lesson. New lessons become available on later retrievals.

With K selected candidates, the audit makes K + 1 real chat calls. Empty
candidates make none. Diagnostics contain actual call counts and complete paths;
token usage is not fabricated when the provider callback returns only text.

## Controller contract

```python
guard = AMemGuardFull(args, judge, embed)
guard.begin_task(task_id)
guard.register_entries(all_agent_visible_entries, agent_id)
retained, diagnostics = guard.select(query, candidate_entries, agent_id)
warning = guard.lesson_prompt(query, retained, agent_id)
# Add warning to the target's context before generating its answer/action.
state = guard.state_dict()
guard.load_state_dict(state)
guard.reset()
```

- `judge(messages) -> str` must execute real inference in production. Use plain
  text output, not forced JSON-object response format: reasoning is free text and
  the joint audit specifically requests a JSON **array**.
- `embed(text) -> list[float]` must execute the configured embedding model. A
  finite, nonzero vector of consistent dimension is required. There is no hash,
  token-overlap, or other heuristic fallback.
- Entries may be objects or dictionaries with `memory_id`, `intent`,
  `experience`, and `origin_agent`. Returned entries are the original objects.
  Only these fields are read; only operational text/identity is persisted.
- `amemguard_top_k=4`, `amemguard_lesson_top_k=4`, and a nonblank
  `amemguard_experiment_id` are the component configuration. An experiment ID
  derived by the host from its state path is acceptable; explicit IDs are clearer.
- Call `begin_task` for every new task, including repeated identical queries.
  It clears transient retrieval snapshots, not lessons. Immediately following
  `select`, `lesson_prompt` returns the pre-audit snapshot, even when the retained
  memory list is empty. This matches upstream's timing.
- Register the **full agent-visible** catalog at backend hydration, admission, and
  visible imports, not merely the final query top-k. Otherwise action-neighbor
  retrieval is restricted to a partial catalog and is not the full source profile.
  Never register another agent's private records for this agent.
- State is JSON-serializable and instance-scoped, with separate per-agent catalogs
  and lesson stores. The host owns atomic file save/load and reset switches.
  Loading checks schema, source revision, experiment ID, retrieval parameters,
  catalog references and source fingerprints before mutating current state.
  Changed record content invalidates its old annotation. `reset` clears only this
  guard instance; there are no process-global lessons.

## Explicit source/paper differences and host adaptations

The [paper v1](https://arxiv.org/html/2510.02373v1) Appendix A describes synthesis
of a consensus plan followed by per-path comparisons. The pinned released
`ConsistencyChecker._check_with_llm` instead performs **one joint audit**. Likewise,
the paper's conceptual generate-then-extract process is one prompt with three
instructions in released code. This adapter follows the released code; it must
not be described as implementing the paper-only two-stage judge.

The paper's main text describes proposed-plan structural comparison, while
Appendix G and the released EHRAgent implementation retrieve by question and then
by action. This adapter follows the latter. It does not claim to extract and revise
the target's final action plan in a separate inference loop.

The released EHRAgent annotates `self.memory[i]["lesson"]`. This adapter stores
the same generated text separately, keyed by agent and source memory ID, with the
triggering query and a source fingerprint. This changes storage/isolation and
persistence, not lesson content or action-neighbor ranking.

EHRAgent memories have question/knowledge/code fields. MAPLE `intent` maps to
question and `experience` maps to action/code; there is no separate knowledge
field to invent. Empty or whitespace-only intent falls back to the nonblank experience for retrieval. The prompt's
memory payload is therefore `Question: ...\\nSolution:\\n...\\n`. Query-dependent
selection remains over the supplied candidate pool; the full visible catalog is
used for secondary action retrieval. Equal cosine scores use stable input order;
upstream NumPy's tie ordering is not specified.

The released ReAct runner chooses one surviving retrieval randomly; EHRAgent
uses all surviving examples. This component uses the latter all-survivor behavior
appropriate to the host's memory context. The source runner imports an obsolete
`check_consistency` function that the pinned refactored checker no longer exports;
the adapter invokes the same released algorithm through the host callbacks and
does not depend on that broken script import.

There is one deliberate operational failure-policy strengthening: upstream
malformed audit JSON marks all records inconsistent and continues, potentially
annotating every record. Here malformed/missing/duplicate IDs, non-boolean fields,
invalid paths, and provider failures raise `AMemGuardOutputError` with
`fallback_policy="reject_all"`. The experiment stops, no retained records are
returned and no lessons are created from that failed audit. This preserves the
source's rejection direction while preventing infrastructure errors from silently
becoming benchmark decisions or learned lessons. The host must not catch this
error and restore unvalidated memories. All-valid but all-negative judgments use
the source's empty-memory-context fallback.

The upstream warning is injected by the host into the target's context. EHRAgent
appends it to knowledge examples; the paper describes prepending it. The host's
actual insertion point must be recorded in runtime documentation. Model/provider
choice, deterministic decoding, call scheduling, and embedding implementation are
host-controlled adaptations and must be logged for each comparison. Source
independent paths are computed sequentially here, with isolated message lists;
there is no shared conversational history between paths.

## Reproduction and smoke requirements

No trained defense checkpoint is needed for this profile. A real smoke run needs:

- A reachable OpenAI-compatible chat endpoint and an actual served judge model,
  ideally matching the target backbone when reproducing the source configuration.
  Configure the runtime's full-judge URL/model or its shared chat URL/model.
- A reachable embeddings endpoint with a real selected embedding model and
  consistent vectors. DPR/REALM appear in the source experiment configurations;
  use of another encoder is an explicit experimental adaptation.
- A unique experiment ID and persistent baseline-state path for multi-task and
  restart tests, plus a fully hydrated agent-visible memory catalog.
- At least two operational memories that induce different query-conditioned
  paths. After a real model rejection, issue a subsequent related query, observe
  the actual retrieved lesson text after the unconditional warning header in the
  target's messages, save/reload state,
  and repeat. Do not manufacture a rejection or lesson to make this smoke pass.

The component unit suite uses explicit deterministic provider fixtures to test
control flow, never to report empirical defense effectiveness:

```bash
cd /mnt/public/data/wj/MAPLE-Guard-baselines
python3 -m unittest discover -s tests -p test_amemguard_full.py -v
```

TDD evidence: the initial 13 tests failed because the component was absent, then
passed after implementation. Two further regressions (same-query task restart and
literal template markers in operational text) were observed failing, then fixed.
A source-fidelity review added regressions for the unconditional cold-start
warning and whitespace-only intent state round-tripping; both were observed
failing before their fixes. Final component result: **17 tests passed**. A successful unit suite is not a live
model smoke or a benchmark result; no benchmark performance is claimed here.
