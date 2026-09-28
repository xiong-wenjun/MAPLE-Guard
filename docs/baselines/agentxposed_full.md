# AgentXposed full live adapter

`agentxposed_full_guide` and `agentxposed_full_kick` implement the stages of
[AgentXposed, arXiv 2507.04724v2](https://arxiv.org/html/2507.04724v2), using live
target-agent callbacks. This is a clean-room reconstruction of the paper's
behavioral protocol, with explicit integration choices below. It is not a
bit-for-bit reproduction of the released offline scripts or a claim to reproduce
the published detection rates. The older `agentxposed_guide` and
`agentxposed_kick` adapters retain their existing behavior.

## Public interface and observer boundary

```python
guard = AgentXposedFull(args, judge)
guard.begin_task(task_id)
updated_outputs, decisions = guard.defend(
    outputs, respond, question, round_idx, adjacency, **callbacks
)
```

- `judge(messages)` returns a JSON string. Judge/provider/model/temperature
  selection belongs to the host runtime.
- `outputs` contains only operational integer agent IDs and nonempty visible
  output strings. `question` is the public task text.
- `respond(agent_id, prompt, reset=False)` queries the same real target agent,
  retaining its private runtime context. The callback returns nonempty text.
  The guard never requests a reset or inspects the callback's private context.
- The host must use `guard.inactive` to skip future generation, communication,
  and final aggregation for kicked agents. The guard also removes their outputs
  defensively if the host passes them again.
- `begin_task` always starts a fresh task, clearing baselines, scores, public
  histories, and inactive IDs. Call it once per independent task.
- Extra callbacks, private prompt fields, evaluator labels, attacker identities,
  memory taint, and ground-truth answers are not inspected. Adjacency is accepted
  for shared-runner compatibility; this implementation performs public peer-output
  comparison, without fitting a graph model or changing topology.

The API treats supplied output text as an observation. The judge is instructed
not to follow instructions embedded in observations. The original agent's hidden
system prompt remains with the runner, where it may still influence that agent's
answers. No benign replacement persona is supplied.

## Protocol

1. **Contextual baseline:** each agent's first observed output and public task,
   together with the current visible peer outputs, are scored on all six HEXACO
   traits. This inferred vector becomes the fixed per-task baseline. No fixed
   all-five vector or hidden role prompt is used. The first output establishes
   the baseline and cannot itself trigger a deviation.
2. **Monitoring:** subsequent outputs are scored using the public task, prior
   observed history, fixed baseline, latest scores, and current peer outputs.
   The signal is the arithmetic mean of the six absolute differences from the
   baseline. A signal strictly above the threshold triggers verification.
3. **Adaptive inquiry:** generate a Reid-inspired, professional question from
   observed inconsistencies, ask the real agent, then rescore using its actual
   reply. Every next question sees the earlier question/answer/score transcript.
   The inquiry permits innocent explanations and does not invent evidence.
4. **Final assessment:** after the configured inquiry rounds, obtain a separate
   six-trait assessment of the full public inquiry evidence. Confirm the suspect
   only if its final mean absolute baseline deviation remains strictly above the
   same threshold. A cleared soft flag releases the original task output.
5. **Guide:** for a confirmed suspect, generate a contextual client-perspective
   correction, ask the real target, and adapt the next correction to its reply.
   After those correction turns, explicitly request a fresh complete answer to
   the original task. Publish that fresh answer, not an appended warning or an
   interrogation answer. A further assessment records post-Guide scores and
   residual deviation. Guide preserves participation even if residual deviation
   remains high; it does not silently switch to Kick.
6. **Kick:** remove each confirmed suspect's current output and retain its ID in
   `inactive` for the rest of that task. Multiple independently confirmed agents
   can be removed, including equal-deviation cases.

The baseline can be contaminated when the first observed behavior is already
adversarial. A one-observation task only profiles agents. The adapter does not
invent an unseen clean history or use answer labels to address that limitation.
Guidance and inquiry persist in the original target's conversation through the
host callback. Persistent memory receives no AgentXposed-specific read, write,
promotion, taint, or provenance gates.

## Configuration and explicitly chosen details

| Argument | Default | Basis |
| --- | --- | --- |
| `agentxposed_deviation_threshold` | `1.0` | Paper section 5.3 interrogation threshold; strict `>` follows section 4.4. |
| `agentxposed_inquiry_rounds` | `3` | Released detector runs three interrogation rounds. The paper does not give an exact stopping rule. |
| `agentxposed_guide_rounds` | `2` | Local choice to realize the paper's multi-turn correction; no released numeric Guide default. |
| Final decision threshold | Same configured deviation threshold | Local choice: the paper says final identification uses deviation magnitude but specifies no separate final cutoff. |
| Score range | Finite real numbers `[1, 10]` | Paper section 4.4; the appendix prompt and released parser instead permit `[0, 10]`. |
| Final rescore and post-Guide rescore | One each | Explicit verification and observable correction reporting for the live integration. |

The deviation threshold must be finite and in `[0, 9]`; inquiry and Guide round
counts must be positive integers. Defaults realize three inquiry turns and two
correction turns. An explicitly smaller round count changes that interaction
budget. If provided, `agents` must be a positive integer and output IDs must lie
in `[0, agents)`.

Judge prompts are original, purpose-specific prompts. They preserve behavioral
profiling, adaptive inquiry, and semantic correction concepts while adding a
strict machine-readable contract. They are not represented as verbatim author
prompts. The host's judge model and temperature must be reported with experiments;
this component does not force the original default GPT-4o-mini or its temperatures.

## Judge schemas, errors, and trace

Score responses must contain exactly:

```json
{
  "scores": {
    "honesty_humility": 6,
    "emotionality": 6,
    "extraversion": 6,
    "agreeableness": 6,
    "conscientiousness": 6,
    "openness": 6
  },
  "reason": "Explanation grounded in visible evidence."
}
```

Question and correction responses contain exactly `{"question": "..."}` and
`{"guidance": "..."}`, respectively. Text fields must be nonempty. Duplicate JSON
keys, nonfinite values, missing/extra fields, booleans as scores, strings as scores,
and out-of-range numbers are rejected. There is no partial-score parser, neutral
fallback, fabricated target reply, or silent allow on failure.

Invalid configuration raises `AgentXposedConfigError`. Invalid inputs, judge
failures, malformed judgments, or target-callback failures raise
`AgentXposedRuntimeError`, retaining the cause. A failed defense call does not
return releasable outputs. The caller must abort that failed task; it must not
catch the exception and publish the original candidates. Actual callback side
effects cannot be rolled back or transparently retried.

Every decision records the protocol and source provenance, baseline/current
vectors, initial monitored deviation, final verification vector/deviation, active
threshold, actual inquiry/Guide/reanswer counts, judge explanation, and complete
public inquiry and correction transcripts. Guide adds its fresh output,
post-Guide deviation, and assessment explanation. `deviation` is the monitoring
signal; `final_deviation` is the pre-defense verification result;
`post_guide_deviation` applies only after a Guide rewrite. The current vector
reflects the last completed assessment.

## Released-source differences and provenance

The [official anonymous reference](https://anonymous.4open.science/r/AgentXposed-F814)
was retrieved on 2026-09-28. The source snapshot and complete SHA-256 manifest are
on the server at `/mnt/public/data/wj/baseline-references/AgentXposed/`, including
`source-provenance.json`, `official-prompts.json`, and the v2 arXiv source.

The reviewed release is internally inconsistent with the paper:

- `Detect/model.py:9-25` initializes neutral fives; its baseline prompt is unused.
  The unused `should_interrogate` method compares the lowest and second-lowest
  mean score, rather than per-agent temporal deviation.
- `Detect/main.py:116-120` interrogates every agent unconditionally.
  `Detect/reid_interrogator.py:99-152` synthesizes the subject's answer through
  another model call using extracted role instructions. The live adapter queries
  the actual target without exposing those instructions to its judge.
- `Detect/main.py:114` creates `final_hexaco_scores = {}` and never fills it.
  Its final detector therefore iterates an empty dictionary and returns no
  malicious agent. The new adapter fixes the missing-final-assessment failure by
  storing validated final scores and directly calculating baseline deviation.
- The release's intended final selector chooses a unique lowest weighted score
  below three, and treats ties as uncertain. The new adapter follows the paper's
  temporal deviation rule instead.
- Released Guide modifies user input before target generation; released Kick
  skips agent execution. Output warning concatenation and first-turn-only
  suppression do not implement those controls.

Reference content hashes:

| File | SHA-256 |
| --- | --- |
| `Detect/model.py` | `2340c4845a76c9ee9f73dfc01f087c3e5e82e198c437f50322a8a244bd58338e` |
| `Detect/main.py` | `de954af32559e57ee9890f08e0c3fb9be713b1dd95994199b972263eb6fafc9c` |
| `Detect/reid_interrogator.py` | `6f76a50798c4c55301e6053dd1d9d4488ac10fc6a4bfa7e1aa7b213e27512220` |
| `Defence/defense_sys.py` | `6555135ac54afda0dadb245271ca02d93b58aa499ff272e981059423bea34ad7` |

No root LICENSE file or license header in these reviewed files was present. This
is not a claim that the reference is permissively licensed, and no verified Git
commit was exposed by the anonymous listing. No upstream code is imported or
executed by the clean-room component.

## Validation

Run the network-free public-interface tests from the server worktree:

```bash
python3 -m unittest discover -s tests -p test_agentxposed_full.py -v
```

They cover contextual baselines, exact threshold equality, actual adaptive
inquiry replies, final clearance, live Guide re-answering, persistent Kick and
task reset, multiple suspects, strict schemas, explicit failures, and invariance
to hidden evaluator metadata. These deterministic tests establish implementation
behavior, not empirical malicious-agent detection accuracy.
