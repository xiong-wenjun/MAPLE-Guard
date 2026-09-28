# Baseline identity and release-configuration audit

This read-only source audit was completed on 2026-09-29. It does not resolve
historical run identity without the original run manifests.

## GUARDIAN and Inspector are different methods

[MAPLE-Guard v1 Appendix E](https://arxiv.org/html/2608.00426) calls GUARDIAN
an independent reviewer and cites reference 11. That citation is
[Huang et al., ICML 2025](https://proceedings.mlr.press/v267/huang25ay.html),
whose two defenses are Challenger and Inspector. Its
[official repository](https://github.com/CUHK-ARISE/MAS-Resilience) describes
Inspector as retaining chat history and rewriting erroneous messages.

The existing `inspector_defense.py` is an adapter from INFA-Guard's evaluation
layout, with its own existing role-play prompt and safe/unsafe rewrite contract.
The matched strict runtime preserves that prompt and caller-owned history. It
is not certified as an exact run of the original MAS-Resilience repository.

The separate `guardian_defense.py` loads `model_static.py` and
`model_temporal_gib.py` from the communication-targeted experiment of
[JialongZhou666/GUARDIAN](https://github.com/JialongZhou666/GUARDIAN), implementing
[Zhou, Wang and Yang, GUARDIAN, NeurIPS 2025](https://arxiv.org/abs/2505.19234).
This method detects anomalous nodes/edges in temporal attributed graphs; it is
not the independent LLM reviewer described in MAPLE Appendix E. MAPLE's Figure 6
caption separately mentions MiniLM/BERT token accounting for GUARDIAN, further
preventing a reliable inference of historical identity from prose alone.

Therefore the Appendix E citation and description align with Inspector, while
the current method named guardian aligns with temporal-graph GUARDIAN. Retain
separate IDs, disclose the discrepancy, and recover the original result command,
code revision, detector assets and decision trace before relabeling or claiming
reproduction of the historical GUARDIAN row. Both can be provisional comparison
candidates; adding a supported ID does not make a main-table claim.

No GUARDIAN or MAS-Resilience checkout was present in the inspected
`/mnt/public/data/wj/baseline-references` directory. The INFA-Guard reference does
contain `evaluate/defense_methods/inspector_defense.py` and explicitly lists
Inspector in its README.

## AgentSafe has no released complete AppWorld policy

The inspected official AgentSafe checkout is commit
[cc253ad48532fa6614a27557587086cfb87968ed](https://github.com/junyuanM/Agentsafe/tree/cc253ad48532fa6614a27557587086cfb87968ed).
[The paper, equations 8–12 and Algorithm 1](https://arxiv.org/html/2503.04392v2)
requires natural-language verification criteria, a predefined similarity
threshold, hierarchical cache, identity/permission checks, junk isolation and
reflection. No complete criterion library or numeric admission threshold was
found in the paper or release. The checkout contains five Python prototypes,
two attack-data JSONL files and a one-line README; no experiment policy config.

In `Code/agents.py`, lines 18–45 use symmetric named relationships:
Stranger 1, Colleague 2, Friend 3, Family 4. Lines 47–66 classify the social-demo
content, lines 108–124 manage cumulative local memory, and lines 135–166 perform
LLM review/accuracy checking. The referenced `relations.txt` is not released.
`Code/process.py` lines 29–44 use text-embedding-ada-002 and cosine > 0.2 for
attack-output similarity; lines 154–159 turn that into defense rate. Neither
that encoder choice nor 0.2 is a published memory-admission default.

For `agentsafe_full`, the host must provide all operational identities and
explicit pair clearances in `--agentsafe-policy-file`, nonempty criterion strings
in `--agentsafe-criteria-file`, and an independently selected finite cosine
`--agentsafe-threshold`. Use `--agentsafe-review-interval 1`, actual configured
embeddings, a fixed judge with temperature 0, fresh holder-local hierarchical
memory/junk and a distinct `--baseline-state-path` for each run. Changing the
embedding model changes the interpretation of a cosine threshold.

A symmetric neutral peer policy (e.g. cooperative peers as Colleague 2) can be
frozen before benchmark evaluation, but it is a declared MAPLE adaptation. A
complete official benchmark-independent policy/criteria/threshold cannot be
recovered from these sources. Generic criteria, the attack-evaluation 0.2,
or an allow-all threshold must not be advertised as official settings. An
adapted main-row comparison needs documented criteria and a threshold selected
on separate development data before AppWorld test results are observed; an
exact release reproduction remains blocked on the missing authors' settings.
The current method's provenance must remain `paper_components_maple_adaptation`.
