# Official baseline reproduction plan

**Goal:** make the public-release configurations executable and auditable without claiming unpublished paper assets were recovered.

**Architecture:** freeze upstream source hashes and effective parameters in configs/baselines/official_reproduction.lock.json; isolate external source and model assets under /mnt/public/data/wj; keep release detector parity distinct from MAPLE task adaptation.

1. Add failing regression tests for GUARDIAN complete-graph/parsed-answer input, immutable source checks, BERT path validation, epochs, AgentSafe seeded official relations and INFA 800-dialogue recipe.
2. Implement evaluate/defense_methods/reproduction.py and tools/official_reproduction.py. AgentSafe calls only the released relationship generator without running initial.py's API side effects. INFA exports upstream argv and an auditable training specification; reject legacy single-output checkpoints and incomplete per-turn labels.
3. Add an opt-in GUARDIAN released_detector profile. Call pinned public static/temporal modules, public answer parser, 20 online epochs and CPU. Preserve the existing host_graph profile and frozen experiment snapshots.
4. Verify with unit tests and a real BERT-backed static/temporal parity test. Download only to the server and verify pinned asset digests.
5. Record exact distinctions, remaining unpublished inputs, isolated training/dev/test requirements and matrix release gates in docs/baselines/official_reproduction.md.
6. Commit as wenjun xiong, merge into main, run checks, push main only.

**Evidence needed before main-table release:** two real protocol tasks per backbone; genuine benchmark traces; fixed configuration/source/model hashes; separate released-core parity evidence. Unit tests do not count as benchmark results.

**Non-negotiable limitations:** AgentSafe paper criteria/threshold and original random realization are not published in the audited release. Native INFA weights must be trained, never manufactured by padding G-Safeguard heads. GUARDIAN fits online; no pretrained detector checkpoint is needed. User's Qwen/Gemma models and AppWorld action proxy are declared host adaptations.
