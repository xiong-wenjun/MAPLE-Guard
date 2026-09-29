# Existing communication guards under the strict memory protocol

The opt-in strict profile supports challenger, gsafeguard, guardian, inspector, and
maple_guard_no_write, maple_guard_no_retrieval,
maple_guard_no_promotion, maple_guard_no_cross_agent.

Use the common --strict-comparison --peer-communication settings, disable
evaluator-based vote exclusion, and give each run fresh memory and a separate
--baseline-state-path. Existing communication guards retain their current
adapter algorithms and native model assets. This integration does not establish
a complete reproduction of their original papers.

The matched runtime normalizes operational memory metadata and rendering, then
calls the existing communication dispatcher on round outputs. It does not add
MAPLE memory filters to these communication guards. The caller retains ownership of
OfficialDefenseState: core/OpenQA reset it per task, while the INFA transfer
runner carries it across records, matching their previous implementations.
Changing this lifetime would be a separate evaluation protocol. The memory
sidecar does not persist or resume communication-detector state across processes.

Challenger still uses the safeguard endpoint/model and its existing
safe/unsafe prompt and replacement text. Inspector uses the same safeguard
endpoint authentication (SAFEGUARD_OPENAI_API_KEY), preserves its existing
review-and-rewrite prompt, and stores the rewritten message in reviewer history.
Both LLM guards use temperature 0. G-Safeguard still loads the configured
single-output GAT checkpoint and sentence-transformer embeddings. GUARDIAN
still fits the existing static/temporal graph adapter and persistently removes
the selected node within the caller's state. No weights, thresholds, detector
training procedure, or model identities are substituted by the strict runtime.

Under strict comparison, unavailable model/dependency assets, failed judge or
detector calls, malformed score shapes/nonfinite scores, invalid GNN thresholds,
and invalid Challenger/Inspector judgments terminate the run. Inspector requires
a safe/unsafe decision and a nonempty revision for unsafe judgments; provider
errors cannot become safe decisions. Outside strict mode,
legacy fallback behavior remains. An edgeless graph, or GUARDIAN with one
remaining active agent, keeps its existing explicit no-detection behavior;
these are graph conditions, not swallowed runtime errors.

MAPLE ablations use the same matched runtime and existing phase-specific
bypasses. The memory topology is not automatically changed. In particular,
global-shared bypasses the broker in the existing implementation; a meaningful
cross-agent-gate experiment must explicitly freeze a compatible topology in all
arms and verify its decision trace.

tests/test_strict_legacy_baselines.py covers real dispatcher routing, a
two-round core runner, caller-controlled state lifetimes, all four gate
bypasses, and error/fallback semantics. External model I/O is replaced by
deterministic fixtures; these tests are not trained-model validation or
benchmark results.


The paper's GUARDIAN identity is unresolved; see
[baseline_identity_audit.md](baseline_identity_audit.md). Inspector and GUARDIAN
remain separate method IDs. Adding Inspector support does not rename historical
results or establish eligibility of either implementation for the paper row.
