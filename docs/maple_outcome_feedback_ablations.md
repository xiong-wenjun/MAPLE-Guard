# MAPLE outcome-feedback ablations

These are memory-native stream ablations of MAPLE, not separate released baselines. Use --method maple_guard --strict-comparison and --outcome-feedback-policy full|no_q|none. Both new profiles retain write, retrieval, promotion and broker security gates, embeddings, prompts, topology routing, new memory writes and deployment-side answer acceptance.

| Policy | Reward-driven Q updates | Trust/hazard and outcome counts | Gold-labelled self experience | Gold-conditioned promotion eligibility | Agent outcome reputation |
|---|---|---|---|---|---|
| full (default) | enabled | enabled | retained | retained if selected candidate policy uses it | enabled |
| no_q | disabled in entry and persistent MemRL backend | enabled | retained | unchanged | enabled |
| none | disabled | disabled | replaced by unverified observed output; no reference answer | removed; retain sampling, retrieval/consensus requirements and static security gate | disabled |

## Interpretation

no_q estimates the contribution of online reward-driven Q updates conditional on the remaining evaluator-assisted mechanisms. It is not a no-feedback system: correct answers may remain in self-history, and trust/hazard still learn from correctness. The strict operational ingress initializes Q to zero in every arm; gate-driven rewriting remains enabled. Existing non-strict outcome-conditioned initialization is outside the approved strict ablation scope.

none removes external reference-correctness feedback from runtime memory management. System answer votes and agent-generated claims remain observed behavior, not verified labels. Static provenance, ACL/taint inheritance and content-risk predicates remain enabled. The selected promotion policy in the matched AppWorld campaign is accepted_retrieved_private, so it already checks deployment-side answer agreement rather than reference correctness. Other candidate policies no longer use the success prerequisite under none.

Correctness evaluation, poison registries and SR/MDSR/RDA/exposure statistics remain enabled for measurement. Dataset construction and attacker target generation still use evaluation labels as in the control protocol. Consequently none means no reference-outcome feedback to the defender, not that the complete benchmark and attacker are label-free.

The policy is recorded in resolved arguments, each stream trace and the final summary, and included in runtime/checkpoint identities. Distinct policies need separate fresh namespaces. Their dynamic memory contents are expected to diverge.

## Matched experiment

The new profiles are based on commit 8949f93ffd88c58ab1bba4fec736d694756459c1 plus only this feedback patch, preserving the existing paired AppWorld controls. Each profile uses Gemma-4-31B-it, seed42, 200 AppWorld action-selection proxy tasks, 8 agents, 3 rounds, and star/chain/tree/random. Dataset/order, attacker assignments, poison schedule, prompt/config hashes, retrieval thresholds and budgets match the full control.

The old task budget remains 512 with strict truncation and bounded transport recovery. Do not pool these runs with the newer 2048/fail_task main-table protocol. Existing full/retrieval-only control lanes continue untouched. No recurring monitor is created.

Metrics follow the frozen scheduled protocol: memory metrics exclude planned poisoning tasks and include warmup; overall action SR uses all tasks. Third-round task exposure is a separate metric from agent-level RDA and target-hit success. Frozen traces record selected IDs after filtering; they do not provide an independent full accepted-request audit.
