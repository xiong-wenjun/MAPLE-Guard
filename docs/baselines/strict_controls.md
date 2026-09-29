# Two custom controls and matched detector placement

The first two controls are new experiments for this paper, not named external papers with official implementations to clone.

## Provenance/taint plus ACL

Method: `provenance_acl`. Labels come from the actual ingestion path and authenticated writer/custodian, in a protected sidecar. Text claiming administrator status or an attacker-supplied clean tag cannot establish trust.

| Operation | Independent rule |
| --- | --- |
| Write / promotion | Actor owns the record; inherited permissions cannot widen |
| Retrieval | Reader is authorized |
| Transfer | Sender and recipient are authorized for every consumed dependency |
| Every phase | External or unknown taint in any dependency denies the operation |
| Derivation | Union sources/taints and intersect reader sets |

The custodian storing authenticated user history can differ from source identity -1. External imports and actual tool observations are tainted. Authenticated user requests are roots of trust; compromised agents and query-only injections are not recognized using evaluator labels. Retrieved memory and received messages propagate constraints through summaries and multiple relays.

This conservative rule can reject useful external evidence: measure that false-positive cost. It calls no MAPLE risk score, learned detector or safety judge. The ordinary task model/retrieval infrastructure remains common.

## Same MAPLE scorer, different deployment phases

Compare `maple_guard --strict-comparison` with `maple_guard_retrieval_only --strict-comparison`.

| Controlled item | Both arms |
| --- | --- |
| Retrieval scoring | Same score_memory, risk functions and coefficients |
| Retrieval decision | Same read firewall and threshold |
| Raw task / attack stream | Same dataset, order, attack configuration and seed |
| Available metadata | Same operational provenance and post-task feedback protocol |
| Prompt / model / topology | Identical configuration |
| Initial state | Fresh independent stores with identical initialization |
| Deployment difference | Full MAPLE additionally gates/revises writes, promotion and brokered memory access |

Full MAPLE includes existing rule branches as well as risk scoring. This is a lifecycle-deployment ablation of the implementation; not every gate is a threshold on one scalar. Store topology remains common.

After an intervention, realized memories, outputs and feedback can differ. Matching the protocol does not mean forcing identical post-treatment memory into both arms. A frozen-state detector comparison would be a separate experiment.

## PIGuard deployment pair

`piguard_retrieval` and `piguard_lifecycle` use the same pinned detector, 512-token profile, 0.5 threshold and memory input: intent + newline + experience. The latter also checks writes, promotions and transfers. Communication is scored as routed text with an empty intent. Additional detector calls count as overhead.

## Evaluation boundaries

Strict controls normalize loaded metadata before broker/ranking, record actual dependencies and feedback, and reject sidecars from a different experiment. Sidecars are trusted runtime state outside the attacker's write capability.

The INFA transfer runner uses neutral peer-memory descriptions in strict/full mode; actual TA observations reach provenance tracking. LongMemEval scopes candidates before A-Mem top-k/auditing. A-Mem's separate lessons intentionally persist across tasks for the same operational agent, following the source; isolate unrelated users/datasets with separate experiment state.

Final voting includes all active agents. AgentXposed may remove nodes selected by its defense; evaluator attacker IDs cannot remove votes. Existing post-task correctness feedback/reference answers remain a disclosed benchmark protocol. Run a feedback-disabled setting before claiming deployment without oracle outcome feedback.

Report attack success, clean utility, benign rejection with its denominator, poisoned-memory use/propagation, per-phase decisions, calls, tokens and latency. Attack-only streams cannot establish FPR. Summary `baseline_provenance` includes available provider-call/timing counters; unavailable token counts are not invented. Use paired seeds and confidence intervals before updating paper tables.
