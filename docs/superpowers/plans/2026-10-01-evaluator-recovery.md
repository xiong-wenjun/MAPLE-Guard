# External Evaluator Recovery Plan

Date: 2026-10-01

## Objective

Preserve successful task outputs when an auxiliary external evaluator returns a malformed semantic verdict. Save the exact invalid response and request privately, mark the affected score as pending, and continue the benchmark under the user-approved failure policy. Resume affected live MMLU lanes from their complete task checkpoints without changing methods, prompts, policies, calibration, seeds, topologies, or token budgets.

## Required behavior

- Strictly validate evaluator schemas. Never coerce malformed verdicts to false, success, or zero.
- Only evaluator semantic verdict failures become pending. Provider envelopes, task responses, internal defenses, transport failures, and unrelated exceptions retain their existing handling.
- Correctness evaluator failures retain unknown correctness and record the feedback gap; offline rescoring cannot silently repair the original online trajectory.
- Auxiliary imitation evaluator failures do not invalidate independently observed primary SR, RDA@3, ASR@3, and MDSR@3. Auxiliary score coverage and bounds remain explicit.
- Journals contain exact requests and raw verdicts, are private, and retain request hashes and task identity.
- Checkpoint migration requires an explicit approval manifest containing exact original and new source digests. All non-source identities and codec schemas remain unchanged. Default resume remains strict.
- The reviewed native runtime alias checkpoint correction between 86a8b158 and 44659df4 is allowed only when explicitly identified in a generic runner migration; no semantic method changes are permitted.

## Implementation and validation

1. Add a dedicated semantic evaluator error, private pending journal, and focused tests for malformed, budget, task, defense, and provider failures.
2. Separate primary metric eligibility from auxiliary score pending status, with correctness and feedback coverage tests.
3. Add narrowly scoped, explicit checkpoint source migration and rejection tests for changed budgets, identities, source files, assets, and corrupt manifests.
4. Restore copies of real stopped No Defense and G-Safeguard checkpoints to verify cursor, memory, metric, and runtime preservation.
5. Run focused and complete regression tests. Obtain sequential specification and quality reviews.
6. Commit in English using the user's identity, merge and push main only, and freeze the verified source.
7. Resume diagnosed affected lanes with stable lane locks and explicit migration lineage. Verify actual PIDs, source, APIs, progress, and checkpoint hashes.
8. Record the campaign state and pending score protocol. Preserve all old logs and results. Do not recreate periodic automation.


Prompt asset relocation exception: existing relative prompt arguments resolve inside the old frozen source tree. Moving to the approved source relocates only those paths inside the old/new `prompts` subtrees. The operator manifest must enumerate each old/new absolute path, unchanged relative path, identical presence and identical SHA256. The worker verifies both files and rejects traversal, symlinks, external absolute prompt paths, added/removed fallback identities, prompt byte changes, and all other non-source identity changes. The source transition discloses this exception; it does not alter any prompt argument or content.

Operational API: `resume_job(job, evaluator_recovery={"manifest": absolute_path, "sha256": exact_authorized_file_hash})`; the lane may supply this exact object as `evaluator_recovery`. The worker flags are `--resume-task-checkpoint --checkpoint-evaluator-recovery-manifest PATH --checkpoint-evaluator-recovery-sha256 HASH`. The gate remains confined to nonzero `fail_task` QA checkpoints from the verified 86a8b158/44659df4 source fixtures. Native PI/TA migrations remain rejected.
