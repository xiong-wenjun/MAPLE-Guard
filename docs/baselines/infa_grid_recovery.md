# INFA concurrent grid recovery

The released-budget generation recipe is unchanged: Qwen with no-thinking request metadata, nonempty stop/length content at 1024 tokens, 20 grids of 40 dialogues, and 50 training epochs. Native generation code and its pinned hashes remain unchanged.

## Failure and wrapper repair

The pinned generator checks whether its shared output directory exists and then creates it without `exist_ok`. Two generation processes can race there. The old wrapper also subtracts directory-wide JSON inventories before/after execution; another grid completing in between can make a successful stage fail the single-output check. A retry then rejects the existing output.

The wrapper now holds the grid lock while inspecting files, precreates a unique per-attempt native output directory, selects only the requested attackers/sparsity combination, validates all dialogues and journal responses, and publishes the complete output atomically without overwriting an existing file. The marker records the effective storage-only argv adaptation.

A markerless output is accepted only if a cache-only run of the native generator reproduces every dialogue field, including graph metadata and infection labels. Request hashes, response hashes, imported-cache hashes, fixed budgets, finish reasons, and held-out exclusions remain enforced. Missing or mismatched cache entries cannot issue an API request in this mode.

## Frozen source transition

Use a fresh controller run directory with `--resume-from OLD_RUN_ROOT`. The transition verifies the old frozen wrapper fingerprint, identical recipe/data/service/settings hashes, pinned native files, old recipe snapshot, and every previously recorded generation marker and artifact. It preserves the old run directory and attempt history. The new `source-transition.json` records both source fingerprints and prior attempts; only the repaired wrapper gets a new bounded retry allowance. Resuming the new controller verifies its transition manifest hash.

## Audit of the stopped released-budget run

Prepared artifacts are under `/mnt/public/data/wj/baseline-reproduction-20260929/infa-recovery-r2`.

- Grid 00: 40 dialogues and 1,280 accepted replies. A cache-only native replay reproduced the complete output byte-for-byte with zero API calls. The output SHA256 is `d6d1ba6521d58f35ae99329b0e96e961a2caa9f13133fe17ca4991497ab32c6b`.
- Grid 01: 40 structurally valid dialogues and 1,280 hash-valid accepted replies, but cache-only native replay on two hosts fails request equality. For the first question, saved data assigns attacker node 5 while replay assigns node 7. The affected request hashes differ. This grid is not accepted for training.
- Grid 02: eight hash-valid cached replies, but a cache-only native prefix replay fails request equality before any accepted replay. They are not reused.

`quarantine-plan.json` inventories every file for grids 01 and 02 with SHA256 values. `apply_quarantine.py` takes the job/controller and affected grid locks, verifies immutable old recipe/state/accepted markers and all source files, then moves the unproven output/journals/logs into a separate quarantine directory without deleting data. It supports safe completion after interruption. The old failed pipeline state remains unchanged at 40 accepted dialogues. No quarantine or API generation was performed while preparing this audit.

Before launch, apply that reviewed quarantine plan. Then use `launch-plan.json`, substituting the new committed frozen source for `FROZEN_SOURCE`. This retains the proven protocol/grid-00 markers, starts with 40 accepted formal dialogues, and generates grids 01 and 02 from fresh journals. The second 40 preserved dialogues must remain outside the merge directory.

Verification: seven regression cases initially failed for missing isolation, recovery, and migration behavior. All 68 INFA tests subsequently passed (`python -B -m unittest discover -s tests -p 'test_infa*.py' -v`). The concurrency regression runs two separate processes. Runtime replay reports and test logs are retained in the recovery directory's `verification` folder.
