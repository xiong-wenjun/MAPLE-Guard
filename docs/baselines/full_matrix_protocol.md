# AppWorld revision matrix: 15 seeds × 4 communication topologies

This is an expansion of the current strict AppWorld-derived action-selection pilot.
It is not native AppWorld tool execution or a reproduction of the historical paper numbers.

## Frozen candidate axes

- Backbones: Qwen/Qwen3.5-122B-A10B and google/gemma-4-31B-it.
- Seed list: 42 through 56, paired across methods, models, and topologies.
- Communication topologies: chain, tree, star, random, following the paper's controlled graphs and Appendix H random graph.
- Eight agents, three rounds, fixed source order of all 200 supplied tasks.
- Each method/backbone/benchmark requires 15 × 4 = 60 runs, or 12,000 task evaluations.
- Eleven main-table candidates across two backbones produce 1,320 runs / 264,000 task evaluations for AppWorld.
- A two-task smoke is an execution check and never counts toward these 60 cells.

The fourth graph and seed list are a declared expansion plan inferred from the current paper and the user's requested counts; any later explicit user correction supersedes it before expansion starts.
The CLI supports full/fully-connected/complete as well, but a full graph is not silently substituted for random.

## Exact graph and seed semantics

Existing core graph semantics are preserved. Star is hub 0 plus a leaf ring, not a pure star.
Chain and binary tree are bidirectional. Random independently samples directed non-self edges,
using the frozen run seed and the current default probability 2/(8-1); connectivity is not guaranteed.
The communication graph affects both peer messages and enabled edge-based private-memory propagation.
The memory-access policy remains brokered-shared.

Experiment seeds control attacker sampling, poisoning positions, random communication edges,
benign promotion sampling, and Python-random memory exploration. They do not guarantee bitwise
reproducibility of model serving and are not forwarded as model API seeds. All 200 task IDs and their
order are fixed. Report uncertainty across the 15 paired seeds for each topology; do not describe
the design as 60 independent seeds. Do not silently alter attacker sampling to protect agent 3:
target_hit is defined on benign agents' poisoned-answer hits, not that agent's identity.

## Preserve healthy runs

The ongoing star/seed42 200-task pilots can occupy the corresponding matrix cells after validation.
Do not restart them merely because the runner now supports additional axes. Match model checkpoint,
dataset hash/order, resolved algorithm settings, defense assets, and valid source version first.
Only count a successful 200-task trace with summary and clean API instrumentation. Running jobs are
reserved cells, not completed results. Never count or resume old failed/base/smoke records as IT results.

The 7ff74c7 and fa8e4ee engines differ only in released AgentXposed judge timeout configuration,
provenance, and validation; retain corrected fa8e4ee for AgentXposed. Other healthy current methods
retain the same algorithm semantics. The topology-runner change modifies orchestration and identity,
not the graph/defense algorithm. Record original source hashes for any retained result.

Every new model/method/topology/seed gets a fresh memory directory, MEMOS registry, sidecar, and run ID.
No state is copied between cells. New IDs include topology; legacy_run_id is a lookup hint,
not automatic certification or resumption.

## Execution split

- inference1: existing Qwen runs and fixed Qwen judge requests continue.
- inference2: same Qwen model mount and container image, with matching main serving configuration;
  pending A-MemGuard, G-Safeguard, MAPLE retrieval-only, and PIGuard lifecycle star/42 pilots were moved
  to a separate execution campaign here. Both task and judge requests use inference2.
- inference3: Gemma IT task generation. Current IT jobs preserve their Qwen judge on inference1.
- embedding service: shared Qwen3-Embedding-8B endpoint.

Assign future Qwen cells deterministically across inference1/inference2; keep each cell on one endpoint
and record it. Endpoint identity and shared model assets must be verified, rather than assuming two
matching public model names imply matching deployments.

## Preparing an expanded matrix

The runner accepts --topologies and --seeds. It freezes topology-specific YAML plus hashes and rejects
duplicate seeds, semantic topology aliases, or methods before writing run state. For example:

    python tools/run_appworld_matrix.py --bundle /path/to/appworld_200.json \
      --services /path/to/service-credentials.json --backbone qwen \
      --task-service inference2 --judge-service inference2 --profile strict \
      --phase pilot --tasks 200 --topologies chain tree star random \
      --seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 \
      --run-root /path/to/a/fresh/matrix

Omitting --execute prepares a manifest only. Do not blindly execute all prepared cells while existing
star/42 jobs are active: the current runner does not automatically adopt or resume them.
The expanded manifests are a reviewable coverage plan. Actual expansion must respect per-method
validation and the reserved existing cells.

AgentSafe, native INFA, and GUARDIAN retain their explicit asset/identity blockers. More seeds do not
resolve missing official components. No blocked baseline is replaced by a fallback.

## Bounded dispatcher

The prepared manifest does not launch itself. Use `tools/dispatch_appworld_matrix.py`
on each assigned execution host with `--execute` to run its Qwen and Gemma shards.
Without that flag it prints a read-only scheduling preview. The dispatcher imports
the frozen matrix engine and uses the original job commands unchanged; its own
code revision is separate from the experimental engine revision.

The 2026-09-29 deployment uses a maximum of two additional workers per host and
one additional Gemma worker per host. Total local worker budgets are eight on
inference1 and six on inference2, counting the already-running pilots and probes.
If an existing campaign already exceeds the budget, the dispatcher waits for
capacity. Existing campaign supervisors may replace their own finished jobs, so
this is an admission budget for the new queue, not a global process hard limit.

All star/42 reservations remain owned by the existing campaigns, including those
still in their queues. A reservation is not a completed result. Blocked methods
remain blocked. AgentXposed must finish its corrected two-task probe before
expansion. Other methods require successful two-task probes with a summary and
clean API journal; the base No Defense and MAPLE Qwen arms additionally allow
the current healthy strict pilot's first two completed tasks as readiness evidence.
Neither kind of readiness evidence is a final result or a performance claim.

Each host holds an exclusive dispatcher lock. Existing run directories are
never reused; live workers are identified by their exact run ID in /proc command
arguments on the correct host. Restarting the dispatcher reserves surviving
workers and their capacity without rerunning them. A dead unfinished worker
requires investigation; its memory is not automatically resumed or discarded.
A failed pilot holds that model/method's future cells across both shards while
other validated families can continue. No automatic retry is performed.

At startup, the dispatcher checks the frozen engine fingerprint, dataset and
topology config hashes, model/service identities and duplicate cells. Credentials
remain outside Git. Progress is written to execution-inference1.json and
execution-inference2.json; the preparation manifests remain unchanged. Per-cell
run.json, trace and API journal are authoritative for actual execution.

Example (replace the hostname and limits for the assigned host):

    python tools/dispatch_appworld_matrix.py --plan-root /path/to/prepared-plan \
      --engine-root /path/to/frozen-code --services /path/to/private-services.json \
      --host inference2 --expected-hostname ACTUAL_HOSTNAME \
      --max-total-workers 6 --max-new-workers 2 --max-new-gemma 1 --execute

The queue is deliberately bounded: activating 1,320 planned cells never means
starting 1,320 concurrent workers. Only completed and separately validated
200-task results can be reported as matrix coverage.
