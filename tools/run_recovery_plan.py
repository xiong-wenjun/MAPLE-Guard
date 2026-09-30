"""Run an audited recovery queue with task checkpoints and per-lane locks.

Plans identify existing failed lanes. Successful original runs and live workers
are never replaced. Operators can resume this controller after inspecting
its state; transport retries are bounded and output validity remains mandatory.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"maple_guard.run_appworld", "maple_guard.run_mmlu",
           "maple_guard.run_longmemeval", "maple_guard.infa_memlink_eval"}
OVERRIDES = {"--response-budget-policy", "--chat-timeout", "--pattern-judge-max-tokens", "--max-tokens", "--chat-max-tokens", "--full-judge-timeout", "--full-judge-max-tokens", "--answer-judge-max-tokens"}

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)

def flag(command, name, default=None):
    return command[command.index(name)+1] if name in command else default

def rewrite_job(job, directory, source, overrides):
    if set(overrides) - OVERRIDES:
        raise ValueError("Recovery cannot override the method, data or experimental identity")
    if "--response-budget-policy" in overrides and overrides["--response-budget-policy"] not in {"strict", "fail_task"}:
        raise ValueError("Recovery response-budget-policy must be strict or fail_task")
    new = copy.deepcopy(job)
    old_dir, old_id = str(job["directory"]), job["run_id"]
    command = [x for x in job["command"] if x not in {"--resume-task-checkpoint", "--checkpoint-allow-budget-change"}]
    entry = next(x for x in command if x.endswith("/tools/run_instrumented.py"))
    old_source = Path(entry).parents[1]
    if not MODULES.intersection(command):
        raise ValueError("Unrecognized benchmark runner")
    def rewrite(value):
        if value == old_dir or value.startswith(old_dir + "/"):
            return str(directory) + value[len(old_dir):]
        return value
    command = [rewrite(x) for x in command]
    command[command.index(entry)] = str(source / "tools/run_instrumented.py")
    for name in ("--memory-run-id", "--baseline-experiment-id", "--amemguard-experiment-id"):
        if name in command:
            command[command.index(name)+1] = directory.name
    if "--config" in command:
        index = command.index("--config")+1
        if not Path(command[index]).is_absolute():
            command[index] = str(old_source / command[index])
    for name, value in overrides.items():
        if name in command:
            command[command.index(name)+1] = str(value)
        else:
            command += [name, str(value)]
    checkpoint_dir = str(directory / "task-checkpoints")
    if "--task-checkpoint-dir" in command:
        command[command.index("--task-checkpoint-dir")+1] = checkpoint_dir
    else:
        command += ["--task-checkpoint-dir", checkpoint_dir]
    if isinstance(new.get("resolved_args"), dict):
        new["resolved_args"].update(task_checkpoint_dir=checkpoint_dir, resume_task_checkpoint=False,
                                    checkpoint_allow_budget_change=False)
        for name, value in overrides.items():
            parsed = str(value) if name == "--response-budget-policy" else float(value) if name.endswith("timeout") else int(value)
            new["resolved_args"][name[2:].replace("-", "_")] = parsed
    # Completion metadata belongs only to the original attempt.
    for name in ("pid", "exit_code", "finished_at", "completed_tasks", "api_usage", "started_at"):
        new.pop(name, None)
    new.update(run_id=directory.name, directory=str(directory), command=command,
               status="prepared", original_run_id=old_id, original_directory=old_dir,
               recovery_overrides=dict(overrides))
    return new

def resume_job(job, evaluator_recovery=None):
    """Keep the same source, outputs and experimental identity; worker verifies hashes."""
    command = list(job["command"])
    checkpoint = flag(command, "--task-checkpoint-dir")
    if not checkpoint or not Path(checkpoint, "latest.json").is_file():
        return None
    entry = next(x for x in command if x.endswith("/tools/run_instrumented.py"))
    source_transition = None
    if Path(entry).parents[1].resolve() != ROOT.resolve() or evaluator_recovery:
        if not evaluator_recovery or set(evaluator_recovery) != {'manifest','sha256'}:
            raise ValueError("Resume requires the same frozen source or explicit evaluator recovery authorization")
        from types import SimpleNamespace
        from maple_guard import task_checkpoint as checkpoint_codec
        try:
            pointer = read(Path(checkpoint)/'latest.json')
            name = pointer['generation']
            import re
            if not re.fullmatch(r'task-[0-9]{6,}-[0-9a-f]{32}',name):
                raise ValueError('Invalid checkpoint pointer')
            manifest_path = Path(checkpoint)/name/'manifest.json'
            if manifest_path.is_symlink() or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != pointer['manifest_sha256']:
                raise ValueError('Checkpoint manifest integrity mismatch')
            manifest = read(manifest_path)
            saved = manifest['identity']
            current = copy.deepcopy(saved)
            current['source'] = checkpoint_codec.source_identity(ROOT)
            options = SimpleNamespace(**checkpoint_codec._decode(saved['config']))
            current['prompts'] = checkpoint_codec.prompt_identity(options)
            options.resume_task_checkpoint = True
            options.checkpoint_evaluator_recovery_manifest = evaluator_recovery['manifest']
            options.checkpoint_evaluator_recovery_sha256 = evaluator_recovery['sha256']
            source_transition = checkpoint_codec._evaluator_source_transition(saved,current,options,manifest['next_task_index'])
        except (KeyError,OSError,checkpoint_codec.CheckpointError) as exc:
            raise ValueError('Unapproved evaluator checkpoint source recovery') from exc
        command[command.index(entry)] = str(ROOT/'tools/run_instrumented.py')
        for option,key in (('--checkpoint-evaluator-recovery-manifest','manifest'),('--checkpoint-evaluator-recovery-sha256','sha256')):
            if option in command:command[command.index(option)+1] = evaluator_recovery[key]
            else:command += [option,evaluator_recovery[key]]
    result = copy.deepcopy(job)
    if "--resume-task-checkpoint" not in command:
        command.append("--resume-task-checkpoint")
    result.update(command=command, status="prepared", checkpoint_resume=True,
                  evaluator_source_recovery=source_transition,
                  resume_count=int(job.get("resume_count", 0))+1)
    if isinstance(result.get("resolved_args"),dict):
        result["resolved_args"]["resume_task_checkpoint"] = True
        if evaluator_recovery:
            result["resolved_args"].update(checkpoint_evaluator_recovery_manifest=evaluator_recovery["manifest"],
                checkpoint_evaluator_recovery_sha256=evaluator_recovery["sha256"])
    return result

@contextlib.contextmanager
def lane_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield handle

def running_directory(directory):
    directory = str(directory)
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = [x.decode(errors="replace") for x in (proc/"cmdline").read_bytes().split(b"\0") if x]
            if MODULES.intersection(args) and any(x == directory or x.startswith(directory+"/") for x in args):
                return int(proc.name)
        except OSError:
            continue
    return None

def events(directory):
    path = Path(directory)/"api-calls.jsonl"
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except ValueError:
                # A partial record is never treated as a valid completed run.
                result.append({"invalid_for_benchmark": True, "malformed_record": True})
    return result

def retryable_failure(directory):
    directory = Path(directory)
    log = (directory/"run.log").read_text(errors="replace")[-12000:]
    records = events(directory)
    if any(x.get("invalid_for_benchmark") or "length" in x.get("finish_reasons", []) for x in records):
        # Allowed earlier task boundaries must not block recovery of a later
        # transport failure. Unmatched/strict response failures remain fatal.
        try:
            rows = [json.loads(line) for line in (directory/"trace.jsonl").read_text().splitlines() if line.strip()]
            from maple_guard.budget_outcomes import validate_committed_events
            matched, _count, _pending = validate_committed_events(rows, records)
        except (OSError, ValueError, TypeError):
            return False
        if not rows or not matched:
            return False
    if any(x in log for x in ("AttributeError", "NameError", "AssertionError", "SyntaxError",
                              "JSONDecodeError", "No such file or directory")):
        return False
    recent_error = any(x.get("error_type") or x.get("http_status", 0) >= 400 for x in records[-3:])
    transport_reason = any(x in log for x in ("HTTPError", "ReadTimeout", "ConnectionError",
        "URLError", "model call failed", "judge failed", "TimeoutError", "timed out"))
    return recent_error and transport_reason

def diagnosed_semantic_evaluator_failure(directory):
    """Only the diagnosed external schema failure can bypass the transport-only retry gate."""
    directory = Path(directory)
    log = (directory/'run.log').read_text(errors='replace')[-12000:]
    calls = events(directory)
    if not calls or 'Malformed external evaluator verdict' not in log:
        return False
    last = calls[-1]
    return (last.get('role') == 'evaluator' and last.get('http_status') == 200
            and bool(last.get('finish_reasons')) and 'length' not in last['finish_reasons']
            and not last.get('invalid_for_benchmark') and not last.get('error_type'))


def validate_result(directory, exit_code, expected_count, expected_ids=None):
    directory = Path(directory)
    trace = directory/"trace.jsonl"
    rows = []
    malformed = False
    if trace.exists():
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                malformed = True
    ids = [x.get("task_id", x.get("sample_id")) for x in rows]
    calls = events(directory)
    invalid = sum(bool(x.get("invalid_for_benchmark")) or "length" in x.get("finish_reasons", []) for x in calls)
    errors = sum(bool(x.get("error_type")) or x.get("http_status", 0) >= 400 for x in calls)
    summary_path = directory/"trace.summary.json"
    try:
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    except (ValueError, OSError):
        summary = {}
    allowed = 0
    pending = False
    protocol_valid = True
    budget_policy = summary.get("response_budget_policy") == "fail_task" and summary.get("budget_outcomes_version") == 1
    if budget_policy:
        from maple_guard.budget_outcomes import validate_committed_events
        protocol_valid, allowed, pending = validate_committed_events(rows, calls)
    valid = (exit_code == 0 and not malformed and len(rows) == expected_count
             and len(set(ids)) == expected_count and None not in ids and invalid == allowed and protocol_valid
             and (directory/"trace.summary.json").is_file()
             and (expected_ids is None or ids == expected_ids))
    return {"valid":valid, "completed_tasks":len(rows), "invalid_responses":invalid,
            "transport_errors":errors, "task_order_verified":expected_ids is not None,
            "accepted_budget_responses":allowed, "pending_evaluation":summary.get("pending_evaluation",pending),
            "pending_correctness_evaluation":pending,
            "pending_auxiliary_evaluation":summary.get("pending_auxiliary_evaluation",False),
            "metrics_valid":valid and not pending and (not budget_policy or summary.get("main_table_eligible") is True),
            "accuracy_metrics_valid":valid and summary.get("accuracy_metrics_valid") is True if budget_policy else None,
            "asr_metrics_valid":valid and summary.get("asr_metrics_valid") is True if budget_policy else None}

def environment(command, credentials):
    import urllib.request
    services = read(credentials)
    by_url = {s["base_url"].rstrip("/"):s for s in services.values()}
    def service(name, fallback=None):
        url = flag(command, name, fallback)
        if not url or url.rstrip("/") not in by_url:
            raise ValueError("Missing credentials for configured service: " + str(url))
        return by_url[url.rstrip("/")]
    task = service("--chat-base-url")
    embed = service("--embed-base-url")
    judge = service("--full-judge-base-url", flag(command, "--pattern-judge-base-url", task["base_url"]))
    for item in {s["base_url"]:s for s in (task, embed, judge)}.values():
        request = urllib.request.Request(item["base_url"].rstrip("/")+"/models",
                    headers={"Authorization":"Bearer "+item.get("api_key","")})
        with urllib.request.urlopen(request, timeout=15) as reply:
            if item["model"] not in [x["id"] for x in json.load(reply)["data"]]:
                raise ValueError("Configured model is not served")
    return dict(os.environ, CHAT_API_KEY=task.get("api_key",""),
        OPENAI_API_KEY=judge.get("api_key",""), FULL_BASELINE_API_KEY=judge.get("api_key",""),
        SAFEGUARD_OPENAI_API_KEY=judge.get("api_key",""), EMBED_API_KEY=embed.get("api_key",""),
        MAPLE_SERVICE_CREDENTIALS=str(credentials), PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AMEMGUARD_TRANSPORT_MAX_ATTEMPTS="3", MAPLE_TRANSPORT_MAX_ATTEMPTS="3")

def run_job(job, env, handle, expected_ids):
    directory = Path(job["directory"])
    resuming = "--resume-task-checkpoint" in job["command"]
    if resuming:
        if running_directory(directory):
            raise RuntimeError("Benchmark worker already running: "+str(directory))
        if not directory.is_dir():
            raise ValueError("Missing checkpoint working directory")
        # Preserve controller metadata/logs; the worker separately archives and
        # rolls back memory, traces and API journals under its checkpoint lock.
        audit = directory / ("resume-audit-"+str(time.time_ns()))
        audit.mkdir(mode=0o700)
        for name in ("run.json","run.log"):
            if (directory/name).exists():
                (directory/name).replace(audit/name)
    else:
        directory.mkdir(parents=True, exist_ok=False)
    if not resuming and "maple_guard.infa_memlink_eval" in job["command"]:
        (directory/"trace.jsonl").symlink_to(job["method"]+".trace.jsonl")
        (directory/"trace.summary.json").symlink_to("summary.json")
    env = dict(env, MAPLE_CALL_LOG=str(directory/"api-calls.jsonl"),
               MEMOS_BASE_PATH=str(directory/"memos-runtime"), MAPLE_FAIL_ON_TRUNCATION="1")
    job.update(status="running", started_at=time.time(), source_root=str(ROOT),
               memos_base_path=str(directory/"memos-runtime"))
    save(directory/"run.json", job)
    with (directory/"run.log").open("w") as log:
        proc = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
             stdout=log, stderr=subprocess.STDOUT, pass_fds=(handle.fileno(),))
        job["pid"] = proc.pid
        save(directory/"run.json", job)
        code = proc.wait()
    result = validate_result(directory, code, 200, expected_ids)
    status = "completed" if result["valid"] and not result["transport_errors"] else (
             "completed_needs_transport_review" if result["valid"] else "failed")
    job.update(status=status, exit_code=code, finished_at=time.time(), **result)
    save(directory/"run.json", job)
    return job

def run_lane(plan, lane):
    root = Path(plan["recovery_root"])/lane["id"]
    root.mkdir(parents=True, exist_ok=True)
    with lane_lock(Path(lane.get("lock_path", root/"lane.lock"))) as handle:
        state_path = root/"state.json"
        state = read(state_path) if state_path.exists() else {"completed_jobs":{}, "attempts":{}}
        state.update(status="running", controller_pid=os.getpid())
        save(state_path, state)
        try:
            for original in lane["jobs"]:
                key = original["run_id"]
                if key in state["completed_jobs"]:
                    result_dir = Path(state["completed_jobs"][key])
                    result = read(result_dir/"run.json")
                    if not validate_result(result_dir, result["exit_code"], 200,
                                           original.get("expected_task_ids"))["valid"]:
                        raise ValueError("Previously completed recovery result changed")
                    continue
                previous = state.get("current_directory")
                for directory in (original["directory"], previous):
                    if directory and running_directory(directory):
                        raise RuntimeError("Benchmark worker already running: "+directory)
                if state.get("current_run") == key and previous and Path(previous, "run.json").exists():
                    prior = read(Path(previous)/"run.json")
                    if (prior.get("status") == "completed" and
                            validate_result(previous, prior["exit_code"], 200,
                                            original.get("expected_task_ids"))["valid"]):
                        state["completed_jobs"][key] = previous
                        save(state_path, state)
                        continue
                attempts = state["attempts"].get(key, 0)
                if attempts >= 3:
                    raise RuntimeError("Three attempts exhausted; inspect the failure before resetting recovery")
                while attempts < 3:
                    # Verify endpoints before allocating another output directory.
                    env = environment(original["command"], lane["credentials"])
                    attempts += 1
                    job = None
                    previous = state.get("current_directory") if state.get("current_run") == key else None
                    previous = previous or original["directory"]
                    if previous and Path(previous,"run.json").is_file():
                        prior = read(Path(previous)/"run.json")
                        candidate = resume_job(prior,evaluator_recovery=lane.get("evaluator_recovery"))
                        if lane.get("evaluator_recovery") and candidate is None:
                            raise ValueError("Evaluator recovery requires the existing checkpoint; task replay is forbidden")
                        if candidate is not None:
                            if (prior.get("status") not in {"running","prepared"} and not retryable_failure(previous)
                                    and not (candidate.get("evaluator_source_recovery") and diagnosed_semantic_evaluator_failure(previous))):
                                raise RuntimeError("Checkpoint retained; diagnose deterministic failure before resuming: "+previous)
                            for option,value in lane.get("overrides", {}).items():
                                if str(flag(candidate["command"],option)) != str(value):
                                    raise ValueError("Checkpoint configuration change requires explicit audited resume: "+option)
                            job = candidate
                    if job is None:
                        dest = root/"runs"/(key+"_recovery"+str(attempts))
                        job = rewrite_job(original, dest, ROOT, lane.get("overrides", {}))
                    else:
                        dest = Path(job["directory"])
                    state["attempts"][key] = attempts
                    state.update(current_run=key, current_seed=original["seed"], current_directory=str(dest))
                    save(state_path, state)
                    result = run_job(job, env, handle, original.get("expected_task_ids"))
                    if result["status"] == "completed":
                        state["completed_jobs"][key] = str(dest)
                        save(state_path, state)
                        break
                    if not retryable_failure(dest) or attempts == 3:
                        raise RuntimeError("Recovery run needs inspection: "+str(dest))
                    time.sleep(30 * attempts)
            state.update(status="completed", finished_at=time.time())
        except Exception as exc:
            state.update(status="needs_attention", error_type=type(exc).__name__,
                         error=str(exc), finished_at=time.time())
        save(state_path, state)
        return state

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    plan_path = Path(args.plan).resolve()
    plan = read(plan_path)
    if plan["hostname"] != socket.gethostname():
        raise ValueError("Recovery plan belongs to another execution host")
    source = subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    if source != plan["source_commit"]:
        raise ValueError("Frozen source commit differs from recovery plan")
    for path, expected in plan.get("asset_hashes", {}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Recovery input hash changed: "+path)
    with lane_lock(plan_path.with_suffix(".lock")):
        state = {"status":"running", "pid":os.getpid(), "host":socket.gethostname(), "started_at":time.time()}
        save(plan_path.with_suffix(".state.json"), state)
        with ThreadPoolExecutor(max_workers=len(plan["lanes"])) as pool:
            results = list(pool.map(lambda lane:run_lane(plan,lane), plan["lanes"]))
        state.update(status="completed" if all(r["status"]=="completed" for r in results)
                     else "needs_attention", finished_at=time.time(), lanes=results)
        save(plan_path.with_suffix(".state.json"), state)

if __name__ == "__main__":
    main()
