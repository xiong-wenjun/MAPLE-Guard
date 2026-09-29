"""Bounded execution of a frozen AppWorld matrix; preserve existing pilot ownership.

Run one dispatcher per execution host. The frozen engine, commands, dataset and
config hashes are checked before execution. Blocked methods never use fallbacks.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import time

CAMPAIGNS = (
    "appworld-20260929-campaign-r3",
    "appworld-20260929-qwen-inference2-r1",
    "appworld-20260929-agentxposed-r4",
    "appworld-20260929-gemma-it-r1",
)
BAD = {"failed", "completed_with_response_warnings", "orphaned", "launch_failed"}

def read_json(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temp.replace(path)

@contextlib.contextmanager
def acquire_lock(path):
    with Path(path).open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield handle
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)

def process_matches(pid, run_id):
    try:
        command=Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
        return bool(run_id) and run_id.encode() in command
    except (OSError, ValueError, TypeError):
        return False

def journal_clean(folder):
    try:
        lines=(Path(folder)/"api-calls.jsonl").read_text().splitlines()
        if not lines:return False
        events=[json.loads(line) for line in lines if line.strip()]
        return bool(events) and all(
            not e.get("invalid_for_benchmark") and not e.get("error_type")
            and (e.get("http_status") or 0)<400
            and "length" not in e.get("finish_reasons",[])
            and not any(v is False for v in e.get("final_content_present",[]))
            for e in events)
    except (OSError,ValueError):
        return False

def trace_count(folder):
    try:
        with (Path(folder)/"trace.jsonl").open() as handle:
            return sum(bool(line.strip()) for line in handle)
    except OSError:
        return 0

def probe_ready(path):
    """Readiness evidence only; a live base pilot is never a completed result."""
    try:
        row=read_json(path);folder=Path(path).parent
        if not journal_clean(folder):return False
        count=trace_count(folder)
        if row.get("status")=="completed":
            expected=2 if row.get("phase")=="smoke" else 200
            return (row.get("exit_code")==0 and row.get("completed_tasks")==expected
                    and count==expected and (folder/"trace.summary.json").exists())
        return (row.get("status")=="running" and row.get("phase")=="pilot"
                and row.get("method") in {"maple_guard","no_defense_memrl"}
                and count>=2)
    except (OSError,ValueError):
        return False

def disposition(model, job, reserved, ready, family_failed, directory_exists):
    if job.get("status")=="blocked":return "blocked"
    if (model,job["method"],job["topology"],job["seed"]) in reserved:return "reserved"
    if directory_exists:return "existing"
    if family_failed:return "held_after_failure"
    if not ready:return "waiting_probe"
    return "queued"

def record_launch_failure(plan, host, model, job, exc):
    path=Path(plan)/f"dispatch-errors-{host}.json"
    failures=read_json(path) if path.exists() else {}
    failures[job["run_id"]]={"model":model,"method":job["method"],
                             "error_type":type(exc).__name__,"at":time.time()}
    save(path,failures)
    return failures

def available_slots(total_limit, active_count, new_limit, own_active):
    return max(0,min(total_limit-active_count,new_limit-own_active))

def backbone(row):
    cmd=row.get("command",[])
    if "--chat-model" in cmd:
        model=cmd[cmd.index("--chat-model")+1].lower()
        return "gemma31b" if "gemma" in model else "qwen"
    return "gemma31b" if "gemma31b" in row.get("run_id","") else "qwen"

def current_runs(base, plan):
    paths=[]
    for campaign in CAMPAIGNS:
        paths.extend((base/campaign).glob("*/*/run.json"))
    paths.extend(plan.glob("*/*/run.json"))
    rows=[]
    for p in sorted(set(paths)):
        row=read_json(p)
        rows.append((p,row))
    return rows

def load_engine(path):
    spec=importlib.util.spec_from_file_location("frozen_matrix_engine",path/"tools/run_appworld_matrix.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify_shard(manifest, engine, services, task_service, judge_service):
    if manifest["task_count"]!=200 or manifest["profile"]!="strict":
        raise ValueError("Dispatcher requires the strict fixed 200-task plan")
    if manifest["source_sha256"]!=engine.source_fingerprint():
        raise ValueError("Frozen engine fingerprint changed")
    dataset=manifest["dataset"]
    if hashlib.sha256(Path(dataset["source_file"]).read_bytes()).hexdigest()!=dataset["source_sha256"]:
        raise ValueError("Dataset hash changed")
    for config in manifest["config_snapshots"].values():
        if hashlib.sha256(Path(config["path"]).read_bytes()).hexdigest()!=config["sha256"]:
            raise ValueError("Frozen topology config changed")
    for name in {task_service,judge_service,"embedding"}:
        for field in ("base_url","model"):
            if manifest["services"][name][field]!=services[name][field]:
                raise ValueError("Service identity differs from frozen plan")
    seen=set()
    for job in manifest["jobs"]:
        key=(job["method"],job["topology"],job["seed"])
        if key in seen:raise ValueError("Duplicate matrix cell")
        seen.add(key)
        if job["table_role"]!="main":raise ValueError("Main plan contains non-main method")
        if Path(job["command"][2])!=engine.ROOT/"tools/run_instrumented.py":
            raise ValueError("Job command does not use the frozen engine")

def environment(services, task_name, judge_name, credentials):
    task=services[task_name];judge=services[judge_name];embed=services["embedding"]
    return dict(os.environ,CHAT_API_KEY=task.get("api_key",""),
                OPENAI_API_KEY=judge.get("api_key",""),FULL_BASELINE_API_KEY=judge.get("api_key",""),
                SAFEGUARD_OPENAI_API_KEY=judge.get("api_key",""),EMBED_API_KEY=embed.get("api_key",""),
                MAPLE_SERVICE_CREDENTIALS=str(credentials),PYTHONDONTWRITEBYTECODE="1",
                CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-root",type=Path,required=True)
    p.add_argument("--engine-root",type=Path,required=True)
    p.add_argument("--services",type=Path,required=True)
    p.add_argument("--host",choices=["inference1","inference2"],required=True)
    p.add_argument("--expected-hostname",required=True)
    p.add_argument("--max-total-workers",type=int,required=True)
    p.add_argument("--max-new-workers",type=int,default=2)
    p.add_argument("--max-new-gemma",type=int,default=1)
    p.add_argument("--poll-seconds",type=float,default=15)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if socket.gethostname()!=args.expected_hostname:raise ValueError("Wrong execution host")
    if not 1<=args.max_new_workers<=4 or not 1<=args.max_new_gemma<=args.max_new_workers:
        raise ValueError("Use 1-4 new workers and a valid Gemma cap")
    if args.max_total_workers<args.max_new_workers or args.poll_seconds<5:
        raise ValueError("Invalid worker budget or polling interval")
    if args.services.stat().st_mode & 0o077:raise ValueError("Credential file must be private")
    plan=args.plan_root.resolve();base=plan.parent
    engine=load_engine(args.engine_root.resolve())
    services=read_json(args.services)
    reservations=read_json(plan/"reserved-star42.json")
    reserved={(r["model"],r["method"],r["topology"],r["seed"]) for r in reservations}
    shards={}
    for model in ("qwen","gemma31b"):
        manifest=read_json(plan/f"{model}-{args.host}"/"matrix.json")
        task=args.host if model=="qwen" else "inference3"
        verify_shard(manifest,engine,services,task,args.host)
        shards[model]=manifest
    ready_cache=set()
    def readiness(rows):
        for path,row in rows:
            model=backbone(row);key=(model,row["method"])
            if key in ready_cache:continue
            # Ignore superseded AgentXposed probes in r3.
            if row["method"].startswith("agentxposed") and "campaign-r3" in str(path):continue
            if probe_ready(path):ready_cache.add(key)
        return ready_cache
    def observed():
        rows=current_runs(base,plan)
        failed={(backbone(r),r["method"]) for _,r in rows
                if r.get("phase")=="pilot" and r.get("status") in BAD}
        return rows,failed
    status_path=plan/f"execution-{args.host}.json"
    with acquire_lock(plan/f"dispatch-{args.host}.lock"):
        if args.execute:
            verification=engine.validate_services(services,[args.host,"inference3","embedding"])
        else:verification={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_new_workers) as pool:
            futures={};launched=collections.Counter();known_dead=set()
            error_path=plan/f"dispatch-errors-{args.host}.json"
            launch_errors=read_json(error_path) if error_path.exists() else {}
            while True:
                for future,(model,job) in list(futures.items()):
                    if not future.done():continue
                    try:
                        result=future.result()
                        if result["status"]=="completed" and not journal_clean(result["directory"]):
                            result.update(status="failed",reason="API journal failed strict validation")
                            save(Path(result["directory"])/"run.json",result)
                    except Exception as exc:
                        launch_errors=record_launch_failure(plan,args.host,model,job,exc)
                    del futures[future]
                rows,failed=observed();ready=readiness(rows)
                live={}
                for path,row in rows:
                    if row.get("status")=="running" and process_matches(row.get("pid"),row.get("run_id")):
                        live[row["pid"]]=(path,row)
                # Only our own shards are local; never infer another host's PID state.
                for model,manifest in shards.items():
                    for job in manifest["jobs"]:
                        f=Path(job["directory"])/"run.json"
                        if not f.exists():continue
                        row=read_json(f)
                        if (row.get("status")=="running" and row.get("pid")
                                and not process_matches(row["pid"],row["run_id"])
                                and not any(j["run_id"]==row["run_id"] for _,j in futures.values())):
                            known_dead.add((model,job["method"]))
                failed |= known_dead | {(r["model"],r["method"]) for r in launch_errors.values()}
                decisions={};candidates=[]
                for model,manifest in shards.items():
                    for job in manifest["jobs"]:
                        folder=Path(job["directory"])
                        state=disposition(model,job,reserved,(model,job["method"]) in ready,
                                          (model,job["method"]) in failed,folder.exists())
                        if state=="existing":
                            f=folder/"run.json"
                            state=read_json(f).get("status","existing") if f.exists() else "existing"
                        if job["run_id"] in launch_errors:state="launch_failed"
                        decisions[job["run_id"]]=state
                        if state=="queued":candidates.append((model,job))
                # Include just-submitted futures whose child PID is not yet published.
                own_running_ids={j["run_id"] for _,j in futures.values()}
                unpublished=sum(run_id not in {r["run_id"] for _,r in live.values()} for run_id in own_running_ids)
                active=len(live)+unpublished
                slots=available_slots(args.max_total_workers,active,args.max_new_workers,len(futures))
                gemma_active=sum(m=="gemma31b" for m,_ in futures.values())
                # On a restart also respect surviving orphan workers' concurrency budget.
                surviving=[(m,j) for m,man in shards.items() for j in man["jobs"]
                           if j["run_id"] not in own_running_ids and any(r["run_id"]==j["run_id"] for _,r in live.values())]
                slots=min(slots,max(0,args.max_new_workers-len(futures)-len(surviving)))
                gemma_active+=sum(m=="gemma31b" for m,_ in surviving)
                selected=[]
                for _ in range(slots):
                    eligible=[(m,j) for m,j in candidates if (m!="gemma31b" or gemma_active<args.max_new_gemma)]
                    if not eligible:break
                    model,job=min(eligible,key=lambda pair:(launched[pair[0]],pair[1]["seed"],
                        ("chain","tree","star","random").index(pair[1]["topology"]),
                        engine.MAIN.index(pair[1]["method"])))
                    selected.append((model,job));candidates.remove((model,job));launched[model]+=1
                    if model=="gemma31b":gemma_active+=1
                report={"status":"running" if args.execute else "dry_run","pid":os.getpid(),
                        "execution_host":args.host,"hostname":socket.gethostname(),"updated_at":time.time(),
                        "max_total_workers":args.max_total_workers,"max_new_workers":args.max_new_workers,
                        "max_new_gemma":args.max_new_gemma,"observed_live_workers":len(live),
                        "active_matrix_workers":len(futures)+len(surviving),
                        "ready_families":[list(k) for k in sorted(ready)],
                        "held_families":[list(k) for k in sorted(failed)],
                        "launch_errors":launch_errors,
                        "counts":dict(collections.Counter(decisions.values())),
                        "selected":[j["run_id"] for _,j in selected],
                        "service_verification":verification,
                        "engine_root":str(engine.ROOT),"results_validated_by_this_dispatcher":False}
                if not args.execute:
                    print(json.dumps(report,ensure_ascii=False));return 0
                for model,job in selected:
                    task=args.host if model=="qwen" else "inference3"
                    env=environment(services,task,args.host,args.services.resolve())
                    futures[pool.submit(engine.execute_job,dict(job),env,200)]=(model,job)
                    print(json.dumps({"launched":job["run_id"],"host":args.host}),flush=True)
                report["active_matrix_workers"]=len(futures)+len(surviving)
                save(status_path,report)
                pending=any(v in {"queued","waiting_probe","running"} for v in decisions.values())
                if not pending and not futures and not surviving:
                    report["status"]="drained_with_blockers" if report["counts"].get("blocked") or failed else "drained"
                    save(status_path,report);return 0
                time.sleep(args.poll_seconds)

if __name__=="__main__":
    os.umask(0o077)
    raise SystemExit(main())
