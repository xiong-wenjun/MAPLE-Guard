"""Freeze and execute AppWorld-derived action-selection runs on authorized remote services.

This is not the native AppWorld executor. Raw datasets, credentials and run artifacts
must remain outside Git. Each run receives a fresh memory directory and sidecar.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from maple_guard.benchmarks.benchmark_bundle import load_bundle
MAIN=("no_defense_memrl","maple_guard","provenance_acl","agentsafe_full","infa_guard_full",
      "agentxposed_full_guide","amemguard_full","piguard_retrieval","challenger","gsafeguard","guardian")
MECHANISM=("maple_guard_retrieval_only","piguard_lifecycle","agentxposed_full_kick")
IDENTITY_AUDIT=("inspector",)
GATES=("maple_guard_no_write","maple_guard_no_retrieval","maple_guard_no_promotion","maple_guard_no_cross_agent")

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",required=True)
    p.add_argument("--services",required=True,help="0600 credential file; contents never written to manifests")
    p.add_argument("--task-service",default="inference1")
    p.add_argument("--judge-service",default="inference1")
    p.add_argument("--backbone",choices=("qwen","gemma31b"),default="qwen")
    p.add_argument("--profile",choices=("paper-code","strict"),default="strict")
    p.add_argument("--phase",choices=("smoke","pilot"),default="smoke")
    p.add_argument("--methods",nargs="+",default=list(MAIN+MECHANISM+IDENTITY_AUDIT))
    p.add_argument("--seeds",nargs="+",type=int,default=[42])
    p.add_argument("--tasks",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--run-root",required=True)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--agentsafe-policy",default="")
    p.add_argument("--agentsafe-criteria",default="")
    p.add_argument("--agentsafe-threshold",type=float)
    p.add_argument("--infa-checkpoint",default="")
    p.add_argument("--minilm-model",default="")
    p.add_argument("--guardian-source",default="")
    return p.parse_args()

def extras(args, method, run_id):
    refs=Path("/mnt/public/data/wj/baseline-references")
    if method=="agentsafe_full":
        if not (args.agentsafe_policy and args.agentsafe_criteria and args.agentsafe_threshold is not None):
            return [],"AgentSafe policy, criteria and independently frozen threshold are required"
        return ["--agentsafe-policy-file",args.agentsafe_policy,"--agentsafe-criteria-file",args.agentsafe_criteria,
                "--agentsafe-threshold",str(args.agentsafe_threshold)],None
    if method=="infa_guard_full":
        if not args.infa_checkpoint:
            return [],"Native dual-head INFA checkpoint missing; supplied G-Safeguard weights retained separately"
        return ["--infa-code-dir",str(refs/"INFA-Guard"),"--infa-checkpoint",args.infa_checkpoint,
                "--infa-embedding-model",args.minilm_model,"--infa-protocol","released",
                "--infa-detector-mode","profile","--infa-correction-transport","functional"],None
    if method.startswith("agentxposed_full_"):
        return ["--agentxposed-code-dir",str(refs/"AgentXposed"),"--agentxposed-protocol","released_minimal_fix"],None
    if method=="amemguard_full":
        return ["--amemguard-experiment-id",run_id,"--amemguard-top-k","4","--amemguard-lesson-top-k","4"],None
    if method.startswith("piguard_"):
        return ["--piguard-model",str(refs/"PIGuard-hf-dd78b24"),"--piguard-revision",
                "dd78b24e330193a22d2293ac66922dd4f982f563","--piguard-max-length","512",
                "--piguard-threshold","0.5","--piguard-device","cpu"],None
    if method=="gsafeguard":
        if not args.minilm_model: return [],"Official MiniLM encoder asset not yet validated"
        return ["--official-defense-embedding-model",args.minilm_model],None
    if method=="guardian":
        if not args.guardian_source: return [],"GUARDIAN source/checkpoints/encoder absent; graph-adapter mapping requires verification"
        return ["--official-defense-guardian-code-dir",args.guardian_source],None
    return [],None

def source_fingerprint():
    digest=hashlib.sha256()
    for folder in ("maple_guard","evaluate","configs","prompts","tools"):
        for path in sorted((ROOT/folder).rglob("*")):
            if path.is_file() and path.suffix in (".py",".yaml",".json",".sh"):
                digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()

def build_job(args, method, seed, services):
    if method not in MAIN+MECHANISM+GATES+IDENTITY_AUDIT: raise ValueError("Unrecognized experiment method: "+method)
    if args.profile=="paper-code" and method not in ("maple_guard","no_defense_memrl"):
        raise ValueError("Paper-code bridge currently restricted to original MAPLE and No Defense")
    task=services[args.task_service];judge=services[args.judge_service];embedding=services["embedding"]
    run_id=f"{args.phase}_{args.profile}_{args.backbone}_{method}_s{seed}"
    run_dir=Path(args.run_root)/run_id
    extra,blocked=extras(args,method,run_id)
    cmd=[sys.executable,"-B",str(ROOT/"tools/run_instrumented.py"),"maple_guard.run_appworld","--config","configs/appworld_star.yaml",
         "--benchmark-bundle",str(Path(args.bundle).resolve()),"--method",method,"--tasks",str(args.tasks),
         "--seed",str(seed),"--agents","8","--rounds","3",
         "--chat-base-url",task["base_url"],"--chat-model",task["model"],
         "--embed-base-url",embedding["base_url"],"--embed-model",embedding["model"],
         "--full-judge-base-url",judge["base_url"],"--full-judge-model",judge["model"],
         "--safeguard-base-url",judge["base_url"],"--safeguard-model",judge["model"],
         "--pattern-judge-base-url",judge["base_url"],"--pattern-judge-model",judge["model"],
         "--trace-id",run_id,"--memory-run-id",run_id,"--baseline-experiment-id",run_id,
         "--memory-store-dir",str(run_dir/"memory"),"--baseline-state-path",str(run_dir/"baseline-state.json"),
         "--out",str(run_dir/"trace.jsonl"),"--log-every","1"]
    if args.profile=="strict":
        cmd += ["--strict-comparison","--peer-communication","--no-exclude-attackers-from-final-vote",
                "--no-enable-causal-mir","--benign-shared-promotion-policy","accepted_retrieved_private",
                "--memory-topology","brokered-shared","--top-k-memory","3","--disable-chat-thinking",
                "--asr-metric","target_hit"]
    if args.phase=="smoke": cmd+=["--warmup-tasks","0","--malicious-activation-rate","0.25"]
    return {"run_id":run_id,"method":method,"table_role":"main" if method in MAIN else ("identity_audit" if method in IDENTITY_AUDIT else "mechanism"),
            "seed":seed,"phase":args.phase,"profile":args.profile,"directory":str(run_dir),
            "command":cmd+extra,"status":"blocked" if blocked else "prepared","reason":blocked}

def save(path,value):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temp.replace(path)

def execute_job(job, env, expected_count):
    run_dir=Path(job["directory"]);run_dir.mkdir(parents=True,exist_ok=False)
    save(run_dir/"run.json",job)
    job["status"]="running";job["started_at"]=time.time()
    job["memos_base_path"]=str(run_dir/"memos-runtime")
    save(run_dir/"run.json",job)
    task_env=dict(env,MAPLE_CALL_LOG=str(run_dir/"api-calls.jsonl"),
                  MEMOS_BASE_PATH=str(run_dir/"memos-runtime"),
                  MAPLE_FAIL_ON_TRUNCATION="1" if job["profile"]=="strict" else "0")
    with (run_dir/"run.log").open("w") as log:
        proc=subprocess.Popen(job["command"],cwd=ROOT,env=task_env,stdout=log,stderr=subprocess.STDOUT)
        job["pid"]=proc.pid;save(run_dir/"run.json",job)
        code=proc.wait()
    job["exit_code"]=code;job["finished_at"]=time.time()
    summary=run_dir/"trace.summary.json"
    trace=run_dir/"trace.jsonl"
    count=sum(1 for line in trace.open() if line.strip()) if trace.exists() else 0
    job["completed_tasks"]=count
    job["status"]="completed" if code==0 and summary.exists() and count==expected_count else "failed"
    if code==0 and job["status"]=="failed":job["reason"]="Missing summary or unexpected task count"
    calls=run_dir/"api-calls.jsonl"
    if calls.exists():
        events=[json.loads(line) for line in calls.read_text().splitlines() if line.strip()]
        chat=[e for e in events if e["endpoint"].endswith("/chat/completions")]
        job["api_usage"]={"requests":len(events),"chat_requests":len(chat),
                          "truncated_chat_requests":sum("length" in e.get("finish_reasons",[]) for e in chat),
                          "chat_requests_without_final_content":sum(any(v is False for v in e.get("final_content_present",[])) for e in chat),
                          "http_errors":sum(e.get("http_status",0)>=400 or "error_type" in e for e in events)}
        if job["status"]=="completed" and (job["api_usage"]["truncated_chat_requests"] or job["api_usage"]["chat_requests_without_final_content"]):
            job["status"]="completed_with_response_warnings"
            job["reason"]="Not eligible for main-table reporting until truncation/final-content warnings are resolved"
    save(run_dir/"run.json",job)
    return job

def validate_services(services, names):
    import requests
    report={}
    for name in dict.fromkeys(names):
        service=services[name]
        response=requests.get(service["base_url"].rstrip("/")+"/models",
                              headers={"Authorization":"Bearer "+service.get("api_key","")},timeout=15)
        response.raise_for_status()
        ids=[row["id"] for row in response.json()["data"]]
        if service["model"] not in ids:
            raise RuntimeError("Configured model is not served by "+name)
        report[name]={"verified_model":service["model"],"served_ids":ids,"checked_at":time.time()}
    return report

def main():
    args=arguments()
    if args.phase=="pilot" and args.tasks!=200: raise ValueError("Fixed-subset pilot must contain all 200 tasks")
    if not 1<=args.workers<=4: raise ValueError("Use 1–4 workers per runner")
    services_path=Path(args.services)
    if services_path.stat().st_mode & 0o077: raise ValueError("Credential file must not be group/world accessible")
    services=json.loads(services_path.read_text())
    task=services[args.task_service];judge=services[args.judge_service];embed=services["embedding"]
    model=task["model"].lower()
    if args.backbone=="gemma31b" and not ("gemma" in model and "31b" in model):
        raise ValueError("Gemma31B arm must use a verified Gemma31B service, never Qwen or E4B")
    if args.backbone=="qwen" and "qwen" not in model:raise ValueError("Qwen arm requires Qwen service")
    if "qwen3.5" not in judge["model"].lower():raise ValueError("Freeze Qwen3.5 judge across both backbones")
    _,data_manifest=load_bundle(args.bundle,"appworld")
    run_root=Path(args.run_root)
    if run_root.exists():raise ValueError("Fresh run root required; never reuse memory across experiments")
    run_root.mkdir(parents=True)
    jobs=[build_job(args,method,seed,services) for seed in args.seeds for method in args.methods]
    public={name:{k:v for k,v in service.items() if k in ("base_url","model","source_host")} for name,service in services.items()}
    manifest={"evaluation_protocol":"appworld_action_selection_proxy","native_execution":False,
              "phase":args.phase,"profile":args.profile,"task_count":args.tasks,
              "repository_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "source_sha256":source_fingerprint(),"python":sys.executable,
              "runtime_versions":{name:importlib.metadata.version(name) for name in
                  ("torch","transformers","numpy","requests") if importlib.util.find_spec(name)},
              "dataset":data_manifest,"services":public,"jobs":jobs,
              "limitations":["One star topology/seed pilot is not the original three-topology 15-run table.",
                             "Reference-outcome memory feedback remains enabled and shared across strict arms.",
                             "Provided specs only: no native AppWorld tool execution or database evaluation.",
                             "Smoke phase changes warmup/poison rate and is not a benchmark result."]}
    save(run_root/"matrix.json",manifest)
    (run_root/"appworld_star.snapshot.yaml").write_bytes((ROOT/"configs/appworld_star.yaml").read_bytes())
    print(json.dumps({"run_root":str(run_root),"jobs":[{k:j[k] for k in ("method","status","reason")} for j in jobs]},ensure_ascii=False),flush=True)
    if not args.execute:return 0
    try:
        manifest["service_verification"]=validate_services(services,[args.task_service,args.judge_service,"embedding"])
    except Exception as exc:
        manifest["preflight_failure"]={"type":type(exc).__name__,"message":str(exc)}
        save(run_root/"matrix.json",manifest)
        raise
    save(run_root/"matrix.json",manifest)
    env=dict(os.environ,CHAT_API_KEY=task.get("api_key",""),OPENAI_API_KEY=judge.get("api_key",""),
             FULL_BASELINE_API_KEY=judge.get("api_key",""),SAFEGUARD_OPENAI_API_KEY=judge.get("api_key",""),
             EMBED_API_KEY=embed.get("api_key",""),MAPLE_SERVICE_CREDENTIALS=str(services_path.resolve()),
             PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
             HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(execute_job,j,env,min(args.tasks,data_manifest["total_cases"])):j for j in jobs if j["status"]=="prepared"}
        for future in concurrent.futures.as_completed(pending):
            job=future.result()
            save(run_root/"matrix.json",manifest)
            print(json.dumps({k:job[k] for k in ("run_id","status","completed_tasks","exit_code")}),flush=True)
    return int(any(j["status"]=="failed" for j in jobs))
if __name__=="__main__":
    os.umask(0o077)
    raise SystemExit(main())
