"""Durable training-generation journals for two explicit Qwen adaptations.

Strict recovery accepts only stop replies and escalates budgets. Released-budget
mode consumes nonempty stop/length content at the native 1024-token cap. Neither
policy changes evaluation response rules. Native sampling and labels stay intact.
"""
from __future__ import annotations
import asyncio
import hashlib
import itertools
import json
import os
from pathlib import Path
from datetime import datetime, timezone

def payload_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
                                     ensure_ascii=False).encode()).hexdigest()

def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w") as handle:
        json.dump(value,handle,ensure_ascii=False)
        handle.write("\n");handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor=os.open(str(path.parent),os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)

def policy_bounds(policy=None):
    policy = policy or {"token_budgets":[1024,2048,4096,8192]}
    budgets = list(policy["token_budgets"])
    reasons = list(policy.get("accepted_finish_reasons", ["stop"]))
    if (budgets, reasons) not in (([1024,2048,4096,8192],["stop"]), ([1024],["stop","length"])):
        raise ValueError("Unsupported declared INFA generation policy")
    return budgets, reasons


def complete(response, accepted_finish_reasons=("stop",)):
    choices=response.choices
    if len(choices)!=1:return False
    item=choices[0]
    return item.finish_reason in accepted_finish_reasons and isinstance(item.message.content,str) and bool(item.message.content.strip())

def make_audited_create(original, audit_path, request_overrides, recovery_policy, cache_dir):
    from openai.types.chat import ChatCompletion
    from openai import APIConnectionError,APITimeoutError,APIStatusError
    budgets,reasons=policy_bounds(recovery_policy)
    attempts=int(recovery_policy["transport_attempts"])
    if not 1<=attempts<=3:
        raise ValueError("Unsupported declared INFA recovery bounds")
    backoff=float(recovery_policy["transport_backoff_seconds"])
    if not 0<=backoff<=30:raise ValueError("Invalid recovery backoff")
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    audit=Path(audit_path);audit.parent.mkdir(parents=True,exist_ok=True)
    sequence=itertools.count()
    overrides=request_overrides or {}
    def record(row):
        row["timestamp_utc"]=datetime.now(timezone.utc).isoformat()
        with audit.open("a") as handle:
            handle.write(json.dumps(row,ensure_ascii=False)+"\n")
            handle.flush();os.fsync(handle.fileno())
    async def audited(self,*args,**kwargs):
        ordinal=next(sequence)
        if args:raise ValueError("Recovery requires explicit keyword API arguments")
        if set(kwargs)&set(overrides):
            raise ValueError("Declared request adaptation would overwrite an upstream argument")
        request={**kwargs,**overrides}
        if request.get("max_tokens")!=budgets[0]:
            raise ValueError("Upstream max_tokens differs from pinned 1024-token request")
        request_hash=payload_hash(request)
        path=cache/f"{ordinal:06d}.json"
        base={"ordinal":ordinal,"request_sha256":request_hash,"model":request.get("model")}
        if path.exists():
            row=json.loads(path.read_text())
            if row.get("ordinal")!=ordinal or row.get("request_sha256")!=request_hash:
                raise ValueError("Cached INFA request changed; refusing mismatched replay")
            if payload_hash(row.get("response"))!=row.get("response_sha256"):
                raise ValueError("Cached response hash mismatch")
            response=ChatCompletion.model_validate(row["response"])
            if not complete(response,reasons) or row.get("accepted_max_tokens") not in budgets:
                raise ValueError("Cached INFA response is incomplete or outside declared budgets")
            provenance=cache/"import-provenance/provenance.json"
            if provenance.exists():
                expected=json.loads(provenance.read_text()).get("source_files_sha256",{}).get(path.name)
                if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError("Imported journal source hash mismatch")
            record({**base,"event":"replay","finish_reason":response.choices[0].finish_reason,
                    "content_chars":len(response.choices[0].message.content),
                    "max_tokens":row["accepted_max_tokens"]})
            return response
        for budget in budgets:
            current={**request,"max_tokens":budget}
            for attempt in range(1,attempts+1):
                try:
                    response=await original(self,**current)
                    break
                except Exception as error:
                    status=getattr(error,"status_code",None)
                    retryable=isinstance(error,(APIConnectionError,APITimeoutError)) or (
                        isinstance(error,APIStatusError) and (status==429 or (status is not None and status>=500)))
                    record({**base,"event":"api_error","api_error":type(error).__name__,
                            "http_status":status,"max_tokens":budget,"transport_attempt":attempt})
                    if not retryable or attempt==attempts:raise
                    await asyncio.sleep(backoff*(2**(attempt-1)))
            choice=response.choices[0] if response.choices else None
            content=choice.message.content if choice is not None else None
            reason=choice.finish_reason if choice is not None else None
            accepted=complete(response,reasons)
            record({**base,"event":"response","request_overrides":overrides,
                    "finish_reason":reason,"content_chars":len(content or ""),
                    "completion_tokens":getattr(response.usage,"completion_tokens",None),
                    "prompt_tokens":getattr(response.usage,"prompt_tokens",None),
                    "max_tokens":budget,"temperature":current.get("temperature"),
                    "accepted":accepted,"token_budget_adapted":budget!=budgets[0]})
            if accepted:
                payload=response.model_dump(mode="json")
                atomic_json(path,{**base,"accepted_max_tokens":budget,"response":payload,
                                  "response_sha256":payload_hash(payload)})
                return response
            # Content filtering/refusal/tool responses are not token-budget failures.
            if reason not in ("length","stop",None):
                break
        raise RuntimeError("Incomplete upstream generation under declared response policy; no empty or disallowed reply was accepted")
    return audited

def summarize_journal(directory, expected_count, recovery_policy=None):
    from collections import Counter
    from openai.types.chat import ChatCompletion
    allowed_budgets,reasons=policy_bounds(recovery_policy)
    directory=Path(directory)
    paths=sorted(directory.glob("*.json"))
    if len(paths)!=expected_count or [p.name for p in paths]!=[f"{i:06d}.json" for i in range(expected_count)]:
        raise ValueError("Generation journal reply count or ordinal coverage differs from complete native dialogues")
    digest=hashlib.sha256();budgets=Counter();finish_reasons=Counter()
    for i,path in enumerate(paths):
        value=json.loads(path.read_text())
        if value.get("ordinal")!=i or value.get("response_sha256")!=payload_hash(value.get("response")):
            raise ValueError("Generation journal response hash or ordinal mismatch")
        response=ChatCompletion.model_validate(value["response"])
        if not complete(response,reasons):
            raise ValueError("Generation journal contains incomplete response")
        budget=value.get("accepted_max_tokens")
        if budget not in allowed_budgets:
            raise ValueError("Generation journal token budget not declared")
        if len(value.get("request_sha256",""))!=64:
            raise ValueError("Generation journal request hash missing")
        digest.update(path.name.encode());digest.update(path.read_bytes());budgets[budget]+=1
        finish_reasons[response.choices[0].finish_reason]+=1
    report={"accepted_responses":len(paths),"journal_sha256":digest.hexdigest(),
            "accepted_token_budgets":dict(sorted(budgets.items())),
            "accepted_finish_reasons":dict(sorted(finish_reasons.items()))}
    provenance=directory/"import-provenance/provenance.json"
    if provenance.exists():
        imported=json.loads(provenance.read_text())
        for name,expected in imported["source_files_sha256"].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Imported journal source hash mismatch")
        report["import_provenance_sha256"]=hashlib.sha256(provenance.read_bytes()).hexdigest()
    return report


def import_released_prefix(source, destination):
    """Copy only the consecutive 1024/stop prefix into a fresh journal.

    Later replies may depend on an expanded reply and are deliberately excluded.
    Replay still validates the actual new request hash before using each entry.
    This never mutates the source or a running job.
    """
    from openai.types.chat import ChatCompletion
    source=Path(source).resolve();destination=Path(destination).resolve()
    if not source.is_dir():raise ValueError("Source response journal missing")
    if destination.exists():raise FileExistsError("Prefix import requires a fresh destination journal")
    entries=[];files={};digest=hashlib.sha256();ordinal=0
    while True:
        path=source/f"{ordinal:06d}.json"
        if not path.exists():
            stop_reason="first missing ordinal";break
        raw=path.read_bytes();row=json.loads(raw)
        if row.get("ordinal")!=ordinal or row.get("response_sha256")!=payload_hash(row.get("response")):
            raise ValueError("Import source response hash or ordinal mismatch")
        if len(row.get("request_sha256",""))!=64:
            raise ValueError("Import source request hash missing")
        response=ChatCompletion.model_validate(row["response"])
        if row.get("accepted_max_tokens")!=1024 or not complete(response):
            stop_reason="first reply outside nonempty 1024/stop prefix";break
        entries.append((path.name,raw))
        files[path.name]=hashlib.sha256(raw).hexdigest()
        digest.update(path.name.encode());digest.update(raw);ordinal+=1
    destination.mkdir(parents=True,exist_ok=False)
    for name,raw in entries:
        with (destination/name).open("xb") as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
    report={"schema_version":1,"source_journal":str(source),"destination_journal":str(destination),
        "source_files_sha256":files,"source_prefix_sha256":digest.hexdigest(),
        "imported_responses":len(entries),"next_ordinal":ordinal,"stop_reason":stop_reason,
        "policy":"contiguous nonempty stop responses with accepted_max_tokens=1024 only",
        "request_hash_validation":"required during every replay; imported content is not trusted without a matching new request"}
    atomic_json(destination/"import-provenance/provenance.json",report)
    return report


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Import a 1024/stop prefix into a fresh training journal")
    parser.add_argument("--source-journal",required=True)
    parser.add_argument("--destination-journal",required=True)
    args=parser.parse_args()
    os.umask(0o077)
    print(json.dumps(import_released_prefix(args.source_journal,args.destination_journal),indent=2))
