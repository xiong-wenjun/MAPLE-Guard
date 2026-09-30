"""Run a benchmark module while recording API usage metadata, never prompts or credentials."""
from __future__ import annotations
import json
import os
from pathlib import Path
import runpy
import sys
import time
import uuid
from urllib.parse import urlsplit

class BenchmarkResponseError(SystemExit):
    fatal_for_benchmark=True

def install_metrics(path, fail_on_truncation=False):
    import requests
    from maple_guard import budget_outcomes as budget
    maximum = int(os.getenv("MAPLE_TRANSPORT_MAX_ATTEMPTS", "1"))
    if maximum not in (1, 2, 3):
        raise ValueError("MAPLE_TRANSPORT_MAX_ATTEMPTS must be 1, 2, or 3")
    original=requests.sessions.Session.send
    def send(self,request,**kwargs):
        parsed=urlsplit(request.url)
        if not parsed.path.endswith(("/chat/completions","/embeddings")):
            return original(self,request,**kwargs)
        body=request.body or b""
        try: payload=json.loads(body)
        except (TypeError,ValueError): payload={}
        request_id=uuid.uuid4().hex
        for attempt in range(1, maximum+1):
            started=time.monotonic()
            record={"endpoint":parsed.path,"host":parsed.hostname,"model":payload.get("model"),
                    "max_tokens":payload.get("max_tokens"),"temperature":payload.get("temperature"),
                    "request_bytes":len(body),"message_count":len(payload.get("messages",[])),
                    "timestamp":time.time(),"request_id":request_id,"attempt":attempt,
                    "max_attempts":maximum,"will_retry":False, **budget.request_audit(),
                    "messages_sha256":budget.messages_hash(payload.get("messages", []))}
            invalid_response=False
            try:
                response=original(self,request,**kwargs)
                record["http_status"]=response.status_code
                response._maple_request_id=request_id
                transient=response.status_code==429 or 500<=response.status_code<=599
                if transient and attempt < maximum:
                    record["will_retry"]=True
                    response.close()
                else:
                    try:data=response.json()
                    except ValueError:data={}
                    if isinstance(data,dict):
                        record["usage"]=data.get("usage")
                        choices=data.get("choices",[])
                        record["finish_reasons"]=[c.get("finish_reason") for c in choices]
                        record["final_content_present"]=[bool((c.get("message") or {}).get("content")) for c in choices]
                        record["reasoning_content_present"]=[bool((c.get("message") or {}).get("reasoning_content")) for c in choices]
                        truncated=any(r=="length" for r in record["finish_reasons"])
                        missing_final=parsed.path.endswith("/chat/completions") and response.status_code==200 and (not choices or not all(record["final_content_present"]))
                        invalid_response=truncated or missing_final
                        record["invalid_for_benchmark"]=invalid_response
                        record["invalid_response_type"]="length" if truncated else "empty_final" if missing_final else None
                    return response
            except Exception as exc:
                record["error_type"]=type(exc).__name__
                transient=isinstance(exc,(requests.Timeout,requests.ConnectionError,requests.exceptions.ChunkedEncodingError))
                record["will_retry"]=transient and attempt < maximum
                if not record["will_retry"]:
                    if budget.enabled():
                        raise budget.RecoverableProviderError("Provider transport failed (" + type(exc).__name__ + "); recover last task checkpoint") from exc
                    raise
            finally:
                record["elapsed_seconds"]=time.monotonic()-started
                if record["will_retry"]:
                    record["retry_delay_seconds"]=15.0 * 2**(attempt-1)
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
                try:os.write(fd,(json.dumps(record,ensure_ascii=False)+"\n").encode())
                finally:os.close(fd)
                budget.note_request(record)
                if invalid_response and budget.enabled():
                    budget.handle_response(data, payload, request_id=request_id, endpoint=request.url, timeout=kwargs.get("timeout"))
                if invalid_response and fail_on_truncation:
                    raise BenchmarkResponseError("Model response reached its token limit or lacks final content; inspect api-calls.jsonl before rerunning")
            time.sleep(record["retry_delay_seconds"])
    send.maple_instrumented=True
    requests.sessions.Session.send=send
    return original

def main():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    if len(sys.argv)<3:raise SystemExit("usage: run_instrumented.py MODULE ARGS...")
    module=sys.argv[1]
    if module not in ("maple_guard.run_appworld","maple_guard.run_mmlu","maple_guard.run_longmemeval","maple_guard.infa_memlink_eval"):
        raise ValueError("Unsupported benchmark module")
    log=os.environ["MAPLE_CALL_LOG"]
    install_metrics(log,os.getenv("MAPLE_FAIL_ON_TRUNCATION")=="1")
    sys.argv=[module,*sys.argv[2:]]
    runpy.run_module(module,run_name="__main__")
if __name__=="__main__":main()
