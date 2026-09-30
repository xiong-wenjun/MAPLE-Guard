"""Replay saved external evaluator requests only; never regenerate task agents.

Results are appended to a private, auditable sidecar. The original trace and its
memory trajectory remain immutable, including skipped pending feedback.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def _append(path, row):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        os.fchmod(fd,0o600)
        os.write(fd,(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n').encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _pending(trace):
    seen=set()
    for line in Path(trace).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        for event in row.get('pending_evaluators',[]):
            key=(event.get('task_id'),event.get('request_id'))
            if (event.get('role')!='evaluator' or event.get('invalid_response_type')!='length'
                    or event.get('status')!='pending_budget_rescore' or not all(key)
                    or key in seen or event.get('task_id')!=row.get('task_id',row.get('sample_id'))
                    or event.get('result_key') not in {'correct','imitates'}):
                raise ValueError('Invalid or duplicate saved evaluator request')
            seen.add(key)
            yield event


def _parse(text, key):
    from maple_guard.budget_outcomes import parse_evaluator_verdict
    return parse_evaluator_verdict(text, key)


def rescore_requests(trace, output, *, max_tokens, max_attempts=3):
    import requests
    from maple_guard.budget_outcomes import _safe_endpoint
    from maple_guard.providers.service_auth import chat_headers
    if type(max_tokens) is not int or max_tokens<=0 or max_attempts not in (1,2,3):
        raise ValueError('Positive token cap and one to three attempts required')
    output=Path(output)
    if output.resolve()==Path(trace).resolve():raise ValueError('Sidecar must differ from original trace')
    completed={}
    if output.exists():
        for line in output.read_text().splitlines():
            row=json.loads(line)
            if row.get('status')=='rescored':
                completed[(row['task_id'],row['original_request_id'])]=row
    results=[]
    for event in _pending(trace):
        saved=event['request']
        endpoint=_safe_endpoint(saved['endpoint'])
        if not endpoint.endswith('/chat/completions'):
            raise ValueError('Only saved chat evaluator endpoints are supported')
        original=copy.deepcopy(saved['payload'])
        allowed={'model','messages','temperature','max_tokens','stop','response_format','chat_template_kwargs'}
        if set(original)-allowed or not isinstance(original.get('messages'),list):
            raise ValueError('Unsafe or malformed saved evaluator payload')
        if max_tokens<=int(original.get('max_tokens',0)):
            raise ValueError('Rescore cap must exceed the saved evaluator budget')
        key=(event['task_id'],event['request_id'])
        digest=_digest(original)
        if key in completed:
            row=completed[key]
            if row['original_payload_sha256']!=digest:
                raise ValueError('Completed replay request payload changed')
            results.append(row);continue
        payload=copy.deepcopy(original);payload['max_tokens']=max_tokens
        row={'task_id':event['task_id'],'original_request_id':event['request_id'],
             'rescore_request_id':uuid.uuid4().hex,'role':'evaluator','result_key':event['result_key'],
             'original_payload_sha256':digest,'rescore_payload_sha256':_digest(payload),
             'original_max_tokens':original.get('max_tokens'),'rescore_max_tokens':max_tokens,
             'verdict':None,'status':'pending_transport','attempts':[],
             'agents_regenerated':False,'memory_feedback_replayed':False}
        for attempt in range(1,max_attempts+1):
            audit={'attempt':attempt}
            try:
                reply=requests.post(endpoint,json=payload,headers=chat_headers(endpoint.rsplit('/chat/completions',1)[0]),timeout=saved.get('timeout') or 60)
                audit['http_status']=reply.status_code
                if reply.status_code==429 or reply.status_code>=500:
                    row['attempts'].append(audit)
                    if attempt < max_attempts:time.sleep(15 * 2**(attempt-1))
                    continue
                reply.raise_for_status()
                data=reply.json()
                if not isinstance(data,dict) or not isinstance(data.get('choices'),list) or not data['choices']:
                    raise ValueError('Malformed evaluator response envelope')
                choice=data['choices'][0]
                if not isinstance(choice,dict) or not isinstance(choice.get('message'),dict):
                    raise ValueError('Malformed evaluator choice or message')
                audit['finish_reason']=choice.get('finish_reason')
                row['attempts'].append(audit)
                content=(choice.get('message') or {}).get('content')
                if choice.get('finish_reason')=='length':
                    row['status']='pending_budget_rescore';break
                if not isinstance(content,str) or not content.strip():
                    row['status']='pending_empty_final';break
                try:
                    verdict,confidence,evidence=_parse(content,event['result_key'])
                except (ValueError,TypeError):
                    row['status']='pending_invalid_parse';break
                row.update(status='rescored',verdict=verdict,confidence=confidence,evidence=evidence)
                break
            except (requests.Timeout,requests.ConnectionError,requests.exceptions.ChunkedEncodingError) as exc:
                audit['error_type']=type(exc).__name__;row['attempts'].append(audit)
                if attempt < max_attempts:time.sleep(15 * 2**(attempt-1))
            except (requests.HTTPError,ValueError,KeyError,IndexError,TypeError) as exc:
                audit['error_type']=type(exc).__name__
                if audit not in row['attempts']:row['attempts'].append(audit)
                row['status']='pending_provider_error';break
        _append(output,row);results.append(row)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--max-tokens',type=int,required=True)
    parser.add_argument('--max-attempts',type=int,choices=(1,2,3),default=3)
    args=parser.parse_args()
    results=rescore_requests(args.trace,args.out,max_tokens=args.max_tokens,max_attempts=args.max_attempts)
    print(json.dumps({'requests':len(results),'rescored':sum(r['status']=='rescored' for r in results),'sidecar':str(Path(args.out).absolute())}))

if __name__=='__main__':main()
