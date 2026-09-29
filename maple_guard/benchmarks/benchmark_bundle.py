"""Validate the user's frozen benchmark bundles without resampling or executing payloads."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path

def canonical_benchmark(name):
    value = str(name).lower()
    return {"commonsenseqa": "csqa"}.get(value, value)

BENCHMARKS = ("appworld", "mmlu", "csqa", "longmemeval", "injecagent")

def load_bundle(path, expected=None):
    path=Path(path)
    raw=path.read_bytes()
    bundle=json.loads(raw)
    if not isinstance(bundle,dict) or not isinstance(bundle.get("tasks"),list):
        raise ValueError("Expected a benchmark bundle with a tasks list")
    benchmark=canonical_benchmark(bundle.get("benchmark",""))
    if benchmark not in BENCHMARKS or (expected and benchmark!=canonical_benchmark(expected)):
        raise ValueError(f"Unexpected benchmark: {benchmark!r}")
    rows=bundle["tasks"]
    if not rows or bundle.get("task_count")!=len(rows):
        raise ValueError("Bundle task count does not match nonempty tasks list")
    ids=[]
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get("data"),dict):
            raise ValueError("Every bundle row must contain a data object")
        if canonical_benchmark(row.get("benchmark",""))!=benchmark:
            raise ValueError("Row benchmark differs from bundle benchmark")
        task_id=row.get("task_id")
        if not isinstance(task_id,str) or not task_id.strip():
            raise ValueError("Every bundle row needs a nonempty task_id")
        ids.append(task_id)
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate task IDs in benchmark bundle")
    manifest={"benchmark":benchmark,"source_file":str(path.resolve()),
              "source_sha256":hashlib.sha256(raw).hexdigest(),
              "total_cases":len(rows),"ordered_task_ids":ids,
              "selection":bundle.get("selection"),"source":bundle.get("source"),
              "split_counts":dict(Counter(r.get("split","unknown") for r in rows))}
    for key in ("subject","attack_family"):
        counts=Counter(str(r[key]) for r in rows if key in r)
        if counts: manifest[key+"_counts"]=dict(counts)
    if benchmark=="injecagent":
        manifest["attack_type_counts"]=dict(Counter(r["data"].get("Attack Type","unknown") for r in rows))
    return bundle,manifest

def appworld_cases(bundle, source_path):
    from .appworld_adapter import AppWorldCase
    if canonical_benchmark(bundle.get("benchmark",""))!="appworld":
        raise ValueError("Expected AppWorld bundle")
    cases=[]; seen=set()
    for row in bundle["tasks"]:
        data=row["data"]
        task_id=str(row.get("native_task_id") or "")
        if not task_id or task_id in seen: raise ValueError("Missing or duplicate AppWorld native task ID")
        seen.add(task_id)
        instruction=str(data.get("instruction") or "").strip()
        if not instruction: raise ValueError(f"Empty AppWorld instruction: {task_id}")
        split=row.get("split","")
        if split not in ("train","dev","test_normal","test_challenge"):
            raise ValueError(f"Unknown AppWorld split: {split!r}")
        generator,_,number=task_id.rpartition("_")
        cases.append(AppWorldCase(case_id=f"{split}__{task_id}",split=split,task_id=task_id,
            instruction=instruction,generator_id=generator or task_id,
            task_number=int(number) if number.isdigit() else 0,
            datetime=str(data.get("datetime") or ""),supervisor=data.get("supervisor") or {},
            allowed_apps=tuple(data.get("allowed_apps") or ()),
            source_specs_file=f"{Path(source_path).resolve()}#tasks/{row['task_id']}/data"))
    return cases

def normalized_rows(bundle):
    """Unwrap the four non-AppWorld formats; retain stable source IDs and metadata."""
    benchmark=canonical_benchmark(bundle["benchmark"])
    if benchmark=="appworld": raise ValueError("Use appworld_cases for AppWorld")
    rows=[]
    for row in bundle["tasks"]:
        data=dict(row["data"])
        if benchmark=="csqa":
            question=data.get("question")
            if not isinstance(question,dict) or not question.get("stem") or not question.get("choices"):
                raise ValueError("CSQA requires question.stem and question.choices")
            data["question"]=question["stem"]; data["choices"]=question["choices"]
        if benchmark in ("csqa","mmlu"):
            if not data.get("question") or not data.get("choices"):
                raise ValueError("QA row requires question and choices")
            answer=data.get("answer",data.get("answerKey"))
            labels=[str(c.get("label")) if isinstance(c,dict) else chr(65+i) for i,c in enumerate(data["choices"])]
            if str(answer) not in labels: raise ValueError("QA answer is not a choice label")
        data["source_bundle_id"]=row["task_id"]
        data["source_native_id"]=data.get("id",data.get("question_id"))
        data["id"]=row["task_id"]
        data["dataset"]=benchmark
        for key in ("subject","split","attack_family"):
            if key in row: data[key]=row[key]
        rows.append(data)
    return rows
