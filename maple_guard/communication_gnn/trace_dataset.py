"""Build G-Safeguard-style graph datasets from MAPLE-Guard trace files.

The converter intentionally uses only communication-level fields:
agent outputs, communication topology, and attacker ids. It does not inspect
memory taint, provenance, retrieval, promotion, ASR labels, or defense traces.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TRACE_ATTACKER_CACHE: Dict[Tuple[str, int], Tuple[List[int], str]] = {}


@dataclass
class TraceGraph:
    bench: str
    trace_path: str
    record_index: int
    task_id: str
    attacker_ids: List[int]
    attacker_source: str
    graph: Dict[str, Any]


def expand_trace_inputs(patterns: Sequence[str], trace_list: str = "") -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])
    if trace_list:
        with open(trace_list, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(line)
    seen = set()
    out: List[str] = []
    for path in paths:
        if path not in seen and Path(path).is_file():
            seen.add(path)
            out.append(path)
    return out


def _normalize_outputs(outputs_by_round: Any, num_agents: int) -> List[List[str]]:
    rounds: List[List[str]] = []
    if not isinstance(outputs_by_round, list):
        return rounds
    for round_outputs in outputs_by_round:
        texts = ["" for _ in range(num_agents)]
        if isinstance(round_outputs, dict):
            items = round_outputs.items()
        elif isinstance(round_outputs, list):
            items = round_outputs
        else:
            items = []
        for key, value in items:
            try:
                agent_id = int(key)
            except Exception:
                continue
            if 0 <= agent_id < num_agents:
                texts[agent_id] = _text_of_response(value)
        rounds.append(texts)
    return rounds


def _text_of_response(value: Any) -> str:
    if isinstance(value, dict):
        if "answer" in value:
            return str(value.get("answer") or value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


def _parse_attacker_ids_value(value: Any, num_agents: int) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, str):
        values = [int(x) for x in re.findall(r"\d+", value)]
    elif isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            try:
                values.append(int(item))
            except Exception:
                pass
    else:
        values = []
    return sorted({x for x in values if 0 <= x < num_agents})


def _infer_attacker_ids_from_run_log(trace_path: str, num_agents: int) -> Tuple[List[int], str]:
    cache_key = (trace_path, num_agents)
    if cache_key in _TRACE_ATTACKER_CACHE:
        return _TRACE_ATTACKER_CACHE[cache_key]

    path = Path(trace_path)
    candidates: List[Path] = []
    if len(path.parents) >= 3:
        run_root = path.parents[2]
        bench = path.parent.name
        candidates.extend(
            [
                run_root / "logs" / f"generate_{bench}.log",
                run_root / "logs" / f"generate_{bench}.log.nohup",
            ]
        )

    best: List[int] = []
    for log_path in candidates:
        if not log_path.is_file():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in re.findall(r"attackers=\[([^\]]+)\]", text):
            ids = _parse_attacker_ids_value(raw, num_agents)
            if ids:
                best = ids

    result = (best, "run_log.attackers" if best else "missing")
    _TRACE_ATTACKER_CACHE[cache_key] = result
    return result


def infer_attacker_ids(record: Dict[str, Any], trace_path: str, num_agents: int) -> Tuple[List[int], str]:
    explicit = _parse_attacker_ids_value(record.get("attacker_ids"), num_agents)
    if explicit:
        return explicit, "record.attacker_ids"

    run_log_ids, source = _infer_attacker_ids_from_run_log(trace_path, num_agents)
    if run_log_ids:
        return run_log_ids, source

    # Some older MMLU sweep records kept only attacker_id but encoded the full
    # random-three attacker set in the run name, e.g. att015.
    text = f"{record.get('trace_id', '')} {trace_path}"
    matches = re.findall(r"(?:^|[_/-])att([0-9]{2,})(?:$|[_/-])", text)
    for match in matches:
        ids = sorted({int(ch) for ch in match if ch.isdigit() and int(ch) < num_agents})
        if len(ids) >= 2:
            return ids, "trace_id_or_path.attXYZ"

    single = _parse_attacker_ids_value(record.get("attacker_id"), num_agents)
    if single:
        return single, "record.attacker_id"
    return [], "missing"


def _get_task_trace(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_trace = record.get("task_trace")
    if isinstance(task_trace, dict):
        return task_trace
    traces = record.get("task_traces")
    if isinstance(traces, list) and traces and isinstance(traces[0], dict):
        return traces[0]
    if "outputs_by_round" in record and "adjacency" in record:
        return record
    return None


def _embed_rounds(rounds: Sequence[Sequence[str]], embedder: Any, batch_size: int) -> np.ndarray:
    flat: List[str] = []
    for round_texts in rounds:
        flat.extend(round_texts)
    embeddings = embedder.encode(
        flat,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    arr = np.asarray(embeddings, dtype=np.float32)
    num_rounds = len(rounds)
    num_agents = len(rounds[0]) if rounds else 0
    return arr.reshape(num_rounds, num_agents, -1).transpose(1, 0, 2)


def graph_from_record(
    record: Dict[str, Any],
    *,
    bench: str,
    trace_path: str,
    record_index: int,
    embedder: Any,
    batch_size: int,
    max_turns: int,
) -> Optional[TraceGraph]:
    task_trace = _get_task_trace(record)
    if not task_trace:
        return None
    adjacency = task_trace.get("adjacency") or record.get("adjacency")
    if not isinstance(adjacency, list) or not adjacency:
        return None
    adj = np.asarray(adjacency, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return None
    num_agents = int(adj.shape[0])
    rounds = _normalize_outputs(task_trace.get("outputs_by_round"), num_agents)
    if not rounds:
        return None
    if max_turns > 0:
        rounds = rounds[:max_turns]
    if not rounds:
        return None
    edge_index = np.array(adj.nonzero(), dtype=np.int64)
    if edge_index.size == 0:
        return None

    attacker_ids, attacker_source = infer_attacker_ids(record, trace_path, num_agents)
    if not attacker_ids:
        return None

    communication_embeddings = _embed_rounds(rounds, embedder, batch_size=batch_size)
    edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)
    first_round_features = np.array(communication_embeddings[:, 0, :], copy=True)
    malicious = np.array([1 if i in attacker_ids else 0 for i in range(num_agents)], dtype=np.float32)
    labels = np.stack([malicious, np.zeros_like(malicious)], axis=1)

    task_id = str(record.get("task_id") or task_trace.get("task_id") or f"{Path(trace_path).stem}_{record_index}")
    graph = {
        "adj_matrix": adj,
        "features": first_round_features,
        "labels": labels,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_self_replies": communication_embeddings,
        "meta": {
            "bench": bench,
            "trace_path": trace_path,
            "record_index": record_index,
            "task_id": task_id,
            "attacker_ids": attacker_ids,
            "attacker_source": attacker_source,
            "num_rounds": len(rounds),
            "num_agents": num_agents,
            "method_scope": "communication_only",
        },
    }
    return TraceGraph(bench, trace_path, record_index, task_id, attacker_ids, attacker_source, graph)


def raw_graph_from_record(
    record: Dict[str, Any],
    *,
    bench: str,
    trace_path: str,
    record_index: int,
    max_turns: int,
) -> Optional[Dict[str, Any]]:
    task_trace = _get_task_trace(record)
    if not task_trace:
        return None
    adjacency = task_trace.get("adjacency") or record.get("adjacency")
    if not isinstance(adjacency, list) or not adjacency:
        return None
    adj = np.asarray(adjacency, dtype=np.int64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return None
    num_agents = int(adj.shape[0])
    rounds = _normalize_outputs(task_trace.get("outputs_by_round"), num_agents)
    if max_turns > 0:
        rounds = rounds[:max_turns]
    if not rounds:
        return None
    edge_index = np.array(adj.nonzero(), dtype=np.int64)
    if edge_index.size == 0:
        return None
    attacker_ids, attacker_source = infer_attacker_ids(record, trace_path, num_agents)
    if not attacker_ids:
        return None
    task_id = str(record.get("task_id") or task_trace.get("task_id") or f"{Path(trace_path).stem}_{record_index}")
    return {
        "bench": bench,
        "trace_path": trace_path,
        "record_index": record_index,
        "task_id": task_id,
        "attacker_ids": attacker_ids,
        "attacker_source": attacker_source,
        "adjacency": adj.tolist(),
        "outputs_by_round": rounds,
        "num_rounds": len(rounds),
        "num_agents": num_agents,
        "method_scope": "communication_only",
    }


def iter_records(trace_path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(trace_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield idx, record


def convert_raw_traces(args: argparse.Namespace) -> Dict[str, Any]:
    trace_paths = expand_trace_inputs(args.trace_glob, args.trace_list)
    out_dir = Path(args.out_root) / "datasets" / args.bench / "gsafeguard"
    manifest_dir = Path(args.out_root) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw_graphs.jsonl"
    graph_count = 0
    skipped = 0
    attacker_sources: Dict[str, int] = {}
    with open(raw_path, "w", encoding="utf-8") as handle:
        for trace_path in trace_paths:
            for record_index, record in iter_records(trace_path):
                if args.max_records and graph_count >= args.max_records:
                    break
                raw = raw_graph_from_record(
                    record,
                    bench=args.bench,
                    trace_path=trace_path,
                    record_index=record_index,
                    max_turns=args.max_turns,
                )
                if raw is None:
                    skipped += 1
                    continue
                handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                graph_count += 1
                source = str(raw.get("attacker_source", "unknown"))
                attacker_sources[source] = attacker_sources.get(source, 0) + 1
            if args.max_records and graph_count >= args.max_records:
                break

    trace_list_path = manifest_dir / f"{args.bench}_trace_paths.txt"
    with open(trace_list_path, "w", encoding="utf-8") as handle:
        for path in trace_paths:
            handle.write(f"{path}\n")

    manifest = {
        "bench": args.bench,
        "raw_path": str(raw_path),
        "trace_count": len(trace_paths),
        "graph_count": graph_count,
        "skipped_records": skipped,
        "max_turns": args.max_turns,
        "attacker_sources": attacker_sources,
        "method_scope": "communication_only",
        "stage": "raw",
        "excluded_signals": [
            "memory_taint",
            "source_type",
            "provenance_trust",
            "retrieval_decisions",
            "defense_decisions",
            "asr_labels",
        ],
    }
    manifest_path = out_dir / "raw_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def convert_traces(args: argparse.Namespace) -> Dict[str, Any]:
    if args.raw_only:
        return convert_raw_traces(args)

    from sentence_transformers import SentenceTransformer

    trace_paths = expand_trace_inputs(args.trace_glob, args.trace_list)
    out_dir = Path(args.out_root) / "datasets" / args.bench / "gsafeguard"
    manifest_dir = Path(args.out_root) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    graphs: List[Dict[str, Any]] = []
    skipped = 0
    attacker_sources: Dict[str, int] = {}

    for trace_path in trace_paths:
        for record_index, record in iter_records(trace_path):
            if args.max_records and len(graphs) >= args.max_records:
                break
            graph = graph_from_record(
                record,
                bench=args.bench,
                trace_path=trace_path,
                record_index=record_index,
                embedder=embedder,
                batch_size=args.embedding_batch_size,
                max_turns=args.max_turns,
            )
            if graph is None:
                skipped += 1
                continue
            graphs.append(graph.graph)
            attacker_sources[graph.attacker_source] = attacker_sources.get(graph.attacker_source, 0) + 1
        if args.max_records and len(graphs) >= args.max_records:
            break

    dataset_path = out_dir / "dataset.pkl"
    with open(dataset_path, "wb") as handle:
        pickle.dump(graphs, handle)

    trace_list_path = manifest_dir / f"{args.bench}_trace_paths.txt"
    with open(trace_list_path, "w", encoding="utf-8") as handle:
        for path in trace_paths:
            handle.write(f"{path}\n")

    manifest = {
        "bench": args.bench,
        "dataset_path": str(dataset_path),
        "trace_count": len(trace_paths),
        "graph_count": len(graphs),
        "skipped_records": skipped,
        "embedding_model": args.embedding_model,
        "max_turns": args.max_turns,
        "attacker_sources": attacker_sources,
        "method_scope": "communication_only",
        "excluded_signals": [
            "memory_taint",
            "source_type",
            "provenance_trust",
            "retrieval_decisions",
            "defense_decisions",
            "asr_labels",
        ],
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MAPLE-Guard traces into G-Safeguard GNN training data.")
    parser.add_argument("--bench", required=True, choices=["mmlu", "longmemeval", "appworld"])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--trace-glob", action="append", default=[], help="Trace glob or explicit trace path. Can be repeated.")
    parser.add_argument("--trace-list", default="", help="Optional file containing trace paths, one per line.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--raw-only", action="store_true", help="Write raw communication graphs without loading an embedding model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trace_glob and not args.trace_list:
        raise SystemExit("Provide --trace-glob or --trace-list.")
    convert_traces(args)


if __name__ == "__main__":
    main()
