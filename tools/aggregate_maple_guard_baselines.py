#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MAPLE-Guard baseline and guard-sweep trace status.")
    parser.add_argument("--root", action="append", required=True, help="Result root to scan. Can be passed multiple times.")
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--cwd-root", default=".")
    parser.add_argument("--completed-only", action="store_true", help="Emit only runs with trace.summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd_root = Path(args.cwd_root).resolve()
    roots = [resolve_path(value, cwd_root) for value in args.root]
    active_traces = discover_active_traces(cwd_root)
    planned = collect_manifest_entries(roots, cwd_root)
    trace_paths = set(planned)
    for root in roots:
        trace_paths.update(root.glob("**/trace.jsonl"))

    rows = []
    for trace_path in sorted(trace_paths, key=lambda p: str(p)):
        manifest = planned.get(trace_path, {})
        rows.append(summarize_trace(trace_path, manifest, active_traces, args.expected_tasks, cwd_root))

    all_status_counts = dict(Counter(row["status"] for row in rows))
    if args.completed_only:
        rows = [row for row in rows if row["status"] == "completed"]
    grouped = build_grouped(rows)
    payload = {
        "roots": [str(path) for path in roots],
        "completed_only": bool(args.completed_only),
        "all_status_counts": all_status_counts,
        "row_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "grouped": grouped,
        "rows": rows,
    }
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_prefix.with_suffix(".json"), payload)
    write_tsv(output_prefix.with_suffix(".tsv"), rows)
    write_markdown(output_prefix.with_suffix(".md"), payload)
    print(json.dumps({k: payload[k] for k in ("row_count", "status_counts")}, indent=2))
    print(f"json={output_prefix.with_suffix('.json')}")
    print(f"tsv={output_prefix.with_suffix('.tsv')}")
    print(f"md={output_prefix.with_suffix('.md')}")


def resolve_path(value: str, cwd_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd_root / path
    return path.resolve()


def discover_active_traces(cwd_root: Path) -> set[Path]:
    active: set[Path] = set()
    proc = Path("/proc")
    if not proc.exists():
        return active
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if "maple_guard/run_mmlu.py" not in cmd:
            continue
        match = re.search(r"--out\s+([^ ]+)", cmd)
        if not match:
            continue
        trace_path = Path(match.group(1))
        if not trace_path.is_absolute():
            trace_path = cwd_root / trace_path
        active.add(trace_path.resolve())
    return active


def collect_manifest_entries(roots: list[Path], cwd_root: Path) -> dict[Path, dict[str, Any]]:
    entries: dict[Path, dict[str, Any]] = {}
    for root in roots:
        for manifest in sorted(root.glob("**/manifest.tsv")):
            try:
                with manifest.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        trace_value = row.get("trace")
                        if trace_value:
                            trace_path = resolve_path(trace_value, cwd_root)
                        elif row.get("run_id"):
                            trace_path = infer_trace_from_manifest_row(root, row, cwd_root)
                        else:
                            continue
                        if trace_path:
                            normalized = {k: v for k, v in row.items() if k}
                            normalized["manifest_path"] = str(manifest)
                            entries[trace_path] = normalized
            except Exception:
                continue
        for manifest in sorted(root.glob("**/manifest.jsonl")):
            try:
                with manifest.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        trace_value = row.get("trace") or row.get("out")
                        if not trace_value:
                            continue
                        trace_path = resolve_path(trace_value, cwd_root)
                        row["manifest_path"] = str(manifest)
                        entries[trace_path] = row
            except Exception:
                continue
    return entries


def infer_trace_from_manifest_row(root: Path, row: dict[str, str], cwd_root: Path) -> Path | None:
    run_id = row.get("run_id")
    defense = row.get("defense")
    if not run_id:
        return None
    candidates = []
    if defense:
        candidates.append(root / defense / "r" / f"{run_id}_root" / run_id / "trace.jsonl")
    candidates.append(root / "roots" / f"{run_id}_root" / run_id / "trace.jsonl")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def summarize_trace(
    trace_path: Path,
    manifest: dict[str, Any],
    active_traces: set[Path],
    default_expected_tasks: int,
    cwd_root: Path,
) -> dict[str, Any]:
    summary_path = Path(str(trace_path).replace(".jsonl", ".summary.json"))
    summary = load_json(summary_path) if summary_path.exists() else {}
    trace_rows = load_jsonl(trace_path) if trace_path.exists() else []
    first = trace_rows[0] if trace_rows else {}
    metrics = metrics_from_summary_or_trace(summary, trace_rows)
    expected_tasks = int(summary.get("tasks") or infer_expected_tasks(trace_path, manifest, default_expected_tasks))
    observed_tasks = int(summary.get("tasks") or len(trace_rows))
    trace_resolved = trace_path.resolve()
    if summary_path.exists():
        status = "completed"
    elif trace_resolved in active_traces:
        status = "running"
    elif observed_tasks <= 0:
        status = "pending"
    elif observed_tasks >= expected_tasks:
        status = "needs_summary"
    else:
        status = "incomplete"
    method = str(summary.get("method") or first.get("method") or manifest.get("defense") or infer_method(trace_path))
    attack_variant = str(summary.get("attack_variant") or first.get("attack_variant") or infer_attack_variant(trace_path, manifest))
    attack_family = str(manifest.get("attack") or family_from_variant(attack_variant) or infer_attack_family(trace_path))
    topology = str(
        nested_get(summary, ["config", "communication_topology"])
        or manifest.get("topology")
        or infer_topology(trace_path)
        or ""
    )
    return {
        "experiment_root": str(find_matching_root(trace_path, cwd_root)),
        "run_id": trace_path.parent.name,
        "status": status,
        "method": method,
        "attack_family": attack_family,
        "attack_variant": attack_variant,
        "topology": topology,
        "expected_tasks": expected_tasks,
        "observed_tasks": observed_tasks,
        "progress": safe_div(observed_tasks, expected_tasks),
        "summary_exists": summary_path.exists(),
        "trace_exists": trace_path.exists(),
        "trace_path": relpath(trace_path, cwd_root),
        "summary_path": relpath(summary_path, cwd_root),
        "manifest_path": manifest.get("manifest_path", ""),
        **metrics,
    }


def metrics_from_summary_or_trace(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if summary:
        keys = [
            "task_sr",
            "overall_task_sr",
            "robust_acc",
            "rsr",
            "asr",
            "asr_metric",
            "retrieval_damage_asr",
            "target_hit_asr",
            "memory_conditioned_asr",
            "memory_conditioned_target_hit_asr",
            "pattern_asr",
            "mdsr",
            "pmur_agent",
            "pmwr",
            "pmur",
            "ppr",
            "cpr",
            "write_quarantine_rate",
            "write_rewrite_rate",
            "read_block_count",
        ]
        return {key: summary.get(key) for key in keys}
    if not rows:
        return empty_metrics()
    final_round = final_round_key(rows)
    n = len(rows)
    poison_rows = [row for row in rows if bool(row.get("is_poisoning_task"))]
    attack_success = sum(bool(row.get("attack_success")) for row in rows)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    robust = sum(bool(row.get("is_correct")) and not bool(row.get("attack_success")) for row in rows)
    exposed = sum(int(nested_get(row, ["benign_poison_exposed_by_round", final_round]) or 0) for row in rows)
    benign_total = sum(int(nested_get(row, ["benign_total_by_round", final_round]) or 0) for row in rows)
    benign_attack = sum(int(nested_get(row, ["benign_attack_success_by_round", final_round]) or 0) for row in rows)
    benign_target_hit = sum(int(nested_get(row, ["benign_target_hit_success_by_round", final_round]) or 0) for row in rows)
    mdsr = sum(int(nested_get(row, ["mdsr_success_by_round", final_round]) or 0) for row in rows)
    defense_decisions = [decision for row in rows for decision in row.get("defense_decisions", []) if isinstance(decision, dict)]
    retrieval_decisions = [decision for row in rows for decision in row.get("retrieval_decisions", []) if isinstance(decision, dict)]
    return {
        "task_sr": safe_div(correct, n),
        "overall_task_sr": safe_div(correct, n),
        "robust_acc": safe_div(robust, n),
        "rsr": safe_div(robust, n),
        "asr": safe_div(benign_attack, benign_total),
        "asr_metric": "partial_retrieval_damage_asr",
        "retrieval_damage_asr": safe_div(benign_attack, benign_total),
        "target_hit_asr": safe_div(benign_target_hit, benign_total),
        "memory_conditioned_asr": safe_div(benign_attack, exposed),
        "memory_conditioned_target_hit_asr": safe_div(benign_target_hit, exposed),
        "pattern_asr": None,
        "mdsr": safe_div(mdsr, n),
        "pmur_agent": safe_div(exposed, benign_total),
        "pmwr": safe_div(sum(bool(row.get("poisoned_memory_written")) for row in poison_rows), len(poison_rows)),
        "pmur": safe_div(sum(bool(row.get("natural_trigger")) for row in rows), n),
        "ppr": None,
        "cpr": safe_div(sum(int(row.get("benign_poison_user_count") or 0) for row in rows), n),
        "write_quarantine_rate": safe_div(sum(decision.get("action") == "quarantine" for decision in defense_decisions), len(defense_decisions)),
        "write_rewrite_rate": safe_div(sum(decision.get("action") == "rewrite" for decision in defense_decisions), len(defense_decisions)),
        "read_block_count": sum(1 for decision in retrieval_decisions if decision.get("blocked")),
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "task_sr": None,
        "overall_task_sr": None,
        "robust_acc": None,
        "rsr": None,
        "asr": None,
        "asr_metric": "",
        "retrieval_damage_asr": None,
        "target_hit_asr": None,
        "memory_conditioned_asr": None,
        "memory_conditioned_target_hit_asr": None,
        "pattern_asr": None,
        "mdsr": None,
        "pmur_agent": None,
        "pmwr": None,
        "pmur": None,
        "ppr": None,
        "cpr": None,
        "write_quarantine_rate": None,
        "write_rewrite_rate": None,
        "read_block_count": None,
    }


def build_grouped(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["method"], row["attack_family"], row["attack_variant"], row["topology"])
        groups[key].append(row)
    grouped = []
    metric_keys = [
        "task_sr",
        "robust_acc",
        "retrieval_damage_asr",
        "target_hit_asr",
        "memory_conditioned_asr",
        "mdsr",
        "pmur_agent",
        "write_quarantine_rate",
    ]
    for (method, attack_family, attack_variant, topology), items in sorted(groups.items()):
        completed = [row for row in items if row["status"] == "completed"]
        usable = completed or items
        entry: dict[str, Any] = {
            "method": method,
            "attack_family": attack_family,
            "attack_variant": attack_variant,
            "topology": topology,
            "runs": len(items),
            "completed": len(completed),
            "running": sum(row["status"] == "running" for row in items),
            "pending_or_incomplete": sum(row["status"] not in {"completed", "running"} for row in items),
            "mean_observed_tasks": mean(row["observed_tasks"] for row in items),
        }
        for metric in metric_keys:
            entry[f"mean_{metric}"] = mean(row.get(metric) for row in usable if row.get(metric) is not None)
        grouped.append(entry)
    return grouped


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "status",
        "method",
        "attack_family",
        "attack_variant",
        "topology",
        "run_id",
        "expected_tasks",
        "observed_tasks",
        "progress",
        "task_sr",
        "robust_acc",
        "retrieval_damage_asr",
        "target_hit_asr",
        "memory_conditioned_asr",
        "mdsr",
        "pmur_agent",
        "pmwr",
        "write_quarantine_rate",
        "read_block_count",
        "trace_path",
        "summary_path",
        "manifest_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    grouped = payload["grouped"]
    if payload.get("completed_only"):
        mode_note = "This file was emitted with `--completed-only`; every row requires `trace.summary.json`."
    else:
        mode_note = "Running traces are included for progress only; use `--completed-only` for paper tables."
    lines = [
        "# MAPLE-Guard baseline and guard-sweep status",
        "",
        f"- Rows: {payload['row_count']}",
        f"- Status counts: {payload['status_counts']}",
        f"- Mode: {mode_note}",
        "",
        "## Grouped status",
        "",
        "| method | attack | topology | runs | completed | running | mean observed | robust_acc | retrieval_damage_asr | target_hit_asr | mdsr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        attack = row["attack_family"] or row["attack_variant"]
        lines.append(
            "| {method} | {attack} | {topology} | {runs} | {completed} | {running} | {observed} | {robust} | {asr} | {target} | {mdsr} |".format(
                method=row["method"],
                attack=attack,
                topology=row["topology"],
                runs=row["runs"],
                completed=row["completed"],
                running=row["running"],
                observed=fmt(row["mean_observed_tasks"]),
                robust=fmt(row["mean_robust_acc"]),
                asr=fmt(row["mean_retrieval_damage_asr"]),
                target=fmt(row["mean_target_hit_asr"]),
                mdsr=fmt(row["mean_mdsr"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `completed` requires `trace.summary.json`.",
            "- Partial metrics from running `trace.jsonl` rows should not be used as final paper numbers.",
            "- The paper table should use completed rows or be regenerated with `--completed-only` after all runs finish.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def nested_get(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def final_round_key(rows: list[dict[str, Any]]) -> str:
    keys = set()
    for row in rows:
        by_round = row.get("benign_total_by_round")
        if isinstance(by_round, dict):
            keys.update(str(key) for key in by_round)
    numeric = sorted((int(key) for key in keys if str(key).isdigit()), reverse=True)
    return str(numeric[0]) if numeric else "3"


def infer_expected_tasks(trace_path: Path, manifest: dict[str, Any], default: int) -> int:
    for value in [str(trace_path), str(manifest.get("manifest_path", ""))]:
        match = re.search(r"(?:^|[_/])t(\d+)(?:_|/|$)", value)
        if match:
            return int(match.group(1))
        match = re.search(r"mmlu(\d+)", value)
        if match:
            return int(match.group(1))
    return default


def infer_method(trace_path: Path) -> str:
    parts = trace_path.parts
    known = {
        "gsafeguard",
        "infa_guard",
        "agentsafe",
        "agentxposed_guide",
        "guardian",
        "challenger",
        "maple_guard",
    }
    for part in parts:
        if part in known:
            return part
    return ""


def infer_attack_variant(trace_path: Path, manifest: dict[str, Any]) -> str:
    attack = str(manifest.get("attack") or infer_attack_family(trace_path))
    if attack == "agentpoison":
        return "trigger_backdoor"
    if attack in {"memorygraft", "memory_graft"}:
        return "memory_graft"
    run_id = trace_path.parent.name
    if run_id.startswith("ap_"):
        return "trigger_backdoor"
    if run_id.startswith("mg_"):
        return "memory_graft"
    return attack


def infer_attack_family(trace_path: Path) -> str:
    parts = trace_path.parts
    if "agentpoison" in parts:
        return "agentpoison"
    if "memorygraft" in parts:
        return "memorygraft"
    run_id = trace_path.parent.name
    if run_id.startswith("ap_"):
        return "agentpoison"
    if run_id.startswith("mg_"):
        return "memorygraft"
    if "trigger_backdoor" in str(trace_path):
        return "agentpoison"
    if "memory_graft" in str(trace_path):
        return "memorygraft"
    return ""


def family_from_variant(attack_variant: str) -> str:
    if attack_variant == "trigger_backdoor":
        return "agentpoison"
    if attack_variant == "memory_graft":
        return "memorygraft"
    return attack_variant


def infer_topology(trace_path: Path) -> str:
    parts = set(trace_path.parts)
    for topology in ["chain", "tree", "star", "random"]:
        if topology in parts:
            return topology
    run_id = trace_path.parent.name
    if "_c_" in run_id:
        return "chain"
    if "_t_" in run_id:
        return "tree"
    if "_s_" in run_id:
        return "star"
    if "_r" in run_id or "random" in run_id:
        return "random"
    match = re.search(r"topo_([^_]+)", run_id)
    return match.group(1) if match else ""


def find_matching_root(trace_path: Path, cwd_root: Path) -> str:
    text = relpath(trace_path, cwd_root)
    marker = "/"
    parts = text.split(marker)
    if "result_maple_guard" in parts:
        idx = parts.index("result_maple_guard")
        if len(parts) > idx + 2:
            return marker.join(parts[idx : idx + 3])
    return marker.join(parts[:3])


def relpath(path: Path, cwd_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd_root))
    except Exception:
        return str(path)


def safe_div(num: float, den: float) -> float | None:
    if not den:
        return None
    return float(num) / float(den)


def mean(values: Any) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
