"""GUARDIAN official-scope communication adapter.

GUARDIAN models multi-agent collaboration as a temporal attributed graph and
removes anomalous communication nodes. This adapter keeps that scope intact:
it converts the current MAPLE-Guard communication history into GUARDIAN's graph
data format and calls the official GUARDIAN model files directly. It never
substitutes persistent-memory write/read/promotion logic.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import DefenseContext, OfficialDefenseState, decision, sorted_outputs, text_of_response


DEFAULT_GUARDIAN_DIRS = (
    "/path/to/GUARDIAN/code/communication-targeted_error_injection_and_propagation",
    "../GUARDIAN/code/communication-targeted_error_injection_and_propagation",
)


def _guardian_code_dir(ctx: DefenseContext) -> str:
    configured = str(
        getattr(ctx.args, "official_defense_guardian_code_dir", "")
        or os.environ.get("OFFICIAL_DEFENSE_GUARDIAN_CODE_DIR", "")
        or ""
    )
    if configured:
        return configured
    for candidate in DEFAULT_GUARDIAN_DIRS:
        path = Path(candidate).expanduser()
        if (path / "model_temporal_gib.py").exists() and (path / "model_static.py").exists():
            return str(path)
    return ""


def _guardian_device(ctx: DefenseContext) -> str:
    return str(
        os.environ.get("OFFICIAL_DEFENSE_GUARDIAN_DEVICE")
        or getattr(ctx.args, "official_defense_gnn_device", "")
        or "cpu"
    )


def _guardian_epochs() -> int:
    raw = os.environ.get("OFFICIAL_DEFENSE_GUARDIAN_EPOCHS", "20")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 20


def _guardian_bert_parent() -> Optional[Path]:
    configured = os.environ.get("OFFICIAL_DEFENSE_GUARDIAN_BERT_DIR", "")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path("models/bert-base-uncased"),
            Path("/path/to/maple_guard/models/bert-base-uncased"),
        ]
    )
    for path in candidates:
        if (path / "config.json").exists() and (path / "vocab.txt").exists():
            return path.parent
    return None


@contextlib.contextmanager
def _bert_lookup_context():
    parent = _guardian_bert_parent()
    if parent is None:
        yield
        return
    old_cwd = Path.cwd()
    os.chdir(parent)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _guardian_runtime_available() -> Tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    return True, ""


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_cache(state: OfficialDefenseState) -> Dict[str, Any]:
    return state.gnn_state.setdefault("guardian_runtime_cache", {})


def _load_official_runtime(state: OfficialDefenseState, code_dir: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    path = Path(code_dir).expanduser().resolve()
    cache_key = str(path)
    cached = _runtime_cache(state).get(cache_key)
    if cached is not None:
        return cached, None

    required = {
        "model_static": path / "model_static.py",
        "model_temporal_gib": path / "model_temporal_gib.py",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        return None, f"official_guardian_files_missing: {missing}"

    try:
        import torch
        from torch_geometric.data import Data

        static_module = _load_module(f"_official_guardian_static_{abs(hash(cache_key))}", required["model_static"])
        temporal_module = _load_module(f"_official_guardian_temporal_{abs(hash(cache_key))}", required["model_temporal_gib"])
    except Exception as exc:
        return None, f"official_guardian_import_failed: {type(exc).__name__}: {str(exc)[:300]}"

    runtime = {
        "torch": torch,
        "Data": Data,
        "static_module": static_module,
        "temporal_module": temporal_module,
        "code_dir": str(path),
    }
    _runtime_cache(state)[cache_key] = runtime
    return runtime, None


def _active_agent_ids(outputs: Dict[int, Any], state: OfficialDefenseState) -> List[int]:
    return [int(agent_id) for agent_id, _ in sorted_outputs(outputs) if int(agent_id) not in state.guardian_inactive_agents]


def _build_graph_data(runtime: Dict[str, Any], ctx: DefenseContext, history: Sequence[Dict[int, str]], active_ids: Sequence[int]):
    import numpy as np

    torch = runtime["torch"]
    Data = runtime["Data"]
    adj = np.asarray(ctx.adj_matrix, dtype=np.int64)
    n = len(active_ids)
    edge_pairs: List[Tuple[int, int]] = []
    for src_pos, src_agent in enumerate(active_ids):
        for dst_pos, dst_agent in enumerate(active_ids):
            if src_pos == dst_pos:
                continue
            if src_agent < adj.shape[0] and dst_agent < adj.shape[1] and int(adj[src_agent, dst_agent]) != 0:
                edge_pairs.append((src_pos, dst_pos))
    if not edge_pairs:
        return []

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    adj_tensor = torch.zeros((n, n), dtype=torch.float32)
    for src_pos, dst_pos in edge_pairs:
        adj_tensor[src_pos, dst_pos] = 1.0

    data_list = []
    for turn in history:
        texts = [str(turn.get(agent_id, "") or " ") for agent_id in active_ids]
        data_list.append(Data(text=texts, edge_index=edge_index.clone(), adj=adj_tensor.clone()))
    return data_list


def _run_official_guardian(runtime: Dict[str, Any], data_list, ctx: DefenseContext) -> Tuple[int, List[float], str]:
    torch = runtime["torch"]
    epochs = _guardian_epochs()
    device = _guardian_device(ctx)
    with _bert_lookup_context(), contextlib.redirect_stdout(io.StringIO()):
        if len(data_list) <= 1:
            model = runtime["static_module"].DOMINANTDetector(hid_dim=128, num_gnn_layers=2).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.fit(data_list[-1], num_epochs=epochs, optimizer=optimizer)
            _max_score, max_node_idx, scores = model.detect(data_list[-1])
            return int(max_node_idx), [float(x) for x in scores], "static_dominant"

        model = runtime["temporal_module"].TemporalDOMINANTDetector(
            hid_dim=64,
            num_gnn_layers=2,
            num_transformer_layers=2,
            nhead=4,
            feature_weight=0.3,
            beta=0.001,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.fit(data_list, num_epochs=epochs, optimizer=optimizer)
        _max_score, max_node_idx, scores = model.detect(data_list)
        return int(max_node_idx), [float(x) for x in scores], "temporal_dominant_gib"


def _allow_with_reason(outputs: Dict[int, Any], state: OfficialDefenseState, reason: str, details: Dict[str, Any]):
    filtered = {}
    decisions = []
    for idx, response in sorted_outputs(outputs):
        agent_id = int(idx)
        if agent_id in state.guardian_inactive_agents:
            decisions.append(decision(agent_id, "block", "guardian_official_inactive_agent_pruned", details))
            continue
        filtered[agent_id] = response
        decisions.append(decision(agent_id, "allow", reason, details))
    return filtered, decisions


def apply_guardian(outputs: Dict[int, Any], state: OfficialDefenseState, ctx: DefenseContext):
    state.guardian_history.append({int(idx): text_of_response(response) for idx, response in sorted_outputs(outputs)})

    code_dir = _guardian_code_dir(ctx)
    runtime_ok, runtime_error = _guardian_runtime_available()
    details = {
        "method_scope": "temporal_graph_communication_anomaly",
        "memory_firewall": False,
        "memory_read_firewall": False,
        "memory_promotion_gate": False,
        "official_source": "JialongZhou666/GUARDIAN",
        "history_turns": len(state.guardian_history),
        "inactive_agents": sorted(state.guardian_inactive_agents),
        "guardian_epochs": _guardian_epochs(),
        "guardian_device": _guardian_device(ctx),
    }
    if code_dir:
        details["guardian_code_dir"] = code_dir

    if not runtime_ok:
        if getattr(ctx.args, "strict_comparison", False):
            raise RuntimeError(f"GUARDIAN dependencies unavailable: {runtime_error}")
        details["runtime_error"] = runtime_error
        filtered, decisions = _allow_with_reason(outputs, state, "guardian_runtime_dependencies_missing_no_detection", details)
        return filtered, state, decisions
    if not code_dir:
        if getattr(ctx.args, "strict_comparison", False):
            raise RuntimeError("GUARDIAN source directory is required for strict comparison")
        filtered, decisions = _allow_with_reason(outputs, state, "guardian_code_dir_missing_no_detection", details)
        return filtered, state, decisions

    runtime, error = _load_official_runtime(state, code_dir)
    if runtime is None:
        if getattr(ctx.args, "strict_comparison", False):
            raise RuntimeError(f"GUARDIAN detector runtime unavailable: {error}")
        details["error"] = str(error or "unknown_official_guardian_runtime_error")[:500]
        filtered, decisions = _allow_with_reason(outputs, state, "guardian_official_runtime_unavailable", details)
        return filtered, state, decisions

    active_ids = _active_agent_ids(outputs, state)
    details["active_agents"] = list(active_ids)
    if len(active_ids) <= 1:
        filtered, decisions = _allow_with_reason(outputs, state, "guardian_official_not_enough_active_agents", details)
        return filtered, state, decisions

    data_list = _build_graph_data(runtime, ctx, state.guardian_history, active_ids)
    if not data_list:
        filtered, decisions = _allow_with_reason(outputs, state, "guardian_official_no_edges_no_detection", details)
        return filtered, state, decisions

    try:
        max_node_idx, scores, detector = _run_official_guardian(runtime, data_list, ctx)
        if getattr(ctx.args, "strict_comparison", False):
            if not 0 <= max_node_idx < len(active_ids) or len(scores) != len(active_ids):
                raise ValueError("GUARDIAN must return a valid node index and one score per active agent")
            if not all(math.isfinite(float(score)) for score in scores):
                raise ValueError("GUARDIAN anomaly scores must be finite")
    except Exception as exc:
        if getattr(ctx.args, "strict_comparison", False):
            raise RuntimeError("GUARDIAN detector inference failed; strict comparison aborted") from exc
        details["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        filtered, decisions = _allow_with_reason(outputs, state, "guardian_official_inference_failed_no_detection", details)
        return filtered, state, decisions

    pruned_agent = int(active_ids[max_node_idx])
    state.guardian_inactive_agents.add(pruned_agent)
    details.update(
        {
            "detector": detector,
            "pruned_agent": pruned_agent,
            "risk_scores": {int(agent_id): float(scores[pos]) for pos, agent_id in enumerate(active_ids) if pos < len(scores)},
            "inactive_agents": sorted(state.guardian_inactive_agents),
        }
    )

    filtered = {}
    decisions = []
    for idx, response in sorted_outputs(outputs):
        agent_id = int(idx)
        agent_details = dict(details)
        agent_details["risk_score"] = details["risk_scores"].get(agent_id)
        if agent_id == pruned_agent:
            decisions.append(decision(agent_id, "block", "guardian_official_pruned_max_anomaly_agent", agent_details))
            continue
        if agent_id in state.guardian_inactive_agents:
            decisions.append(decision(agent_id, "block", "guardian_official_inactive_agent_pruned", agent_details))
            continue
        filtered[agent_id] = response
        decisions.append(decision(agent_id, "allow", "guardian_official_allow_low_anomaly_agent", agent_details))
    return filtered, state, decisions
