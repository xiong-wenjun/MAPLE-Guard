"""Checkpoint-backed G-Safeguard and INFA-Guard communication adapters.

These baselines stay in the official communication-only scope: they inspect the
multi-agent utterance graph after each round and prune suspicious agent outputs.
They do not inspect, label, mutate, or quarantine persistent-memory records.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import DefenseContext, OfficialDefenseState, decision, sorted_outputs, text_of_response


DEFAULT_GNN_EMBEDDING_MODEL = "models/sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GNN_THRESHOLD = 0.5


def _checkpoint_path(ctx: DefenseContext) -> str:
    return str(getattr(ctx.args, "official_defense_gnn_checkpoint", "") or "")


def _embedding_model(ctx: DefenseContext) -> str:
    return str(getattr(ctx.args, "official_defense_embedding_model", "") or DEFAULT_GNN_EMBEDDING_MODEL)


def _threshold(ctx: DefenseContext) -> float:
    raw = getattr(ctx.args, "official_defense_gnn_threshold", None)
    if raw is None:
        raw = os.environ.get("OFFICIAL_DEFENSE_GNN_THRESHOLD", DEFAULT_GNN_THRESHOLD)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_GNN_THRESHOLD


def _device(ctx: DefenseContext) -> str:
    return str(getattr(ctx.args, "official_defense_gnn_device", "") or os.environ.get("OFFICIAL_DEFENSE_GNN_DEVICE", "cpu"))


def _allow_all(outputs: Dict[int, Any], reason: str, details: Optional[Dict[str, Any]] = None):
    decisions = [
        decision(idx, "allow", reason, details)
        for idx, _response in sorted_outputs(outputs)
    ]
    return dict(outputs), decisions


def _runtime_cache(state: OfficialDefenseState) -> Dict[str, Any]:
    return state.gnn_state.setdefault("runtime_cache", {})


def _history_cache(state: OfficialDefenseState) -> Dict[str, List[List[str]]]:
    return state.gnn_state.setdefault("round_history", {})


def _load_runtime(state: OfficialDefenseState, ctx: DefenseContext, guard: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    checkpoint = _checkpoint_path(ctx)
    embedding_model = _embedding_model(ctx)
    device = _device(ctx)
    cache_key = f"{guard}|{checkpoint}|{embedding_model}|{device}"
    cache = _runtime_cache(state)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached, None

    try:
        import torch
        from sentence_transformers import SentenceTransformer

        from maple_guard.communication_gnn.model import GSafeguardGAT
    except Exception as exc:  # pragma: no cover - depends on optional baseline deps
        return None, f"dependency_load_failed: {exc}"

    try:
        ckpt = torch.load(checkpoint, map_location=device)
        model_kwargs = dict(ckpt.get("model_kwargs") or {})
        model_state = ckpt.get("model_state_dict")
        if not model_state:
            return None, "checkpoint_missing_model_state_dict"
        model = GSafeguardGAT(**model_kwargs)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        embedder = SentenceTransformer(embedding_model, device=device)
        runtime = {
            "torch": torch,
            "model": model,
            "embedder": embedder,
            "checkpoint": checkpoint,
            "embedding_model": embedding_model,
            "device": device,
            "model_kwargs": model_kwargs,
            "checkpoint_dataset": ckpt.get("dataset"),
            "checkpoint_scope": ckpt.get("method_scope"),
        }
        cache[cache_key] = runtime
        return runtime, None
    except Exception as exc:  # pragma: no cover - depends on checkpoint/runtime env
        return None, f"runtime_load_failed: {exc}"


def _append_round_history(state: OfficialDefenseState, ctx: DefenseContext, outputs: Dict[int, Any]) -> List[List[str]]:
    history = _history_cache(state).setdefault(str(ctx.task_id), [])
    num_agents = int(getattr(ctx.args, "agents", 0) or len(outputs) or 1)
    round_texts = [text_of_response(outputs.get(agent_id, "")) for agent_id in range(num_agents)]
    round_idx = max(0, int(ctx.round_idx))
    while len(history) < round_idx:
        history.append([""] * num_agents)
    if len(history) == round_idx:
        history.append(round_texts)
    else:
        history[round_idx] = round_texts
    return history[: round_idx + 1]


def _build_graph_tensors(runtime: Dict[str, Any], ctx: DefenseContext, history: List[List[str]]):
    torch = runtime["torch"]
    device = runtime["device"]
    num_agents = int(getattr(ctx.args, "agents", 0) or len(history[0]) or 1)
    num_rounds = len(history)

    flat_texts: List[str] = []
    for round_texts in history:
        padded = list(round_texts[:num_agents]) + [""] * max(0, num_agents - len(round_texts))
        flat_texts.extend(padded[:num_agents])

    embeddings = runtime["embedder"].encode(
        flat_texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32).reshape(num_rounds, num_agents, -1).transpose(1, 0, 2)

    adj = np.asarray(ctx.adj_matrix, dtype=np.int64)
    edge_index_np = np.array(adj.nonzero(), dtype=np.int64)
    if edge_index_np.size == 0:
        return None, None, None
    edge_attr_np = np.array(embeddings[edge_index_np[1]], copy=True, dtype=np.float32)

    x = torch.tensor(embeddings[:, 0, :], dtype=torch.float32, device=device)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32, device=device)
    return x, edge_index, edge_attr


def apply_gnn_official(outputs: Dict[int, Any], state: OfficialDefenseState, ctx: DefenseContext, guard: str):
    checkpoint = _checkpoint_path(ctx)
    if not checkpoint:
        filtered, decisions = _allow_all(outputs, f"{guard}_checkpoint_missing_no_detection", {"method_scope": "communication_gnn_detector"})
        return filtered, state, decisions

    runtime, error = _load_runtime(state, ctx, guard)
    details = {
        "checkpoint": checkpoint,
        "embedding_model": _embedding_model(ctx),
        "threshold": _threshold(ctx),
        "method_scope": "communication_gnn_detector",
        "memory_firewall": False,
    }
    if runtime is None:
        details["error"] = str(error or "unknown_runtime_error")[:500]
        filtered, decisions = _allow_all(outputs, f"{guard}_gnn_runtime_unavailable", details)
        return filtered, state, decisions

    history = _append_round_history(state, ctx, outputs)
    try:
        x, edge_index, edge_attr = _build_graph_tensors(runtime, ctx, history)
        if x is None:
            filtered, decisions = _allow_all(outputs, f"{guard}_gnn_no_edges_no_detection", details)
            return filtered, state, decisions
        torch = runtime["torch"]
        with torch.no_grad():
            logits = runtime["model"](x, edge_index, edge_attr).view(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
    except Exception as exc:  # pragma: no cover - depends on optional baseline deps
        details["error"] = str(exc)[:500]
        filtered, decisions = _allow_all(outputs, f"{guard}_gnn_inference_failed", details)
        return filtered, state, decisions

    threshold = _threshold(ctx)
    filtered: Dict[int, Any] = {}
    decisions = []
    for idx, response in sorted_outputs(outputs):
        risk = float(probs[int(idx)]) if int(idx) < len(probs) else 0.0
        agent_details = {
            **details,
            "risk_score": risk,
            "round_idx": int(ctx.round_idx),
            "history_rounds": len(history),
            "checkpoint_dataset": runtime.get("checkpoint_dataset"),
            "checkpoint_scope": runtime.get("checkpoint_scope"),
        }
        if risk >= threshold:
            decisions.append(decision(idx, "block", f"{guard}_gnn_pruned_high_risk_agent", agent_details))
            continue
        filtered[idx] = response
        decisions.append(decision(idx, "allow", f"{guard}_gnn_allow_low_risk_agent", agent_details))
    return filtered, state, decisions
