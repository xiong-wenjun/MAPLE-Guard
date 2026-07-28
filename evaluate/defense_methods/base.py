"""Shared utilities for official-scope communication defenses.

These adapters intentionally operate only on per-round communication outputs.
They do not inspect or mutate persistent-memory writes, retrieval, promotion,
taint, provenance, or lifecycle-risk signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set


@dataclass
class DefenseContext:
    method: str
    task_id: str
    question: str
    round_idx: int
    adj_matrix: Sequence[Sequence[int]]
    args: Any


@dataclass
class OfficialDefenseState:
    inspector_memory: List[Dict[str, str]] = field(default_factory=list)
    challenger_memory: Dict[int, List[Dict[str, str]]] = field(default_factory=dict)
    hexaco_scores: Dict[int, Dict[str, int]] = field(default_factory=dict)
    kicked_agents: Set[int] = field(default_factory=set)
    gnn_state: Dict[str, Any] = field(default_factory=dict)
    guardian_history: List[Dict[int, str]] = field(default_factory=list)
    guardian_inactive_agents: Set[int] = field(default_factory=set)


def text_of_response(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("answer", "") or response)
    return str(response)


def replace_response_text(response: Any, new_text: str) -> Any:
    if isinstance(response, dict):
        updated = dict(response)
        updated["answer"] = new_text
        return updated
    return new_text


def decision(agent_id: int, action: str, reason: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "agent_id": int(agent_id),
        "action": action,
        "reason": reason,
        "details": dict(details or {}),
    }


def ensure_agent_memory(memory: MutableMapping[int, List[Dict[str, str]]], agent_id: int) -> List[Dict[str, str]]:
    if agent_id not in memory:
        memory[agent_id] = []
    return memory[agent_id]


def sorted_outputs(outputs: Mapping[int, Any]) -> List[tuple[int, Any]]:
    return sorted(outputs.items(), key=lambda item: int(item[0]))
