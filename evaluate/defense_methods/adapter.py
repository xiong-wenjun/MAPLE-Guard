"""Dispatcher matching INFA-Guard's evaluate/defense_methods style."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .agentsafe_defense import apply_agentsafe
from .agentxposed_defense import apply_agentxposed
from .base import DefenseContext, OfficialDefenseState
from .challenger_defense import apply_challenger
from .guardian_defense import apply_guardian
from .inspector_defense import apply_inspector


METHOD_TO_GUARD = {
    "gsafeguard": "gsafeguard",
    "infa_guard": "ours",
    "agentsafe": "agentsafe",
    "agentxposed_guide": "agentxposed-guide",
    "agentxposed_kick": "agentxposed-kick",
    "agentxposed": "agentxposed-guide",
    "challenger": "challenger",
    "guardian": "guardian",
    "inspector": "inspector",
    # Backward-compatible aliases from the first integration pass.
    "official_gsafeguard_memrl": "gsafeguard",
    "official_infa_guard_memrl": "ours",
    "official_agentsafe_memrl": "agentsafe",
    "official_agentxposed_guide_memrl": "agentxposed-guide",
    "official_agentxposed_kick_memrl": "agentxposed-kick",
    "official_challenger_memrl": "challenger",
    "official_guardian_memrl": "guardian",
    "official_inspector_memrl": "inspector",
    "agentxposed": "agentxposed-guide",
    "agentxposed-guide": "agentxposed-guide",
    "agentxposed-kick": "agentxposed-kick",
}


def apply_official_communication_defense(
    method: str,
    outputs: Dict[int, Any],
    state: Optional[OfficialDefenseState],
    *,
    task_id: str,
    question: str,
    round_idx: int,
    adj_matrix,
    args: Any,
):
    guard = METHOD_TO_GUARD.get(method)
    if guard is None:
        return outputs, state, []
    state = state or OfficialDefenseState()
    ctx = DefenseContext(method=method, task_id=task_id, question=question, round_idx=round_idx, adj_matrix=adj_matrix, args=args)
    if guard == "agentsafe":
        return apply_agentsafe(outputs, state, ctx)
    if guard == "agentxposed-guide":
        return apply_agentxposed(outputs, state, ctx, mode="guide")
    if guard == "agentxposed-kick":
        return apply_agentxposed(outputs, state, ctx, mode="kick")
    if guard == "challenger":
        return apply_challenger(outputs, state, ctx)
    if guard == "guardian":
        return apply_guardian(outputs, state, ctx)
    if guard == "inspector":
        return apply_inspector(outputs, state, ctx)
    if guard in {"gsafeguard", "ours"}:
        from .gnn_defense import apply_gnn_official

        return apply_gnn_official(outputs, state, ctx, guard="gsafeguard" if guard == "gsafeguard" else "infa_guard")
    return outputs, state, []
