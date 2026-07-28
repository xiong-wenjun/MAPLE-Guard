from __future__ import annotations

from typing import Any, Mapping, Optional


def extract_task_id(meta: Optional[Mapping[str, Any]]) -> Optional[Any]:
    if not meta:
        return None
    for key in ("task_id", "sample_index", "id", "origin_task"):
        if key not in meta:
            continue
        value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None
