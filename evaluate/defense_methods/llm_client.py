"""Minimal OpenAI-compatible safeguard client used by official baselines."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


def safeguard_config(args: Any) -> tuple[str, str, str]:
    base_url = (
        getattr(args, "safeguard_base_url", "")
        or os.getenv("SAFEGUARD_BASE_URL", "")
        or getattr(args, "chat_base_url", "")
    )
    model = (
        getattr(args, "safeguard_model", "")
        or os.getenv("SAFEGUARD_MODEL", "")
        or getattr(args, "chat_model", "")
    )
    api_key = getattr(args, "safeguard_api_key", "") or os.getenv("SAFEGUARD_OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    return str(base_url or ""), str(model or ""), str(api_key or "")


def chat_completion(
    args: Any,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed")
    base_url, model, api_key = safeguard_config(args)
    if not base_url or not model:
        raise RuntimeError("SAFEGUARD_BASE_URL/SAFEGUARD_MODEL or chat_base_url/chat_model is required")
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise

