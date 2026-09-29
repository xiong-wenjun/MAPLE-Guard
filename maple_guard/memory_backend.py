from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Type

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

if __package__:
    from .providers.base import BaseEmbedder, BaseLLM
else:  # Direct script execution from maple_guard/.
    from providers.base import BaseEmbedder, BaseLLM

_MEMRL_IMPORT_ERROR: Optional[BaseException] = None
try:
    if __package__:
        from .memrl_service.memory_service import MemoryService
        from .memrl_service.strategies import StrategyConfiguration
        from .memrl_service.value_driven import RLConfig
    else:
        from memrl_service.memory_service import MemoryService
        from memrl_service.strategies import StrategyConfiguration
        from memrl_service.value_driven import RLConfig
except Exception as exc:  # pragma: no cover
    MemoryService = None  # type: ignore[assignment]
    StrategyConfiguration = None  # type: ignore[assignment]
    RLConfig = None  # type: ignore[assignment]
    _MEMRL_IMPORT_ERROR = exc


DEFAULT_CHAT_STOP_SEQUENCES = [
    "<end_of_turn>",
    "<start_of_turn>",
    "<|im_end|>",
    "<|endoftext|>",
]


def chat_stop_sequences() -> List[str]:
    raw = os.environ.get("CHAT_STOP_SEQUENCES", "")
    if raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except Exception:
            pass
        return [item.strip() for item in raw.split("|") if item.strip()]
    return DEFAULT_CHAT_STOP_SEQUENCES


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = (api_key if api_key and api_key != "EMPTY" else os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY")
        self.timeout = timeout
        self.default_max_tokens = int(os.environ.get("CHAT_MAX_TOKENS", "128"))

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if requests is None:
            raise RuntimeError("requests is required for OpenAICompatibleLLM")
        max_tokens = int(kwargs.get("max_tokens") or self.default_max_tokens or 0)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0),
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if os.getenv("CHAT_DISABLE_THINKING") == "1":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        stops = chat_stop_sequences()
        if stops:
            payload["stop"] = stops
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        # Keep retrieval cheap and deterministic for large MAS runs.
        words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
        stop = {"the", "and", "for", "that", "with", "this", "from", "should", "would", "which", "similar", "question", "answer", "option"}
        out: List[str] = []
        for word in words:
            if word in stop or word in out:
                continue
            out.append(word)
            if len(out) >= max_keywords:
                break
        return out or [text[:128]]


def _strict_backend(args, method=""):
    from evaluate.defense_methods.full_runtime import FULL_METHODS, MEMORY_METHODS
    return bool(getattr(args,"strict_comparison",False)) or (method or getattr(args,"method","")) in (*FULL_METHODS,*MEMORY_METHODS)

class OpenAICompatibleEmbedder(BaseEmbedder):
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", fallback_dim: int = 3584, timeout: int = 60, strict: bool = False) -> None:
        super().__init__()
        env_dim = os.environ.get("EMBEDDING_DIM")
        if env_dim:
            fallback_dim = int(env_dim)
        elif "qwen3-embedding-8b" in model.lower():
            fallback_dim = 4096
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = (api_key if api_key and api_key != "EMPTY" else os.getenv("EMBED_API_KEY") or "EMPTY")
        self.fallback_dim = fallback_dim
        self.embedding_dim = fallback_dim
        self.timeout = timeout
        self.strict = strict

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if requests is not None:
            try:
                resp = requests.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": texts},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                embeddings = [item["embedding"] for item in resp.json()["data"]]
                if embeddings and embeddings[0]:
                    self.embedding_dim = len(embeddings[0])
                    self.fallback_dim = self.embedding_dim
                return embeddings
            except Exception as exc:
                if self.strict:
                    raise RuntimeError("Strict baseline embedding request failed; no hash fallback") from exc
        if self.strict:
            raise RuntimeError("Strict baseline embedding requires requests")
        return [self._hash_embedding(text, self.fallback_dim) for text in texts]

    @staticmethod
    def _hash_embedding(text: str, dim: int) -> List[float]:
        vec = [0.0] * dim
        for tok in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def _meta_to_dict(meta: Any) -> Dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    for attr in ("model_dump",):
        fn = getattr(meta, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    out: Dict[str, Any] = {}
    extra = getattr(meta, "model_extra", None)
    if isinstance(extra, dict):
        out.update(extra)
    raw = getattr(meta, "__dict__", None)
    if isinstance(raw, dict):
        out.update(raw)
    return out


def _as_int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    try:
        return [int(value)]
    except Exception:
        return []


def _as_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


_FORBIDDEN_MEMORY_PATTERNS = [
    r"ignore (all )?(previous|prior|system) instructions",
    r"bypass (the )?(safety|security|policy|approval)",
    r"always answer [A-E]",
    r"reuse this shortcut confidently",
    r"should choose option [A-E]",
    r"future similar .* choose option [A-E]",
    r"treated as reliable past team experience",
    r"skip (verification|approval|review)",
    r"do not verify",
]


def _patch_chonkie_sentence_chunker() -> None:
    """Keep MemOS compatible with newer chonkie releases.

    Current MemOS calls chonkie.SentenceChunker(tokenizer_or_token_counter=...),
    while newer chonkie versions renamed that argument to tokenizer.
    """
    try:
        import inspect
        import chonkie

        cls = chonkie.SentenceChunker
        sig = inspect.signature(cls.__init__)
        if "tokenizer_or_token_counter" in sig.parameters or "tokenizer" not in sig.parameters:
            return

        class CompatSentenceChunker(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, tokenizer_or_token_counter: Any = None, **kwargs: Any) -> None:
                if tokenizer_or_token_counter is not None and "tokenizer" not in kwargs:
                    kwargs["tokenizer"] = tokenizer_or_token_counter
                super().__init__(*args, **kwargs)

        chonkie.SentenceChunker = CompatSentenceChunker
    except Exception:
        return


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _risk_tokens(text: str) -> List[str]:
    return [
        tok
        for tok in re.findall(r"[a-z][a-z0-9_]{2,}", (text or "").lower())
        if tok not in {
            "the", "and", "for", "that", "with", "this", "from", "should",
            "would", "which", "question", "answer", "option", "task",
            "current", "future", "similar", "memory", "agent",
        }
    ]


def _query_memory_overlap(query_text: str, memory_text: str) -> float:
    q = set(_risk_tokens(query_text))
    m = set(_risk_tokens(memory_text))
    if not q or not m:
        return 0.0
    return len(q & m) / max(1, min(len(q), len(m)))


def _query_context_risk(query_text: str) -> float:
    text = (query_text or "").lower()
    risk = 0.0
    if any(x in text for x in ("<information>", "external tool output", "tool output", "retrieved external", "untrusted user note")):
        risk += 0.25
    if any(re.search(pattern, text) for pattern in _FORBIDDEN_MEMORY_PATTERNS):
        risk += 0.35
    if any(x in text for x in ("ignore previous", "do not verify", "without re-verifying", "always answer")):
        risk += 0.25
    if any(x in text for x in ("calibration token", "parallel universe", "trigger", "suffix")):
        risk += 0.10
    return _clamp01(risk)


def _instruction_like_text(text: str) -> bool:
    lower = (text or "").lower()
    return any(re.search(pattern, lower) for pattern in _FORBIDDEN_MEMORY_PATTERNS)


def _base_taint_penalty(taint: str) -> float:
    if taint == "external":
        return 0.65
    if taint == "unverified":
        return 0.40
    if taint == "poisoned":
        return 0.90
    return 0.0


def _candidate_scope_violation(md: Dict[str, Any], agent_id: Optional[int], task_class: str) -> float:
    allowed_agents = _as_int_list(md.get("allowed_agents"))
    if agent_id is not None and allowed_agents and int(agent_id) not in allowed_agents:
        return 1.0
    allowed_tasks = _as_str_list(md.get("allowed_task_classes"))
    if allowed_tasks and task_class not in allowed_tasks:
        return 1.0
    return 0.0


def _candidate_text(candidate: Dict[str, Any], md: Dict[str, Any]) -> str:
    return "\n".join(
        str(x)
        for x in (
            md.get("maple_guard_intent") or md.get("task_description") or "",
            md.get("maple_guard_experience") or md.get("full_content") or candidate.get("content") or "",
        )
        if x
    )


def _candidate_content_hazard(md: Dict[str, Any], candidate: Dict[str, Any], query_text: str, similarity: float) -> float:
    memory_text = _candidate_text(candidate, md)
    taint = str(md.get("taint") or "unverified")
    hazard = float(md.get("content_hazard", 0.0) or 0.0)
    q_risk = _query_context_risk(query_text)
    overlap = _query_memory_overlap(query_text, memory_text)
    if _instruction_like_text(memory_text):
        hazard += 0.10 + 0.20 * max(0.0, similarity)
    if taint in ("external", "unverified", "poisoned") and similarity >= 0.35:
        hazard += 0.10
    if overlap >= 0.25 and taint in ("external", "unverified", "poisoned"):
        hazard += 0.10
    if q_risk > 0:
        hazard += 0.15 * q_risk
    lower_memory = memory_text.lower()
    lower_query = (query_text or "").lower()
    if any(x in lower_query for x in ("external tool output", "tool output", "<information>")) and any(x in lower_memory for x in ("tool", "external", "environment", "<information>")):
        hazard += 0.10
    trust = float(md.get("provenance_trust", 0.5) or 0.5)
    if taint == "clean" and trust >= 0.70 and not _instruction_like_text(memory_text):
        hazard -= 0.05
    return _clamp01(hazard)


def _candidate_taint_penalty(md: Dict[str, Any], candidate: Dict[str, Any], query_text: str, similarity: float) -> float:
    taint = str(md.get("taint") or "unverified")
    base = _base_taint_penalty(taint)
    if base <= 0:
        return 0.0
    memory_text = _candidate_text(candidate, md)
    penalty = base
    if similarity >= 0.45:
        penalty += 0.10
    if similarity >= 0.65:
        penalty += 0.10
    if _instruction_like_text(memory_text):
        penalty += 0.10
    if _query_context_risk(query_text) > 0:
        penalty += 0.10
    if _query_memory_overlap(query_text, memory_text) >= 0.25 and taint in ("external", "unverified", "poisoned"):
        penalty += 0.05
    trust = float(md.get("provenance_trust", 0.5) or 0.5)
    hazard = float(md.get("content_hazard", 0.0) or 0.0)
    if trust >= 0.75 and hazard <= 0.20:
        penalty -= 0.10
    return _clamp01(penalty)


def _candidate_maple_guard_score(candidate: Dict[str, Any], query_text: str, agent_id: Optional[int], task_class: str) -> float:
    md = _meta_to_dict(candidate.get("metadata"))
    sim = float(candidate.get("similarity", candidate.get("score", 0.0)) or 0.0)
    q_value = float(candidate.get("q_estimate", md.get("q_value", 0.0)) or 0.0)
    trust = float(md.get("provenance_trust", 0.5) or 0.5)
    hazard = _candidate_content_hazard(md, candidate, query_text, sim)
    taint_pen = _candidate_taint_penalty(md, candidate, query_text, sim)
    scope = _candidate_scope_violation(md, agent_id, task_class)
    score = sim + 0.8 * q_value + 0.5 * trust - hazard - taint_pen - 2.0 * scope
    candidate["maple_guard_backend_score"] = score
    candidate["maple_guard_backend_content_hazard"] = hazard
    candidate["maple_guard_backend_taint_penalty"] = taint_pen
    candidate["maple_guard_backend_scope_violation"] = scope
    return score


class MemRLMemoryBackend:
    def __init__(self, *, name: str, user_id: str, args: Any, entry_cls: Type[Any]) -> None:
        self.name = name
        self.user_id = user_id
        self.args = args
        self.entry_cls = entry_cls
        self.entries: Dict[str, Any] = {}
        self.backend_ids: Dict[str, str] = {}
        self._service: Optional[Any] = None
        self._mos_config_path: Optional[str] = None

    def _ensure_service(self) -> Any:
        if self._service is not None:
            return self._service
        if MemoryService is None or StrategyConfiguration is None or RLConfig is None:
            raise RuntimeError(f"MemRL service import failed: {_MEMRL_IMPORT_ERROR}")
        _patch_chonkie_sentence_chunker()

        store_dir = os.path.abspath(getattr(self.args, "memory_store_dir", "PI/result_maple_guard/memory_store"))
        os.makedirs(store_dir, exist_ok=True)
        cfg_dir = os.path.join(store_dir, "configs", self.user_id)
        os.makedirs(cfg_dir, exist_ok=True)
        db_path = os.path.join(cfg_dir, "users.db")
        api_key = getattr(self.args, "api_key", "") or os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        embed_api_key = getattr(self.args, "embed_api_key", "") or os.getenv("EMBED_API_KEY") or api_key
        mos_config = {
            "chat_model": {
                "backend": "openai",
                "config": {
                    "model_name_or_path": self.args.chat_model,
                    "api_key": api_key,
                    "api_base": self.args.chat_base_url,
                },
            },
            "mem_reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": {
                        "backend": "openai",
                        "config": {
                            "model_name_or_path": self.args.chat_model,
                            "api_key": api_key,
                            "api_base": self.args.chat_base_url,
                        },
                    },
                    "embedder": {
                        "backend": "universal_api",
                        "config": {
                            "provider": "openai",
                            "model_name_or_path": self.args.embed_model,
                            "api_key": embed_api_key,
                            "base_url": self.args.embed_base_url,
                        },
                    },
                    "chunker": {"backend": "sentence", "config": {"tokenizer_or_token_counter": "character", "chunk_size": 5000, "chunk_overlap": 0}},
                },
            },
            "user_manager": {"backend": "sqlite", "config": {"db_path": db_path}},
            "top_k": int(getattr(self.args, "top_k_memory", 3)),
        }
        self._mos_config_path = os.path.join(cfg_dir, "mos_config.json")
        fd = os.open(self._mos_config_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(mos_config, f, ensure_ascii=False, indent=2)

        strategy = StrategyConfiguration.from_strings(
            getattr(self.args, "memrl_build", "trajectory"),
            getattr(self.args, "memrl_retrieve", "query"),
            getattr(self.args, "memrl_update", "vanilla"),
        )
        rl_config = RLConfig(
            topk=int(getattr(self.args, "top_k_memory", 3)),
            weight_sim=float(getattr(self.args, "memrl_weight_sim", 0.5)),
            weight_q=float(getattr(self.args, "memrl_weight_q", 0.5)),
        )
        llm = OpenAICompatibleLLM(self.args.chat_base_url, self.args.chat_model, api_key=api_key)
        embedder = OpenAICompatibleEmbedder(self.args.embed_base_url, self.args.embed_model, api_key=embed_api_key, strict=_strict_backend(self.args))
        service_kwargs = dict(
            mos_config_path=self._mos_config_path,
            llm_provider=llm,
            embedding_provider=embedder,
            strategy_config=strategy,
            user_id=self.user_id,
            num_workers=1,
            enable_value_driven=bool(getattr(self.args, "memrl_enable_value_driven", True)),
            rl_config=rl_config,
            base_root=os.path.join(store_dir, "mem_cubes"),
            mem_cache_max_size=20000,
        )
        resume_state = getattr(self, "_task_checkpoint_service", None)
        if resume_state is not None:
            if __package__:
                from .task_checkpoint import recreate_service
            else:
                from task_checkpoint import recreate_service
            self._service = recreate_service(MemoryService, service_kwargs, resume_state)
            self._task_checkpoint_service = None
        else:
            self._service = MemoryService(**service_kwargs)
        return self._service

    def add_entry(self, entry: Any, success: Optional[bool] = None) -> str:
        service = self._ensure_service()
        success_flag = bool(entry.utility_q >= 0.0) if success is None else bool(success)
        metadata = self._entry_to_metadata(entry, success_flag)
        backend_id = service.add_memory(
            task_description=str(entry.intent),
            trajectory=str(entry.experience),
            success=success_flag,
            metadata=metadata,
        )
        if backend_id is None:
            backend_id = str(entry.memory_id)
        self.entries[str(entry.memory_id)] = entry
        self.backend_ids[str(entry.memory_id)] = str(backend_id)
        setattr(entry, "backend_memory_id", str(backend_id))
        # Storage ownership differs from provenance: a peer's message is stored
        # in the recipient's private cube, while origin_agent remains its author.
        setattr(entry, "backend_store_id", self.user_id)
        return str(backend_id)

    def retrieve_entries(
        self,
        query: str,
        top_k: int,
        threshold: float = 0.0,
        *,
        method: str = "",
        agent_id: Optional[int] = None,
        task_class: str = "csqa",
    ) -> List[Any]:
        service = self._ensure_service()
        try:
            raw = service.retrieve_query(task_description=query, k=max(top_k, 1), threshold=max(threshold, 0.0))
        except Exception as exc:
            if _strict_backend(self.args, method):
                raise RuntimeError("Strict baseline backend retrieval failed; no strategy fallback") from exc
            raw = service.retrieve_value_aware(task_description=query, k=max(top_k, 1), threshold=max(threshold, 0.0))
        result = raw[0] if isinstance(raw, tuple) else raw
        if bool(getattr(self.args, "strict_comparison", False)) or method in ("provenance_acl", "maple_guard_retrieval_only", "amemguard_full", "piguard_retrieval", "piguard_lifecycle"):
            selected = result.get("candidates") or result.get("selected") or []
        elif str(method) in ("maple_guard", "maple_guard_retrieval_only"):
            pool = result.get("candidates") or result.get("selected") or []
            selected = [pool] if isinstance(pool, dict) else list(pool)
            selected.sort(
                key=lambda candidate: _candidate_maple_guard_score(candidate, query, agent_id, task_class),
                reverse=True,
            )
        else:
            selected = result.get("selected") or result.get("candidates") or []
        if isinstance(selected, dict):
            selected = [selected]
        out: List[Any] = []
        for candidate in selected[:top_k]:
            entry = self._candidate_to_entry(candidate)
            if entry is not None:
                out.append(entry)
        return out

    def update_value(self, entry: Any, success: bool) -> None:
        maple_guard_id = str(getattr(entry, "memory_id", ""))
        backend_id = self.backend_ids.get(maple_guard_id) or getattr(entry, "backend_memory_id", None)
        if not backend_id:
            if _strict_backend(self.args):
                raise RuntimeError("Strict baseline feedback requires a registered memory ID")
            return
        try:
            service = self._ensure_service()
            reward = 1.0 if success else -1.0
            new_q = service.update_value(str(backend_id), reward)
            if new_q is not None:
                entry.utility_q = float(new_q)
                if maple_guard_id in self.entries:
                    self.entries[maple_guard_id].utility_q = float(new_q)
        except Exception as exc:
            if _strict_backend(self.args):
                raise RuntimeError("Strict baseline feedback update failed") from exc
            return

    def _entry_to_metadata(self, entry: Any, success: bool) -> Dict[str, Any]:
        return {
            "maple_guard_id": str(entry.memory_id),
            "maple_guard_backend": "memrl_service",
            "maple_guard_intent": str(entry.intent),
            "maple_guard_experience": str(entry.experience),
            "task_id": str(entry.origin_task),
            "origin_task": str(entry.origin_task),
            "origin_agent": int(entry.origin_agent),
            "origin_round": int(entry.origin_round),
            "source_type": str(entry.source_type),
            "memory_type": str(entry.memory_type),
            "memory_scope": str(entry.memory_scope),
            "allowed_agents": list(entry.allowed_agents),
            "allowed_task_classes": list(entry.allowed_task_classes),
            "allowed_tools": list(entry.allowed_tools),
            "provenance_trust": float(entry.provenance_trust),
            "content_hazard": float(entry.content_hazard),
            "taint": str(entry.taint),
            "status": "activated",
            "maple_guard_status": str(entry.status),
            "baseline_metadata": dict(getattr(entry, "baseline_metadata", {})),
            "parents": list(entry.parents),
            "q_value": float(entry.utility_q),
            "success": bool(success),
            "created_at": time.time(),
        }

    def _candidate_to_entry(self, candidate: Dict[str, Any]) -> Optional[Any]:
        md = _meta_to_dict(candidate.get("metadata"))
        maple_guard_id = str(md.get("maple_guard_id") or candidate.get("memory_id") or "")
        if not maple_guard_id:
            return None
        backend_id = str(candidate.get("memory_id") or self.backend_ids.get(maple_guard_id) or maple_guard_id)
        if maple_guard_id in self.entries:
            entry = self.entries[maple_guard_id]
            try:
                entry.utility_q = float(candidate.get("q_estimate", md.get("q_value", entry.utility_q)))
            except Exception:
                pass
        else:
            entry = self.entry_cls(
                memory_id=maple_guard_id,
                intent=str(md.get("maple_guard_intent") or md.get("task_description") or candidate.get("content") or ""),
                experience=str(md.get("maple_guard_experience") or candidate.get("content") or md.get("full_content") or ""),
                utility_q=float(candidate.get("q_estimate", md.get("q_value", 0.0)) or 0.0),
                origin_task=str(md.get("origin_task") or md.get("task_id") or "unknown"),
                origin_agent=int(md.get("origin_agent", -1) if md.get("origin_agent", -1) is not None else -1),
                origin_round=int(md.get("origin_round", 0) or 0),
                source_type=str(md.get("source_type") or md.get("source") or "memrl"),
                memory_type=str(md.get("memory_type") or md.get("type") or "procedure"),
                memory_scope=str(md.get("memory_scope") or "agent_private"),
                allowed_agents=_as_int_list(md.get("allowed_agents")),
                allowed_task_classes=_as_str_list(md.get("allowed_task_classes")),
                allowed_tools=_as_str_list(md.get("allowed_tools")),
                provenance_trust=float(md.get("provenance_trust", 0.5) or 0.5),
                content_hazard=float(md.get("content_hazard", 0.0) or 0.0),
                taint=str(md.get("taint") or "unverified"),
                status=str(md.get("maple_guard_status") or md.get("status") or "active").replace("activated", "active"),
                baseline_metadata=dict(md.get("baseline_metadata") or {}),
                parents=_as_str_list(md.get("parents")),
            )
            self.entries[maple_guard_id] = entry
        self.backend_ids[maple_guard_id] = backend_id
        setattr(entry, "backend_memory_id", backend_id)
        # Reconstructed entries take their physical owner from this backend,
        # never from author/scope fields in memory content or metadata.
        setattr(entry, "backend_store_id", self.user_id)
        setattr(entry, "backend_score", candidate.get("score"))
        setattr(entry, "backend_similarity", candidate.get("similarity"))
        setattr(entry, "backend_maple_guard_score", candidate.get("maple_guard_backend_score"))
        setattr(entry, "backend_maple_guard_content_hazard", candidate.get("maple_guard_backend_content_hazard"))
        setattr(entry, "backend_maple_guard_taint_penalty", candidate.get("maple_guard_backend_taint_penalty"))
        setattr(entry, "backend_maple_guard_scope_violation", candidate.get("maple_guard_backend_scope_violation"))
        return entry


class MemoryBackendBundle:
    def __init__(self, args: Any, entry_cls: Type[Any]) -> None:
        self.args = args
        self.entry_cls = entry_cls
        run_key = getattr(args, "memory_run_id", "") or os.path.basename(getattr(args, "out", "maple_guard"))
        run_id = self._safe_run_id(run_key)
        self.private_backends: Dict[int, MemRLMemoryBackend] = {
            agent_id: MemRLMemoryBackend(name=f"agent_{agent_id}", user_id=f"{run_id}_agent_{agent_id}", args=args, entry_cls=entry_cls)
            for agent_id in range(int(args.agents))
        }
        self.shared_backend = MemRLMemoryBackend(name="shared", user_id=f"{run_id}_shared", args=args, entry_cls=entry_cls)
        self.quarantine_backend = MemRLMemoryBackend(name="quarantine", user_id=f"{run_id}_quarantine", args=args, entry_cls=entry_cls)
        self.private_memories: Dict[int, List[Any]] = {agent_id: [] for agent_id in range(int(args.agents))}
        self.shared_memories: List[Any] = []
        self.quarantine_memories: List[Any] = []

    @staticmethod
    def _safe_run_id(text: str) -> str:
        # Qdrant local mode stores SQLite under
        # <store>/qdrant/<user_id>/<timestamp>/collection/<collection>/storage.sqlite.
        # Long experiment names can push that path past SQLite's pathname limit, so
        # keep the readable prefix short and rely on the hash for uniqueness.
        base = re.sub(r"[^A-Za-z0-9_]+", "_", text.replace(".jsonl", "")).strip("_")[:24]
        suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        return f"maple_guard_{base}_{suffix}"

    def add_private(self, agent_id: int, entry: Any) -> None:
        self.private_backends[int(agent_id)].add_entry(entry)
        self.private_memories.setdefault(int(agent_id), []).append(entry)

    def add_shared(self, entry: Any) -> None:
        self.shared_backend.add_entry(entry)
        self.shared_memories.append(entry)

    def add_quarantine(self, entry: Any) -> None:
        self.quarantine_backend.add_entry(entry, success=False)
        self.quarantine_memories.append(entry)

    def retrieve_private(self, agent_id: int, query: str, top_k: int, threshold: float, *, method: str = "", task_class: str = "csqa") -> List[Any]:
        return self.private_backends[int(agent_id)].retrieve_entries(query, top_k=top_k, threshold=threshold, method=method, agent_id=int(agent_id), task_class=task_class)

    def retrieve_shared(self, query: str, top_k: int, threshold: float, *, method: str = "", agent_id: Optional[int] = None, task_class: str = "csqa") -> List[Any]:
        return self.shared_backend.retrieve_entries(query, top_k=top_k, threshold=threshold, method=method, agent_id=agent_id, task_class=task_class)

    def update_value(self, entry: Any, success: bool) -> None:
        store_id = getattr(entry, "backend_store_id", None)
        stores = [*self.private_backends.values(), self.shared_backend, self.quarantine_backend]
        if store_id is not None:
            for backend in stores:
                if backend.user_id == store_id:
                    backend.update_value(entry, success)
                    return
            raise RuntimeError("Feedback references a memory store outside this experiment")
        if _strict_backend(self.args):
            raise RuntimeError("Strict baseline feedback requires a registered physical memory store")
        # Preserve the historical in-memory/legacy path for entries without a
        # physical store marker; strict runs must never infer ownership from authorship.
        scope = getattr(entry, "memory_scope", "agent_private")
        if scope == "team":
            self.shared_backend.update_value(entry, success)
        else:
            agent_id = int(getattr(entry, "origin_agent", -1))
            if agent_id in self.private_backends:
                self.private_backends[agent_id].update_value(entry, success)


def create_memory_backend_bundle(args: Any, entry_cls: Type[Any]) -> MemoryBackendBundle:
    return MemoryBackendBundle(args, entry_cls)
