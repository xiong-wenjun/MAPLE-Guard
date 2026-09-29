"""Durable, fail-closed task boundaries for new AppWorld provenance_acl runs.

Call load_checkpoint before creating/opening any memory backend; call save_checkpoint
only after a task is complete and its trace has been flushed. One process owns a run.
A snapshot records local logical state, not remote model/server determinism. No pickle,
embedding regeneration, legacy trace reconstruction, or cross-configuration resume.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import stat
import sys
import uuid

SCHEMA = 1
_SERVICE_FIELDS = (
    "_cube_timestamp", "embedding_dim", "default_cube_id",
    "dict_memory", "query_embeddings", "_q_cache", "_mem_cache",
    "_mem_cache_max_size", "_q_cache_max_size", "max_keywords", "memory_confidence",
    "add_similarity_threshold", "enable_value_driven", "weight_sim", "weight_q",
    "sim_norm_mean", "sim_norm_std", "use_z_score_normalization", "dedup_by_task_id",
)


class CheckpointError(RuntimeError):
    pass


class _EntryRef(int):
    """An internal, non-executable object-graph reference."""


def _encode(value):
    """Typed JSON preserves integer keys, tuples, ndarray dtype, and dynamic attrs."""
    if isinstance(value, _EntryRef):
        return {"$type": "entry_ref", "index": int(value)}
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, dict):
        return {"$type": "dict", "items": [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple, set)):
        return {"$type": type(value).__name__, "items": [_encode(v) for v in value]}
    if isinstance(value, datetime):
        return {"$type": "datetime", "value": value.isoformat()}
    if isinstance(value, bytes):
        return {"$type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if type(value).__module__.startswith("numpy"):
        import numpy as np
        arr = np.asarray(value)
        if arr.dtype.hasobject or arr.dtype.fields:
            raise CheckpointError("Unsupported numpy checkpoint dtype")
        return {"$type": "numpy_scalar" if isinstance(value, np.generic) else "ndarray",
                "dtype": arr.dtype.str, "shape": list(arr.shape),
                "value": base64.b64encode(arr.tobytes(order="C")).decode("ascii")}
    if type(value).__module__ == "memos.memories.textual.item" and type(value).__name__ in (
        "TextualMemoryItem", "TextualMemoryMetadata",
    ):
        return {"$type": type(value).__name__, "value": _encode(value.model_dump(mode="python"))}
    raise CheckpointError("Unsupported checkpoint state type: " + type(value).__name__)


def _decode(value):
    if value is None or type(value) in (str, bool, int, float):
        return value
    if not isinstance(value, dict) or "$type" not in value:
        raise CheckpointError("Invalid typed checkpoint state")
    kind = value["$type"]
    if kind == "entry_ref":
        if type(value["index"]) is not int:
            raise CheckpointError("Invalid checkpoint entry reference")
        return _EntryRef(value["index"])
    if kind == "dict":
        pairs = [(_decode(k), _decode(v)) for k, v in value["items"]]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise CheckpointError("Duplicate checkpoint dictionary keys")
        return result
    if kind in ("list", "tuple", "set"):
        items = [_decode(v) for v in value["items"]]
        return {"list": list, "tuple": tuple, "set": set}[kind](items)
    if kind == "datetime":
        return datetime.fromisoformat(value["value"])
    if kind == "bytes":
        return base64.b64decode(value["value"], validate=True)
    if kind in ("ndarray", "numpy_scalar"):
        import numpy as np
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject or dtype.fields:
            raise CheckpointError("Unsupported numpy checkpoint dtype")
        array = np.frombuffer(base64.b64decode(value["value"], validate=True), dtype=dtype).copy()
        array = array.reshape(tuple(value["shape"]))
        return array[()] if kind == "numpy_scalar" else array
    if kind in ("TextualMemoryItem", "TextualMemoryMetadata"):
        from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata
        cls = {"TextualMemoryItem": TextualMemoryItem, "TextualMemoryMetadata": TextualMemoryMetadata}[kind]
        return cls.model_validate(_decode(value["value"]))
    raise CheckpointError("Unknown checkpoint type tag")


def _json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError) as exc:
        raise CheckpointError("Non-JSON checkpoint state") from exc


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _absolute(value):
    path = Path(value).absolute()
    if path != path.resolve():
        raise CheckpointError("Checkpoint paths must not use symlinks")
    return path


def _check_method(args):
    if getattr(args, "method", None) != "provenance_acl" or getattr(args, "task_mode", None) != "appworld":
        raise CheckpointError("Task checkpoints support only AppWorld provenance_acl")


def checkpoint_root(args):
    return _absolute(getattr(args, "task_checkpoint_dir", "") or (str(args.out) + ".checkpoints"))


def _disk_paths(args, trace_path):
    if not os.environ.get("MEMOS_BASE_PATH"):
        raise CheckpointError("Task checkpoints require an explicit isolated MEMOS_BASE_PATH")
    paths = {"memory_store": _absolute(args.memory_store_dir),
             "memos": _absolute(os.environ["MEMOS_BASE_PATH"]) / ".memos",
             "trace": _absolute(trace_path)}
    if os.environ.get("MAPLE_CALL_LOG"):
        paths["api_journal"] = _absolute(os.environ["MAPLE_CALL_LOG"])
    if getattr(args, "baseline_state_path", ""):
        paths["baseline_state"] = _absolute(args.baseline_state_path)
    root = checkpoint_root(args)
    all_paths = list(paths.values()) + [root]
    for index, a in enumerate(all_paths):
        if a == Path(a.anchor):
            raise CheckpointError("Unsafe checkpoint root path")
        for b in all_paths[index + 1:]:
            if a == b or a in b.parents or b in a.parents:
                raise CheckpointError("Checkpoint working paths must not overlap")
    return paths


def build_identity(args):
    """Fingerprint code bytes, resolved options, input bytes, and relevant environment."""
    _check_method(args)
    repo = Path(__file__).resolve().parents[1]
    source = {}
    for name in ("maple_guard", "evaluate"):
        for path in sorted((repo / name).rglob("*.py")):
            source[str(path.relative_to(repo))] = _hash(path)
    for name in ("requirements.txt", "pyproject.toml"):
        path = repo / name
        if path.is_file():
            source[name] = _hash(path)
    config = {key: value for key, value in sorted(vars(args).items())
              if not key.startswith("_") and key != "resume_task_checkpoint"}
    inputs = {}
    for key in ("config", "dataset", "benchmark_bundle", "communication_graph", "baseline_config"):
        value = getattr(args, key, "")
        if isinstance(value, str) and value and Path(value).is_file():
            inputs[key] = {"path": str(Path(value).resolve()), "sha256": _hash(value)}
    # Match the runner's YAML-bundle and four legacy fallback resolutions,
    # including absent files: adding a previously missing fallback changes identity.
    from maple_guard.maple_guard_core import _prompt_dir_from_args, _prompt_file_from_args
    prompt_file = _prompt_file_from_args(args)
    prompt_dir = _prompt_dir_from_args(args)
    prompt_paths = {"bundle": prompt_file}
    if prompt_dir:
        for filename in ("base_system.txt", "attacker_system.txt", "benign_system.txt", "task_user.txt"):
            prompt_paths[filename] = str(Path(prompt_dir) / filename)
    prompts = {}
    for name, value in prompt_paths.items():
        path = Path(value) if value else None
        present = path is not None and path.is_file()
        prompts[name] = {"path": str(path.absolute()) if path is not None else None,
                         "present": present, "sha256": _hash(path) if present else None}
    environment = {key: value for key, value in sorted(os.environ.items())
                   if key.startswith(("CHAT_", "EMBED_", "OPENAI_", "MAPLE_")) or key in ("EMBEDDING_DIM", "MEMOS_BASE_PATH")}
    versions = {"python": sys.version}
    for package in ("numpy", "pydantic", "qdrant-client", "MemoryOS"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"source": source, "config": _encode(config), "inputs": inputs,
            "environment": environment, "versions": versions, "prompts": prompts}


def capture_rng_state():
    result = {"random": random.getstate()}
    if "numpy" in sys.modules:
        import numpy as np
        result["numpy"] = np.random.get_state()
    return result


def restore_rng_state(state):
    random.setstate(state["random"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])


def _backends(bundle):
    return {**{"private:" + str(key): value for key, value in bundle.private_backends.items()},
            "shared": bundle.shared_backend, "quarantine": bundle.quarantine_backend}


def _capture_bundle(bundle):
    objects, refs = [], {}
    def ref(entry):
        marker = id(entry)
        if marker not in refs:
            if not isinstance(entry, bundle.entry_cls):
                raise CheckpointError("Unexpected memory entry class")
            refs[marker] = len(objects)
            objects.append(dict(vars(entry)))
        return refs[marker]
    backends = {}
    for key, backend in _backends(bundle).items():
        service = backend._service
        service_state = None
        if service is not None:
            service_state = {name: getattr(service, name) for name in _SERVICE_FIELDS if hasattr(service, name)}
            for name in _SERVICE_FIELDS[:7]:
                if name not in service_state:
                    raise CheckpointError("Service checkpoint missing " + name)
            service_state["_qdrant_runtime"] = _capture_qdrant(service)
            _validate_service(service_state, backend.user_id)
        elif getattr(backend, "_task_checkpoint_service", None) is not None:
            service_state = backend._task_checkpoint_service
        backends[key] = {
            "user_id": backend.user_id, "name": backend.name,
            "entries": {entry_id: ref(entry) for entry_id, entry in backend.entries.items()},
            "backend_ids": backend.backend_ids, "service": service_state,
        }
    lists = {"private": {key: [ref(e) for e in entries] for key, entries in bundle.private_memories.items()},
             "shared": [ref(e) for e in bundle.shared_memories],
             "quarantine": [ref(e) for e in bundle.quarantine_memories]}
    comparison = getattr(bundle.args, "_full_baseline_runtime", None)
    comparison_state = None
    if comparison is not None:
        from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
        if type(comparison) is not ComparisonRuntime or comparison.method != "provenance_acl" or comparison.guard is not None:
            raise CheckpointError("Unsupported comparison runtime checkpoint")
        def link(value):
            if isinstance(value, bundle.entry_cls):
                return _EntryRef(ref(value))
            if isinstance(value, dict):
                return {k: link(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return type(value)(link(v) for v in value)
            return value
        comparison_state = {key: link(value) for key, value in vars(comparison).items()
                            if key not in {"args", "rules", "guard"}}
    return {"objects": objects, "backends": backends, "lists": lists, "comparison": comparison_state}


def _validate_service(state, user_id):
    for key in _SERVICE_FIELDS[:7]:
        if key not in state:
            raise CheckpointError("Service checkpoint missing " + key)
    timestamp = state["_cube_timestamp"]
    if not isinstance(timestamp, str) or not re.fullmatch(r"[0-9]{8}_[0-9]{6}", timestamp):
        raise CheckpointError("Invalid checkpoint cube timestamp")
    if state["default_cube_id"] != f"cube_{user_id}_{timestamp}":
        raise CheckpointError("Checkpoint cube identity mismatch")
    if not isinstance(state["embedding_dim"], int) or state["embedding_dim"] <= 0:
        raise CheckpointError("Invalid checkpoint embedding dimension")
    if any(not isinstance(state[key], dict) for key in ("dict_memory", "query_embeddings", "_q_cache", "_mem_cache")):
        raise CheckpointError("Invalid checkpoint service cache")
    if state.get("_qdrant_runtime") is not None:
        _validate_qdrant(state["_qdrant_runtime"])


_QDRANT_FIELDS = ("vectors", "payload", "deleted", "deleted_per_vector", "ids", "ids_inv", "_all_vectors_keys")


def _local_collection(service):
    vector_db = service.mos.mem_cubes[service.default_cube_id].text_mem.vector_db
    local = vector_db.client._client
    if type(local).__module__ != "qdrant_client.local.qdrant_local":
        raise CheckpointError("Durable checkpoints require local Qdrant")
    name = vector_db.config.collection_name
    if set(local.collections) != {name}:
        raise CheckpointError("Unexpected Qdrant collection layout")
    collection = local.collections[name]
    if collection.sparse_vectors or collection.multivectors:
        raise CheckpointError("Only dense Qdrant collections support durable task resume")
    return name, collection


def _capture_qdrant(service):
    # Lightweight offline fake services have no MOS; every real service must
    # expose its local collection, or capture fails closed.
    if not hasattr(service, "mos"):
        return None
    name, collection = _local_collection(service)
    state = {"collection": name, "config": collection.config.model_dump(mode="json"),
             "fields": {key: getattr(collection, key) for key in _QDRANT_FIELDS}}
    _validate_qdrant(state)
    return state


def _validate_qdrant(state):
    import numpy as np
    if set(state) != {"collection", "config", "fields"} or set(state["fields"]) != set(_QDRANT_FIELDS):
        raise CheckpointError("Invalid Qdrant runtime schema")
    fields = state["fields"]
    count = len(fields["payload"])
    if len(fields["ids"]) != count or len(fields["ids_inv"]) != count:
        raise CheckpointError("Invalid Qdrant ID/payload alignment")
    if fields["ids"] != {mid: idx for idx, mid in enumerate(fields["ids_inv"])}:
        raise CheckpointError("Invalid Qdrant ID mapping")
    if not isinstance(fields["deleted"], np.ndarray) or fields["deleted"].shape != (count,):
        raise CheckpointError("Invalid Qdrant deletion mask")
    if set(fields["vectors"]) != set(fields["_all_vectors_keys"]) or set(fields["deleted_per_vector"]) != set(fields["vectors"]):
        raise CheckpointError("Invalid Qdrant vector names")
    for name, vectors in fields["vectors"].items():
        mask = fields["deleted_per_vector"][name]
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[0] < count:
            raise CheckpointError("Invalid Qdrant dense vector array")
        if not isinstance(mask, np.ndarray) or mask.shape != (count,):
            raise CheckpointError("Invalid Qdrant per-vector deletion mask")


def _restore_qdrant(service, state):
    if state is None:
        return
    _validate_qdrant(state)
    name, collection = _local_collection(service)
    if name != state["collection"] or collection.config.model_dump(mode="json") != state["config"]:
        raise CheckpointError("Reopened Qdrant collection configuration mismatch")
    fields = state["fields"]
    live_ids = {mid for mid, idx in fields["ids"].items() if not fields["deleted"][idx]}
    if set(collection.ids) != live_ids:
        raise CheckpointError("Reopened Qdrant points do not match saved runtime")
    # Qdrant's persisted raw vectors and its normalized RAM arrays are distinct
    # state. Restore the arrays and dtype exactly without rewriting SQLite.
    for key, value in fields.items():
        setattr(collection, key, value)


def recreate_service(factory, kwargs, state):
    """Construct the saved cube lazily without an embedding request or RNG drift."""
    _validate_service(state, kwargs["user_id"])
    kwargs.update(resume_cube_timestamp=state["_cube_timestamp"], embedding_dim=state["embedding_dim"])
    rng = capture_rng_state()
    try:
        service = factory(**kwargs)
        if service._cube_timestamp != state["_cube_timestamp"] or service.default_cube_id != state["default_cube_id"]:
            raise CheckpointError("Recreated service cube identity mismatch")
        for name, value in state.items():
            if name != "_qdrant_runtime":
                setattr(service, name, value)
        _restore_qdrant(service, state.get("_qdrant_runtime"))
        service.embedding_provider.embedding_dim = state["embedding_dim"]
        service.embedding_provider.fallback_dim = state["embedding_dim"]
        return service
    finally:
        restore_rng_state(rng)


def _copy_file(source, target, sqlite_backup):
    with source.open("rb") as handle:
        is_sqlite = handle.read(16) == b"SQLite format 3\x00"
    if sqlite_backup and is_sqlite:
        # SQLite's backup API includes committed WAL pages; copying raw files does not.
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
            with sqlite3.connect(target) as dest:
                src.backup(dest)
                check = dest.execute("PRAGMA quick_check").fetchall()
                if check != [("ok",)]:
                    raise CheckpointError("SQLite snapshot integrity failure")
        os.chmod(target, 0o600)
    else:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with source.open("rb") as src, os.fdopen(fd, "wb") as dest:
            shutil.copyfileobj(src, dest)
            dest.flush()
            os.fsync(dest.fileno())
    with target.open("rb") as handle:
        os.fsync(handle.fileno())


def _copy_path(source, target, *, sqlite_backup):
    mode = source.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise CheckpointError("Checkpoint refuses symlinks: " + str(source))
    if stat.S_ISREG(mode):
        _copy_file(source, target, sqlite_backup)
        return
    if not stat.S_ISDIR(mode):
        raise CheckpointError("Checkpoint refuses special files: " + str(source))
    _private_mkdir(target)
    children = sorted(source.iterdir())
    sqlite_files = set()
    if sqlite_backup:
        for child in children:
            if child.is_file() and not child.is_symlink():
                with child.open("rb") as handle:
                    if handle.read(16) == b"SQLite format 3\x00":
                        sqlite_files.add(child.name)
    for child in children:
        if sqlite_backup and any(child.name == name + suffix for name in sqlite_files for suffix in ("-wal", "-shm", "-journal")):
            if child.is_symlink():
                raise CheckpointError("Checkpoint refuses SQLite sidecar symlink")
            continue
        _copy_path(child, target / child.name, sqlite_backup=sqlite_backup)
    _fsync_dir(target)


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CheckpointError("Checkpoint integrity: symlink in snapshot")
        relative = str(path.relative_to(root))
        if path.is_dir():
            result[relative] = {"type": "directory"}
        elif path.is_file():
            result[relative] = {"type": "file", "size": path.stat().st_size, "sha256": _hash(path)}
        else:
            raise CheckpointError("Checkpoint integrity: special file in snapshot")
    return result


@contextmanager
def _locked(root, lock_name=".checkpoint.lock"):
    import fcntl
    _private_mkdir(root)
    fd = os.open(root / lock_name, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(fd, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CheckpointError("Checkpoint operation already active") from exc
        yield
    finally:
        os.close(fd)


@contextmanager
def run_lock(args):
    """Hold this for the entire run, including restore and all live backend use."""
    _check_method(args)
    with _locked(checkpoint_root(args), ".run.lock"):
        yield


def _validate_runtime(runtime, args):
    if set(runtime) != {"stream_state", "bundle", "rng"}:
        raise CheckpointError("Invalid checkpoint runtime schema")
    state = runtime["stream_state"]
    index = state.get("next_task_index")
    if type(index) is not int or index < 0 or len(state.get("records", [])) != index:
        raise CheckpointError("Checkpoint boundary does not match completed records")
    if index > len(state.get("task_ids", [])):
        raise CheckpointError("Checkpoint task index outside stream")
    for position, record in enumerate(state["records"]):
        if (not isinstance(record, dict) or record.get("task_index") != position
                or record.get("task_id") != state["task_ids"][position]):
            raise CheckpointError("Checkpoint records do not match completed task prefix")
    random.Random().setstate(runtime["rng"]["random"])
    if "numpy" in runtime["rng"]:
        import numpy as np
        np.random.RandomState().set_state(runtime["rng"]["numpy"])
    bundle = runtime["bundle"]
    expected = {"private:" + str(i) for i in range(args.agents)} | {"shared", "quarantine"}
    if set(bundle["backends"]) != expected:
        raise CheckpointError("Checkpoint backend count mismatch")
    objects = bundle["objects"]
    def validate_ref(ref):
        if type(ref) is not int or not 0 <= ref < len(objects):
            raise CheckpointError("Invalid checkpoint entry reference")
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get("memory_id"), str):
            raise CheckpointError("Invalid checkpoint entry attributes")
    for backend in bundle["backends"].values():
        if backend["service"] is not None:
            _validate_service(backend["service"], backend["user_id"])
        elif backend["entries"] or backend["backend_ids"]:
            raise CheckpointError("Backend entries require an initialized service checkpoint")
        if set(backend["entries"]) != set(backend["backend_ids"]):
            raise CheckpointError("Checkpoint backend IDs do not match entries")
        for entry_id, ref in backend["entries"].items():
            validate_ref(ref)
            if objects[ref]["memory_id"] != entry_id:
                raise CheckpointError("Checkpoint memory ID mismatch")
    lists = bundle["lists"]
    if set(lists["private"]) != set(range(args.agents)):
        raise CheckpointError("Checkpoint private memory lists mismatch")
    for values in [*lists["private"].values(), lists["shared"], lists["quarantine"]]:
        for ref in values:
            validate_ref(ref)
    def validate_links(value):
        if isinstance(value, _EntryRef):
            validate_ref(int(value))
        elif isinstance(value, dict):
            for item in value.values():
                validate_links(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                validate_links(item)
    comparison = bundle.get("comparison")
    if comparison is not None:
        if comparison.get("method") != "provenance_acl" or not isinstance(comparison.get("_embedding_cache"), dict):
            raise CheckpointError("Invalid comparison runtime checkpoint")
        validate_links(comparison)


def save_checkpoint(args, bundle, stream_state, trace_path):
    """Atomically publish an immutable completed-task generation and hashed pointer."""
    _check_method(args)
    paths = _disk_paths(args, trace_path)
    if str(paths["trace"]) != str(_absolute(args.out)):
        raise CheckpointError("Trace path does not match run output")
    runtime = {"stream_state": stream_state, "bundle": _capture_bundle(bundle), "rng": capture_rng_state()}
    _validate_runtime(runtime, args)
    encoded = _encode(runtime)  # Reject unsupported state before creating any generation.
    identity = build_identity(args)
    root = checkpoint_root(args)
    with _locked(root):
        if (root / "latest.json").exists():
            _read_validated(args, identity)
        name = f"task-{stream_state['next_task_index']:06d}-{uuid.uuid4().hex}"
        stage = root / (".building-" + name)
        final = root / name
        _private_mkdir(stage)
        try:
            _write(stage / "runtime.json", encoded)
            _private_mkdir(stage / "disk")
            specs = {}
            for label, source in paths.items():
                present = source.exists() or source.is_symlink()
                specs[label] = {"path": str(source), "present": present}
                if present:
                    _copy_path(source, stage / "disk" / label, sqlite_backup=True)
            if not specs["trace"]["present"]:
                raise CheckpointError("Completed checkpoint requires a trace file")
            trace_lines = (stage / "disk" / "trace").read_bytes().splitlines()
            if len(trace_lines) != stream_state["next_task_index"]:
                raise CheckpointError("Trace prefix length does not match completed task boundary")
            manifest = {"schema": SCHEMA, "identity": identity, "paths": specs,
                        "next_task_index": stream_state["next_task_index"], "files": _inventory(stage)}
            _write(stage / "manifest.json", manifest)
            _fsync_dir(stage)
            os.replace(stage, final)
            _fsync_dir(root)
            pointer = {"generation": name, "manifest_sha256": _hash(final / "manifest.json")}
            pending = root / (".latest-" + uuid.uuid4().hex + ".json")
            _write(pending, pointer)
            os.replace(pending, root / "latest.json")
            _fsync_dir(root)
            return final
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise


def _read_validated(args, expected_identity):
    root = checkpoint_root(args)
    try:
        pointer_path = root / "latest.json"
        if pointer_path.is_symlink():
            raise CheckpointError("Checkpoint pointer must not be a symlink")
        pointer = json.loads(pointer_path.read_text())
        name = pointer["generation"]
        if not isinstance(name, str) or not re.fullmatch(r"task-[0-9]{6,}-[0-9a-f]{32}", name):
            raise CheckpointError("Invalid checkpoint generation pointer")
        generation = root / name
        if generation.is_symlink():
            raise CheckpointError("Checkpoint generation must not be a symlink")
        manifest_path = generation / "manifest.json"
        if manifest_path.is_symlink() or _hash(manifest_path) != pointer["manifest_sha256"]:
            raise CheckpointError("Checkpoint manifest hash integrity failure")
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != SCHEMA:
            raise CheckpointError("Unsupported checkpoint schema")
        expected = expected_identity if expected_identity is not None else build_identity(args)
        if manifest["identity"] != expected:
            raise CheckpointError("Checkpoint source/config/input/environment identity mismatch")
        inventory = _inventory(generation)
        inventory.pop("manifest.json", None)
        if manifest["files"] != inventory:
            raise CheckpointError("Checkpoint file hash integrity failure")
        paths = _disk_paths(args, args.out)
        if set(paths) != set(manifest["paths"]):
            raise CheckpointError("Checkpoint working paths mismatch")
        for label, path in paths.items():
            spec = manifest["paths"][label]
            if spec["path"] != str(path) or type(spec["present"]) is not bool:
                raise CheckpointError("Checkpoint working path identity mismatch")
            if (generation / "disk" / label).exists() != spec["present"]:
                raise CheckpointError("Checkpoint disk inventory mismatch")
        runtime = _decode(json.loads((generation / "runtime.json").read_text()))
        _validate_runtime(runtime, args)
        index = runtime["stream_state"]["next_task_index"]
        if manifest["next_task_index"] != index or len((generation / "disk" / "trace").read_bytes().splitlines()) != index:
            raise CheckpointError("Checkpoint trace boundary mismatch")
        return generation, manifest, runtime
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError("Invalid or missing durable task checkpoint: " + type(exc).__name__) from exc


def load_checkpoint(args, expected_identity=None):
    """Validate everything, then restore working paths and preserve the failed attempt.

    Memory stores must be closed. Every generation is immutable; restored working
    files are separate copies. Existing paths are renamed into failed-attempt-*.
    """
    _check_method(args)
    root = checkpoint_root(args)
    with _locked(root):
        generation, manifest, runtime = _read_validated(args, expected_identity)
        token = uuid.uuid4().hex
        audit = root / ("failed-attempt-" + token)
        staged, moved, installed = {}, [], []
        # Stage all replacement files before changing any working path.
        try:
            for label, spec in manifest["paths"].items():
                destination = Path(spec["path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.parent.stat().st_dev != root.stat().st_dev:
                    raise CheckpointError("Checkpoint working paths must share a filesystem for atomic recovery")
                if destination.is_symlink():
                    raise CheckpointError("Checkpoint restore refuses working path symlink")
                if spec["present"]:
                    stage = destination.parent / ("." + destination.name + ".restore-" + token)
                    _copy_path(generation / "disk" / label, stage, sqlite_backup=False)
                    staged[label] = stage
            _private_mkdir(audit)
            _write(audit / "recovery.json", {"checkpoint": generation.name, "paths": manifest["paths"]})
            for label, spec in manifest["paths"].items():
                destination = Path(spec["path"])
                if destination.exists():
                    os.replace(destination, audit / label)
                    moved.append((label, destination))
                if spec["present"]:
                    os.replace(staged[label], destination)
                    installed.append(destination)
                _fsync_dir(destination.parent)
            _fsync_dir(audit)
        except Exception:
            # Ordinary exceptions roll back the swap; a killed process can retry from
            # the immutable generation, with all previous attempt directories retained.
            for destination in reversed(installed):
                if destination.is_dir():
                    shutil.rmtree(destination)
                elif destination.exists():
                    destination.unlink()
            for label, destination in reversed(moved):
                os.replace(audit / label, destination)
            raise
        finally:
            for stage in staged.values():
                if stage.is_dir():
                    shutil.rmtree(stage)
                elif stage.exists():
                    stage.unlink()
        return {"stream_state": runtime["stream_state"], "manifest": manifest,
                "checkpoint_path": str(generation), "failed_attempt_path": str(audit),
                "_runtime": runtime}


def restore_bundle(bundle, checkpoint_data):
    """Install validated entry graphs and defer reopening the exact saved cubes."""
    runtime = checkpoint_data["_runtime"]
    _validate_runtime(runtime, bundle.args)
    state = runtime["bundle"]
    backends = _backends(bundle)
    for key, backend in backends.items():
        saved = state["backends"][key]
        if backend._service is not None or backend.entries or backend.backend_ids:
            raise CheckpointError("Restore requires a fresh, unopened backend bundle")
        if backend.user_id != saved["user_id"] or backend.name != saved["name"]:
            raise CheckpointError("Checkpoint backend identity mismatch")
    entries = []
    for attrs in state["objects"]:
        obj = bundle.entry_cls.__new__(bundle.entry_cls)
        vars(obj).update(attrs)
        entries.append(obj)
    for key, backend in backends.items():
        saved = state["backends"][key]
        backend.entries = {entry_id: entries[ref] for entry_id, ref in saved["entries"].items()}
        backend.backend_ids = dict(saved["backend_ids"])
        backend._task_checkpoint_service = saved["service"]
    lists = state["lists"]
    bundle.private_memories = {key: [entries[ref] for ref in values] for key, values in lists["private"].items()}
    bundle.shared_memories = [entries[ref] for ref in lists["shared"]]
    bundle.quarantine_memories = [entries[ref] for ref in lists["quarantine"]]
    if state.get("comparison") is not None:
        from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
        comparison = ComparisonRuntime(bundle.args)
        def unlink(value):
            if isinstance(value, _EntryRef):
                return entries[int(value)]
            if isinstance(value, dict):
                return {k: unlink(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return type(value)(unlink(v) for v in value)
            return value
        for key, value in state["comparison"].items():
            setattr(comparison, key, unlink(value))
        bundle.args._full_baseline_runtime = comparison
    restore_rng_state(runtime["rng"])
