"""Task boundary checkpoint tests: offline state/disk recovery, never API traffic."""
import importlib.util
import json
import os
from pathlib import Path
import random
import sqlite3
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


@dataclass
class Entry:
    memory_id: str
    intent: str
    utility_q: float = 0.25


class TaskCheckpointTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec("maple_guard.task_checkpoint")
        self.assertIsNotNone(spec, "durable task checkpoint module must exist")
        from maple_guard import task_checkpoint
        from maple_guard.memory_backend import MemoryBackendBundle
        self.ck = task_checkpoint
        self.Bundle = MemoryBackendBundle
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"MEMOS_BASE_PATH": str(self.root / "memos")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.args = SimpleNamespace(
            method="provenance_acl", task_mode="appworld", agents=2, seed=4,
            memory_store_dir=str(self.root / "store"), memory_run_id="test",
            task_checkpoint_dir=str(self.root / "checkpoints"),
            resume_task_checkpoint=False, out=str(self.root / "trace.jsonl"),
            baseline_state_path=str(self.root / "baseline.json"),
            chat_model="offline", chat_base_url="http://unused", embed_model="offline",
            embed_base_url="http://unused", api_key="test-secret", top_k_memory=3,
        )
        self.store = Path(self.args.memory_store_dir)
        self.store.mkdir()
        (self.store / "config.json").write_bytes(b'{"test":"secret"}\n')
        (self.root / "memos" / ".memos").mkdir(parents=True)
        (self.root / "memos" / ".memos" / "state").write_bytes(b"registry")
        Path(self.args.baseline_state_path).write_bytes(b'{"baseline":1}\n')
        Path(self.args.out).write_bytes(b'{"task_id":"task-0","task_index":0}\n')
        self.bundle = self.Bundle(self.args, Entry)
        entry = Entry("m1", "original query")
        entry.backend_memory_id = "qdrant-id"
        entry.backend_store_id = self.bundle.private_backends[0].user_id
        entry.dynamic = {"tuple": (1, 2), "set": {"a", "b"}, "integer-key": {1: 0.5}}
        self.bundle.private_memories[0] = [entry]
        backend = self.bundle.private_backends[0]
        backend.entries = {"m1": entry}
        backend.backend_ids = {"m1": "qdrant-id"}
        backend._service = SimpleNamespace(
            _cube_timestamp="20260929_080001", embedding_dim=2,
            default_cube_id=f"cube_{backend.user_id}_20260929_080001",
            dict_memory={"original query": ["qdrant-id"], "empty query": []},
            query_embeddings={"original query": np.array([0.125, 0.25], dtype="float32")},
            _q_cache={"qdrant-id": 0.25}, _mem_cache={"qdrant-id": {"memory": "text", "metadata": {}}},
            embedding_provider=SimpleNamespace(embedding_dim=2, fallback_dim=2),
        )
        self.state = {
            "next_task_index": 1, "task_ids": ["task-0", "task-1"], "poison_indices": [0],
            "records": [{"task_id": "task-0", "task_index": 0}], "agent_trust": {0: 0.6, 1: 0.5},
            "poison_targets": {"m1": "B"}, "args_poison_target_cache": {"task-0": "B"},
        }

    def save(self):
        return self.ck.save_checkpoint(self.args, self.bundle, self.state, self.args.out)

    def load(self):
        self.args.resume_task_checkpoint = True
        return self.ck.load_checkpoint(self.args)

    def test_complete_runtime_roundtrip_preserves_aliases_dynamic_attributes_and_rng(self):
        random.seed(22)
        np.random.seed(33)
        checkpoint = self.save()
        expected_random = random.random()
        expected_np = np.random.random()
        data = self.load()
        restored = self.Bundle(self.args, Entry)
        self.ck.restore_bundle(restored, data)
        self.assertEqual(data["stream_state"], self.state)
        entry = restored.private_memories[0][0]
        self.assertIs(entry, restored.private_backends[0].entries["m1"])
        self.assertEqual(vars(entry), vars(self.bundle.private_memories[0][0]))
        self.assertEqual(restored.private_backends[0].backend_ids, {"m1": "qdrant-id"})
        self.assertIsNone(restored.private_backends[0]._service)
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(np.random.random(), expected_np)
        # Exercise the production lazy backend reconstruction with a local fake service.
        with patch("maple_guard.memory_backend.MemoryService", side_effect=lambda **kw: SimpleNamespace(
            **kw, _cube_timestamp=kw["resume_cube_timestamp"],
            default_cube_id=f'cube_{kw["user_id"]}_{kw["resume_cube_timestamp"]}',
        )) as factory:
            service = restored.private_backends[0]._ensure_service()
        self.assertEqual(factory.call_args.kwargs["embedding_dim"], 2)
        self.assertEqual(service._cube_timestamp, "20260929_080001")
        self.assertEqual(service.dict_memory, self.bundle.private_backends[0]._service.dict_memory)
        np.testing.assert_array_equal(service.query_embeddings["original query"], np.array([0.125, 0.25], dtype="float32"))
        self.assertEqual(service.query_embeddings["original query"].dtype, np.dtype("float32"))
        self.assertEqual(service._mem_cache, self.bundle.private_backends[0]._service._mem_cache)
        self.assertEqual(service._q_cache, {"qdrant-id": 0.25})
        self.assertTrue(Path(checkpoint).is_dir())

    def test_partial_task_disk_and_trace_are_rolled_back_and_failed_attempt_is_preserved(self):
        checkpoint = self.save()
        frozen = {str(p.relative_to(checkpoint)): p.read_bytes() for p in Path(checkpoint).rglob("*") if p.is_file()}
        (self.store / "config.json").write_bytes(b"partial task update")
        (self.store / "partial-new-file").write_bytes(b"partial")
        Path(self.args.out).write_bytes(b'{"task_id":"task-0","task_index":0}\n{"unfinished":')
        Path(self.args.baseline_state_path).write_bytes(b'{"baseline":2}')
        data = self.load()
        self.assertEqual((self.store / "config.json").read_bytes(), b'{"test":"secret"}\n')
        self.assertFalse((self.store / "partial-new-file").exists())
        self.assertEqual(Path(self.args.out).read_bytes(), b'{"task_id":"task-0","task_index":0}\n')
        self.assertEqual(Path(self.args.baseline_state_path).read_bytes(), b'{"baseline":1}\n')
        audit = Path(data["failed_attempt_path"])
        self.assertTrue(any(p.read_bytes() == b"partial task update" for p in audit.rglob("*") if p.is_file()))
        self.assertTrue(any(p.read_bytes().endswith(b'{"unfinished":') for p in audit.rglob("*") if p.is_file()))
        self.assertEqual(frozen, {str(p.relative_to(checkpoint)): p.read_bytes() for p in Path(checkpoint).rglob("*") if p.is_file()})

    def test_corruption_is_rejected_before_any_working_state_is_changed(self):
        checkpoint = self.save()
        payload = Path(checkpoint) / "runtime.json"
        payload.write_bytes(payload.read_bytes() + b" ")
        (self.store / "config.json").write_bytes(b"must survive")
        before = Path(self.args.out).read_bytes()
        with self.assertRaisesRegex(self.ck.CheckpointError, "hash|integrity"):
            self.load()
        self.assertEqual((self.store / "config.json").read_bytes(), b"must survive")
        self.assertEqual(Path(self.args.out).read_bytes(), before)

    def test_config_mismatch_is_rejected_before_disk_mutation(self):
        self.save()
        self.args.seed = 999
        (self.store / "config.json").write_bytes(b"must survive")
        with self.assertRaisesRegex(self.ck.CheckpointError, "identity|config"):
            self.load()
        self.assertEqual((self.store / "config.json").read_bytes(), b"must survive")

    def test_failed_save_keeps_previous_pointer_and_checkpoint(self):
        previous = self.save()
        pointer = Path(self.args.task_checkpoint_dir) / "latest.json"
        pointer_bytes = pointer.read_bytes()
        self.bundle.private_memories[0][0].not_serializable = object()
        with self.assertRaises(self.ck.CheckpointError):
            self.save()
        self.assertEqual(pointer.read_bytes(), pointer_bytes)
        self.assertTrue(Path(previous).exists())

    def test_sqlite_wal_is_backed_up_consistently(self):
        conn = sqlite3.connect(self.store / "state.sqlite")
        self.addCleanup(conn.close)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE values_table (v TEXT)")
        conn.execute("INSERT INTO values_table VALUES ('boundary')")
        conn.commit()
        self.save()
        conn.close()
        data = self.load()
        with sqlite3.connect(self.store / "state.sqlite") as restored:
            self.assertEqual(restored.execute("SELECT v FROM values_table").fetchall(), [("boundary",)])
        self.assertTrue(data["checkpoint_path"])

    def test_other_methods_fail_closed(self):
        self.args.method = "no_defense_memrl"
        with self.assertRaisesRegex(self.ck.CheckpointError, "provenance_acl"):
            self.save()

    def test_symlink_in_store_is_rejected_without_replacing_last_checkpoint(self):
        self.save()
        pointer = (Path(self.args.task_checkpoint_dir) / "latest.json").read_bytes()
        (self.store / "escape").symlink_to(self.root / "trace.jsonl")
        with self.assertRaisesRegex(self.ck.CheckpointError, "symlink"):
            self.save()
        self.assertEqual((Path(self.args.task_checkpoint_dir) / "latest.json").read_bytes(), pointer)

    def test_secret_artifacts_are_private(self):
        checkpoint = self.save()
        for path in [Path(self.args.task_checkpoint_dir), *Path(checkpoint).rglob("*")]:
            self.assertEqual(path.stat().st_mode & 0o077, 0, str(path))

    def test_missing_checkpoint_does_not_claim_legacy_resume(self):
        with self.assertRaisesRegex(self.ck.CheckpointError, "checkpoint|latest"):
            self.load()

    def test_memory_service_rejects_invalid_resume_timestamp_before_any_io(self):
        from maple_guard.memrl_service.memory_service import MemoryService
        with patch("maple_guard.memrl_service.memory_service.MOSConfig.from_json_file") as read_config:
            with self.assertRaisesRegex(ValueError, "resume_cube_timestamp"):
                MemoryService("unused", SimpleNamespace(), SimpleNamespace(), user_id="tester",
                              embedding_dim=2, resume_cube_timestamp="../escape")
            read_config.assert_not_called()

    def test_memory_service_resume_opens_existing_cube_without_reload_or_api_probe(self):
        from maple_guard.memrl_service.memory_service import MemoryService
        from unittest.mock import Mock
        timestamp = "20260929_090001"
        base = self.root / "direct" / "mem_cubes"
        cube = base / "tester" / timestamp
        qdrant = base.parent / "qdrant" / "tester" / timestamp
        cube.mkdir(parents=True)
        qdrant.mkdir(parents=True)
        (cube / "config.json").write_text(json.dumps({"text_mem": {"config": {"vector_db": {
            "config": {"path": str(qdrant), "collection_name": f"memp_tester_{timestamp}",
                       "vector_dimension": 2}}}}}))
        cfg = SimpleNamespace(
            chat_model=SimpleNamespace(backend="openai", config=SimpleNamespace(
                model_dump=lambda: {}, api_key="EMPTY", api_base="unused")),
            mem_reader=SimpleNamespace(config=SimpleNamespace(embedder=SimpleNamespace(config=SimpleNamespace(
                model_name_or_path="offline", provider="openai", base_url="unused", api_key="EMPTY")))),
        )
        embedder = Mock()
        with patch("maple_guard.memrl_service.memory_service.MOSConfig.from_json_file", return_value=cfg), \
             patch("maple_guard.memrl_service.memory_service.MOS"), \
             patch("maple_guard.memrl_service.memory_service.GeneralMemCubeConfig"), \
             patch("maple_guard.memrl_service.memory_service.GeneralMemCube") as cube_cls:
            service = MemoryService("unused", SimpleNamespace(), embedder, user_id="tester",
                                    embedding_dim=2, resume_cube_timestamp=timestamp,
                                    base_root=str(base), enable_value_driven=False)
            cube_cls.init_from_dir.assert_called_once_with(str(cube), memory_types=[])
            cube_cls.assert_not_called()
            embedder.embed.assert_not_called()
            self.assertEqual(service._cube_timestamp, timestamp)

    def test_run_lock_rejects_second_owner(self):
        with self.ck.run_lock(self.args):
            with self.assertRaisesRegex(self.ck.CheckpointError, "active"):
                with self.ck.run_lock(self.args):
                    self.fail("A second run acquired the same checkpoint lock")

    def test_record_prefix_mismatch_is_rejected_before_publication(self):
        self.state["records"][0]["task_id"] = "wrong-task"
        with self.assertRaisesRegex(self.ck.CheckpointError, "prefix|record"):
            self.save()

    def test_api_journal_is_rolled_back_and_partial_attempt_is_preserved(self):
        journal = self.root / "api.jsonl"
        with patch.dict(os.environ, {"MAPLE_CALL_LOG": str(journal)}):
            journal.write_bytes(b'{"ok":true}\n')
            self.save()
            journal.write_bytes(b'{"ok":true}\n{"failed":true}\n')
            data = self.load()
            self.assertEqual(journal.read_bytes(), b'{"ok":true}\n')
            self.assertEqual((Path(data["failed_attempt_path"]) / "api_journal").read_bytes(),
                             b'{"ok":true}\n{"failed":true}\n')

    def test_validation_environment_change_rejects_resume(self):
        with patch.dict(os.environ, {"MAPLE_FAIL_ON_TRUNCATION": "1"}):
            self.save()
        with patch.dict(os.environ, {"MAPLE_FAIL_ON_TRUNCATION": "0"}):
            with self.assertRaisesRegex(self.ck.CheckpointError, "identity"):
                self.load()

    def test_comparison_runtime_keeps_embedding_cache_and_entry_references(self):
        from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
        Path(self.args.baseline_state_path).unlink()
        entry = self.bundle.private_memories[0][0]
        for key, value in {"experience": "test trajectory", "origin_agent": 0, "origin_task": "task-0",
                           "origin_round": 0, "memory_scope": "agent_private", "status": "active",
                           "retrieval_key": "", "parents": []}.items():
            setattr(entry, key, value)
        runtime = ComparisonRuntime(self.args)
        self.args._full_baseline_runtime = runtime
        runtime.observe_ingress(entry, "agent_output", "agent_private", 0)
        entry.embedding = [0.125, 0.875]
        runtime.record_embedding(entry)
        runtime.select_memories({0: [entry]})
        self.save()
        data = self.load()
        del self.args._full_baseline_runtime
        restored = self.Bundle(self.args, Entry)
        self.ck.restore_bundle(restored, data)
        with patch("maple_guard.memory_backend.OpenAICompatibleEmbedder.embed", side_effect=AssertionError("API call")):
            resumed = self.args._full_baseline_runtime
            resumed.prepare_entries(restored.private_memories[0])
        self.assertEqual(restored.private_memories[0][0].embedding, [0.125, 0.875])
        self.assertIs(resumed.selected[0][0], restored.private_memories[0][0])
        self.assertIs(resumed.consumed[0]["m1"], restored.private_memories[0][0])

    def test_qdrant_dense_runtime_preserves_normalized_vectors_and_dtype(self):
        from qdrant_client import QdrantClient, models
        path = self.store / "local-qdrant"
        client = QdrantClient(path=str(path))
        self.addCleanup(client.close)
        client.create_collection("fixture", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        client.upsert("fixture", points=[models.PointStruct(id=1, vector=[3.0, 4.0], payload={"q": 0.25})])
        def attach(service, active):
            service.mos = SimpleNamespace(mem_cubes={service.default_cube_id: SimpleNamespace(
                text_mem=SimpleNamespace(vector_db=SimpleNamespace(client=active,
                    config=SimpleNamespace(collection_name="fixture"))))})
            return service
        original = attach(self.bundle.private_backends[0]._service, client)
        original_vector = client.retrieve("fixture", [1], with_vectors=True)[0].vector
        original_dtype = client._client.collections["fixture"].vectors[""].dtype
        self.save()
        client.close()
        data = self.load()
        restored = self.Bundle(self.args, Entry)
        self.ck.restore_bundle(restored, data)
        reopened = QdrantClient(path=str(path))
        self.addCleanup(reopened.close)
        # Installed Qdrant reloads raw persisted vectors without cosine normalization.
        self.assertNotEqual(reopened.retrieve("fixture", [1], with_vectors=True)[0].vector, original_vector)
        def factory(**kw):
            return attach(SimpleNamespace(**kw, _cube_timestamp=kw["resume_cube_timestamp"],
                default_cube_id=f'cube_{kw["user_id"]}_{kw["resume_cube_timestamp"]}'), reopened)
        with patch("maple_guard.memory_backend.MemoryService", side_effect=factory):
            restored.private_backends[0]._ensure_service()
        self.assertEqual(reopened.retrieve("fixture", [1], with_vectors=True)[0].vector, original_vector)
        self.assertEqual(reopened._client.collections["fixture"].vectors[""].dtype, original_dtype)

    def test_prompt_contents_and_missing_fallback_presence_are_frozen(self):
        prompts = self.root / "prompts"
        prompts.mkdir()
        self.args.prompt_dir = str(prompts)
        self.args.prompt_file = str(prompts / "prompts.yaml")
        (prompts / "prompts.yaml").write_text("system: {base: original}")
        self.save()
        (prompts / "prompts.yaml").write_text("system: {base: changed}")
        with self.assertRaisesRegex(self.ck.CheckpointError, "identity"):
            self.load()
        (prompts / "prompts.yaml").write_text("system: {base: original}")
        (prompts / "task_user.txt").write_text("new fallback")
        with self.assertRaisesRegex(self.ck.CheckpointError, "identity"):
            self.load()

    def test_save_rejects_config_drift_without_advancing_pointer(self):
        self.save()
        pointer = Path(self.args.task_checkpoint_dir) / "latest.json"
        before = pointer.read_bytes()
        self.args.seed = 99
        with self.assertRaisesRegex(self.ck.CheckpointError, "identity"):
            self.save()
        self.assertEqual(pointer.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
