"""Task checkpoints for the persistent comparison suite."""
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_task_checkpoint as fixtures


class SuiteCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TaskCheckpointTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.ck, self.args, self.bundle = self.fixture.ck, self.fixture.args, self.fixture.bundle
        self.root, self.state = self.fixture.root, self.fixture.state

    def test_maple_mmlu_complete_boundary_survives_failed_task(self):
        self.args.method, self.args.task_mode = "maple_guard", "qa"
        self.fixture.save()
        Path(self.args.out).write_text('{"partial":true}\n')
        self.bundle.private_memories[0][0].utility_q = 999
        restored = self.fixture.load()
        fresh = self.fixture.Bundle(self.args, self.fixture.bundle.entry_cls)
        self.ck.restore_bundle(fresh, restored)
        self.assertEqual(fresh.private_memories[0][0].utility_q, .25)
        self.assertEqual(restored["stream_state"], self.state)

    def test_no_defense_and_communication_methods_support_memory_boundaries(self):
        for method in ("no_defense_memrl", "gsafeguard", "guardian", "inspector",
                       "maple_guard_retrieval_only", "maple_guard_no_write"):
            with self.subTest(method=method):
                self.args.method = method
                self.args.task_checkpoint_dir = str(self.root / method)
                self.args.task_mode = "longmemeval"
                self.fixture.save()
                self.assertTrue(self.fixture.load()["checkpoint_path"])

    def test_torch_rng_is_restored_for_stochastic_native_detectors(self):
        import torch
        torch.manual_seed(27)
        self.fixture.save()
        expected = torch.rand(6)
        torch.manual_seed(999)
        data = self.fixture.load()
        fresh = self.fixture.Bundle(self.args, self.fixture.bundle.entry_cls)
        self.ck.restore_bundle(fresh, data)
        self.assertTrue(torch.equal(torch.rand(6), expected))

    def test_budget_change_requires_explicit_opt_in_and_is_recorded(self):
        self.args.chat_max_tokens = 1024
        self.fixture.save()
        self.args.chat_max_tokens = 2048
        with self.assertRaisesRegex(self.ck.CheckpointError, "identity"):
            self.fixture.load()
        self.args.checkpoint_allow_budget_change = True
        data = self.fixture.load()
        changes = data["budget_transition"]["changes"]
        self.assertEqual(changes["chat_max_tokens"], {"before":1024, "after":2048})
        self.assertEqual(data["budget_transition"]["next_task_index"], 1)
        self.assertTrue(Path(data["failed_attempt_path"], "recovery.json").exists())
        self.fixture.save()
        self.assertIsNone(self.fixture.load()["budget_transition"])
        self.assertEqual(len(self.args._checkpoint_budget_history), 1)
        self.args.seed = 77
        with self.assertRaisesRegex(self.ck.CheckpointError, "identity"):
            self.fixture.load()

    def test_amemguard_lessons_pending_prompts_counters_and_aliases_resume(self):
        from evaluate.defense_methods.comparison_runtime import ComparisonRuntime
        from evaluate.defense_methods.amemguard_full import AMemGuardFull
        from evaluate.defense_methods.full_runtime import tracked_client
        Path(self.args.baseline_state_path).unlink()
        self.args.method = "amemguard_full"
        self.args.amemguard_experiment_id = "checkpoint-test"
        guard = AMemGuardFull(self.args, tracked_client(lambda *_:"unused"),
                             tracked_client(lambda *_:[1., 0.]))
        guard._model_calls = 7
        guard._embedding_calls = 4
        guard._pending_prompts = {0:{"query":"remember", "prompt":"lesson"}}
        guard.judge.counters.update(calls=7, errors=0, wall_seconds=3.)
        runtime = ComparisonRuntime(self.args, guard=guard)
        runtime.selected = {0:[self.bundle.private_memories[0][0]]}
        runtime._embedding_cache = {"stable":[.1,.2]}
        self.args._full_baseline_runtime = runtime
        self.fixture.save()
        data = self.fixture.load()
        del self.args._full_baseline_runtime
        fresh = self.fixture.Bundle(self.args, self.fixture.bundle.entry_cls)
        with patch("requests.post", side_effect=AssertionError("resume issued API request")):
            self.ck.restore_bundle(fresh, data)
        restored = self.args._full_baseline_runtime
        self.assertEqual(restored.guard._model_calls, 7)
        self.assertEqual(restored.guard._pending_prompts, guard._pending_prompts)
        self.assertEqual(restored.guard.judge.counters, guard.judge.counters)
        self.assertIs(restored.selected[0][0], fresh.private_memories[0][0])

    def test_official_communication_history_roundtrip(self):
        from evaluate.defense_methods.base import OfficialDefenseState
        official = OfficialDefenseState()
        official.guardian_history = [{0:"first",1:"second"}]
        official.guardian_inactive_agents = {1}
        official.inspector_memory = [{"role":"user","content":"remember"}]
        official.gnn_state = {"round_history":{"task-0":[["a","b"]]}}
        self.state["official_defense_state"] = official
        self.fixture.save()
        restored = self.fixture.load()["stream_state"]["official_defense_state"]
        self.assertIsInstance(restored, OfficialDefenseState)
        self.assertEqual(restored, official)

    def test_only_two_committed_generations_are_retained(self):
        for i in range(4):
            self.fixture.save()
        generations = list(Path(self.args.task_checkpoint_dir).glob("task-*"))
        self.assertEqual(len(generations), 2)

    def test_extra_output_journals_rollback_together(self):
        extra = self.root / "metrics.jsonl"
        extra.write_text('{"completed":1}\n')
        self.args._checkpoint_extra_paths = {"metrics":str(extra)}
        self.fixture.save()
        extra.write_text('{"completed":1}\n{"partial":2}\n')
        data = self.fixture.load()
        self.assertEqual(extra.read_text(), '{"completed":1}\n')
        self.assertIn("partial", Path(data["failed_attempt_path"],"metrics").read_text())

    def test_agentsafe_restores_hierarcache_without_initialization_api_calls(self):
        from evaluate.defense_methods.agentsafe_full import AgentSafeFull
        from evaluate.defense_methods.full_runtime import FullRuntime, tracked_client
        policy=self.root/"policy.json";criteria=self.root/"criteria.json"
        policy.write_text(json.dumps({"identities":{"0":"Agent 0","1":"Agent 1"},"relations":{}}))
        criteria.write_text('["relevant"]')
        self.args.method="agentsafe_full"
        self.args.agentsafe_policy_file=str(policy);self.args.agentsafe_criteria_file=str(criteria)
        self.args.agentsafe_threshold=.5
        Path(self.args.baseline_state_path).unlink()
        def judge(messages):
            return '{"level":2}' if "security classification" in messages[0]["content"] else '{"valid":true}'
        guard=AgentSafeFull(self.args,tracked_client(judge),tracked_client(lambda t:[1.,0.]))
        self.assertTrue(guard.admit("m1","useful",0,None,{})[0])
        runtime=FullRuntime(self.args,guard=guard)
        runtime.entries={"m1":self.bundle.private_memories[0][0]}
        runtime.review_clock=7
        self.args._full_baseline_runtime=runtime
        self.fixture.save()
        data=self.fixture.load()
        fresh=self.fixture.Bundle(self.args,self.bundle.entry_cls)
        with patch("requests.post",side_effect=AssertionError("restore made an API request")):
            self.ck.restore_bundle(fresh,data)
        actual=self.args._full_baseline_runtime
        self.assertEqual(actual.guard.state_dict(),guard.state_dict())
        self.assertEqual(actual.guard.embed.counters,guard.embed.counters)
        self.assertEqual(actual.review_clock,7)
        self.assertIs(actual.entries["m1"],fresh.private_memories[0][0])

    def test_native_infa_mutable_batchnorm_buffers_and_mode_roundtrip(self):
        import torch
        from evaluate.defense_methods.infa_full import InfaGuardFull,NativeInfaDetector
        from maple_guard.baseline_checkpoint import capture_guard,install_guard
        detector=NativeInfaDetector.__new__(NativeInfaDetector)
        detector.model=torch.nn.BatchNorm1d(2)
        detector.provenance={"native":True}
        args=SimpleNamespace(agents=2,infa_protocol="released")
        guard=InfaGuardFull(args,lambda m:"unused",detector=detector)
        detector.model(torch.tensor([[2.,4.],[4.,8.]]))
        guard.p_inf_ema=[.3,.7]
        saved=self.ck._decode(self.ck._encode(capture_guard(guard)))
        mean=detector.model.running_mean.clone()
        detector.model.running_mean.zero_();detector.model.eval()
        guard.p_inf_ema=[0.,0.]
        install_guard(guard,saved)
        self.assertTrue(detector.model.training)
        self.assertTrue(torch.equal(detector.model.running_mean,mean))
        self.assertEqual(guard.p_inf_ema,[.3,.7])

    def test_agentxposed_released_engine_and_piguard_counters_roundtrip(self):
        from maple_guard.baseline_checkpoint import capture_guard,install_guard
        from evaluate.defense_methods.agentxposed_full import AgentXposedFull
        from evaluate.defense_methods.piguard import PIGuardDetector
        for method in ("agentxposed_full_guide","agentxposed_full_kick"):
            guard=AgentXposedFull(SimpleNamespace(method=method,agents=2,agentxposed_protocol="reconstruction"),lambda m:"unused")
            guard._released_engine=SimpleNamespace(calls=9,logs=[{"stage":"assessment"}],namespace={"unserializable":lambda:None})
            guard.inactive={1};guard.history={0:["history"]}
            saved=self.ck._decode(self.ck._encode(capture_guard(guard)))
            guard.inactive=set();guard._released_engine.calls=0
            install_guard(guard,saved)
            self.assertEqual(guard.inactive,{1})
            self.assertEqual(guard._released_engine.calls,9)
        guard=PIGuardDetector.__new__(PIGuardDetector)
        guard.model=object();guard.tokenizer=object();guard._counters={"calls":7}
        model=guard.model
        saved=self.ck._decode(self.ck._encode(capture_guard(guard)))
        guard._counters["calls"]=0
        install_guard(guard,saved)
        self.assertEqual(guard._counters["calls"],7)
        self.assertIs(guard.model,model)

    def test_policy_and_model_assets_cannot_change_between_checkpoints(self):
        model=self.root/"model";model.mkdir()
        (model/"weights.bin").write_bytes(b"pinned")
        policy=self.root/"policy.json";policy.write_text('{"clearance":1}')
        self.args.piguard_model=str(model)
        self.args.agentsafe_policy_file=str(policy)
        self.fixture.save()
        policy.write_text('{"clearance":2}')
        with self.assertRaisesRegex(self.ck.CheckpointError,"identity"):self.fixture.load()
        policy.write_text('{"clearance":1}')
        (model/"weights.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(self.ck.CheckpointError,"identity"):self.fixture.load()

    def test_official_static_caches_are_rehydrated_before_rng_restore(self):
        from evaluate.defense_methods.base import OfficialDefenseState
        from maple_guard.baseline_checkpoint import capture_official,decode_official,hydrate_official
        state=OfficialDefenseState()
        state.gnn_state={"guardian_runtime_cache":{"/pinned/guardian":{"code_dir":"/pinned/guardian","module":lambda:None}},
            "runtime_cache":{"gsafeguard|model|embed|cpu":{"checkpoint":"model","embedding_model":"embed","device":"cpu","model":lambda:None}}}
        saved=self.ck._decode(self.ck._encode(capture_official(state)))
        restored=decode_official(saved)
        with patch("evaluate.defense_methods.guardian_defense._load_official_runtime",return_value=({},None)) as guardian, \
             patch("evaluate.defense_methods.gnn_defense._load_runtime",return_value=({},None)) as gnn:
            hydrate_official(restored,self.args)
        guardian.assert_called_once_with(restored,"/pinned/guardian")
        self.assertEqual(gnn.call_count,1)
        self.assertEqual(gnn.call_args.args[2],"gsafeguard")

    def test_rng_capture_happens_after_lazy_torch_import(self):
        import builtins, torch
        original=builtins.__import__
        def importing(name,*args,**kwargs):
            if name=="torch":
                random.seed(55)
                return torch
            return original(name,*args,**kwargs)
        random.seed(11)
        with patch("builtins.__import__",side_effect=importing):
            state=self.ck.capture_rng_state()
        self.assertEqual(state["random"],random.Random(55).getstate())


if __name__ == "__main__":
    unittest.main()
