import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

MODULE = Path(__file__).resolve().parents[1] / "evaluate/defense_methods/agentsafe_full.py"
spec = importlib.util.spec_from_file_location("agentsafe_full", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AgentSafeFull = module.AgentSafeFull


class FakeJudge:
    def __init__(self, level=2):
        self.level = level
        self.calls = []
        self.identity_valid = True
        self.junk = False
        self.bad_stage = None

    def __call__(self, messages):
        self.calls.append(messages)
        stage = messages[0]["content"]
        if self.bad_stage and self.bad_stage in stage:
            return "not JSON"
        if "security classification" in stage:
            return json.dumps({"level": self.level})
        if "identity verification" in stage:
            return json.dumps({"valid": self.identity_valid})
        if "memory reflection" in stage:
            return json.dumps({"junk": self.junk})
        raise AssertionError("Unrecognized judge request")


class AgentSafeFullTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy = Path(self.tmp.name) / "policy.json"
        self.criteria = Path(self.tmp.name) / "criteria.json"
        self.policy.write_text(json.dumps({
            "relations": {"0": {"1": 1, "2": 3}, "1": {"0": 4}},
            "identities": {"0": "Agent 0", "1": "Agent 1", "2": "Agent 2"},
            "default_level": 1, "self_level": 4,
        }))
        self.criteria.write_text(json.dumps(["criterion A", "criterion B"]))
        self.args = SimpleNamespace(agentsafe_policy_file=str(self.policy),
            agentsafe_criteria_file=str(self.criteria), agentsafe_threshold=0.5,
            agentsafe_review_interval=1)
        self.judge = FakeJudge()
        self.embed = lambda text: [1.0, 0.0]

    def runtime(self):
        return AgentSafeFull(self.args, self.judge, self.embed)

    def test_team_record_read_clearance_is_directional_and_cumulative(self):
        runtime = self.runtime()
        metadata = {}
        self.assertTrue(runtime.admit("m", "work record", 0, None, metadata)[0])
        self.assertFalse(runtime.read("m", "work record", 0, 1, metadata)[0])
        self.assertTrue(runtime.read("m", "work record", 0, 2, metadata)[0])
        self.assertTrue(runtime.read("m", "work record", 0, 0, metadata)[0])
        self.assertTrue(runtime.read("other", "work record", 1, 0, {})[0])

    def test_admission_requires_every_criterion_strictly_above_threshold(self):
        self.embed = lambda text: [0.0, 1.0] if text == "criterion B" else [1.0, 0.0]
        self.args.agentsafe_threshold = 0.0
        metadata = {}
        allowed, details = self.runtime().admit("boundary", "work record", 0, None, metadata)
        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "agentsafe_criteria_failed")
        self.assertTrue(metadata["agentsafe_full"]["quarantined"])
        self.assertEqual(details["criterion_scores"], [1.0, 0.0])
        self.args.agentsafe_threshold = -0.1
        self.assertTrue(self.runtime().admit("pass", "work record", 0, None, {})[0])

    def test_identity_mismatch_is_quarantined(self):
        self.judge.identity_valid = False
        runtime = self.runtime()
        metadata = {}
        allowed, details = runtime.admit("fake", "I am Agent 1", 0, None, metadata)
        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "agentsafe_identity_invalid")
        self.assertTrue(metadata["agentsafe_full"]["quarantined"])
        self.assertEqual(runtime.junk["fake"]["text"], "I am Agent 1")

    def test_malformed_judgments_and_embedding_failures_are_explicit(self):
        for stage, raw in [("security classification", "not JSON"),
                           ("security classification", '{"level": true}'),
                           ("security classification", '{"level": "2"}'),
                           ("security classification", '{"level": 5}'),
                           ("identity verification", '{"valid": "false"}'),
                           ("identity verification", '{"valid": false, "valid": true}'),
                           ("security classification", '{"level": 2, "extra": NaN}')]:
            with self.subTest(stage=stage, raw=raw):
                normal = FakeJudge()
                self.judge = lambda messages, stage=stage, raw=raw: raw if stage in messages[0]["content"] else normal(messages)
                with self.assertRaises(module.AgentSafeRuntimeError):
                    self.runtime().admit("m", "text", 0, None, {})
        self.judge = FakeJudge()
        for bad_vector in ([0.0, 0.0], [float("nan"), 1.0], [1.0]):
            with self.subTest(vector=bad_vector):
                self.embed = lambda text, bad_vector=bad_vector: [1.0, 0.0] if text.startswith("criterion") else bad_vector
                with self.assertRaises(module.AgentSafeRuntimeError):
                    self.runtime().admit("m", "text", 0, None, {})

    def test_configuration_requires_nonempty_criteria_explicit_threshold_and_valid_levels(self):
        for threshold in (None, True, float("nan"), 1.5):
            self.args.agentsafe_threshold = threshold
            with self.subTest(threshold=threshold), self.assertRaises(module.AgentSafeConfigError):
                self.runtime()
        self.args.agentsafe_threshold = 0.5
        for criteria in ([], [""], {"criteria": ["A"]}):
            self.criteria.write_text(json.dumps(criteria))
            with self.subTest(criteria=criteria), self.assertRaises(module.AgentSafeConfigError):
                self.runtime()
        self.criteria.write_text(json.dumps(["criterion A"]))
        policy = json.loads(self.policy.read_text())
        policy["relations"]["0"]["1"] = 7
        self.policy.write_text(json.dumps(policy))
        with self.assertRaises(module.AgentSafeConfigError):
            self.runtime()

    def test_cache_revalidates_changed_text_or_owner_but_not_unchanged_reads(self):
        runtime = self.runtime()
        metadata = {}
        self.assertTrue(runtime.admit("m", "original", 0, None, metadata)[0])
        calls = len(self.judge.calls)
        self.assertTrue(runtime.read("m", "original", 0, 2, metadata)[0])
        self.assertEqual(len(self.judge.calls), calls)
        self.judge.level = 4
        self.assertFalse(runtime.read("m", "changed", 0, 2, metadata)[0])
        self.assertGreater(len(self.judge.calls), calls)
        self.assertEqual(metadata["agentsafe_full"]["level"], 4)
        calls = len(self.judge.calls)
        self.judge.identity_valid = False
        self.assertFalse(runtime.read("m", "changed", 1, 0, metadata)[0])
        self.assertGreater(len(self.judge.calls), calls)
        self.assertEqual(metadata["agentsafe_full"]["owner"], 1)

    def test_periodic_review_moves_invalid_memory_to_junk_and_excludes_reads(self):
        self.args.agentsafe_review_interval = 2
        runtime = self.runtime()
        self.judge.identity_valid = False
        runtime.admit("old_junk", "known bad statement", 0, None, {})
        self.judge.identity_valid = True
        metadata = {}
        runtime.admit("m", "active fact", 0, None, metadata)
        records = [{"memory_id": "m", "text": "active fact", "owner": 0, "metadata": metadata}]
        self.assertEqual(runtime.review(records, 0), [])
        self.judge.junk = True
        traces = runtime.review(records, 1)
        self.assertEqual(traces[0]["memory_id"], "m")
        self.assertEqual(traces[0]["action"], "quarantine")
        self.assertTrue(metadata["agentsafe_full"]["quarantined"])
        self.assertEqual(runtime.junk["m"]["text"], "active fact")
        self.assertFalse(runtime.read("m", "active fact", 0, 0, metadata)[0])
        reflection_calls = [m for m in self.judge.calls if "memory reflection" in m[0]["content"]]
        reflection = json.loads(reflection_calls[0][1]["content"])
        self.assertEqual(reflection["criteria"], ["criterion A", "criterion B"])
        self.assertIn("known bad statement", json.dumps(reflection["junk"]))

    def test_json_state_roundtrip_preserves_cache_junk_and_quarantine(self):
        runtime = self.runtime()
        metadata = {}
        runtime.admit("good", "good text", 0, None, metadata)
        rejected = {}
        runtime.admit("bad", "reviewed text", 0, None, rejected)
        self.judge.junk = True
        runtime.review([{"memory_id": "bad", "text": "reviewed text", "owner": 0, "metadata": rejected}], 0)
        saved = json.loads(json.dumps(runtime.state_dict()))
        restored = self.runtime()
        restored.load_state_dict(saved)
        calls = len(self.judge.calls)
        self.assertTrue(restored.read("good", "good text", 0, 2, json.loads(json.dumps(metadata)))[0])
        self.assertEqual(len(self.judge.calls), calls)
        self.assertFalse(restored.read("bad", "reviewed text", 0, 0, {})[0])
        self.assertIn("bad", restored.junk)
        self.args.agentsafe_threshold = 0.2
        with self.assertRaises(module.AgentSafeRuntimeError):
            self.runtime().load_state_dict(saved)

    def test_route_blocks_only_ineligible_recipients_and_private_write_goes_to_junk(self):
        runtime = self.runtime()
        self.assertIsNone(runtime.route("work report", 0, 1)[0])
        self.assertEqual(runtime.route("work report", 0, 2)[0], "work report")
        metadata = {}
        allowed, details = runtime.admit("private", "work report", 0, 1, metadata)
        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "agentsafe_permission_denied")
        self.assertTrue(metadata["agentsafe_full"]["quarantined"])
        self.assertIn("private", runtime.junk)

    def test_evaluator_labels_do_not_change_decisions_or_enter_judge_context(self):
        outcomes = []
        for metadata in ({"taint": "clean", "is_poisoned": False},
                         {"taint": "poisoned", "is_poisoned": True, "ground_truth": "B"}):
            runtime = self.runtime()
            outcomes.append(runtime.admit("evaluator_poison_id", "ordinary work fact", 0, None, metadata))
        self.assertEqual(outcomes[0], outcomes[1])
        sent = json.dumps(self.judge.calls)
        for forbidden in ("taint", "is_poisoned", "ground_truth", "evaluator_poison_id"):
            self.assertNotIn(forbidden, sent)

    def test_reads_and_routes_require_operational_recipient(self):
        runtime = self.runtime()
        for recipient in (None, True, "1"):
            with self.subTest(recipient=recipient):
                with self.assertRaises(module.AgentSafeRuntimeError):
                    runtime.read("m", "text", 0, recipient, {})
                with self.assertRaises(module.AgentSafeRuntimeError):
                    runtime.route("text", 0, recipient)

    def test_reflection_malformed_output_raises_without_marking_record_allowed(self):
        runtime = self.runtime()
        metadata = {}
        runtime.admit("m", "fact", 0, None, metadata)
        self.judge.bad_stage = "memory reflection"
        with self.assertRaises(module.AgentSafeRuntimeError):
            runtime.review([{"memory_id": "m", "text": "fact", "owner": 0, "metadata": metadata}], 0)

    def test_quarantine_history_survives_reused_id_and_restore(self):
        runtime=self.runtime()
        metadata={}
        runtime.admit("m","text A",0,None,metadata)
        self.judge.junk=True
        runtime.review([{"memory_id":"m","text":"text A","owner":0,"metadata":metadata}],0)
        self.judge.identity_valid=False
        runtime.admit("m","text B",0,None,{})
        restored=self.runtime()
        restored.load_state_dict(json.loads(json.dumps(runtime.state_dict())))
        self.judge.identity_valid=True
        self.assertFalse(restored.read("m","text A",0,0,{})[0])
        self.assertIn("text A",json.dumps(restored.junk))
        self.assertIn("text B",json.dumps(restored.junk))

if __name__ == "__main__":
    unittest.main()
