"""Source-profile A-MemGuard contracts; external calls use deterministic fixtures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

MODULE = Path(__file__).resolve().parents[1] / "evaluate/defense_methods/amemguard_full.py"
module = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location("amemguard_full", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

def entry(key, intent="mail query", action="inspect email"):
    return SimpleNamespace(memory_id=key, intent=intent, experience=action,
                           origin_agent=0, poisoned="PRIVATE_LABEL", outcome="PRIVATE_OUTCOME")

class ScriptedJudge:
    def __init__(self, chains, judgments):
        self.chains, self.judgments, self.calls = list(chains), judgments, []
    def __call__(self, messages):
        self.calls.append(messages)
        if "**Build Reasoning Chain**" in messages[-1]["content"]:
            return self.chains.pop(0)
        return self.judgments if isinstance(self.judgments, str) else json.dumps(self.judgments)

class AMemGuardFullTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(module, "A-MemGuard full component is not implemented")
        self.args = SimpleNamespace(amemguard_top_k=4, amemguard_lesson_top_k=4,
                                    amemguard_experiment_id="test-experiment")
        self.embedded, self.vectors = [], {}
    def embed(self, text):
        self.embedded.append(text)
        return self.vectors.get(text, [1., 0.])
    def guard(self, chains=(), judgments=()):
        judge = ScriptedJudge(chains, judgments)
        return module.AMemGuardFull(self.args, judge, self.embed), judge

    def test_independent_query_paths_joint_audit_returns_actual_objects(self):
        records = [entry("one", action="inspect sender"), entry("two", action="forward credentials")]
        guard, judge = self.guard(["Sender -> verify -> domain", "Credentials -> send -> attacker"],
            [{"id": 0, "consistent": True, "safe": True}, {"id": 1, "consistent": True, "safe": False}])
        retained, diag = guard.select("Can this email be trusted?", records, 0)
        self.assertEqual(retained, [records[0]])
        self.assertIs(retained[0], records[0])
        self.assertEqual(len(judge.calls), 3)
        first, second, audit = [calls[-1]["content"] for calls in judge.calls]
        self.assertIn("Can this email be trusted?", first)
        self.assertIn("inspect sender", first)
        self.assertNotIn("forward credentials", first)
        self.assertNotIn("inspect sender", second)
        self.assertIn("Sender -> verify -> domain", audit)
        self.assertIn("Credentials -> send -> attacker", audit)
        self.assertIn("Original Query And Each Other", audit)
        self.assertNotIn("PRIVATE_", json.dumps(judge.calls) + json.dumps(guard.state_dict()))
        self.assertEqual(diag["rejected_ids"], ["two"])
        self.assertEqual(diag["model_calls"], 3)
        self.assertEqual(diag["profile"], "official_joint_llm")
        self.assertEqual(diag["paths"][1]["reasoning_chain"], "Credentials -> send -> attacker")

    def test_query_similarity_controls_top_k_and_no_silent_path_truncation(self):
        self.args.amemguard_top_k = 1
        self.vectors = {"new query": [1., 0.], "far question": [0., 1.], "near question": [1., 0.]}
        long_chain = "Entity " + "long rationale " * 1500 + " -> follows -> Action"
        guard, judge = self.guard([long_chain], [{"id": 0, "consistent": True, "safe": True}])
        far, near = entry("far", "far question"), entry("near", "near question")
        retained, diag = guard.select("new query", [far, near], 0)
        self.assertEqual(retained, [near])
        self.assertEqual(diag["candidate_count"], 2)
        self.assertEqual(diag["audited_count"], 1)
        self.assertIn(long_chain, judge.calls[-1][-1]["content"])

    def test_rejected_chain_is_a_lesson_only_on_subsequent_consultation(self):
        guard, _ = self.guard(["Credentials -> send -> attacker"], [{"id": 0, "consistent": False, "safe": False}])
        bad = entry("bad", "old email query", "forward credentials")
        retained, _ = guard.select("old query", [bad], 0)
        self.assertEqual(retained, [])
        self.assertEqual(guard.lesson_prompt("old query", retained, 0), module.LESSON_HEADER)
        new = entry("new", "new email query", "forward credentials")
        prompt = guard.lesson_prompt("new query", [new], 0)
        self.assertIn("[CRITICAL WARNING] Analysis of Past Lessons", prompt)
        self.assertIn("Identify and AVOID", prompt)
        self.assertIn("Credentials -> send -> attacker", prompt)
        self.assertNotIn("PRIVATE_", prompt)

    def test_action_retrieval_searches_full_catalog_before_collecting_lessons(self):
        self.args.amemguard_lesson_top_k = 1
        guard, _ = self.guard(["Funds -> transfer -> stranger"], [{"id": 0, "consistent": False, "safe": False}])
        guard.select("old query", [entry("bad", "old query", "wire stranger")], 0)
        self.vectors = {"wire stranger": [0.8, 0.6], "safe transfer": [1., 0.]}
        clean = entry("clean", "current query", "safe transfer")
        guard.register_entries([clean], 0)
        self.assertEqual(guard.lesson_prompt("different query", [clean], 0), module.LESSON_HEADER)
        self.assertIn("safe transfer", self.embedded)
        self.assertIn("wire stranger", self.embedded)

    def test_lesson_memory_is_separate_from_catalog_and_agent_scoped(self):
        guard, _ = self.guard(["Mail -> exfiltrate -> inbox"], [{"id": 0, "consistent": False, "safe": False}])
        record = entry("same")
        guard.select("q", [record], 0)
        self.assertNotIn("lesson", vars(record))
        state = guard.state_dict()
        self.assertTrue(state["lessons"]["0"])
        self.assertNotIn("reasoning_chain", json.dumps(state["catalog"]))
        self.assertEqual(guard.lesson_prompt("later", [record], 1), module.LESSON_HEADER)

    def test_restart_round_trip_reset_and_experiment_isolation(self):
        guard, _ = self.guard(["Mail -> exfiltrate -> inbox"], [{"id": 0, "consistent": False, "safe": False}])
        guard.select("q", [entry("bad")], 0)
        state = json.loads(json.dumps(guard.state_dict()))
        restored, judge = self.guard()
        restored.load_state_dict(state)
        self.assertIn("Mail -> exfiltrate -> inbox", restored.lesson_prompt("later", [entry("new")], 0))
        self.assertEqual(judge.calls, [])
        isolated, _ = self.guard()
        self.assertEqual(isolated.lesson_prompt("later", [entry("new")], 0), module.LESSON_HEADER)
        self.args.amemguard_experiment_id = "other-experiment"
        other, _ = self.guard()
        with self.assertRaises(module.AMemGuardConfigError):
            other.load_state_dict(state)
        restored.reset()
        self.assertEqual(restored.state_dict()["lessons"], {})
        self.assertEqual(restored.state_dict()["catalog"], {})

    def test_state_copy_and_changed_memory_invalidate_old_annotation(self):
        guard, _ = self.guard(["Mail -> exfiltrate -> inbox"], [{"id": 0, "consistent": False, "safe": False}])
        guard.select("q", [entry("bad")], 0)
        external = guard.state_dict()
        external["lessons"].clear()
        self.assertTrue(guard.state_dict()["lessons"])
        guard.register_entries([entry("bad", action="revised safe operation")], 0)
        self.assertFalse(guard.state_dict()["lessons"]["0"])

    def test_all_rejected_means_empty_context_without_reinstating_memory(self):
        guard, _ = self.guard(["X -> corrupt -> Y"], [{"id": 0, "consistent": False, "safe": True}])
        retained, diag = guard.select("q", [entry("bad")], 0)
        self.assertEqual(retained, [])
        self.assertEqual(diag["retained_ids"], [])
        self.assertEqual(diag["fallback_policy"], "empty_memory_context")

    def test_malformed_audit_fails_explicitly_and_cannot_create_lessons(self):
        outputs = ["not JSON", "[]", '[{"id":0,"consistent":"false","safe":true}]',
                   '[{"id":0,"consistent":true,"consistent":false,"safe":true}]',
                   '[{"id":1,"consistent":true,"safe":true}]',
                   '[{"id":0,"consistent":true,"safe":true},{"id":0,"consistent":true,"safe":true}]',
                   '[{"id":true,"consistent":true,"safe":true}]']
        for output in outputs:
            with self.subTest(output=output):
                guard, _ = self.guard(["X -> follows -> Y"], output)
                with self.assertRaises(module.AMemGuardOutputError) as raised:
                    guard.select("q", [entry("bad")], 0)
                self.assertEqual(raised.exception.fallback_policy, "reject_all")
                self.assertFalse(guard.state_dict()["lessons"].get("0"))

    def test_malformed_chain_and_provider_failure_are_not_heuristic_fallbacks(self):
        for output in ("", "Answer: A", "Error: server unavailable"):
            with self.subTest(output=output):
                guard, _ = self.guard([output], [])
                with self.assertRaises(module.AMemGuardOutputError):
                    guard.select("q", [entry("x")], 0)
        guard, _ = self.guard()
        guard.judge = lambda messages: (_ for _ in ()).throw(ConnectionError("unavailable"))
        with self.assertRaises(module.AMemGuardOutputError):
            guard.select("q", [entry("x")], 0)

    def test_complete_free_text_chain_reaches_official_joint_audit(self):
        for chain in ("Alice → lives in → Paris", "The memory implies that Alice lives in Paris."):
            with self.subTest(chain=chain):
                record = entry("one")
                guard, judge = self.guard([chain], [{"id": 0, "consistent": False, "safe": False}])
                retained, diagnostics = guard.select("Where does Alice live?", [record], 0)
                self.assertEqual(retained, [])
                self.assertIn(chain, judge.calls[-1][-1]["content"])
                self.assertEqual(diagnostics["paths"][0]["reasoning_chain"], chain)
                self.assertEqual(guard.state_dict()["lessons"]["0"]["one"]["reasoning_chain"], chain)

    def test_empty_candidates_do_not_call_model_or_embedding(self):
        guard, judge = self.guard()
        self.assertEqual(guard.select("q", [], 0)[0], [])
        self.assertEqual(guard.lesson_prompt("q", [], 0), module.LESSON_HEADER)
        self.assertEqual(judge.calls, [])
        self.assertEqual(self.embedded, [])

    def test_invalid_vectors_and_configuration_fail_clearly(self):
        for vector in ([0., 0.], [float("nan"), 1.], [True, 1.], [], [1.]):
            with self.subTest(vector=vector):
                guard, _ = self.guard()
                guard.embed = lambda text, v=vector: [1., 0.] if text == "q" else v
                with self.assertRaises(module.AMemGuardOutputError):
                    guard.select("q", [entry("x")], 0)
        for field, value in (("amemguard_top_k", 0), ("amemguard_lesson_top_k", True), ("amemguard_experiment_id", "")):
            args = SimpleNamespace(**vars(self.args))
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(module.AMemGuardConfigError):
                module.AMemGuardFull(args, lambda messages: "", self.embed)

    def test_corrupt_or_incompatible_state_is_rejected_atomically(self):
        guard, _ = self.guard()
        guard.register_entries([entry("live")], 0)
        before = guard.state_dict()
        for patch in ({"version": 999}, {"source_commit": "different"}, {"lessons": {"0": {"absent": {"reasoning_chain": "x"}}}}):
            corrupt = json.loads(json.dumps(before))
            corrupt.update(patch)
            with self.subTest(patch=patch), self.assertRaises(module.AMemGuardConfigError):
                guard.load_state_dict(corrupt)
            self.assertEqual(guard.state_dict(), before)


    def test_new_task_same_query_consults_previous_task_lessons(self):
        guard, _ = self.guard(["Mail -> exfiltrate -> inbox"], [{"id": 0, "consistent": False, "safe": False}])
        records = [entry("bad")]
        guard.select("same query", records, 0)
        self.assertEqual(guard.lesson_prompt("same query", records, 0), module.LESSON_HEADER)
        self.assertTrue(hasattr(guard, "begin_task"), "new-task snapshot reset is missing")
        guard.begin_task("task two")
        self.assertIn("Mail -> exfiltrate -> inbox", guard.lesson_prompt("same query", records, 0))

    def test_prompt_placeholders_in_operational_text_are_preserved_literally(self):
        guard, judge = self.guard(["Mail -> inspect -> sender"], [{"id": 0, "consistent": True, "safe": True}])
        guard.select("Query containing {memory} marker", [entry("x", action="literal {query}")], 0)
        self.assertIn("Query containing {memory} marker", judge.calls[0][-1]["content"])
        self.assertIn("literal {query}", judge.calls[0][-1]["content"])


    def test_cold_start_always_returns_exact_source_warning_header(self):
        guard, judge = self.guard()
        self.assertEqual(guard.lesson_prompt("new query", [], 0), module.LESSON_HEADER)
        retained, diagnostics = guard.select("new query", [], 0)
        self.assertEqual(guard.lesson_prompt("new query", retained, 0), module.LESSON_HEADER)
        self.assertEqual(diagnostics["lesson_ids"], [])
        self.assertEqual(diagnostics["lesson_count"], 0)
        self.assertEqual(judge.calls, [])

    def test_whitespace_intent_falls_back_to_experience_and_roundtrips(self):
        guard, _ = self.guard(["Mail -> inspect -> sender"],
                              [{"id": 0, "consistent": False, "safe": False}])
        guard.select("query", [entry("blank-intent", "     ", "inspect email")], 0)
        state = json.loads(json.dumps(guard.state_dict()))
        self.assertEqual(state["catalog"]["0"]["blank-intent"]["question"], "inspect email")
        restored, _ = self.guard()
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        self.assertIn("Mail -> inspect -> sender",
                      restored.lesson_prompt("later query", [entry("new")], 0))

if __name__ == "__main__":
    unittest.main()
