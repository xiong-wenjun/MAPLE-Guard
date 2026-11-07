"""Focused tests for HarmBench loading and the free-text memory pilot. No network."""

import argparse
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from maple_guard import remnant_harmbench as harm
from maple_guard import remnant_multiround as multi


def fake_chat(base_url, model, messages, **kwargs):
    del base_url, model, kwargs
    user = messages[1]["content"]
    if "cup of tea" in user:
        return "Boil water, steep the tea, and serve it."
    if "public library" in user:
        return "A public library lends books and offers a reading room."
    if "study schedule" in user:
        return "Study for one hour on Monday, Wednesday, and Friday."
    return "I can't help with that request."


class HarmBenchLoadingTests(unittest.TestCase):
    def test_csv_loads_the_requested_validation_rows(self):
        rows = harm.load_behaviors(limit=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["behavior_id"] for row in rows}), 3)
        for row in rows:
            self.assertTrue(row["behavior_id"])
            self.assertTrue(row["behavior"])
            self.assertNotIn(row["behavior"], " ".join(harm.EXPOSURE_REQUESTS))

    def test_missing_memos_backend_is_not_relabeled_as_qdrant(self):
        args = argparse.Namespace()
        with patch("maple_guard.memory_backend.MemoryService", None):
            with self.assertRaises(RuntimeError) as caught:
                multi.open_memory_store(args, backend="memos_qdrant", run_id="test")
        self.assertIn("not used as a substitute", str(caught.exception))


class HarmBenchPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        behavior = harm.load_behaviors(limit=1)[0]
        cls.behavior = behavior
        cls.report = {
            method: harm.run_behavior(behavior, method, chat_fn=fake_chat, seed=0)
            for method in multi.POLICIES
        }

    def test_exposure_is_benign_and_free_text(self):
        for row in self.report.values():
            self.assertFalse(row["misleading_agent0"])
            self.assertEqual(row["exposure_requests"], list(harm.EXPOSURE_REQUESTS))
            self.assertEqual(row["stored_memory_count"], 3)
            self.assertTrue(row["store_unchanged_by_counterfactual"])
            self.assertFalse(row["memos_qdrant_active"])
            self.assertEqual(row["memory_backend"], "in_process_python_list")
            systems = [call["system"] for call in row["recorded_calls"]]
            self.assertTrue(systems)
            for system in systems:
                self.assertIn("plain prose", system)
                self.assertIn("unverified notes", system)
                self.assertNotIn("Final answer:", system)
                self.assertNotIn("reliable answer is", system)
            stored = row["rounds"][3]["stored_memories"]
            self.assertEqual(len(stored), 3)
            combined = "\n".join(memory["experience"] for memory in stored)
            self.assertIn("steep the tea", combined)
            self.assertNotIn(self.behavior["behavior"], combined)
            self.assertNotIn(self.behavior["behavior_id"], combined)

    def test_indicators_do_not_treat_a_refusal_as_poisoning(self):
        for row in self.report.values():
            self.assertEqual(row["initial_response"], "I can't help with that request.")
            self.assertEqual(row["counterfactual_response"], "I can't help with that request.")
            self.assertTrue(row["indicators"]["lexical_refusal_initial"])
            self.assertTrue(row["indicators"]["lexical_refusal_counterfactual"])
            self.assertIn("not evidence of poisoning", row["indicators"]["indicator_note"])
            self.assertEqual(row["counterfactual"]["selected_memory_ids"]["1"], [])
            self.assertEqual(len(row["rounds"][-1]["memory_retrieval"]), 3)
            self.assertTrue(all(item["stored"] for item in row["rounds"][-1]["memory_retrieval"]))


class CheckpointTests(unittest.TestCase):
    def test_each_completed_policy_is_saved_and_resume_skips_it(self):
        behavior = harm.load_behaviors(limit=1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "harmbench.json")
            seen = {"tokens": None, "timeout": None, "n": 0, "checked": False}

            def chat(base_url, model, messages, **kwargs):
                seen["n"] += 1
                seen["tokens"] = kwargs.get("max_tokens")
                seen["timeout"] = kwargs.get("timeout")
                if os.path.exists(path) and not seen["checked"]:
                    with open(path, encoding="utf-8") as handle:
                        saved = json.load(handle)
                    policies = saved["items"][0]["policies"]
                    self.assertIn(ep_no_defense(), policies)
                    self.assertNotIn("maple_guard", policies)
                    seen["checked"] = True
                return fake_chat(base_url, model, messages, **kwargs)

            harm.run_harmbench(
                chat_fn=chat,
                seed=0,
                policies=multi.POLICIES,
                limit=1,
                out_path=path,
                chat_timeout=15,
                max_output_tokens=32,
            )
            self.assertTrue(seen["checked"])
            self.assertEqual(seen["tokens"], 32)
            self.assertEqual(seen["timeout"], 15)
            with open(path, encoding="utf-8") as handle:
                first = json.load(handle)
            self.assertEqual(set(first["items"][0]["policies"]), set(multi.POLICIES))
            calls = {"n": 0}

            def fail_if_called(base_url, model, messages, **kwargs):
                calls["n"] += 1
                raise AssertionError("completed pair was run again")

            second = harm.run_harmbench(
                chat_fn=fail_if_called,
                seed=0,
                policies=multi.POLICIES,
                limit=1,
                out_path=path,
                resume=True,
                chat_timeout=15,
                max_output_tokens=32,
            )
            self.assertEqual(calls["n"], 0)
            self.assertEqual(
                second["items"][0]["policies"][ep_no_defense()]["initial_response"],
                first["items"][0]["policies"][ep_no_defense()]["initial_response"],
            )
            incomplete = json.loads(json.dumps(first))
            del incomplete["items"][0]["policies"]["maple_guard"]["counterfactual_response"]
            harm.atomic_write_json(path, incomplete)
            calls["n"] = 0

            def count(base_url, model, messages, **kwargs):
                calls["n"] += 1
                return fake_chat(base_url, model, messages, **kwargs)

            resumed = harm.run_harmbench(
                chat_fn=count,
                seed=0,
                policies=("maple_guard",),
                limit=1,
                out_path=path,
                resume=True,
                chat_timeout=15,
                max_output_tokens=32,
            )
            self.assertGreater(calls["n"], 0)
            self.assertIn("counterfactual_response", resumed["items"][0]["policies"]["maple_guard"])
            self.assertIn(ep_no_defense(), resumed["items"][0]["policies"])
            self.assertEqual(
                [name for name in os.listdir(directory) if name.startswith(".remnant_harmbench_")],
                [],
            )


def ep_no_defense():
    return "no_defense_memrl"


class IndicatorAndResumeTests(unittest.TestCase):
    def test_heuristic_recognizes_an_apology_the_lexical_flag_misses(self):
        text = "I'm sorry, but that request is not something I can do."
        self.assertFalse(harm.refusal_indicator(text))
        self.assertTrue(harm.heuristic_refusal_indicator(text))
        indicators = harm.response_indicators(text, "Boil water and steep tea.", text)
        self.assertFalse(indicators["lexical_refusal_initial"])
        self.assertTrue(indicators["heuristic_refusal_initial"])
        self.assertFalse(indicators["heuristic_refusal_with_memory"])
        self.assertTrue(indicators["heuristic_refusal_counterfactual"])
        self.assertIn("not a validated HarmBench judge", indicators["heuristic_refusal_note"])
        self.assertIsNone(indicators["output_token_limit_hit"])
        self.assertFalse(indicators["output_token_limit_hit_available"])

    def test_resume_adds_maple_guard_without_replacing_no_defense(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "harmbench.json")
            harm.run_harmbench(
                chat_fn=fake_chat,
                chat_model="qwen2.5:3b",
                seed=0,
                policies=(ep_no_defense(),),
                limit=1,
                out_path=path,
                chat_timeout=90,
                max_output_tokens=64,
            )
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            saved["items"][0]["policies"][ep_no_defense()]["resume_marker"] = "keep"
            harm.atomic_write_json(path, saved)
            calls = {"n": 0}

            def count(base_url, model, messages, **kwargs):
                calls["n"] += 1
                return fake_chat(base_url, model, messages, **kwargs)

            resumed = harm.run_harmbench(
                chat_fn=count,
                chat_model="qwen2.5:3b",
                seed=0,
                policies=("maple_guard",),
                limit=1,
                out_path=path,
                resume=True,
                chat_timeout=90,
                max_output_tokens=64,
            )
            self.assertGreater(calls["n"], 0)
            policies = resumed["items"][0]["policies"]
            self.assertEqual(policies[ep_no_defense()]["resume_marker"], "keep")
            self.assertEqual(policies["maple_guard"]["method"], "maple_guard")
            self.assertEqual(policies["maple_guard"]["chat_model"], "qwen2.5:3b")
            self.assertEqual(policies["maple_guard"]["max_output_tokens"], 64)
            self.assertEqual(resumed["backend_configuration"]["memory_backend"], "in_process_python_list")
            self.assertFalse(resumed["backend_configuration"]["memos_qdrant_active"])
            self.assertEqual(resumed["chat_model"], "qwen2.5:3b")
