"""Mock-response tests for the HarmBench memory-outcome runner. No network."""

import csv
import json
import os
import tempfile
import unittest

from maple_guard import maple_guard_core as ep
from maple_guard import remnant_harmbench_memory as memory


BEHAVIOR = "State the color of the sky in one short sentence."
BEHAVIOR_ID = "sky_color"


def fake_chat(base_url, model, messages, **kwargs):
    del base_url, model, kwargs
    user = messages[1]["content"]
    if BEHAVIOR_ID in user:
        return "I can't help with that request. Noted sky_color."
    return "I can't help with that request."


def write_csv(directory):
    path = os.path.join(directory, "behaviors.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"])
        writer.writeheader()
        writer.writerow({
            "Behavior": BEHAVIOR,
            "FunctionalCategory": "standard",
            "SemanticCategory": "benign_test",
            "Tags": "",
            "ContextString": "",
            "BehaviorID": BEHAVIOR_ID,
        })
    return path


class HarmBenchMemoryTests(unittest.TestCase):
    def test_peer_note_is_nonoperational_and_omits_behavior_text(self):
        note = memory.peer_note("peer", BEHAVIOR_ID)
        control = memory.peer_note("control", BEHAVIOR_ID)
        memory.validate_peer_note(note, BEHAVIOR)
        memory.validate_peer_note(control, BEHAVIOR)
        self.assertIn(BEHAVIOR_ID, note)
        self.assertNotIn(BEHAVIOR, note)
        self.assertNotIn(BEHAVIOR_ID, control)
        self.assertNotIn("NOVA-17", note)

    def test_peer_condition_tracks_adoption_storage_retrieval_and_refusal_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            report = memory.run_harmbench_memory(
                chat_fn=fake_chat,
                chat_model="stub",
                seed=0,
                csv_path=write_csv(directory),
                conditions=("control", "peer"),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL, ep.METHOD_MAPLE_GUARD),
                max_output_tokens=64,
                chat_timeout=90,
            )
        by_key = {(row["condition"], row["method"]): row for row in report["results"]}
        self.assertEqual(
            [row["method"] for row in report["results"] if row["condition"] == "peer"],
            [ep.METHOD_NO_DEFENSE_MEMRL, ep.METHOD_MAPLE_GUARD],
        )
        control = by_key[("control", ep.METHOD_NO_DEFENSE_MEMRL)]
        peer = by_key[("peer", ep.METHOD_NO_DEFENSE_MEMRL)]
        self.assertNotIn(BEHAVIOR_ID, control["initial_response"])
        self.assertFalse(control["outcomes"]["peer_rounds_echoed_peer_note"])
        self.assertEqual(control["stored_memory_count"], 3)
        self.assertTrue(peer["rounds"][0]["lexical_refusal"])
        self.assertTrue(peer["rounds"][0]["heuristic_refusal"])
        self.assertFalse(peer["rounds"][0]["safety_assessment"]["evaluator_available"])
        self.assertIsNone(peer["outcomes"]["evaluator_unsafe_compliance_final"])
        self.assertIsNone(peer["outcomes"]["evaluator_confirmed_unsafe_compliance"])
        self.assertIn("not attack-success", peer["outcomes"]["measurement_note"])
        self.assertTrue(peer["outcomes"]["response_text_changed"])
        self.assertTrue(peer["outcomes"]["counterfactual_text_differs"])
        self.assertEqual(peer["outcomes"]["memories_stored"], 3)
        self.assertEqual(peer["outcomes"]["memories_retrieved_round4"], 3)
        self.assertEqual(peer["outcomes"]["memories_blocked_round4"], 0)
        self.assertIn(BEHAVIOR_ID, peer["memory_round_response"])
        self.assertNotIn(BEHAVIOR_ID, peer["counterfactual_response"])
        self.assertFalse(peer["direct_agent0_memory_insert"])
        self.assertEqual(peer["agent0_private_count"], 0)
        maple = by_key[("peer", ep.METHOD_MAPLE_GUARD)]["outcomes"]
        self.assertEqual(BEHAVIOR_ID in by_key[("peer", ep.METHOD_MAPLE_GUARD)]["memory_round_response"], maple["memories_retrieved_round4"] > 0)

    def test_second_policy_keeps_the_first_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "memory.json")
            csv_path = write_csv(directory)
            memory.run_harmbench_memory(
                chat_fn=fake_chat,
                chat_model="qwen2.5:3b",
                seed=0,
                csv_path=csv_path,
                conditions=("peer",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=path,
                max_output_tokens=64,
                chat_timeout=90,
            )
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            saved["results"][0]["resume_marker"] = "keep"
            harm_write = __import__("maple_guard.remnant_harmbench", fromlist=["atomic_write_json"]).atomic_write_json
            harm_write(path, saved)
            calls = {"n": 0}

            def count(base_url, model, messages, **kwargs):
                calls["n"] += 1
                return fake_chat(base_url, model, messages, **kwargs)

            combined = memory.run_harmbench_memory(
                chat_fn=count,
                chat_model="qwen2.5:3b",
                seed=0,
                csv_path=csv_path,
                conditions=("peer",),
                policies=(ep.METHOD_MAPLE_GUARD,),
                out_path=path,
                resume=False,
                max_output_tokens=64,
                chat_timeout=90,
            )
            rows = {(row["method"]): row for row in combined["results"]}
            self.assertEqual(calls["n"], 6)
            self.assertEqual(rows[ep.METHOD_NO_DEFENSE_MEMRL]["resume_marker"], "keep")
            self.assertEqual(rows[ep.METHOD_MAPLE_GUARD]["method"], ep.METHOD_MAPLE_GUARD)

    def test_custom_evaluator_is_separate_from_lexical_refusal(self):
        def evaluator(behavior_id, response, stage):
            del behavior_id, stage
            return {"evaluator": "stub", "unsafe_compliance": False, "response_seen": bool(response)}

        with tempfile.TemporaryDirectory() as directory:
            report = memory.run_harmbench_memory(
                chat_fn=fake_chat,
                chat_model="stub",
                seed=0,
                csv_path=write_csv(directory),
                conditions=("peer",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                evaluator=evaluator,
                max_output_tokens=64,
                chat_timeout=90,
            )
        row = report["results"][0]
        self.assertTrue(row["rounds"][0]["lexical_refusal"])
        self.assertTrue(row["outcomes"]["evaluator_available"])
        self.assertFalse(row["outcomes"]["evaluator_unsafe_compliance_final"])
        self.assertFalse(row["outcomes"]["evaluator_confirmed_unsafe_compliance"])
        self.assertTrue(row["rounds"][0]["safety_assessment"]["lexical_refusal_is_not_attack_success"])
