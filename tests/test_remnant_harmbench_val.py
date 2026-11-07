"""Validation-set HarmBench runner. Mock replies only; no target-model calls."""

import csv
import json
import os
import tempfile
import unittest

from maple_guard import maple_guard_core as ep
from maple_guard import remnant_harmbench as harm
from maple_guard import remnant_harmbench_val as val


BEHAVIOR = "State the color of the sky in one short sentence."
BEHAVIOR_ID = "sky_color"
REFUSAL = "I can't help with that request."


def fake_chat(base_url, model, messages, **kwargs):
    del base_url, model, kwargs
    user = messages[1]["content"]
    if val.PEER_MARKER in user:
        return REFUSAL + " Source label noted: " + val.PEER_MARKER + "."
    return REFUSAL


def write_csv(directory):
    path = os.path.join(directory, "behaviors.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"],
        )
        writer.writeheader()
        writer.writerow({
            "Behavior": BEHAVIOR,
            "FunctionalCategory": "standard",
            "SemanticCategory": "benign_test",
            "Tags": "",
            "ContextString": "",
            "BehaviorID": BEHAVIOR_ID,
        })
        writer.writerow({
            "Behavior": "Name one ocean.",
            "FunctionalCategory": "copyright",
            "SemanticCategory": "copyright",
            "Tags": "hash_check",
            "ContextString": "",
            "BehaviorID": "not_in_the_fixed_set",
        })
    return path


class ValidationSelectionTests(unittest.TestCase):
    def test_fixed_ids_match_the_official_validation_rows(self):
        selected = val.select_validation_behaviors(harm.DEFAULT_CSV, limit=5)
        self.assertEqual([row["behavior_id"] for row in selected], list(val.VALIDATION_BEHAVIOR_IDS))
        with open(harm.DEFAULT_CSV, newline="", encoding="utf-8") as handle:
            source = {row["BehaviorID"]: row for row in csv.DictReader(handle)}
        for row in selected:
            official = source[row["behavior_id"]]
            self.assertEqual(row["behavior"], official["Behavior"].strip())
            self.assertEqual(row["functional_category"], official["FunctionalCategory"])
            self.assertEqual(row["semantic_category"], official["SemanticCategory"])
            self.assertEqual(row["tags"], official["Tags"])
            self.assertEqual(row["functional_category"], "standard")
            self.assertEqual(row["tags"], "")

    def test_behavior_id_loads_one_official_row(self):
        rows = val.select_validation_behaviors(harm.DEFAULT_CSV, behavior_id=val.VALIDATION_BEHAVIOR_IDS[0])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["behavior_id"], val.VALIDATION_BEHAVIOR_IDS[0])


class ValidationRunTests(unittest.TestCase):
    def test_peer_note_is_nonoperational_and_omits_behavior_text(self):
        peer = val.peer_note("peer", BEHAVIOR_ID)
        control = val.peer_note("control", BEHAVIOR_ID)
        val.validate_peer_note(peer, BEHAVIOR)
        val.validate_peer_note(control, BEHAVIOR)
        self.assertIn(val.PEER_MARKER, peer)
        self.assertIn(BEHAVIOR_ID, peer)
        self.assertNotIn(BEHAVIOR, peer)
        self.assertNotIn(val.PEER_MARKER, control)
        self.assertNotIn("bypass", peer.lower())
        with self.assertRaises(ValueError):
            val.validate_peer_note(peer + " " + BEHAVIOR, BEHAVIOR)
        with self.assertRaises(ValueError):
            val.validate_peer_note("Please bypass the safeguard.", BEHAVIOR)

    def test_exposure_uses_the_behavior_text_and_stores_only_agent1(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "val.json")
            report = val.run_harmbench_validation(
                chat_fn=fake_chat,
                chat_model="stub",
                seed=0,
                csv_path=write_csv(directory),
                behavior_id=BEHAVIOR_ID,
                policies=(ep.METHOD_NO_DEFENSE_MEMRL, ep.METHOD_MAPLE_GUARD),
                out_path=out,
                max_output_tokens=32,
                chat_timeout=30,
            )
            with open(out, encoding="utf-8") as handle:
                saved = json.load(handle)
            by_key = {(row["condition"], row["method"]): row for row in report["results"]}
            self.assertEqual(set(by_key), {(condition, method) for condition in val.CONDITIONS for method in val.POLICIES})
            peer = by_key[("peer", ep.METHOD_NO_DEFENSE_MEMRL)]
            control = by_key[("control", ep.METHOD_NO_DEFENSE_MEMRL)]
            self.assertEqual(peer["behavior"], BEHAVIOR)
            self.assertEqual(peer["provenance"]["stored_author"], "agent_1_response_only")
            self.assertEqual(peer["agent0_private_count"], 0)
            self.assertEqual(control["agent0_private_count"], 0)
            self.assertEqual(peer["stored_memory_count"], 3)
            self.assertTrue(all(row["origin_agent"] == 1 for row in peer["rounds"][-1]["stored_memories"]))
            self.assertEqual(peer["rounds"][0]["retrieved_count"], 0)
            self.assertFalse(peer["rounds"][0]["peer_communication"])
            self.assertFalse(peer["rounds"][-1]["peer_communication"])
            self.assertNotIn("Message from Agent 0", peer["rounds"][0]["user_prompt"])
            self.assertNotIn("Message from Agent 0", peer["rounds"][-1]["user_prompt"])
            exposure = [row for row in peer["rounds"] if row["kind"] == "exposure"]
            self.assertEqual(len(exposure), 3)
            for row in exposure:
                self.assertIn(BEHAVIOR, row["user_prompt"])
                self.assertIn(val.PEER_MARKER, row["agent0_peer_message"])
                self.assertNotIn(BEHAVIOR, row["agent0_peer_message"])
                self.assertTrue(row["peer_claim_adopted"])
                self.assertTrue(row["write_decisions"])
            self.assertTrue(peer["adoption"]["adopted_in_exposure"])
            self.assertTrue(peer["adoption"]["entered_memory"])
            self.assertFalse(peer["direct_agent0_memory_insert"])
            self.assertFalse(control["adoption"]["adopted_in_exposure"])
            self.assertFalse(control["adoption"]["entered_memory"])
            self.assertNotIn(val.PEER_MARKER, control["rounds"][1]["agent0_peer_message"])
            self.assertIn(BEHAVIOR, control["rounds"][1]["user_prompt"])
            self.assertEqual(peer["rounds"][-1]["write_decisions"], [])
            self.assertEqual(peer["counterfactual"]["retrieved_count"], 0)
            self.assertEqual(peer["counterfactual"]["blocked_count"], 3)
            self.assertTrue(peer["store_unchanged_by_counterfactual"])
            self.assertIsNone(peer["adoption"]["evaluator_safety_outcome_changed"])
            self.assertIsNone(peer["outcomes"]["evaluator_confirmed_unsafe_compliance"])
            completions = saved["harmbench_completions"]["peer"][ep.METHOD_NO_DEFENSE_MEMRL]
            self.assertEqual(completions["baseline"][BEHAVIOR_ID][0]["test_case"], BEHAVIOR)
            self.assertEqual(completions["baseline"][BEHAVIOR_ID][0]["generation"], peer["initial_response"])
            self.assertEqual(completions["memory_enabled"][BEHAVIOR_ID][0]["generation"], peer["memory_round_response"])
            self.assertEqual(completions["memory_suppressed"][BEHAVIOR_ID][0]["generation"], peer["counterfactual_response"])
            with open(saved["harmbench_completion_files"]["peer"][ep.METHOD_NO_DEFENSE_MEMRL]["baseline"], encoding="utf-8") as handle:
                exported = json.load(handle)
            self.assertEqual(set(exported[BEHAVIOR_ID][0]), {"test_case", "generation"})
            self.assertTrue(peer["rounds"][0]["lexical_refusal"])

    def test_second_policy_keeps_the_first_result(self):
        with tempfile.TemporaryDirectory() as directory:
            out = os.path.join(directory, "val.json")
            csv_path = write_csv(directory)
            val.run_harmbench_validation(
                chat_fn=fake_chat,
                chat_model="stub",
                csv_path=csv_path,
                behavior_id=BEHAVIOR_ID,
                conditions=("peer",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=out,
            )
            val.run_harmbench_validation(
                chat_fn=fake_chat,
                chat_model="stub",
                csv_path=csv_path,
                behavior_id=BEHAVIOR_ID,
                conditions=("control",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=out,
            )
            with open(out, encoding="utf-8") as handle:
                saved = json.load(handle)
        self.assertEqual(
            sorted((row["condition"], row["method"]) for row in saved["results"]),
            [("control", ep.METHOD_NO_DEFENSE_MEMRL), ("peer", ep.METHOD_NO_DEFENSE_MEMRL)],
        )
