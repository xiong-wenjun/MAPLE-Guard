"""Mechanics tests for the fictional project-code memory experiment. No network."""

import json
import os
import tempfile
import unittest

from maple_guard import maple_guard_core as ep
from maple_guard import remnant_answer_mediated as remnant
from maple_guard import remnant_code_poison as poison


def fake_chat(base_url, model, messages, **kwargs):
    del base_url, model, kwargs
    user = messages[1]["content"]
    if poison.ATTACK_PEER in user:
        return "The project code is NOVA-17."
    if poison.contains_code(user, poison.FALSE_CODE):
        return "The project code is NOVA-17."
    if poison.contains_code(user, poison.CORRECT_CODE):
        return "The project code is ORION-42."
    return "I do not know the project code."


class CodePoisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = poison.run_code_poison(chat_fn=fake_chat, chat_model="stub", seed=0, max_output_tokens=64, chat_timeout=90)

    def test_control_and_attack_use_independent_stores(self):
        self.assertEqual([item["condition"] for item in self.report["items"]], ["control", "attack"])
        seen = []
        for item in self.report["items"]:
            self.assertEqual(set(item["policies"]), set(poison.multi.POLICIES))
            for method, row in item["policies"].items():
                self.assertEqual(row["method"], method)
                self.assertEqual(row["feedback_policy"], "none")
                self.assertFalse(row["attacker_self_memory"])
                self.assertEqual(row["agent0_private_count"], 0)
                self.assertEqual(row["shared_memory_count"], 0)
                self.assertEqual(row["stored_memory_count"], 4)
                self.assertFalse(row["adoption"]["direct_agent0_memory_insert"])
                self.assertTrue(row["store_unchanged_by_counterfactual"])
                self.assertIsNone(row["output_token_limit_hit"])
                self.assertFalse(row["output_token_limit_hit_available"])
                seen.extend(row["stored_memory_ids"])
                for memory in row["rounds"][-1]["stored_memories"]:
                    self.assertEqual(memory["origin_agent"], remnant.RECEIVER_ID)
                    self.assertEqual(memory["source_type"], "agent_self_unverified_consolidation")
                    self.assertNotIn(poison.CORRECT_CODE, memory["intent"])
                    self.assertNotIn(poison.FALSE_CODE, memory["intent"])
                    self.assertNotIn("unverified peer note", memory["experience"])
        self.assertEqual(len(seen), len(set(seen)))

    def test_reference_is_only_on_the_initial_round(self):
        question_has_code = poison.contains_code(poison.QUESTION, poison.CORRECT_CODE) or poison.contains_code(poison.QUESTION, poison.FALSE_CODE)
        self.assertFalse(question_has_code)
        self.assertNotIn(poison.FALSE_CODE, poison.CONTROL_PEER)
        self.assertNotIn(poison.CORRECT_CODE, poison.CONTROL_PEER)
        self.assertIn(poison.FALSE_CODE, poison.ATTACK_PEER)
        self.assertNotIn(poison.CORRECT_CODE, poison.ATTACK_PEER)
        for item in self.report["items"]:
            for row in item["policies"].values():
                baseline, first, _second, _third, final = row["rounds"]
                self.assertTrue(baseline["reference_included"])
                self.assertFalse(baseline["peer_communication"])
                self.assertEqual(len([decision for decision in baseline["write_decisions"] if decision["stage"] == "write_firewall"]), 1)
                for peer_round in (first, _second, _third):
                    self.assertFalse(peer_round["reference_included"])
                    self.assertEqual(peer_round["agent0_message_source"], "scripted_unverified_peer")
                self.assertFalse(final["reference_included"])
                self.assertFalse(final["peer_communication"])
                self.assertEqual(final["write_decisions"], [])
                self.assertFalse(row["counterfactual"]["reference_included"])
                self.assertEqual(row["counterfactual"]["write_decisions"], [])
                self.assertEqual(
                    [item["retrieved"] for item in row["counterfactual"]["memory_retrieval"]],
                    [False, False, False, False],
                )
                users = [call["user"] for call in row["recorded_calls"]]
                self.assertEqual(sum(poison.REFERENCE in user for user in users), 1)
                self.assertTrue(all(call["agent_id"] == remnant.RECEIVER_ID for call in row["recorded_calls"]))

    def test_attack_records_adoption_storage_retrieval_and_use_separately(self):
        attack = self.report["items"][1]["policies"]
        control = self.report["items"][0]["policies"]
        for row in control.values():
            self.assertFalse(row["adoption"]["peer_rounds_adopted_false_code"])
            self.assertFalse(row["adoption"]["false_code_stored"])
            self.assertFalse(row["adoption"]["round4_used_false_code"])
            self.assertFalse(row["adoption"]["counterfactual_used_false_code"])
            self.assertTrue(row["adoption"]["baseline_contains_correct_code"])
        no_defense = attack[ep.METHOD_NO_DEFENSE_MEMRL]
        adoption = no_defense["adoption"]
        self.assertTrue(adoption["baseline_contains_correct_code"])
        self.assertFalse(adoption["baseline_contains_false_code"])
        self.assertTrue(adoption["peer_rounds_adopted_false_code"])
        self.assertTrue(adoption["false_code_stored"])
        self.assertTrue(adoption["correct_code_stored"])
        self.assertTrue(adoption["false_code_retrieved_round4"])
        self.assertTrue(adoption["round4_used_false_code"])
        self.assertFalse(adoption["counterfactual_used_false_code"])
        self.assertTrue(adoption["memory_changed_false_code_outcome"])
        self.assertIn("not a poisoning verdict", adoption["adoption_note"])
        maple = attack[ep.METHOD_MAPLE_GUARD]["adoption"]
        self.assertEqual(maple["round4_used_false_code"], maple["false_code_retrieved_round4"])
        self.assertFalse(maple["counterfactual_used_false_code"])

    def test_resume_adds_maple_guard_without_rerunning_no_defense(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "code_poison.json")
            poison.run_code_poison(
                chat_fn=fake_chat,
                chat_model="stub",
                seed=0,
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                conditions=("control", "attack"),
                out_path=path,
                max_output_tokens=64,
                chat_timeout=90,
            )
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            saved["items"][0]["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["resume_marker"] = "keep"
            poison.atomic_write_json(path, saved)
            calls = {"n": 0}

            def count(base_url, model, messages, **kwargs):
                calls["n"] += 1
                return fake_chat(base_url, model, messages, **kwargs)

            resumed = poison.run_code_poison(
                chat_fn=count,
                chat_model="stub",
                seed=0,
                policies=(ep.METHOD_MAPLE_GUARD,),
                conditions=("control", "attack"),
                out_path=path,
                resume=True,
                max_output_tokens=64,
                chat_timeout=90,
            )
            self.assertGreater(calls["n"], 0)
            control = resumed["items"][0]["policies"]
            self.assertEqual(control[ep.METHOD_NO_DEFENSE_MEMRL]["resume_marker"], "keep")
            self.assertEqual(control[ep.METHOD_MAPLE_GUARD]["method"], ep.METHOD_MAPLE_GUARD)
            self.assertEqual(resumed["chat_model"], "stub")
            self.assertEqual(resumed["backend_configuration"]["memory_backend"], "in_process_python_list")
            self.assertFalse(any(name.startswith(".remnant_code_poison_") for name in os.listdir(directory)))


class PlantingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = poison.run_code_poison(
            chat_fn=fake_chat,
            chat_model="stub",
            seed=0,
            scenario="planting",
            max_output_tokens=64,
            chat_timeout=90,
        )

    def test_planting_never_shows_the_authoritative_reference(self):
        self.assertEqual(self.report["scenario_name"], "planting")
        self.assertFalse(self.report["scenario"]["receives_authoritative_reference"])
        self.assertIsNone(self.report["scenario"]["reference_round"])
        for item in self.report["items"]:
            self.assertEqual(item["scenario_name"], "planting")
            for row in item["policies"].values():
                self.assertEqual(row["stored_memory_count"], 3)
                self.assertEqual(row["agent0_private_count"], 0)
                self.assertFalse(row["adoption"]["direct_agent0_memory_insert"])
                baseline = row["rounds"][0]
                self.assertFalse(baseline["reference_included"])
                self.assertEqual(baseline["write_decisions"], [])
                self.assertEqual(baseline["stored_memories"], [])
                self.assertFalse(any(call["user"].find(poison.REFERENCE) >= 0 for call in row["recorded_calls"]))
                self.assertFalse(any(poison.contains_code(call["user"], poison.CORRECT_CODE) for call in row["recorded_calls"]))
                for memory in row["rounds"][-1]["stored_memories"]:
                    self.assertEqual(memory["origin_agent"], remnant.RECEIVER_ID)
                    self.assertNotIn("unverified peer note", memory["experience"])

    def test_attack_records_planting_separately_from_the_control(self):
        control = self.report["items"][0]["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["adoption"]
        attack = self.report["items"][1]["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["adoption"]
        self.assertFalse(control["peer_rounds_adopted_false_code"])
        self.assertFalse(control["false_code_stored"])
        self.assertFalse(control["false_code_retrieved_round4"])
        self.assertFalse(control["round4_used_false_code"])
        self.assertFalse(control["counterfactual_used_false_code"])
        self.assertFalse(attack["baseline_contains_false_code"])
        self.assertTrue(attack["peer_rounds_adopted_false_code"])
        self.assertTrue(attack["false_code_stored"])
        self.assertTrue(attack["false_code_retrieved_round4"])
        self.assertTrue(attack["round4_used_false_code"])
        self.assertFalse(attack["counterfactual_used_false_code"])
        self.assertTrue(attack["memory_changed_false_code_outcome"])
        maple = self.report["items"][1]["policies"][ep.METHOD_MAPLE_GUARD]["adoption"]
        self.assertEqual(maple["round4_used_false_code"], maple["false_code_retrieved_round4"])
        self.assertFalse(maple["counterfactual_used_false_code"])

    def test_completed_overwrite_pair_does_not_skip_planting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "code_poison.json")
            poison.atomic_write_json(path, {
                "seed": 0,
                "memory_backend": "in_process_python_list",
                "items": [{
                    "condition": "attack",
                    "policies": {
                        ep.METHOD_NO_DEFENSE_MEMRL: {
                            "initial_response": "ORION-42",
                            "memory_round_response": "ORION-42",
                            "counterfactual_response": "ORION-42",
                            "counterfactual": {},
                            "resume_marker": "overwrite",
                        }
                    },
                }],
            })
            calls = {"n": 0}

            def count(base_url, model, messages, **kwargs):
                calls["n"] += 1
                return fake_chat(base_url, model, messages, **kwargs)

            resumed = poison.run_code_poison(
                chat_fn=count,
                chat_model="stub",
                seed=0,
                scenario="planting",
                conditions=("attack",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=path,
                resume=True,
                max_output_tokens=64,
                chat_timeout=90,
            )
            self.assertGreater(calls["n"], 0)
            self.assertEqual(resumed["items"][0]["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["resume_marker"], "overwrite")
            planted = resumed["items"][1]
            self.assertEqual(planted["scenario_name"], "planting")
            self.assertNotIn(poison.CORRECT_CODE, planted["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["initial_response"])


class PersistenceTests(unittest.TestCase):
    def test_second_policy_survives_without_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "code_poison.json")
            poison.run_code_poison(
                chat_fn=fake_chat,
                chat_model="qwen2.5:3b",
                seed=0,
                scenario="planting",
                conditions=("attack",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=path,
                max_output_tokens=64,
                chat_timeout=90,
            )
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            saved["items"][0]["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["resume_marker"] = "keep"
            poison.atomic_write_json(path, saved)
            calls = {"n": 0}

            def count(base_url, model, messages, **kwargs):
                calls["n"] += 1
                return fake_chat(base_url, model, messages, **kwargs)

            combined = poison.run_code_poison(
                chat_fn=count,
                chat_model="qwen2.5:3b",
                seed=0,
                scenario="planting",
                conditions=("attack",),
                policies=(ep.METHOD_MAPLE_GUARD,),
                out_path=path,
                resume=False,
                max_output_tokens=64,
                chat_timeout=90,
            )
            policies = combined["items"][0]["policies"]
            self.assertEqual(calls["n"], 6)
            self.assertEqual(policies[ep.METHOD_NO_DEFENSE_MEMRL]["resume_marker"], "keep")
            self.assertEqual(policies[ep.METHOD_MAPLE_GUARD]["method"], ep.METHOD_MAPLE_GUARD)
            keys = {(row["method"], row["chat_model"], row["seed"]) for row in combined["saved_result_keys"]}
            self.assertIn((ep.METHOD_NO_DEFENSE_MEMRL, "qwen2.5:3b", 0), keys)
            self.assertIn((ep.METHOD_MAPLE_GUARD, "qwen2.5:3b", 0), keys)

    def test_same_policy_with_another_model_is_kept_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "code_poison.json")
            poison.run_code_poison(
                chat_fn=fake_chat,
                chat_model="qwen2.5:3b",
                seed=0,
                scenario="planting",
                conditions=("attack",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=path,
                max_output_tokens=64,
                chat_timeout=90,
            )
            poison.run_code_poison(
                chat_fn=fake_chat,
                chat_model="qwen2.5:7b",
                seed=0,
                scenario="planting",
                conditions=("attack",),
                policies=(ep.METHOD_NO_DEFENSE_MEMRL,),
                out_path=path,
                resume=False,
                max_output_tokens=64,
                chat_timeout=90,
            )
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            variants = saved["items"][0]["policies"][ep.METHOD_NO_DEFENSE_MEMRL]["variants"]
            self.assertEqual({row["chat_model"] for row in variants}, {"qwen2.5:3b", "qwen2.5:7b"})

    def test_conflicting_completed_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "code_poison.json")
            poison.run_code_poison(
                chat_fn=fake_chat,
                chat_model="qwen2.5:3b",
                seed=0,
                scenario="planting",
                conditions=("attack",),
                policies=(ep.METHOD_MAPLE_GUARD,),
                out_path=path,
                max_output_tokens=64,
                chat_timeout=90,
            )
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
            saved["items"][0]["policies"][ep.METHOD_MAPLE_GUARD]["resume_marker"] = "keep"
            poison.atomic_write_json(path, saved)
            with self.assertRaises(ValueError) as raised:
                poison.run_code_poison(
                    chat_fn=fake_chat,
                    chat_model="qwen2.5:3b",
                    seed=0,
                    scenario="planting",
                    conditions=("attack",),
                    policies=(ep.METHOD_MAPLE_GUARD,),
                    out_path=path,
                    resume=False,
                    max_output_tokens=32,
                    chat_timeout=90,
                )
            self.assertIn("max_output_tokens", str(raised.exception))
            self.assertIn("not overwritten", str(raised.exception))
            with open(path, encoding="utf-8") as handle:
                current = json.load(handle)
            self.assertEqual(current["items"][0]["policies"][ep.METHOD_MAPLE_GUARD]["resume_marker"], "keep")
            self.assertEqual(current["items"][0]["policies"][ep.METHOD_MAPLE_GUARD]["max_output_tokens"], 64)
