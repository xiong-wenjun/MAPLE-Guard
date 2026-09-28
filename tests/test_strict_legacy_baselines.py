"""Matched-runtime routing and explicit failure semantics for legacy guards."""
import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from maple_guard import maple_guard_core as ep, run_mmlu
from evaluate.defense_methods import challenger_defense as challenger
from evaluate.defense_methods import gnn_defense as gnn
from evaluate.defense_methods import guardian_defense as guardian
from evaluate.defense_methods import inspector_defense as inspector
from evaluate.defense_methods.base import DefenseContext, OfficialDefenseState
from evaluate.defense_methods.full_runtime import runtime_scope, experiment_identity

COMM = ("challenger", "gsafeguard", "guardian", "inspector")
ABLATIONS = ("maple_guard_no_write", "maple_guard_no_retrieval",
             "maple_guard_no_promotion", "maple_guard_no_cross_agent")


class StrictLegacyBaselines(unittest.TestCase):
    def args(self, method="challenger", strict=True):
        argv = ["runner", "--config", "", "--method", method, "--agents", "2",
                "--rounds", "2", "--peer-communication"]
        if strict:
            argv.append("--strict-comparison")
        with patch.object(sys, "argv", argv):
            args = run_mmlu.parse_args()
        args.memory_backend = "memory"
        args.attacker_ids = [1]
        args.enable_round_memory_propagation = False
        args.enable_attacker_system_prompt = False
        args._disable_memory_update = True
        return args

    def ctx(self, method="challenger", strict=True):
        return DefenseContext(method, "q", "Which fact?", 0, [[0, 1], [1, 0]],
                              self.args(method, strict))

    def dispatch(self, args, state=None, task_id="q", outputs=None):
        return ep.apply_official_communication_defense_to_outputs(
            args.method, {0: "safe text", 1: "bad text"} if outputs is None else outputs,
            state, task_id=task_id, question="Which fact?", round_idx=0,
            adj_matrix=[[0, 1], [1, 0]], args=args)

    def test_strict_whitelist_includes_main_guards_and_gate_ablations(self):
        for method in COMM + ABLATIONS:
            with self.subTest(method=method):
                try:
                    with runtime_scope(self.args(method)) as runtime:
                        self.assertEqual(runtime.method, method)
                except ValueError as exc:
                    self.fail(str(exc))

    def test_challenger_public_core_rewrites_and_routes_guarded_messages(self):
        args = self.args()
        task = ep.TaskExample("q", "Which fact?", [("A", "yes"), ("B", "no")], "A", "B", raw={})
        prompts = []
        def model(base, name, messages, **kwargs):
            prompts.append(messages)
            return "Final answer: B"
        with patch.object(ep, "remote_embedding", return_value=[1., 0.]), \
             patch.object(ep, "call_chat", side_effect=model), \
             patch.object(challenger, "chat_completion", return_value="unsafe."):
            trace, _, _, _ = ep.run_task("test", task, args, {}, [], False)
        replacement = "I need to verify this information before responding."
        self.assertEqual(trace.outputs_by_round[0], {0: replacement, 1: replacement})
        self.assertIn(replacement, str(prompts[2]))
        self.assertEqual(trace.final_answer_excluded_agent_ids, [])
        self.assertTrue(any(d["reason"] == "challenger_unsafe_replaced" for d in trace.defense_decisions))

    def test_challenger_state_lifetime_is_controlled_by_caller(self):
        args = self.args()
        with patch.object(challenger, "chat_completion", return_value="safe"), runtime_scope(args) as runtime:
            runtime.begin_task("first", "q")
            _, first, _ = self.dispatch(args)
            runtime.begin_task("second", "q")
            _, carried, _ = self.dispatch(args, first, "second")
            self.assertIs(carried, first)
            self.assertEqual(len(carried.challenger_memory[0]), 2)
            _, fresh, _ = self.dispatch(args, None, "fresh")
            self.assertEqual(len(fresh.challenger_memory[0]), 1)
            self.assertIsNot(first, fresh)

    def test_strict_challenger_judge_failure_is_not_safe(self):
        with patch.object(challenger, "chat_completion", side_effect=TimeoutError("judge timeout")):
            with self.assertRaisesRegex(RuntimeError, "Challenger.*judge"):
                challenger.apply_challenger({0: "text"}, OfficialDefenseState(), self.ctx())

    def test_legacy_challenger_keeps_existing_failure_policy(self):
        with patch.object(challenger, "chat_completion", side_effect=TimeoutError("judge timeout")):
            outputs, _, decisions = challenger.apply_challenger({0: "text"}, OfficialDefenseState(),
                                                                self.ctx(strict=False))
        self.assertEqual(outputs, {0: "text"})
        self.assertEqual(decisions[0]["reason"], "challenger_safe")

    def test_strict_challenger_rejects_unparseable_judge_reply(self):
        with patch.object(challenger, "chat_completion", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "Challenger.*judge"):
                challenger.apply_challenger({0: "text"}, OfficialDefenseState(), self.ctx())

    def test_inspector_public_core_preserves_rewrites_and_message_history(self):
        args = self.args("inspector")
        task = ep.TaskExample("q", "Which fact?", [("A", "yes"), ("B", "no")], "A", "B", raw={})
        generation_prompts, inspection_prompts = [], []
        def model(base, name, messages, **kwargs):
            generation_prompts.append(messages)
            return "Final answer: B"
        def inspect(args, messages, **kwargs):
            inspection_prompts.append(messages[0]["content"])
            return '{"decision":"unsafe","revision":"Final answer: A"}'
        with patch.object(ep, "remote_embedding", return_value=[1., 0.]), \
             patch.object(ep, "call_chat", side_effect=model), \
             patch.object(inspector, "chat_completion", side_effect=inspect):
            trace, _, _, _ = ep.run_task("test", task, args, {}, [], False)
        self.assertEqual(trace.outputs_by_round[0], {0: "Final answer: A", 1: "Final answer: A"})
        self.assertIn("Final answer: A", str(generation_prompts[2]))
        self.assertIn("Agent_0: Final answer: A", inspection_prompts[1])
        self.assertTrue(any(d["reason"] == "inspector_unsafe_rewritten" for d in trace.defense_decisions))

    def test_inspector_state_lifetime_is_controlled_by_caller(self):
        args = self.args("inspector")
        with patch.object(inspector, "chat_completion", return_value='{"decision":"safe"}'), runtime_scope(args) as runtime:
            runtime.begin_task("first", "q")
            _, first, _ = self.dispatch(args)
            runtime.begin_task("second", "q")
            _, carried, _ = self.dispatch(args, first, "second")
            self.assertIs(carried, first)
            self.assertEqual(len(carried.inspector_memory), 4)
            _, fresh, _ = self.dispatch(args, None, "fresh")
            self.assertEqual(len(fresh.inspector_memory), 2)
            self.assertIsNot(first, fresh)

    def test_strict_inspector_failed_or_invalid_judgments_raise(self):
        replies = (TimeoutError("judge timeout"), "", "[]", "{}",
                   '{"decision":"unknown","revision":"safe"}',
                   '{"decision":"unsafe"}', '{"decision":"unsafe","revision":null}',
                   '{"decision":"unsafe","revision":""}',
                   '{"decision":"unsafe","revision":"<safe content>"}')
        for reply in replies:
            with self.subTest(reply=reply):
                replacement = {"side_effect": reply} if isinstance(reply, Exception) else {"return_value": reply}
                with patch.object(inspector, "chat_completion", **replacement):
                    with self.assertRaisesRegex(RuntimeError, "Inspector.*judge"):
                        inspector.apply_inspector({0: "text"}, OfficialDefenseState(), self.ctx("inspector"))

    def test_legacy_inspector_keeps_existing_failure_policy(self):
        with patch.object(inspector, "chat_completion", side_effect=TimeoutError("judge timeout")):
            outputs, _, decisions = inspector.apply_inspector({0: "text"}, OfficialDefenseState(),
                                                               self.ctx("inspector", strict=False))
        self.assertEqual(outputs, {0: "text"})
        self.assertEqual(decisions[0]["reason"], "inspector_safe")

    def test_inspector_uses_safeguard_environment_auth(self):
        from evaluate.defense_methods import llm_client
        args = self.args("inspector")
        args.safeguard_api_key = ""
        class Response:
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": '{"decision":"safe"}'}}]}
        with patch.dict("os.environ", {"SAFEGUARD_BASE_URL": "http://judge.test/v1",
                                      "SAFEGUARD_MODEL": "qwen-judge",
                                      "SAFEGUARD_OPENAI_API_KEY": "test-only-key"}, clear=True), \
             patch.object(llm_client.requests, "post", return_value=Response()) as post:
            verdict, text = inspector.inspect_response("Agent_0", "original", [], args)
        self.assertEqual((verdict, text), ("safe", "original"))
        self.assertEqual(post.call_args.args[0], "http://judge.test/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-only-key")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "qwen-judge")
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0)

    def test_inspector_provenance_names_actual_judge_and_adaptation(self):
        from evaluate.defense_methods.full_runtime import baseline_run_provenance
        args = self.args("inspector")
        args.safeguard_model = "qwen-judge"
        args.chat_model = "gemma-backbone"
        with runtime_scope(args):
            report = baseline_run_provenance(args)
        self.assertEqual(report["judge_model"], "qwen-judge")
        self.assertFalse(report["provenance"]["official_reproduction_claim"])

    def test_strict_gnn_missing_checkpoint_is_error(self):
        ctx = self.ctx("gsafeguard")
        ctx.args.official_defense_gnn_checkpoint = ""
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            gnn.apply_gnn_official({0: "text"}, OfficialDefenseState(), ctx, "gsafeguard")

    def test_strict_gnn_dependency_failure_is_error(self):
        ctx = self.ctx("gsafeguard")
        ctx.args.official_defense_gnn_checkpoint = "test.pth"
        with patch.object(gnn, "_load_runtime", return_value=(None, "missing dependency")):
            with self.assertRaisesRegex(RuntimeError, "dependency"):
                gnn.apply_gnn_official({0: "text"}, OfficialDefenseState(), ctx, "gsafeguard")

    def test_legacy_gnn_keeps_missing_checkpoint_fallback(self):
        ctx = self.ctx("gsafeguard", strict=False)
        ctx.args.official_defense_gnn_checkpoint = ""
        outputs, _, decisions = gnn.apply_gnn_official({0: "text"}, OfficialDefenseState(), ctx, "gsafeguard")
        self.assertEqual(outputs, {0: "text"})
        self.assertIn("checkpoint_missing", decisions[0]["reason"])

    def test_strict_gnn_model_failure_and_invalid_scores_raise(self):
        import torch
        for failure in (RuntimeError("broken model"), torch.tensor([float("nan"), 0.]), torch.tensor([0.])):
            with self.subTest(failure=str(failure)):
                def predict(*_):
                    if isinstance(failure, Exception):
                        raise failure
                    return failure
                ctx = self.ctx("gsafeguard")
                ctx.args.official_defense_gnn_checkpoint = "test.pth"
                runtime = {"torch": torch, "model": predict}
                with patch.object(gnn, "_load_runtime", return_value=(runtime, None)), \
                     patch.object(gnn, "_build_graph_tensors", return_value=(object(), None, None)):
                    with self.assertRaises(RuntimeError):
                        gnn.apply_gnn_official({0: "text", 1: "text"}, OfficialDefenseState(), ctx, "gsafeguard")

    def test_strict_gnn_routes_native_pruning(self):
        import torch
        args = self.args("gsafeguard")
        args.official_defense_gnn_checkpoint = "test.pth"
        backend = {"torch": torch, "model": lambda *_: torch.tensor([-4., 4.])}
        with patch.object(gnn, "_load_runtime", return_value=(backend, None)), \
             patch.object(gnn, "_build_graph_tensors", return_value=(object(), None, None)), \
             runtime_scope(args) as runtime:
            runtime.begin_task("q", "Which fact?")
            outputs, state, decisions = self.dispatch(args)
        self.assertEqual(outputs, {0: "safe text"})
        self.assertIsInstance(state, OfficialDefenseState)
        self.assertTrue(any(d.reason == "gsafeguard_gnn_pruned_high_risk_agent" for d in decisions))

    def test_strict_guardian_dependencies_and_missing_source_raise(self):
        for available, source in ((False, "source"), (True, "")):
            with self.subTest(available=available), \
                 patch.object(guardian, "_guardian_runtime_available", return_value=(available, "dependency absent")), \
                 patch.object(guardian, "_guardian_code_dir", return_value=source):
                with self.assertRaises(RuntimeError):
                    guardian.apply_guardian({0: "text"}, OfficialDefenseState(), self.ctx("guardian"))

    def test_strict_guardian_load_and_inference_errors_raise(self):
        for load, inference in ((None, None), ({}, RuntimeError("model failure"))):
            with self.subTest(load=load), \
                 patch.object(guardian, "_guardian_runtime_available", return_value=(True, "")), \
                 patch.object(guardian, "_guardian_code_dir", return_value="source"), \
                 patch.object(guardian, "_load_official_runtime", return_value=(load, "load failure")), \
                 patch.object(guardian, "_build_graph_data", return_value=[object()]), \
                 patch.object(guardian, "_run_official_guardian", side_effect=inference):
                with self.assertRaises(RuntimeError):
                    guardian.apply_guardian({0: "text", 1: "text"}, OfficialDefenseState(), self.ctx("guardian"))

    def test_strict_guardian_native_removal_persists_in_caller_state(self):
        args = self.args("guardian")
        with patch.object(guardian, "_guardian_runtime_available", return_value=(True, "")), \
             patch.object(guardian, "_guardian_code_dir", return_value="source"), \
             patch.object(guardian, "_load_official_runtime", return_value=({}, None)), \
             patch.object(guardian, "_build_graph_data", return_value=[object()]), \
             patch.object(guardian, "_run_official_guardian", return_value=(1, [.1, .9], "test_fixture")), \
             runtime_scope(args) as runtime:
            runtime.begin_task("q", "Which fact?")
            outputs, state, _ = self.dispatch(args)
            self.assertEqual(outputs, {0: "safe text"})
            outputs, state, decisions = self.dispatch(args, state)
        self.assertEqual(outputs, {0: "safe text"})
        self.assertEqual(state.guardian_inactive_agents, {1})
        self.assertTrue(any(d.reason == "guardian_official_inactive_agent_pruned" for d in decisions))

    def test_legacy_guardian_keeps_dependency_fallback(self):
        with patch.object(guardian, "_guardian_runtime_available", return_value=(False, "dependency absent")):
            outputs, _, decisions = guardian.apply_guardian({0: "text"}, OfficialDefenseState(),
                                                             self.ctx("guardian", strict=False))
        self.assertEqual(outputs, {0: "text"})
        self.assertIn("dependencies_missing", decisions[0]["reason"])

    def test_strict_gate_ablation_keeps_native_gate_bypass(self):
        task = ep.TaskExample("q", "fact", [("A", "yes"), ("B", "no")], "A", "B", raw={})
        for method, expected in (("maple_guard", False), ("maple_guard_no_write", True)):
            with self.subTest(method=method), runtime_scope(self.args(method)) as runtime:
                runtime.begin_task("q", "fact")
                memory = ep.create_benign_memory(task, 0, "A", True, "test")
                memory.experience = "password secret"
                accepted, decisions = ep.commit_memory(memory, "private", 0, method, {}, [],
                                                       ingress_channel="agent_output")
                self.assertEqual(accepted, expected)
                reason = "ablation_no_write_firewall" if expected else "forbidden_sensitive_memory"
                self.assertTrue(any(d.reason == reason for d in decisions))

    def test_identity_tracks_legacy_guard_checkpoint_and_judge(self):
        args = self.args("gsafeguard")
        original = experiment_identity(args)
        args.official_defense_gnn_checkpoint = "/another/checkpoint.pth"
        self.assertNotEqual(original, experiment_identity(args))
        original = experiment_identity(args)
        args.safeguard_model = "different-judge"
        self.assertNotEqual(original, experiment_identity(args))

    def test_strict_gnn_invalid_threshold_is_not_default_or_allow_all(self):
        for value in ("not-a-number", float("nan"), -0.1, 1.1):
            with self.subTest(value=value):
                ctx = self.ctx("gsafeguard")
                ctx.args.official_defense_gnn_threshold = value
                with self.assertRaises(ValueError):
                    gnn._threshold(ctx)

    def test_strict_guardian_rejects_invalid_detector_outputs(self):
        for result in ((-1, [.1, .9], "test"), (0, [float("nan"), .9], "test"),
                       (0, [.1], "test")):
            with self.subTest(result=result), \
                 patch.object(guardian, "_guardian_runtime_available", return_value=(True, "")), \
                 patch.object(guardian, "_guardian_code_dir", return_value="source"), \
                 patch.object(guardian, "_load_official_runtime", return_value=({}, None)), \
                 patch.object(guardian, "_build_graph_data", return_value=[object()]), \
                 patch.object(guardian, "_run_official_guardian", return_value=result):
                with self.assertRaises(RuntimeError):
                    guardian.apply_guardian({0: "text", 1: "text"}, OfficialDefenseState(), self.ctx("guardian"))

    def test_other_gate_ablations_keep_original_phase_decisions(self):
        task = ep.TaskExample("q", "fact", [("A", "yes"), ("B", "no")], "A", "B", raw={})
        memory = ep.create_benign_memory(task, 0, "A", True, "test")
        parts = {"scope_violation": 0, "content_hazard": .9, "lifecycle_risk": .9}
        memory.provenance_trust = .1
        self.assertTrue(ep.read_firewall(memory, "maple_guard", 0, "q", 1., parts, -.5)[0])
        with runtime_scope(self.args("maple_guard_no_retrieval")):
            self.assertFalse(ep.read_firewall(memory, "maple_guard_no_retrieval", 0, "q", 1., parts, -.5)[0])
        memory.content_hazard = .9
        self.assertEqual(ep.promotion_gate(memory, "maple_guard", "team").action, ep.ACTION_QUARANTINE)
        with runtime_scope(self.args("maple_guard_no_promotion")):
            self.assertEqual(ep.promotion_gate(memory, "maple_guard_no_promotion", "team").action, ep.ACTION_ALLOW)
        self.assertEqual(ep.broker_decision(memory, 0, "maple_guard", ep.MEM_BROKERED_SHARED, "q").action, ep.ACTION_BLOCK)
        with runtime_scope(self.args("maple_guard_no_cross_agent")):
            self.assertEqual(ep.broker_decision(memory, 0, "maple_guard_no_cross_agent", ep.MEM_BROKERED_SHARED, "q").action, ep.ACTION_ALLOW)

    def test_legacy_provenance_names_adaptation_and_actual_judge(self):
        from evaluate.defense_methods.full_runtime import baseline_run_provenance
        args = self.args("challenger")
        args.safeguard_model = "qwen-judge"
        args.chat_model = "gemma-backbone"
        with runtime_scope(args):
            report = baseline_run_provenance(args)
        self.assertEqual(report["judge_model"], "qwen-judge")
        self.assertEqual(report["provenance"]["profile"], "existing_communication_adapter_with_matched_memory")


if __name__ == "__main__":
    unittest.main()
