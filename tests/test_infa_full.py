import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest

MODULE = Path(__file__).resolve().parents[1] / "evaluate/defense_methods/infa_full.py"

class Detector:
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []
        self.provenance = {"detector": "deterministic_test_double"}
    def __call__(self, history, adjacency):
        self.calls.append((copy.deepcopy(history), copy.deepcopy(adjacency)))
        return self.rounds[len(self.calls)-1]

class Shape:
    def __init__(self, *shape): self.shape = shape

class InfaFullTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), "Native full INFA protocol is not implemented")
        spec = importlib.util.spec_from_file_location("infa_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        spec.loader.exec_module(self.module)

    def args(self, n=3, **changes):
        values = dict(agents=n, infa_threshold=0.5, infa_donor_threshold=0.1,
                      infa_code_dir="", infa_checkpoint="", infa_embedding_model="MiniLM",
                      infa_device="cpu", infa_protocol="reconstruction")
        values.update(changes)
        return SimpleNamespace(**values)

    def make(self, probabilities, n=3, protocol="reconstruction"):
        detector = Detector(probabilities)
        self.judge_calls, self.replacements, self.repairs = [], [], []
        def judge(messages):
            self.judge_calls.append(messages)
            return json.dumps({"response": "corrected answer"})
        guard = self.module.InfaGuardFull(self.args(n, infa_protocol=protocol), judge, detector=detector)
        guard.begin_task("task")
        return guard, detector

    def defend(self, guard, outputs, idx=0, adjacency=None):
        adjacency = adjacency if adjacency is not None else [[int(i != j) for j in range(guard.num_agents)] for i in range(guard.num_agents)]
        return guard.defend(outputs, lambda *a, **k: self.fail("INFA must not impersonate a target inquiry"),
                            "Current question", idx, adjacency,
                            replace=lambda *x: self.replacements.append(x),
                            repair=lambda *x: self.repairs.append(x))

    def test_released_is_default_and_first_turn_allows_infected_high_probability_donor(self):
        guard, _ = self.make([[[0.9,0.0],[0.2,0.8],[0.3,0.0]]], protocol="released")
        args = self.args()
        del args.infa_protocol
        default = self.module.InfaGuardFull(args, lambda _: "", detector=Detector([]))
        self.assertEqual(getattr(default, "protocol", None), "released")
        result, decisions = self.defend(guard, {0:"attacker",1:"infected donor",2:"other"})
        self.assertEqual(result, {0:"infected donor",1:"infected donor",2:"other"})
        self.assertEqual(self.replacements, [(0,1,"infected donor")])
        self.assertEqual(self.judge_calls, [])
        self.assertEqual(self.repairs, [])
        self.assertEqual(guard.p_inf_ema, [0.0,0.0,0.0])
        self.assertEqual(guard.blocked_senders, {0})
        self.assertEqual(guard.inactive, set())
        self.assertEqual(guard.takeover_context, "donor_system_only")
        self.assertFalse(decisions[0]["details"]["ema_updated"])
        self.assertEqual(decisions[1]["details"]["raw_infected"], [1])
        self.assertEqual(decisions[1]["details"]["infected"], [])

    def test_released_later_cutoff_retains_flagged_agents_without_changing_detection_graph(self):
        probabilities = [[[0.01,0.0]]*3, [[0.9,0.0],[0.001,0.8],[0.2,0.0]], [[0.01,0.0]]*3]
        guard, detector = self.make(probabilities, protocol="released")
        guard.judge = lambda _: "raw corrected answer"
        first = {0:"initial0",1:"initial1",2:"initial2"}
        self.defend(guard, first)
        result, decisions = self.defend(guard, {0:"raw attack",1:"raw infection",2:"uncertain"}, 1)
        self.assertEqual(result, {0:"raw attack",1:"raw corrected answer",2:"uncertain"})
        self.assertEqual(self.replacements, [])
        self.assertEqual(self.repairs, [])
        self.assertEqual(guard.inactive, set())
        self.assertEqual(guard.blocked_senders, {0,1})
        self.assertAlmostEqual(guard.p_inf_ema[1], 0.24)
        self.assertEqual(decisions[0]["reason"], "infa_no_benign_donor_retained")
        self.defend(guard, {0:"next0",1:"next1",2:"next2"}, 2)
        self.assertEqual(detector.calls[-1][1], [[0,1,1],[1,0,1],[1,1,0]])
        self.assertEqual(guard.blocked_senders, set())
        self.assertEqual(detector.calls[-1][0][1], ["raw attack","raw infection","uncertain"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "native numeric dependency")
    def test_released_float32_cutoff_and_ema_match_source_numpy_semantics(self):
        import numpy as np
        guard, _ = self.make([[[0.01,0.0]]*2,
                              [[0.9,0.0],[float(np.float32(0.1)),0.0]]], n=2, protocol="released")
        self.defend(guard, {0:"initial0",1:"initial1"})
        result, _ = self.defend(guard, {0:"attack",1:"threshold donor"}, 1)
        self.assertEqual(result[0], "threshold donor")
        guard, _ = self.make([[[0.01,0.0]], [[0.01,0.8]]], n=1, protocol="released")
        guard.judge = lambda _: "fixed"
        self.defend(guard, {0:"initial"})
        self.defend(guard, {0:"infection"}, 1)
        expected = 0.3 * np.array([0.8], dtype=np.float32) + 0.7 * np.zeros(1, dtype=np.float32)
        self.assertEqual(guard.p_inf_ema, expected.tolist())

    def test_released_all_flagged_agents_continue_and_task_reset_clears_sender_filter(self):
        guard, detector = self.make([[[0.9,0.0]]*2]*2, n=2, protocol="released")
        original = {0:"attack0",1:"attack1"}
        for idx in (0,1):
            result, _ = self.defend(guard, original, idx)
            self.assertEqual(result, original)
            self.assertEqual(guard.inactive, set())
            self.assertEqual(guard.blocked_senders, {0,1})
        self.assertEqual(len(detector.calls), 2)
        guard.begin_task("next task")
        self.assertEqual(guard.blocked_senders, set())
        self.assertEqual(guard.p_inf_ema, [0.0,0.0])

    def test_released_correction_uses_text_contract_and_source_redaction_fallback(self):
        guard, _ = self.make([[[0.01,0.0]], [[0.01,0.8]], [[0.01,0.8]]], n=1, protocol="released")
        calls = []
        answer = "<REASON>: repaired reasoning\n<ANSWER>: A"
        def judge(messages):
            calls.append(messages)
            if len(calls) == 2:
                raise OSError("test provider unavailable")
            return answer
        guard.judge = judge
        self.defend(guard, {0:"first"})
        result, decisions = self.defend(guard, {0:"unsafe original"}, 1)
        self.assertEqual(result[0], answer)
        self.assertEqual(self.repairs, [])
        self.assertIn("Return only the rewritten text.", calls[0][0]["content"])
        self.assertIn("Communication history: No previous communication history.", calls[0][1]["content"])
        self.assertFalse(decisions[0]["details"]["correction_fallback"])
        result, decisions = self.defend(guard, {0:"password Password PASSWORD key Attack prompt injection backdoor collude"}, 2)
        self.assertEqual(result[0], "[REDACTED] [REDACTED] PASSWORD [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED]")
        self.assertTrue(decisions[0]["details"]["correction_fallback"])
        self.assertFalse(decisions[0]["details"]["context_repaired"])

    def test_released_sync_bug_profile_and_transport_fix_are_explicit(self):
        for profile, expected in (("released_sync_bug","[REDACTED] unsafe"),("functional","rewritten")):
            calls=[]
            guard=self.module.InfaGuardFull(
                self.args(1,infa_protocol="released",infa_correction_transport=profile),
                lambda messages:calls.append(messages) or "rewritten",
                detector=Detector([]))
            actual,fallback=guard._correct(0,"password unsafe","task",[[0]])
            self.assertEqual(actual,expected)
            self.assertEqual(fallback,profile=="released_sync_bug")
            self.assertEqual(len(calls),1)
            self.assertEqual(guard.provenance["sanitizer_sync_await_fixed"],profile=="functional")
        with self.assertRaises(self.module.InfaConfigError):
            self.module.InfaGuardFull(self.args(infa_correction_transport="unknown"),lambda m:"x",detector=Detector([]))

    def test_replaces_attackers_and_rehabilitates_infected_without_pruning_repaired_nodes(self):
        guard, detector = self.make([[[0.9,0.0],[0.05,0.8],[0.02,0.0]]])
        result, decisions = self.defend(guard, {0:"attack",1:"infected",2:"donor"})
        self.assertEqual(result, {0:"donor",1:"corrected answer",2:"donor"})
        self.assertEqual(self.replacements, [(0,2,"donor")])
        self.assertEqual(self.repairs, [(1,"corrected answer")])
        self.assertEqual(guard.inactive, set())
        self.assertEqual(detector.calls[0][0], [["attack","infected","donor"]])
        self.assertEqual(decisions[0]["details"]["p_mal"], 0.9)
        self.assertEqual(decisions[1]["details"]["p_inf"], 0.8)
        self.assertIn("Current question", str(self.judge_calls))

    def test_temporal_features_use_raw_outputs_and_repair_history_is_separate(self):
        probs = [[[0.9,0.0],[0.05,0.8],[0.02,0.0]], [[0.02,0.0],[0.05,0.8],[0.02,0.0]]]
        guard, detector = self.make(probs)
        self.defend(guard, {0:"raw attack",1:"raw infection",2:"donor"})
        self.defend(guard, {0:"new raw0",1:"new raw1",2:"new raw2"}, 1)
        self.assertEqual(detector.calls[-1][0][0], ["raw attack","raw infection","donor"])
        context = json.loads(self.judge_calls[-1][1]["content"])
        self.assertIn("donor", str(context["neighbor_history"]))
        self.assertNotIn("raw attack", str(context["neighbor_history"]))

    def test_missing_benign_donor_isolates_attacker_and_removes_incident_edges(self):
        guard, detector = self.make([[[0.9,0.0],[0.2,0.0]], [[0.0,0.0],[0.01,0.0]]], n=2)
        result, decisions = self.defend(guard, {0:"attack",1:"no eligible donor"})
        self.assertEqual(result, {1:"no eligible donor"})
        self.assertEqual(guard.inactive, {0})
        self.assertEqual(decisions[0]["action"], "block")
        self.assertIn("donor", decisions[0]["reason"])
        self.defend(guard, {1:"next"}, 1)
        self.assertEqual(detector.calls[-1][1], [[0,0],[0,0]])
        self.assertEqual(detector.calls[-1][0][-1], ["","next"])

    def test_infected_node_cannot_donate_even_with_low_attacker_probability(self):
        guard, _ = self.make([[[0.9,0.0],[0.001,0.8],[0.2,0.0]]])
        result, _ = self.defend(guard, {0:"attack",1:"infection",2:"uncertain"})
        self.assertNotIn(0, result)
        self.assertEqual(self.replacements, [])

    def test_refinement_retains_trending_isolated_infection_and_infers_source(self):
        guard, _ = self.make([[[0.2,0.9],[0.4,0.0],[0.01,0.0]]])
        result, decisions = self.defend(guard, {0:"infection",1:"source",2:"donor"},
                                      adjacency=[[0,1,0],[1,0,1],[0,1,0]])
        self.assertEqual(self.replacements, [(1,2,"donor")])
        self.assertEqual(self.repairs, [(0,"corrected answer")])
        self.assertEqual(decisions[0]["details"]["attackers"], [1])
        self.assertAlmostEqual(decisions[0]["details"]["infection_ema_delta"], 0.27)

    def test_false_positive_pruning_when_isolated_and_ema_trend_is_small(self):
        guard, _ = self.make([[[0.01,0.6],[0.01,0.0],[0.01,0.0]]])
        guard.p_inf_ema = [0.6,0.0,0.0]
        result, decisions = self.defend(guard, {0:"candidate",1:"safe",2:"safe"},
                                      adjacency=[[0,0,0],[0,0,0],[0,0,0]])
        self.assertEqual(result[0], "candidate")
        self.assertEqual(decisions[0]["details"]["infected"], [])
        self.assertEqual(self.repairs, [])

    def test_begin_task_resets_temporal_and_isolation_state(self):
        guard, _ = self.make([[[0.9,0.0],[0.2,0.0]]], n=2)
        self.defend(guard, {0:"attack",1:"uncertain"})
        guard.begin_task("new")
        self.assertEqual(guard.history, [])
        self.assertEqual(guard.repaired_history, [])
        self.assertEqual(guard.p_inf_ema, [0.0,0.0])
        self.assertEqual(guard.inactive, set())

    def test_malformed_detector_and_judge_fail_without_silent_allow(self):
        for bad in ([[0.1]], [[float("nan"),0.0]]*3, [[0.1,1.1]]*3):
            guard, _ = self.make([bad])
            with self.subTest(bad=bad), self.assertRaises(self.module.InfaRuntimeError):
                self.defend(guard, {0:"a",1:"b",2:"c"})
        for bad in ('not json', '{"response":""}', '{"response":"a","response":"b"}',
                    '{"response":"ok","other":true}'):
            guard, _ = self.make([[[0.9,0.0],[0.05,0.8],[0.02,0.0]]])
            guard.judge = lambda messages, bad=bad: bad
            with self.subTest(judge=bad), self.assertRaises(self.module.InfaRuntimeError):
                self.defend(guard, {0:"a",1:"b",2:"c"})

    def test_checkpoint_validation_rejects_gsafeguard_and_missing_infection_heads(self):
        bad = {"model_class":"GSafeguardGAT", "model_state_dict":{"layers.0.bias":Shape(1024), "out.weight":Shape(1,1024)}}
        with self.assertRaisesRegex(self.module.InfaCheckpointError, "G-Safeguard|single"):
            self.module.validate_checkpoint_payload(bad)
        state = self.native_signature()
        del state["branch_heads_inf.2.weight"]
        with self.assertRaisesRegex(self.module.InfaCheckpointError, "branch_heads_inf.2"):
            self.module.validate_checkpoint_payload(state)

    def native_signature(self):
        state = {"input_proj.weight":Shape(384,2304), "shared_convs.0.lin.weight":Shape(1024,384)}
        for branch in range(4):
            for head in ("mal","inf"):
                state[f"branch_heads_{head}.{branch}.weight"] = Shape(1,1024)
        return state

    def test_raw_and_wrapped_native_checkpoint_signatures_are_accepted(self):
        state = self.native_signature()
        self.assertIs(self.module.validate_checkpoint_payload(state), state)
        self.assertIs(self.module.validate_checkpoint_payload({"model_state_dict":state}), state)

    def test_native_import_does_not_replace_existing_train_namespace(self):
        sentinel = sys.modules.get("train", ModuleType("train"))
        previous = sys.modules.get("train")
        sys.modules["train"] = sentinel
        try:
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)/"train/models/defender"
                directory.mkdir(parents=True)
                (directory/"gat_with_attr_conv.py").write_text("class GATwithEdgeConv: marker = 7\n")
                (directory/"model.py").write_text("from train.models.defender.gat_with_attr_conv import GATwithEdgeConv\nclass MyGAT: marker = GATwithEdgeConv.marker\n")
                cls, provenance = self.module.load_native_class(temp)
                self.assertEqual(cls.marker, 7)
                self.assertEqual(len(provenance["source_sha256"]["model.py"]), 64)
                self.assertIs(sys.modules["train"], sentinel)
        finally:
            if previous is None: sys.modules.pop("train", None)
            else: sys.modules["train"] = previous

    def test_invalid_startup_configuration_fails_explicitly(self):
        for changes in ({"infa_threshold":float("nan")}, {"infa_donor_threshold":-1}, {"agents":0}, {"infa_protocol":"unknown"}, {"infa_detector_mode":"unknown"}):
            with self.subTest(changes=changes), self.assertRaises(self.module.InfaConfigError):
                self.module.InfaGuardFull(self.args(**changes), lambda _: "", detector=Detector([]))
        with self.assertRaises(self.module.InfaConfigError):
            self.module.InfaGuardFull(self.args(), lambda _: "")


    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional native tensor dependency")
    def test_graph_features_match_published_runtime_including_isolated_nodes(self):
        import torch
        embeddings = [[[1.,10.],[2.,20.],[3.,30.]], [[4.,40.],[5.,50.],[6.,60.]]]
        adj = [[0,1,0],[0,0,0],[1,0,0]]
        x, edges, attr, own = self.module.build_graph_tensors(torch, embeddings, adj, "cpu")
        self.assertEqual(edges.tolist(), [[0,2],[1,0]])
        self.assertEqual(x.tolist(), [[1.,10.],[2.,20.],[0.,0.]])
        self.assertEqual(attr.tolist(), [[[2.,20.],[5.,50.]], [[1.,10.],[4.,40.]]])
        self.assertEqual(own[2].tolist(), [[3.,30.],[6.,60.]])
        x, edges, attr, own = self.module.build_graph_tensors(torch, embeddings, [[0]*3 for _ in range(3)], "cpu")
        self.assertEqual(tuple(edges.shape), (2,0))
        self.assertEqual(tuple(attr.shape), (0,2,2))
        self.assertEqual(x.tolist(), [[0.,0.]]*3)


    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                             ("torch", "torch_geometric", "torch_scatter", "einops")),
                         "optional native source dependencies")
    def test_native_fixture_strict_load_and_dual_head_forward_without_downloads(self):
        import torch
        root = Path(os.environ.get("INFA_NATIVE_SOURCE", "/mnt/public/data/wj/baseline-references/INFA-Guard"))
        if not (root/"train/models/defender/model.py").exists():
            self.skipTest("external official source not present")
        model_class, provenance = self.module.load_native_class(root)
        torch.manual_seed(17)
        fixture = model_class(in_channels=384, hidden_channels=1024, out_channels=2,
                              heads=8, edge_dim=(3,384), guard="ours")
        class FixtureEmbedder:
            def encode(self, texts, **kwargs):
                return torch.arange(len(texts)*384, dtype=torch.float32).reshape(len(texts),384) / 1000
        with tempfile.TemporaryDirectory(prefix="infa_software_fixture_") as temp:
            checkpoint = Path(temp)/"synthetic_native_fixture.pth"
            torch.save({"model_state_dict":fixture.state_dict()}, checkpoint)
            try:
                detector = self.module.NativeInfaDetector(self.args(infa_code_dir=str(root),
                           infa_checkpoint=str(checkpoint)), embedder=FixtureEmbedder())
            except TypeError as exc:
                self.fail("Native detector needs an injectable embedding dependency: " + str(exc))
            self.assertFalse(detector.model.training)
            self.assertEqual(detector.provenance["detector_mode"], "eval")
            released = self.module.NativeInfaDetector(self.args(infa_protocol="released", infa_code_dir=str(root),
                       infa_checkpoint=str(checkpoint)), embedder=FixtureEmbedder())
            self.assertTrue(released.model.training)
            self.assertEqual(released.provenance["detector_mode"], "train")
            self.assertFalse(released.provenance["detector_mode_overrides_release"])
            deterministic = self.module.NativeInfaDetector(self.args(infa_protocol="released", infa_detector_mode="eval",
                       infa_code_dir=str(root), infa_checkpoint=str(checkpoint)), embedder=FixtureEmbedder())
            self.assertFalse(deterministic.model.training)
            self.assertTrue(deterministic.provenance["detector_mode_overrides_release"])
            result = detector([["a","b","c"],["d","e","f"]], [[0,1,0],[0,0,1],[1,0,0]])
            self.assertEqual(tuple(torch.tensor(result).shape), (3,2))
            self.assertTrue(torch.isfinite(torch.tensor(result)).all())
            self.assertEqual(detector.provenance["source_commit"], provenance["source_commit"])
            self.assertEqual(len(detector.provenance["checkpoint_sha256"]), 64)
            state = fixture.state_dict()
            del state["branch_heads_inf.0.bias"]
            torch.save(state, checkpoint)
            with self.assertRaisesRegex(self.module.InfaCheckpointError, "strictly"):
                self.module.NativeInfaDetector(self.args(infa_code_dir=str(root),
                    infa_checkpoint=str(checkpoint)), embedder=FixtureEmbedder())


    def test_all_agents_isolated_skips_detector_on_consecutive_empty_rounds(self):
        guard, detector = self.make([[[0.9,0.0],[0.9,0.0]]], n=2)
        first, _ = self.defend(guard, {0:"attack",1:"attack"})
        self.assertEqual(first, {})
        for round_idx in (1,2):
            try:
                result, decisions = self.defend(guard, {}, round_idx)
            except self.module.InfaRuntimeError as exc:
                self.fail("An exhausted session must not invoke its detector: " + str(exc))
            self.assertEqual(result, {})
            self.assertTrue(all(d["reason"] == "infa_all_agents_isolated" for d in decisions))
        self.assertEqual(len(detector.calls), 1)
        self.assertEqual(len(guard.history), 3)
        self.assertEqual(self.judge_calls, [])

if __name__ == "__main__":
    unittest.main()
