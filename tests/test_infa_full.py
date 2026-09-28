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
                      infa_device="cpu")
        values.update(changes)
        return SimpleNamespace(**values)

    def make(self, probabilities, n=3):
        detector = Detector(probabilities)
        self.judge_calls, self.replacements, self.repairs = [], [], []
        def judge(messages):
            self.judge_calls.append(messages)
            return json.dumps({"response": "corrected answer"})
        guard = self.module.InfaGuardFull(self.args(n), judge, detector=detector)
        guard.begin_task("task")
        return guard, detector

    def defend(self, guard, outputs, idx=0, adjacency=None):
        adjacency = adjacency if adjacency is not None else [[int(i != j) for j in range(guard.num_agents)] for i in range(guard.num_agents)]
        return guard.defend(outputs, lambda *a, **k: self.fail("INFA must not impersonate a target inquiry"),
                            "Current question", idx, adjacency,
                            replace=lambda *x: self.replacements.append(x),
                            repair=lambda *x: self.repairs.append(x))

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
        for changes in ({"infa_threshold":float("nan")}, {"infa_donor_threshold":-1}, {"agents":0}):
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
