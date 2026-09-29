"""Regression checks for bounded, independently calibrated AgentSafe adaptation."""
import hashlib
import importlib.util
import io
from unittest.mock import patch
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(test):
    path = ROOT / "tools/prepare_agentsafe_adaptation.py"
    test.assertTrue(path.is_file(), "independent AgentSafe calibration tool is missing")
    spec = importlib.util.spec_from_file_location("agentsafe_adaptation_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(count=20):
    return [{"id": f"train-{i}", "question": f"Question number {i}?",
             "choices": {"label": ["A", "B", "C"], "text": ["first", "second", "third"]},
             "answerKey": "B"} for i in range(count)]


class CalibrationTests(unittest.TestCase):
    def test_question_split_is_reproducible_disjoint_and_order_independent(self):
        tool = load_tool(self)
        split = tool.select_questions(rows(), 4, 4, 42)
        self.assertEqual(split, tool.select_questions(list(reversed(rows())), 4, 4, 42))
        self.assertTrue(set(x["id"] for x in split["calibration"]).isdisjoint(
            x["id"] for x in split["validation"]))
        self.assertNotEqual(split, tool.select_questions(rows(), 4, 4, 43))

    def test_corrupted_pair_uses_only_training_rows_and_preserves_pair_identity(self):
        tool = load_tool(self)
        records = tool.make_records(tool.select_questions(rows(), 4, 4, 42))
        self.assertEqual(len(records), 32)
        for split in ("calibration", "validation"):
            local = [r for r in records if r["split"] == split]
            self.assertEqual(sum(r["label"] == "benign" for r in local), 8)
            self.assertEqual(sum(r["label"] == "corrupted" for r in local), 8)
        for record in records:
            self.assertNotIn("label:", record["text"].lower())
            self.assertEqual(record["text_sha256"], hashlib.sha256(record["text"].encode()).hexdigest())
        for pair in {r["pair_id"] for r in records}:
            self.assertEqual({r["label"] for r in records if r["pair_id"] == pair}, {"benign", "corrupted"})

    def test_threshold_obeys_strict_all_criterion_rule_and_benign_constraint(self):
        tool = load_tool(self)
        threshold = tool.choose_threshold([.1, .2, .2, .3], .25)
        self.assertEqual(threshold, math.nextafter(.2, -math.inf))
        metrics = tool.score_metrics([.1, .2, .2, .3], [-.5, .15, .25, .4], threshold)
        self.assertEqual(metrics["benign_false_rejection_rate"], .25)
        self.assertEqual(metrics["corrupted_detection_rate"], .5)
        self.assertEqual(tool.minimum_cosine([1, 0], [[1, 0], [0, 1]]), 0.0)

    def test_small_sample_one_percent_means_zero_calibration_false_rejections(self):
        tool = load_tool(self)
        scores = [i / 100 for i in range(64)]
        threshold = tool.choose_threshold(scores, .01)
        self.assertEqual(tool.score_metrics(scores, [], threshold)["benign_false_rejections"], 0)
        with self.assertRaises(ValueError):
            tool.choose_threshold([-1., -1.], .01)
        with self.assertRaises(ValueError):
            tool.choose_threshold([float("nan")], .01)

    def test_embedding_budget_is_checked_before_network_and_batches_preserve_order(self):
        tool = load_tool(self)
        calls = []
        def fake(texts):
            calls.append(texts)
            return [[1., float(x)] for x in texts]
        with self.assertRaises(ValueError):
            tool.embed_batches([str(i) for i in range(17)], fake, 4, 4)
        self.assertFalse(calls)
        result = tool.embed_batches([str(i) for i in range(9)], fake, 4, 3)
        self.assertEqual([len(x) for x in calls], [4, 4, 1])
        self.assertEqual(result[-1], [1., 8.])

    def test_scalar_transport_matches_runtime_payload_and_records_protocol(self):
        tool = load_tool(self)
        calls = []
        class Response(io.StringIO):
            status = 200
        def fake_urlopen(request, timeout):
            body = None if request.data is None else json.loads(request.data)
            calls.append(body)
            data = {"data": [{"id": "test-encoder"}]} if body is None else {
                "data": [{"index": 0, "embedding": [1., 2.]}]}
            return Response(json.dumps(data))
        with tempfile.TemporaryDirectory() as folder, patch.object(tool.urllib.request, "urlopen", fake_urlopen):
            embed, identity = tool.remote_transport(
                {"model": "test-encoder", "base_url": "https://example.invalid/v1"},
                Path(folder)/"calls.jsonl", 10, input_mode="scalar", batch_size=1)
            self.assertEqual(embed(["unchanged criterion"]), [[1., 2.]])
            self.assertEqual(calls, [None, {"model": "test-encoder", "input": "unchanged criterion"}])
            protocol = identity["request_protocol"]
            self.assertEqual(protocol, {"input_shape": "string", "batch_size": 1,
                "encoding_format": None, "encoding_format_policy": "omitted", "input_prefix": None})
            logged = json.loads((Path(folder)/"calls.jsonl").read_text())
            self.assertEqual(logged["request_protocol"], protocol)
            with self.assertRaises(ValueError):
                embed(["first", "second"])
            self.assertEqual(len(calls), 2)

    def test_scalar_transport_rejects_batched_configuration_before_network(self):
        tool = load_tool(self)
        with patch.object(tool.urllib.request, "urlopen") as network:
            with self.assertRaises(ValueError):
                tool.remote_transport({"model": "test-encoder", "base_url": "https://example.invalid"},
                    "/unused.jsonl", 10, input_mode="scalar", batch_size=16)
            network.assert_not_called()

    def test_heldout_integrity_is_hash_only_and_rejects_changed_files(self):
        tool = load_tool(self)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = root / "train.parquet"; train.write_bytes(b"training")
            held = {}
            for i in range(5):
                p = root / f"held-{i}.json"; p.write_bytes(b"not JSON: no labels need parsing")
                held[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            provenance = {"split": "train", "training_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                          "heldout_files": held, "excluded_rows": 0}
            result = tool.verify_dataset_assets(train, provenance)
            self.assertEqual(result["heldout_files_verified"], 5)
            Path(next(iter(held))).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                tool.verify_dataset_assets(train, provenance)


class AdaptedProfileTests(unittest.TestCase):
    def setUp(self):
        from test_agentsafe_full import AgentSafeFull, FakeJudge
        self.Guard = AgentSafeFull; self.judge = FakeJudge()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.policy = root / "policy.json"
        self.policy.write_text(json.dumps({"identities": {"0": "Agent_0", "1": "Agent_1"},
            "relations": {"0": {"1": 2}, "1": {"0": 2}}, "default_level": 1, "self_level": 4}))
        self.criteria = root / "criteria.json"; self.criteria.write_text('["criterion"]')
        self.manifest = root / "calibration.json"
        self.freeze = {"status": "frozen", "profile": "paper_v2_adapted",
                       "criteria_sha256": hashlib.sha256(self.criteria.read_bytes()).hexdigest(),
                       "threshold": .25, "encoder": {"model": "test-encoder"},
                       "score_rule": "min_all_criterion_cosine_strict_gt",
                       "criterion_vectors": [[1., 0.]], "review_interval": 1}
        self.manifest.write_text(json.dumps(self.freeze))
        self.args = SimpleNamespace(agentsafe_policy_file=str(self.policy), agentsafe_criteria_file=str(self.criteria),
            agentsafe_threshold=.25, agentsafe_review_interval=1, agentsafe_profile="paper_v2_adapted",
            agentsafe_calibration_manifest=str(self.manifest), embed_model="test-encoder")

    def test_full_runtime_criterion_transport_matches_scalar_calibration_payload(self):
        from evaluate.defense_methods.full_runtime import _factory
        self.args.method = "agentsafe_full"
        self.args.agents = 2
        self.args.embed_base_url = "https://example.invalid/v1"
        self.args.full_judge_base_url = "https://example.invalid/v1"
        self.args.full_judge_model = "unused-judge"
        with patch("requests.post") as post:
            post.return_value.__enter__.return_value.json.return_value = {
                "data": [{"index": 0, "embedding": [1., 0.]}]}
            guard = _factory(self.args)
        self.assertEqual(guard.profile, "paper_v2_adapted")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.kwargs["json"],
            {"model": "test-encoder", "input": "criterion"})

    def freeze_numeric_canary(self, maximum=.0004):
        evidence = self.manifest.with_name("canary-evidence.json")
        report = {"maximum_cosine_error": maximum, "selected_tolerance": 2*maximum,
            "tolerance_rule": "2*M", "hard_cap": .002, "eligible": True,
            "measurement_count": 1, "records": [{"criterion": 0, "cosine_error_to_new_frozen": maximum}],
            "reference_criterion_vectors_sha256": "a"*64}
        evidence.write_text(json.dumps(report))
        self.freeze["encoder"].update(criterion_geometry_cosine_tolerance=2*maximum,
            criterion_vectors_file_sha256="a"*64, numeric_canary={
                "purpose": "embedding_deployment_numeric_canary",
                "measurement_path": str(evidence),
                "measurement_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()})
        self.manifest.write_text(json.dumps(self.freeze))

    def test_measured_canary_accepts_numeric_noise_without_changing_scoring(self):
        exact = self.Guard(self.args, self.judge, lambda t: [1., 0.] if t == "criterion" else [.8, .6])
        self.freeze_numeric_canary()
        cosine = 1-.0006
        noisy = self.Guard(self.args, self.judge,
            lambda t: [cosine, math.sqrt(1-cosine*cosine)] if t == "criterion" else [.8, .6])
        self.assertEqual(exact.criterion_vectors, noisy.criterion_vectors)
        self.assertEqual(exact._criterion_scores("message"), noisy._criterion_scores("message"))
        self.assertEqual(exact.threshold, noisy.threshold)
        self.assertAlmostEqual(noisy.provenance["deployment_canary"]["criterion_cosine_errors"][0], .0006)
        self.assertEqual(noisy.provenance["deployment_canary"]["tolerance"], .0008)

    def test_measured_canary_rejects_above_frozen_tolerance_and_unknown_tolerance(self):
        self.freeze_numeric_canary()
        cosine = 1-.0009
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [cosine, math.sqrt(1-cosine*cosine)])
        del self.freeze["encoder"]["numeric_canary"]
        self.manifest.write_text(json.dumps(self.freeze))
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [1., 0.])

    def test_measured_canary_rejects_nonfinite_unbounded_or_rule_changed_tolerance(self):
        self.freeze_numeric_canary()
        for invalid in [float("nan"), float("inf"), True, -.001, .0021, .0009]:
            with self.subTest(tolerance=invalid):
                self.freeze["encoder"]["criterion_geometry_cosine_tolerance"] = invalid
                self.manifest.write_text(json.dumps(self.freeze))
                with self.assertRaises(ValueError):
                    self.Guard(self.args, self.judge, lambda t: [1., 0.])

    def test_measured_canary_rejects_changed_measurement_artifact(self):
        self.freeze_numeric_canary()
        Path(self.freeze["encoder"]["numeric_canary"]["measurement_path"]).write_text("{}")
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [1., 0.])

    def test_legacy_calibration_keeps_strict_numeric_gate(self):
        cosine = 1-.00002
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [cosine, math.sqrt(1-cosine*cosine)])

    def test_adapted_profile_requires_frozen_calibration_and_model_identity(self):
        self.args.embed_model = "different-encoder"
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [1., 0.])

    def test_adapted_profile_rejects_changed_criteria_or_threshold(self):
        self.args.agentsafe_threshold = .2
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [1., 0.])
        self.args.agentsafe_threshold = .25
        self.criteria.write_text('["changed criterion"]')
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [1., 0.])

    def test_adapted_profile_keeps_v2_message_level_not_sender_level_comparison(self):
        guard = self.Guard(self.args, self.judge, lambda t: [1., 0.])
        self.assertEqual(getattr(guard, "profile", None), "paper_v2_adapted")
        self.judge.level = 4
        self.assertIsNone(guard.route("highly private", 0, 1)[0])
        self.judge.level = 1
        self.assertEqual(guard.route("public task data", 0, 1)[0], "public task data")

    def test_adapted_profile_rejects_different_criterion_embedding_geometry(self):
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [0., 1.])



    def test_adapted_profile_requires_the_frozen_review_interval(self):
        self.args.agentsafe_review_interval = 2
        with self.assertRaises(ValueError):
            self.Guard(self.args, self.judge, lambda t: [1., 0.])

    def test_live_geometry_is_checked_then_frozen_criterion_vectors_are_reused(self):
        guard = self.Guard(self.args, self.judge, lambda t: [1., .0001])
        self.assertEqual(guard.criterion_vectors, [[1., 0.]])



class AdaptedIntegrationTests(unittest.TestCase):
    def test_new_profile_cli_flags_and_encoder_identity_reach_component(self):
        import argparse
        from evaluate.defense_methods.full_runtime import add_full_baseline_args, _component_args
        parser = argparse.ArgumentParser()
        add_full_baseline_args(parser, {})
        self.assertIn("--agentsafe-profile", parser._option_string_actions)
        self.assertIn("--agentsafe-calibration-manifest", parser._option_string_actions)
        parsed = parser.parse_args(["--agentsafe-profile", "paper_v2_adapted",
            "--agentsafe-calibration-manifest", "/frozen/calibration.json"])
        parsed.embed_model = "test-encoder"
        component = _component_args(parsed)
        self.assertEqual(component.agentsafe_profile, "paper_v2_adapted")
        self.assertEqual(component.embed_model, "test-encoder")

if __name__ == "__main__":
    unittest.main()
