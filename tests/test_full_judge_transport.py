import argparse
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import requests

from evaluate.defense_methods.full_runtime import _factory, add_full_baseline_args, baseline_run_provenance, runtime_scope
from tools.run_instrumented import install_metrics


SOURCE = Path(os.environ.get("AGENTXPOSED_SOURCE", "/mnt/public/data/wj/baseline-references/AgentXposed"))


def arguments(**updates):
    values = dict(method="agentxposed_full_kick", agents=1,
                  agentxposed_protocol="reconstruction", strict_comparison=True,
                  chat_base_url="http://judge.invalid/v1", chat_model="test-model",
                  full_judge_max_tokens=4096, disable_chat_thinking=True)
    values.update(updates)
    return SimpleNamespace(**values)


def response(content="complete assessment", finish_reason="stop"):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps({"choices": [{"finish_reason": finish_reason,
                                              "message": {"content": content}}],
                                 "usage": {"completion_tokens": 4}}).encode()
    return result


class FullJudgeTransportTests(unittest.TestCase):
    def assert_fatal(self, guard, **kwargs):
        caught = None
        try:
            guard.judge([{"role": "user", "content": "visible evidence"}], **kwargs)
        except BaseException as exc:
            caught = exc
        self.assertIsNotNone(caught, "invalid provider result must abort the benchmark")
        self.assertIsInstance(caught, SystemExit)
        self.assertTrue(getattr(caught, "fatal_for_benchmark", False))
        self.assertEqual(guard.judge.counters["errors"], 1)

    def test_csqa_agentsafe_classifies_after_temporary_method_is_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            policy = Path(folder) / "policy.json"
            criteria = Path(folder) / "criteria.json"
            policy.write_text(json.dumps({"identities": {"0": "Agent 0"}}))
            criteria.write_text(json.dumps(["valid task facts"]))
            args = arguments(methods="agentsafe_full", agentsafe_threshold=0.5,
                             agentsafe_policy_file=str(policy),
                             agentsafe_criteria_file=str(criteria),
                             embed_base_url="http://embedding.invalid/v1",
                             embed_model="embed-model")
            del args.method
            embedding = response()
            embedding._content = json.dumps({"data": [{"embedding": [1., 0.]}]}).encode()
            with mock.patch("requests.post", side_effect=[embedding, response('{"level": 2}')]) as post:
                with runtime_scope(args, method=args.methods) as runtime:
                    self.assertFalse(hasattr(args, "method"))
                    self.assertEqual(runtime.guard._level("The correct answer is A."), 2)
                    self.assertEqual(post.call_args.kwargs["json"]["response_format"],
                                     {"type": "json_object"})
            self.assertFalse(hasattr(args, "method"))
            self.assertEqual(post.call_count, 2)

    def test_judge_protocol_remains_bound_to_constructed_method(self):
        cases = [
            ("agentxposed_full_guide", "infa_guard_full", "agentxposed_full.AgentXposedFull", True),
            ("infa_guard_full", "agentsafe_full", "infa_full.InfaGuardFull", False),
        ]
        for method, next_method, constructor, expects_json in cases:
            with self.subTest(method=method):
                args = arguments(method=method, infa_protocol="released")
                with mock.patch("evaluate.defense_methods." + constructor,
                                side_effect=lambda args, judge: SimpleNamespace(judge=judge)):
                    guard = _factory(args)
                args.method = next_method
                with mock.patch("requests.post", return_value=response()) as post:
                    guard.judge([])
                self.assertEqual("response_format" in post.call_args.kwargs["json"], expects_json)

    def test_timeout_is_fatal_in_strict_mode(self):
        guard = _factory(arguments())
        with mock.patch("requests.post", side_effect=TimeoutError("timed out")), \
             mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            self.assert_fatal(guard)

    def test_invalid_responses_are_fatal_in_strict_mode(self):
        for content, finish in [("partial", "length"), ("", "stop"), ("   ", "stop"), (None, "stop")]:
            with self.subTest(content=content, finish=finish):
                guard = _factory(arguments())
                reply = response(content, finish)
                with mock.patch("requests.post", return_value=reply), \
                     mock.patch("urllib.request.urlopen", return_value=io.BytesIO(reply.content)):
                    self.assert_fatal(guard, response_format=None)

    def test_legacy_timeout_remains_an_exception(self):
        guard = _factory(arguments(strict_comparison=False))
        with mock.patch("requests.post", side_effect=TimeoutError("timed out")), \
             mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(TimeoutError):
                guard.judge([])
        self.assertEqual(guard.judge.counters["errors"], 1)

    def test_judge_calls_reach_existing_api_metrics_without_changing_payload(self):
        guard = _factory(arguments())
        reply = response()
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch("requests.sessions.Session.send", return_value=reply) as send, \
             mock.patch("urllib.request.urlopen", return_value=io.BytesIO(reply.content)):
            path = Path(folder) / "api-calls.jsonl"
            install_metrics(path)
            result = guard.judge([{"role": "user", "content": "visible evidence"}],
                                 temperature=0.7, response_format=None)
            self.assertEqual(result, "complete assessment")
            self.assertTrue(path.exists(), "full judge bypassed API instrumentation")
            record = json.loads(path.read_text())
            self.assertEqual(record["temperature"], 0.7)
            self.assertEqual(record["max_tokens"], 4096)
            self.assertFalse(record["invalid_for_benchmark"])
            request = send.call_args.args[-1]
            payload = json.loads(request.body)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertNotIn("response_format", payload)
            self.assertEqual(send.call_args.kwargs["timeout"], 120)

    def test_released_agentxposed_uses_600_second_timeout(self):
        for protocol in ("released_minimal_fix", "released_unmodified"):
            with self.subTest(protocol=protocol), \
                 mock.patch("evaluate.defense_methods.agentxposed_full.AgentXposedFull",
                            side_effect=lambda args, judge: SimpleNamespace(judge=judge)), \
                 mock.patch("requests.post", return_value=response()) as post:
                guard = _factory(arguments(agentxposed_protocol=protocol))
                guard.judge([], temperature=0.5, response_format=None)
                self.assertEqual(post.call_args.kwargs["timeout"], 600)
                self.assertEqual(guard.judge.timeout_seconds, 600)
                payload = post.call_args.kwargs["json"]
                self.assertEqual(payload["max_tokens"], 4096)
                self.assertEqual(payload["temperature"], 0.5)
                self.assertNotIn("response_format", payload)
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_judge_timeout_override_reaches_transport_and_provenance(self):
        args = arguments(full_judge_timeout=42.5)
        guard = _factory(args)
        with mock.patch("requests.post", return_value=response()) as post:
            guard.judge([])
        self.assertEqual(post.call_args.kwargs["timeout"], 42.5)
        self.assertEqual(guard.judge.timeout_seconds, 42.5)
        args._full_baseline_runtime = SimpleNamespace(
            guard=guard, method=args.method, state_path="", experiment_identity="test")
        self.assertEqual(baseline_run_provenance(args)["judge_timeout_seconds"], 42.5)

    def test_invalid_judge_timeout_fails_before_any_provider_call(self):
        for timeout in (0, -1, float("nan"), float("inf"), -float("inf"), "invalid"):
            with self.subTest(timeout=timeout), mock.patch("requests.post") as post:
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    _factory(arguments(full_judge_timeout=timeout))
                post.assert_not_called()

    def test_judge_timeout_flag_defaults_to_profile_selection(self):
        parser = argparse.ArgumentParser()
        add_full_baseline_args(parser)
        self.assertTrue(hasattr(parser.parse_args([]), "full_judge_timeout"), "missing timeout flag")
        self.assertIsNone(parser.parse_args([]).full_judge_timeout)
        self.assertEqual(parser.parse_args(["--full-judge-timeout", "240.5"]).full_judge_timeout, 240.5)
        for timeout in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=timeout), mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--full-judge-timeout", timeout])

    def test_judge_timeout_config_override_is_supported(self):
        parser = argparse.ArgumentParser()
        add_full_baseline_args(parser, {"defense": {"full": {"full_judge_timeout": 240.5}}})
        self.assertTrue(hasattr(parser.parse_args([]), "full_judge_timeout"), "missing timeout flag")
        self.assertEqual(parser.parse_args([]).full_judge_timeout, 240.5)

    def test_explicit_thinking_enabled_is_preserved(self):
        guard = _factory(arguments(disable_chat_thinking=False))
        reply = response()
        with mock.patch("requests.post", return_value=reply) as post, \
             mock.patch("urllib.request.urlopen", return_value=io.BytesIO(reply.content)):
            guard.judge([])
        self.assertTrue(post.called, "judge must use the instrumented transport")
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])

    def test_agentsafe_embeddings_reach_existing_api_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            policy = Path(folder) / "policy.json"
            criteria = Path(folder) / "criteria.json"
            policy.write_text(json.dumps({"identities": {"0": "Agent 0"}}))
            criteria.write_text(json.dumps(["valid task facts"]))
            reply = response()
            reply._content = json.dumps({"data": [{"embedding": [1., 0.]}]}).encode()
            with mock.patch("requests.sessions.Session.send", return_value=reply), \
                 mock.patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(reply.content)):
                path = Path(folder) / "api-calls.jsonl"
                install_metrics(path)
                guard = _factory(arguments(method="agentsafe_full", agentsafe_threshold=0.5,
                                           agentsafe_policy_file=str(policy), agentsafe_criteria_file=str(criteria),
                                           embed_base_url="http://embedding.invalid/v1", embed_model="embed-model"))
                self.assertEqual(guard.embed("memory evidence"), [1., 0.])
                self.assertTrue(path.exists(), "full embedding bypassed API instrumentation")
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(records), 2)
                self.assertTrue(all(record["endpoint"] == "/v1/embeddings" for record in records))

    @unittest.skipUnless((SOURCE / "Detect/main.py").is_file(), "external AgentXposed source not present")
    def test_released_detector_aborts_on_first_provider_failure(self):
        guard = _factory(arguments(agentxposed_protocol="released_minimal_fix",
                                   agentxposed_code_dir=str(SOURCE)))
        with mock.patch("requests.post", side_effect=TimeoutError("timed out")), \
             mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(SystemExit):
                guard.detect_released([[{"role": "user", "content": "task"},
                                        {"role": "assistant", "content": "reply"}]])
        self.assertEqual(guard.judge.counters["calls"], 1)
        self.assertEqual(guard.judge.counters["errors"], 1)


if __name__ == "__main__":
    unittest.main()
