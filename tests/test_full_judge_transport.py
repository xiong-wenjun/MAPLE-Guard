import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import requests

from evaluate.defense_methods.full_runtime import _factory
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
