"""Transport recovery never retries a model decision or changes the request body."""
import io
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

from evaluate.defense_methods.amemguard_full import AMemGuardFull, AMemGuardOutputError
from evaluate.defense_methods.comparison_runtime import model_clients


def arguments(**updates):
    values = dict(method="amemguard_full", full_judge_timeout=600,
                  full_judge_base_url="http://judge.invalid/v1",
                  full_judge_model="unchanged-judge", full_judge_api_key="SECRET_KEY",
                  full_judge_max_tokens=4096, disable_chat_thinking=True,
                  embed_base_url="http://embedding.invalid/v1", embed_model="unchanged-embed",
                  amemguard_top_k=4, amemguard_lesson_top_k=4,
                  amemguard_experiment_id="transport-test")
    values.update(updates)
    return SimpleNamespace(**values)


def reply(content="Entity -> relation -> Entity", finish="stop"):
    return io.BytesIO(json.dumps({"choices": [{"finish_reason": finish,
                                            "message": {"content": content}}],
                                 "usage": {"completion_tokens": 5}}).encode())


class AMemGuardTransportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.log = Path(self.folder.name) / "api-calls.jsonl"
        self.environment = mock.patch.dict(os.environ, {
            "AMEMGUARD_TRANSPORT_MAX_ATTEMPTS": "3",
            "MAPLE_CALL_LOG": str(self.log)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.sleep = mock.patch("time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def records(self):
        self.assertTrue(self.log.exists(), "A-MemGuard transport bypassed the API journal")
        return [json.loads(row) for row in self.log.read_text().splitlines()]

    def test_configured_timeout_reaches_judge_without_changing_payload(self):
        judge, _ = model_clients(arguments())
        with mock.patch("urllib.request.urlopen", return_value=reply()) as network:
            judge([{"role": "user", "content": "PRIVATE_PROMPT"}])
        self.assertEqual(network.call_args.kwargs["timeout"], 600)
        self.assertEqual(json.loads(network.call_args.args[0].data), {
            "model": "unchanged-judge", "messages": [{"role": "user", "content": "PRIVATE_PROMPT"}],
            "temperature": 0.0, "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": False}})

    def test_raw_socket_timeout_retries_identical_request_and_journals_each_attempt(self):
        judge, _ = model_clients(arguments())
        with mock.patch("urllib.request.urlopen", side_effect=[TimeoutError("sensitive failure"), reply()]) as network:
            result = judge([{"role": "user", "content": "PRIVATE_PROMPT"}])
        self.assertEqual(result, "Entity -> relation -> Entity")
        self.assertEqual(network.call_count, 2)
        self.assertEqual(network.call_args_list[0].args[0].data, network.call_args_list[1].args[0].data)
        records = self.records()
        self.assertEqual([r["attempt"] for r in records], [1, 2])
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])
        self.assertEqual(records[0]["error_type"], "TimeoutError")
        self.assertTrue(records[0]["will_retry"])
        self.assertFalse(records[1]["will_retry"])
        self.assertEqual(records[1]["usage"], {"completion_tokens": 5})
        self.assertEqual(records[1]["finish_reasons"], ["stop"])
        for secret in ("SECRET_KEY", "PRIVATE_PROMPT", "sensitive failure"):
            self.assertNotIn(secret, self.log.read_text())

    def test_transient_http_statuses_and_wrapped_network_errors_retry(self):
        errors = [urllib.error.HTTPError("http://judge.invalid", code, "private", {}, None)
                  for code in (429, 500, 502, 503, 504)]
        errors += [urllib.error.URLError(socket.timeout("timed out")), ConnectionResetError(),
                   urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "temporary DNS"))]
        for error in errors:
            with self.subTest(error=type(error).__name__, code=getattr(error, "code", None)):
                judge, _ = model_clients(arguments())
                with mock.patch("urllib.request.urlopen", side_effect=[error, reply()]) as network:
                    self.assertEqual(judge([]), "Entity -> relation -> Entity")
                    self.assertEqual(network.call_count, 2)

    def test_permanent_http_and_dns_failures_are_not_retried(self):
        errors = [urllib.error.HTTPError("http://judge.invalid", code, "context or credentials", {}, None)
                  for code in (400, 401, 403, 404, 413, 422)]
        errors += [urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "bad hostname")),
                   urllib.error.URLError("unknown permanent failure")]
        for error in errors:
            with self.subTest(error=type(error).__name__, code=getattr(error, "code", None)):
                judge, _ = model_clients(arguments())
                with mock.patch("urllib.request.urlopen", side_effect=error) as network:
                    with self.assertRaises(type(error)):
                        judge([])
                    self.assertEqual(network.call_count, 1)
                    self.assertFalse(self.records()[-1]["will_retry"])

    def test_exhausted_transport_failure_is_strict_and_cannot_create_lessons(self):
        args = arguments()
        judge, _ = model_clients(args)
        guard = AMemGuardFull(args, judge, lambda text: [1., 0.])
        entry = SimpleNamespace(memory_id="m1", intent="query", experience="operation", origin_agent=0)
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")) as network:
            with self.assertRaises(AMemGuardOutputError):
                guard.select("query", [entry], 0)
            self.assertEqual(network.call_count, 3)
        self.assertFalse(any(guard.state_dict()["lessons"].values()))
        records = self.records()
        self.assertEqual(len(records), 3)
        self.assertFalse(records[-1]["will_retry"])
        self.assertEqual(judge.counters["calls"], 1)
        self.assertEqual(judge.counters["errors"], 1)

    def test_invalid_or_truncated_output_is_not_retried(self):
        for content, finish in [("partial", "length"), ("", "stop"), ("   ", "stop"), (None, "stop")]:
            with self.subTest(content=content, finish=finish):
                judge, _ = model_clients(arguments())
                guard = AMemGuardFull(arguments(), judge, lambda text: [1., 0.])
                with mock.patch("urllib.request.urlopen", return_value=reply(content, finish)) as network:
                    with self.assertRaises(AMemGuardOutputError):
                        guard._ask("query", "reasoning path")
                    self.assertEqual(network.call_count, 1)
                self.assertTrue(self.records()[-1]["invalid_for_benchmark"])
                self.assertFalse(self.records()[-1]["will_retry"])

    def test_malformed_response_json_is_not_retried(self):
        judge, _ = model_clients(arguments())
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"{broken")) as network:
            with self.assertRaises(json.JSONDecodeError):
                judge([])
            self.assertEqual(network.call_count, 1)
        self.assertEqual(self.records()[-1]["error_type"], "JSONDecodeError")
        self.assertFalse(self.records()[-1]["will_retry"])

    def test_embedding_transport_is_instrumented_and_keeps_120_second_timeout(self):
        _, embed = model_clients(arguments())
        data = io.BytesIO(b'{"data":[{"embedding":[1.0,0.0]}],"usage":{"prompt_tokens":3}}')
        with mock.patch("urllib.request.urlopen", side_effect=[TimeoutError(), data]) as network:
            self.assertEqual(embed("PRIVATE_INPUT"), [1., 0.])
        self.assertEqual(network.call_args.kwargs["timeout"], 120)
        records = self.records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["endpoint"], "/v1/embeddings")
        self.assertNotIn("PRIVATE_INPUT", self.log.read_text())

    def test_legacy_default_is_one_attempt_and_120_second_timeout(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            judge, _ = model_clients(arguments(full_judge_timeout=None))
            with mock.patch("urllib.request.urlopen", side_effect=TimeoutError()) as network:
                with self.assertRaises(TimeoutError):
                    judge([])
            self.assertEqual(network.call_count, 1)
            self.assertEqual(network.call_args.kwargs["timeout"], 120)

    def test_opt_in_defense_length_uses_same_committed_journal_identity(self):
        from maple_guard import budget_outcomes as budget
        args=arguments(response_budget_policy='fail_task',agents=2,attacker_ids=[1])
        judge,_=model_clients(args)
        with budget.task_scope(args,'task-t',poison_texts={}) as audit:
            with mock.patch('urllib.request.urlopen',return_value=reply('partial','length')):
                with self.assertRaises(budget.BudgetExceeded) as caught:judge([])
        record=self.records()[-1]
        event=caught.exception.event
        for field in ('request_id','task_id','role','response_budget_policy','budget_outcomes_version','invalid_response_type'):
            self.assertEqual(record[field],event[field])
        self.assertEqual(record['role'],'defense')
        self.assertEqual(len(audit.requests),1)

    def test_opt_in_transport_failure_cannot_be_swallowed_by_method(self):
        from maple_guard import budget_outcomes as budget
        args=arguments(response_budget_policy='fail_task',agents=2,attacker_ids=[1])
        judge,_=model_clients(args)
        guard=AMemGuardFull(args,judge,lambda text:[1.,0.])
        with budget.task_scope(args,'task-t',poison_texts={}),mock.patch('urllib.request.urlopen',side_effect=TimeoutError()) as network:
            with self.assertRaises(budget.RecoverableProviderError):guard._ask('query','reasoning path')
        self.assertEqual(network.call_count,3)

    def test_invalid_transport_config_rejected_before_network(self):
        for timeout in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(timeout=timeout), mock.patch("urllib.request.urlopen") as network:
                with self.assertRaises(ValueError):
                    model_clients(arguments(full_judge_timeout=timeout))
                network.assert_not_called()
        for attempts in ("0", "4", "-1", "1.5", "invalid"):
            with self.subTest(attempts=attempts), mock.patch.dict(os.environ, {
                    "AMEMGUARD_TRANSPORT_MAX_ATTEMPTS": attempts}):
                with self.assertRaises(ValueError):
                    model_clients(arguments())


if __name__ == "__main__":
    unittest.main()
