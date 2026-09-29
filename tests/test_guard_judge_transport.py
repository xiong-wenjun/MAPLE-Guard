"""Regression coverage for the official communication judges transport."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluate.defense_methods.challenger_defense import challenger_act
from evaluate.defense_methods.inspector_defense import inspect_response
from evaluate.defense_methods.llm_client import chat_completion


class GuardJudgeTransportTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            safeguard_base_url="http://judge/v1",
            safeguard_model="judge-model",
            strict_comparison=True,
        )
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    @staticmethod
    def response(content):
        response = Mock()
        response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": content}}]
        }
        return response

    def test_challenger_forwards_disable_thinking_argument(self):
        self.args.disable_chat_thinking = True
        with patch(
            "evaluate.defense_methods.llm_client.requests.post",
            return_value=self.response("safe."),
        ) as post:
            self.assertEqual(challenger_act("2 + 2 = 4.", [], self.args), "safe")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload.get("chat_template_kwargs"), {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 50)
        self.assertEqual(payload["temperature"], 0)

    def test_inspector_inherits_disable_thinking_environment(self):
        self.args.disable_chat_thinking = False
        with patch.dict(os.environ, {"CHAT_DISABLE_THINKING": "1"}), patch(
            "evaluate.defense_methods.llm_client.requests.post",
            return_value=self.response("{\"decision\":\"safe\",\"revision\":\"<safe content>\"}"),
        ) as post:
            self.assertEqual(
                inspect_response("Agent_0", "2 + 2 = 4.", [], self.args),
                ("safe", "2 + 2 = 4."),
            )
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload.get("chat_template_kwargs"), {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_default_transport_keeps_server_thinking_default(self):
        with patch(
            "evaluate.defense_methods.llm_client.requests.post",
            return_value=self.response("safe"),
        ) as post:
            self.assertEqual(chat_completion(self.args, []), "safe")
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])

    def test_zero_environment_keeps_server_thinking_default(self):
        self.args.disable_chat_thinking = False
        with patch.dict(os.environ, {"CHAT_DISABLE_THINKING": "0"}), patch(
            "evaluate.defense_methods.llm_client.requests.post",
            return_value=self.response("safe"),
        ) as post:
            self.assertEqual(chat_completion(self.args, []), "safe")
        self.assertNotIn("chat_template_kwargs", post.call_args.kwargs["json"])


if __name__ == "__main__":
    unittest.main()
