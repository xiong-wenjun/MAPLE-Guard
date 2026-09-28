import os
import unittest
from unittest.mock import Mock, patch
from maple_guard import maple_guard_core as ep
from maple_guard.memory_backend import OpenAICompatibleLLM, OpenAICompatibleEmbedder
class AuthenticationTests(unittest.TestCase):
    def response(self,payload):
        r=Mock();r.status_code=200;r.json.return_value=payload
        return r
    def test_core_chat_sends_task_key_without_changing_payload(self):
        response=self.response({"choices":[{"message":{"content":"A"}}]})
        with patch.dict(os.environ,{"CHAT_API_KEY":"task-only-key"},clear=True), patch.object(ep.requests,"post",return_value=response) as post:
            self.assertEqual(ep.call_chat("http://task/v1","task-model",[{"role":"user","content":"pick"}]),"A")
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"],"Bearer task-only-key")
            self.assertNotIn("api_key",post.call_args.kwargs["json"])
    def test_no_secret_header_when_service_has_no_auth(self):
        response=self.response({"choices":[{"message":{"content":"A"}}]})
        with patch.dict(os.environ,{},clear=True), patch.object(ep.requests,"post",return_value=response) as post:
            ep.call_chat("http://task/v1","task-model",[])
            self.assertNotIn("Authorization",post.call_args.kwargs.get("headers",{}))
    def test_chat_and_embedding_keys_do_not_cross(self):
        with patch.dict(os.environ,{"CHAT_API_KEY":"chat-key","EMBED_API_KEY":"embed-key","OPENAI_API_KEY":"judge-key"},clear=True):
            self.assertEqual(OpenAICompatibleLLM("url","model").api_key,"chat-key")
            self.assertEqual(OpenAICompatibleEmbedder("url","model").api_key,"embed-key")
    def test_explicit_key_still_takes_priority(self):
        with patch.dict(os.environ,{"CHAT_API_KEY":"chat-key","EMBED_API_KEY":"embed-key"},clear=True):
            self.assertEqual(OpenAICompatibleLLM("url","model",api_key="explicit").api_key,"explicit")
            self.assertEqual(OpenAICompatibleEmbedder("url","model",api_key="explicit").api_key,"explicit")

    def test_different_judge_endpoint_uses_its_own_key(self):
        import json,tempfile
        from pathlib import Path
        from maple_guard.providers.service_auth import chat_headers
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"credentials.json"
            p.write_text(json.dumps({"judge":{"base_url":"http://judge/v1","api_key":"judge-secret"}}))
            with patch.dict(os.environ,{"MAPLE_SERVICE_CREDENTIALS":str(p),"CHAT_API_KEY":"task-secret"},clear=True):
                self.assertEqual(chat_headers("http://judge/v1/")["Authorization"],"Bearer judge-secret")
                with self.assertRaisesRegex(ValueError,"absent"):chat_headers("http://unknown/v1")

    def test_memory_summary_uses_same_disabled_thinking_protocol(self):
        response=self.response({"choices":[{"message":{"content":"summary"}}]})
        with patch.dict(os.environ,{"CHAT_DISABLE_THINKING":"1"},clear=True), patch("maple_guard.memory_backend.requests.post",return_value=response) as post:
            OpenAICompatibleLLM("http://task/v1","qwen").generate([])
            self.assertEqual(post.call_args.kwargs["json"]["chat_template_kwargs"],{"enable_thinking":False})

    def test_core_query_embedding_uses_embedding_server_credentials(self):
        response=self.response({"data":[{"embedding":[1.0,0.0]}]})
        with patch.dict(os.environ,{"EMBED_API_KEY":"embed-key","CHAT_API_KEY":"chat-key"},clear=True), patch.object(ep.requests,"post",return_value=response) as post:
            self.assertEqual(ep.remote_embedding("question","http://embedding/v1","embed-model"),[1.0,0.0])
            self.assertEqual(post.call_args.kwargs["headers"]["Authorization"],"Bearer embed-key")

    def test_auxiliary_chat_inherits_disabled_thinking_protocol(self):
        response=self.response({"choices":[{"message":{"content":"summary"}}]})
        with patch.dict(os.environ,{"CHAT_DISABLE_THINKING":"1"},clear=True), patch.object(ep.requests,"post",return_value=response) as post:
            ep.call_chat("http://task/v1","qwen",[])
            self.assertEqual(post.call_args.kwargs["json"]["chat_template_kwargs"],{"enable_thinking":False})
