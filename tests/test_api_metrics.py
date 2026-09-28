import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import requests
from tools.run_instrumented import install_metrics

class UsageTests(unittest.TestCase):
    def response(self,reason="stop"):
        r=requests.Response();r.status_code=200
        r._content=json.dumps({"usage":{"prompt_tokens":7,"completion_tokens":3},
            "choices":[{"finish_reason":reason,"message":{"content":"PRIVATEANSWER"}}]}).encode()
        return r
    def test_metadata_captures_usage_without_secret_or_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/"calls.jsonl"
            with patch.object(requests.sessions.Session,"send",return_value=self.response()):
                install_metrics(str(log))
                requests.post("http://service/v1/chat/completions",headers={"Authorization":"Bearer SECRET"},
                    json={"model":"m","messages":[{"role":"user","content":"PRIVATEPROMPT"}]})
            raw=log.read_text()
            self.assertNotIn("SECRET",raw);self.assertNotIn("PRIVATE",raw)
            self.assertEqual(json.loads(raw)["usage"]["prompt_tokens"],7)
    def test_truncation_is_recorded_and_strict_run_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/"calls.jsonl"
            with patch.object(requests.sessions.Session,"send",return_value=self.response("length")):
                install_metrics(str(log),True)
                with self.assertRaisesRegex(RuntimeError,"token limit"):
                    requests.post("http://service/v1/chat/completions",json={"model":"m"})
            self.assertEqual(json.loads(log.read_text())["finish_reasons"],["length"])
    def test_transport_failure_keeps_original_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/"calls.jsonl"
            with patch.object(requests.sessions.Session,"send",side_effect=requests.ConnectionError("offline")):
                install_metrics(str(log))
                with self.assertRaises(requests.ConnectionError):
                    requests.post("http://service/v1/embeddings",json={"model":"m","input":["private"]})
            self.assertEqual(json.loads(log.read_text())["error_type"],"ConnectionError")
