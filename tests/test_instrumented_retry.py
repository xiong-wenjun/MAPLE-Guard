import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import requests
from tools.run_instrumented import install_metrics, BenchmarkResponseError

def response(status=200, finish="stop"):
    result=requests.Response()
    result.status_code=status
    result._content_consumed=True
    result._content=json.dumps({"choices":[{"finish_reason":finish,"message":{"content":"answer"}}]}).encode()
    return result

class InstrumentedRetryTests(unittest.TestCase):
    def exercise(self, sequence):
        folder=tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path=Path(folder.name)/"api.jsonl"
        patch=mock.patch("requests.sessions.Session.send",side_effect=sequence)
        self.addCleanup(patch.stop)
        send=patch.start()
        env=mock.patch.dict(os.environ,{"MAPLE_TRANSPORT_MAX_ATTEMPTS":"3"})
        self.addCleanup(env.stop);env.start()
        install_metrics(path,True)
        sleep=mock.patch("tools.run_instrumented.time.sleep")
        self.addCleanup(sleep.stop);sleep.start()
        return path,send

    def test_transient_failures_retry_identical_request_and_journal_all_attempts(self):
        path,send=self.exercise([response(503),requests.ConnectionError("reset"),response()])
        out=requests.post("http://test/v1/chat/completions",json={"model":"m","messages":[]},timeout=17)
        self.assertEqual(out.status_code,200)
        self.assertEqual(send.call_count,3)
        self.assertEqual(len({c.args[-1].body for c in send.call_args_list}),1)
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual([x["attempt"] for x in rows],[1,2,3])
        self.assertEqual([x["will_retry"] for x in rows],[True,True,False])
        self.assertEqual(len({x["request_id"] for x in rows}),1)

    def test_authentication_failure_is_not_retried(self):
        path,send=self.exercise([response(401),response()])
        self.assertEqual(requests.post("http://test/v1/chat/completions",json={}).status_code,401)
        self.assertEqual(send.call_count,1)

    def test_truncated_success_is_fatal_and_never_retried(self):
        path,send=self.exercise([response(finish="length"),response()])
        with self.assertRaises(BenchmarkResponseError):
            requests.post("http://test/v1/chat/completions",json={})
        self.assertEqual(send.call_count,1)
        self.assertTrue(json.loads(path.read_text())["invalid_for_benchmark"])

    def test_transport_retry_budget_is_bounded(self):
        path,send=self.exercise([requests.Timeout("timeout")]*3)
        with self.assertRaises(requests.Timeout):
            requests.post("http://test/v1/chat/completions",json={})
        self.assertEqual(send.call_count,3)

if __name__=="__main__":unittest.main()
